#!/usr/bin/env python3
"""
redaction_eval.py - measure how much PII a redaction step removed.

    python First_pipeline/firstpp_fixed/eval/redaction_eval.py before_redact.md after_redact.md
    python First_pipeline/firstpp_fixed/eval/redaction_eval.py v1.md v1_redacted_v3.md --models gravitee,gliner,pii-tracer
    python First_pipeline/firstpp_fixed/eval/redaction_eval.py a.md b.md --models none          # regex/checksum layer only
    python First_pipeline/firstpp_fixed/eval/redaction_eval.py a.md b.md \
        --config First_pipeline/firstpp_fixed/eval/pii_config.example.json --fail-under 90

Flow
  1. each .md -> plain text via md_to_text.py. The .txt is written next to its .md as <md_stem>.txt
     (one txt per iteration, so v1/v2/v3 never collide) and is reused only when a sidecar
     <stem>.txt.meta.json proves it was made from exactly this .md content.
  2. detect PII  = regex + checksum validators (pii_patterns.py)  +  optional ML models
     (gravitee bert-small-pii, GLiNER-PII, PII-Tracer, any HF token-classification model)
     -> spans are merged into one entity list. Same detectors run on both files.
  3. score = risk-weighted residual exposure, with severity caps for leaked high-value IDs.
  4. console report + <after_stem>__vs__<before_stem>.eval.json next to the after file.

Score (0-100)
  value weight   = base risk(type) x confidence(validated / model prob / context)
                   x linkage (name or other IDs nearby) x (1 + 0.25*log2(occurrences))
  residual       = sum(weight x leak_fraction)   over PII found in BEFORE that still shows in AFTER
                 + new_exposure_factor x weight   of PII in AFTER that BEFORE never had
  raw score      = 100 x (1 - residual / total_before_weight)
  cap            : a leaked CRITICAL value (Aadhaar, SSN, card, account, passport, PAN...) caps the
                   score at 79 (-5 per extra one); a leaked HIGH value caps it at 94 (-2 per extra one)
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import md_to_text
from pii_patterns import (
    BUILTIN_RULES, GLINER_LABELS, MASK_RE, PARTIAL_RUN_TYPES, TEXT_TYPES, Config, Entity, RawSpan,
    find_regex_spans, load_config, map_model_label, mask_preview, merge_spans, normalize_value,
    normalize_for_detection, tier_of, weight_for,
)

# --------------------------------------------------------------------------- tunables
CRITICAL_CAP, CRITICAL_STEP, CRITICAL_FLOOR = 79.0, 5.0, 40.0
HIGH_CAP, HIGH_STEP, HIGH_FLOOR = 94.0, 2.0, 80.0
LINK_WINDOW = 250                 # chars: "nearby" for the linkage factor
MIN_CONF = 0.25                   # entities below this are ignored
PASS_SCORE, REVIEW_SCORE = 95.0, 80.0
TIER_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def log(msg: str):
    print(msg, file=sys.stderr)


# =============================================================================
# 1. md -> txt (with safe re-use)
# =============================================================================
def prepare_text(md_path: str, table_style: str, force: bool) -> dict:
    md = Path(md_path).expanduser().resolve()
    if not md.is_file():
        sys.exit(f"File not found: {md}")
    raw = md.read_bytes()
    md_hash = hashlib.sha256(raw).hexdigest()

    if md.suffix.lower() == ".txt":                       # already plain text
        return {"md": md, "txt": md, "text": raw.decode("utf-8-sig", "replace"),
                "status": "plain-text input", "md_sha": md_hash}

    txt = md.with_suffix(".txt")                          # same folder, same stem
    meta = md.with_name(md.stem + ".txt.meta.json")
    want = {"md_sha256": md_hash, "table_style": table_style, "converter": "md_to_text/1"}
    if not force and txt.is_file() and meta.is_file():
        try:
            if json.loads(meta.read_text()) == want:
                return {"md": md, "txt": txt, "text": txt.read_text(encoding="utf-8"),
                        "status": "reused existing txt", "md_sha": md_hash}
        except Exception:
            pass
    text = md_to_text.convert(raw.decode("utf-8-sig", "replace"), table_style=table_style)
    txt.write_text(text, encoding="utf-8", newline="\n")
    meta.write_text(json.dumps(want), encoding="utf-8")
    return {"md": md, "txt": txt, "text": text, "status": "converted", "md_sha": md_hash}


# =============================================================================
# 2. Model detectors (all optional, loaded lazily)
# =============================================================================
def iter_chunks(text: str, size: int, overlap: int):
    n, pos = len(text), 0
    while pos < n:
        end = min(n, pos + size)
        if end < n:                                        # snap to a newline/space near the end
            cut = max(text.rfind("\n", pos + int(size * 0.6), end), text.rfind(" ", pos + int(size * 0.8), end))
            end = cut if cut > pos else end
        yield pos, text[pos:end]
        if end >= n:
            break
        pos = max(end - overlap, pos + 1)


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _pick_device(device: str):
    if device == "cpu":
        return -1, "cpu"
    try:
        import torch
        if device in ("auto", "cuda") and torch.cuda.is_available():
            return 0, "cuda"
    except Exception:
        pass
    return -1, "cpu"


class HFTokenDetector:
    """Any Hugging Face token-classification PII model (gravitee-io/bert-small-pii-detection, ...)."""

    def __init__(self, name: str, model_id: str, device: str, threshold: float):
        from transformers import pipeline
        dev, self.device_name = _pick_device(device)
        self.name, self.threshold = name, threshold
        self.pipe = pipeline("token-classification", model=model_id, aggregation_strategy="simple", device=dev)

    def _run(self, chunk: str):
        try:
            return self.pipe(chunk)
        except Exception:
            if len(chunk) < 120:
                return []
            mid = len(chunk) // 2
            left = self._run(chunk[:mid])
            right = [dict(r, start=r["start"] + mid, end=r["end"] + mid) for r in self._run(chunk[mid:])]
            return left + right

    def detect(self, text: str) -> list[RawSpan]:
        out = []
        for off, ch in iter_chunks(text, 600, 80):
            for r in self._run(ch):
                etype, mult = map_model_label(r.get("entity_group") or r.get("entity", ""))
                score = float(r["score"]) * mult
                if etype is None or score < self.threshold:
                    continue
                s, e = _trim(text, r["start"] + off, r["end"] + off)
                if e > s:
                    out.append(RawSpan(s, e, text[s:e], etype, score, f"model:{self.name}"))
        return out


class GlinerDetector:
    """GLiNER-style zero-shot NER (nvidia/gliner-pii, knowledgator/gliner-pii-*, ...)."""

    def __init__(self, name: str, model_id: str, device: str, threshold: float):
        from gliner import GLiNER
        self.name, self.threshold = name, threshold
        self.model = GLiNER.from_pretrained(model_id)
        _, self.device_name = _pick_device(device)
        if self.device_name == "cuda":
            try:
                self.model = self.model.to("cuda")
            except Exception:
                self.device_name = "cpu"
        self.labels = list(GLINER_LABELS)

    def detect(self, text: str) -> list[RawSpan]:
        out = []
        for off, ch in iter_chunks(text, 700, 100):
            for r in self.model.predict_entities(ch, self.labels, threshold=self.threshold):
                etype = GLINER_LABELS.get(r["label"], "OTHER_PII")
                s, e = _trim(text, r["start"] + off, r["end"] + off)
                if e > s:
                    out.append(RawSpan(s, e, text[s:e], etype, float(r["score"]), f"model:{self.name}"))
        return out


class PIITracerDetector:
    """perplexity-ai/PII-Tracer  (AutoModel + trust_remote_code, model.predict(text) -> (spans, sensitivity)).
    Span structure is read defensively; if your version returns something else, adapt _coerce()."""

    def __init__(self, name: str, model_id: str, device: str, threshold: float):
        from transformers import AutoModel
        self.name, self.threshold = name, threshold
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
        _, self.device_name = _pick_device(device)
        if self.device_name == "cuda":
            self.model = self.model.to("cuda")

    @staticmethod
    def _get(sp, *names, default=None):
        for n in names:
            if isinstance(sp, dict) and n in sp:
                return sp[n]
            if hasattr(sp, n):
                return getattr(sp, n)
        return default

    def _coerce(self, sp):
        if isinstance(sp, (tuple, list)) and len(sp) >= 3:
            s, e, lab = sp[0], sp[1], sp[2]
            score = sp[3] if len(sp) > 3 else 0.8
        else:
            s = self._get(sp, "start", "begin")
            e = self._get(sp, "end", "stop")
            lab = self._get(sp, "label", "category", "type", "tag", "entity")
            score = self._get(sp, "score", "confidence", "prob", default=0.8)
        if s is None or e is None or lab is None:
            raise RuntimeError(f"Unrecognised PII-Tracer span format: {sp!r} - adapt PIITracerDetector._coerce")
        return int(s), int(e), str(lab), float(score)

    def detect(self, text: str) -> list[RawSpan]:
        out = []
        for off, ch in iter_chunks(text, 6000, 200):
            spans, _sens = self.model.predict(ch)
            for sp in spans:
                s, e, lab, score = self._coerce(sp)
                etype, mult = map_model_label(lab)
                score *= mult
                if etype is None or score < self.threshold:
                    continue
                s, e = _trim(text, s + off, e + off)
                if e > s:
                    out.append(RawSpan(s, e, text[s:e], etype, score, f"model:{self.name}"))
        return out


MODEL_ALIASES = {
    "gravitee": ("hf", "gravitee-io/bert-small-pii-detection"),
    "gliner": ("gliner", "nvidia/gliner-pii"),
    "pii-tracer": ("tracer", "perplexity-ai/PII-Tracer"),
}
KINDS = {"hf": HFTokenDetector, "gliner": GlinerDetector, "tracer": PIITracerDetector}


def build_detectors(spec: str, device: str, threshold: float):
    """spec: 'gravitee,gliner' | 'none' | 'hf:<repo>' | 'gliner:<repo>' | 'tracer:<repo>' (comma separated)."""
    detectors, notes = [], []
    for tok in [t.strip() for t in spec.split(",") if t.strip() and t.strip().lower() not in ("none", "regex")]:
        if tok.lower() in MODEL_ALIASES:
            kind, mid = MODEL_ALIASES[tok.lower()]
            name = tok.lower()
        elif ":" in tok and tok.split(":", 1)[0] in KINDS:
            kind, mid = tok.split(":", 1)
            name = mid.split("/")[-1]
        else:
            notes.append(f"{tok}: unknown model (use gravitee | gliner | pii-tracer | hf:<repo> | gliner:<repo> | tracer:<repo>)")
            continue
        try:
            t0 = time.time()
            det = KINDS[kind](name, mid, device, threshold)
            detectors.append(det)
            notes.append(f"{name}: loaded on {det.device_name} ({time.time() - t0:.0f}s)")
        except Exception as ex:                                    # missing lib, offline, bad id ...
            notes.append(f"{name}: NOT USED ({type(ex).__name__}: {str(ex)[:140]})")
    return detectors, notes


# =============================================================================
# 3. Detection (+ per-file cache)
# =============================================================================
def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def fingerprint(text: str, det_names: list[str], cfg: Config, threshold: float) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(json.dumps([sorted(det_names), threshold, cfg.raw], sort_keys=True).encode())
    h.update(_file_sha(Path(__file__).with_name("pii_patterns.py")).encode())
    return h.hexdigest()


def detect_entities(text: str, detectors, cfg: Config) -> list[Entity]:
    rx = find_regex_spans(text, BUILTIN_RULES + cfg.rules)
    ml: list[RawSpan] = []
    for d in detectors:
        ml.extend(d.detect(text))
    ents = merge_spans(text, rx, ml)
    return [e for e in ents if e.conf >= MIN_CONF and not cfg.allowed(e)]


def cached_detect(doc: dict, detectors, cfg: Config, threshold: float, use_cache: bool):
    cache = doc["txt"].with_name(doc["txt"].stem + ".pii.json")
    fp = fingerprint(doc["text"], [d.name for d in detectors], cfg, threshold)
    if use_cache and cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fp:
                return [Entity.from_dict(d) for d in data["entities"]], "cache"
        except Exception:
            pass
    ents = detect_entities(doc["text"], detectors, cfg)
    if use_cache:
        cache.write_text(json.dumps({"fingerprint": fp, "entities": [e.to_dict() for e in ents]}), encoding="utf-8")
    return ents, "fresh"


# =============================================================================
# 4. Scoring
# =============================================================================
@dataclass
class Group:
    etype: str
    key: str
    ents: list
    weight: float = 0.0
    base: float = 0.0
    tier: str = "LOW"
    leak: float = 0.0
    leak_kind: str = ""

    @property
    def text(self) -> str:
        return self.ents[0].text

    @property
    def n(self) -> int:
        return len(self.ents)


def assign_weights(ents: list[Entity], cfg: Config) -> None:
    """Per-occurrence weight = base risk x confidence x linkage."""
    ents.sort(key=lambda e: e.start)
    starts = [e.start for e in ents]
    for e in ents:
        base = weight_for(e.etype, cfg)
        cf = max(e.conf, 0.9) if e.validated else e.conf
        lo = bisect.bisect_left(starts, e.start - LINK_WINDOW)
        hi = bisect.bisect_right(starts, e.start + LINK_WINDOW)
        near = [o for o in ents[lo:hi] if o is not e and o.key != e.key]
        has_person = any(o.etype == "PERSON" for o in near) and e.etype != "PERSON"
        id_types = {o.etype for o in near if weight_for(o.etype, cfg) >= 5}
        if e.etype == "PERSON":
            link = 1.2 if id_types else 1.0
        else:
            link = 1.0 + (0.2 if has_person else 0.0) + (0.1 if len(id_types) >= 2 else 0.0)
        e.weight = base * cf * min(link, 1.3)


def make_groups(ents: list[Entity], cfg: Config) -> dict:
    by: dict = defaultdict(list)
    for e in ents:
        by[(e.etype, e.key)].append(e)
    groups = {}
    for (et, key), lst in by.items():
        g = Group(et, key, lst)
        g.base = weight_for(et, cfg)
        g.tier = tier_of(g.base)
        mean_w = sum(x.weight for x in lst) / len(lst)
        g.weight = mean_w * (1 + 0.25 * math.log2(len(lst)))
        groups[(et, key)] = g
    return groups


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"\w+", s) if len(t) >= 3}


def leak_fraction(g: Group, a_fold: str, a_alnum: str, a_by_type: dict, a_keys_by_type: dict):
    et, key = g.etype, g.key
    if et in TEXT_TYPES:
        if len(key) >= 4 and key in a_fold:
            return 1.0, "exact"
        if et == "EMAIL" and "@" in key and len(key.split("@")[0]) >= 5 and key.split("@")[0] in a_fold:
            return 0.5, "local-part"
        toks = _tokens(key)
        if toks and et != "EMAIL":
            best = max((len(toks & _tokens(ag.key)) / len(toks) for ag in a_by_type.get(et, [])), default=0.0)
            if best >= 0.34:
                return best, "partial-tokens"
        return 0.0, ""
    if et == "PARTIAL_ID" or len(key) < 6:
        return (1.0, "exact") if key in a_keys_by_type.get(et, set()) else (0.0, "")
    if key in a_alnum:
        return 1.0, "exact"
    n = len(key)
    if et in PARTIAL_RUN_TYPES and n >= 10:
        need = max(7, math.ceil(0.6 * n))
        for k in range(n - 1, need - 1, -1):
            for i in range(n - k + 1):
                if key[i:i + k] in a_alnum:
                    return k / n, f"partial-run({k}/{n})"
    return 0.0, ""


def strip_spans(text: str, ents: list[Entity]) -> str:
    out, pos = [], 0
    for e in sorted(ents, key=lambda x: x.start):
        if e.start >= pos:
            out.append(text[pos:e.start])
            pos = e.end
    out.append(text[pos:])
    return " ".join(out)


def retention(b_text, b_ents, a_text, a_ents) -> float:
    def words(t):
        return Counter(w.lower() for w in re.findall(r"\w{2,}", t))
    cb = words(strip_spans(b_text, b_ents))
    ca = words(MASK_RE.sub(" ", strip_spans(a_text, a_ents)))
    tot = sum(cb.values())
    return sum(min(v, ca.get(w, 0)) for w, v in cb.items()) / tot if tot else 1.0


def evaluate(b_text, a_text, b_ents, a_ents, cfg: Config, new_factor: float) -> dict:
    assign_weights(b_ents, cfg)
    assign_weights(a_ents, cfg)
    bg, ag = make_groups(b_ents, cfg), make_groups(a_ents, cfg)

    a_fold = normalize_value("PERSON", a_text)
    a_alnum = re.sub(r"[^A-Za-z0-9]", "", normalize_for_detection(a_text)).upper()
    a_by_type: dict = defaultdict(list)
    a_keys_by_type: dict = defaultdict(set)
    for g in ag.values():
        a_by_type[g.etype].append(g)
        a_keys_by_type[g.etype].add(g.key)

    for g in bg.values():
        g.leak, g.leak_kind = leak_fraction(g, a_fold, a_alnum, a_by_type, a_keys_by_type)

    b_keys = {g.key for g in bg.values()}
    b_tokens_by_type: dict = defaultdict(list)
    for g in bg.values():
        b_tokens_by_type[g.etype].append(_tokens(g.key))
    new_groups = []
    for g in ag.values():
        if g.key in b_keys:
            continue
        if g.etype in TEXT_TYPES:
            toks = _tokens(g.key)
            if toks and any(len(toks & bt) / len(toks) >= 0.5 for bt in b_tokens_by_type.get(g.etype, [])):
                continue
        new_groups.append(g)

    w_before = sum(g.weight for g in bg.values())
    residual = sum(g.weight * g.leak for g in bg.values())
    new_exposure = sum(g.weight for g in new_groups) * new_factor
    raw = None if w_before == 0 else max(0.0, 100.0 * (1 - (residual + new_exposure) / w_before))

    leaked = [g for g in bg.values() if g.leak >= 0.5]
    crit = [g for g in leaked if g.tier == "CRITICAL"]
    crit_new = [g for g in new_groups if g.tier == "CRITICAL" and max(e.conf for e in g.ents) >= 0.8]
    high = [g for g in leaked if g.tier == "HIGH"]
    n_crit, n_high = len(crit) + len(crit_new), len(high)

    cap = 100.0
    cap_reason = []
    if n_crit:
        cap = min(cap, max(CRITICAL_FLOOR, CRITICAL_CAP - CRITICAL_STEP * (n_crit - 1)))
        cap_reason.append(f"{n_crit} critical value(s) still exposed")
    if n_high:
        cap = min(cap, max(HIGH_FLOOR, HIGH_CAP - HIGH_STEP * (n_high - 1)))
        cap_reason.append(f"{n_high} high-risk value(s) still exposed")
    score = None if raw is None else min(raw, cap)

    if score is None:
        verdict = "N/A"
    elif score >= PASS_SCORE and not n_crit and not n_high:
        verdict = "PASS"
    elif score >= REVIEW_SCORE:
        verdict = "REVIEW"
    else:
        verdict = "FAIL"

    # per-type table
    types = sorted({g.etype for g in bg.values()} | {g.etype for g in new_groups})
    per_type = []
    for t in types:
        gb = [g for g in bg.values() if g.etype == t]
        gn = [g for g in new_groups if g.etype == t]
        wb = sum(g.weight for g in gb)
        res = sum(g.weight * g.leak for g in gb)
        per_type.append({
            "type": t, "tier": tier_of(weight_for(t, cfg)),
            "before_unique": len(gb), "before_occurrences": sum(g.n for g in gb),
            "leaked_unique": sum(1 for g in gb if g.leak >= 0.5),
            "partially_leaked": sum(1 for g in gb if 0 < g.leak < 0.5),
            "new_in_after": len(gn),
            "weight_before": round(wb, 2), "weight_residual": round(res, 2),
            "reduction_pct": None if wb == 0 else round(100 * (1 - res / wb), 1),
        })
    per_type.sort(key=lambda r: (-TIER_RANK[r["tier"]], -r["weight_before"]))

    unique_b = len(bg)
    count_reduction = None if unique_b == 0 else 100.0 * (1 - sum(g.leak for g in bg.values()) / unique_b)
    mask_b, mask_a = len(MASK_RE.findall(b_text)), len(MASK_RE.findall(a_text))

    return {
        "score": None if score is None else round(score, 1),
        "raw_score_before_caps": None if raw is None else round(raw, 1),
        "cap_applied": None if cap >= 100 else round(cap, 1),
        "cap_reasons": cap_reason,
        "verdict": verdict,
        "weighted_pii_reduction_pct": None if w_before == 0 else round(100 * (1 - residual / w_before), 1),
        "unique_value_reduction_pct": None if count_reduction is None else round(count_reduction, 1),
        "entities_before": len(b_ents), "entities_after": len(a_ents),
        "unique_before": unique_b, "unique_still_leaking": len(leaked),
        "weight_before": round(w_before, 2), "weight_residual": round(residual, 2),
        "weight_new_in_after": round(sum(g.weight for g in new_groups), 2),
        "critical_leaks": n_crit, "high_leaks": n_high,
        "text_retention_pct": round(100 * retention(b_text, b_ents, a_text, a_ents), 1),
        "mask_markers_before": mask_b, "mask_markers_after": mask_a,
        "per_type": per_type,
        "_leaks": sorted([g for g in bg.values() if g.leak > 0], key=lambda g: -g.weight * g.leak),
        "_new": sorted(new_groups, key=lambda g: -g.weight),
    }


# =============================================================================
# 5. Reporting
# =============================================================================
def leak_rows(groups, show_values: bool, limit: Optional[int] = None):
    rows = []
    for g in groups[:limit]:
        e0 = g.ents[0]
        rows.append({
            "type": g.etype, "tier": g.tier, "value": g.text if show_values else mask_preview(g.text),
            "occurrences": g.n, "line_first": e0.line, "leak_fraction": round(g.leak, 2),
            "leak_kind": g.leak_kind, "confidence": round(max(e.conf for e in g.ents), 2),
            "sources": sorted(set().union(*[e.sources for e in g.ents])),
            "weight": round(g.weight, 2),
        })
    return rows


def print_report(rep: dict, b: dict, a: dict, notes: list, show_values: bool, warnings: list):
    line = "=" * 74
    print(f"\n{line}\n PII REDACTION EVALUATION\n{line}")
    print(f" before : {b['md'].name}  ->  {b['txt'].name}   [{b['status']}]")
    print(f" after  : {a['md'].name}  ->  {a['txt'].name}   [{a['status']}]")
    for n in notes:
        print(f" detector  {n}")
    for w in warnings:
        print(f" WARNING   {w}")
    if rep["score"] is None:
        print("\n No PII was detected in the BEFORE document, so there is nothing to compare.")
        return
    print(f"\n SCORE   {rep['score']:5.1f} / 100      VERDICT  {rep['verdict']}")
    if rep["cap_applied"] is not None:
        print(f"         (raw {rep['raw_score_before_caps']}, capped at {rep['cap_applied']}: {'; '.join(rep['cap_reasons'])})")
    print(f"\n weighted PII reduction : {rep['weighted_pii_reduction_pct']}%")
    print(f" unique-value reduction : {rep['unique_value_reduction_pct']}%   "
          f"({rep['unique_still_leaking']} of {rep['unique_before']} unique values still visible)")
    print(f" entities before/after  : {rep['entities_before']} / {rep['entities_after']}")
    print(f" risk weight            : before {rep['weight_before']}  residual {rep['weight_residual']}  "
          f"new-in-after {rep['weight_new_in_after']}")
    print(f" leaked critical / high : {rep['critical_leaks']} / {rep['high_leaks']}")
    print(f" non-PII text retained  : {rep['text_retention_pct']}%   "
          f"(mask markers {rep['mask_markers_before']} -> {rep['mask_markers_after']})")

    print(f"\n {'TYPE':<14}{'TIER':<9}{'BEFORE':>7}{'LEAKED':>8}{'PARTIAL':>9}{'NEW':>5}{'REDUCED':>9}")
    for r in rep["per_type"]:
        red = "-" if r["reduction_pct"] is None else f"{r['reduction_pct']}%"
        print(f" {r['type']:<14}{r['tier']:<9}{r['before_unique']:>7}{r['leaked_unique']:>8}"
              f"{r['partially_leaked']:>9}{r['new_in_after']:>5}{red:>9}")

    leaks = leak_rows(rep["_leaks"], show_values, 15)
    if leaks:
        print("\n TOP LEAKS (still visible in AFTER)")
        for r in leaks:
            print(f"  [{r['tier']:<8}] {r['type']:<13} {r['value']:<26} x{r['occurrences']:<3} "
                  f"line {r['line_first']:<5} leak {r['leak_fraction']:<4} ({r['leak_kind']})")
    new = leak_rows(rep["_new"], show_values, 8)
    if new:
        print("\n NEW / UNMATCHED PII in AFTER (missed by BEFORE detection, or introduced)")
        for r in new:
            print(f"  [{r['tier']:<8}] {r['type']:<13} {r['value']:<26} x{r['occurrences']:<3} "
                  f"line {r['line_first']:<5} conf {r['confidence']}")
    print()


# =============================================================================
# 6. CLI
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score how much PII a redaction removed (before.md vs after.md).")
    ap.add_argument("before", help="markdown (or .txt) BEFORE redaction")
    ap.add_argument("after", help="markdown (or .txt) AFTER redaction")
    ap.add_argument("--models", default="gravitee,gliner",
                    help="comma list: gravitee, gliner, pii-tracer, hf:<repo>, gliner:<repo>, tracer:<repo>, or 'none' "
                         "(default gravitee,gliner; missing models are skipped with a note)")
    ap.add_argument("--threshold", type=float, default=0.5, help="min model confidence (default 0.5)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--config", help="JSON: extra regex rules, weights, allowlist (see docstring of pii_patterns)")
    ap.add_argument("--table-style", choices=["aligned", "tsv", "pipe"], default="aligned")
    ap.add_argument("--force-convert", action="store_true", help="re-run md -> txt even if a valid txt exists")
    ap.add_argument("--no-cache", action="store_true", help="do not read/write <name>.pii.json detection caches")
    ap.add_argument("--new-exposure-factor", type=float, default=0.5,
                    help="how much PII found only in AFTER counts against the score (default 0.5)")
    ap.add_argument("--show-values", action="store_true", help="print/store real PII values (default: masked previews)")
    ap.add_argument("--out-dir", help="where to write the .eval.json (default: next to the AFTER file)")
    ap.add_argument("--fail-under", type=float, help="exit code 1 if score is below this (for CI)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    b = prepare_text(args.before, args.table_style, args.force_convert)
    a = prepare_text(args.after, args.table_style, args.force_convert)

    warnings = []
    if b["md_sha"] == a["md_sha"]:
        warnings.append("before and after files are identical - nothing was redacted")
    sb, sa = b["md"].stem.lower(), a["md"].stem.lower()
    common = len(__import__("os").path.commonprefix([sb, sa]))
    if common < min(4, len(sb), len(sa)) and difflib.SequenceMatcher(None, sb, sa).ratio() < 0.35:
        warnings.append(f"file names look unrelated ('{b['md'].stem}' vs '{a['md'].stem}') - "
                        f"make sure both belong to the same PDF")

    detectors, notes = build_detectors(args.models, args.device, args.threshold)
    notes.insert(0, "regex+checksum: always on")
    use_cache = not args.no_cache
    b_ents, b_src = cached_detect(b, detectors, cfg, args.threshold, use_cache)
    a_ents, a_src = cached_detect(a, detectors, cfg, args.threshold, use_cache)
    notes.append(f"entities: before from {b_src}, after from {a_src}")

    rep = evaluate(b["text"], a["text"], b_ents, a_ents, cfg, args.new_exposure_factor)
    if rep["text_retention_pct"] < 50:
        warnings.append(f"only {rep['text_retention_pct']}% of the non-PII text survives - wrong pair of files, "
                        f"or over-redaction")
    elif rep["text_retention_pct"] < 85:
        warnings.append(f"only {rep['text_retention_pct']}% of the non-PII text survives - possible over-redaction")

    print_report(rep, b, a, notes, args.show_values, warnings)

    out_dir = Path(args.out_dir) if args.out_dir else a["md"].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a['md'].stem}__vs__{b['md'].stem}.eval.json"
    payload = {k: v for k, v in rep.items() if not k.startswith("_")}
    payload.update({
        "before": str(b["md"]), "after": str(a["md"]),
        "detectors": notes, "warnings": warnings,
        "settings": {"models": args.models, "threshold": args.threshold,
                     "new_exposure_factor": args.new_exposure_factor, "config": args.config},
        "leaks": leak_rows(rep["_leaks"], args.show_values),
        "new_in_after": leak_rows(rep["_new"], args.show_values),
    })
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" report saved -> {out}")

    if args.fail_under is not None and (rep["score"] is None or rep["score"] < args.fail_under):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
