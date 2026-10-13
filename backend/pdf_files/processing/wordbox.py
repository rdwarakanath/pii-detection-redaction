"""Word-level boxes for PII redaction.

Idea: TEXT comes from the best engine for the block (PP-StructureV3 / PaddleOCR-VL / native layer), GEOMETRY comes from a
locator that returns word boxes (Tesseract run inside the block crop, or the exact native PDF word boxes).  The two are
joined by a character-level alignment, so the locator's misreads, glued words and split words do not matter:

    "Documentcontrol" (VL)  <->  "Document" "control" (Tesseract)  ->  one token, one box spanning both words
    "Version6.0"      (VL)  <->  "Version6.0" (Tesseract)           ->  fine
    "Version 6.0"     (VL)  <->  "Version6.0" (Tesseract)           ->  two tokens, Tesseract's box split proportionally
    "EMP-38104"       (VL)  <->  "EMP-3B104"  (Tesseract)           ->  still aligned (8 of 9 characters match)

Every output word carries `start`/`end` offsets into the page-level `words_text`, so PII spans found in that string map
straight to boxes.  Tokens that cannot be aligned are NEVER dropped: they get the box of the whole OCR line they sit in
(over-redaction is the fail-safe) plus a proportional estimate (`est_bbox_px`), and are marked in `status`.

Word dict fields
    text        token text as read by the text engine (or the locator text for status "locator-only")
    start, end  character offsets in the page `words_text`
    bbox_px     box in the engine raster (None for native words); bbox_pdf in unrotated PDF points (always set)
    source      native | aligned | fallback
    status      exact | ok | split | weak | fallback-line | fallback-block | locator-only
    locator_conf  mean Tesseract word confidence 0-100 of the words behind the box (None for native). It is a Tesseract
                number: never compare it with Paddle scores.
    block       StructBlock idx (-1 = not inside any block);  part/parts: a token wrapped over a line break has one entry
                per line
"""
from __future__ import annotations

import difflib
import html as _html
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]

_PAIR_LIMIT = 4_000_000          # len(ref chars) * len(locator chars) above which the anchored (two-stage) alignment is used
_MIN_COVER_OK = 0.5              # share of a token's alphanumerics that must be matched for status ok/split (else weak)
_TAG = re.compile(r"<[^>]+>")


# ------------------------------------------------------------------ text helpers
def clean_text(text: str) -> str:
    """Block text -> the string offsets refer to: HTML tags removed (tables / VL output), nothing else changed."""
    t = text or ""
    return _TAG.sub(" ", t) if "<" in t else t


def _norm_chars(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


# ------------------------------------------------------------------ alignment
@dataclass
class LocWord:
    text: str
    box: Box
    line: int = 0                 # unique line key (words of one OCR line share it)
    conf: Optional[float] = None  # locator confidence (Tesseract 0-100); None for native words


def _pairs(r: str, t: str) -> List[Tuple[int, int]]:
    """Matched (ref index, locator index) character pairs between two normalised strings."""
    if not r or not t:
        return []
    sm = difflib.SequenceMatcher(None, r, t, autojunk=False)
    out: List[Tuple[int, int]] = []
    for a, b, n in sm.get_matching_blocks():
        out.extend((a + k, b + k) for k in range(n))
    return out


def _anchored_pairs(r: str, r_owner: List[int], r_tok_norm: List[str],
                    t: str, t_owner: List[int], t_word_norm: List[str]) -> List[Tuple[int, int]]:
    """Large blocks: anchor on identical whole words first (cheap, list-level), then align characters in the gaps."""
    r_start, t_start = {}, {}
    pos = 0
    for j, s in enumerate(r_tok_norm):
        r_start[j] = pos
        pos += len(s)
    pos = 0
    for i, s in enumerate(t_word_norm):
        t_start[i] = pos
        pos += len(s)
    sm = difflib.SequenceMatcher(None, r_tok_norm, t_word_norm, autojunk=False)
    out: List[Tuple[int, int]] = []
    ra = ta = 0                                            # char cursors (end of last anchor)
    for a, b, n in sm.get_matching_blocks():
        if n:
            ca = r_start[a]
            cb = t_start[b]
            gap_r, gap_t = r[ra:ca], t[ta:cb]
            if len(gap_r) * len(gap_t) <= _PAIR_LIMIT:
                out.extend((ra + i, ta + j) for i, j in _pairs(gap_r, gap_t))
            ea = r_start[a + n - 1] + len(r_tok_norm[a + n - 1])
            eb = t_start[b + n - 1] + len(t_word_norm[b + n - 1])
            out.extend(zip(range(ca, ea), range(cb, eb)))
            ra, ta = ea, eb
    gap_r, gap_t = r[ra:], t[ta:]
    if len(gap_r) * len(gap_t) <= _PAIR_LIMIT:
        out.extend((ra + i, ta + j) for i, j in _pairs(gap_r, gap_t))
    return out


def align(ref_text: str, loc: Sequence[LocWord]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Align reference tokens (whitespace separated) to locator words.

    Returns (tokens, unclaimed): tokens in reading order, each
        {text, start, end, cover, pieces: [(line_key, box, conf, split_flag)], status}
    with `pieces` empty when nothing matched; `unclaimed` = indices of locator words no token matched at all."""
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", ref_text)]
    r_norm_tok = [_norm_chars(t) for t, _, _ in toks]
    t_norm_word = [_norm_chars(w.text) for w in loc]
    r = "".join(r_norm_tok)
    t = "".join(t_norm_word)
    r_owner = [j for j, s in enumerate(r_norm_tok) for _ in s]
    t_owner = [i for i, s in enumerate(t_norm_word) for _ in s]
    t_local = [k for s in t_norm_word for k in range(len(s))]     # char index inside its locator word

    pairs = (_pairs(r, t) if len(r) * len(t) <= _PAIR_LIMIT
             else _anchored_pairs(r, r_owner, r_norm_tok, t, t_owner, t_norm_word))

    # per token: {locator word -> [char indices inside that word]};  per locator word: claimant tokens
    claim: Dict[int, Dict[int, List[int]]] = {}
    for ri, ti in pairs:
        claim.setdefault(r_owner[ri], {}).setdefault(t_owner[ti], []).append(t_local[ti])
    by_word: Dict[int, List[Tuple[int, int, int]]] = {}        # locator word -> [(kmin, kmax, token)]
    for j, d in claim.items():
        for i, ks in d.items():
            by_word.setdefault(i, []).append((min(ks), max(ks), j))

    # x-extent of every (token, locator word): whole box if the token is the only claimant, else a proportional slice
    seg: Dict[Tuple[int, int], Box] = {}
    split_flag: Dict[Tuple[int, int], bool] = {}
    for i, cl in by_word.items():
        x0, y0, x1, y1 = loc[i].box
        L = max(len(t_norm_word[i]), 1)
        if len(cl) == 1:
            seg[(cl[0][2], i)] = (x0, y0, x1, y1)
            split_flag[(cl[0][2], i)] = False
            continue
        cl = sorted(cl, key=lambda c: (c[0], c[1]))
        cuts = [0.0]
        for a, b in zip(cl, cl[1:]):
            cuts.append((a[1] + 1 + b[0]) / 2.0)               # midway between one token's last and the next one's first char
        cuts.append(float(L))
        w = x1 - x0
        for n, (kmin, kmax, j) in enumerate(cl):
            seg[(j, i)] = (x0 + w * cuts[n] / L, y0, x0 + w * cuts[n + 1] / L, y1)
            split_flag[(j, i)] = True

    tokens: List[Dict[str, Any]] = []
    for j, (text, s, e) in enumerate(toks):
        n_alnum = len(r_norm_tok[j])
        d = claim.get(j, {})
        matched = sum(len(v) for v in d.values())
        cover = (matched / n_alnum) if n_alnum else 0.0
        # one piece per locator line (a token wrapped over a line break is two boxes, not one giant rectangle)
        by_line: Dict[int, List[int]] = {}
        for i in d:
            by_line.setdefault(loc[i].line, []).append(i)
        pieces = []
        for lk, ids in sorted(by_line.items(), key=lambda kv: min(loc[i].box[1] for i in kv[1])):
            bs = [seg[(j, i)] for i in ids]
            confs = [loc[i].conf for i in ids if loc[i].conf is not None]
            pieces.append((lk, (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)),
                           (sum(confs) / len(confs)) if confs else None, any(split_flag[(j, i)] for i in ids)))
        if n_alnum == 0:
            status = "punct"                                    # pure punctuation: no box of its own
        elif not pieces:
            status = "unaligned"
        elif cover < _MIN_COVER_OK:
            status = "weak"
        else:
            status = "split" if any(p[3] for p in pieces) else "ok"
        tokens.append({"text": text, "start": s, "end": e, "cover": round(cover, 3), "pieces": pieces, "status": status})
    claimed = set(by_word)
    return tokens, [i for i in range(len(loc)) if t_norm_word[i] and i not in claimed]


# ------------------------------------------------------------------ fallback boxes for tokens the locator could not place
def _line_for_token(token: str, lines: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], int]]:
    """OCR line (from the text engine) that contains the token, plus the token's character offset in that line's text."""
    tn = _norm_chars(token)
    if not tn:
        return None
    for ln in lines:
        text = ln.get("text") or ""
        if tn in _norm_chars(text):
            m = re.search(re.escape(token), text)
            if m:
                return ln, m.start()
            return ln, -1
    return None


def _estimate(ln: Dict[str, Any], off: int, token: str) -> Optional[Box]:
    """Proportional position of a token inside its line box (fonts are not monospaced: this is an estimate, padded)."""
    box = ln.get("bbox_px")
    text = ln.get("text") or ""
    if not box or off < 0 or not text:
        return None
    x0, y0, x1, y1 = box
    cw = (x1 - x0) / max(len(text), 1)
    return (max(x0, x0 + off * cw - 2 * cw), y0, min(x1, x0 + (off + len(token)) * cw + 2 * cw), y1)


# ------------------------------------------------------------------ block / page assembly
def words_for_block(*, block_idx: int, text: str, bbox: Box, line_boxes: List[Dict[str, Any]],
                    loc: Sequence[LocWord], to_pdf: Optional[Callable[..., Box]], native: bool,
                    offset: int = 0, min_extra_alnum: int = 2) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    """One block -> (cleaned text, word dicts, counters).  `loc` is in raster px unless `native` (then PDF points)."""
    ref = clean_text(text)
    tokens, unclaimed = align(ref, loc)
    words: List[Dict[str, Any]] = []
    cnt = {"tokens": 0, "ok": 0, "split": 0, "weak": 0, "fallback": 0, "locator_only": 0}

    def emit(w: Dict[str, Any]) -> None:
        words.append(w)

    def make(token, start, end, box_px, status, source, conf, part=0, parts=1, est=None):
        if native:
            bpdf, bpx = tuple(round(v, 2) for v in box_px), None
        else:
            bpx = tuple(round(v, 1) for v in box_px)
            bpdf = tuple(round(v, 2) for v in to_pdf(*box_px)) if to_pdf else None
        d = {"text": token, "start": offset + start, "end": offset + end, "bbox_px": list(bpx) if bpx else None,
             "bbox_pdf": list(bpdf) if bpdf else None, "source": source, "status": status,
             "locator_conf": None if conf is None else round(conf, 1), "block": block_idx, "part": part, "parts": parts}
        if est is not None:
            d["est_bbox_px"] = [round(v, 1) for v in est]
        return d

    for tk in tokens:
        if tk["status"] == "punct":
            continue
        cnt["tokens"] += 1
        weak = tk["status"] in ("unaligned", "weak")
        if not weak:
            pcs = tk["pieces"]
            for n, (_, box, conf, _) in enumerate(pcs):
                emit(make(tk["text"], tk["start"], tk["end"], box,
                          "exact" if native else tk["status"], "native" if native else "aligned", conf, n, len(pcs)))
            cnt["split" if tk["status"] == "split" else "ok"] += 1
            continue
        # fail-safe: the whole OCR line (or the block) is the box; keep a proportional estimate next to it
        hit = None if native else _line_for_token(tk["text"], line_boxes)
        if hit and hit[0].get("bbox_px"):
            ln, off = hit
            emit(make(tk["text"], tk["start"], tk["end"], tuple(ln["bbox_px"]), "fallback-line", "fallback", None,
                      est=_estimate(ln, off, tk["text"])))
        else:
            emit(make(tk["text"], tk["start"], tk["end"], tuple(bbox), "fallback-block", "fallback", None))
        cnt["weak" if tk["status"] == "weak" else "fallback"] += 1

    # locator words no token claimed: text the engine did not return (possible missed content) -> keep, flagged
    extra_text = ""
    base = offset + len(ref)
    for i in unclaimed:
        w = loc[i]
        if sum(ch.isalnum() for ch in w.text) < min_extra_alnum:
            continue
        s = base + len(extra_text) + 1
        extra_text += "\n" + w.text
        words.append(make(w.text, s - offset, s - offset + len(w.text), w.box, "locator-only",
                          "native" if native else "aligned", w.conf))
        cnt["locator_only"] += 1
    return ref + extra_text, words, cnt


def locate_block(img: np.ndarray, bbox: Box, runner, cfg, *, label: str = "text", n_lines: int = 0,
                 margin: int = 6) -> List[LocWord]:
    """Run Tesseract inside one block crop; return word boxes in the pixel space of `img` (only words centred in the block)."""
    from . import preprocess as pp
    from .ocr_engine import Attempt
    import cv2
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    x0, y0 = max(int(bbox[0]) - margin, 0), max(int(bbox[1]) - margin, 0)
    x1, y1 = min(int(bbox[2]) + margin + 1, W), min(int(bbox[3]) + margin + 1, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return []
    crop = np.ascontiguousarray(g[y0:y1, x0:x1])
    tbl = label == "table"
    first = Attempt("wb/psm7", "norm", 7) if (n_lines == 1 and not tbl) else Attempt("wb/psm6", "norm", 6, remove_lines=tbl)
    ladder = [first,
              Attempt("wb/clahe psm6", "clahe", 6, scale=1.4, remove_lines=tbl),
              Attempt("wb/sauvola psm6", "sauvola", 6, scale=1.8, remove_lines=tbl)]
    words, _, _ = runner.ladder(crop, ladder, cfg.accept_score)
    out: List[LocWord] = []
    tol = 3.0
    for w in words:
        bx = (w.x0 + x0, w.y0 + y0, w.x1 + x0, w.y1 + y0)
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        if bbox[0] - tol <= cx <= bbox[2] + tol and bbox[1] - tol <= cy <= bbox[3] + tol and w.text.strip():
            out.append(LocWord(w.text, bx, w.line, w.conf))
    out.sort(key=lambda lw: (lw.line, lw.box[0]))
    # reading order: lines top-to-bottom (by their first word's top), words left-to-right
    first_y: Dict[int, float] = {}
    for lw in out:
        first_y[lw.line] = min(first_y.get(lw.line, 1e9), lw.box[1])
    out.sort(key=lambda lw: (round(first_y[lw.line] / 4), lw.line, lw.box[0]))
    # Tesseract boxes hug the ink ("owner:" has no ascenders, "Policy" has both): give every word of a line the line's full
    # vertical extent so redaction boxes cover ascenders/descenders and look uniform. Horizontal extents stay tight.
    top: Dict[int, float] = {}
    bot: Dict[int, float] = {}
    for lw in out:
        top[lw.line] = min(top.get(lw.line, 1e9), lw.box[1])
        bot[lw.line] = max(bot.get(lw.line, -1e9), lw.box[3])
    return [LocWord(lw.text, (lw.box[0], top[lw.line], lw.box[2], bot[lw.line]), lw.line, lw.conf) for lw in out]


def native_loc_words(native: list, box_pdf: Box) -> List[LocWord]:
    """Exact word boxes from the trusted PDF text layer whose centre lies in `box_pdf` (reading order by line, then x)."""
    ins = [w for w in native if box_pdf[0] <= (w[0] + w[2]) / 2 <= box_pdf[2] and box_pdf[1] <= (w[1] + w[3]) / 2 <= box_pdf[3]]
    lines: Dict[Tuple[int, int], list] = {}
    for w in ins:
        lines.setdefault((w[5], w[6]), []).append(w)
    ordered = sorted(lines.items(), key=lambda kv: (round(kv[1][0][1] / 4), kv[1][0][0]))
    out: List[LocWord] = []
    for n, (_, ws) in enumerate(ordered):
        for w in sorted(ws, key=lambda x: x[0]):
            out.append(LocWord(w[4], (w[0], w[1], w[2], w[3]), n, None))
    return out


def native_page_words(native: list) -> Tuple[str, List[Dict[str, Any]]]:
    """Digital / native-only page: every native word is exact.  Text is built block -> line -> word."""
    text, words = "", []
    last_block = None
    for blk, ln, ws in _group_native(native):
        sep = "" if not text else ("\n\n" if blk != last_block else "\n")
        text += sep
        for k, w in enumerate(ws):
            if k:
                text += " "
            words.append({"text": w[4], "start": len(text), "end": len(text) + len(w[4]), "bbox_px": None,
                          "bbox_pdf": [round(v, 2) for v in w[:4]], "source": "native", "status": "exact",
                          "locator_conf": None, "block": -1, "part": 0, "parts": 1})
            text += w[4]
        last_block = blk
    return text, words


def _group_native(native: list):
    groups: Dict[Tuple[int, int], list] = {}
    for w in native:
        if str(w[4]).strip():
            groups.setdefault((w[5], w[6]), []).append(w)
    for (blk, ln), ws in sorted(groups.items()):
        yield blk, ln, sorted(ws, key=lambda x: x[7] if len(x) > 7 else x[0])


def attach_words(blocks: list, raster, runner, cfg, native: Optional[list] = None,
                 page_type: str = "scanned") -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Build page-level (words_text, words, stats, warnings) for an OCR page. Never raises for a single bad block."""
    from .ocr_engine import block_plain_text
    native = native or []
    text_parts: List[str] = []
    words: List[Dict[str, Any]] = []
    stats = {"tokens": 0, "ok": 0, "split": 0, "weak": 0, "fallback": 0, "locator_only": 0, "blocks_failed": 0}
    warns: List[str] = []
    offset = 0
    for b in blocks:
        if b.label in ("image", "header_image", "footer_image", "chart"):
            continue
        plain = block_plain_text(b)
        if not plain.strip():
            continue
        try:
            if b.source == "native" and b.bbox_pdf:
                loc = native_loc_words(native, tuple(b.bbox_pdf))
                is_native, to_pdf = True, None
            else:
                loc = locate_block(raster.img, b.bbox, runner, cfg, label=b.label, n_lines=len(b.line_boxes))
                is_native, to_pdf = False, raster.geo.box
            txt, ws, cnt = words_for_block(block_idx=b.idx, text=plain, bbox=b.bbox, line_boxes=b.line_boxes, loc=loc,
                                           to_pdf=to_pdf, native=is_native, offset=offset)
        except Exception as e:                                   # one block must not cost the page its other words
            stats["blocks_failed"] += 1
            warns.append(f"word boxes failed for block {b.idx}: {type(e).__name__}: {str(e)[:100]}")
            log.warning("word boxes failed for block %s", b.idx, exc_info=True)
            continue
        for k in cnt:
            stats[k] += cnt[k]
        words += ws
        text_parts.append(txt)
        offset += len(txt) + 2                                   # blocks are joined by a blank line
    # hybrid pages: trusted native words outside every block (stamps, overlays) must not escape redaction
    if page_type == "hybrid" and native:
        rest = [w for w in native if str(w[4]).strip() and not any(
            bb.bbox_pdf and bb.bbox_pdf[0] <= (w[0] + w[2]) / 2 <= bb.bbox_pdf[2]
            and bb.bbox_pdf[1] <= (w[1] + w[3]) / 2 <= bb.bbox_pdf[3] for bb in blocks)]
        if rest:
            t, ws = native_page_words(rest)
            for w in ws:
                w["start"] += offset
                w["end"] += offset
            words += ws
            text_parts.append(t)
            stats["native_outside_blocks"] = len(ws)
    n_weak = stats["weak"] + stats["fallback"]
    if n_weak:
        warns.append(f"word boxes: {n_weak} token(s) could not be placed on a locator word and carry a whole-line box "
                     "(fail-safe over-redaction); see status fallback-line / fallback-block / weak")
    return "\n\n".join(text_parts), words, stats, warns
