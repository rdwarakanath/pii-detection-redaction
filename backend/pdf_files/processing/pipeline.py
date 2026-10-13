"""PDF -> Markdown using PP-StructureV3, lazy PaddleOCR-VL, then Tesseract as the last-resort OCR tier.

Per page:
  classify (digital / scanned / hybrid / vector / blank)
    digital + trusted text layer ........ native text (no OCR)
    everything else ..................... render (300 DPI) -> orient/deskew -> PP-StructureV3
  quality gate (text-RECOGNITION confidence >= 85% + completeness / junk / spacing / order / table / uncovered-ink checks)
    pass ................................ keep the PP-StructureV3 page untouched
    fail -> release PP model, then lazily load PaddleOCR-VL
      layout failed / PP crashed ....... VL rereads whole page; accept only if content checks pass
      layout usable, weak blocks ....... VL rereads only weak block crops and verifies each result
      VL rejected / load fails / weak .. release VL, then use positioned Tesseract as the last-resort OCR tier
    layout "failed" ONLY because ink lies outside every block (nothing structural wrong): the Paddle structure is kept; the
      fallback re-reads the page (vl) or the failing blocks, and the page is flagged as possibly incomplete
  every candidate weak -> retry the page at higher DPI with targeted preprocessing; every attempt is kept
  still weak -> flagged `needs_review`

Markdown is the only deliverable. Block / attempt JSON is internal (`<out>/_internal/`), for QA and debugging.
Errors that mean an engine is broken (not one bad page) raise ocr_engine.EngineUnavailableError and stop the run.
"""
from __future__ import annotations
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf

from . import preprocess as pp
from . import textlayer
from . import wordbox
from .config import Config
from .models import PageResult, Word
from .ocr_engine import (EngineUnavailableError, GateResult, PaddleSettings, Runner, StructBlock, StructPage,
                         assemble_markdown, block_markdown, block_plain_text, get_structure_engine, get_vl_engine,
                         is_clear_improvement, lines_to_text, match_lines_to_blocks, quality_gate,
                         release_structure_engine, release_vl_engine, tesseract_lines, tesseract_rebuild, text_contained, _inter_frac,
                         text_quality, verify_vl_page, vl_repair_blocks)
log = logging.getLogger(__name__)

# Labels for blocks that are peripheral to the document's substantive content.
# When the ONLY remaining weak blocks are peripheral (headers, footers, page numbers),
# the page is NOT flagged as needs_review — the body text is fine.
_PERIPHERAL_LABELS = {"header", "footer", "number"}


def _split_weak(still_weak: List[int], blocks: List[StructBlock]) -> tuple:
    """Partition weak block indices into (substantive, peripheral).
    Returns (substantive_weak, peripheral_weak, peripheral_only)."""
    by_idx = {b.idx: b for b in blocks}
    substantive = [i for i in still_weak if by_idx.get(i) is None or by_idx[i].label not in _PERIPHERAL_LABELS]
    peripheral = [i for i in still_weak if by_idx.get(i) is not None and by_idx[i].label in _PERIPHERAL_LABELS]
    return substantive, peripheral, (len(substantive) == 0 and len(peripheral) > 0)


# ====================================================================== result type
@dataclass
class PageMarkdown:
    page_no: int                                   # 1-based; original page identity
    page_type: str                                 # digital | scanned | hybrid | vector | blank
    markdown: str = ""
    source: str = ""                               # native | paddle | paddle+tesseract | paddle+vl | vl | tesseract-rebuild | blank | error
    structure: str = "ok"                          # ok | best-effort | native
    score: Optional[float] = 100.0                 # Paddle page text-RECOGNITION confidence x100; 100 for native / blank;
                                                   # None when the text came from an engine without such a number (vl / rebuild)
    warnings: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)            # Markdown of each table (count shown by the CLI)
    stats: Dict[str, Any] = field(default_factory=dict)
    attempts: List[Dict[str, Any]] = field(default_factory=list)   # internal: every attempt + selection reason
    blocks: List[Dict[str, Any]] = field(default_factory=list)     # internal: final blocks with px + PDF boxes
    words: List[Dict[str, Any]] = field(default_factory=list)      # word boxes (see wordbox.py); start/end index `words_text`
    words_text: str = ""                                           # the string PII detection should run on for this page


# ====================================================================== classification (unchanged logic)
def classify(page: pymupdf.Page, cfg: Config) -> dict:
    words = page.get_text("words")
    text = page.get_text()
    invisible = []
    try:
        for sp in page.get_texttrace():
            if sp.get("type") == 3:
                invisible.append(pymupdf.Rect(sp["bbox"]))
    except Exception:
        pass
    vis = []
    for w in words:
        r = pymupdf.Rect(w[:4])
        c = pymupdf.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        if any(c in ir for ir in invisible) and not cfg.trust_invisible_text:
            continue
        vis.append(w)
    n_bad = text.count("\ufffd") + text.count("(cid:") * 6 + sum(1 for ch in text if 0xE000 <= ord(ch) <= 0xF8FF)
    garbage = n_bad / max(len(text), 1)
    untrusted = garbage > cfg.garbage_max
    if untrusted:
        vis = []
    area = page.rect.get_area() or 1
    img_cov = 0.0
    try:
        img_cov = min(1.0, sum(pymupdf.Rect(i["bbox"]).intersect(page.rect).get_area() for i in page.get_image_info()) / area)
    except Exception:
        pass
    try:
        n_draw = len(page.get_drawings())
    except Exception:
        n_draw = 0
    n = len(vis)
    if n >= cfg.min_native_words and img_cov < cfg.hybrid_image_cov and n_draw < cfg.drawing_heavy:
        ptype = "digital"
    elif n >= cfg.min_native_words:
        ptype = "hybrid"
    elif img_cov >= 0.3:
        ptype = "scanned"
    elif n_draw >= cfg.drawing_heavy:
        ptype = "vector"
    else:
        ptype = "scanned" if img_cov > 0.01 or n == 0 else "hybrid"
    return {"type": ptype, "native": vis, "img_cov": img_cov, "drawings": n_draw,
            "garbage": garbage, "untrusted": untrusted, "n_invisible": len(invisible)}


# ====================================================================== geometry / rendering
class Geo:
    """working-raster px  ->  unrotated PDF points (original page coordinates)."""
    def __init__(self, page: pymupdf.Page, scale: float, A_inv: np.ndarray):
        self.scale, self.A, self.derot = scale, A_inv, page.derotation_matrix

    def box(self, x0, y0, x1, y1) -> Tuple[float, float, float, float]:
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], float) @ self.A.T
        ps = [pymupdf.Point(p[0] / self.scale, p[1] / self.scale) * self.derot for p in pts]
        xs, ys = [p.x for p in ps], [p.y for p in ps]
        return (min(xs), min(ys), max(xs), max(ys))


def render_gray(page: pymupdf.Page, dpi: int) -> np.ndarray:
    s = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csGRAY, alpha=False)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    return np.ascontiguousarray(a)


@dataclass
class Raster:
    img: np.ndarray            # image sent to BOTH engines (polarity-fixed, optional targeted preprocessing)
    geo: Geo
    dpi: int
    variant: Optional[str]
    ink: float


def make_raster(page: pymupdf.Page, info: dict, cfg: Config, dpi: int, variant: Optional[str] = None,
                rot: Optional[int] = None, skew: Optional[float] = None, log: Optional[list] = None
                ) -> Tuple[Raster, int, float]:
    """Render -> orientation (OSD) -> deskew -> polarity fix -> optional variant.
    `rot` / `skew` are detected once (first call) and reused for retries so every DPI shares one geometry."""
    work = render_gray(page, dpi)
    A = np.eye(3)
    if rot is None:
        rot = 0
        if info["type"] in ("scanned", "hybrid") and cfg.detect_orientation:
            try:
                import pytesseract
                osd = pytesseract.image_to_osd(pp.resize(work, 0.5), config="-c min_characters_to_try=40",
                                               output_type=pytesseract.Output.DICT)
                r, oc = int(osd.get("rotate", 0)), float(osd.get("orientation_conf", 0))
                if r in (90, 180, 270) and oc >= 3.0:
                    rot = r
                    if log is not None:
                        log.append({"stage": "orientation", "rotate_cw": r, "conf": oc})
            except Exception:
                pass
    if rot:
        work, M = pp.rotate_k90(work, rot // 90)
        A = A @ M
    if skew is None:
        skew = 0.0
        if info["type"] == "scanned" and cfg.deskew:
            ang = pp.estimate_skew(work, cfg.deskew_max_deg)
            if abs(ang) >= cfg.deskew_min_deg:
                skew = ang
                if log is not None:
                    log.append({"stage": "deskew", "deg": round(ang, 2)})
    if skew:
        work, M = pp.rotate_affine(work, skew)
        A = A @ M
    ink = pp.ink_fraction(work)
    img = work
    if cfg.fix_polarity:
        rev = pp.find_reverse_blocks(work, cfg.dark_fill_thresh, cfg.dark_min_w, cfg.dark_min_h, cfg.dark_min_fill)
        if rev:
            img = pp.invert_blocks(work, rev)
    if variant:
        img = pp.make_variant(img, variant)
    return Raster(img, Geo(page, dpi / 72.0, A), dpi, variant, ink), rot, skew


# ====================================================================== native text helpers
def native_markdown(page: pymupdf.Page, info: dict) -> str:
    """Trusted text layer -> Markdown via the existing reading-order logic (no OCR)."""
    res = PageResult(0, page.rect.width, page.rect.height, page.rotation, info["type"])
    for w in info["native"]:
        res.words.append(Word(w[4], 100.0, tuple(w[:4]), "native", block=w[5], line=w[5] * 1000 + w[6]))
    return textlayer.page_markdown(res)


def apply_native_text(blocks: List[StructBlock], native: list, geo: Geo, ps: PaddleSettings) -> int:
    """Hybrid pages: where the trusted text layer covers a text block, use it instead of OCR text."""
    n_used = 0
    for b in blocks:
        if b.label in ("table", "image", "chart", "formula"):
            continue
        box = geo.box(*b.bbox)
        inside = [w for w in native if box[0] <= (w[0] + w[2]) / 2 <= box[2] and box[1] <= (w[1] + w[3]) / 2 <= box[3]]
        if len(inside) < 3:
            continue
        lines: Dict[Tuple[int, int], list] = {}
        for w in inside:
            lines.setdefault((w[5], w[6]), []).append(w)
        ordered = sorted(lines.values(), key=lambda l: (round(l[0][1] / 4), l[0][0]))
        text = "\n".join(" ".join(x[4] for x in sorted(l, key=lambda x: x[0])) for l in ordered)
        n_nat = sum(c.isalnum() for c in text)
        n_ocr = sum(c.isalnum() for c in b.text)
        if n_nat >= ps.min_native_cover * max(n_ocr, 1):
            b.text, b.source, b.rec_conf, b.note = text, "native", None, "trusted native text layer"
            n_used += 1
    return n_used


# ====================================================================== one attempt = Paddle (+ fallback)
@dataclass
class Outcome:
    dpi: int
    variant: Optional[str]
    source: str                         # paddle | paddle+tesseract | paddle+vl | vl | tesseract-rebuild
    structure: str                      # ok | best-effort
    blocks: List[StructBlock]
    gate: GateResult                    # gate on the raw PP-StructureV3 page
    weak: bool                          # still weak AFTER fallback
    weak_blocks: List[int]
    rec_conf: Optional[float]
    flags: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def markdown(self, ps: PaddleSettings) -> str:
        return assemble_markdown(self.blocks, ps.markdown_ignore_labels)

    def alnum(self) -> int:
        return sum(c.isalnum() for b in self.blocks for c in block_plain_text(b))


def _tess_weak(text: str, conf: float, cfg: Config, ps: PaddleSettings) -> bool:
    """Judge Tesseract text on Tesseract's own 0-100 scale (cfg.accept_score) plus the shared, engine-independent text checks.
    Paddle's recognition threshold (0-1) is a different quantity and is never reused here."""
    q = text_quality(text)
    return q["alnum"] == 0 or conf < cfg.accept_score or (q["tokens"] >= 3 and q["err"] > ps.max_err_rate)


def _mean_conf(lines) -> float:
    n = sum(l.n_words for l in lines)
    return sum(l.conf * l.n_words for l in lines) / n if n else 0.0


def _append_tesseract_orphans(blocks: List[StructBlock], lines: list, geo: "Geo") -> int:
    """Preserve substantial positioned Tesseract text that falls outside every PP block as reviewable extra blocks."""
    added = 0
    next_idx = max((b.idx for b in blocks), default=-1) + 1
    for line in lines:
        if sum(ch.isalnum() for ch in line.text) < 3:
            continue
        block = StructBlock(next_idx, "text", line.text, line.bbox, None, "tesseract", 1,
                            note="orphan line retained as last-resort recovered content; verify placement",
                            bbox_pdf=geo.box(*line.bbox),
                            line_boxes=[{"text": line.text, "bbox_px": list(line.bbox),
                                         "bbox_pdf": list(geo.box(*line.bbox)), "tesseract_conf": line.conf}])
        blocks.append(block)
        next_idx += 1
        added += 1
    return added


def _load_vl(ps: PaddleSettings):
    """Load PaddleOCR-VL only when PP quality checks fail; the caller may then continue to Tesseract if loading fails."""
    try:
        return get_vl_engine(ps)
    except EngineUnavailableError:
        raise
    except Exception as e:
        raise EngineUnavailableError(
            f"PaddleOCR-VL could not be loaded ({type(e).__name__}: {str(e)[:200]}). It needs a compatible GPU build and "
            "a paddleocr release that ships PaddleOCRVL; Tesseract will be tried as the last resort") from e


def _set_line_pdf(blocks: List[StructBlock], geo: "Geo") -> None:
    """Add PDF-space coordinates to OCR line boxes while preserving their engine pixel boxes."""
    for block in blocks:
        for line in block.line_boxes:
            box = line.get("bbox_px")
            if box and len(box) == 4:
                line["bbox_pdf"] = list(geo.box(*box))


def _vl_whole_page(ra: "Raster", info: dict, cfg: Config, ps: PaddleSettings, sp: StructPage,
                   detail: Dict[str, Any]) -> Tuple[Optional[List[StructBlock]], List[str]]:
    """Re-read the whole page with PaddleOCR-VL and verify it. Tesseract runs only after rejection or unresolved content."""
    vlp = _load_vl(ps).parse_page(ra.img)
    for b in vlp.blocks:
        b.bbox_pdf = ra.geo.box(*b.bbox)
    _set_line_pdf(vlp.blocks, ra.geo)
    n_native = apply_native_text(vlp.blocks, info["native"], ra.geo, ps) if info["type"] == "hybrid" else 0
    ok, reasons, vgate = verify_vl_page(vlp, sp, ra.ink, ps, cfg.min_ink_frac, ra.img)
    vd = detail["vl"] = {"mode": "page", "seconds": vlp.seconds, "n_blocks": len(vlp.blocks), "native_blocks": n_native,
                         "metrics": vgate.metrics, "error": vlp.error or None}
    vd["accepted"], vd["reasons"] = ok, reasons
    return (vlp.blocks if ok else None), reasons


_COVERED_MIN = 0.85          # share of an orphan line's characters already present in the blocks beside it


def _row_blocks(box, blocks: List[StructBlock]) -> List[StructBlock]:
    """Blocks that share the vertical band of `box` (>=50% of the box height), left to right."""
    h = max(box[3] - box[1], 1.0)
    out = [b for b in blocks if b.text.strip() and min(box[3], b.bbox[3]) - max(box[1], b.bbox[1]) >= 0.5 * h]
    return sorted(out, key=lambda b: b.bbox[0])


def _drop_covered_orphans(orphans: list, blocks: List[StructBlock]) -> Tuple[list, list]:
    """A Tesseract line that spans several Paddle blocks (header "title ......... doc-id", footer "text ... Page 2 of 35")
    overlaps none of them by 50% and so looks like missing content, although every character is already in those
    blocks. Such lines are duplicates, not recoveries: drop them. Returns (real_orphans, covered_orphans)."""
    real, covered = [], []
    for l in orphans:
        row = " ".join(b.text for b in _row_blocks(l.bbox, blocks))
        (covered if row and text_contained(l.text, row) >= _COVERED_MIN else real).append(l)
    return real, covered


def _line_confirms_block(b: StructBlock, lines: list) -> bool:
    """A Tesseract line that spans the block (block lies inside the line) and contains all of the block's characters
    independently confirms the block's text, even though the line was not 'assigned' to it."""
    for l in lines:
        if _inter_frac(b.bbox, l.bbox) >= 0.6 and text_contained(b.text, l.text) >= 0.9:
            return True
    return False


def _repair_with_tesseract(sp: StructPage, gate: GateResult, assigned: dict, cfg: Config, ps: PaddleSettings,
                           detail: Dict[str, Any], all_lines: Optional[list] = None) -> Tuple[List[str], List[int], int]:
    """Layout passed: keep Paddle structure, swap text only where Tesseract is clearly better."""
    flags: List[str] = []
    still_weak: List[int] = []
    replaced = 0
    for b in sp.blocks:
        if b.idx not in gate.failing:
            continue
        if b.label == "table":                       # cell-level replacement is a later step: keep + flag
            b.note += " | table kept from Paddle (Tesseract cannot rebuild cell structure)"
            flags.append(f"block {b.idx}: weak table kept, uncertain")
            still_weak.append(b.idx)
            continue
        ls = assigned.get(b.idx, [])
        if not ls and all_lines and _line_confirms_block(b, all_lines):
            b.note += " | independently confirmed by Tesseract (line spans block)"
            detail.setdefault("tesseract_confirmed_blocks", []).append(b.idx)
            continue
        if not ls:
            flags.append(f"block {b.idx}: no Tesseract line overlaps; kept Paddle text, uncertain")
            still_weak.append(b.idx)
            continue
        t_text = lines_to_text(ls)
        better, why, metrics = is_clear_improvement(b.text, t_text, ps)
        detail.setdefault("block_compare", {})[b.idx] = {"decision": why, **metrics}
        if not better and why == "candidates agree":
            if not _tess_weak(t_text, _mean_conf(ls), cfg, ps):
                b.note += " | independently confirmed by Tesseract"
                detail.setdefault("tesseract_confirmed_blocks", []).append(b.idx)
                continue
            flags.append(f"block {b.idx}: engines agree, but Tesseract text is also weak")
            still_weak.append(b.idx)
            continue
        if better:
            b.text, b.source, b.rec_conf = t_text, "tesseract", None
            b.line_boxes = [{"text": line.text, "bbox_px": list(line.bbox), "tesseract_conf": line.conf}
                            for line in ls]
            replaced += 1
            b.note = f"text replaced: {why}; was: {b.note}"
            if _tess_weak(t_text, _mean_conf(ls), cfg, ps):
                still_weak.append(b.idx)
                flags.append(f"block {b.idx}: replaced but Tesseract text is also weak")
        else:
            b.note += f" | kept existing text: {why}"
            flags.append(f"block {b.idx}: {why}; kept existing text, uncertain")
            still_weak.append(b.idx)
    return flags, still_weak, replaced


def _repair_with_vl(ra: "Raster", sp: StructPage, gate: GateResult, assigned: dict, ps: PaddleSettings,
                    detail: Dict[str, Any]) -> Tuple[List[str], List[int], int]:
    """Layout passed: re-read ONLY the failing blocks (crops) with PaddleOCR-VL; each replacement is verified against the
    independent Tesseract text for that block (and healthy Paddle text) inside vl_repair_blocks. The rest of the page is untouched."""
    refs = {i: lines_to_text(ls) for i, ls in assigned.items()}
    rep = vl_repair_blocks(_load_vl(ps), ra.img, sp.blocks, gate.failing, ps, refs)
    flags = [f"block {i}: {why}; kept Paddle text, uncertain" for i, why in sorted(rep.kept.items())]
    flags += [f"block {i}: PaddleOCR-VL error ({why}); kept Paddle text, uncertain" for i, why in sorted(rep.errors.items())]
    flags += [f"block {i}: replaced by PaddleOCR-VL but there was no trusted reference to verify it against, uncertain"
              for i in rep.unverified]
    detail["vl"] = {"mode": "blocks", "seconds": rep.seconds, "replaced": rep.replaced, "kept": rep.kept,
                    "errors": rep.errors, "unverified": rep.unverified, "metrics": rep.metrics}
    # A VL replacement without a trusted reference is not yet convincing: allow the final Tesseract
    # tier to compare that region too, then keep it flagged if Tesseract cannot establish an improvement.
    still_weak = sorted(set(rep.still_weak) | set(rep.unverified))
    return flags, still_weak, len(rep.replaced)


def run_attempt(page: pymupdf.Page, info: dict, cfg: Config, ps: PaddleSettings, runner: Runner, dpi: int,
                variant: Optional[str], rot, skew, plog: list) -> Tuple[Outcome, int, float]:
    ra, rot, skew = make_raster(page, info, cfg, dpi, variant, rot, skew, plog)
    engine = get_structure_engine(ps)
    sp: StructPage = engine.parse(ra.img)
    for b in sp.blocks:
        b.bbox_pdf = ra.geo.box(*b.bbox)
    _set_line_pdf(sp.blocks, ra.geo)
    n_native = apply_native_text(sp.blocks, info["native"], ra.geo, ps) if info["type"] == "hybrid" else 0
    gate = quality_gate(sp, ra.ink, ps, cfg.min_ink_frac, ra.img)          # the raster enables the uncovered-ink check
    detail: Dict[str, Any] = {"enabled_modules": ps.enabled_modules(), "text_det_limit_side_len": ps.limit_side_len,
                              "n_blocks": len(sp.blocks),
                              "native_blocks": n_native, "gate_reasons": gate.reasons, "gate_warnings": gate.warnings,
                              "gate_metrics": gate.metrics, "paddle_error": sp.error or None,
                              "paddle_markdown_chars": len(sp.raw_markdown), "paddle_seconds": sp.seconds}
    mk = lambda **kw: Outcome(dpi, variant, gate=gate, rec_conf=gate.rec_conf, detail=detail, **kw)

    if gate.ok:
        return mk(source="paddle", structure="ok", blocks=sp.blocks, weak=False, weak_blocks=[]), rot, skew
    if not ps.fallback_enabled:
        sub_w, _, peri_only = _split_weak(gate.failing, sp.blocks)
        return mk(source="paddle", structure="ok", blocks=sp.blocks, weak=bool(sub_w), weak_blocks=gate.failing), rot, skew

    use_vl = cfg.fallback_engine == "vl"
    detail["fallback_engine"] = cfg.fallback_engine

    # Tesseract is not run until VL has been rejected or has left weak blocks unresolved.
    lines: list = []
    if not use_vl:
        t0 = time.time()
        lines = tesseract_lines(runner, ra.img, cfg)
        detail["tesseract_lines"] = len(lines)
        detail["tesseract_seconds"] = round(time.time() - t0, 1)

    # "layout failed" caused ONLY by ink outside every block means nothing is structurally wrong with what Paddle did find:
    # keep its structure instead of discarding it (re-gate without the raster to tell the two apart).
    ink_only = False
    if not gate.layout_ok and gate.metrics.get("uncovered_ink", 0.0) > ps.max_uncovered_ink:
        ink_only = quality_gate(sp, ra.ink, ps, cfg.min_ink_frac).layout_ok
    structural = (not gate.layout_ok) and not ink_only
    flags: List[str] = []

    if structural:
        if use_vl:
            release_structure_engine(ps)
            try:
                vl_blocks, vl_reasons = _vl_whole_page(ra, info, cfg, ps, sp, detail)
            except EngineUnavailableError as e:
                vl_blocks, vl_reasons = None, [str(e)]
                detail["vl"] = {"accepted": False, "error": str(e)}
            finally:
                release_vl_engine(ps)
            if vl_blocks is not None:
                flags.append("layout failed: page re-read by PaddleOCR-VL and verified")
                return mk(source="vl", structure="ok", blocks=vl_blocks, weak=False, weak_blocks=[], flags=flags), rot, skew
            flags.append("PaddleOCR-VL result rejected: " + "; ".join(vl_reasons))
            t0 = time.time()
            lines = tesseract_lines(runner, ra.img, cfg)
            detail["tesseract_lines"] = len(lines)
            detail["tesseract_seconds"] = round(time.time() - t0, 1)
        blocks = tesseract_rebuild(lines)
        for b in blocks:
            b.bbox_pdf = ra.geo.box(*b.bbox)
        _set_line_pdf(blocks, ra.geo)
        conf = _mean_conf(lines)
        weak = (not lines) or _tess_weak(" ".join(b.text for b in blocks), conf, cfg, ps)
        detail["tesseract_mean_conf"] = round(conf, 1)
        flags.append("layout failed: page rebuilt from Tesseract positioned lines; structure is best-effort")
        if weak:
            flags.append("Tesseract rebuild is also weak")
        return mk(source="tesseract-rebuild", structure="best-effort", blocks=blocks, weak=weak,
                  weak_blocks=[b.idx for b in blocks] if weak else [], flags=flags), rot, skew

    # ---------------- layout usable (possibly incomplete): keep Paddle structure
    vl_whole_attempted = False
    if use_vl and ink_only:                    # content is missing, so re-read the whole page rather than only failing blocks
        vl_whole_attempted = True
        release_structure_engine(ps)
        try:
            vl_blocks, vl_reasons = _vl_whole_page(ra, info, cfg, ps, sp, detail)
        except EngineUnavailableError as e:
            vl_blocks, vl_reasons = None, [str(e)]
            detail["vl"] = {"accepted": False, "error": str(e)}
        finally:
            release_vl_engine(ps)
        if vl_blocks is not None:
            flags.append(f"{gate.metrics['uncovered_ink']:.0%} of the page ink lay outside every Paddle block: "
                         "page re-read by PaddleOCR-VL and verified")
            return mk(source="vl", structure="ok", blocks=vl_blocks, weak=False, weak_blocks=[], flags=flags), rot, skew
        flags.append("PaddleOCR-VL result rejected: " + "; ".join(vl_reasons))

    if use_vl:
        if vl_whole_attempted:
            f, still_weak, replaced = [], list(gate.failing), 0
        else:
            release_structure_engine(ps)
            try:
                f, still_weak, replaced = _repair_with_vl(ra, sp, gate, {}, ps, detail)
            except EngineUnavailableError as e:
                f, still_weak, replaced = [f"PaddleOCR-VL unavailable: {e}"], list(gate.failing), 0
                detail["vl"] = {"accepted": False, "error": str(e)}
            finally:
                release_vl_engine(ps)
        flags += f
        vl_replaced = replaced
        if still_weak or ink_only:
            # Only now run Tesseract, as the last-resort recognizer for unresolved regions.
            t0 = time.time()
            lines = tesseract_lines(runner, ra.img, cfg)
            detail["tesseract_lines"] = len(lines)
            detail["tesseract_seconds"] = round(time.time() - t0, 1)
            assigned, orphans = match_lines_to_blocks(lines, sp.blocks, ps.match_min_frac)
            import dataclasses
            gate_tess = dataclasses.replace(gate, failing=still_weak)
            f2, still_weak, replaced2 = _repair_with_tesseract(sp, gate_tess, assigned, cfg, ps, detail, all_lines=lines)
            flags += f2
            replaced += replaced2
            confirmed = set(detail.get("tesseract_confirmed_blocks", []))
            if confirmed:
                flags = [flag for flag in flags if not any(
                    flag.startswith(f"block {idx}: replaced by PaddleOCR-VL but there was no trusted reference")
                    for idx in confirmed)]
        else:
            assigned, orphans = {}, []
            replaced = vl_replaced
    else:
        assigned, orphans = match_lines_to_blocks(lines, sp.blocks, ps.match_min_frac)
        f, still_weak, replaced = _repair_with_tesseract(sp, gate, assigned, cfg, ps, detail, all_lines=lines)
        flags += f
    orphan_review = False
    if orphans:
        orphans, covered = _drop_covered_orphans(orphans, sp.blocks)
        if covered:
            detail["orphans_already_covered"] = [{"text": l.text[:80], "bbox": l.bbox} for l in covered[:20]]
    if orphans:
        orphan_chars = sum(sum(c.isalnum() for c in l.text) for l in orphans)
        detail["orphan_lines"] = [{"text": l.text[:80], "bbox": l.bbox} for l in orphans[:20]]
        if orphan_chars >= 15:
            orphan_review = True
            n_added = _append_tesseract_orphans(sp.blocks, orphans, ra.geo)
            detail["tesseract_orphan_blocks_added"] = n_added
            flags.append(f"{n_added} Tesseract line(s), {orphan_chars} characters, recovered outside Paddle blocks; "
                         "verify their placement")
    if ink_only:
        flags.append(f"{gate.metrics['uncovered_ink']:.0%} of the page ink lies outside every detected block: "
                     "possible missing content (not recovered)")
    detail["blocks_replaced"] = replaced
    _set_line_pdf(sp.blocks, ra.geo)
    selected_sources = {b.source for b in sp.blocks}
    selected_sources.discard("native")
    if "tesseract" in selected_sources and "vl" in selected_sources:
        source = "paddle+vl+tesseract"
    elif "tesseract" in selected_sources:
        source = "paddle+tesseract"
    elif "vl" in selected_sources:
        source = "paddle+vl"
    else:
        source = "paddle"
    # A page is only "weak" (needs_review) when substantive blocks remain unfixed.
    # Peripheral blocks (headers, footers, page numbers) that couldn't be repaired are noted in warnings
    # but do NOT doom the whole page — the body text quality is what matters.
    sub_weak, peri_weak, peri_only = _split_weak(still_weak, sp.blocks)
    page_weak = bool(sub_weak) or ink_only or orphan_review
    if peri_only:
        flags.append(f"{len(peri_weak)} peripheral block(s) (header/footer) remain uncertain but body text is OK")
    return mk(source=source, structure="ok", blocks=sp.blocks, weak=page_weak,
              weak_blocks=still_weak, flags=flags), rot, skew


def _rank(o: Outcome) -> tuple:
    """Selection across attempts (higher is better): not weak > fewer weak blocks > more content > fewer error
    proxies. Raw engine confidences are NOT part of the ranking."""
    text = " ".join(block_plain_text(b) for b in o.blocks)
    return (not o.weak, -len(o.weak_blocks), o.alnum(), -text_quality(text)["err"])


def _summ(o: Outcome, ps: PaddleSettings, selected: bool, reason: str) -> Dict[str, Any]:
    return {"dpi": o.dpi, "variant": o.variant, "source": o.source, "structure": o.structure, "weak": o.weak,
            "weak_blocks": o.weak_blocks, "paddle_rec_conf": None if o.rec_conf is None else round(o.rec_conf, 3),
            "gate_reasons": o.gate.reasons, "flags": o.flags, "detail": o.detail, "selected": selected,
            "selection_reason": reason, "markdown": o.markdown(ps)}


# ====================================================================== per page
_RUNNER: Optional[Runner] = None
_DOC: Dict[str, pymupdf.Document] = {}
_DOC_SIG: Dict[str, Tuple[int, int]] = {}


def _file_sig(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def _native_words(pm: PageMarkdown, info: dict, cfg: Config) -> None:
    """Digital / native-only page: exact word boxes straight from the trusted text layer."""
    if not cfg.word_boxes:
        return
    pm.words_text, pm.words = wordbox.native_page_words(info["native"])
    pm.stats["word_boxes"] = {"source": "native", "n_words": len(pm.words)}


def _ocr_words(pm: PageMarkdown, page: pymupdf.Page, info: dict, cfg: Config, best: Outcome, rot, skew) -> None:
    """OCR page: locate words inside every block (Tesseract) and align the block's own text to them. Never raises."""
    try:
        ra, _, _ = make_raster(page, info, cfg, best.dpi, None, rot, skew)     # same dpi / rotation / deskew as the engines
        pm.words_text, pm.words, st, warns = wordbox.attach_words(
            best.blocks, ra, _RUNNER, cfg, native=info["native"], page_type=info["type"])
        pm.stats["word_boxes"] = {"source": "locator+alignment", "n_words": len(pm.words), **st}
        pm.warnings += warns
        if cfg.debug_dir and pm.words:
            _debug_words(cfg.debug_dir, pm.page_no, ra.img, pm.words)
    except Exception as e:
        log.warning("word boxes failed for page %d", pm.page_no, exc_info=True)
        pm.warnings.append(f"word boxes failed: {type(e).__name__}: {str(e)[:120]}; line boxes only")


def _debug_words(out_dir: str, page_no: int, raster: np.ndarray, words: List[Dict[str, Any]]) -> None:
    """Word boxes on the engine raster: green = aligned, orange = split inside a locator word, red = fallback (whole line /
    block), blue = locator-only. Native words have no pixel box and are skipped."""
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.cvtColor(raster, cv2.COLOR_GRAY2BGR) if raster.ndim == 2 else raster.copy()
    col = {"ok": (60, 160, 60), "split": (0, 140, 255), "locator-only": (200, 120, 0)}
    for w in words:
        if not w.get("bbox_px"):
            continue
        x0, y0, x1, y1 = (int(v) for v in w["bbox_px"])
        cv2.rectangle(img, (x0, y0), (x1, y1), col.get(w["status"], (0, 0, 255)), 1)
    cv2.imwrite(os.path.join(out_dir, f"page_{page_no:03d}_words.png"), img)


def process_page(args) -> PageMarkdown:
    global _RUNNER
    pdf_path, pno, cfg, ps = args
    t0 = time.time()
    if _RUNNER is None or _RUNNER.cfg != cfg:                 # a changed config (lang, tesseract path...) needs a fresh Runner
        _RUNNER = Runner(cfg)
    sig = _file_sig(pdf_path)
    if pdf_path not in _DOC or _DOC_SIG.get(pdf_path) != sig:    # a replaced / edited file must not be served from the cache
        stale = _DOC.pop(pdf_path, None)
        if stale is not None:
            try:
                stale.close()
            except Exception:
                pass
        _DOC[pdf_path], _DOC_SIG[pdf_path] = pymupdf.open(pdf_path), sig
    page = _DOC[pdf_path][pno]
    info = classify(page, cfg)
    pm = PageMarkdown(pno + 1, info["type"])
    pm.stats.update({k: info[k] for k in ("img_cov", "drawings", "garbage", "untrusted", "n_invisible")})
    pm.stats.update({"n_ocr_words": 0, "n_low_conf": 0})
    if info["untrusted"]:
        pm.warnings.append("native text layer looked corrupt (cid/U+FFFD); ignored and OCR'd")

    def done(pm: PageMarkdown) -> PageMarkdown:
        pm.stats["seconds"] = round(time.time() - t0, 2)
        return pm

    # ---- digital: trust the text layer, no OCR
    if info["type"] == "digital":
        pm.markdown, pm.source, pm.structure = native_markdown(page, info), "native", "native"
        _native_words(pm, info, cfg)
        return done(pm)

    # ---- does anything remain that the text layer does not cover? (avoid OCRing trusted text needlessly)
    resid = render_gray(page, cfg.base_dpi)
    for w in info["native"]:
        r = (pymupdf.Rect(w[:4]) * pymupdf.Matrix(page.rotation_matrix) * (cfg.base_dpi / 72.0)).normalize()
        resid[max(int(r.y0) - 1, 0):int(r.y1) + 2, max(int(r.x0) - 1, 0):int(r.x1) + 2] = 255
    if pp.ink_fraction(resid) < cfg.min_ink_frac:
        if info["native"]:
            pm.markdown, pm.source, pm.structure = native_markdown(page, info), "native", "native"
            pm.attempts.append({"stage": "native-only", "reason": "no residual ink beyond trusted text layer"})
            _native_words(pm, info, cfg)
        else:
            pm.page_type, pm.source = "blank", "blank"
            pm.attempts.append({"stage": "blank", "reason": "no ink"})
        return done(pm)
    del resid

    # ---- primary + fallback + retries (retry only while every candidate so far is weak)
    plog: list = []
    outcomes: List[Outcome] = []
    first, rot, skew = run_attempt(page, info, cfg, ps, _RUNNER, cfg.base_dpi, None, None, None, plog)
    outcomes.append(first)
    for dpi in ps.retry_dpis:
        if not all(o.weak for o in outcomes):
            break
        o, rot, skew = run_attempt(page, info, cfg, ps, _RUNNER, dpi, ps.retry_variant, rot, skew, plog)
        outcomes.append(o)

    best = max(outcomes, key=_rank)           # max() keeps the earliest of equals -> ties stay with Paddle @ base DPI
    if len(outcomes) == 1:
        reason = "passed quality gate" if not best.weak else "weak; no retry configured, PP-StructureV3 path retained"
    elif best is outcomes[0]:
        reason = "no retry was clearly better; first PP-StructureV3 result retained"
    else:
        reason = f"retry at {best.dpi} DPI/{best.variant} ranked higher (not weak > fewer weak blocks > completeness > error proxy)"
    pm.attempts = [{"stage": "preprocess", "log": plog}] + [_summ(o, ps, o is best, reason if o is best else "") for o in outcomes]

    pm.source, pm.structure = best.source, best.structure
    pm.markdown = best.markdown(ps)
    # A missing Paddle recognition score is unknown, not 100%.
    if best.source.startswith("paddle"):
        pm.score = round(best.rec_conf * 100, 1) if best.rec_conf is not None else None
    else:
        pm.score = None
    pm.tables = [block_markdown(b) for b in best.blocks if b.label == "table"]
    pm.blocks = [{"idx": b.idx, "label": b.label, "source": b.source, "bbox_px": [round(v, 1) for v in b.bbox],
                 "bbox_pdf": [round(v, 1) for v in b.bbox_pdf] if b.bbox_pdf else None,
                 "rec_conf": None if b.rec_conf is None else round(b.rec_conf, 3), "line_boxes": b.line_boxes,
                 "note": b.note.strip(" |"), "text": b.text,
                 "plain_text": wordbox.clean_text(block_plain_text(b))} for b in best.blocks]
    pm.stats.update({"n_ocr_words": sum(len(b.text.split()) for b in best.blocks if b.source != "native"),
                     "n_low_conf": len(best.weak_blocks), "attempts": len(outcomes), "selected_dpi": best.dpi,
                     "enabled_modules": ps.enabled_modules(), "fallback_engine": cfg.fallback_engine if ps.fallback_enabled else None,
                     "n_unverified": len(set((best.detail.get("vl") or {}).get("unverified", [])) -
                                         set(best.detail.get("tesseract_confirmed_blocks", [])))})
    pm.warnings += best.flags
    if cfg.word_boxes:
        _ocr_words(pm, page, info, cfg, best, rot, skew)
    if best.structure == "best-effort":
        pm.warnings.append("structure best-effort: tables / columns / headings not verified")
    if best.weak:
        pm.stats["needs_review"] = True
        if not ps.fallback_enabled:
            pm.warnings.append("needs_review: weak result and no fallback engine is enabled (--no-fallback)")
        elif cfg.fallback_engine == "vl":
            pm.warnings.append("needs_review: unresolved or unverified content remains after the PP-StructureV3, "
                               "PaddleOCR-VL and Tesseract tiers; review this page against the page image")
        else:
            pm.warnings.append("needs_vlm: PP-StructureV3 and Tesseract both weak after retry; escalate this page/region to "
                               "PaddleOCR-VL (--fallback-engine vl) or review it against the page image")
            pm.stats["needs_vlm"] = True
    if cfg.debug_dir:
        try:                                          # the overlay uses the exact raster the engines saw (rotation / deskew applied)
            dbg, _, _ = make_raster(page, info, cfg, best.dpi, best.variant, rot, skew)
            _debug_overlay(cfg.debug_dir, pno + 1, dbg.img, best)
        except Exception:
            log.warning("debug overlay for page %d failed", pno + 1, exc_info=True)
    return done(pm)


def _debug_overlay(out_dir: str, page_no: int, raster: np.ndarray, o: Outcome) -> None:
    """Block boxes on the engine raster (green = Paddle, orange = Tesseract, purple = VL, blue = native, red = other).
    `raster` must be the image the engines saw: block boxes are in ITS pixel space (after rotation / deskew)."""
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.cvtColor(raster, cv2.COLOR_GRAY2BGR) if raster.ndim == 2 else raster.copy()
    col = {"paddle": (60, 160, 60), "tesseract": (0, 140, 255), "vl": (180, 60, 180), "native": (200, 120, 0)}
    for b in o.blocks:
        x0, y0, x1, y1 = (int(v) for v in b.bbox)
        cv2.rectangle(img, (x0, y0), (x1, y1), col.get(b.source, (0, 0, 255)), 2)
        cv2.putText(img, f"{b.idx}:{b.label}", (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dir, f"page_{page_no:03d}.png"),
                cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


# ====================================================================== document
def parse_pages(spec: Optional[str], n: int) -> List[int]:
    """Parse a 1-based page spec like '1-5,9' into a sorted list of 0-based indices within [0, n).
    Reversed ranges (e.g. '5-3') are silently skipped with a warning instead of yielding nothing."""
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b.strip() else int(a)
        if hi < lo:
            log.warning("parse_pages: reversed range %r skipped (lo=%d > hi=%d); check your --pages argument", part, lo, hi)
            continue
        out += range(lo - 1, hi)
    return [p for p in out if 0 <= p < n]


def process_pdf(path: str, cfg: Optional[Config] = None, pages: Optional[str] = None,
                paddle: Optional[PaddleSettings] = None) -> List[PageMarkdown]:
    cfg = cfg or Config()
    ps = paddle or cfg.paddle_settings()      # all PP-StructureV3 options + gate thresholds come from Config
    with pymupdf.open(path) as d:
        idx = parse_pages(pages, len(d))
    # PP-StructureV3 loads several models and is multi-threaded itself: default to ONE worker process.
    # (cfg.workers > 1 loads one model set per process; only do that with enough RAM / GPU memory.)
    workers = cfg.workers if cfg.workers > 0 else 1
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")       # Tesseract fallback: one thread per process
    jobs = [(path, i, cfg, ps) for i in idx]
    if workers <= 1 or len(jobs) == 1:
        return [process_page(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(process_page, jobs))


def write_outputs(results: List[PageMarkdown], out_dir: str, stem: str,
                  json_stem: Optional[str] = None) -> Dict[str, str]:
    """Markdown is the sole deliverable. Per-page blocks/attempts go to <out>/_internal/ for QA only."""
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        for r in results:
            flags = f"source={r.source} structure={r.structure}"
            if r.stats.get("n_low_conf"):
                flags += f" uncertain_blocks={r.stats['n_low_conf']}"
            if r.stats.get("n_unverified"):
                flags += f" unverified_blocks={r.stats['n_unverified']}"
            if r.stats.get("needs_review") or r.stats.get("needs_vlm"):
                flags += " needs_review"
            f.write(f"<!-- page {r.page_no} | {flags} -->\n{r.markdown}\n\n")
    internal_dir = os.path.join(out_dir, "_internal")
    os.makedirs(internal_dir, exist_ok=True)
    int_path = os.path.join(internal_dir, f"{json_stem or stem}.pages.json")
    with open(int_path, "w", encoding="utf-8") as f:
        json.dump({"total_seconds": round(sum(r.stats.get("seconds", 0) for r in results), 1),
                   "pages": [asdict(r) for r in results]}, f, ensure_ascii=False, indent=1, default=str)
    return {"markdown": md_path, "internal_json": int_path}
