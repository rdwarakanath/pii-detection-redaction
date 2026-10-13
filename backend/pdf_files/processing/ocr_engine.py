"""OCR engines, quality scoring and the confidence-driven escalation ladder.

Primary engine : PaddleOCR PP-StructureV3 (layout + OCR + reading order + tables -> per-block results)
Fallback order: PaddleOCR-VL (whole page or per-block crops), then positioned Tesseract as last resort. Raw
                 confidences of different engines are never compared; candidates are judged by the project checks.

Revision notes:
  * Gate: Paddle-specific checks (run-together words, stray leading glyph, uncovered ink), recognition confidence
    weighted by OCR characters (not HTML length), tiny blocks (page numbers) no longer trigger fallback.
  * text_quality: no longer penalises dot-leaders / list enumerators / non-Latin scripts; flags run-together words.
  * Engines: one cached PP-StructureV3 instance per *engine* options (gate thresholds no longer reload models),
    unsupported kwargs are logged, repeated engine failures raise EngineUnavailableError (fail loudly), known
    oneDNN/PIR and GPU-OOM errors carry an actionable hint.
  * PaddleOCR-VL: lazy engine, page parse, per-block crop recognition, verification helpers (VL has no per-token
    confidence, so its output is verified instead of thresholded) and `vl_repair_blocks` / `verify_vl_page`.
  * Table parsing tolerates unclosed <td>/<tr> and Markdown tables.
  * pytesseract is imported lazily inside the Tesseract class; the module loads without it, but fallback-enabled runs
    require the Tesseract package, executable and language data.

Flow per page (PP-StructureV3 -> quality gate -> fallback):
  1. PP-StructureV3 produces per-block results with recognition confidence scores.
  2. quality_gate checks: rec_conf >= rec_conf_threshold (default 0.85), layout completeness, junk/spacing/table checks.
  3. Gate PASS -> keep the page unchanged.
  4. Gate FAIL -> lazily load PaddleOCR-VL, re-read the whole page or failing blocks, and validate its content.
  5. If VL cannot load or its result is rejected / still weak, run Tesseract as the final OCR fallback.
  6. Still weak -> needs_review flag. Higher-DPI retries are opt-in because they add time and memory pressure.
"""
from __future__ import annotations
import difflib
import gc
import html as _html
import inspect
import json
import logging
import re
import time
from html.parser import HTMLParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from . import preprocess as pp
from .config import Config

log = logging.getLogger(__name__)


@dataclass
class RawWord:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    block: int = 0
    line: int = 0     # unique line key within one recognise() call

    @property
    def box(self):
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Attempt:
    name: str
    variant: str = "norm"
    psm: int = 6
    scale: float = 1.0          # multiplier on the auto (cap-height driven) scale
    invert: bool = False
    extra: str = ""
    engine: str = "tesseract"
    remove_lines: bool = False


# ------------------------------------------------------------------ scoring
_VOWEL = re.compile(r"[aeiouyAEIOUY]")


def _junk(w: RawWord) -> bool:
    t = w.text
    alnum = sum(c.isalnum() for c in t)
    if alnum == 0:
        return len(t) > 1 or w.conf < 60
    if alnum / len(t) < 0.5:
        return True
    # The vowel rule only makes sense for ASCII Latin words: Cyrillic / Greek / Indic / CJK words never contain
    # [aeiouy] and would all be flagged as junk.
    if len(t) >= 5 and t.isascii() and t.isalpha() and not t.isupper() and not _VOWEL.search(t):
        return True
    return len(t) == 1 and w.conf < 50


def quality(words: List[RawWord]) -> float:
    """0-100. Length-weighted confidence minus a penalty for junk tokens. Empty => 0."""
    if not words:
        return 0.0
    chars = sum(len(w.text) for w in words) or 1
    conf = sum(w.conf * len(w.text) for w in words) / chars
    junk = sum(1 for w in words if _junk(w)) / len(words)
    return max(0.0, conf - 45.0 * junk)


# ------------------------------------------------------------------ engines (Tesseract + optional alternatives)
class Tesseract:
    """Tesseract OCR engine wrapper. pytesseract is imported lazily so the module loads even when Tesseract is absent."""
    name = "tesseract"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import pytesseract as _pt
            self._pt = _pt
            if cfg.tesseract_cmd:
                _pt.pytesseract.tesseract_cmd = cfg.tesseract_cmd
        except ImportError as e:
            raise RuntimeError(
                "pytesseract is not installed. Run: pip install pytesseract  (and install the Tesseract binary). "
                "Tesseract is required as the last-resort OCR tier whenever fallback is enabled."
            ) from e

    def run(self, img: np.ndarray, att: Attempt) -> List[RawWord]:
        pt = self._pt
        conf = (f"--oem 1 --psm {att.psm} --dpi 300 -c preserve_interword_spaces=1 "
                f"-c tessedit_do_invert=0 {att.extra}")
        d = pt.image_to_data(img, lang=self.cfg.lang, config=conf, output_type=pt.Output.DICT)
        out, keys = [], {}
        for i, t in enumerate(d["text"]):
            t = (t or "").strip()
            c = float(d["conf"][i])
            if not t or c < 0:
                continue
            k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lid = keys.setdefault(k, len(keys))
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append(RawWord(t, c, x, y, x + w, y + h, d["block_num"][i] * 100 + d["par_num"][i], lid))
        return out


class EasyOCR:
    name = "easyocr"

    def __init__(self, cfg: Config):
        import easyocr   # optional dependency
        self.r = easyocr.Reader([cfg.lang[:2]], gpu=False, verbose=False)

    def run(self, img: np.ndarray, att: Attempt) -> List[RawWord]:
        out = []
        for i, (poly, text, conf) in enumerate(self.r.readtext(img)):
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            out.append(RawWord(text.strip(), conf * 100, min(xs), min(ys), max(xs), max(ys), i, i))
        return [w for w in out if w.text]


class RapidOCR:
    name = "rapidocr"

    def __init__(self, cfg: Config):
        from rapidocr_onnxruntime import RapidOCR as R   # optional dependency
        self.r = R()

    def run(self, img: np.ndarray, att: Attempt) -> List[RawWord]:
        res, _ = self.r(img)
        out = []
        for i, (box, text, conf) in enumerate(res or []):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append(RawWord(text.strip(), float(conf) * 100, min(xs), min(ys), max(xs), max(ys), i, i))
        return [w for w in out if w.text]


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engines: Dict[str, Any] = {}   # engines are created lazily on first use (Tesseract needs pytesseract installed)
        self._alt_failed = False
        self._warned: set = set()          # log each distinct engine failure once (a missing binary fails every call)

    def _engine(self, name: str):
        if name not in self.engines:
            if name == "tesseract":
                self.engines[name] = Tesseract(self.cfg)
            elif name == "easyocr":
                self.engines[name] = EasyOCR(self.cfg)
            elif name == "rapidocr":
                self.engines[name] = RapidOCR(self.cfg)
            else:
                raise ValueError(f"Unsupported OCR engine: {name}")
        return self.engines[name]

    def recognize(self, g: np.ndarray, att: Attempt, base_scale: Optional[float] = None) -> List[RawWord]:
        """OCR one crop. Returned boxes are in the pixel space of `g` (resize + padding undone)."""
        if g.size == 0 or min(g.shape) < 6:
            return []
        img = 255 - g if att.invert else g
        if att.remove_lines:
            img = pp.remove_lines(img, kh=max(20, g.shape[1] // 20), kv=max(20, g.shape[0] // 8))
        s0 = base_scale if base_scale is not None else pp.scale_for(
            img, self.cfg.target_cap_px, self.cfg.scale_min, self.cfg.scale_max)
        f = float(np.clip(s0 * att.scale, 0.3, 5.0))
        img = pp.make_variant(pp.resize(img, f), att.variant)
        P = 12
        img = pp.pad(img, P)
        words = self._engine(att.engine).run(img, att)
        for w in words:
            w.x0, w.x1 = (w.x0 - P) / f, (w.x1 - P) / f
            w.y0, w.y1 = (w.y0 - P) / f, (w.y1 - P) / f
        return words

    # -- escalation -------------------------------------------------------
    def ladder(self, g: np.ndarray, attempts: List[Attempt], accept: float, base_scale: Optional[float] = None
               ) -> Tuple[List[RawWord], float, List[dict]]:
        cur: List[RawWord] = []
        cur_s, log_ = -1.0, []
        for att in attempts:
            try:
                new = self.recognize(g, att, base_scale)
            except Exception as e:                      # engine missing / crashed: log and move on
                msg = f"{type(e).__name__}: {str(e)[:120]}"
                log_.append({"attempt": att.name, "error": str(e)[:120]})
                key = (att.engine, msg)
                if key not in self._warned:
                    self._warned.add(key)
                    log.warning("OCR engine %s failed (attempt %s): %s", att.engine, att.name, msg)
                continue
            cand = merge_by_line(cur, new) if cur else new
            s = quality(cand)
            log_.append({"attempt": att.name, "score": round(quality(new), 1), "merged": round(s, 1), "n": len(new)})
            if s > cur_s:
                cur, cur_s = cand, s
            if cur_s >= accept:
                break
        return cur, max(cur_s, 0.0), log_


# ------------------------------------------------------------------ line voting
def _lines(words: List[RawWord]):
    d = {}
    for w in words:
        d.setdefault(w.line, []).append(w)
    out = []
    for ws in d.values():
        out.append((min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws), ws))
    return out


def _overlap(a, b) -> bool:
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return False
    return iy / min(a[3] - a[1], b[3] - b[1]) > 0.5 and ix / min(a[2] - a[0], b[2] - b[0]) > 0.3


def merge_by_line(a: List[RawWord], b: List[RawWord]) -> List[RawWord]:
    """Per-line best-of-N voting between two OCR runs of the same region."""
    la, lb = _lines(a), _lines(b)
    keep = [True] * len(la)
    add = []
    for bl in lb:
        hits = [i for i, al in enumerate(la) if keep[i] and _overlap(al, bl)]
        sb = quality(bl[4])
        if not hits:
            if sb >= 55:
                add.append(bl)
            continue
        sa = quality([w for i in hits for w in la[i][4]])
        if sb > sa + 3:
            for i in hits:
                keep[i] = False
            add.append(bl)
    out = [w for i, al in enumerate(la) if keep[i] for w in al[4]]
    nxt = max([w.line for w in out] + [-1]) + 1
    for bl in add:
        for w in bl[4]:
            out.append(RawWord(w.text, w.conf, w.x0, w.y0, w.x1, w.y1, w.block, nxt))
        nxt += 1
    return out


# ------------------------------------------------------------------ standard ladders (Tesseract; region/cell ladders are
# only used by the legacy tables.py and can be deleted together with it)
def page_text_ladder(cfg: Config) -> List[Attempt]:
    L = [Attempt("norm/psm4", "norm", 4),
         Attempt("clahe/psm6", "clahe", 6),
         Attempt("sauvola/psm3 x1.4", "sauvola", 3, scale=1.4),
         Attempt("denoise/psm6 x1.8", "denoise", 6, scale=1.8)]
    if cfg.alt_engine:
        L.append(Attempt("alt", "gray", 0, engine=cfg.alt_engine))
    return L


def region_ladder(cfg: Config) -> List[Attempt]:
    """Figures, screenshots, logos, badges: sparse text, more aggressive scaling, both polarities."""
    L = [Attempt("sparse/psm11", "norm", 11, scale=1.3),
         Attempt("sparse/psm12 x2", "clahe", 12, scale=2.0),
         Attempt("inv/psm11 x2", "norm", 11, scale=2.0, invert=True),
         Attempt("sauvola/psm6 x2.5", "sauvola", 6, scale=2.5),
         Attempt("clahe/psm11 x3", "clahe", 11, scale=3.0)]
    if cfg.alt_engine:
        L.append(Attempt("alt", "gray", 0, engine=cfg.alt_engine))
    return L


def cell_ladder(cfg: Config, single_line: bool) -> List[Attempt]:
    p = 7 if single_line else 6
    L = [Attempt("cell/norm", "norm", p, scale=1.5, remove_lines=True),
         Attempt("cell/clahe x2", "clahe", p, scale=2.2, remove_lines=True),
         Attempt("cell/sauvola", "sauvola", p, scale=2.0, remove_lines=True),
         Attempt("cell/word", "norm", 8 if single_line else 11, scale=2.5, remove_lines=True),
         Attempt("cell/inv", "norm", p, scale=2.0, invert=True, remove_lines=True)]
    if cfg.alt_engine:
        L.append(Attempt("alt", "gray", 0, engine=cfg.alt_engine))
    return L


# ====================================================================================================
#  PP-StructureV3 (primary) -- settings, result parsing, Markdown rendering
# ====================================================================================================
# PaddleOCR language codes that use the Latin script / space-separated words. The spacing and stray-glyph checks are only
# meaningful for these; for other scripts they are skipped (never guessed).
_LATIN_PADDLE_LANGS = frozenset({
    "en", "fr", "german", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "cs", "ro", "tr", "hu", "id", "ms",
    "vi", "latin", "af", "ca", "hr", "sk", "sl", "et", "lv", "lt", "is", "ga", "eu", "mt", "sq", "cy", "la"})


def _latin_lang(lang: Optional[str]) -> bool:
    return (lang or "").lower() in _LATIN_PADDLE_LANGS


@dataclass(frozen=True)
class PaddleSettings:
    """PP-StructureV3 / PaddleOCR-VL options + quality-gate thresholds. Frozen so settings are hashable.

    Defaults follow implementation_guidelines.md: start from PP-StructureV3 defaults; orientation / unwarping
    and the optional formula / chart / seal modules are OFF until a document set needs them. The enabled modules
    are written to the internal JSON because they change both quality and runtime.
    NOTE: use_doc_unwarping changes page geometry, so block boxes no longer map back to the PDF page and
    box-overlap fallback is unreliable. Keep it off unless you accept that.

    Engines are cached by `engine_key()` / `vl_key()` (model options only), NOT by the whole settings object, so changing
    a gate threshold never reloads a model set.
    """
    lang: str = "en"
    device: Optional[str] = None                  # None = PaddleOCR default; "cpu" | "gpu:0"
    enable_mkldnn: bool = False                   # CPU oneDNN acceleration; false uses the plain Paddle CPU path (avoids crash)
    use_doc_orientation_classify: bool = False    # pipeline already does Tesseract OSD + deskew
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    use_seal_recognition: bool = False
    use_table_recognition: bool = True
    use_formula_recognition: bool = False
    use_chart_recognition: bool = False
    use_region_detection: bool = True
    limit_side_len: int = 4000

    # ---- quality gate (text-RECOGNITION confidence only; never layout-detection scores) -----------
    rec_conf_threshold: float = 0.85              # below -> fallback for that page/region
    max_err_rate: float = 0.12                    # junk + mixed-alnum token share that fails a block
    max_overlap_iou: float = 0.6                  # two text blocks overlapping more than this -> layout failed
    max_order_jump_frac: float = 0.25             # share of reading-order transitions that go back up a column
    min_table_fill: float = 0.35                  # tables with fewer non-empty cells than this are incoherent
    min_chars_on_inked_page: int = 20             # inked page with fewer alnum chars -> incomplete

    # ---- fallback / retry --------------------------------------------------------------------------
    fallback_enabled: bool = True
    match_min_frac: float = 0.5                   # share of a Tesseract line that must lie inside a Paddle block
    retry_dpis: Tuple[int, ...] = ()               # opt-in: full retries add runtime and memory pressure
    retry_variant: str = "norm"                   # targeted preprocessing for retries (preprocess.make_variant)
    min_native_cover: float = 0.7                 # hybrid: native text replaces OCR text when it covers this much
    markdown_ignore_labels: Tuple[str, ...] = ()  # PP-StructureV3 drops header/footer/number by default; we keep them

    # ---- additional gate checks (new; defaults chosen so existing behaviour is only made stricter where Paddle fails)
    min_gate_chars: int = 3                       # blocks whose OCR text is shorter (page numbers) skip the rec-conf check
    max_uncovered_ink: float = 0.15               # ink outside every non-empty block above this => layout incomplete
    min_space_ratio: float = 0.09                 # (words-1)/alnum below this in Latin prose => words run together
    min_spacing_chars: int = 25                   # only blocks with at least this many alnum chars get the spacing check
    abort_after_errors: int = 3                   # consecutive engine exceptions before EngineUnavailableError; 0 = never

    # ---- PaddleOCR-VL (fallback engine) --------------------------------------------------------------
    vl_pipeline_version: Optional[str] = None     # None = library default ("v1"); newer releases accept e.g. "v1.5"
    vl_backend: Optional[str] = None              # None = native Paddle; "vllm-server" etc. for a serving backend
    vl_server_url: Optional[str] = None           # used with a *-server backend
    vl_device: Optional[str] = None               # None = same as `device`; VL needs a GPU (no CPU / Arm support)
    vl_use_chart_recognition: bool = False
    vl_use_ocr_for_image_block: bool = False      # OCR text inside image blocks (logos); more content, more noise
    vl_crop_margin_px: int = 12                   # context added around a block before region recognition
    vl_min_len_ratio: float = 0.5                 # VL alnum / Paddle alnum must lie in [min, max] ...
    vl_max_len_ratio: float = 2.0                 # ... otherwise VL is treated as truncated / hallucinated
    vl_min_agreement: float = 0.35                # VL vs trusted reference similarity below this => reject VL
    vl_loop_repeats: int = 7                      # same phrase repeated this often back-to-back => degenerate VL output

    def enabled_modules(self) -> Dict[str, bool]:
        return {k: getattr(self, k) for k in (
            "use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation", "use_seal_recognition",
            "use_table_recognition", "use_formula_recognition", "use_chart_recognition", "use_region_detection",
            )}

    def engine_key(self) -> tuple:
        """Everything that changes which PP-StructureV3 models are loaded / how they run (no gate thresholds)."""
        return (self.lang, self.device, self.enable_mkldnn, self.limit_side_len) + tuple(self.enabled_modules().values())

    def vl_key(self) -> tuple:
        return (self.vl_device or self.device, self.vl_pipeline_version, self.vl_backend, self.vl_server_url,
                self.vl_use_chart_recognition, self.vl_use_ocr_for_image_block)


@dataclass
class StructBlock:
    idx: int                         # position in PP-StructureV3 reading order
    label: str                       # doc_title | paragraph_title | text | table | formula | image | ...
    text: str                        # plain text; HTML (or Markdown) for tables
    bbox: Tuple[float, float, float, float]       # px of the image sent to the engine (page identity kept by caller)
    rec_conf: Optional[float] = None  # 0-1 mean text-RECOGNITION score of OCR lines inside the block; None = unknown
    source: str = "paddle"           # paddle | native | tesseract | vl
    n_lines: int = 0
    note: str = ""                   # uncertainty / replacement reason
    bbox_pdf: Optional[Tuple[float, float, float, float]] = None
    rec_chars: int = 0               # characters of OCR text behind rec_conf (weight for the page mean); 0 = unknown
    line_boxes: List[Dict[str, Any]] = field(default_factory=list)  # recognized text lines with px/PDF boxes and score


@dataclass
class StructPage:
    blocks: List[StructBlock]
    raw_markdown: str = ""           # engine's own Markdown (debug reference only)
    n_ocr_lines: int = 0
    error: str = ""
    engine: str = "paddle"           # paddle | vl
    seconds: float = 0.0


_TEXT_LABELS = {"doc_title", "paragraph_title", "text", "abstract", "content", "reference", "reference_content",
                "figure_title", "table_title", "chart_title", "footnote", "vision_footnote", "header", "footer",
                "number", "aside_text", "vertical_text", "algorithm", "formula_number", "seal"}
_NO_TEXT_LABELS = {"image", "header_image", "footer_image", "chart"}
# Labels holding running prose / titles where word spacing and a stray leading glyph can be judged.
_PROSE_LABELS = {"doc_title", "paragraph_title", "text", "abstract", "content", "reference", "reference_content",
                 "figure_title", "table_title", "chart_title", "footnote", "vision_footnote", "header", "footer"}


def _as_list(v) -> list:
    """None / list / tuple / ndarray -> list (avoids ambiguous-truth-value errors on arrays)."""
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return list(v)
    except TypeError:
        return []


def _box4(b) -> Optional[Tuple[float, float, float, float]]:
    try:
        b = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(b) == 4:
        return (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
    if len(b) >= 8:                                   # polygon -> bbox
        xs, ys = b[0::2], b[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _poly_box(p) -> Optional[Tuple[float, float, float, float]]:
    try:
        xs, ys = [float(q[0]) for q in p], [float(q[1]) for q in p]
        return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, IndexError, ValueError):
        return _box4(p)


def _center_in(box, outer) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _inter_frac(a, b) -> float:
    """Share of box `a` that lies inside box `b`."""
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    return max(ix, 0) * max(iy, 0) / area


def _iou(a, b) -> float:
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


def parse_structure_result(data: Dict[str, Any], markdown: str = "", source: str = "paddle") -> StructPage:
    """PP-StructureV3 / PaddleOCR-VL JSON (res.json) -> StructPage. Tolerant of the {"res": {...}} wrapper and missing keys.
    PaddleOCR-VL results carry no `overall_ocr_res`, so their blocks get rec_conf=None (they are verified, not thresholded)."""
    d = data.get("res", data) if isinstance(data, dict) else {}
    if not isinstance(d, dict):
        d = {}
    ocr = d.get("overall_ocr_res") or {}
    texts, scores = _as_list(ocr.get("rec_texts")), _as_list(ocr.get("rec_scores"))
    boxes = _as_list(ocr.get("rec_boxes"))
    if not boxes:
        boxes = _as_list(ocr.get("rec_polys")) or _as_list(ocr.get("dt_polys"))
    lines = []
    for i, t in enumerate(texts):
        if i >= len(scores) or i >= len(boxes):
            break
        bb = _poly_box(boxes[i])
        if bb and str(t).strip():
            lines.append((bb, str(t), float(scores[i])))
    blocks: List[StructBlock] = []
    for it in _as_list(d.get("parsing_res_list")):
        if not isinstance(it, dict):
            continue
        bb = _box4(it.get("block_bbox") or it.get("bbox"))
        if bb is None:
            continue
        label = str(it.get("block_label") or it.get("label") or "text")
        content = str(it.get("block_content") or it.get("content") or "")
        inside = [(t, s) for b, t, s in lines if _center_in(b, bb)]
        w = sum(len(t) for t, _ in inside)
        conf = (sum(s * len(t) for t, s in inside) / w) if w else None
        block_lines = [{"text": t, "bbox_px": list(b), "rec_conf": float(score)} for b, t, score in lines
                       if _center_in(b, bb)]
        blocks.append(StructBlock(len(blocks), label, content, bb, conf, source, len(inside), rec_chars=w,
                                  line_boxes=block_lines))
    return StructPage(blocks, markdown, len(lines), engine="vl" if source == "vl" else "paddle")


# ---------------------------------------------------------------- tables / markdown rendering
class _TableParser(HTMLParser):
    """Minimal table reader. Tolerant of unclosed <td>/<th>/<tr> (VL output is not always well-formed)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[dict]] = []
        self._row = None
        self._cell = None

    @staticmethod
    def _n(v) -> int:
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            return 1

    def _close_cell(self):
        if self._cell is not None:
            self._cell["text"] = " ".join("".join(self._cell["t"]).split())
            if self._row is None:
                self._row = []
            self._row.append(self._cell)
            self._cell = None

    def _close_row(self):
        self._close_cell()
        if self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def finish(self):
        self._close_row()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._close_row()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = []
            self._cell = {"t": [], "cs": self._n(a.get("colspan")), "rs": self._n(a.get("rowspan"))}
        elif tag == "br" and self._cell is not None:
            self._cell["t"].append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._close_cell()
        elif tag in ("tr", "table"):
            self._close_row()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["t"].append(data)


def html_table_to_grid(markup: str) -> Tuple[List[List[str]], bool, bool]:
    """HTML table -> (grid, coherent, has_spans). Spans are flattened (text kept in the top-left cell)."""
    p = _TableParser()
    try:
        p.feed(markup)
        p.close()
        p.finish()
    except Exception:
        return [], False, False
    occ: Dict[Tuple[int, int], str] = {}
    spans = False
    for r, row in enumerate(p.rows):
        c = 0
        for cell in row:
            while (r, c) in occ:
                c += 1
            spans = spans or cell["cs"] > 1 or cell["rs"] > 1
            for dr in range(cell["rs"]):
                for dc in range(cell["cs"]):
                    occ[(r + dr, c + dc)] = cell["text"] if (dr == 0 and dc == 0) else ""
            c += cell["cs"]
    if not occ:
        return [], False, False
    n_r, n_c = max(r for r, _ in occ) + 1, max(c for _, c in occ) + 1
    grid = [[occ.get((r, c), "") for c in range(n_c)] for r in range(n_r)]
    coherent = all(all((r, c) in occ for c in range(n_c)) for r in range(len(p.rows)))
    return grid, coherent, spans


_MD_SEP_ROW = re.compile(r"\|?(?:\s*:?-{3,}:?\s*\|?)+")


def table_text_to_grid(text: str) -> Tuple[List[List[str]], bool]:
    """Table block text (HTML *or* Markdown pipe table) -> (grid, coherent). ([], False) if it is neither."""
    t = (text or "").strip()
    if t.startswith("|"):
        rows: List[List[str]] = []
        for line in t.splitlines():
            line = line.strip()
            if not line.startswith("|") or _MD_SEP_ROW.fullmatch(line):
                continue
            body = line[1:]
            if body.endswith("|") and not body.endswith("\\|"):
                body = body[:-1]
            rows.append([c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", body)])
        if not rows:
            return [], False
        width = max(len(r) for r in rows)
        coherent = all(len(r) == width for r in rows)
        return [r + [""] * (width - len(r)) for r in rows], coherent
    if "<" in t:
        grid, coherent, _ = html_table_to_grid(t)
        return grid, coherent
    return [], False


def grid_to_markdown(grid: List[List[str]]) -> str:
    if not grid:
        return ""
    esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
    n = len(grid[0])
    out = ["| " + " | ".join(esc(x) for x in grid[0]) + " |", "|" + "---|" * n]
    out += ["| " + " | ".join(esc(x) for x in r) + " |" for r in grid[1:]]
    return "\n".join(out)


_LIST_START = re.compile(r"^\s*(?:[\u2022\u00b7\u25cf\u25aa*-]\s+|\(?\d{1,3}[.)]\s+|\(?[a-zA-Z][.)]\s+)")


def join_lines(lines: List[str]) -> str:
    """Join OCR lines into one paragraph: de-hyphenate, keep list items on their own line."""
    from .textlayer import _dehyph
    lines = [l.strip() for l in lines if l.strip()]
    merged: List[str] = []
    for l in _dehyph(lines):
        if merged and not _LIST_START.match(l):
            merged[-1] += " " + l
        else:
            merged.append(l)
    return "\n".join(merged)


def _plain(label: str, text: str) -> str:
    """Block text as plain text (tables: one row per line, cells space-separated; HTML and Markdown tables both handled)."""
    t = text or ""
    if label == "table":
        if t.lstrip().startswith("|"):
            grid, _ = table_text_to_grid(t)
            if grid:
                return "\n".join(" ".join(r) for r in grid)
        elif "<" in t:
            grid, _, _ = html_table_to_grid(t)
            if grid:
                return "\n".join(" ".join(r) for r in grid)
            return _html.unescape(re.sub(r"<[^>]+>", " ", t))
    return t


def block_plain_text(b: StructBlock) -> str:
    return _plain(b.label, b.text)


def block_markdown(b: StructBlock, ignore: Tuple[str, ...] = ()) -> str:
    """One block -> Markdown. Per-block rendering is what lets us swap a block's text without touching the rest."""
    if b.label in ignore:
        return ""
    if b.label == "table":
        if b.text.lstrip().startswith("|"):                       # already Markdown (e.g. rebuilt / VL)
            return b.text
        grid, _, _ = html_table_to_grid(b.text)
        return grid_to_markdown(grid) if grid else b.text.strip()
    t = b.text.strip()
    if not t:
        return ""
    if b.label == "formula":
        return t if t.startswith("$") else f"$$\n{t}\n$$"
    body = join_lines(t.split("\n")) if b.label not in ("image", "chart") else t
    return {"doc_title": "# ", "paragraph_title": "## "}.get(b.label, "") + body


def assemble_markdown(blocks: List[StructBlock], ignore: Tuple[str, ...] = ()) -> str:
    return "\n\n".join(m for m in (block_markdown(b, ignore) for b in blocks) if m.strip())


# ---------------------------------------------------------------- text-quality proxies
_MIXED = re.compile(r"[A-Za-z]+\d+[A-Za-z]+|\d+[A-Za-z]+\d+|[a-z][A-Z][a-z]")
_ID_LIKE = re.compile(r"^[A-Z]{2,6}[-_]?[A-Z0-9-]*\d[A-Z0-9-]*$|^\d+(st|nd|rd|th|[kKmM%])$|^v?\d+(\.\d+)+$|@")
# Tokens that are layout punctuation or list enumerators, not recognition errors: "........", "-----", "(a)", "iv.", "1)".
_PUNCT_RUN = re.compile(r"^[.\-_=\u2014\u2013\u00b7\u2026\u2022*~|/\\]+$")
_ENUM = re.compile(r"^[(\[]?[A-Za-z0-9]{1,4}[)\].:]$|^[(\[][A-Za-z0-9]{1,4}[)\]]$")
# Non-ASCII, non-Latin-extended glyph glued to the next letter/digit at the start of a block (logo read as a character:
# "\u4e09CADENCE"). Common typographic leaders (bullets, quotes, dashes, currency, \u00a7 \u00a9 \u00ae \u2122) are excluded.
_STRAY_LEAD = re.compile(r"^[^\x00-\x7f\u00c0-\u024f\u2018-\u201f\u2022\u2013\u2014\u2026\u00a7\u00a9\u00ae\u2122"
                         r"\u00b0\u00b1\u00b6\u00bf\u00a1\u20a0-\u20cf\s]+(?=[A-Za-z0-9])")
_MIN_SPACE_RATIO = 0.09
_MIN_SPACING_CHARS = 25


def _latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 0x250) / len(letters)


def run_together(text: str, min_ratio: float = _MIN_SPACE_RATIO, min_chars: int = _MIN_SPACING_CHARS) -> bool:
    """True when Latin prose has (almost) no spaces between words ("Documentreference GRP-POL-001 |Version6.0").

    English averages roughly one word break per 5-6 alphanumerics (~0.18); below `min_ratio` the words ran together.
    Skipped for short text, non-Latin scripts, mostly-numeric text (IDs, hashes) and link-like text (URLs, e-mails).

    Pipe ``|``, middle-dot ``·`` (\\u00b7), bullet ``•`` (\\u2022), em-dash ``—`` (\\u2014) and hyphen ``-`` are treated as word
    separators in addition to whitespace, because headers / footers commonly use them (e.g.
    ``GRP-POL-001 | Version 6.0 | Internal``).  Without this, every page with such a header is falsely flagged."""
    # Normalise visual separators to spaces so they count as word breaks.
    norm = re.sub(r'[|\u00b7\u2022\u2014\u2013\-]', ' ', text)
    toks = norm.split()
    alnum = sum(c.isalnum() for c in text)
    if alnum < min_chars or not toks:
        return False
    if sum(c.isalpha() for c in text) < 0.7 * alnum or _latin_share(text) < 0.9:
        return False
    if sum(1 for t in toks if "@" in t or "/" in t or "://" in t or t.lower().startswith("www.")) >= 0.5 * len(toks):
        return False
    words = [t for t in toks if any(c.isalnum() for c in t)]
    return (len(words) - 1) / alnum < min_ratio


def text_quality(text: str, check_spacing: bool = True, min_space_ratio: float = _MIN_SPACE_RATIO,
                 min_spacing_chars: int = _MIN_SPACING_CHARS) -> Dict[str, float]:
    """Cheap, engine-independent proxies for recognition errors (NOT accuracy; accuracy needs reference text).

    `err` = junk-token share + mixed-alphanumeric-token share (+0.5 when words ran together and check_spacing is on, so
    that it clearly exceeds max_err_rate and any 'clearly fewer errors' margin). Dot-leaders and list enumerators are not
    counted as junk."""
    toks = [t for t in text.split() if t]
    alnum = sum(c.isalnum() for c in text)
    if not toks:
        return {"alnum": 0, "tokens": 0, "junk": 0.0, "mixed": 0.0, "runtogether": 0.0, "err": 0.0}
    junk = sum(1 for t in toks
               if not (_PUNCT_RUN.match(t) or _ENUM.match(t)) and _junk(RawWord(t, 100.0, 0, 0, 0, 0))) / len(toks)
    mixed = sum(1 for t in toks if len(t) >= 4 and _MIXED.search(t) and not _ID_LIKE.search(t)) / len(toks)
    rt = bool(check_spacing and run_together(text, min_space_ratio, min_spacing_chars))
    return {"alnum": alnum, "tokens": len(toks), "junk": round(junk, 4), "mixed": round(mixed, 4),
            "runtogether": 1.0 if rt else 0.0, "err": round(junk + mixed + (0.5 if rt else 0.0), 4)}


def is_clear_improvement(paddle_text: str, tess_text: str, s: Optional[PaddleSettings] = None
                         ) -> Tuple[bool, str, Dict[str, Any]]:
    """Should the alternative text (Tesseract) replace PP-StructureV3 text for the same block? Raw confidences are NOT used.

    A) fills missing content : alternative has >=20% more alphanumerics and is not noisier.
    B) fewer recognition errs: similar length (not truncated / not hallucinated) and clearly lower error proxy.
    Anything else keeps the PP-StructureV3 text (the caller records the uncertainty)."""
    kw = {} if s is None else {"min_space_ratio": s.min_space_ratio, "min_spacing_chars": s.min_spacing_chars}
    pq, tq = text_quality(paddle_text, **kw), text_quality(tess_text, **kw)
    sim = difflib.SequenceMatcher(None, paddle_text, tess_text, autojunk=False).ratio() \
        if max(len(paddle_text), len(tess_text)) < 4000 else None
    ratio = tq["alnum"] / max(pq["alnum"], 1)
    m = {"paddle": pq, "tesseract": tq, "alnum_ratio": round(ratio, 2), "similarity": None if sim is None else round(sim, 3)}
    if tq["alnum"] == 0:
        return False, "tesseract produced no text", m
    if sim is not None and sim >= 0.98:
        return False, "candidates agree", m
    # Same words, different layout (line breaks, punctuation, quotes, order): compare on normalized text, but ONLY when
    # both candidates are equally clean, so that fused Paddle text is still replaced by a properly spaced Tesseract text.
    seq_n, bag_n = _agreement(paddle_text, tess_text), _bag_agreement(paddle_text, tess_text)
    m["agreement"] = {"sequence": round(seq_n, 3), "bag": round(bag_n, 3)}
    if (not pq["runtogether"] and not tq["runtogether"] and abs(pq["err"] - tq["err"]) <= 0.02
            and 0.9 <= ratio <= 1.1 and max(seq_n, bag_n) >= 0.95):
        return False, "candidates agree", m
    if ratio >= 1.2 and tq["err"] <= pq["err"] + 0.02:
        return True, "tesseract recovered missing content", m
    if 0.85 <= ratio <= 1.3 and tq["err"] + 0.05 <= pq["err"]:
        return True, "tesseract has clearly fewer recognition errors", m
    return False, "no clear improvement", m


# ---------------------------------------------------------------- quality gate
@dataclass
class GateResult:
    ok: bool
    layout_ok: bool
    rec_conf: Optional[float]               # OCR-char-weighted mean over OCR-sourced blocks (0-1)
    failing: List[int]                      # block idx that must go to fallback
    reasons: List[str]
    metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)   # non-failing observations (e.g. minor uncovered ink)


def ink_coverage_gap(img: Optional[np.ndarray], blocks: List[StructBlock], ink_thresh: int = 170,
                     margin: int = 8, stride: int = 2) -> Optional[float]:
    """Share of dark pixels in `img` that lie outside every block that actually holds content (text, table, image/chart).

    A completeness test that needs no reference: text the layout/OCR missed is ink nobody claimed. `img` must be the raster
    that was sent to the engine (block boxes are in its pixel space). Returns None when it cannot be computed."""
    if img is None or getattr(img, "size", 0) == 0 or not blocks:
        return None
    g = img if img.ndim == 2 else img[..., :3].min(axis=2)
    ink = g[::stride, ::stride] < ink_thresh
    total = int(ink.sum())
    if total < 50:
        return 0.0
    covered = np.zeros(ink.shape, dtype=bool)
    for b in blocks:
        if not (b.text.strip() or b.label in _NO_TEXT_LABELS):
            continue                                   # an empty block must not hide missing text
        x0, y0 = max(int((b.bbox[0] - margin) / stride), 0), max(int((b.bbox[1] - margin) / stride), 0)
        x1, y1 = int((b.bbox[2] + margin) / stride) + 1, int((b.bbox[3] + margin) / stride) + 1
        covered[y0:y1, x0:x1] = True
    return float((ink & ~covered).sum()) / total


def quality_gate(sp: StructPage, ink_fraction: float, s: PaddleSettings, min_ink: float = 0.0004,
                 img: Optional[np.ndarray] = None) -> GateResult:
    """Recognition confidence + completeness / junk / spacing / reading-order / table checks.
    Native-sourced blocks are trusted and skipped. Layout-detection scores are never read here.
    Works for PP-StructureV3 pages and (verified) PaddleOCR-VL pages: VL blocks have rec_conf=None, so only the
    content checks apply to them.  Pass `img` (the raster sent to the engine) to enable the uncovered-ink check."""
    reasons: List[str] = []
    warnings: List[str] = []
    failing: List[int] = []
    layout_ok = True
    if sp.error:
        return GateResult(False, False, None, [], [f"{sp.engine}_error: {sp.error}"], {})
    ocr_blocks = [b for b in sp.blocks if b.source in ("paddle", "vl")]
    all_text = " ".join(block_plain_text(b) for b in sp.blocks)
    total_alnum = sum(c.isalnum() for c in all_text)
    if ink_fraction >= min_ink and (not sp.blocks or total_alnum == 0):
        layout_ok = False
        reasons.append("empty_result_on_inked_page")
    elif ink_fraction >= 0.01 and total_alnum < s.min_chars_on_inked_page:
        layout_ok = False
        reasons.append(f"incomplete: only {total_alnum} chars on an inked page")
    uncovered = ink_coverage_gap(img, sp.blocks) if img is not None else None
    if uncovered is not None:
        if uncovered > s.max_uncovered_ink:
            layout_ok = False
            reasons.append(f"incomplete: {uncovered:.0%} of the ink lies outside every block")
        elif uncovered > s.max_uncovered_ink / 3:
            warnings.append(f"uncovered_ink_minor:{uncovered:.2f}")
    txt = [b for b in sp.blocks if b.label in _TEXT_LABELS and b.text.strip()]
    overlaps = sum(1 for i, a in enumerate(txt) for b in txt[i + 1:] if _iou(a.bbox, b.bbox) > s.max_overlap_iou)
    if overlaps:
        layout_ok = False
        reasons.append(f"overlapping_blocks:{overlaps}")
    if len(txt) >= 4:
        jumps = 0
        for a, b in zip(txt, txt[1:]):
            same_col = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0]) > 0.5 * min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
            if same_col and b.bbox[3] < a.bbox[1]:                    # next block sits entirely above the previous one
                jumps += 1
        if jumps / (len(txt) - 1) > s.max_order_jump_frac:
            layout_ok = False
            reasons.append(f"reading_order_jumps:{jumps}")
    latin = _latin_lang(s.lang)
    for b in ocr_blocks:
        if b.label in _NO_TEXT_LABELS and not b.text.strip():
            continue
        why = []
        tiny = 0 < b.rec_chars < s.min_gate_chars            # page numbers etc.: one low score must not trigger a fallback
        if b.rec_conf is not None and not tiny and b.rec_conf < s.rec_conf_threshold:
            why.append(f"rec_conf {b.rec_conf:.2f}<{s.rec_conf_threshold}")
        if b.label == "table":
            grid, coherent = table_text_to_grid(b.text)
            if not grid:
                why.append("table_unparsable")
            else:
                filled = sum(1 for r in grid for c in r if c.strip()) / max(sum(len(r) for r in grid), 1)
                if not coherent:
                    why.append("table_ragged")
                if filled < s.min_table_fill:
                    why.append(f"table_sparse:{filled:.2f}")
                nf, nl = table_fused_cells(b.text, min_ratio=s.min_space_ratio) if latin else (0, 0)
                if nf >= _TABLE_FUSED_MIN_CELLS and nf >= 0.2 * max(nl, 1):
                    why.append(f"table_run_together:{nf}/{nl}")
        elif b.label != "formula":
            prose = latin and b.label in _PROSE_LABELS
            q = text_quality(b.text, check_spacing=prose, min_space_ratio=s.min_space_ratio,
                             min_spacing_chars=s.min_spacing_chars)
            if q["runtogether"]:
                why.append("run_together")
            elif q["tokens"] >= 3 and q["err"] > s.max_err_rate:
                why.append(f"err_rate {q['err']:.2f}")
            if prose and _STRAY_LEAD.match(b.text.lstrip()):
                why.append("stray_glyph")
        if why:
            failing.append(b.idx)
            b.note = "; ".join(why)
    rb = [b for b in ocr_blocks if b.rec_conf is not None]
    wts = [(b.rec_chars or len(b.text)) for b in rb]                  # OCR characters, not HTML length
    w = sum(wts)
    rec = (sum(b.rec_conf * x for b, x in zip(rb, wts)) / w) if w else None
    if rec is not None and rec < s.rec_conf_threshold:
        reasons.append(f"page_rec_conf {rec:.2f}<{s.rec_conf_threshold}")
    if failing:
        reasons.append(f"failing_blocks:{len(failing)}")
    metrics = {"blocks": len(sp.blocks), "ocr_lines": sp.n_ocr_lines, "alnum": total_alnum}
    if uncovered is not None:
        metrics["uncovered_ink"] = round(uncovered, 4)
    return GateResult(layout_ok and not failing, layout_ok, rec, failing, reasons, metrics, warnings)


# ====================================================================================================
#  Engine plumbing shared by PP-StructureV3 and PaddleOCR-VL
# ====================================================================================================
class EngineUnavailableError(RuntimeError):
    """An engine failed several times in a row: the installation / runtime is broken, not one bad page. Raised instead of
    silently degrading every page to the fallback path."""


def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs, []
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []
    keep = {k: v for k, v in kwargs.items() if k in params}
    return keep, [k for k in kwargs if k not in params]


def _error_hint(msg: str) -> str:
    low = msg.lower()
    if "convertpirattribute2runtimeattribute" in low or "onednn" in low or "mkldnn" in low:
        return ("known PaddlePaddle PIR/oneDNN CPU inference error: disable oneDNN (enable_mkldnn=False / --no-mkldnn) "
                "or use the tested paddlepaddle==3.2.2")
    if "out of memory" in low or "resourceexhausted" in low or ("cuda" in low and "memory" in low):
        return "GPU out of memory: lower the render DPI, run one worker, or serve the VL model via a separate backend"
    return ""


def _free_gpu() -> None:
    gc.collect()
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _result_json(r: Any) -> Dict[str, Any]:
    """PaddleOCR result object -> dict (`.json` is a property in 3.x, tolerate a method / JSON string / dict-like)."""
    data = getattr(r, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            data = None
    if isinstance(data, dict):
        return data
    try:
        return dict(r)
    except Exception:
        return {}


def _result_markdown(r: Any) -> str:
    md = getattr(r, "markdown", None)
    if callable(md):
        md = md()
    if isinstance(md, dict):
        return md.get("markdown_texts", "") or ""
    return md if isinstance(md, str) else ""


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.ascontiguousarray(np.dstack([img, img, img]))
    return img


class _EngineBase:
    name = ""

    def __init__(self, s: PaddleSettings):
        self.settings = s
        self.dropped: List[str] = []
        self.last_error = ""
        self._consec_failures = 0
        self.pipe = None

    def _ok(self) -> None:
        self._consec_failures = 0

    def _fail(self, e: Exception) -> str:
        """Record one engine exception; returns the message to store on the page. Raises EngineUnavailableError after
        `abort_after_errors` consecutive failures."""
        hint = _error_hint(str(e))
        msg = f"{type(e).__name__}: {str(e)[:240]}" + (f" | hint: {hint}" if hint else "")
        self.last_error = msg
        self._consec_failures += 1
        log.error("%s failed (%d in a row): %s", self.name, self._consec_failures, msg)
        n = self.settings.abort_after_errors
        if n and self._consec_failures >= n:
            raise EngineUnavailableError(
                f"{self.name} failed {self._consec_failures} times in a row; aborting instead of degrading every page. "
                f"Last error: {msg}") from e
        _free_gpu()
        return msg

    def close(self) -> None:
        pipe, self.pipe = self.pipe, None
        if pipe is not None and hasattr(pipe, "close"):
            try:
                pipe.close()
            except Exception:
                pass


# ====================================================================================================
#  PP-StructureV3 engine wrapper
# ====================================================================================================
class PPStructureV3Engine(_EngineBase):
    """Thin wrapper over paddleocr.PPStructureV3 (PaddleOCR 3.x). One instance per engine_key."""
    name = "pp-structurev3"

    def __init__(self, s: PaddleSettings):
        super().__init__(s)
        try:
            from paddleocr import PPStructureV3
        except ImportError as e:                                   # fail loudly: this is the primary engine
            raise RuntimeError("PP-StructureV3 is not installed. Run: pip install \"paddlepaddle>=3.0,<4.0\" "
                               "\"paddleocr[doc-parser]>=3.0,<4.0\"") from e
        kw = dict(s.enabled_modules())
        kw["lang"] = s.lang
        kw["enable_mkldnn"] = s.enable_mkldnn
        kw["text_det_limit_side_len"] = s.limit_side_len
        kw["text_det_limit_type"] = "max"
        if s.device:
            kw["device"] = s.device
        kw, self.dropped = _filter_kwargs(PPStructureV3.__init__, kw)
        if self.dropped:
            log.warning("PP-StructureV3 ignored unsupported option(s) %s: the installed paddleocr does not accept them, "
                        "so the corresponding settings have NO effect", self.dropped)
        self.pipe = PPStructureV3(**kw)

    def parse(self, img: np.ndarray) -> StructPage:
        """img: uint8 gray or BGR page raster. Returns blocks in the pixel space of `img`."""
        img = _to_bgr(img)
        t0 = time.time()
        try:
            results = list(self.pipe.predict(img))
            if not results:
                return StructPage([], error="empty prediction")
            r = results[0]
            sp = parse_structure_result(_result_json(r), _result_markdown(r))
            sp.seconds = round(time.time() - t0, 2)
            self._ok()
            return sp
        except Exception as e:             # a crashed page must not kill the document (unless it keeps happening)
            return StructPage([], error=self._fail(e), seconds=round(time.time() - t0, 2))


_STRUCT_ENGINES: Dict[tuple, Any] = {}


def get_structure_engine(s: PaddleSettings):
    key = s.engine_key()
    if key not in _STRUCT_ENGINES:
        _STRUCT_ENGINES[key] = PPStructureV3Engine(s)
    return _STRUCT_ENGINES[key]


def release_structure_engine(s: Optional[PaddleSettings] = None) -> None:
    """Close and drop cached PP-StructureV3 engines so VL can use GPU memory without model overlap."""
    keys = list(_STRUCT_ENGINES) if s is None else [s.engine_key()]
    for key in keys:
        engine = _STRUCT_ENGINES.pop(key, None)
        if engine is not None:
            engine.close()
    _free_gpu()


# ====================================================================================================
#  PaddleOCR-VL (fallback engine)
#
#  PaddleOCR-VL returns text blocks in the same `parsing_res_list` schema as PP-StructureV3 but NO per-token recognition
#  scores, so a VL result cannot be accepted by a confidence threshold. It is accepted by *verification*: not empty, no
#  repetition loop, plausible length against the failed candidate, readable text, and agreement with a trusted reference
#  where one exists (the Paddle text when it looks healthy, or an independent text such as Tesseract's).
#  Region recognition uses `use_layout_detection=False` + `prompt_label` (the docs state prompt_label only takes effect
#  with layout detection off).  The VL model needs a GPU (no CPU / Arm support); load it lazily and release it when done.
# ====================================================================================================
_MIN_CHAR_AGREEMENT = 0.85      # VL vs spacing-damaged Paddle characters: below this the VL text is left "unverified"
_TABLE_FUSED_MIN_CELLS = 3      # a table is "run together" when at least this many (and >=20% of the long) cells are fused
VL_PROMPT_BY_LABEL = {"table": "table", "formula": "formula", "chart": "chart"}   # everything else -> "ocr"


@dataclass
class VLRegion:
    text: str = ""
    prompt: str = "ocr"
    seconds: float = 0.0
    error: str = ""


class PaddleOCRVLEngine(_EngineBase):
    name = "paddleocr-vl"

    def __init__(self, s: PaddleSettings):
        super().__init__(s)
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as e:
            raise RuntimeError("PaddleOCR-VL is not installed or this paddleocr version predates it. Run: "
                               "pip install -U \"paddleocr[doc-parser]\" (needs a GPU build of PaddlePaddle)") from e
        kw: Dict[str, Any] = {
            "use_layout_detection": True,
            "use_chart_recognition": s.vl_use_chart_recognition,
            "use_ocr_for_image_block": s.vl_use_ocr_for_image_block,
            "format_block_content": True,
            "use_doc_orientation_classify": False,   # the pipeline already orients/deskews; boxes must keep mapping to the page
            "use_doc_unwarping": False,
        }
        device = s.vl_device or s.device
        if device:
            kw["device"] = device
        if s.vl_pipeline_version:
            kw["pipeline_version"] = s.vl_pipeline_version
        if s.vl_backend:
            kw["vl_rec_backend"] = s.vl_backend
        if s.vl_server_url:
            kw["vl_rec_server_url"] = s.vl_server_url
        kw, self.dropped = _filter_kwargs(PaddleOCRVL.__init__, kw)
        if self.dropped:
            log.warning("PaddleOCR-VL ignored unsupported option(s) %s: the installed paddleocr does not accept them, "
                        "so the corresponding settings have NO effect", self.dropped)
        self.pipe = PaddleOCRVL(**kw)

    def parse_page(self, img: np.ndarray) -> StructPage:
        """Whole-page parse. Blocks have source="vl" and rec_conf=None (verify with `verify_vl_page`)."""
        img = _to_bgr(img)
        t0 = time.time()
        try:
            results = list(self.pipe.predict(img, use_layout_detection=True))
            if not results:
                return StructPage([], error="empty prediction", engine="vl")
            r = results[0]
            sp = parse_structure_result(_result_json(r), _result_markdown(r), source="vl")
            sp.seconds = round(time.time() - t0, 2)
            self._ok()
            return sp
        except Exception as e:
            return StructPage([], error=self._fail(e), engine="vl", seconds=round(time.time() - t0, 2))

    def recognize_region(self, crop: np.ndarray, label: str = "text") -> VLRegion:
        """Recognise one cropped block (layout detection off, prompt chosen from the block label)."""
        prompt = VL_PROMPT_BY_LABEL.get(label, "ocr")
        t0 = time.time()
        try:
            results = list(self.pipe.predict(_to_bgr(crop), use_layout_detection=False, prompt_label=prompt))
            if not results:
                return VLRegion(prompt=prompt, error="empty prediction", seconds=round(time.time() - t0, 2))
            r = results[0]
            md = _result_markdown(r)
            sp = parse_structure_result(_result_json(r), md, source="vl")
            parts = [b.text.strip() for b in sp.blocks if b.text.strip()]
            text = "\n".join(parts) if parts else md.strip()
            self._ok()
            return VLRegion(text=text, prompt=prompt, seconds=round(time.time() - t0, 2))
        except Exception as e:
            return VLRegion(prompt=prompt, error=self._fail(e), seconds=round(time.time() - t0, 2))


_VL_ENGINES: Dict[tuple, PaddleOCRVLEngine] = {}


def get_vl_engine(s: PaddleSettings) -> PaddleOCRVLEngine:
    """Lazy: the VL model is only loaded the first time a page / block actually needs it."""
    key = s.vl_key()
    if key not in _VL_ENGINES:
        _VL_ENGINES[key] = PaddleOCRVLEngine(s)
    return _VL_ENGINES[key]


def release_vl_engine(s: Optional[PaddleSettings] = None) -> None:
    """Close and drop the cached VL engine(s) (all of them when `s` is None) and free GPU memory."""
    keys = list(_VL_ENGINES) if s is None else [s.vl_key()]
    for k in keys:
        eng = _VL_ENGINES.pop(k, None)
        if eng is not None:
            eng.close()
    _free_gpu()


# ---------------------------------------------------------------- VL verification
def find_degenerate(text: str, is_table: bool = False, repeats: int = 7) -> str:
    """Non-empty reason string when text looks like a generation loop (same phrase back-to-back many times)."""
    t = re.sub(r"[-_=.\u00b7\u2026*~|]{3,}", " ", text or "")[:6000]     # rules / dot leaders are not repetition
    if not t.strip():
        return ""
    reps = max(25 if is_table else repeats, 2)      # tables legitimately repeat cells ("N/A", "Yes")
    m = re.search(r"(.{3,80}?)\1{%d,}" % (reps - 1), t, re.S)
    if m and any(c.isalnum() for c in m.group(1)):
        return f"repetition loop ({m.group(1)[:30]!r} repeated {reps}+ times)"
    return ""


def _norm_alnum(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum())


def _agreement(a: str, b: str) -> float:
    """0-1 similarity of two texts ignoring case, spacing and punctuation (so a missing space does not count as a
    disagreement)."""
    x, y = _norm_alnum(a), _norm_alnum(b)
    if not x or not y:
        return 0.0
    if max(len(x), len(y)) <= 3000:
        return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio()
    # long texts: character-trigram overlap (linear time)
    gx = [x[i:i + 3] for i in range(len(x) - 2)]
    gy = [y[i:i + 3] for i in range(len(y) - 2)]
    cy: Dict[str, int] = {}
    for g in gy:
        cy[g] = cy.get(g, 0) + 1
    hit = 0
    for g in gx:
        if cy.get(g, 0) > 0:
            cy[g] -= 1
            hit += 1
    return 2.0 * hit / max(len(gx) + len(gy), 1)


def _bag_agreement(a: str, b: str) -> float:
    """Order-insensitive word overlap (case / punctuation ignored): 1.0 = same multiset of words."""
    ta = re.findall(r"[a-z0-9]+", a.lower())
    tb = re.findall(r"[a-z0-9]+", b.lower())
    if not ta or not tb:
        return 0.0
    cb: Dict[str, int] = {}
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    hit = 0
    for t in ta:
        if cb.get(t, 0) > 0:
            cb[t] -= 1
            hit += 1
    return 2.0 * hit / (len(ta) + len(tb))


def text_contained(needle: str, haystack: str) -> float:
    """Share of `needle` (alphanumerics only, case-insensitive, spacing ignored) found, in order, inside `haystack`.
    Used to decide whether a short block / OCR line is already covered by a longer text."""
    x, y = _norm_alnum(needle), _norm_alnum(haystack)
    if not x or not y:
        return 0.0
    m = difflib.SequenceMatcher(None, x, y, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / len(x)


def collapse_spaces(text: str) -> str:
    """Collapse runs of spaces / tabs inside each line (VL output sometimes carries double spaces); newlines are kept."""
    return "\n".join(re.sub(r"[ \t]{2,}", " ", ln).rstrip() for ln in (text or "").split("\n"))


def table_fused_cells(text: str, min_chars: int = 20, min_ratio: float = _MIN_SPACE_RATIO) -> Tuple[int, int]:
    """(cells whose words ran together, cells long enough to judge) for a table block. Short cells are skipped: a single
    long word is legitimate, a 20+ character cell with almost no word breaks is not."""
    grid, _ = table_text_to_grid(text) if text.lstrip().startswith("|") else (None, False)
    if not grid and "<" in (text or ""):
        grid, _, _ = html_table_to_grid(text)
    n_long = n_fused = 0
    for row in grid or []:
        for cell in row:
            if sum(c.isalnum() for c in cell) >= min_chars:
                n_long += 1
                if run_together(cell, min_ratio, min_chars):
                    n_fused += 1
    return n_fused, n_long


def verify_vl_text(paddle_text: str, vl_text: str, s: PaddleSettings, label: str = "text",
                   reference: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """May `vl_text` replace `paddle_text` for one block? Never uses engine confidences.

    Rejects: empty / loop / noisy / ragged-table output, implausible length, and disagreement with a trusted reference.
    `reference` is an independent text for the same region (e.g. Tesseract); the Paddle text counts as a reference only
    when it looks healthy itself. With no trusted reference the VL text is accepted but marked unverified
    (metrics["verified_against"] == "none") so the caller can flag it."""
    m: Dict[str, Any] = {}
    is_table = label == "table"
    vl_plain, pd_plain = _plain(label, vl_text), _plain(label, paddle_text)
    vq = text_quality(vl_plain, check_spacing=_latin_lang(s.lang) and label in _PROSE_LABELS,
                      min_space_ratio=s.min_space_ratio, min_spacing_chars=s.min_spacing_chars)
    m["vl"] = vq
    if vq["alnum"] == 0:
        return False, "vl produced no text", m
    loop = find_degenerate(vl_plain, is_table, s.vl_loop_repeats)
    if loop:
        return False, f"vl output degenerate: {loop}", m
    if is_table:
        grid, coherent = table_text_to_grid(vl_text)
        if not grid:
            return False, "vl table unparsable", m
        filled = sum(1 for r in grid for c in r if c.strip()) / max(sum(len(r) for r in grid), 1)
        m["vl_table_fill"] = round(filled, 3)
        if not coherent:
            return False, "vl table ragged", m
        if filled < s.min_table_fill:
            return False, f"vl table sparse ({filled:.2f})", m
        nf, nl = table_fused_cells(vl_text, min_ratio=s.min_space_ratio)
        if nf >= _TABLE_FUSED_MIN_CELLS and nf >= 0.2 * max(nl, 1):
            return False, f"vl table cells still run together ({nf}/{nl})", m
    elif label != "formula" and (vq["runtogether"] or (vq["tokens"] >= 3 and vq["err"] > s.max_err_rate)):
        return False, f"vl text noisy (err {vq['err']:.2f})", m
    pa = sum(c.isalnum() for c in pd_plain)
    ratio = vq["alnum"] / max(pa, 1)
    m["alnum_ratio"] = round(ratio, 2)
    if pa >= 20 and not (s.vl_min_len_ratio <= ratio <= s.vl_max_len_ratio):
        return False, f"vl length implausible (ratio {ratio:.2f} vs failed candidate)", m
    pq = text_quality(pd_plain, check_spacing=_latin_lang(s.lang) and label in _PROSE_LABELS,
                      min_space_ratio=s.min_space_ratio, min_spacing_chars=s.min_spacing_chars)
    paddle_trusted = pa >= 20 and pq["err"] <= s.max_err_rate and not pq["runtogether"]
    refs = []
    if reference and any(c.isalnum() for c in reference):
        refs.append(("reference", reference))
    if paddle_trusted:
        refs.append(("paddle", pd_plain))
    if not refs:
        # The commonest Paddle failure is words fused together ("Policyowner"): its CHARACTERS are still right, only the
        # spaces are missing. `_agreement` ignores spacing, so such text is a valid character-level reference as long as it
        # is not junk. A VL text that reproduces those characters AND restores the spaces is therefore verified.
        spacing_only = pa >= 20 and pq["junk"] <= s.max_err_rate and (pq["runtogether"] or is_table)
        if spacing_only:
            chars = _agreement(vl_plain, pd_plain)
            m["agreement"] = {"paddle-chars": round(chars, 3)}
            if chars >= _MIN_CHAR_AGREEMENT:
                m["verified_against"] = "paddle-chars"
                return True, f"vl verified against paddle characters ({chars:.2f}); spacing repaired", m
        m["verified_against"] = "none"
        return True, "vl accepted; no trusted reference to verify against", m
    sims = [(name, _agreement(vl_plain, txt)) for name, txt in refs]
    best_name, best = max(sims, key=lambda x: x[1])
    m["agreement"] = {n: round(v, 3) for n, v in sims}
    if best < s.vl_min_agreement:
        return False, f"vl disagrees with every trusted reference (best {best:.2f})", m
    m["verified_against"] = best_name
    return True, f"vl verified against {best_name} ({best:.2f})", m


def verify_vl_page(vl_page: StructPage, paddle_page: Optional[StructPage], ink_fraction: float, s: PaddleSettings,
                   min_ink: float = 0.0004, img: Optional[np.ndarray] = None
                   ) -> Tuple[bool, List[str], GateResult]:
    """Accept a whole-page VL parse? Runs the content checks of `quality_gate` (VL blocks carry no recognition
    confidence) plus loop and length checks against the failed Paddle page. Returns (ok, reasons, gate)."""
    gate = quality_gate(vl_page, ink_fraction, s, min_ink, img)
    reasons = list(gate.reasons)
    ok = gate.ok
    if vl_page.error:
        return False, reasons, gate
    for b in vl_page.blocks:
        loop = find_degenerate(block_plain_text(b), b.label == "table", s.vl_loop_repeats)
        if loop:
            ok = False
            reasons.append(f"block {b.idx}: vl output degenerate: {loop}")
    if paddle_page is not None and not paddle_page.error:
        pa = sum(c.isalnum() for b in paddle_page.blocks for c in block_plain_text(b))
        va = sum(c.isalnum() for b in vl_page.blocks for c in block_plain_text(b))
        if pa >= 30:
            ratio = va / pa
            gate.metrics["vl_alnum_ratio"] = round(ratio, 2)
            if ratio < s.vl_min_len_ratio:
                ok = False
                reasons.append(f"vl_missing_content: {ratio:.2f} of the Paddle character count")
            elif ratio > s.vl_max_len_ratio:
                ok = False
                reasons.append(f"vl_excess_content: {ratio:.2f} of the Paddle character count")
    return ok, reasons, gate


def vl_parse_page(img: np.ndarray, s: PaddleSettings) -> StructPage:
    """Convenience: lazy-load the VL engine and parse a whole page raster."""
    return get_vl_engine(s).parse_page(img)


def crop_block(img: np.ndarray, bbox: Tuple[float, float, float, float], margin: int = 12) -> Optional[np.ndarray]:
    """Crop `bbox` (+margin) out of the page raster; None when the crop would be too small to recognise."""
    H, W = img.shape[:2]
    x0, y0 = max(int(bbox[0] - margin), 0), max(int(bbox[1] - margin), 0)
    x1, y1 = min(int(bbox[2] + margin) + 1, W), min(int(bbox[3] + margin) + 1, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return np.ascontiguousarray(_to_bgr(img[y0:y1, x0:x1]))


@dataclass
class VLRepairReport:
    replaced: List[int] = field(default_factory=list)            # block idx whose text was replaced by verified VL text
    kept: Dict[int, str] = field(default_factory=dict)           # block idx -> why the Paddle text was kept
    errors: Dict[int, str] = field(default_factory=dict)         # block idx -> VL engine error
    unverified: List[int] = field(default_factory=list)          # replaced, but no trusted reference existed
    seconds: float = 0.0
    metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    @property
    def still_weak(self) -> List[int]:
        """Blocks VL could not fix: the caller should flag these (needs_review)."""
        return sorted(set(self.kept) | set(self.errors))


def vl_repair_blocks(engine: PaddleOCRVLEngine, img: np.ndarray, blocks: List[StructBlock], failing: List[int],
                     s: PaddleSettings, references: Optional[Dict[int, str]] = None) -> VLRepairReport:
    """Layout passed but some blocks failed the gate: re-recognise only those blocks with VL and replace the text when
    `verify_vl_text` accepts it. Mutates `blocks` (text, source, rec_conf, note); positions / labels are untouched, so
    the rest of the page is not disturbed. `references` maps block idx -> independent text (e.g. Tesseract lines)."""
    rep = VLRepairReport()
    t0 = time.time()
    by_idx = {b.idx: b for b in blocks}
    for idx in failing:
        b = by_idx.get(idx)
        if b is None:
            continue
        if b.label in _NO_TEXT_LABELS or not (b.label in _TEXT_LABELS or b.label in ("table", "formula")):
            rep.kept[idx] = f"label '{b.label}' is not recoverable by VL"
            continue
        # A block a few pixels wide/high holds no readable text; a VL answer for it would be unverifiable invention.
        if (b.bbox[2] - b.bbox[0]) < 10 or (b.bbox[3] - b.bbox[1]) < 10:
            rep.kept[idx] = "block too small to recognise"
            continue
        crop = crop_block(img, b.bbox, s.vl_crop_margin_px)
        if crop is None:
            rep.kept[idx] = "block too small to crop"
            continue
        region = engine.recognize_region(crop, b.label)
        if region.error:
            rep.errors[idx] = region.error
            b.note += f" | VL error: {region.error}"
            continue
        ok, why, metrics = verify_vl_text(b.text, region.text, s, b.label, (references or {}).get(idx))
        metrics["prompt"], metrics["seconds"] = region.prompt, region.seconds
        rep.metrics[idx] = {"decision": why, **metrics}
        if ok:
            b.note = f"text replaced by VL ({why}); was: {b.note}"
            b.text, b.source, b.rec_conf, b.rec_chars = (region.text if b.label == "table" else collapse_spaces(region.text)), "vl", None, 0
            rep.replaced.append(idx)
            if metrics.get("verified_against") == "none":
                rep.unverified.append(idx)
        else:
            rep.kept[idx] = why
            b.note += f" | kept Paddle: {why}"
    rep.seconds = round(time.time() - t0, 2)
    return rep


# ====================================================================================================
#  Tesseract fallback with positions (kept: independent verifier / legacy fallback)
# ====================================================================================================
@dataclass
class TLine:
    text: str
    conf: float                              # Tesseract word-confidence mean (0-100); only used to judge Tesseract itself
    bbox: Tuple[float, float, float, float]  # px of the page image
    col: int = 0
    para: int = 0
    n_words: int = 0


def tesseract_lines(runner: "Runner", img: np.ndarray, cfg: Config) -> List[TLine]:
    """Positioned Tesseract lines (TSV word boxes grouped by line) in reading order: column, then top-to-bottom."""
    from . import layout as lo
    cap = pp.estimate_cap_height(img) or 25.0
    cols = lo.find_columns(img, cap) if cfg.split_columns else [(0, img.shape[1])]
    out: List[TLine] = []
    for ci, (x0, x1) in enumerate(cols):
        crop = img[:, x0:x1]
        if pp.ink_fraction(crop) < cfg.min_ink_frac:
            continue
        words, _, _ = runner.ladder(crop, page_text_ladder(cfg), cfg.accept_score)
        by_line: Dict[int, List[RawWord]] = {}
        for w in words:
            by_line.setdefault(w.line, []).append(w)
        for ws in by_line.values():
            ws.sort(key=lambda w: w.x0)
            chars = sum(len(w.text) for w in ws) or 1
            out.append(TLine(" ".join(w.text for w in ws), sum(w.conf * len(w.text) for w in ws) / chars,
                             (min(w.x0 for w in ws) + x0, min(w.y0 for w in ws),
                              max(w.x1 for w in ws) + x0, max(w.y1 for w in ws)), ci, ws[0].block, len(ws)))
    out.sort(key=lambda l: (l.col, l.bbox[1], l.bbox[0]))
    return out


def match_lines_to_blocks(lines: List[TLine], blocks: List[StructBlock], min_frac: float = 0.5
                          ) -> Tuple[Dict[int, List[TLine]], List[TLine]]:
    """Assign each Tesseract line to the PP-StructureV3 block holding most of it (same page, bbox overlap).
    Returns (block idx -> lines, orphans). Orphans lie outside every block: possibly missing content."""
    assigned: Dict[int, List[TLine]] = {}
    orphans: List[TLine] = []
    for l in lines:
        best, best_f = None, 0.0
        for b in blocks:
            f = _inter_frac(l.bbox, b.bbox)
            if f > best_f:
                best, best_f = b, f
        if best is None or best_f < min_frac:
            orphans.append(l)
        elif best.label in _TEXT_LABELS:
            assigned.setdefault(best.idx, []).append(l)
        # lines inside tables / figures / formulas are covered by that block; not orphans, not text candidates
    for v in assigned.values():
        v.sort(key=lambda l: (l.col, round(l.bbox[1] / max(l.bbox[3] - l.bbox[1], 1)), l.bbox[0]))
    return assigned, orphans


def lines_to_text(lines: List[TLine]) -> str:
    return join_lines([l.text for l in lines])


def tesseract_rebuild(lines: List[TLine]) -> List[StructBlock]:
    """Layout failed: rebuild the page from positioned lines in reading order. Paragraphs come from Tesseract's own
    paragraph ids plus a vertical-gap test. No headings / tables are claimed: structure is best-effort."""
    blocks: List[StructBlock] = []
    cur: List[TLine] = []

    def flush():
        if not cur:
            return
        bb = (min(l.bbox[0] for l in cur), min(l.bbox[1] for l in cur), max(l.bbox[2] for l in cur), max(l.bbox[3] for l in cur))
        n = sum(l.n_words for l in cur) or 1
        blocks.append(StructBlock(len(blocks), "text", lines_to_text(cur), bb, None, "tesseract", len(cur),
                                  note=f"best-effort rebuild; tesseract mean conf {sum(l.conf * l.n_words for l in cur) / n:.0f}",
                                  line_boxes=[{"text": line.text, "bbox_px": list(line.bbox),
                                               "tesseract_conf": line.conf} for line in cur]))

    for l in lines:
        if cur:
            p = cur[-1]
            h = max(p.bbox[3] - p.bbox[1], 1)
            if l.col != p.col or l.para != p.para or l.bbox[1] - p.bbox[3] > 1.2 * h:
                flush()
                cur = []
        cur.append(l)
    flush()
    return blocks
