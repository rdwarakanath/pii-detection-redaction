"""Central configuration. Every threshold lives here so the pipeline can be tuned without code changes.

Layout of this file
  1. Primary engine  : PaddleOCR PP-StructureV3  (``paddle_*`` fields -> ``Config.paddle_settings()``)
     1b. Gate / fallback / retry thresholds (the quality gate that decides when the primary result is not good enough)
  2. Fallback engines: PaddleOCR-VL (lazy, verified rather than thresholded), then Tesseract as last resort
  3. Shared pipeline : rendering, page classification, page geometry, polarity
  4. Tesseract fallback tuning: layout and escalation ladders (used after VL fails the quality gate)
  5. Legacy fields   : kept so old call-sites / saved configs still load; not read by the PaddleOCR pipeline

The ``paddle_*`` / gate / ``vl_*`` fields mirror ``ocr_engine.PaddleSettings`` one-to-one (same defaults, same meaning).
``Config.paddle_settings()`` is the single place that converts one into the other, so the thresholds documented in
implementation_guidelines.md / mindmap.md are defined once, here.

Scales that must never be mixed (they are different quantities, see implementation_guidelines.md):
  * ``rec_conf_threshold``  : PaddleOCR text-RECOGNITION confidence, 0-1.      (layout-detection scores are never used)
  * ``accept_score`` / ``cell_accept`` : Tesseract's own 0-100 word-confidence scale, used only inside Tesseract ladders.
  * ``vl_*``                : PaddleOCR-VL has no per-token confidence; its text is accepted by verification checks.

Config validates itself on construction (``__post_init__``): out-of-range thresholds raise ``ValueError`` immediately instead of
failing mid-run, and risky-but-legal combinations raise a ``UserWarning``.
"""
from __future__ import annotations
import dataclasses
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:                                   # ocr_engine imports Config -> import PaddleSettings lazily
    from .ocr_engine import PaddleSettings

# Tesseract language code -> PaddleOCR language code. Explicit on purpose: Tesseract uses 3-letter codes whose first two letters
# are NOT reliably the PaddleOCR code (pol->"po", ces->"ce", tur->"tu", ind->"in", swe->"sw" are all wrong), so there is no
# prefix guessing. Codes not listed here must be given through ``paddle_lang`` (a PaddleOCR code such as "en", "german",
# "japan"); paddle_lang_code() raises instead of silently selecting a different language. Whether a given language is available
# also depends on the installed PaddleOCR / PP-StructureV3 models: the library rejects codes it does not support at start-up.
TESSERACT_TO_PADDLE_LANG = {
    # Chinese / Japanese / Korean
    "chi_sim": "ch", "chi_tra": "chinese_cht", "jpn": "japan", "kor": "korean",
    # Latin script
    "eng": "en", "fra": "fr", "deu": "german", "spa": "es", "ita": "it", "por": "pt", "nld": "nl", "swe": "sv",
    "dan": "da", "nor": "no", "fin": "fi", "isl": "is", "pol": "pl", "ces": "cs", "slk": "sk", "slv": "sl", "hrv": "hr",
    "bos": "bs", "hun": "hu", "ron": "ro", "tur": "tr", "vie": "vi", "ind": "id", "msa": "ms", "tgl": "tl", "swa": "sw",
    "afr": "af", "sqi": "sq", "est": "et", "lav": "lv", "lit": "lt", "mlt": "mt", "gle": "ga", "cym": "cy", "eus": "eu",
    "cat": "ca", "glg": "gl", "oci": "oc", "ltz": "lb", "roh": "rm", "aze": "az", "uzb": "uz", "kur": "ku", "lat": "la",
    "mri": "mi", "que": "qu",
    # Cyrillic
    "rus": "ru", "ukr": "uk", "bel": "be", "bul": "bg", "mon": "mn",
    # Arabic script
    "ara": "ar", "fas": "fa", "urd": "ur", "uig": "ug", "pus": "ps", "snd": "sd",
    # Devanagari
    "hin": "hi", "mar": "mr", "nep": "ne", "san": "sa", "mai": "mai",
    # Other scripts
    "tam": "ta", "tel": "te", "kan": "ka", "ell": "el", "tha": "th",
    # Tesseract script models
    "Latin": "latin", "Cyrillic": "cyrillic", "Arabic": "arabic", "Devanagari": "devanagari",
}
_PADDLE_LANG_CODES = frozenset(TESSERACT_TO_PADDLE_LANG.values())      # PaddleOCR codes may also be passed as `lang` directly

FALLBACK_ENGINES = ("tesseract", "vl")
ALT_ENGINES = (None, "easyocr", "rapidocr")

# The implementation includes whole-page and block-crop VL, validation, and Tesseract last-resort fallback.
VL_TIER_WIRED = True


@dataclass
class Config:
    # =====================================================================================================
    # 1. PRIMARY ENGINE - PaddleOCR PP-StructureV3 (PaddleOCR 3.x, paddlepaddle 3.x)
    #    Start from PP-StructureV3 defaults; turn optional modules on only when the document set needs them,
    #    because every module changes quality AND runtime (enabled modules are logged per page).
    # =====================================================================================================
    paddle_lang: Optional[str] = None        # None = derive from `lang` via TESSERACT_TO_PADDLE_LANG (eng -> en)
    paddle_device: Optional[str] = None      # None = PaddleOCR default | "cpu" | "gpu:0"
    paddle_enable_mkldnn: bool = False        # CPU inference acceleration. Kept False by default because of the oneDNN/PIR error
                                              # ("ConvertPirAttribute2RuntimeAttribute", seen with PaddlePaddle 3.3.1 on
                                              # Windows CPU). Irrelevant on GPU.
    paddle_use_doc_orientation_classify: bool = False   # off: pipeline already does Tesseract OSD + deskew. If you turn it ON,
                                                        # PP-StructureV3 rotates the image internally and block boxes are no
                                                        # longer in the pixel space of the raster we hold (see warnings)
    paddle_use_doc_unwarping: bool = False   # keep OFF: unwarping changes page geometry -> boxes no longer map to
                                             # the PDF page and box-overlap fallback becomes unreliable
    paddle_use_textline_orientation: bool = False
    paddle_use_seal_recognition: bool = False
    paddle_use_table_recognition: bool = True       # tables are part of the primary path
    paddle_use_formula_recognition: bool = False    # enable only if the documents contain formulas
    paddle_use_chart_recognition: bool = False      # enable only if the documents contain charts
    paddle_use_region_detection: bool = True        # region detection -> reading order for multi-column pages
    paddle_limit_side_len: int = 4000               # PP-StructureV3 text detector max side; retain the tested memory-safe default

    # ---- 1b. quality gate: text-RECOGNITION confidence + content checks (never layout-detection scores) ---------------
    rec_conf_threshold: float = 0.85         # < 0.85 on a page/block -> fallback for that page/region (Paddle 0-1 scale only)
    max_err_rate: float = 0.12               # junk + mixed-alphanumeric token share that fails a block
    max_overlap_iou: float = 0.6             # two text blocks overlapping more than this -> layout failed
    max_order_jump_frac: float = 0.25        # share of reading-order transitions that jump back up a column
    min_table_fill: float = 0.35             # tables with fewer non-empty cells than this are incoherent
    min_chars_on_inked_page: int = 20        # inked page with fewer alnum chars than this -> incomplete
    min_gate_chars: int = 3                  # blocks with fewer OCR characters (page numbers) skip the rec-confidence check
    max_uncovered_ink: float = 0.15          # share of ink outside every block above this -> layout incomplete (needs the
                                             # pipeline to pass the raster to quality_gate; 1/3 of it logs a warning)
    min_space_ratio: float = 0.09            # (words-1)/alnum below this in Latin prose -> words ran together (English ~0.18)
    min_spacing_chars: int = 25              # only blocks with at least this many alphanumerics get the spacing check
    abort_after_errors: int = 3              # consecutive engine exceptions before EngineUnavailableError (0 = never abort);
                                             # stops a broken install silently degrading every page to the fallback

    # ---- fallback / retry --------------------------------------------------------------------------------
    fallback_enabled: bool = True            # False = PP-StructureV3 only (no fallback engine); failures are only flagged
    fallback_engine: str = "vl"               # PP-StructureV3 -> lazy PaddleOCR-VL -> Tesseract last resort
    match_min_frac: float = 0.5              # share of a Tesseract line that must lie inside a Paddle block
    retry_dpis: Tuple[int, ...] = ()          # disabled by default: retries add a full inference pass and GPU memory pressure
                                             # Note: Paddle resizes inputs internally, so extra DPI helps less than it does
                                             # for Tesseract; measure before adding more entries (each costs a full pass)
    retry_variant: str = "norm"              # targeted preprocessing for retries (preprocess.make_variant)
    min_native_cover: float = 0.7            # hybrid pages: native text replaces OCR text above this coverage
    markdown_ignore_labels: Tuple[str, ...] = ()    # block labels dropped from Markdown; () keeps header/footer/number
    word_boxes: bool = True                  # emit word-level boxes (wordbox.py) for PII redaction: Tesseract locates words inside
                                             # every OCR block, the block's own text is aligned to them; native pages use the exact
                                             # PDF word boxes. Costs one extra Tesseract pass per OCR block. False = lines only.

    # =====================================================================================================
    # 2. FALLBACK ENGINES
    #    2a. PaddleOCR-VL (vision-language model). GPU only (no CPU / Arm support). It returns no per-token confidence, so
    #        its text is VERIFIED (ocr_engine.verify_vl_text / verify_vl_page) instead of compared with a threshold.
    #    2b. Tesseract (conditional) + optional last-resort alternative engine
    # =====================================================================================================
    # ---- 2a. PaddleOCR-VL --------------------------------------------------------------------------------
    vl_device: Optional[str] = None          # None = same as paddle_device. Must be a GPU, e.g. "gpu:0"
    vl_pipeline_version: Optional[str] = None   # None = library default; set (e.g. "v1.5") only if your paddleocr accepts it
    vl_backend: Optional[str] = None         # None = native Paddle inference; e.g. "vllm-server" to offload to a server
    vl_server_url: Optional[str] = None      # used with a *-server backend
    vl_use_chart_recognition: bool = False   # slower; enable only if the documents contain charts
    vl_use_ocr_for_image_block: bool = False # also OCR text inside image blocks (logos): more content, more noise
    vl_crop_margin_px: int = 12              # context added around a failing block before region recognition
    vl_min_len_ratio: float = 0.5            # VL alnum / failed-candidate alnum must lie in [min, max]; outside it VL is
    vl_max_len_ratio: float = 2.0            # treated as truncated or hallucinated and the Paddle text is kept
    vl_min_agreement: float = 0.35           # VL vs trusted reference similarity below this -> VL rejected
    vl_loop_repeats: int = 7                 # same phrase repeated this often back-to-back -> degenerate VL output

    # ---- 2b. Tesseract -----------------------------------------------------------------------------------
    lang: str = "eng"                        # TESSERACT language code (also feeds paddle_lang when that is None, through
                                             # TESSERACT_TO_PADDLE_LANG). "eng+fra" goes to Tesseract as is; PaddleOCR uses
                                             # the first component. Unmapped codes need an explicit paddle_lang
    tesseract_cmd: Optional[str] = None      # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    alt_engine: Optional[str] = None         # None | "easyocr" | "rapidocr"  (extra Tesseract-ladder attempt only;
                                             # PaddleOCR is now the PRIMARY engine, so "paddleocr" is not valid here)
    workers: int = 0                         # 0 -> 1 process (PP-StructureV3 loads several models and is itself
                                             # multi-threaded); >1 loads one model set per process - needs the RAM/GPU
                                             # (with the VL tier, one VL model per process as well)
    debug_dir: Optional[str] = None          # optional directory for page overlay images

    # =====================================================================================================
    # 3. SHARED PIPELINE - rendering, page classification, geometry, polarity
    # =====================================================================================================
    # ---- rendering / resolution -------------------------------------------
    base_dpi: int = 300                      # OCR-required content is rendered at 300 DPI first (see mindmap)
    target_cap_px: int = 30                  # desired capital-letter height fed to Tesseract (fallback only)
    scale_min: float = 0.6
    scale_max: float = 3.0

    # ---- page classification (digital / scanned / hybrid / vector / blank) --
    min_native_words: int = 8                # fewer visible native words -> treat as having no text layer
    garbage_max: float = 0.08                # >8% (cid:)/U+FFFD/private-use chars -> native layer untrusted
    hybrid_image_cov: float = 0.03           # image area fraction that makes a digital page "hybrid"
    drawing_heavy: int = 400                 # many vector paths + little text => maybe outlined text
    trust_invisible_text: bool = False       # False: ignore existing OCR layers (render mode 3) and re-OCR
    min_ink_frac: float = 0.0004             # residual raster with less ink than this is treated as blank

    # ---- page geometry (applied to the raster sent to ALL engines) ---------
    detect_orientation: bool = True          # Tesseract OSD (needs osd.traineddata); silently skipped if missing
    deskew: bool = True
    deskew_min_deg: float = 0.3
    deskew_max_deg: float = 6.0

    # ---- polarity (white-on-dark blocks, coloured pills, dark screenshots) -
    fix_polarity: bool = True
    dark_fill_thresh: int = 170
    dark_min_w: int = 30
    dark_min_h: int = 14
    dark_min_fill: float = 0.50

    # =====================================================================================================
    # 4. TESSERACT-FALLBACK TUNING (read only when the fallback runs)
    # =====================================================================================================
    # ---- column / table geometry used to order Tesseract lines -------------
    table_min_lines: int = 3
    table_min_width_frac: float = 0.25
    table_row_gap_in: float = 0.45           # max vertical gap (inches) between rules of one table
    split_columns: bool = True

    # ---- Tesseract escalation ladder (own 0-100 scale; never compared with Paddle confidence) --------------
    accept_score: float = 82.0               # stop escalating when region score >= this
    cell_accept: float = 78.0

    # =====================================================================================================
    # 5. LEGACY (pre-PaddleOCR pipeline). Not read by pipeline.py / ocr_engine.py; kept so existing
    #    Config(...) calls and saved settings keep loading. Safe to delete together with tables.py cell OCR.
    # =====================================================================================================
    detect_tables: bool = True
    borderless_tables: bool = True
    refine_conf: float = 55.0
    refine_min_words: int = 2
    max_refine_regions: int = 14
    refine_gain: float = 4.0
    drop_junk_words: bool = True
    postprocess: bool = True
    digit_fix: bool = True

    # -----------------------------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Normalise values that come back as lists from JSON / YAML / CLI so downstream hashing (frozen PaddleSettings) works.
        self.retry_dpis = tuple(int(d) for d in (self.retry_dpis or ()))
        self.markdown_ignore_labels = tuple(self.markdown_ignore_labels or ())
        problems = self.validate()
        if problems:
            raise ValueError("Invalid Config: " + "; ".join(problems))
        for msg in self.risk_warnings():
            warnings.warn(msg, UserWarning, stacklevel=3)

    # ------------------------------------------------------------------ validation
    def validate(self) -> List[str]:
        """Return a list of problems (empty = valid). Called on construction; callable again after mutating fields."""
        bad: List[str] = []

        def unit(*names: str) -> None:
            for n in names:
                v = getattr(self, n)
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0.0 <= float(v) <= 1.0:
                    bad.append(f"{n}={v!r} must be within 0..1")

        def at_least(lo: float, *names: str) -> None:
            for n in names:
                v = getattr(self, n)
                if not isinstance(v, (int, float)) or isinstance(v, bool) or v < lo:
                    bad.append(f"{n}={v!r} must be >= {lo}")

        unit("rec_conf_threshold", "max_err_rate", "max_overlap_iou", "max_order_jump_frac", "min_table_fill",
             "match_min_frac", "min_native_cover", "max_uncovered_ink", "vl_min_agreement", "garbage_max",
             "hybrid_image_cov", "dark_min_fill", "min_ink_frac")
        if not (isinstance(self.min_space_ratio, (int, float)) and 0.0 < self.min_space_ratio <= 1.0):
            bad.append(f"min_space_ratio={self.min_space_ratio!r} must be within (0, 1]")
        at_least(0, "min_chars_on_inked_page", "min_gate_chars", "min_spacing_chars", "abort_after_errors",
                 "vl_crop_margin_px", "workers", "min_native_words", "drawing_heavy", "table_min_lines")
        at_least(2, "vl_loop_repeats")
        if not (isinstance(self.vl_min_len_ratio, (int, float)) and isinstance(self.vl_max_len_ratio, (int, float))
                and 0 < self.vl_min_len_ratio <= self.vl_max_len_ratio):
            bad.append(f"vl_min_len_ratio={self.vl_min_len_ratio!r} / vl_max_len_ratio={self.vl_max_len_ratio!r} "
                       "must satisfy 0 < min <= max")
        for n in ("base_dpi",):
            v = getattr(self, n)
            if not isinstance(v, int) or not 72 <= v <= 1200:
                bad.append(f"{n}={v!r} must be an integer DPI within 72..1200")
        for d in self.retry_dpis:
            if not 72 <= d <= 1200:
                bad.append(f"retry_dpis entry {d} must be within 72..1200")
        if self.scale_min > self.scale_max:
            bad.append(f"scale_min={self.scale_min} must be <= scale_max={self.scale_max}")
        if self.deskew_min_deg > self.deskew_max_deg:
            bad.append(f"deskew_min_deg={self.deskew_min_deg} must be <= deskew_max_deg={self.deskew_max_deg}")
        if not isinstance(self.lang, str) or not self.lang.strip():
            bad.append("lang must be a non-empty Tesseract language code")
        if self.fallback_engine not in FALLBACK_ENGINES:
            bad.append(f"fallback_engine={self.fallback_engine!r} must be one of {FALLBACK_ENGINES}")
        if self.alt_engine not in ALT_ENGINES:
            bad.append(f"alt_engine={self.alt_engine!r} must be one of {ALT_ENGINES} ('paddleocr' is the primary engine)")
        try:
            from .preprocess import VARIANTS                 # fail now, not on the first retry after a full Paddle pass
            if self.retry_variant not in VARIANTS:
                bad.append(f"retry_variant={self.retry_variant!r} must be one of {VARIANTS}")
        except ImportError:                                  # preprocess needs OpenCV; skip the check rather than block import
            pass
        return bad

    def risk_warnings(self) -> List[str]:
        """Legal but risky combinations (emitted once as UserWarning at construction)."""
        w: List[str] = []
        if self.paddle_use_doc_unwarping:
            w.append("paddle_use_doc_unwarping=True changes page geometry: block boxes will not map back to the PDF page and "
                     "box-overlap fallback / coordinates become unreliable")
        if self.paddle_use_doc_orientation_classify:
            w.append("paddle_use_doc_orientation_classify=True rotates the image inside PP-StructureV3, so block boxes are no "
                     "longer in the pixel space of the raster the pipeline holds (the pipeline already orients via OSD)")
        if self.fallback_enabled and self.fallback_engine == "vl":
            if not VL_TIER_WIRED:
                w.append("fallback_engine='vl' is selected but pipeline.py does not call the PaddleOCR-VL tier yet "
                         "(config.VL_TIER_WIRED is False): the Tesseract fallback will still run")
            dev = (self.vl_device or self.paddle_device or "").lower()
            if dev.startswith("cpu"):
                w.append(f"PaddleOCR-VL needs a GPU but the effective VL device is {dev!r}")
            if self.workers > 1:
                w.append("workers>1 loads one PP-StructureV3 AND one PaddleOCR-VL model set per process: check GPU memory")
        return w

    # ------------------------------------------------------------------ derived helpers
    def use_vl_fallback(self) -> bool:
        """True when the PaddleOCR-VL tier is the selected fallback (and a fallback is enabled at all)."""
        return self.fallback_enabled and self.fallback_engine == "vl"

    def use_tesseract_fallback(self) -> bool:
        """True when Tesseract is the selected fallback (and a fallback is enabled at all)."""
        return self.fallback_enabled and self.fallback_engine == "tesseract"

    def needs_tesseract(self) -> bool:
        """Tesseract is the final OCR tier whenever fallback is enabled."""
        return self.fallback_enabled

    def paddle_lang_code(self) -> str:
        """PaddleOCR language: explicit `paddle_lang`, else the mapped Tesseract `lang`.
        Multi-language ("eng+fra") and vertical ("jpn_vert") Tesseract codes use their first / base component; a PaddleOCR code
        passed as `lang` ("en", "german") is accepted as is. Anything else raises ValueError (no prefix guessing)."""
        if self.paddle_lang:
            return self.paddle_lang
        base = self.lang.split("+")[0].strip()
        if base.endswith("_vert"):
            base = base[:-len("_vert")]
        if base in TESSERACT_TO_PADDLE_LANG:
            return TESSERACT_TO_PADDLE_LANG[base]
        if base in _PADDLE_LANG_CODES:
            return base
        raise ValueError(f"No PaddleOCR language known for Tesseract code {base!r}: set Config.paddle_lang explicitly "
                         "(a PaddleOCR code such as 'en', 'german', 'japan') or extend TESSERACT_TO_PADDLE_LANG")

    def paddle_settings(self) -> "PaddleSettings":
        """Build the (frozen, hashable) PaddleSettings the engines are configured from.
        PP-StructureV3 / PaddleOCR-VL instances are cached by the *model options* inside it (engine_key / vl_key), so gate
        thresholds can change without reloading models."""
        from .ocr_engine import PaddleSettings
        return PaddleSettings(
            lang=self.paddle_lang_code(),
            device=self.paddle_device,
            enable_mkldnn=self.paddle_enable_mkldnn,
            use_doc_orientation_classify=self.paddle_use_doc_orientation_classify,
            use_doc_unwarping=self.paddle_use_doc_unwarping,
            use_textline_orientation=self.paddle_use_textline_orientation,
            use_seal_recognition=self.paddle_use_seal_recognition,
            use_table_recognition=self.paddle_use_table_recognition,
            use_formula_recognition=self.paddle_use_formula_recognition,
            use_chart_recognition=self.paddle_use_chart_recognition,
            use_region_detection=self.paddle_use_region_detection,
            limit_side_len=self.paddle_limit_side_len,
            rec_conf_threshold=self.rec_conf_threshold,
            max_err_rate=self.max_err_rate,
            max_overlap_iou=self.max_overlap_iou,
            max_order_jump_frac=self.max_order_jump_frac,
            min_table_fill=self.min_table_fill,
            min_chars_on_inked_page=self.min_chars_on_inked_page,
            min_gate_chars=self.min_gate_chars,
            max_uncovered_ink=self.max_uncovered_ink,
            min_space_ratio=self.min_space_ratio,
            min_spacing_chars=self.min_spacing_chars,
            abort_after_errors=self.abort_after_errors,
            fallback_enabled=self.fallback_enabled,
            match_min_frac=self.match_min_frac,
            retry_dpis=tuple(self.retry_dpis),
            retry_variant=self.retry_variant,
            min_native_cover=self.min_native_cover,
            markdown_ignore_labels=tuple(self.markdown_ignore_labels),
            vl_pipeline_version=self.vl_pipeline_version,
            vl_backend=self.vl_backend,
            vl_server_url=self.vl_server_url,
            vl_device=self.vl_device,
            vl_use_chart_recognition=self.vl_use_chart_recognition,
            vl_use_ocr_for_image_block=self.vl_use_ocr_for_image_block,
            vl_crop_margin_px=self.vl_crop_margin_px,
            vl_min_len_ratio=self.vl_min_len_ratio,
            vl_max_len_ratio=self.vl_max_len_ratio,
            vl_min_agreement=self.vl_min_agreement,
            vl_loop_repeats=self.vl_loop_repeats,
        )

    # ------------------------------------------------------------------ (de)serialisation of saved settings
    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly snapshot (tuples become lists; Config() / from_dict() turn them back)."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Rebuild from a saved snapshot. Unknown keys (from a newer/older version) are ignored with a warning, missing keys
        take their defaults, so saved settings keep loading."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            warnings.warn(f"Config.from_dict ignored unknown key(s): {unknown}", UserWarning, stacklevel=2)
        return cls(**{k: v for k, v in data.items() if k in known})
