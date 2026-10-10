"""Command line: PP-StructureV3 primary, lazy PaddleOCR-VL fallback, Tesseract final fallback.

Markdown is the only deliverable; per-page blocks / attempts go to ``<out>/_internal/`` for QA.

What the CLI adds on top of ``process_pdf``
  * every Config option that matters (fallback engine, PaddleOCR-VL, gate tuning, abort threshold) is a flag; bad values are
    reported as usage errors (exit 2) instead of tracebacks;
  * pages are processed one at a time (``--workers`` <= 1), each result is checkpointed, and a page that crashes is recorded
    as an error page instead of killing the document; ``--resume`` continues an interrupted run. Checkpoints are tied to the
    PDF, the effective settings, the library versions and the pipeline code, so stale ones are never reused;
  * an engine that keeps failing (``EngineUnavailableError``) stops the run with a clear message, and the pages finished so far
    are written to ``<stem>.partial.md`` rather than lost;
  * a start-up check for Tesseract / paddleocr, per-page progress, and a report that lists the pages needing attention.

Exit codes:  0 success | 1 unexpected error | 2 usage / configuration error | 3 run aborted (engine unavailable or repeated page
failures) | 4 finished, but some pages need review (only with --strict) | 130 interrupted
"""
from __future__ import annotations
import argparse
import dataclasses
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import textwrap
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pymupdf

# Native Paddle/CUDA crashes may bypass Python exception handling. Keep a Python stack trace enabled
# so Windows access violations leave actionable diagnostics in the terminal/run log.
faulthandler.enable(all_threads=True)

from .config import ALT_ENGINES, FALLBACK_ENGINES, Config
from .ocr_engine import EngineUnavailableError, release_vl_engine
from .pipeline import PageMarkdown, process_pdf, write_outputs

log = logging.getLogger(__name__)

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_ABORTED, EXIT_REVIEW, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130


# ====================================================================== diagnostics
_DISTS = ("paddlepaddle", "paddlepaddle-gpu", "paddleocr", "PyMuPDF", "opencv-python-headless", "numpy", "pytesseract")


def _dist_versions() -> Dict[str, str]:
    out = {}
    for d in _DISTS:
        try:
            out[d] = importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError:
            out[d] = "not installed"
    return out


class _VersionsAction(argparse.Action):
    """--versions: print the library versions that decide whether a run works (see requirements.txt) and exit."""

    def __init__(self, option_strings, dest, **kw):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, help=kw.get("help"))

    def __call__(self, parser, namespace, values, option_string=None):
        lines = [f"{k:24} {v}" for k, v in _dist_versions().items()]
        try:
            import pytesseract
            lines.append(f"{'tesseract (binary)':24} {pytesseract.get_tesseract_version()}")
        except Exception:
            lines.append(f"{'tesseract (binary)':24} not found")
        print("\n".join(lines))
        parser.exit(EXIT_OK)


# ====================================================================== arguments
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="PDF -> Markdown. PP-StructureV3 is primary, PaddleOCR-VL is loaded lazily on quality failure, and "
                    "Tesseract is the last-resort OCR tier when VL fails or is rejected. "
                    "Fallback runs when text-recognition confidence < threshold or the "
                    "quality checks fail. Digital pages with a trusted text layer are never OCR'd.",
        epilog="exit codes: 0 ok | 1 error | 2 usage/config error | 3 aborted (partial output kept; see --resume) | "
               "4 pages need review (with --strict) | 130 interrupted")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="ocr_out")
    ap.add_argument("--pages", help="1-based pages, e.g. 1-5,9")
    ap.add_argument("--dpi", type=int, default=300, help="base render DPI (default 300; valid 72-1200)")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = 1 process (default). >1 creates per-process OCR model instances and disables per-page "
                         "checkpoints / error isolation (the batch is one unit)")
    run = ap.add_argument_group("run control")
    run.add_argument("--resume", action="store_true",
                     help="reuse pages already finished by an earlier run (same PDF, settings, library versions and code); "
                          "anything else is recomputed")
    run.add_argument("--no-checkpoints", action="store_true", help="do not write per-page checkpoints (disables --resume)")
    run.add_argument("--strict", action="store_true", help="exit 4 if any page still needs review after fallback")
    run.add_argument("-q", "--quiet", action="store_true", help="no progress lines or report table (warnings and errors still shown)")
    run.add_argument("-v", "--verbose", action="store_true", help="debug logging (library and per-page diagnostics)")
    run.add_argument("--versions", action=_VersionsAction, help="print library / Tesseract versions and exit")
    run.add_argument("--debug", action="store_true", help="write overlay PNGs to <out>/debug")

    lang = ap.add_argument_group("language")
    lang.add_argument("--lang", default="eng",
                      help="Tesseract language code (default eng; 'eng+fra' allowed). The PaddleOCR language is looked up from "
                           "it; codes with no mapping need --paddle-lang")
    lang.add_argument("--paddle-lang", help="PaddleOCR language code, overrides the lookup (e.g. en, german, japan, ch)")

    prim = ap.add_argument_group("primary engine: PP-StructureV3")
    prim.add_argument("--device", help="PaddleOCR device: cpu | gpu:0 (default: PaddleOCR decides)")
    mkldnn = prim.add_mutually_exclusive_group()
    mkldnn.add_argument("--enable-mkldnn", dest="enable_mkldnn", action="store_true",
                        help="enable oneDNN/MKLDNN for Paddle CPU inference (off by default)")
    mkldnn.add_argument("--no-mkldnn", dest="enable_mkldnn", action="store_false",
                        help="explicitly disable oneDNN/MKLDNN (already the default)")
    prim.set_defaults(enable_mkldnn=False)
    prim.add_argument("--no-tables", action="store_true", help="disable PP-StructureV3 table recognition")
    prim.add_argument("--formula", action="store_true", help="enable formula recognition (slower)")
    prim.add_argument("--chart", action="store_true", help="enable chart recognition (slower)")
    prim.add_argument("--seal", action="store_true", help="enable seal recognition")
    prim.add_argument("--textline-orientation", action="store_true", help="enable text-line orientation classifier")

    gate = ap.add_argument_group("quality gate (when the primary result is not good enough)")
    gate.add_argument("--rec-threshold", type=float,
                      help="text-recognition confidence (0-1) below which the fallback runs (default 0.85)")
    gate.add_argument("--max-uncovered-ink", type=float,
                      help="share (0-1) of page ink outside every detected block above which the layout counts as incomplete "
                           "(default 0.15). Takes effect once pipeline.py passes the page raster to the quality gate")
    gate.add_argument("--min-space-ratio", type=float,
                      help="(words-1)/characters below which Latin text counts as having run-together words (default 0.09)")
    gate.add_argument("--abort-after-errors", type=int,
                      help="stop the run after this many consecutive engine failures / identical page errors; 0 = never "
                           "(default 3)")

    fb = ap.add_argument_group("fallback")
    fb_off = fb.add_mutually_exclusive_group()
    fb_off.add_argument("--no-fallback", action="store_true",
                        help="PP-StructureV3 only: never run a fallback engine; weak output is flagged")
    fb_off.add_argument("--fallback-engine", choices=FALLBACK_ENGINES,
                        help="fallback policy: vl (default) runs VL then Tesseract last; tesseract skips VL")
    retry = fb.add_mutually_exclusive_group()
    retry.add_argument("--no-retry", action="store_true", help="disable the higher-DPI retry for weak pages")
    retry.add_argument("--retry-dpi", type=int, nargs="+",
                       help="DPI(s) used for additional full-page retry passes (disabled by default)")
    fb.add_argument("--tesseract", help="path to tesseract executable")
    fb.add_argument("--alt", choices=[e for e in ALT_ENGINES if e],
                    help="optional extra attempt inside the Tesseract fallback ladders")

    vl = ap.add_argument_group("PaddleOCR-VL fallback (used with --fallback-engine vl)")
    vl.add_argument("--vl-device", help="GPU for PaddleOCR-VL, e.g. gpu:0 (default: same as --device); it has no CPU support")
    vl.add_argument("--vl-backend", help="VL inference backend, e.g. vllm-server (default: native Paddle)")
    vl.add_argument("--vl-server-url", help="server URL for a *-server backend")
    vl.add_argument("--vl-version", help="PaddleOCR-VL pipeline version, only if your paddleocr accepts it (e.g. v1.5)")
    vl.add_argument("--vl-chart", action="store_true", help="enable chart recognition in PaddleOCR-VL (slower)")
    vl.add_argument("--vl-image-ocr", action="store_true",
                    help="also OCR text inside image blocks (logos): more content, more noise")
    vl.add_argument("--vl-min-agreement", type=float,
                    help="similarity (0-1) with a trusted reference below which VL output is rejected (default 0.35)")
    return ap


def config_from_args(a: argparse.Namespace) -> Config:
    """Namespace -> Config. Options left unset keep the Config defaults. May raise ValueError (invalid values)."""
    kw = dict(lang=a.lang, tesseract_cmd=a.tesseract, alt_engine=a.alt, base_dpi=a.dpi, workers=a.workers,
              debug_dir=os.path.join(a.out, "debug") if a.debug else None,
              paddle_enable_mkldnn=a.enable_mkldnn,
              paddle_use_table_recognition=not a.no_tables,
              paddle_use_formula_recognition=a.formula,
              paddle_use_chart_recognition=a.chart,
              paddle_use_seal_recognition=a.seal,
              paddle_use_textline_orientation=a.textline_orientation,
              fallback_enabled=not a.no_fallback,
              vl_use_chart_recognition=a.vl_chart,
              vl_use_ocr_for_image_block=a.vl_image_ocr)
    optional = {"paddle_lang": a.paddle_lang, "paddle_device": a.device, "rec_conf_threshold": a.rec_threshold,
                "max_uncovered_ink": a.max_uncovered_ink, "min_space_ratio": a.min_space_ratio,
                "abort_after_errors": a.abort_after_errors, "fallback_engine": a.fallback_engine,
                "vl_device": a.vl_device, "vl_backend": a.vl_backend, "vl_server_url": a.vl_server_url,
                "vl_pipeline_version": a.vl_version, "vl_min_agreement": a.vl_min_agreement}
    kw.update({k: v for k, v in optional.items() if v is not None})
    if a.no_retry:
        kw["retry_dpis"] = ()
    elif a.retry_dpi:
        kw["retry_dpis"] = tuple(a.retry_dpi)
    return Config(**kw)


# ====================================================================== page selection
_SPEC = re.compile(r"^\s*\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*\s*$")


def parse_page_spec(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """'1-5,9' -> [(1, 5), (9, 9)] (1-based, inclusive). None -> None (all pages). ValueError on bad syntax or reversed ranges."""
    if spec is None:
        return None
    if not _SPEC.match(spec):
        raise ValueError(f"--pages {spec!r}: expected numbers and ranges like 1-5,9")
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b.strip() else int(a)
        if lo < 1 or hi < lo:
            raise ValueError(f"--pages: invalid range {part.strip()!r} (pages start at 1 and ranges must ascend)")
        out.append((lo, hi))
    return out


def resolve_pages(ranges: Optional[List[Tuple[int, int]]], n: int) -> Tuple[List[int], int]:
    """Ranges -> ascending, de-duplicated 1-based pages that exist, plus the highest page requested beyond the document.
    Pages are always processed and written in page order (the order given in --pages does not matter)."""
    if ranges is None:
        return list(range(1, n + 1)), 0
    pages, beyond = set(), 0
    for lo, hi in ranges:
        beyond = max(beyond, hi if hi > n else 0)
        pages.update(range(lo, min(hi, n) + 1))
    return sorted(pages), beyond


def _compress(pages: Sequence[int]) -> str:
    """[1,2,3,5,9,10] -> '1-3,5,9-10' (the form process_pdf's `pages` argument takes)."""
    out, i = [], 0
    ps = list(pages)
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[j] + 1:
            j += 1
        out.append(str(ps[i]) if i == j else f"{ps[i]}-{ps[j]}")
        i = j + 1
    return ",".join(out)


# ====================================================================== checkpoints
def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_hash() -> str:
    """Hash of the modules that decide page results: a code change invalidates old checkpoints (cli.py itself does not)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for name in ("pipeline.py", "ocr_engine.py", "config.py", "preprocess.py", "textlayer.py", "layout.py", "tables.py",
                 "postprocess.py", "models.py"):
        p = os.path.join(here, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                h.update(name.encode() + f.read())
    return h.hexdigest()


def fingerprint(pdf: str, cfg: Config) -> str:
    """Identity of everything a page result depends on. Run-control settings (workers, debug_dir, tesseract path) are excluded."""
    d = cfg.to_dict()
    for k in ("workers", "debug_dir", "tesseract_cmd"):
        d.pop(k, None)
    h = hashlib.sha256()
    for part in (_file_sha256(pdf), json.dumps(d, sort_keys=True, default=str), json.dumps(_dist_versions(), sort_keys=True),
                 _code_hash()):
        h.update(part.encode())
    return h.hexdigest()


class Checkpoints:
    """Per-page JSON results under <out>/_internal/checkpoints/<stem>/, guarded by a fingerprint manifest."""

    def __init__(self, out_dir: str, stem: str, fp: str, enabled: bool):
        self.root = os.path.join(out_dir, "_internal", "checkpoints", stem)
        self.fp, self.enabled = fp, enabled

    def _page_file(self, page_no: int) -> str:
        return os.path.join(self.root, f"page_{page_no:04d}.json")

    @staticmethod
    def _atomic_write(path: str, text: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    def prepare(self, resume: bool, warn) -> Dict[int, PageMarkdown]:
        """Reusable pages (resume only). Without --resume, or when the fingerprint differs, old checkpoints are discarded."""
        if not self.enabled:
            return {}
        manifest = os.path.join(self.root, "manifest.json")
        loaded: Dict[int, PageMarkdown] = {}
        if os.path.isdir(self.root):
            same = False
            try:
                with open(manifest, encoding="utf-8") as f:
                    same = json.load(f).get("fingerprint") == self.fp
            except (OSError, ValueError):
                pass
            if resume and same:
                for name in sorted(os.listdir(self.root)):
                    m = re.fullmatch(r"page_(\d{4})\.json", name)
                    if not m:
                        continue
                    try:
                        with open(os.path.join(self.root, name), encoding="utf-8") as f:
                            pm = PageMarkdown(**json.load(f))
                    except (OSError, ValueError, TypeError):
                        continue                                          # unreadable / from another layout: recompute
                    if pm.page_no == int(m.group(1)) and pm.source != "error":
                        loaded[pm.page_no] = pm
            else:
                if resume and not same:
                    warn("checkpoints belong to a different PDF, settings, library versions or code: discarded, "
                         "starting from scratch")
                shutil.rmtree(self.root, ignore_errors=True)
        elif resume:
            warn("--resume: no checkpoints found for this document; starting from scratch")
        try:
            os.makedirs(self.root, exist_ok=True)
            self._atomic_write(manifest, json.dumps({"fingerprint": self.fp, "created": time.strftime("%Y-%m-%d %H:%M:%S")}))
        except OSError as e:
            warn(f"cannot write checkpoints ({e}); continuing without")
            self.enabled = False
        return loaded

    def save(self, pm: PageMarkdown, warn) -> None:
        if not self.enabled or pm.source == "error":
            return                                                        # error pages are retried on --resume
        try:
            self._atomic_write(self._page_file(pm.page_no),
                               json.dumps(dataclasses.asdict(pm), ensure_ascii=False, default=str))
        except OSError as e:
            warn(f"cannot write checkpoint for page {pm.page_no} ({e}); further checkpoints disabled")
            self.enabled = False


# ====================================================================== run
@dataclass
class RunOutcome:
    results: List[PageMarkdown] = field(default_factory=list)
    resumed: List[int] = field(default_factory=list)
    aborted: str = ""                  # reason the run stopped early
    interrupted: bool = False


def needs_review(pm: PageMarkdown) -> bool:
    return bool(pm.stats.get("needs_review") or pm.stats.get("needs_vlm") or pm.source == "error")


def _error_page(page_no: int, why: str) -> PageMarkdown:
    return PageMarkdown(page_no, "unknown", "", "error", "error", 0.0, [f"page failed: {why}"], [],
                        {"needs_review": True, "error": why, "n_ocr_words": 0, "n_low_conf": 0, "seconds": 0})


def run_document(pdf: str, cfg: Config, pages: List[int], ckpt: Checkpoints, resume: bool, say, warn) -> RunOutcome:
    """Process `pages` (1-based). One page at a time unless cfg.workers > 1."""
    out = RunOutcome()
    done = ckpt.prepare(resume, warn)
    results: Dict[int, PageMarkdown] = {p: done[p] for p in pages if p in done}
    out.resumed = sorted(results)
    pending = [p for p in pages if p not in results]
    if out.resumed:
        say(f"resuming: {len(out.resumed)} page(s) reused, {len(pending)} to process")

    def finish():
        out.results = [results[p] for p in sorted(results)]
        return out

    if not pending:
        return finish()

    if cfg.workers > 1:                                   # batch: one model set per process, no per-page isolation
        try:
            for pm in process_pdf(pdf, cfg, pages=_compress(pending)):
                results[pm.page_no] = pm
                ckpt.save(pm, warn)
        except EngineUnavailableError as e:
            out.aborted = str(e)
        except KeyboardInterrupt:
            out.interrupted = True
        except Exception as e:
            log.debug("batch failed", exc_info=True)
            out.aborted = (f"{type(e).__name__}: {e}. A worker failed and the batch was lost; re-run with --workers 1 to "
                           "isolate the page and keep per-page checkpoints")
        return finish()

    last_sig, repeats = None, 0
    for k, pno in enumerate(pending, 1):
        try:
            pm = process_pdf(pdf, cfg, pages=str(pno))[0]
        except EngineUnavailableError as e:
            out.aborted = str(e)
            break
        except KeyboardInterrupt:
            out.interrupted = True
            break
        except Exception as e:                            # one bad page must not lose the document
            log.debug("page %d failed", pno, exc_info=True)
            sig = f"{type(e).__name__}: {str(e)[:200]}"
            repeats = repeats + 1 if sig == last_sig else 1
            last_sig = sig
            results[pno] = pm = _error_page(pno, sig)
            warn(f"page {pno} failed ({sig}); recorded as an error page")
            if cfg.abort_after_errors and repeats >= cfg.abort_after_errors:
                out.aborted = f"{repeats} pages in a row failed with the same error: {sig}"
                break
        else:
            last_sig, repeats = None, 0
            results[pno] = pm
            ckpt.save(pm, warn)
        rec = f"{pm.score:5.1f}%" if pm.score is not None and pm.source.startswith("paddle") else "    - "
        say(f"[{k:>{len(str(len(pending)))}}/{len(pending)}] page {pno:>3}  {pm.page_type:8} {pm.source:18} {rec} "
            f"{pm.stats.get('seconds', 0):>6}s" + ("  NEEDS REVIEW" if needs_review(pm) else ""))
    return finish()


# ====================================================================== start-up checks
def preflight(cfg: Config) -> List[str]:
    """Problems that would otherwise surface late and quietly (a missing Tesseract only logs inside the fallback)."""
    msgs: List[str] = []
    if importlib.util.find_spec("paddleocr") is None:
        msgs.append("paddleocr is not installed: only digital pages with a trusted text layer can be processed")
    if cfg.needs_tesseract() or cfg.detect_orientation:
        try:
            import pytesseract
            if cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
            pytesseract.get_tesseract_version()
            langs = set(pytesseract.get_languages(config=""))
            missing = [l for l in cfg.lang.split("+") if l not in langs]
            if missing:
                msgs.append(f"Tesseract language data missing for {missing}: Tesseract fallback / verification will fail")
            if cfg.detect_orientation and "osd" not in langs:
                msgs.append("Tesseract osd.traineddata not found: page-orientation detection will be skipped")
        except Exception:
            what = "the Tesseract fallback and verification" if cfg.needs_tesseract() else "orientation detection"
            msgs.append(f"Tesseract was not found (--tesseract PATH to point at it): {what} will be unavailable")
    return msgs


# ====================================================================== report
def _fmt_row(pm: PageMarkdown, resumed: bool, width: int) -> str:
    rec = f"{pm.score:.1f}" if pm.score is not None and pm.source.startswith("paddle") else "-"
    head = (f"{pm.page_no:>3} {pm.page_type:8} {pm.source:18} {rec:>6} {pm.stats.get('n_ocr_words', 0):>6} "
            f"{pm.stats.get('n_low_conf', 0):>4} {len(pm.tables):>3} {pm.stats.get('seconds', 0):>6}  ")
    note = ("[resumed] " if resumed else "") + "; ".join(pm.warnings)
    room = max(width - len(head), 20)
    return head + (note if len(note) <= room else note[:room - 1] + "\u2026")


def print_report(results: List[PageMarkdown], resumed: Sequence[int], elapsed: float) -> None:
    width = shutil.get_terminal_size((120, 20)).columns
    print(f"{'pg':>3} {'type':8} {'source':18} {'rec%':>6} {'ocr_w':>6} {'weak':>4} {'tbl':>3} {'sec':>6}  warnings")
    for r in results:
        print(_fmt_row(r, r.page_no in set(resumed), width))
    attention = [r for r in results if needs_review(r) or r.stats.get("n_low_conf") or r.structure == "best-effort"]
    if attention:
        print("\npages needing attention:")
        for r in attention:
            tag = "NEEDS REVIEW" if needs_review(r) else "uncertain"
            print(f"  page {r.page_no} [{tag}]")
            for w in r.warnings:
                print(textwrap.fill(w, width=max(width - 6, 40), initial_indent="    - ", subsequent_indent="      "))
    srcs = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r.source for r in results).items()))
    page_secs = sum(float(r.stats.get("seconds", 0) or 0) for r in results if r.page_no not in set(resumed))
    print(f"\n{len(results)} page(s): {srcs} | {sum(1 for r in results if needs_review(r))} need review | "
          f"wall {elapsed:.1f}s (page time {page_secs:.1f}s" + (f", {len(resumed)} reused" if resumed else "") + ")")


# ====================================================================== main
def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.ERROR if a.quiet else logging.WARNING,
                        format="%(levelname)s [%(name)s] %(message)s", stream=sys.stderr)

    def say(msg: str) -> None:
        if not a.quiet:
            print(msg, file=sys.stderr, flush=True)

    def warn(msg: str) -> None:
        print(f"warning: {msg}", file=sys.stderr, flush=True)

    if a.resume and a.no_checkpoints:
        ap.error("--resume needs checkpoints; drop --no-checkpoints")
    try:
        ranges = parse_page_spec(a.pages)
    except ValueError as e:
        ap.error(str(e))
    if not os.path.isfile(a.pdf):
        ap.error(f"PDF not found: {a.pdf}")
    if a.tesseract and not (os.path.isfile(a.tesseract) or shutil.which(a.tesseract)):
        ap.error(f"--tesseract: executable not found: {a.tesseract}")
    try:
        with warnings.catch_warnings(record=True) as caught:        # Config's risky-combination warnings, shown plainly
            warnings.simplefilter("always")
            cfg = config_from_args(a)
        cfg.paddle_lang_code()                                       # unmapped language: fail now, not on the first page
    except ValueError as e:
        ap.error(str(e))
    for w in caught:
        warn(str(w.message))
    try:
        with pymupdf.open(a.pdf) as d:
            n_pages = len(d)
    except Exception as e:
        ap.error(f"cannot open {a.pdf}: {e}")
    pages, beyond = resolve_pages(ranges, n_pages)
    if beyond:
        warn(f"--pages asked for page {beyond}; the document has {n_pages} page(s): later pages ignored")
    if not pages:
        ap.error(f"no pages selected (document has {n_pages} page(s))")
    for m in preflight(cfg):
        warn(m)

    stem = os.path.splitext(os.path.basename(a.pdf))[0]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    try:
        ckpt = Checkpoints(a.out, stem, fingerprint(a.pdf, cfg) if not a.no_checkpoints else "", not a.no_checkpoints)
        outcome = run_document(a.pdf, cfg, pages, ckpt, a.resume, say, warn)
    except Exception as e:                                           # failure outside page processing
        log.debug("run failed", exc_info=True)
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if cfg.use_vl_fallback():                                    # free the VL model's GPU memory
            try:
                release_vl_engine()
            except Exception:
                pass
    elapsed = time.time() - t0

    stopped = outcome.aborted or outcome.interrupted
    internal = os.path.join(a.out, "_internal")
    if stopped:
        if outcome.results:                                          # keep what was finished, clearly marked partial
            paths = write_outputs(outcome.results, a.out, stem + ".partial", json_stem=stem)
            warn(f"stopped after {len(outcome.results)} of {len(pages)} page(s); partial output: {paths['markdown']}")
        if outcome.aborted:
            print(f"error: run aborted: {outcome.aborted}", file=sys.stderr)
            if not a.no_checkpoints and outcome.results:
                print("fix the cause, then re-run the same command with --resume to continue", file=sys.stderr)
            return EXIT_ABORTED
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED

    paths = write_outputs(outcome.results, a.out, stem)
    for stale in (os.path.join(a.out, f"{stem}.partial.md"), os.path.join(internal, f"{stem}.partial.pages.json")):
        try:
            os.remove(stale)
        except OSError:
            pass
    if not a.quiet:
        print_report(outcome.results, outcome.resumed, elapsed)
    print(paths["markdown"] if a.quiet else f"markdown: {paths['markdown']}\ninternal: {paths['internal_json']}")
    if a.strict and any(needs_review(r) for r in outcome.results):
        return EXIT_REVIEW
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
