# First_pipeline: error status and fixes

Basis: static review of the code in `firstpp1.zip` (the version you changed after the 28-page run), plus the three run logs and the 28 checkpoints. I could not run Paddle or a GPU here, so "not fixed" means the code still contains the cause. "Unverified" means it needs a real run to confirm. I cannot diff against the pre-change code, so I judged only what is in the zip.

Installed versions, from `install.log`: paddlepaddle-gpu 3.2.1 (cu126), paddleocr 3.7.0, paddlex 3.7.2.

## Summary

| # | Error | Status in current code |
|---|-------|------------------------|
| 1 | GPU out of memory on PP-StructureV3 → VL (RTX 2050, 4 GB) | **Not fixed** |
| 2 | CPU PP-StructureV3 + GPU VL crashes before page 1 | **Not fixed** |
| 3 | All pages flagged NEEDS REVIEW / `needs_vlm` | **Unverified** (blocked by #1) |
| 4 | Confidence (90–98%) is not accuracy; 85% target unverified | **Open** (needs reference pages) |
| 5 | `paddle_limit_side_len=9600` has no effect | **Not fixed** (real bug) |
| 6 | Runtime about 75 s/page, over the 25–30 min target | **Not fixed** |
| 7 | cuDNN mismatch (Paddle built for 9.9, machine has 9.5) | **Not fixed** (environment) |
| 8 | Text found by Tesseract outside Paddle blocks is never inserted | **Not fixed** |
| 9 | Coordinates only at block level; no word/line boxes saved | **Open** |
| 10 | `paddleocr[doc-parser]` is unpinned (resolved to 3.7.0) | **Not fixed** (risk) |

What did change in the code (good): the VL tier is now wired in (`VL_TIER_WIRED = True`). It has whole-page and block-crop VL, verification, and a VL → Tesseract cascade. VL is released in `cli.py` at the end of the run. None of this was exercised by a successful run, so it is untested on your machine.

---

## 1. GPU out of memory (PP-StructureV3 → PaddleOCR-VL): NOT FIXED

**Evidence in code**
- `get_structure_engine()` caches the PP-StructureV3 engine in `_STRUCT_ENGINES` and **nothing ever releases it** (there is no `release_structure_engine`).
- `get_vl_engine()` loads PaddleOCR-VL lazily on the first failing page, so both model sets sit on a 4 GB card at the same time.
- The only OOM handling is `_error_hint()` (a message) and `_free_gpu()` (`gc` + `empty_cache`). That does not free models that are still referenced.
- After 3 consecutive failures the run aborts. That is correct behaviour, but it is not a fix.

**Fix (pick one, in order of effort)**
1. **Two-pass run (recommended for 4 GB).** Pass 1: PP-StructureV3 on all pages. Release it. Pass 2: VL only on flagged pages or blocks. Add to `ocr_engine.py`:
   ```python
   def release_structure_engine(s=None):
       keys = list(_STRUCT_ENGINES) if s is None else [s.engine_key()]
       for k in keys:
           eng = _STRUCT_ENGINES.pop(k, None)
           if eng is not None:
               eng.close()
       _free_gpu()
   ```
   Call it before `_load_vl()`, and after VL finishes the flagged pages, reload PP for the next pages (or split the run into two CLI invocations that resume from checkpoints).
2. **Serve VL separately** (`--vl-backend vllm-server --vl-server-url ...`, run under WSL or Docker as the guidelines recommend). The VL weights then stay out of your Python process. `vllm-server` is a valid backend in paddleocr 3.7.0.
3. Drop the 400 DPI retry while VL is loaded (see #6), since larger rasters raise peak GPU memory.

## 2. CPU PP-StructureV3 + GPU VL crash: NOT FIXED

- Exit code `-1073741819` is a Windows access violation (`0xC0000005`). The log stops right after the PP model-loading lines, with no Python traceback.
- Nothing in the code diagnoses or guards this.

**Fix**
- Get a traceback: set `PYTHONFAULTHANDLER=1` or call `faulthandler.enable()` at the top of `cli.py`, then run on a single page (`--pages 1`).
- Test PaddleOCR-VL **alone** with a 5-line script on one image. If it also crashes natively on Windows, move VL to WSL or Docker (guideline: native Windows support is unvalidated).
- Do not mix CPU and GPU Paddle in one process. The two-pass design in #1 avoids this.

## 3. Every page NEEDS REVIEW: UNVERIFIED

- In the Tesseract-fallback run, `needs_vlm` is set on every page where the result stays weak (`pipeline.py`, around line 596). That meant all 28 pages.
- Two causes show up in the checkpoints: "no Tesseract line overlaps; kept Paddle text, uncertain" (block 1 on 27 of 28 pages), and ink outside blocks.
- The new VL cascade should reduce this, but only once #1 and #2 work.

**Fix**: first fix #1. Then re-run with `--fallback-engine vl` and compare the share of pages flagged. Also make `vl` the default in `config.py` (`fallback_engine`) so you do not run the legacy path by accident, as the guidelines already say.

## 4. Confidence is not accuracy: OPEN

- Paddle recognition confidence averaged 96%, yet 178 of 504 blocks (35%) failed the gate as `run_together` (words fused) and were replaced by Tesseract text.
- Real errors remain in the output, for example "Group **Rislk** Management Policy" and a stray "三" before CADENCE on page 1. The partial VL run read both correctly.

**Fix**
- Hand-correct 5–6 representative pages (scanned, table, multi-column) as reference text.
- Run `eval.py` for CER and WER per page. This is the only way to check the 85% target.
- Investigate why Paddle fuses words on about a third of the blocks (see #5 as one possible contributor; this is unproven).

## 5. `paddle_limit_side_len = 9600` does nothing: NOT FIXED (real bug)

- `PaddleSettings.enabled_modules()` passes the key `limit_side_len` to `PPStructureV3`. In paddleocr 3.7.0 that pipeline has no such parameter. The real ones are `text_det_limit_side_len` and `text_det_limit_type`.
- The run log proves it had no effect: "Resized image size (3400x4400) exceeds max_side_limit of 4000".
- The config comment is therefore misleading.

**Fix**
- In `PPStructureV3Engine.__init__`, map the setting:
  ```python
  kw["text_det_limit_side_len"] = s.limit_side_len
  kw["text_det_limit_type"] = "max"
  ```
  and stop sending `limit_side_len`. Take it out of `enabled_modules()` and log it separately.
- On a 4 GB GPU, raising the limit raises memory use. A 300 DPI page only shrinks from 4400 to 4000 px (about 9%), so keeping 4000 may be the better choice for now. If so, delete the 9600 default and fix the comment.

## 6. Runtime above target: NOT FIXED

- About 75 s/page; 28 pages took about 34.9 min, so 35 pages is roughly 44 min against the 25–30 min goal.
- Contributors: the 400 DPI retry (`retry_dpis=(400,)`) runs a full second pass whenever the first result is weak (page 1 did this); Tesseract runs on every page that fails the gate, plus OSD.

**Fix**: set `retry_dpis=()` (CLI: `--no-retry` if present, otherwise `--retry-dpi` with no values), and measure. Paddle resizes inputs internally, so higher DPI helps little. Re-measure once VL replaces Tesseract.

## 7. cuDNN mismatch: NOT FIXED (environment)

- Log warning: Paddle is compiled with cuDNN 9.9 but the machine has 9.5, "which may cause serious incompatible bug".

**Fix**: install a cuDNN build matching the Paddle wheel (9.9 for the cu126 wheel), or use the Docker or WSL setup from the PaddleOCR-VL docs. This may also be related to #2, but that is unproven.

## 8. Missing content is only flagged: NOT FIXED

- On 25 of 28 pages Tesseract found about 100 characters outside every Paddle block. The code logs "possible missing content (not inserted)" and moves on.

**Fix**: in `run_attempt`, when `orphan_chars >= 15`, send the orphan line bounding box (padded) to `vl_repair_blocks` or `recognize_region`. If VL is unavailable, insert the Tesseract lines as extra blocks with `source="tesseract-orphan"`, their own `bbox_px` and `bbox_pdf`, and a review note, instead of dropping them.

## 9. Coordinates: block-level only: OPEN

- All 504 blocks have `bbox_px` and `bbox_pdf` (PDF points on the unrotated page), which is good. Tesseract-replaced blocks keep Paddle's box.
- Word and line boxes are not stored. The `Word` model exists in `models.py` but is not used by the PP path.
- The Markdown carries no coordinates, and `pages.json` is written only when the whole run ends. A stopped run (like your 28-page one) therefore has coordinates only in the per-page checkpoints.

**Fix**: persist line-level boxes from PP-StructureV3's `overall_ocr_res` (`rec_boxes`, `rec_texts`, `rec_scores`) into each block, converted with `ra.geo.box(...)`. Write the consolidated `pages.json` from the checkpoints at the end of every run, including partial runs.

## 10. Unpinned `paddleocr[doc-parser]`: NOT FIXED (risk)

- `requirements.txt` pins `paddlepaddle-gpu==3.2.1` but not paddleocr, so pip resolved 3.7.0 and paddlex 3.7.2.
- The guidelines call for a tested pin. Different releases change parameter names (see #5) and VL behaviour (`pipeline_version` v1 / v1.5 / v1.6).

**Fix**: after a working run, `pip freeze | findstr /i "paddle"` and pin those exact versions in `requirements.txt`. The resume fingerprint already includes library versions, so a version change will correctly invalidate checkpoints.

---

## Suggested order of work
1. #5 (small code fix), #6 (config change), #10 (pin).
2. #7 and #2 (environment): get a traceback and a stable VL smoke test.
3. #1 (two-pass or server VL), then re-run with `--fallback-engine vl`.
4. #8 and #9 (completeness and coordinates).
5. #4: build the reference pages and measure CER and WER.

---

## Implementation status (2026-10-10)

Applied to the current source files, but **not validated with a real Paddle/VL page run**:

- #1: PP-StructureV3 is explicitly closed and removed from its cache before VL initializes; VL is released after use. This prevents both model sets from being intentionally held in the process at once. The per-page model switching may add substantial load time, and individual PP or VL inference may still exceed 4 GB.
- #2: Python fault handling is enabled so native crashes can emit a stack trace. This is diagnostic only; the CPU-PP/GPU-VL Windows access violation is not proven fixed.
- #3: VL is now the default fallback policy, Tesseract OCR runs only after VL is rejected/unavailable or leaves weak blocks, and missing VL verification references are sent to the Tesseract tier. Page flags remain unverified until a successful representative run.
- #4: still open. The 0.85 threshold remains a Paddle confidence routing signal. No corrected reference pages were supplied, so the 85% accuracy target cannot be claimed.
- #5: `limit_side_len` is sent as `text_det_limit_side_len` with `text_det_limit_type="max"`; the default is 4000 to avoid the review's high-memory 9600 setting.
- #6: higher-DPI retries are disabled by default. Runtime target still requires a measured run.
- #7: unresolved environment mismatch. `paddlepaddle-gpu==3.2.1` in the installed environment declares `nvidia-cudnn-cu12==9.5.1.17`, while the run log reports a cuDNN 9.9 build expectation. Forcing cuDNN 9.9 into requirements would conflict with the installed Paddle wheel metadata; resolve by selecting and validating a compatible Paddle/CUDA environment.
- #8: substantial positioned Tesseract lines outside PP blocks are retained as separate text blocks with pixel/PDF boxes and a review note instead of being dropped.
- #9: PP recognized lines and Tesseract lines now retain line-level boxes. Partial runs write the consolidated JSON report to the usual `<stem>.pages.json` path while keeping Markdown named `<stem>.partial.md`.
- #10: PaddleOCR 3.7.0 and PaddleX 3.7.2 are pinned to the versions in the recorded install log.

Python syntax compilation and CLI help succeeded after these edits. GPU inference, native Windows stability, OCR quality, line-box geometry, and runtime remain unverified; no full-document run was performed.
