# OCR Pipeline Implementation Guidelines

## Objective

Build a page-aware PDF-to-Markdown pipeline with PP-StructureV3 as primary, lazy PaddleOCR-VL as the second tier, and Tesseract as the last-resort OCR tier. Keep page/block JSON internal for QA. PII detection and redaction are out of scope until OCR quality is accepted.

## Page routing

1. Classify each page before OCR. Extract a trustworthy digital text layer natively; do not send it through OCR unnecessarily. Route blank pages to a recorded blank result. Route scanned, hybrid, corrupt-text-layer, and vector-text pages through the OCR path.
2. Render OCR pages at the configured DPI (start at 300). Keep the raster-to-PDF coordinate mapping and original page number for every block and retry.
3. Run PP-StructureV3 first. Use its page/block structure and recognition output to reconstruct Markdown. Record enabled modules and engine/library versions.

## Quality gate and fallback

1. Treat 0.85 as the initial PP text-recognition confidence threshold, not as a claim of 85% document accuracy. Do not use layout-detection confidence as an OCR-quality score. Missing confidence is unknown, never 100%.
2. Also fail the gate for missing/empty content, run-together text, junk glyphs, implausible text, excessive uncovered ink, broken reading order, or incoherent tables. Record each reason.
3. PP-StructureV3 is tier one. If its page/layout gate fails, release its model and lazily load PaddleOCR-VL for a whole-page read. If the layout is usable but blocks fail, load VL only then and read those block crops.
4. VL has no directly comparable per-token confidence. Check non-empty content, completeness, repetition, plausible length, table coherence, and agreement with healthy PP text when available. Do not run Tesseract before VL by default and never compare raw Paddle and Tesseract confidence numbers.
5. If VL cannot load, is rejected, or leaves weak content unresolved, release it and run positioned Tesseract as the last OCR tier. Replace a block only when Tesseract is clearly better. Preserve Tesseract lines outside PP blocks as position-bearing text blocks and mark their placement for review. Never silently drop recognized content.
6. If the quality gate passes for PP, do not load VL or run Tesseract OCR. The default is one worker and no higher-DPI retry to limit GPU memory and runtime. Retries must be explicitly enabled and measured.
7. Keep a failed page isolated. Record engine errors and attempts, checkpoint completed pages, and support resuming only when the PDF, settings, versions, and relevant code fingerprint match. Repeated engine-wide errors must stop the run clearly while retaining completed-page output.

## Current baseline operation

The default CLI policy is PP-StructureV3 → lazy PaddleOCR-VL → Tesseract last resort. `--fallback-engine tesseract` remains an explicit compatibility mode that skips VL. PaddleOCR-VL requires a compatible GPU runtime. The pipeline releases the PP model before loading VL and releases VL before the next page's PP pass to avoid keeping both model sets resident together.

## Output and evaluation

- Write one merged Markdown document in input page order. Preserve headings, paragraphs, lists, reading order, and tables where the parser supports them; Markdown is reconstructed content, not a pixel-faithful page copy.
- Keep detailed blocks, line and block pixel/PDF coordinates, per-page scores, fallback attempts, reasons, versions, and timings in the internal JSON report. Persist a consolidated JSON report for complete and partial runs. Do not use filename glob order to assemble page output.
- Evaluate against manually corrected page references spanning digital, scanned, hybrid, multi-column, and table pages. Report CER, WER, page runtime, selected engine, and fallback/review status. The acceptance target is at least 85% text correctness on the representative sample, alongside acceptable reading order and table structure.
- A 25–30 minute run for 35 pages is acceptable only if the measured quality gate is met. Do not infer document accuracy from engine confidence.
- Continue to PII design only after the Markdown and location data have been reviewed and the evaluation target is met.

## Runtime and dependencies

- Pin the recorded PaddleOCR/PaddleX versions in `requirements.txt`; do not install CPU and GPU Paddle wheels into the same environment. Validate the exact Paddle/CUDA/cuDNN combination before a document run.
- The currently installed Paddle GPU wheel declares cuDNN 9.5.1.17 while the prior run log reports a cuDNN 9.9 build expectation. Do not force a conflicting cuDNN pin; select a matching supported Paddle environment and verify it on the target machine.
- Match the PaddlePaddle GPU wheel to the installed CUDA/runtime and platform. The previous PP-StructureV3 CPU trial on PaddlePaddle 3.2.2 is evidence for that CPU setup only; it does not prove the combined GPU environment works.
- PaddlePaddle 3.3.1 on Windows CPU previously hit a PIR/oneDNN inference error. Keep oneDNN disabled for that known configuration and surface failures; do not claim this workaround fixes other builds without a run.
- PaddleOCR-VL's official setup guidance currently recommends WSL or Docker for Windows. Native Windows support must be validated on the actual machine before treating it as reliable.
