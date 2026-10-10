# OCR Pipeline Mindmap

```text
PDF
└── Inspect each page
    ├── Trusted digital text → native extraction
    ├── Blank → record blank, skip OCR
    └── Scanned / hybrid / corrupt text layer / vector text
        └── Render page (default 300 DPI; retain page and coordinate mapping)
            └── PP-StructureV3 (primary)
                ├── Layout, reading order, OCR, tables and enabled modules
                ├── Save block and line scores/boxes internally
                └── Quality gate: 0.85 recognition confidence + content/layout checks
                    ├── Pass → use PP result; do not load VL or run Tesseract OCR
                    └── Fail / PP error
                        ├── Release PP model
                        └── Lazy-load PaddleOCR-VL
                            ├── Layout failed → VL reads whole page
                            ├── Localized weak blocks → VL reads only those crops
                            ├── Validate content; no comparison of raw engine scores
                            ├── Pass → accept verified VL page/block results
                            └── Rejected / unavailable / still weak
                                ├── Release VL model
                                └── Tesseract (last-resort OCR)
                                    ├── Match positioned lines to PP blocks
                                    ├── Replace only clearly improved block text
                                    └── Preserve substantial unmatched lines as positioned blocks
    └── Assemble pages in page order
        ├── User output: merged Markdown
        ├── Internal QA: JSON with page/block/line text, pixel/PDF boxes, scores,
        │   sources, attempts, warnings and timings
        ├── Save consolidated JSON for complete and partial runs
        └── Evaluate against corrected references (CER/WER) before PII design
```

## Rules in brief

- Default OCR order is PP-StructureV3 → lazy PaddleOCR-VL → Tesseract last resort. A passing PP result does not initialize VL.
- Release PP before loading VL and release VL before another page's PP inference to avoid both large models staying resident together.
- The 0.85 Paddle recognition-confidence threshold is a routing signal, not proof of 85% accuracy. Validate text correctness against corrected references.
- Never compare Paddle and Tesseract confidence values as if they shared a scale. Missing confidence is unknown.
- Whole-page VL is for a failed page/layout; block-crop VL is for localized failures. Verify before accepting either result.
- Tesseract text outside PP blocks must be retained with its bounding boxes and marked for placement review, not silently dropped.
- Do not feed Markdown directly into PII processing until OCR output and coordinate handling have been reviewed.
