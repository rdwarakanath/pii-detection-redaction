# DOCX redaction

The DOCX redactor uses the existing extractor and Presidio detector. It replaces supported findings with exactly `xxxxxxxxxx` in Word text nodes, including headers, footers, notes and comments, and redacts author and last-modifier metadata. It preserves unaffected run properties. The original is never overwritten.

```powershell
python -m tests.test_docx_redactor
python -m tests.test_docx_image_redaction
python -m tests.audit_redacted_docx
```

The sample output is `data/output/Cadence_TPRM_Training_redacted.docx`. Close it in Word before regenerating. A failed save leaves the previous output intact.

## Image policy

`config/docx_image_redactions.json` contains reviewed media digests and rectangles for the Cadence training sample. Four screenshots contain 26 personal name, username or search regions. Pillow replaces those pixels with a white background and the required placeholder. This is actual image-data replacement, not a Word overlay. Unaffected pixels, image dimensions and other assets are preserved.

Coordinates are applied only when the media part and SHA-256 digest match the reviewed original. A changed or unknown image is left unchanged and reported as unreviewed. This policy is specific to these screenshots; it does not automatically discover personal data in other images. Supply another reviewed manifest through `image_manifest_path` for other documents.

The Windows OCR script in `tests/ocr_docx_media.ps1` is an optional local review helper. It reports text and rectangles and never modifies a document. OCR is supporting evidence, not proof that an image is free of personal data. This does not modify either extractor or the separate PDF/OCR pipeline.

## Review and limits

The adjacent `.review.json` records text replacements, exclusions, skipped findings, reviewed image hashes, image regions, fields, external targets and unresolved parts. It contains original sensitive identifiers and should remain local. `review_required` stays true until full-page visual verification is performed.

Date, contents and page-number field instructions with the explicitly reviewed syntax are preserved. Deleted revision text, unknown field instructions, embedded objects, unknown media and external targets are reported for review. The reviewed sample training-resource link is preserved. Unsupported identifiers and image content in new documents can remain.

The sample has 1,292 extracted blocks, 28 text name replacements, 41 reviewed media assets, and four masked screenshots. Edited screenshots have been visually inspected and checked with local OCR. An 84-page Word PDF export was reviewed. It exposed two non-person false positives, which are now excluded, and a Delivery Date field that Word updated automatically. The redactor now preserves the source delivery date as fixed text in the labeled cell. The updated DOCX requires a fresh Word PDF export for final visual verification; the packaged LibreOffice renderer remains unavailable.
