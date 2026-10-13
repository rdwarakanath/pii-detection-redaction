"""Run the documented DOCX/PPTX redactor for one uploaded document.

This is the generic worker entry point used by the FastAPI service. The
sample-specific modules under ``tests`` remain demonstrations only.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from detectors.presidio_detector import PresidioDetector
from redaction import redact_docx, redact_pptx


MARKERS = {".docx": "word/document.xml", ".pptx": "ppt/presentation.xml"}
IMAGE_MANIFEST = Path(__file__).resolve().parent / "config" / "docx_image_redactions.json"


def _package_is_valid(path: Path, suffix: str) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return MARKERS[suffix] in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _summary_docx(report: dict[str, Any]) -> dict[str, Any]:
    # The full DOCX review report contains original sensitive values. Keep it
    # in the job folder; expose counts and review flags only through the API.
    reasons = []
    if not report.get("visual_validation_passed"):
        reasons.append("Visual inspection has not been completed by the service.")
    if report.get("skipped_findings"):
        reasons.append("One or more detector findings were skipped and need review.")
    if report.get("unreviewed_parts"):
        reasons.append("The document contains parts that were not covered by the reviewed redaction rules.")
    return {
        "format": "docx",
        "validation_passed": bool(report.get("validation_passed")),
        "review_required": bool(report.get("review_required", True)),
        "review_reasons": reasons,
        "total_blocks": int(report.get("total_blocks", 0)),
        "redacted_findings": int(report.get("redacted_findings", 0)),
        "skipped_findings_count": len(report.get("skipped_findings", [])),
        "unreviewed_parts_count": len(report.get("unreviewed_parts", [])),
        "reviewed_images_count": len(report.get("reviewed_images", [])),
        "visual_validation_passed": bool(report.get("visual_validation_passed")),
        "metadata_redacted_count": len(report.get("metadata_redacted", [])),
    }


def _summary_pptx(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "pptx",
        "validation_passed": bool(report.get("validation_passed")),
        # The documented extractor covers slide text and table cells. It does
        # not perform a complete visual/media review of arbitrary uploads.
        "review_required": True,
        "review_reasons": [
            "Visual review is required before distribution.",
            "Text inside images and other unsupported PowerPoint content is not covered by the documented extractor.",
        ],
        "total_blocks": int(report.get("total_blocks", 0)),
        "redacted_blocks": int(report.get("redacted_blocks", 0)),
        "redacted_findings": int(report.get("redacted_findings", 0)),
        "skipped_findings_count": len(report.get("skipped_findings", [])),
        "metadata_redacted_count": len(report.get("metadata_redacted", [])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redact one DOCX or PPTX upload")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args(argv)

    source, destination, summary_path = args.input.resolve(), args.output.resolve(), args.summary.resolve()
    suffix = source.suffix.lower()
    if suffix not in MARKERS:
        parser.error("input must be a .docx or .pptx file")
    if not source.is_file() or not _package_is_valid(source, suffix):
        parser.error("input is missing or is not a valid Office Open XML document")
    if source == destination or destination.suffix.lower() != suffix:
        parser.error("output must be a separate file with the same extension as the input")

    detector = PresidioDetector()
    if suffix == ".docx":
        report = redact_docx(
            str(source), str(destination), detector=detector,
            image_manifest_path=str(IMAGE_MANIFEST),
        )
        summary = _summary_docx(report)
    else:
        report = redact_pptx(str(source), str(destination), detector=detector)
        summary = _summary_pptx(report)

    if not destination.is_file() or not _package_is_valid(destination, suffix):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Redactor did not produce a valid Office Open XML output")
    if not summary["validation_passed"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Redactor validation did not pass; output is withheld")

    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
