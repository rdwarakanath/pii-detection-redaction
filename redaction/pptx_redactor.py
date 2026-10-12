
from pathlib import Path
from typing import Any

from pptx import Presentation

from detectors.presidio_detector import PresidioDetector
from extractors.pptx_extractor import extract_pptx
from models import document


PLACEHOLDER = "xxxxxxxxxx"


def _placeholder_for(entity_type: str) -> str:
    """Use the same placeholder for every detected PII entity."""
    return PLACEHOLDER


def _select_non_overlapping_findings(
    findings: list[dict[str, Any]],
    text_length: int,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Keep valid, sufficiently confident, non-overlapping detections.

    When detections overlap, prefer the higher-confidence result.
    """
    candidates = []

    for finding in findings:
        start = int(finding["start"])
        end = int(finding["end"])
        score = float(finding.get("score", 0.0))

        if score < min_score:
            continue

        if start < 0 or end <= start or end > text_length:
            continue

        candidates.append(finding)

    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            int(item["start"]),
            -(int(item["end"]) - int(item["start"])),
        )
    )

    selected = []

    for candidate in candidates:
        start = int(candidate["start"])
        end = int(candidate["end"])

        overlaps = any(
            start < int(existing["end"])
            and end > int(existing["start"])
            for existing in selected
        )

        if not overlaps:
            selected.append(candidate)

    return sorted(selected, key=lambda item: int(item["start"]))


def _redact_text(
    text: str,
    findings: list[dict[str, Any]],
    min_score: float = 0.0,
) -> str:
    """Produce the expected redacted text without modifying a document."""
    selected = _select_non_overlapping_findings(
        findings,
        len(text),
        min_score,
    )

    # Replace from right to left so earlier offsets remain valid.
    for finding in reversed(selected):
        start = int(finding["start"])
        end = int(finding["end"])
        placeholder = _placeholder_for(finding["entity_type"])

        text = text[:start] + placeholder + text[end:]

    return text


def _build_character_map(text_frame):
    """
    Map each character in a PowerPoint text frame to its original run
    and character index. Paragraph separators are represented by None.
    """
    character_map = []
    paragraphs = text_frame.paragraphs

    for paragraph_index, paragraph in enumerate(paragraphs):
        for run in paragraph.runs:
            for character_index in range(len(run.text)):
                character_map.append((run, character_index))

        if paragraph_index < len(paragraphs) - 1:
            character_map.append(None)

    # Fail safely if python-pptx's run content cannot represent the
    # text frame exactly. Do not silently overwrite all formatting.
    if len(character_map) != len(text_frame.text):
        raise ValueError(
            "Could not map PowerPoint text to its original runs. "
            "The redaction was stopped to avoid modifying the wrong text."
        )

    return character_map



def _filter_single_paragraph_findings(
    text_frame,
    block_text: str,
    findings: list[dict[str, Any]],
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Keep only findings that fit safely inside one paragraph."""

    original_text = text_frame.text

    if original_text.strip() != block_text:
        raise ValueError(
            "The extracted text does not match the original PowerPoint text."
        )

    character_map = _build_character_map(text_frame)
    leading_offset = len(original_text) - len(original_text.lstrip())

    safe_findings = []

    for finding in findings:
        start = int(finding["start"])
        end = int(finding["end"])
        score = float(finding.get("score", 0.0))

        if score < min_score:
            continue

        if start < 0 or end <= start or end > len(block_text):
            continue

        absolute_start = leading_offset + start
        absolute_end = leading_offset + end

        if absolute_end > len(character_map):
            continue

        span_map = character_map[absolute_start:absolute_end]

        # None marks a paragraph separator.
        if any(position is None for position in span_map):
            continue

        safe_findings.append(finding)

    return _select_non_overlapping_findings(
        safe_findings,
        len(block_text),
        min_score,
    )



def _replace_span_in_text_frame(
    text_frame,
    block_text: str,
    findings: list[dict[str, Any]],
    min_score: float = 0.0,
) -> int:
    """
    Replace detected spans while retaining unaffected text runs.

    The placeholder inherits the formatting of the run where the
    sensitive span begins. A span crossing paragraph boundaries is
    rejected rather than risking incorrect redaction.
    """
    original_frame_text = text_frame.text

    if original_frame_text.strip() != block_text:
        raise ValueError(
            "The source text does not match the extracted block. "
            "Redaction stopped rather than risk editing the wrong content."
        )

    selected = _select_non_overlapping_findings(
        findings,
        len(block_text),
        min_score,
    )

    if not selected:
        return 0

    # The extractor strips outer whitespace. Account for leading
    # whitespace when translating block offsets to frame offsets.
    leading_offset = len(original_frame_text) - len(
        original_frame_text.lstrip()
    )

    # Descending order keeps the positions of earlier text unchanged.
    for finding in reversed(selected):
        start = leading_offset + int(finding["start"])
        end = leading_offset + int(finding["end"])

        character_map = _build_character_map(text_frame)

        if start < 0 or end > len(character_map) or start >= end:
            raise ValueError(
                "A detected PII span is outside the original text frame."
            )

        span_map = character_map[start:end]

        if any(position is None for position in span_map):
            raise ValueError(
                "A detected PII span crosses paragraph boundaries. "
                "Redaction stopped to avoid changing the wrong text."
            )

        first_run, first_index = span_map[0]
        last_run, last_index = span_map[-1]

        placeholder = _placeholder_for(finding["entity_type"])

        if first_run is last_run:
            first_run.text = (
                first_run.text[:first_index]
                + placeholder
                + first_run.text[last_index + 1:]
            )
        else:
            # Keep the prefix and insert the placeholder into the
            # first run, then remove the rest of the sensitive span.
            first_run.text = (
                first_run.text[:first_index] + placeholder
            )

            runs_in_span = []

            for run, _ in span_map:
                if not runs_in_span or runs_in_span[-1] is not run:
                    runs_in_span.append(run)

            for run in runs_in_span[1:-1]:
                run.text = ""

            last_run.text = last_run.text[last_index + 1:]

    return len(selected)


def _get_text_frame(presentation, block):
    """Locate the text frame identified by an extracted DocumentBlock."""
    location = block.location

    slide_number = int(location["slide"])
    shape_number = int(location["shape"])

    # The current extractor stores these location numbers as 1-based.
    slide = presentation.slides[slide_number - 1]
    shape = slide.shapes[shape_number - 1]

    if block.location_type == "table_cell":
        if not shape.has_table:
            raise ValueError(
                f"Expected a table at slide {slide_number}, "
                f"shape {shape_number}."
            )

        row_number = int(location["row"])
        column_number = int(location["column"])

        cell = shape.table.cell(row_number - 1, column_number - 1)
        return cell.text_frame

    if not shape.has_text_frame:
        raise ValueError(
            f"Expected a text frame at slide {slide_number}, "
            f"shape {shape_number}."
        )

    return shape.text_frame


def redact_pptx(
    input_path: str,
    output_path: str,
    detector: PresidioDetector | None = None,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """
    Redact PII in a PPTX file and save a separate output copy.

    Args:
        input_path: Path to the original .pptx file.
        output_path: Destination for the redacted .pptx file.
        detector: Optional existing PresidioDetector instance.
        min_score: Minimum Presidio confidence score to redact.

    Returns:
        A summary containing block counts, redaction counts,
        output path, and validation status.
    """
    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Input PowerPoint was not found: {source}"
        )

    if source.suffix.lower() != ".pptx":
        raise ValueError("The input file must have a .pptx extension.")

    if destination.suffix.lower() != ".pptx":
        raise ValueError("The output file must have a .pptx extension.")

    if source.resolve() == destination.resolve():
        raise ValueError(
            "Input and output paths must differ. "
            "The original file must not be overwritten."
        )

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0.0 and 1.0.")

    destination.parent.mkdir(parents=True, exist_ok=True)

    detector = detector or PresidioDetector()

    # Use the existing extractor and its DocumentBlock locations.
    document = extract_pptx(str(source))
    if hasattr(detector, "prepare_document"):
        detector.prepare_document(document)

    # Analyze all blocks before modifying the presentation.

    # Load the original presentation before planning redactions.
    presentation = Presentation(str(source))

    # Analyze all blocks before modifying the presentation.
    plans = []
    skipped_findings = []

    for block in document.blocks:
        analysis = detector.analyze_block(block)

        # Locate the original text and filter unsafe cross-paragraph spans.
        text_frame = _get_text_frame(presentation, block)

        safe_findings = _filter_single_paragraph_findings(
            text_frame,
            block.text,
            analysis["findings"],
            min_score,
        )

        for finding in analysis["findings"]:
            if finding not in safe_findings:
                skipped_findings.append({
                    "location": block.location,
                    "finding": finding,
                    "reason": "Below threshold, invalid, overlapping, or crosses a paragraph boundary",
                })

        expected_text = _redact_text(
            block.text,
            safe_findings,
            min_score,
        )

        plans.append(
            {
                "block": block,
                "findings": safe_findings,
                "expected_text": expected_text,
            }
        )


    redacted_blocks = 0
    redacted_findings = 0

    for plan in plans:
        block = plan["block"]
        findings = plan["findings"]

        if not findings:
            continue

        text_frame = _get_text_frame(presentation, block)

        count = _replace_span_in_text_frame(
            text_frame,
            block.text,
            findings,
            min_score,
        )

        if count:
            redacted_blocks += 1
            redacted_findings += count

    metadata_redacted = []
    for field in ("author", "last_modified_by"):
        if getattr(presentation.core_properties, field):
            setattr(presentation.core_properties, field, PLACEHOLDER)
            metadata_redacted.append(field)

    # Save to a new file; never modify the original.
    presentation.save(str(destination))

    # Re-open and re-extract the output to validate every block.
    try:
        redacted_document = extract_pptx(str(destination))

        if len(redacted_document.blocks) != len(plans):
            raise RuntimeError(
                "Validation failed: the number of extracted blocks changed."
            )

        for index, plan in enumerate(plans):
            actual = redacted_document.blocks[index].text
            expected = plan["expected_text"]

            if actual != expected:
                block = plan["block"]
                raise RuntimeError(
                    "Validation failed for "
                    f"slide {block.location.get('slide')}, "
                    f"shape {block.location.get('shape')}: "
                    "the saved text differs from the expected redacted text."
                )

    except Exception:
        # Do not leave a failed output looking like a validated result.
        destination.unlink(missing_ok=True)
        raise

    return {
        "input_path": str(source),
        "output_path": str(destination),
        "total_blocks": len(document.blocks),
        "redacted_blocks": redacted_blocks,
        "redacted_findings": redacted_findings,
        "skipped_findings": skipped_findings,
        "metadata_redacted": metadata_redacted,
        "validation_passed": True,
    }
