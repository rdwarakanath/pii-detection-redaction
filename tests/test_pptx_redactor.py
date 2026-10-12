from pathlib import Path
import re

from pptx import Presentation

from extractors.pptx_extractor import extract_pptx
from redaction import redact_pptx


INPUT_PATH = Path(
    "data/input/"
    "Cadence_Financial_Group_Organizational_Pack.pptx"
)

OUTPUT_PATH = Path(
    "data/output/"
    "Cadence_Financial_Group_Organizational_Pack_redacted.pptx"
)


def main():
    print("\n===== PPTX REDACTION TEST =====")

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Sample PowerPoint not found: {INPUT_PATH}"
        )

    # Keep the original document for comparison.
    original_document = extract_pptx(str(INPUT_PATH))

    result = redact_pptx(
        input_path=str(INPUT_PATH),
        output_path=str(OUTPUT_PATH),
    )

    # Check that a separate output was created.
    assert OUTPUT_PATH.is_file(), "Redacted PPTX was not created."
    assert INPUT_PATH.resolve() != OUTPUT_PATH.resolve()

    # Re-extract the redacted PowerPoint.
    redacted_document = extract_pptx(str(OUTPUT_PATH))

    assert (
        len(original_document.blocks)
        == len(redacted_document.blocks)
    ), "The extracted block count changed."

    assert result["validation_passed"], (
        "Redacted document validation failed."
    )

    placeholders = ("xxxxxxxxxx",)

    placeholder_count = sum(
        block.text.count(placeholder)
        for block in redacted_document.blocks
        for placeholder in placeholders
    )

    assert result["redacted_findings"] > 0, (
        "No PII findings were redacted. Check the detector and input."
    )

    assert placeholder_count > 0, (
        "No redaction placeholders were found in the output."
    )

    # Independent coverage checks for the known synthetic directory fixture.
    output_text = "\n".join(block.text for block in redacted_document.blocks)
    original_names = [
        block.text for block in original_document.blocks
        if block.location_type == "table_cell"
        and block.location.get("slide") == 8
        and block.location.get("column") == 1
        and block.location.get("row", 0) > 1
    ]
    assert len(original_names) == 16, "The directory fixture changed."
    for name in original_names:
        parts = name.split()
        for variant in (name, f"{parts[0][0]}. {' '.join(parts[1:])}"):
            assert variant.casefold() not in output_text.casefold(), f"Residual name: {variant}"
    assert not re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]+", output_text)
    assert not re.search(r"\+1\s+\(555\)\s+\d{4}", output_text)
    email_header = next(
        block for block in redacted_document.blocks
        if block.location == {"slide": 8, "shape": 6, "row": 1, "column": 4}
    )
    assert email_header.text == "Email", "A harmless directory heading was redacted."
    properties = Presentation(str(OUTPUT_PATH)).core_properties
    assert properties.author == "xxxxxxxxxx"
    assert properties.last_modified_by == "xxxxxxxxxx"
    assert not result["skipped_findings"], "Review skipped findings before using this fixture output."
    print("Directory names, phones, emails, heading, metadata, and skipped findings: PASS")

    print(f"Original blocks: {len(original_document.blocks)}")
    print(f"Redacted blocks: {result['redacted_blocks']}")
    print(f"PII findings replaced: {result['redacted_findings']}")
    print(f"Placeholders in output: {placeholder_count}")
    print(f"Output file: {OUTPUT_PATH}")
    print("Validation: PASS")


if __name__ == "__main__":
    main()
