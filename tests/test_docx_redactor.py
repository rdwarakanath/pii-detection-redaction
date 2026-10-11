from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import re
import json

from docx import Document
from detectors.presidio_detector import PresidioDetector
from extractors.docx_extractor import extract_docx
from redaction.docx_redactor import redact_docx, _learn_names


def main():
    detector = PresidioDetector()
    with TemporaryDirectory() as folder:
        source = Path(folder) / "source.docx"
        output = Path(folder) / "output.docx"
        doc = Document()
        paragraph = doc.add_paragraph("Contact ")
        paragraph.add_run("first@").bold = True
        paragraph.add_run("company.example").italic = True
        paragraph.add_run(" on +1 (555) 0114.")
        doc.sections[0].header.paragraphs[0].text = "first@company.example"
        table = doc.add_table(rows=2, cols=1)
        table.cell(0, 0).text = "Users"
        table.cell(1, 0).text = "KC Burney\nDK Koran"
        doc.core_properties.author = "Private Author"
        doc.save(source)
        original_bytes = source.read_bytes()
        result = redact_docx(source, output, detector)
        reopened = Document(output)
        assert source.read_bytes() == original_bytes
        assert reopened.paragraphs[0].text == "Contact xxxxxxxxxx on xxxxxxxxxx."
        assert reopened.paragraphs[0].runs[1].bold
        assert reopened.paragraphs[0].runs[2].italic
        assert reopened.sections[0].header.paragraphs[0].text == "xxxxxxxxxx"
        assert reopened.tables[0].cell(1, 0).text == "xxxxxxxxxx\nxxxxxxxxxx"
        assert reopened.core_properties.author == "xxxxxxxxxx"
        assert not result["skipped_findings"]
        try:
            redact_docx(source, source, detector)
        except ValueError:
            pass
        else:
            raise AssertionError("Overwriting source must fail")
    print("Split runs, formatting, headers, name tables, metadata, and source safety: PASS")
    source = Path("data/input/Cadence_TPRM_Training.docx")
    output = Path("data/output/Cadence_TPRM_Training_redacted.docx")
    result = redact_docx(source, output, detector)
    original, redacted = extract_docx(str(source)), extract_docx(str(output))
    assert len(original.blocks) == len(redacted.blocks) == 1292
    text = "\n".join(block.text for block in redacted.blocks)
    names = _learn_names(original)
    assert "KC Burney" in names and "DK Koran" in names
    for name in names:
        assert name.casefold() not in text.casefold(), f"Residual name: {name}"
    assert "Optiv" in text and "Offboarding Workflow" in text
    assert not re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]+", text)
    assert not result["skipped_findings"], result["skipped_findings"]
    with ZipFile(output) as archive:
        xml = archive.read("docProps/core.xml").decode()
        assert "Madhav Mehta" not in xml and "Divisha Kilari" not in xml
    print(json.dumps({key: result[key] for key in ("total_blocks", "redacted_findings", "validation_passed", "review_required")}, indent=2))
    print("Training document coverage: PASS")


if __name__ == "__main__":
    main()
