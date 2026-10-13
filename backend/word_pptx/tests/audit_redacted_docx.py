"""Independent package-text audit; media still requires manual/OCR review."""
from pathlib import Path
from zipfile import ZipFile
import re
import json
from hashlib import sha256

from lxml import etree
from extractors.docx_extractor import extract_docx
from redaction.docx_redactor import _learn_names


def main(output_path="data/output/Cadence_TPRM_Training_redacted.docx"):
    source = Path("data/input/Cadence_TPRM_Training.docx")
    output = Path(output_path)
    names = _learn_names(extract_docx(str(source)))
    findings, media = [], []
    report_path = output.with_suffix(".review.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    reviewed_media = {entry["part"]: entry["output_sha256"] for entry in report.get("reviewed_images", [])}
    checked_media = 0
    patterns = {
        "email": re.compile(r"\b[\w.!#$%&'*+/=?^`{|}~-]+@[\w.-]+\.[A-Za-z]+"),
        "phone": re.compile(r"(?<!\w)(?:\+\d{1,3}\s*)?(?:\(\d{2,4}\)|\d{2,4})[ .-]+\d{3,4}(?:[ .-]+\d{3,4})?(?!\w)"),
    }
    with ZipFile(output) as archive:
        for part in archive.namelist():
            if part.startswith(("word/media/", "word/embeddings/")):
                if sha256(archive.read(part)).hexdigest() == reviewed_media.get(part):
                    checked_media += 1
                else:
                    media.append(part)
            if not part.endswith((".xml", ".rels")):
                continue
            root = etree.fromstring(archive.read(part))
            # Join adjacent text runs to detect identifiers split across nodes.
            paragraphs = root.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
            values = ["".join(p.itertext()) for p in paragraphs] if paragraphs else [" ".join(root.itertext())]
            values += [value for node in root.iter() for value in node.attrib.values()]
            for value in values:
                for entity, pattern in patterns.items():
                    # Numeric document statistics, versions, and GUID fragments
                    # in package metadata are not phone-number candidates.
                    if entity == "phone" and not part.startswith("word/"):
                        continue
                    for match in pattern.finditer(value):
                        findings.append((part, entity, match.group()))
                for name in names:
                    if name.casefold() in value.casefold():
                        findings.append((part, "known name", name))
    print("===== DOCX PACKAGE AUDIT =====")
    print(f"Possible residual identifiers: {len(findings)}")
    for finding in findings:
        print(finding)
    print(f"Media assets matching the reviewed output hashes: {checked_media}")
    print(f"Images/embedded objects without a matching review: {len(media)}")
    print("No regex matches is not proof that every sensitive entity is removed.")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
