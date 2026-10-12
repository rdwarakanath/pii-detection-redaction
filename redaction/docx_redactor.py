"""Redact Word text in OOXML while preserving run and paragraph properties."""
from pathlib import Path
import json
import re
import tempfile
from zipfile import ZipFile
from hashlib import sha256

from lxml import etree

from detectors.presidio_detector import PresidioDetector, PLACEHOLDER
from extractors.docx_extractor import extract_docx
from models.document import DocumentBlock
from .image_redactor import redact_reviewed_image

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
# Reviewed terminology in the training document, not person identifiers.
NON_PERSON_TERMS = {
    "optiv", "torii", "email", "cloud", "workflow", "onboarding workflow",
    "offboarding workflow", "vendor tier", "send servpro", "launching servpro",
    "launching stafven", "qr_task privacy", "qr_task security", "qr_task",
    "qrn_sensitive targeting data", "contains",
}


def _paragraph_text(paragraph):
    text, mapping = [], []
    for node in paragraph.iter():
        # Nested text-box paragraphs are processed independently.
        if next(node.iterancestors(f"{{{W}}}p"), None) is not paragraph:
            continue
        if node.tag == f"{{{W}}}t":
            for index, char in enumerate(node.text or ""):
                text.append(char)
                mapping.append((node, index))
        elif node.tag in {f"{{{W}}}tab", f"{{{W}}}br", f"{{{W}}}cr"}:
            text.append("\t" if node.tag.endswith("}tab") else "\n")
            mapping.append(None)
    return "".join(text), mapping


def _learn_names(document):
    rows = {}
    names = set()
    for block in document.blocks:
        if block.location_type == "table_cell":
            loc = block.location
            rows.setdefault(loc["table"], {}).setdefault(loc["row"], {})[loc["column"]] = block.text
    for table in rows.values():
        user_columns = {col for col, value in table.get(1, {}).items() if value.strip().casefold() in {"users", "name", "full name"}}
        for row_number, cells in table.items():
            values = [cells[col] for col in user_columns if col in cells and row_number > 1]
            if cells.get(1, "").strip().casefold() in {"client contact", "author(s)", "author", "peer reviewer", "qa reviewer"}:
                values.append(cells.get(2, ""))
            for value in values:
                for name in re.split(r"[,;\n]+", value):
                    parts = name.strip().split()
                    if len(parts) >= 2 and all(re.fullmatch(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", part) for part in parts):
                        names.add(" ".join(parts))
    return names


def _freeze_delivery_dates(root):
    """Preserve the source's cached Delivery Date when Word updates fields."""
    frozen = []
    for row in root.xpath("//w:tr", namespaces=NS):
        cells = row.findall("w:tc", NS)
        if len(cells) < 2 or "".join(cells[0].xpath(".//w:t/text()", namespaces=NS)).strip().casefold() != "delivery date":
            continue
        cell = cells[1]
        instructions = "".join(cell.xpath(".//w:instrText/text()", namespaces=NS)).strip()
        if not re.fullmatch(r'DATE\s+\\@\s*"MMMM d, yyyy"', instructions, re.IGNORECASE):
            continue
        value = "".join(cell.xpath(".//w:t/text()", namespaces=NS))
        if not value.strip():
            raise ValueError("Delivery Date field has no cached value to preserve")
        for node in cell.xpath(".//w:fldChar | .//w:instrText", namespaces=NS):
            node.getparent().remove(node)
        frozen.append({"label": "Delivery Date", "value": value, "reason": "Preserve source date during Word export"})
    return frozen


def redact_docx(input_path, output_path, detector=None, min_score=0.0, image_manifest_path=None):
    """Save a separate redacted copy and a review summary alongside it.

    Covers ordinary Word text in body, tables, headers, footers, notes and
    comments. Media is changed only using hash-guarded, reviewed rectangles.
    Unknown media, revisions and unrecognized field instructions are reported.
    """
    source, destination = Path(input_path), Path(output_path)
    if not source.is_file() or source.suffix.lower() != ".docx":
        raise ValueError("Input must be an existing DOCX file.")
    if destination.suffix.lower() != ".docx" or source.resolve() == destination.resolve():
        raise ValueError("Output must be a separate DOCX file.")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score must be between 0 and 1.")
    original = extract_docx(str(source))
    detector = detector or PresidioDetector()
    detector.prepare_document(original)
    names = _learn_names(original)
    detector.known_names.update(names)
    detector.known_names.update(f"{name.split()[0][0]}. {' '.join(name.split()[1:])}" for name in names)
    report = {"input_path": str(source), "output_path": str(destination), "total_blocks": len(original.blocks),
              "redacted_findings": 0, "replacements": [], "excluded_findings": [],
              "skipped_findings": [], "unreviewed_parts": [], "metadata_redacted": []}
    report["visual_validation_passed"] = False
    report["visual_validation_note"] = "Render and inspect before distribution; text validation does not establish visual quality."
    report["image_redactions"] = []
    report["reviewed_images"] = []
    report["reviewed_fields"] = []
    report["reviewed_external_targets"] = []
    report["preserved_delivery_dates"] = []
    manifest_path = Path(image_manifest_path) if image_manifest_path else Path(__file__).resolve().parents[1] / "config" / "docx_image_redactions.json"
    image_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["images"] if manifest_path.is_file() else {}
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    expected = {}
    with ZipFile(source) as archive:
        parts = {info.filename: (info, archive.read(info.filename)) for info in archive.infolist()}
    for part, (info, data) in list(parts.items()):
        if part.startswith("word/media/"):
            entry = image_manifest.get(part)
            if entry and sha256(data).hexdigest() == entry.get("sha256"):
                redacted_data = redact_reviewed_image(data, entry)
                parts[part] = (info, redacted_data)
                report["reviewed_images"].append({"part": part, "status": entry["review_status"], "source_sha256": entry["sha256"], "output_sha256": sha256(redacted_data).hexdigest()})
                if entry.get("regions"):
                    report["image_redactions"].append({"part": part, "regions": entry["regions"], "placeholder": PLACEHOLDER})
            else:
                report["unreviewed_parts"].append(part + (": source differs from reviewed image" if entry else ""))
            continue
        if part.startswith("word/embeddings/"):
            report["unreviewed_parts"].append(part)
        if not part.endswith(".xml"):
            continue
        root = etree.fromstring(data, parser)
        if part.startswith("word/"):
            if part == "word/document.xml":
                report["preserved_delivery_dates"] = _freeze_delivery_dates(root)
            if root.xpath("//w:del | //w:delText", namespaces=NS):
                report["unreviewed_parts"].append(part + ": deleted text")
            instructions = " ".join(node.text or "" for node in root.xpath("//w:instrText", namespaces=NS))
            # Field instructions can be split across runs; review the combined
            # stream rather than accepting each fragment independently.
            safe_field = (
                r'DATE\s*(?:\\@\s*"(?:MMMM d, yyyy|yyyy)"\s*)?(?:\\\*\s*MERGEFORMAT)?'
                r'|TOC\s+\\o\s*"1-3"\s+\\h\s+\\z\s+\\u'
                r'|PAGEREF\s+_Toc\d+\s+\\h|PAGE\b'
            )
            residual_instructions = re.sub(safe_field, '', instructions, flags=re.IGNORECASE).strip()
            if instructions.strip() and not residual_instructions:
                report["reviewed_fields"].append({"part": part, "reason": "Only date, table-of-contents, and page-number instructions"})
            elif instructions.strip():
                report["unreviewed_parts"].append(part + ": unrecognized field instructions")
            for index, paragraph in enumerate(root.iter(f"{{{W}}}p"), 1):
                text, mapping = _paragraph_text(paragraph)
                if not text.strip():
                    continue
                block = DocumentBlock(text, source.name, "DOCX", "xml_paragraph", {"part": part, "paragraph": index}, "OOXML")
                findings = detector.analyze_block(block)["findings"]
                target = text
                for finding in reversed(findings):
                    start, end = finding["start"], finding["end"]
                    entry = {"location": block.location, **finding}
                    term = finding["text"].strip().casefold()
                    if finding["entity_type"] == "PERSON" and (term in NON_PERSON_TERMS or re.fullmatch(r"[a-z_]+(?:[._][a-z0-9]+)+", term)):
                        report["excluded_findings"].append({**entry, "reason": "Reviewed organization, workflow term, or rule identifier"})
                        continue
                    if finding["score"] < min_score or any(item is None for item in mapping[start:end]):
                        report["skipped_findings"].append({**entry, "reason": "Below threshold or spans a tab/line break"})
                        continue
                    span = mapping[start:end]
                    first, first_offset = span[0]
                    last, last_offset = span[-1]
                    if first is last:
                        first.text = first.text[:first_offset] + PLACEHOLDER + first.text[last_offset + 1:]
                    else:
                        first.text = first.text[:first_offset] + PLACEHOLDER
                        seen = []
                        for node, _ in span:
                            if not seen or seen[-1] is not node:
                                seen.append(node)
                        for node in seen[1:-1]:
                            node.text = ""
                        last.text = last.text[last_offset + 1:]
                    for node, _ in span:
                        node.set(XML_SPACE, "preserve")
                    target = target[:start] + PLACEHOLDER + target[end:]
                    report["replacements"].append(entry)
                    report["redacted_findings"] += 1
                assert _paragraph_text(paragraph)[0] == target, "Run replacement validation failed"
                expected[(part, index)] = target
            # Comments can also store personal author names in attributes.
            for node in root.iter():
                for attr in (f"{{{W}}}author", f"{{{W}}}initials"):
                    if node.get(attr):
                        node.set(attr, PLACEHOLDER)
        if part == "docProps/core.xml":
            for node in root:
                if etree.QName(node).localname in {"creator", "lastModifiedBy"} and node.text:
                    node.text = PLACEHOLDER
                    report["metadata_redacted"].append(etree.QName(node).localname)
        parts[part] = (info, etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True))
    # Hyperlink targets can retain an email even after display text is redacted.
    for part, (info, data) in list(parts.items()):
        if not part.endswith(".rels"):
            continue
        root = etree.fromstring(data, parser)
        for relation in root:
            if relation.get("TargetMode") == "External":
                target = relation.get("Target", "")
                if target.lower().startswith(("mailto:", "tel:")) or any(name.casefold() in target.casefold() for name in names):
                    relation.set("Target", PLACEHOLDER)
                elif target == "https://example.com/training-resource":
                    report["reviewed_external_targets"].append(target)
                else:
                    report["unreviewed_parts"].append(f"{part}: external target {target}")
        parts[part] = (info, etree.tostring(root, xml_declaration=True, encoding="UTF-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".docx", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with ZipFile(staging, "w") as archive:
            for info, data in parts.values():
                archive.writestr(info, data)
        with ZipFile(staging) as archive:
            for part in {key[0] for key in expected}:
                root = etree.fromstring(archive.read(part), parser)
                for index, paragraph in enumerate(root.iter(f"{{{W}}}p"), 1):
                    if (part, index) in expected:
                        assert _paragraph_text(paragraph)[0] == expected[(part, index)]
        reopened = extract_docx(str(staging))
        assert [b.location for b in reopened.blocks] == [b.location for b in original.blocks], "Extracted structure changed"
        report["validation_passed"] = True
        report["review_required"] = bool(report["skipped_findings"] or report["unreviewed_parts"] or not report["visual_validation_passed"])
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
    report_path = destination.with_suffix(".review.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
