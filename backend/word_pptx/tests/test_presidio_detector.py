
from extractors import extract_pptx, extract_docx
from detectors.presidio_detector import PresidioDetector


detector = PresidioDetector()

sample = (
    "Contact Daniela Figueroa at "
    "daniela@example.com or +1 212-555-0100."
)

print("\n===== PRESIDIO SAMPLE TEST =====")

from models.document import DocumentBlock

sample_block = DocumentBlock(
    text=sample,
    source_file="sample.txt",
    file_type="TXT",
    location_type="test",
    location={"test_case": 1},
    extraction_method="manual",
)

sample_result = detector.analyze_block(sample_block)

for finding in sample_result["findings"]:
    print(
        finding["entity_type"],
        repr(finding["text"]),
        finding["score"],
    )

print("\n===== PPTX DOCUMENT TEST =====")

pptx = extract_pptx(
    "data/input/Cadence_Financial_Group_Organizational_Pack.pptx"
)
pptx_results = detector.analyze_document(pptx)

print(f"PPTX blocks analyzed: {len(pptx_results)}")
print(
    "PPTX blocks with findings:",
    sum(bool(item["findings"]) for item in pptx_results),
)

print("\n===== DOCX DOCUMENT TEST =====")

docx = extract_docx(
    "data/input/Cadence_TPRM_Training.docx"
)
docx_results = detector.analyze_document(docx)

print(f"DOCX blocks analyzed: {len(docx_results)}")
print(
    "DOCX blocks with findings:",
    sum(bool(item["findings"]) for item in docx_results),
)
