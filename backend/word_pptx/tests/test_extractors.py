from extractors import extract_pptx, extract_docx


PPTX_PATH = "data/input/Cadence_Financial_Group_Organizational_Pack.pptx"
DOCX_PATH = "data/input/Cadence_TPRM_Training.docx"


# ========================================
# PPTX TEST
# ========================================

print("\n========================================")
print("PPTX VALIDATION")
print("========================================")

pptx_document = extract_pptx(PPTX_PATH)

pptx_table_blocks = [
    block
    for block in pptx_document.blocks
    if block.location_type == "table_cell"
]

print(f"Total blocks: {len(pptx_document.blocks)}")
print(f"Table blocks: {len(pptx_table_blocks)}")

assert len(pptx_document.blocks) > 0
assert len(pptx_table_blocks) > 0

print("PPTX extraction: PASS")


# ========================================
# DOCX TEST
# ========================================

print("\n========================================")
print("DOCX VALIDATION")
print("========================================")

docx_document = extract_docx(DOCX_PATH)

docx_paragraph_blocks = [
    block
    for block in docx_document.blocks
    if block.location_type == "paragraph"
]

docx_table_blocks = [
    block
    for block in docx_document.blocks
    if block.location_type == "table_cell"
]

print(f"Total blocks: {len(docx_document.blocks)}")
print(f"Paragraph blocks: {len(docx_paragraph_blocks)}")
print(f"Table blocks: {len(docx_table_blocks)}")

assert len(docx_document.blocks) > 0
assert len(docx_paragraph_blocks) > 0
assert len(docx_table_blocks) > 0

print("DOCX extraction: PASS")


# ========================================
# FINAL RESULT
# ========================================

print("\n========================================")
print("ALL EXTRACTION TESTS PASSED")
print("========================================")