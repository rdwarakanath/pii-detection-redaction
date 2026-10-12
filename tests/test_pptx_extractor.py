from extractors.pptx_extractor import (
    extract_pptx,
    save_extracted_text,
)


# ========================================
# FILE PATHS
# ========================================

PPTX_PATH = (
    "data/input/"
    "Cadence_Financial_Group_Organizational_Pack.pptx"
)

OUTPUT_PATH = (
    "data/output/"
    "Cadence_Financial_Group_Organizational_Pack_extracted.txt"
)


# ========================================
# EXTRACT PPTX
# ========================================

document = extract_pptx(PPTX_PATH)


# ========================================
# DISPLAY SUMMARY
# ========================================

print("\n========================================")
print("PPTX EXTRACTION TEST")
print("========================================")

print(
    f"File: {document.source_file}"
)

print(
    f"Type: {document.file_type}"
)

print(
    f"Document ID: {document.document_id}"
)

print(
    f"Total blocks: {len(document.blocks)}"
)


# ========================================
# COUNT TABLE BLOCKS
# ========================================

table_blocks = [
    block
    for block in document.blocks
    if block.location_type == "table_cell"
]


print(
    f"Table blocks: {len(table_blocks)}"
)


# ========================================
# SAVE EXTRACTED TEXT
# ========================================

save_extracted_text(
    document,
    OUTPUT_PATH
)


# ========================================
# FINAL RESULT
# ========================================

print("\n========================================")
print("EXTRACTION COMPLETE")
print("========================================")

print(
    f"Extracted text saved to:"
)

print(
    OUTPUT_PATH
)