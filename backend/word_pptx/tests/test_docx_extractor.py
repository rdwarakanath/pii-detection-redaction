from extractors.docx_extractor import extract_docx


docx_path = "data/input/Cadence_TPRM_Training.docx"

document = extract_docx(docx_path)

print("\n========================================")
print("DOCX V2 EXTRACTION TEST")
print("========================================")

print(f"File: {document.source_file}")
print(f"Type: {document.file_type}")
print(f"Document ID: {document.document_id}")
print(f"Total blocks: {len(document.blocks)}")


# ----------------------------------------
# Separate paragraph and table blocks
# ----------------------------------------

paragraph_blocks = [
    block
    for block in document.blocks
    if block.location_type == "paragraph"
]

table_blocks = [
    block
    for block in document.blocks
    if block.location_type == "table_cell"
]


print("\n========================================")
print("SUMMARY")
print("========================================")

print(f"Paragraph blocks: {len(paragraph_blocks)}")
print(f"Table-cell blocks: {len(table_blocks)}")
print(f"Total blocks: {len(document.blocks)}")


# ----------------------------------------
# Show table extraction
# ----------------------------------------

print("\n========================================")
print("TABLE BLOCKS")
print("========================================")

for index, block in enumerate(table_blocks, start=1):

    print("\n----------------------------------------")
    print(f"TABLE BLOCK {index}")
    print(f"Location: {block.location}")
    print(f"Text: {block.text}")