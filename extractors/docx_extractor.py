from pathlib import Path
from uuid import uuid4

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from models.document import DocumentBlock, NormalizedDocument


def extract_docx(file_path: str) -> NormalizedDocument:
    """
    Extract machine-readable content from a DOCX file.

    Extracts:
    1. Paragraph text
    2. Table-cell text

    The original order of paragraphs and tables is preserved.
    """

    path = Path(file_path)

    # ----------------------------------------
    # Validate file
    # ----------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() != ".docx":
        raise ValueError(
            f"Expected a DOCX file, got: {path.suffix}"
        )

    # ----------------------------------------
    # Open document
    # ----------------------------------------

    document_file = Document(path)

    # ----------------------------------------
    # Create normalized document
    # ----------------------------------------

    document = NormalizedDocument(
        document_id=str(uuid4()),
        source_file=path.name,
        file_type="DOCX",
    )

    # ----------------------------------------
    # Counters
    # ----------------------------------------

    paragraph_number = 0
    table_number = 0

    # ----------------------------------------
    # Walk through document in original order
    # ----------------------------------------

    for child in document_file.element.body.iterchildren():

        # ========================================
        # PARAGRAPH
        # ========================================

        if child.tag.endswith("}p"):

            paragraph_number += 1

            paragraph = Paragraph(
                child,
                document_file
            )

            text = paragraph.text.strip()

            # Ignore empty paragraphs
            if not text:
                continue

            block = DocumentBlock(
                text=text,
                source_file=path.name,
                file_type="DOCX",
                location_type="paragraph",
                location={
                    "paragraph": paragraph_number,
                },
                extraction_method="python-docx",
                metadata={},
            )

            document.blocks.append(block)

        # ========================================
        # TABLE
        # ========================================

        elif child.tag.endswith("}tbl"):

            table_number += 1

            table = Table(
                child,
                document_file
            )

            # Go through every row
            for row_number, row in enumerate(
                table.rows,
                start=1
            ):

                # Go through every cell
                for column_number, cell in enumerate(
                    row.cells,
                    start=1
                ):

                    text = cell.text.strip()

                    # Ignore empty cells
                    if not text:
                        continue

                    block = DocumentBlock(
                        text=text,
                        source_file=path.name,
                        file_type="DOCX",
                        location_type="table_cell",
                        location={
                            "table": table_number,
                            "row": row_number,
                            "column": column_number,
                        },
                        extraction_method="python-docx",
                        metadata={},
                    )

                    document.blocks.append(block)

    # ----------------------------------------
    # Return complete document
    # ----------------------------------------

    return document