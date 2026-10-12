from pathlib import Path
from uuid import uuid4

from pptx import Presentation

from models.document import DocumentBlock, NormalizedDocument


def extract_pptx(file_path: str) -> NormalizedDocument:
    """
    Extract machine-readable text from a PPTX file.

    Extracts:
    1. Normal text-containing shapes
    2. Text from table cells

    Each extracted piece of content becomes a DocumentBlock.
    """

    path = Path(file_path)

    # ----------------------------------------
    # Validate file
    # ----------------------------------------

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if path.suffix.lower() != ".pptx":
        raise ValueError(
            f"Expected a PPTX file, got: {path.suffix}"
        )

    # ----------------------------------------
    # Open presentation
    # ----------------------------------------

    presentation = Presentation(path)

    # ----------------------------------------
    # Create normalized document
    # ----------------------------------------

    document = NormalizedDocument(
        document_id=str(uuid4()),
        source_file=path.name,
        file_type="PPTX",
    )

    # ----------------------------------------
    # Process every slide
    # ----------------------------------------

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        # Process every shape on the slide
        for shape_number, shape in enumerate(
            slide.shapes,
            start=1
        ):

            # ========================================
            # TABLE EXTRACTION
            # ========================================

            if shape.has_table:

                table = shape.table

                # Go through every row
                for row_number, row in enumerate(
                    table.rows,
                    start=1
                ):

                    # Go through every column/cell
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
                            file_type="PPTX",
                            location_type="table_cell",
                            location={
                                "slide": slide_number,
                                "shape": shape_number,
                                "row": row_number,
                                "column": column_number,
                            },
                            extraction_method="python-pptx",
                            metadata={
                                "shape_type": str(
                                    shape.shape_type
                                )
                            },
                        )

                        document.blocks.append(block)

                # We already processed this shape
                continue

            # ========================================
            # NORMAL TEXT EXTRACTION
            # ========================================

            # Ignore shapes that don't contain text
            if not shape.has_text_frame:
                continue

            text = shape.text.strip()

            # Ignore empty text boxes
            if not text:
                continue

            block = DocumentBlock(
                text=text,
                source_file=path.name,
                file_type="PPTX",
                location_type="slide",
                location={
                    "slide": slide_number,
                    "shape": shape_number,
                },
                extraction_method="python-pptx",
                metadata={
                    "shape_type": str(
                        shape.shape_type
                    )
                },
            )

            document.blocks.append(block)

    # ----------------------------------------
    # Return complete document
    # ----------------------------------------

    return document


def save_extracted_text(
    document: NormalizedDocument,
    output_path: str
):
    """
    Save extracted PPTX content to a text file.

    The text file contains:
    - Extracted text
    - Location information
    - Extraction method

    This preserves traceability for downstream
    PII detection and redaction.
    """

    output_file = Path(output_path)

    # Create parent directory if it doesn't exist
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------
    # Write extracted content
    # ----------------------------------------

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for index, block in enumerate(
            document.blocks,
            start=1
        ):

            file.write("=" * 60 + "\n")
            file.write(
                f"BLOCK {index}\n"
            )
            file.write("=" * 60 + "\n")

            file.write(
                f"Text: {block.text}\n"
            )

            file.write(
                f"Location Type: "
                f"{block.location_type}\n"
            )

            file.write(
                f"Location: "
                f"{block.location}\n"
            )

            file.write(
                f"Extraction Method: "
                f"{block.extraction_method}\n"
            )

            file.write("\n")