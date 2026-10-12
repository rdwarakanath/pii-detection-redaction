from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DocumentBlock:
    """
    A normalized piece of extracted document content.

    Every extractor (PPTX, DOCX, PDF) should eventually
    produce DocumentBlock objects so downstream components
    do not need to know the original file format.
    """

    text: str

    source_file: str

    file_type: str

    location_type: str

    location: Dict[str, Any]

    extraction_method: str

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDocument:
    """
    Represents an entire processed document.
    """

    document_id: str

    source_file: str

    file_type: str

    blocks: List[DocumentBlock] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)