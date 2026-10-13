import zipfile
from dataclasses import dataclass
from pathlib import Path

ALLOWED_LABEL = "PDF, Word (.docx), PowerPoint (.pptx)"
_TABLE = {
    ".pdf":  ("pdf",  "PDF",  "application/pdf"),
    ".docx": ("docx", "DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".pptx": ("pptx", "PPTX", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}
# DOCX and PPTX are both ZIP files; each has a required part that identifies it.
_MARKER = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml"}

class InvalidFormat(Exception): ...

@dataclass(frozen=True)
class ClassifiedFile:
    key: str
    label: str
    mime: str

def classify(name: str, f) -> ClassifiedFile:
    ext = Path(name).suffix.lower()
    if ext not in _TABLE:
        raise InvalidFormat(f"“{name}” isn't supported. Please upload {ALLOWED_LABEL}.")
    key, label, mime = _TABLE[ext]
    f.seek(0)
    try:
        if key == "pdf":
            if b"%PDF-" not in f.read(1024):
                raise InvalidFormat(f"“{name}” doesn't look like a valid PDF.")
        else:
            try:
                with zipfile.ZipFile(f) as z:
                    ok = _MARKER[key] in z.namelist()
            except zipfile.BadZipFile:
                ok = False
            if not ok:
                raise InvalidFormat(f"“{name}” isn't a valid {label} file.")
    finally:
        f.seek(0)
    return ClassifiedFile(key, label, mime)
