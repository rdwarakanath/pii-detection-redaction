import io
import zipfile
import pytest
from validation import classify, InvalidFormat, ClassifiedFile


def make_zip(namelist):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in namelist:
            z.writestr(name, "test content")
    buf.seek(0)
    return buf


def make_pdf():
    return io.BytesIO(b"%PDF-1.7\n%fake pdf content")


def test_classify_valid_pdf():
    f = make_pdf()
    res = classify("sample.pdf", f)
    assert res == ClassifiedFile(
        key="pdf",
        label="PDF",
        mime="application/pdf",
    )
    assert f.tell() == 0  # seek(0) in finally


def test_classify_valid_docx():
    f = make_zip(["word/document.xml", "docProps/core.xml"])
    res = classify("sample.docx", f)
    assert res == ClassifiedFile(
        key="docx",
        label="DOCX",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert f.tell() == 0


def test_classify_valid_pptx():
    f = make_zip(["ppt/presentation.xml", "docProps/core.xml"])
    res = classify("presentation.pptx", f)
    assert res == ClassifiedFile(
        key="pptx",
        label="PPTX",
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )
    assert f.tell() == 0


def test_classify_txt_renamed_to_pdf():
    f = io.BytesIO(b"Just plain text without PDF signature")
    with pytest.raises(InvalidFormat, match="doesn't look like a valid PDF"):
        classify("document.pdf", f)
    assert f.tell() == 0


def test_classify_docx_renamed_to_pptx():
    # Contains word/document.xml instead of ppt/presentation.xml
    f = make_zip(["word/document.xml"])
    with pytest.raises(InvalidFormat, match="isn't a valid PPTX file"):
        classify("document.pptx", f)
    assert f.tell() == 0


def test_classify_pptx_renamed_to_docx():
    f = make_zip(["ppt/presentation.xml"])
    with pytest.raises(InvalidFormat, match="isn't a valid DOCX file"):
        classify("presentation.docx", f)
    assert f.tell() == 0


def test_classify_unsupported_extensions():
    for ext in [".doc", ".xlsx", ".png", ".txt", ".jpeg", ".csv"]:
        f = io.BytesIO(b"data")
        with pytest.raises(InvalidFormat, match="isn't supported"):
            classify(f"file{ext}", f)


def test_classify_zero_byte_file():
    # PDF zero-byte
    f_pdf = io.BytesIO(b"")
    with pytest.raises(InvalidFormat, match="doesn't look like a valid PDF"):
        classify("empty.pdf", f_pdf)

    # DOCX zero-byte
    f_docx = io.BytesIO(b"")
    with pytest.raises(InvalidFormat, match="isn't a valid DOCX file"):
        classify("empty.docx", f_docx)

    # PPTX zero-byte
    f_pptx = io.BytesIO(b"")
    with pytest.raises(InvalidFormat, match="isn't a valid PPTX file"):
        classify("empty.pptx", f_pptx)


def test_classify_corrupted_zip():
    f = io.BytesIO(b"PK\x03\x04not a full zip")
    with pytest.raises(InvalidFormat, match="isn't a valid DOCX file"):
        classify("corrupted.docx", f)
