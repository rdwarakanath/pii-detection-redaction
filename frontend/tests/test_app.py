import io
import zipfile
from pathlib import Path
import requests_mock
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def make_valid_pdf_bytes():
    return b"%PDF-1.4 test document content"


def make_valid_docx_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", "<document/>")
    return buf.getvalue()


def test_app_initial_state():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.title) == 1
    assert at.title[0].value == "Document Processor"
    assert len(at.file_uploader) == 1
    assert not at.button  # No submit button when no file uploaded
    assert at.session_state["status"] == "idle"
    assert at.session_state["file_sig"] is None


def test_app_upload_wrong_format():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.file_uploader[0].upload("data.xlsx", b"fake excel")
    at.run()

    assert at.session_state["error"] is not None
    assert at.session_state["error"]["kind"] == "format"
    assert "isn't supported" in at.session_state["error"]["message"]
    assert not at.button  # Submit button hidden


def test_app_upload_fake_extension_txt_as_pdf():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.file_uploader[0].upload("fake.pdf", b"This is plain text pretending to be PDF")
    at.run()

    assert at.session_state["error"] is not None
    assert at.session_state["error"]["kind"] == "format"
    assert "doesn't look like a valid PDF" in at.session_state["error"]["message"]
    assert not at.button


def test_app_valid_pdf_upload_and_process_success():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.file_uploader[0].upload("sample.pdf", make_valid_pdf_bytes())
    at.run()

    # Document recognized, button available
    assert at.session_state["doc"] is not None
    assert at.session_state["doc"].key == "pdf"
    assert len(at.button) == 1
    assert at.button[0].label == "Process document"

    # Mock the HTTP call
    with requests_mock.Mocker(real_http=True) as m:
        m.post(
            "http://127.0.0.1:8000/api/v1/jobs",
            json={"job_id": "a" * 32, "status": "queued", "filename": "sample.pdf"},
            status_code=202,
        )
        m.get(
            f"http://127.0.0.1:8000/api/v1/jobs/{'a' * 32}",
            json={
                "job_id": "a" * 32,
                "status": "completed",
                "progress": 100,
                "message": "Redacted PDF is ready.",
                "evaluation": {"score": 96.0, "passed": True, "critical_leaks": 0, "high_leaks": 0},
            },
        )
        m.get(
            f"http://127.0.0.1:8000/api/v1/jobs/{'a' * 32}/result",
            content=b"%PDF-1.4 redacted output",
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="sample_redacted.pdf"',
            },
        )
        m.get(
            f"http://127.0.0.1:8000/api/v1/jobs/{'a' * 32}/report",
            json={
                "job_id": "a" * 32,
                "evaluation": {"score": 96.0, "passed": True, "critical_leaks": 0, "high_leaks": 0},
            },
        )
        at.button[0].click()
        at.run()

    # Status should be done, result set, download button visible
    assert at.session_state["status"] == "completed"
    assert at.session_state["result"] is not None
    assert at.session_state["result"].filename == "sample_redacted.pdf"
    assert len(at.download_button) == 1
    assert "Download sample_redacted.pdf" in at.download_button[0].label


def test_app_server_error_enables_retry():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.file_uploader[0].upload("sample.pdf", make_valid_pdf_bytes())
    at.run()

    with requests_mock.Mocker(real_http=True) as m:
        m.post(
            "http://127.0.0.1:8000/api/v1/jobs",
            json={"detail": "OCR service crashed"},
            status_code=500,
        )
        at.button[0].click()
        at.run()

    # Status reset to idle, server error set, button enabled again
    assert at.session_state["status"] == "idle"
    assert at.session_state["error"] is not None
    assert at.session_state["error"]["kind"] == "server"
    assert "OCR service crashed" in at.session_state["error"]["message"]
    assert len(at.button) == 1
    assert not at.button[0].disabled  # Enabled for retry


def test_app_new_file_clears_old_state():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.file_uploader[0].upload("sample.pdf", make_valid_pdf_bytes())
    at.run()

    with requests_mock.Mocker(real_http=True) as m:
        m.post(
            "http://127.0.0.1:8000/api/v1/jobs",
            json={"job_id": "c" * 32, "status": "queued", "filename": "sample.pdf"},
            status_code=202,
        )
        m.get(
            f"http://127.0.0.1:8000/api/v1/jobs/{'c' * 32}",
            json={
                "job_id": "c" * 32,
                "status": "completed",
                "progress": 100,
                "message": "Redacted PDF is ready.",
                "evaluation": {"score": 96.0, "passed": True},
            },
        )
        m.get(
            f"http://127.0.0.1:8000/api/v1/jobs/{'c' * 32}/result",
            content=b"%PDF-1.4 redacted",
            headers={"Content-Type": "application/pdf"},
        )
        m.get(
            f"http://127.0.0.1:8000/api/v1/jobs/{'c' * 32}/report",
            json={"job_id": "c" * 32, "evaluation": {"score": 96.0, "passed": True}},
        )
        at.button[0].click()
        at.run()

    assert at.session_state["status"] == "completed"
    assert at.session_state["result"] is not None

    # Now upload a different file
    at.file_uploader[0].upload("another.docx", make_valid_docx_bytes())
    at.run()

    # Previous result cleared, status reset to idle
    assert at.session_state["result"] is None
    assert at.session_state["status"] == "idle"
    assert at.session_state["doc"].key == "docx"
