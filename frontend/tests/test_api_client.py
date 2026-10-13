import pytest
import requests
import requests_mock

from api_client import (
    ProcessedFile,
    ServerError,
    UnsupportedFormatError,
    download_result,
    get_job,
    get_report,
    submit_job,
)


BASE_URL = "http://testserver"
JOB_ID = "a" * 32


def test_submit_job_posts_pdf_and_returns_job():
    with requests_mock.Mocker() as mock:
        mock.post(
            f"{BASE_URL}/api/v1/jobs",
            status_code=202,
            json={"job_id": JOB_ID, "status": "queued"},
        )

        job = submit_job(
            BASE_URL,
            "input.pdf",
            b"%PDF-1.4 test",
            idempotency_key="f" * 32,
        )

    assert job["job_id"] == JOB_ID
    assert mock.last_request.headers["Content-Type"].startswith("multipart/form-data")
    assert mock.last_request.headers["Idempotency-Key"] == "f" * 32
    assert "input.pdf" in mock.last_request.text


def test_submit_job_reports_fastapi_validation_error():
    with requests_mock.Mocker() as mock:
        mock.post(
            f"{BASE_URL}/api/v1/jobs",
            status_code=415,
            json={"detail": "The uploaded PDF could not be opened."},
        )

        with pytest.raises(UnsupportedFormatError, match="could not be opened"):
            submit_job(BASE_URL, "bad.pdf", b"not a pdf")


def test_get_job_returns_current_status():
    with requests_mock.Mocker() as mock:
        mock.get(
            f"{BASE_URL}/api/v1/jobs/{JOB_ID}",
            json={"job_id": JOB_ID, "status": "processing", "progress": 45},
        )

        job = get_job(BASE_URL, JOB_ID)

    assert job["status"] == "processing"
    assert job["progress"] == 45


def test_get_job_rejects_mismatched_job_id():
    with requests_mock.Mocker() as mock:
        mock.get(
            f"{BASE_URL}/api/v1/jobs/{JOB_ID}",
            json={"job_id": "b" * 32, "status": "completed"},
        )

        with pytest.raises(ServerError, match="invalid job status"):
            get_job(BASE_URL, JOB_ID)


def test_get_report_returns_evaluation():
    evaluation = {"score": 96.0, "passed": True}
    with requests_mock.Mocker() as mock:
        mock.get(
            f"{BASE_URL}/api/v1/jobs/{JOB_ID}/report",
            json={"job_id": JOB_ID, "evaluation": evaluation},
        )

        report = get_report(BASE_URL, JOB_ID)

    assert report["evaluation"] == evaluation


def test_download_result_checks_pdf_and_filename():
    content = b"%PDF-1.4 redacted output"
    with requests_mock.Mocker() as mock:
        mock.get(
            f"{BASE_URL}/api/v1/jobs/{JOB_ID}/result",
            content=content,
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="redacted.pdf"',
            },
        )

        result = download_result(BASE_URL, JOB_ID)

    assert isinstance(result, ProcessedFile)
    assert result.data == content
    assert result.filename == "redacted.pdf"
    assert result.mime == "application/pdf"


def test_download_result_rejects_wrong_content_type():
    with requests_mock.Mocker() as mock:
        mock.get(
            f"{BASE_URL}/api/v1/jobs/{JOB_ID}/result",
            content=b"not pdf",
            headers={"Content-Type": "text/html"},
        )

        with pytest.raises(ServerError, match="unexpected file type"):
            download_result(BASE_URL, JOB_ID)


def test_network_errors_are_translated():
    with requests_mock.Mocker() as mock:
        mock.get(
            f"{BASE_URL}/api/v1/jobs/{JOB_ID}",
            exc=requests.exceptions.ConnectionError,
        )

        with pytest.raises(ServerError, match="Can't reach the processing server"):
            get_job(BASE_URL, JOB_ID)
