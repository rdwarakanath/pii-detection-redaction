import requests_mock

from api_client import download_result, get_job, get_report, submit_job


def test_job_api_contract_lifecycle():
    base_url = "http://testserver"
    job_id = "0123456789abcdef0123456789abcdef"
    evaluation = {"score": 96.0, "passed": True}
    job = {"job_id": job_id, "filename": "my_sample.pdf", "status": "completed", "evaluation": evaluation}
    report = {"job_id": job_id, "status": "completed", "evaluation": evaluation}
    pdf = b"%PDF-1.4 mock redacted result"

    with requests_mock.Mocker() as mock:
        mock.post(f"{base_url}/api/v1/jobs", status_code=202, json=job)
        mock.get(f"{base_url}/api/v1/jobs/{job_id}", json=job)
        mock.get(f"{base_url}/api/v1/jobs/{job_id}/report", json=report)
        mock.get(
            f"{base_url}/api/v1/jobs/{job_id}/result",
            content=pdf,
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="mock_redacted.pdf"',
            },
        )

        submitted = submit_job(base_url, "my_sample.pdf", b"%PDF-1.4 source")
        current = get_job(base_url, submitted["job_id"])
        final_report = get_report(base_url, submitted["job_id"])
        result = download_result(base_url, submitted["job_id"])

    assert current["status"] == "completed"
    assert final_report["evaluation"]["passed"] is True
    assert result.data == pdf
    assert result.filename == "mock_redacted.pdf"
