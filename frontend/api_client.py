"""HTTP client for the asynchronous local document-processing API."""

from dataclasses import dataclass
from email.message import Message
from typing import Any

import requests


class UnsupportedFormatError(Exception):
    """The API rejected the uploaded document format."""


class ServerError(Exception):
    """The API could not complete an HTTP operation."""


@dataclass(frozen=True)
class ProcessedFile:
    data: bytes
    filename: str
    mime: str


def _message(response: requests.Response, default: str) -> str:
    try:
        payload = response.json()
    except (ValueError, requests.JSONDecodeError):
        return default

    # FastAPI errors use ``detail``; retain compatibility with older API errors.
    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return default


def _filename(response: requests.Response, fallback: str) -> str:
    message = Message()
    message["content-disposition"] = response.headers.get("content-disposition", "")
    return message.get_filename() or fallback


def _request_error(exc: requests.RequestException, operation: str) -> ServerError:
    if isinstance(exc, requests.Timeout):
        return ServerError(f"Timed out while {operation}.")
    if isinstance(exc, requests.ConnectionError):
        return ServerError("Can't reach the processing server. Is it running?")
    return ServerError(f"The request failed while {operation}.")


def _json_response(response: requests.Response, operation: str) -> dict[str, Any]:
    if not response.ok:
        message = _message(response, f"Server error ({response.status_code}) while {operation}.")
        if response.status_code == 415:
            raise UnsupportedFormatError(message)
        raise ServerError(message)
    try:
        payload = response.json()
    except (ValueError, requests.JSONDecodeError) as exc:
        raise ServerError(f"The server returned an invalid response while {operation}.") from exc
    if not isinstance(payload, dict):
        raise ServerError(f"The server returned an invalid response while {operation}.")
    return payload


def submit_job(
    base_url: str,
    filename: str,
    data: bytes,
    mime: str = "application/pdf",
    idempotency_key: str | None = None,
    timeout=(5, 120),
) -> dict[str, Any]:
    """Upload a document and return the newly queued job."""
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/v1/jobs",
            files={"file": (filename, data, mime)},
            headers={"Idempotency-Key": idempotency_key} if idempotency_key else None,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise _request_error(exc, "submitting the document") from exc
    job = _json_response(response, "submitting the document")
    if not isinstance(job.get("job_id"), str) or not job["job_id"]:
        raise ServerError("The server accepted the request but returned no job ID.")
    return job


def get_job(base_url: str, job_id: str, timeout=(5, 20)) -> dict[str, Any]:
    """Fetch the latest state of a submitted job."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}", timeout=timeout
        )
    except requests.RequestException as exc:
        raise _request_error(exc, "checking job status") from exc
    job = _json_response(response, "checking job status")
    if job.get("job_id") != job_id or not isinstance(job.get("status"), str):
        raise ServerError("The server returned an invalid job status response.")
    return job


def get_report(base_url: str, job_id: str, timeout=(5, 20)) -> dict[str, Any]:
    """Fetch the final evaluation report for a job."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}/report", timeout=timeout
        )
    except requests.RequestException as exc:
        raise _request_error(exc, "loading the evaluation report") from exc
    return _json_response(response, "loading the evaluation report")


def download_result(
    base_url: str,
    job_id: str,
    file_format: str = "pdf",
    timeout=(5, 120),
) -> ProcessedFile:
    """Download the format-preserving result for a completed job."""
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/v1/jobs/{job_id}/result", timeout=timeout
        )
    except requests.RequestException as exc:
        raise _request_error(exc, "downloading the result") from exc

    if not response.ok:
        raise ServerError(
            _message(response, f"Server error ({response.status_code}) while downloading the result.")
        )
    expected_mime = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }.get(file_format)
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if expected_mime is None or content_type != expected_mime:
        raise ServerError("The server returned an unexpected file type for the processed document.")
    signature_ok = (
        response.content.startswith(b"%PDF-") if file_format == "pdf"
        else response.content.startswith(b"PK\x03\x04")
    )
    if not signature_ok:
        raise ServerError("The server response is not a valid processed document.")
    return ProcessedFile(
        data=response.content,
        filename=_filename(response, f"redacted_document.{file_format}"),
        mime=expected_mime,
    )
