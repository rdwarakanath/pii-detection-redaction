"""Local FastAPI job API for PDF, DOCX, and PPTX PII redaction.

Run from ``PII_redaction/backend`` with ``python -m uvicorn api:app``.
Document processors run as child processes so native crashes do not terminate
the HTTP server. The one-thread executor serializes processing jobs.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


BACKEND_DIR = Path(__file__).resolve().parent
PDF_PACKAGE_DIR = BACKEND_DIR / "pdf_files"
PROCESSING_DIR = PDF_PACKAGE_DIR / "processing"
EVAL_DIR = PDF_PACKAGE_DIR / "eval"
WORD_PPTX_DIR = BACKEND_DIR / "word_pptx"
JOBS_DIR = Path(os.getenv("PII_JOBS_DIR", str(BACKEND_DIR / "jobs"))).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("PII_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("PII_MAX_PDF_PAGES", "500"))
MAX_ATTEMPTS = 3
PASS_SCORE = 85.0
OCR_DEVICE = os.getenv("PII_OCR_DEVICE", "gpu:0")
OCR_FALLBACK = os.getenv("PII_OCR_FALLBACK_ENGINE", "vl")
COMMAND_TIMEOUT_SECONDS = int(os.getenv("PII_COMMAND_TIMEOUT_SECONDS", "10800"))

_DOC_MARKERS = {".docx": "word/document.xml", ".pptx": "ppt/presentation.xml"}
_RESULT_MIMES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.RLock()
_IDEMPOTENCY_LOCK = asyncio.Lock()
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pii-ocr")

app = FastAPI(title="Local PII Redaction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "PII_FRONTEND_ORIGINS", "http://127.0.0.1:8501,http://localhost:8501"
    ).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS_DIR / job_id


def _save_job(job: dict[str, Any]) -> None:
    job["updated_at"] = _now()
    job_dir = JOBS_DIR / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
    tmp.replace(job_dir / "job.json")


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "filename": job["filename"],
        "format": job["format"],
        "status": job["status"],
        "stage": job.get("stage"),
        "progress": job.get("progress", 0),
        "attempt": job.get("attempt", 0),
        "max_attempts": MAX_ATTEMPTS,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "message": job.get("message"),
        "result_available": bool(job.get("result_path")),
        "evaluation": job.get("evaluation"),
        "error": job.get("error"),
    }


def _set_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS[job_id]
        job.update(changes)
        _save_job(job)
        return dict(job)


def _command(job_id: str, name: str, args: list[str], cwd: Path) -> int:
    """Run a pipeline stage to a per-job log without invoking a shell."""
    log_path = _job_dir(job_id) / f"{name}.log"
    stop_monitor = threading.Event()
    monitor: threading.Thread | None = None

    # OCR can spend several minutes inside one page. Tail its flushed page
    # markers so polling clients see activity instead of a frozen stage label.
    if name.endswith("_ocr"):
        stage_label = "Extracting original PDF" if name == "original_ocr" else "Checking redacted PDF"

        def monitor_ocr_log() -> None:
            offset = 0
            pattern = re.compile(r"^\[(\d+)/(\d+)\] starting page (\d+)")
            while not stop_monitor.is_set():
                try:
                    if log_path.is_file():
                        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                            stream.seek(offset)
                            lines = stream.readlines()
                            offset = stream.tell()
                        for line in lines:
                            match = pattern.match(line.strip())
                            if match:
                                current, total, page = match.groups()
                                _set_job(
                                    job_id,
                                    message=f"{stage_label} — page {page} ({current} of {total})",
                                )
                except (OSError, KeyError):
                    pass
                stop_monitor.wait(0.25)

        monitor = threading.Thread(target=monitor_ocr_log, name=f"{name}-status", daemon=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(args) + "\n\n")
        log.flush()
        if monitor is not None:
            monitor.start()
        try:
            result = subprocess.run(
                args, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                timeout=COMMAND_TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{name} exceeded the configured processing timeout") from exc
        finally:
            stop_monitor.set()
            if monitor is not None:
                monitor.join(timeout=2)
    return result.returncode


def _run_ocr(job_id: str, pdf_path: Path, output_dir: Path, log_prefix: str) -> tuple[Path, Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, "-m", "pdf_files.processing.cli", str(pdf_path),
        "--out", str(output_dir), "--device", OCR_DEVICE,
        "--fallback-engine", OCR_FALLBACK, "--workers", "0", "--strict",
    ]
    code = _command(job_id, f"{log_prefix}_ocr", args, BACKEND_DIR)
    if code not in (0, 4):
        raise RuntimeError(f"OCR failed in stage {log_prefix} (exit code {code})")

    stem = pdf_path.stem
    markdown = output_dir / f"{stem}.md"
    pages_json = output_dir / "_internal" / f"{stem}.pages.json"
    if not markdown.is_file() or not pages_json.is_file():
        raise RuntimeError(f"OCR stage {log_prefix} did not produce Markdown and page JSON")

    payload = json.loads(pages_json.read_text(encoding="utf-8"))
    needs_review = sum(
        1 for page in payload.get("pages", [])
        if page.get("stats", {}).get("needs_review") or page.get("stats", {}).get("needs_vlm")
    )
    return markdown, pages_json, code, needs_review


def _run_pdf_job(job_id: str) -> None:
    job_dir = _job_dir(job_id)
    original_pdf = job_dir / "input.pdf"
    try:
        _set_job(job_id, status="processing", stage="ocr_original", progress=3,
                 message="Extracting the original PDF")
        original_md, current_pages_json, original_code, original_review = _run_ocr(
            job_id, original_pdf, job_dir / "original_ocr", "original"
        )
        current_pdf = original_pdf
        last_redacted: Path | None = None
        last_attempt_summary: dict[str, Any] | None = None
        passed_any = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempt_dir = job_dir / f"attempt_{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            redacted_pdf = attempt_dir / f"redacted_attempt_{attempt}.pdf"
            _set_job(job_id, status="processing", stage="redacting", attempt=attempt,
                     progress=5 + (attempt - 1) * 30,
                     message=f"Redaction attempt {attempt} of {MAX_ATTEMPTS}")

            redact_args = [
                sys.executable, str(PROCESSING_DIR / "redact_pdf.py"),
                str(current_pdf), str(current_pages_json), str(redacted_pdf),
                "--figures", "hybrid", "--dates", "all",
            ]
            redact_code = _command(job_id, f"attempt_{attempt}_redact", redact_args, BACKEND_DIR)
            if not redacted_pdf.is_file():
                raise RuntimeError(f"Redaction attempt {attempt} did not produce a PDF")
            last_redacted = redacted_pdf

            _set_job(job_id, stage="ocr_redacted", progress=10 + (attempt - 1) * 30,
                     message=f"Checking redaction attempt {attempt}")
            after_md, after_pages_json, after_code, after_review = _run_ocr(
                job_id, redacted_pdf, attempt_dir / "redacted_ocr", f"attempt_{attempt}_after"
            )

            _set_job(job_id, stage="evaluating", progress=20 + (attempt - 1) * 30,
                     message=f"Checking PII removal for attempt {attempt}")
            eval_dir = attempt_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_args = [
                sys.executable, str(EVAL_DIR / "redaction_eval.py"),
                str(original_md), str(after_md), "--models", "none", "--no-cache",
                "--out-dir", str(eval_dir),
            ]
            eval_code = _command(job_id, f"attempt_{attempt}_evaluation", eval_args, EVAL_DIR)
            report_path = eval_dir / f"{after_md.stem}__vs__{original_md.stem}.eval.json"
            if eval_code != 0 or not report_path.is_file():
                raise RuntimeError(f"PII evaluation failed in attempt {attempt}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            score = report.get("score")
            critical = int(report.get("critical_leaks", 0))
            high = int(report.get("high_leaks", 0))
            passed = (
                isinstance(score, (int, float)) and float(score) > PASS_SCORE
                and critical == 0 and high == 0 and redact_code == 0
                and original_code == 0 and after_code == 0
                and original_review == 0 and after_review == 0
            )
            last_attempt_summary = {
                "attempt": attempt,
                "score": score,
                "passed": passed,
                "critical_leaks": critical,
                "high_leaks": high,
                "unique_values_still_visible": report.get("unique_still_leaking"),
                "original_pages_needing_review": original_review,
                "redacted_pages_needing_review": after_review,
                "redaction_verified": redact_code == 0,
                "evaluator": "regex+checksum only",
                "warnings": report.get("warnings", []),
            }
            _set_job(job_id, evaluation=last_attempt_summary, progress=25 + (attempt - 1) * 30,
                     message=("Quality threshold passed" if passed else
                              f"Attempt {attempt} did not meet all acceptance checks"))
            if passed:
                passed_any = True
                break

            # Keep the original Markdown fixed for scoring. Retry redaction on
            # the latest redacted PDF using OCR JSON made from that exact PDF.
            current_pdf = redacted_pdf
            current_pages_json = after_pages_json

        if last_redacted is None:
            raise RuntimeError("No redacted PDF was produced")
        final_pdf = job_dir / "result.pdf"
        shutil.copy2(last_redacted, final_pdf)
        evaluation = last_attempt_summary or {
            "attempt": 0, "score": None, "passed": False,
            "warnings": ["No completed evaluation is available."],
        }
        _set_job(
            job_id, status="completed" if passed_any else "completed_review",
            stage="done", progress=100, result_path=str(final_pdf),
            evaluation=evaluation,
            message=("Redacted PDF is ready." if passed_any else
                     "Final redacted PDF is ready, but it did not meet all acceptance checks."),
            error=None,
        )
    except Exception as exc:
        # Preserve a redacted artifact if a later retry/evaluation stage fails.
        existing = sorted(job_dir.glob("attempt_*/redacted_attempt_*.pdf"))
        if existing:
            final_pdf = job_dir / "result.pdf"
            shutil.copy2(existing[-1], final_pdf)
            _set_job(
                job_id, status="completed_review", stage="done", progress=100,
                result_path=str(final_pdf),
                message="A redacted PDF is available, but processing ended before a passing evaluation.",
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            _set_job(job_id, status="failed", stage="failed", progress=100,
                     message="The PDF could not be redacted.",
                     error=f"{type(exc).__name__}: {exc}")


def _run_office_job(job_id: str) -> None:
    """Run the README-documented Office redactor in its isolated Python 3.13 venv."""
    job_dir = _job_dir(job_id)
    with _LOCK:
        job = dict(_JOBS[job_id])
    suffix = "." + job["format"]
    input_path = job_dir / f"input{suffix}"
    output_path = job_dir / f"result{suffix}"
    summary_path = job_dir / "processing_summary.json"
    python_name = "python.exe" if os.name == "nt" else "python"
    python_path = WORD_PPTX_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin") / python_name

    try:
        if not python_path.is_file():
            raise RuntimeError(
                f"Word/PPTX environment is missing: {python_path}. Create the venv and install its requirements first."
            )
        missing_modules = _missing_office_modules(python_path)
        if missing_modules:
            requirements_path = WORD_PPTX_DIR / "requirements.txt"
            invocation = "& " if os.name == "nt" else ""
            install_command = (
                f'{invocation}"{python_path}" -m pip install -r "{requirements_path}"'
            )
            missing = ", ".join(missing_modules)
            raise RuntimeError(
                f"Word/PPTX dependencies are not installed in the dedicated environment "
                f"(missing: {missing}). Install them with: {install_command}"
            )
        _set_job(
            job_id, status="processing", stage="office_redaction", progress=10,
            message=f"Redacting {job['format'].upper()} content",
        )
        args = [
            str(python_path), str(WORD_PPTX_DIR / "process_document.py"),
            str(input_path), str(output_path), str(summary_path),
        ]
        code = _command(job_id, "office_redaction", args, WORD_PPTX_DIR)
        if code != 0:
            raise RuntimeError(f"{job['format'].upper()} redaction failed (exit code {code}); see office_redaction.log")
        if not output_path.is_file() or not _valid_office_file(output_path, suffix):
            raise RuntimeError("Office redactor did not produce a valid output document")
        if not summary_path.is_file():
            raise RuntimeError("Office redactor did not produce its processing summary")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("format") != job["format"] or summary.get("validation_passed") is not True:
            raise RuntimeError("Office output did not pass the redactor's validation")
        if summary.get("review_required") is not True:
            # Visual/media review is outside this upload-time text redactor's
            # validation guarantees; never label such a result as fully cleared.
            summary["review_required"] = True

        _set_job(
            job_id, status="completed_review", stage="done", progress=100,
            result_path=str(output_path), evaluation=summary, error=None,
            message=(f"Redacted {job['format'].upper()} is ready. Review it before distribution."),
        )
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        _set_job(
            job_id, status="failed", stage="failed", progress=100,
            result_path=None, message=f"The {job['format'].upper()} could not be redacted.",
            error=f"{type(exc).__name__}: {exc}",
        )


def _missing_office_modules(python_path: Path) -> list[str]:
    """Check required worker imports without importing heavy OCR/model code."""
    module_names = (
        "presidio_analyzer", "presidio_anonymizer", "docx", "pptx", "lxml",
        "spacy", "en_core_web_lg",
    )
    probe = (
        "import importlib.util, json; "
        f"names = {module_names!r}; "
        "print(json.dumps([name for name in names if importlib.util.find_spec(name) is None]))"
    )
    result = subprocess.run(
        [str(python_path), "-c", probe],
        cwd=str(WORD_PPTX_DIR), capture_output=True, text=True,
        timeout=30, check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        hint = detail[-1] if detail else "unknown environment check error"
        raise RuntimeError(f"Could not check the Word/PPTX environment: {hint}")
    try:
        missing = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Could not read the Word/PPTX environment check result") from exc
    if not isinstance(missing, list) or not all(isinstance(name, str) for name in missing):
        raise RuntimeError("The Word/PPTX environment check returned an invalid result")
    return missing


def _valid_office_file(path: Path, suffix: str) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return _DOC_MARKERS[suffix] in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "local-pii-redaction-api"}


async def _create_job(file: UploadFile, idempotency_key: str | None) -> dict[str, Any]:
    filename = Path((file.filename or "upload").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".pptx"):
        await file.close()
        raise HTTPException(status_code=415, detail="Upload a PDF, DOCX, or PPTX file.")

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    saved_path = job_dir / f"input{suffix}"
    total = 0
    try:
        with saved_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds the configured upload limit.")
                destination.write(chunk)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        await file.close()

    if suffix == ".pdf":
        if total < 5 or saved_path.read_bytes()[:5] != b"%PDF-":
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=415, detail="The uploaded file is not a valid PDF.")
        try:
            with pymupdf.open(saved_path) as document:
                page_count = len(document)
        except Exception as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=415, detail="The uploaded PDF could not be opened.") from exc
        if page_count == 0 or page_count > MAX_PDF_PAGES:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=413, detail=f"PDF must contain 1 to {MAX_PDF_PAGES} pages.")
        file_format, status, stage, message = (
            "pdf", "queued", "queued", "Queued for PDF redaction."
        )
        queued = True
    else:
        if not _valid_office_file(saved_path, suffix):
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=415, detail=f"The uploaded file is not a valid {suffix[1:].upper()} file.")
        file_format, status, stage, message = (
            suffix[1:], "queued", "queued", f"Queued for {suffix[1:].upper()} redaction."
        )
        queued = True

    job = {
        "job_id": job_id, "filename": filename, "format": file_format,
        "idempotency_key": idempotency_key,
        "status": status, "stage": stage, "progress": 0, "attempt": 0,
        "created_at": _now(), "updated_at": _now(), "message": message,
        "error": None, "result_path": None, "evaluation": None,
    }
    with _LOCK:
        _JOBS[job_id] = job
        _save_job(job)
    if queued:
        _WORKER.submit(_run_pdf_job if file_format == "pdf" else _run_office_job, job_id)
    return _public_job(job)


def _existing_idempotent_job(key: str) -> dict[str, Any] | None:
    with _LOCK:
        for job in _JOBS.values():
            if job.get("idempotency_key") == key:
                return job

    # Keep retries safe if the API process restarted after saving the job.
    for record in JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if job.get("idempotency_key") == key:
            with _LOCK:
                _JOBS[job["job_id"]] = job
            return job
    return None


@app.post("/api/v1/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, Any]:
    if idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", idempotency_key):
        await file.close()
        raise HTTPException(status_code=400, detail="Invalid Idempotency-Key header.")

    if idempotency_key is None:
        return await _create_job(file, None)

    # Serialize idempotent submissions so simultaneous retries with one key
    # cannot create separate jobs.
    async with _IDEMPOTENCY_LOCK:
        existing = _existing_idempotent_job(idempotency_key)
        if existing is not None:
            await file.close()
            return _public_job(existing)
        return await _create_job(file, idempotency_key)


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        path = _job_dir(job_id) / "job.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Job not found")
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("status") in ("queued", "processing"):
            job.update(
                status="failed", stage="failed", progress=100,
                message="The API restarted while this job was running. Submit the file again.",
                error="Interrupted by API restart.",
            )
            _save_job(job)
        with _LOCK:
            _JOBS[job_id] = job
    return _public_job(job)


@app.get("/api/v1/jobs/{job_id}/result")
def download_result(job_id: str) -> FileResponse:
    job = _JOBS.get(job_id)
    if job is None:
        get_job(job_id)
        job = _JOBS[job_id]
    result_path = Path(job.get("result_path") or "")
    if job.get("status") not in ("completed", "completed_review") or not result_path.is_file():
        raise HTTPException(status_code=409, detail="No redacted document is ready for this job.")
    file_format = job.get("format")
    if file_format not in _RESULT_MIMES:
        raise HTTPException(status_code=500, detail="The job has an unsupported result format.")
    filename = Path(job["filename"]).stem + "_redacted." + file_format
    return FileResponse(result_path, media_type=_RESULT_MIMES[file_format], filename=filename)


@app.get("/api/v1/jobs/{job_id}/report")
def get_report(job_id: str) -> dict[str, Any]:
    job = _JOBS.get(job_id)
    if job is None:
        get_job(job_id)
        job = _JOBS[job_id]
    if not job.get("evaluation"):
        raise HTTPException(status_code=409, detail="No evaluation report is ready for this job.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "evaluation": job["evaluation"],
        "message": job.get("message"),
    }
