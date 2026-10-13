import html
import os
import uuid
import streamlit as st
from api_client import (
    ServerError,
    UnsupportedFormatError,
    download_result,
    get_job,
    get_report,
    submit_job,
)
from validation import ALLOWED_LABEL, InvalidFormat, classify

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="Document Processor",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

@keyframes shake {
    10%, 90% { transform: translateX(-1px); }
    20%, 80% { transform: translateX(2px); }
    30%, 50%, 70% { transform: translateX(-4px); }
    40%, 60% { transform: translateX(4px); }
}

@keyframes slidein {
    from { opacity: 0; transform: translateY(-8px); }
    to { opacity: 1; transform: none; }
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.45); }
    50% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
}

.notice {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    padding: 1rem 1.1rem;
    margin: 0.75rem 0;
    border-radius: 0.6rem;
    border-left: 6px solid;
    font-size: 0.95rem;
    line-height: 1.45;
}

.notice-icon {
    font-size: 1.5rem;
    line-height: 1;
}

.notice-format {
    background: #fff7ed;
    border-color: #f97316;
    color: #7c2d12;
    animation: shake 0.5s;
}

.notice-server {
    background: #fef2f2;
    border-color: #dc2626;
    color: #7f1d1d;
    animation: slidein 0.35s ease-out, pulse 1.2s ease-out 2;
}

.file-info-badge {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    padding: 0.65rem 1rem;
    margin-bottom: 0.75rem;
    color: #1e293b;
    font-size: 0.95rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

@media (prefers-reduced-motion: reduce) {
    .notice {
        animation: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "status": "idle",
    "file_sig": None,
    "doc": None,
    "job_id": None,
    "idempotency_key": None,
    "job": None,
    "report": None,
    "result": None,
    "error": None,
    "poll_error": None,
    "result_error": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def start_processing():
    st.session_state.update(
        status="submitting",
        idempotency_key=uuid.uuid4().hex,
        error=None,
        poll_error=None,
        result_error=None,
        job_id=None,
        job=None,
        report=None,
        result=None,
    )


def resume_submission():
    """Retry a submission with the same key after an uncertain response."""
    st.session_state.update(status="submitting", error=None)


def notice(kind, title, message, icon):
    st.markdown(
        f'<div class="notice notice-{kind}" role="alert">'
        f'<span class="notice-icon">{icon}</span>'
        f'<div><strong>{html.escape(title)}</strong><br>{html.escape(message)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


st.title("Document Processor")
st.caption(f"Accepted formats: {ALLOWED_LABEL}")

busy = st.session_state.status in {"submitting", "queued", "processing"}
locked = busy or st.session_state.status == "submit_uncertain"
uploaded = st.file_uploader("Drag and drop a file here, or choose one", disabled=locked)

if uploaded is None:
    # Keep state if Streamlit reruns while a previously submitted job is active.
    if not busy:
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
    st.stop()

# New file selected -> reset and classify once
sig = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
if sig != st.session_state.file_sig:
    st.session_state.update({**DEFAULTS, "file_sig": sig})
    try:
        st.session_state.doc = classify(uploaded.name, uploaded)
    except InvalidFormat as exc:
        st.session_state.error = {"kind": "format", "message": str(exc)}

doc = st.session_state.doc
if doc:
    st.markdown(
        f'<div class="file-info-badge">📄 <strong>{html.escape(uploaded.name)}</strong> · {doc.label} · {uploaded.size / 1024:,.0f} KB</div>',
        unsafe_allow_html=True,
    )
    st.button(
        "Retry submission safely" if st.session_state.status == "submit_uncertain" else "Process document",
        type="primary",
        on_click=resume_submission if st.session_state.status == "submit_uncertain" else start_processing,
        disabled=busy,
        key="process_document_button",
    )

if st.session_state.status == "submitting":
    with st.spinner("Submitting your document…"):
        try:
            submitted = submit_job(
                API_BASE_URL,
                uploaded.name,
                uploaded.getvalue(),
                mime=doc.mime,
                idempotency_key=st.session_state.idempotency_key,
            )
            st.session_state.job_id = submitted["job_id"]
            st.session_state.job = submitted
            st.session_state.status = submitted.get("status", "queued")
        except UnsupportedFormatError as exc:
            st.session_state.update(
                status="idle", error={"kind": "format", "message": str(exc)}
            )
        except ServerError as exc:
            st.session_state.update(
                status="submit_uncertain",
                error={"kind": "uncertain", "message": str(exc)},
            )
    st.rerun()

err = st.session_state.error
if err and err["kind"] == "format":
    notice("format", "Unsupported file format", err["message"], "🚫")
elif err:
    if err["kind"] == "uncertain":
        notice(
            "server",
            "Submission status is uncertain",
            f"{err['message']} Use “Retry submission safely” to check or resubmit the same job without creating a duplicate.",
            "⚠️",
        )
    else:
        notice("server", "Something went wrong on the server", err["message"], "⚠️")


TERMINAL_STATES = {"completed", "completed_review", "failed", "unsupported"}


def _format_score(value):
    return f"{value:.1f}" if isinstance(value, (int, float)) else "Not available"


@st.fragment(run_every="2s")
def render_job():
    job_id = st.session_state.get("job_id")
    if not job_id:
        return

    current = st.session_state.get("job") or {}
    if current.get("status") not in TERMINAL_STATES:
        try:
            previous_status = current.get("status")
            current = get_job(API_BASE_URL, job_id)
            st.session_state.job = current
            st.session_state.status = current["status"]
            st.session_state.poll_error = None
            if current["status"] in TERMINAL_STATES and previous_status not in TERMINAL_STATES:
                # Re-run the full page once so the uploader is enabled again.
                st.rerun()
        except ServerError as exc:
            # A transient polling error must not discard the job ID or submit again.
            st.session_state.poll_error = str(exc)
            st.warning("Could not refresh job status. Retrying automatically.")
            st.caption(str(exc))
            return

    status = current.get("status")
    message = current.get("message") or "Processing your document."
    if status in {"queued", "processing"}:
        progress = current.get("progress", 0)
        if not isinstance(progress, (int, float)):
            progress = 0
        progress = max(0, min(100, int(progress)))
        st.progress(progress, text=message)
        attempt = current.get("attempt")
        max_attempts = current.get("max_attempts")
        if isinstance(attempt, int) and isinstance(max_attempts, int) and max_attempts > 0:
            st.caption(f"Redaction attempt {attempt} of {max_attempts}")
        return

    if status not in TERMINAL_STATES:
        st.warning("The server returned an unrecognized job state. Status checks will continue.")
        return

    if status in {"completed", "completed_review"}:
        file_format = current.get("format", "pdf")
        # Download and report are fetched once and retained in session state.
        if st.session_state.result is None and not st.session_state.result_error:
            try:
                st.session_state.result = download_result(API_BASE_URL, job_id, file_format)
            except ServerError as exc:
                st.session_state.result_error = str(exc)
        if st.session_state.report is None:
            try:
                st.session_state.report = get_report(API_BASE_URL, job_id)
            except ServerError:
                # Job status already carries the summary; keep that if the report
                # endpoint is temporarily unavailable.
                st.session_state.report = {
                    "evaluation": current.get("evaluation"),
                    "message": message,
                }

        if status == "completed":
            st.success("Redaction checks passed. Your processed PDF is ready.")
        elif file_format in {"docx", "pptx"}:
            st.warning(f"The redacted {file_format.upper()} is ready. Review it before distribution.")
        else:
            st.warning("A redacted PDF is available, but one or more checks need review.")

        evaluation = (st.session_state.report or {}).get("evaluation") or current.get("evaluation") or {}
        if evaluation and file_format == "pdf":
            score = _format_score(evaluation.get("score"))
            passed = evaluation.get("passed")
            st.markdown(f"**Evaluation score:** {score} / 100  ·  **Checks passed:** {'Yes' if passed else 'No'}")
            st.caption(
                "Critical leaks: "
                f"{evaluation.get('critical_leaks', 'not reported')}  ·  "
                "High leaks: "
                f"{evaluation.get('high_leaks', 'not reported')}  ·  "
                "Pages needing OCR review (original / redacted): "
                f"{evaluation.get('original_pages_needing_review', 'not reported')} / "
                f"{evaluation.get('redacted_pages_needing_review', 'not reported')}"
            )
            warnings = evaluation.get("warnings") or []
            if warnings:
                with st.expander("Review warnings"):
                    for warning in warnings:
                        st.write(f"• {warning}")
        elif evaluation:
            st.markdown("**Processing summary**")
            st.caption(
                f"Text blocks: {evaluation.get('total_blocks', 'not reported')}  ·  "
                f"PII replacements: {evaluation.get('redacted_findings', 'not reported')}  ·  "
                f"Skipped findings: {evaluation.get('skipped_findings_count', 'not reported')}"
            )
            reasons = evaluation.get("review_reasons") or []
            if reasons:
                with st.expander("Why review is required"):
                    for reason in reasons:
                        st.write(f"• {reason}")

        result = st.session_state.result
        if result:
            st.download_button(
                f"Download {result.filename}",
                data=result.data,
                file_name=result.filename,
                mime=result.mime,
                type="primary",
                key=f"download_result_{job_id}",
            )
        elif st.session_state.result_error:
            st.error(f"The result is ready, but could not be downloaded: {st.session_state.result_error}")
            if st.button("Retry result download", key=f"retry_result_{job_id}"):
                st.session_state.result_error = None
                st.rerun(scope="fragment")
        return

    # Keep non-PDF/unavailable processor states generic in the UI.
    error = current.get("error") or message
    st.error(f"Processing could not be completed: {error}")
    if current.get("progress", 0) == 100:
        st.button(
            "Try again",
            type="primary",
            on_click=start_processing,
            key=f"retry_job_{job_id}",
        )


if st.session_state.job_id:
    render_job()
