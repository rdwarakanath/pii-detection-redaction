"""Small local mock for the asynchronous PDF jobs API."""

import cgi
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


JOB_ID = "0123456789abcdef0123456789abcdef"
_RESULT = b"%PDF-1.4 mock redacted result"
_EVALUATION = {
    "attempt": 1,
    "score": 96.0,
    "passed": True,
    "critical_leaks": 0,
    "high_leaks": 0,
    "original_pages_needing_review": 0,
    "redacted_pages_needing_review": 0,
    "warnings": [],
}
_JOB = {
    "job_id": JOB_ID,
    "filename": "mock.pdf",
    "format": "pdf",
    "status": "completed",
    "stage": "done",
    "progress": 100,
    "attempt": 1,
    "max_attempts": 3,
    "message": "Mock redacted PDF is ready.",
    "result_available": True,
    "evaluation": _EVALUATION,
    "error": None,
}


class MockApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/v1/jobs":
            self._send_json(404, {"detail": "Not found"})
            return

        params = parse_qs(parsed.query)
        status_val = self.headers.get("X-Mock-Status") or params.get("status", [None])[0]
        if status_val:
            status = int(status_val)
            if status in (415, 500):
                self._send_json(status, {"detail": "Simulated mock server failure."})
                return

        content_type = self.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_json(400, {"detail": "Expected multipart/form-data."})
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )
        if "file" not in form:
            self._send_json(400, {"detail": "Missing file."})
            return

        file_item = form["file"]
        filename = file_item.filename or "mock.pdf"
        if not filename.lower().endswith(".pdf"):
            self._send_json(415, {"detail": "The mock processor only supports PDF."})
            return

        delay = self.headers.get("X-Mock-Delay") or params.get("delay", [None])[0]
        if delay:
            try:
                time.sleep(float(delay))
            except ValueError:
                pass
        job = {**_JOB, "filename": filename}
        self._send_json(202, job)

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")
        base = f"/api/v1/jobs/{JOB_ID}"
        if path == base:
            self._send_json(200, _JOB)
        elif path == f"{base}/report":
            self._send_json(
                200,
                {"job_id": JOB_ID, "status": "completed", "evaluation": _EVALUATION,
                 "message": _JOB["message"]},
            )
        elif path == f"{base}/result":
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="mock_redacted.pdf"')
            self.send_header("Content-Length", str(len(_RESULT)))
            self.end_headers()
            self.wfile.write(_RESULT)
        else:
            self._send_json(404, {"detail": "Job not found"})

    def log_message(self, format, *args):
        print(f"[Mock Server] {self.address_string()} - {format % args}")


def run_mock_server(host="127.0.0.1", port=8000):
    server = HTTPServer((host, port), MockApiHandler)
    print(f"Mock document processor server running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_mock_server()
