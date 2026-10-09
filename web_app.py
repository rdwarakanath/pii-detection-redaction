"""Local browser interface for the offline PII document scanner."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from urllib.parse import unquote

import pipeline

HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".pptx",
                        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PAGE = Path(__file__).with_name("index.html")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/":
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        try:
            body = PAGE.read_bytes()
        except OSError:
            self._send(500, "text/plain; charset=utf-8", b"Could not read index.html")
            return
        self._send(200, "text/html; charset=utf-8", body)

    def do_POST(self):
        if self.path != "/api/process":
            self._send(404, "application/json; charset=utf-8", b'{"error":"Not found"}')
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self._send(413, "application/json; charset=utf-8", b'{"error":"Upload must be under 25 MB."}')
            return

        raw = self.rfile.read(length)
        filename = Path(unquote(self.headers.get("X-Filename", "uploaded.txt"))).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            self._send(415, "application/json; charset=utf-8",
                      b'{"error":"Supported files: TXT, MD, CSV, PDF, DOCX, PPTX, and common image formats."}')
            return

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="pii_upload_", suffix=suffix, delete=False) as temp:
                temp.write(raw)
                temp_path = temp.name
            detected_type = pipeline.detect_type(temp_path)
            if detected_type == "unknown":
                raise ValueError("The uploaded file could not be identified as a supported type.")
            result = pipeline.process(temp_path, display_name=filename)
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(400, "application/json; charset=utf-8", body)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    print(f"Offline scanner ready at http://{HOST}:{PORT}. Press Ctrl+C to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
