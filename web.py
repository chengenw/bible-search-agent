"""Run the local Bible search web app with `uv run python web.py`."""

import argparse
import json
from functools import cache
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from src.agent import run_agent
from src.config import VERSIONS
from src.vector_store import get_collection

WEB_DIR = Path(__file__).with_name("web")
ASSETS = {
    "/": ("index.html", "text/html"),
    "/search.css": ("search.css", "text/css"),
    "/search.js": ("search.js", "text/javascript"),
}
MAX_QUESTION_LENGTH = 1000
MAX_REQUEST_BYTES = 16 * 1024
DEFAULT_VERSION = "KJV"
collection_lock = Lock()


@cache
def bible_index(version):
    return get_collection(version)


def translation_options():
    options = []
    for version in VERSIONS:
        name = escape(version)
        selected = " selected" if version == DEFAULT_VERSION else ""
        options.append(f'<option value="{name}"{selected}>{name}</option>')
    return "\n".join(options)


class SearchHandler(BaseHTTPRequestHandler):
    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, **data):
        self.send_body(status, "application/json", json.dumps(data).encode("utf-8"))

    def do_GET(self):
        asset = ASSETS.get(self.path.split("?", 1)[0])
        if asset is None:
            self.send_error(404)
            return
        filename, content_type = asset
        body = (WEB_DIR / filename).read_bytes()
        if filename == "index.html":
            body = body.replace(b"<!-- translation options -->", translation_options().encode("utf-8"))
        self.send_body(200, content_type, body)

    def do_POST(self):
        if self.path != "/api/search":
            self.send_error(404)
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, error="Send an application/json request.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_REQUEST_BYTES:
                self.send_json(413, error="The request is too large.")
                return
            payload = json.loads(self.rfile.read(max(0, length)))
        except (ValueError, RecursionError):
            self.send_json(400, error="Send a valid JSON request.")
            return

        value = payload.get("question") if isinstance(payload, dict) else None
        question = value.strip() if isinstance(value, str) else ""
        if not 0 < len(question) <= MAX_QUESTION_LENGTH:
            self.send_json(400, error=f"Enter a question of 1 to {MAX_QUESTION_LENGTH:,} characters.")
            return

        version = payload.get("version", DEFAULT_VERSION)
        if not isinstance(version, str) or version not in VERSIONS:
            self.send_json(400, error="Choose a supported Bible translation.")
            return

        try:
            # Keep simultaneous first requests from both trying to build the index.
            with collection_lock:
                collection = bible_index(version)
            answer, trace = run_agent(question, collection, return_trace=True)
        except Exception as error:
            self.log_error("Search failed: %s", error)
            self.send_json(502, error="Search failed. Check the server output and try again.")
            return

        sources = {}
        for search in trace:
            for verse in search["results"][:3]:
                sources.setdefault(verse["reference"], verse["text"])
        self.send_json(200, answer=answer, sources=sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), SearchHandler) as server:
        print(f"Bible Search is ready at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
