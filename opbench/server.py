"""Same-origin HTTP API + static frontend. Standard library, no install required."""
import argparse
import hmac
import json
import mimetypes
import os
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from . import service
from .catalog import CATEGORIES, expand_template

ROOT = Path(__file__).resolve().parent.parent
MAX_BODY = 32 * 1024 * 1024


def make_handler(db_path, token=""):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpBench/1.0"

        def respond(self, value, status=200, content_type="application/json; charset=utf-8"):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.respond({"error": "API token required"}, 401)
                return False
            return True

        def do_GET(self):
            try:
                self.get()
            except KeyError as exc:
                self.respond({"error": str(exc)}, 404)
            except (ValueError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception:
                logging.exception("GET request failed")
                self.respond({"error": "internal server error"}, 500)

        def get(self):
            url = urlsplit(self.path)
            path, query = url.path, parse_qs(url.query)
            if path == "/api/health":
                return self.respond({"status": "ok", "schema_version": 1, "auth_required": bool(token)})
            if path.startswith("/api/") and not self.authorized():
                return
            if path == "/api/runs":
                return self.respond(service.list_runs(db_path, query.get("archived", ["0"])[0] == "1"))
            if path.startswith("/api/runs/"):
                return self.respond(service.export_report(db_path, path.split("/")[3]))
            if path == "/api/compare":
                return self.respond(service.compare(db_path, query.get("left", [""])[0], query.get("right", [""])[0]))
            if path == "/api/catalog":
                return self.respond(CATEGORIES)
            if path == "/api/devices":
                with service.connect(db_path) as db:
                    rows = db.execute("SELECT id,metadata FROM devices ORDER BY id").fetchall()
                return self.respond([{ "id": r[0], **json.loads(r[1])} for r in rows])
            if path == "/api/templates":
                return self.respond([{"id": p.stem, **json.loads(p.read_text(encoding="utf-8"))} for p in sorted((ROOT/"templates").glob("*.json"))])
            if path.startswith("/api/"):
                return self.respond({"error": "not found"}, 404)
            file = (ROOT/"web"/("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if not file.is_relative_to((ROOT/"web").resolve()) or not file.is_file():
                return self.respond({"error": "not found"}, 404)
            return self.respond(file.read_bytes(), content_type=mimetypes.guess_type(str(file))[0] or "application/octet-stream")

        def do_POST(self):
            try:
                if not self.authorized():
                    return
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    return self.respond({"error": "cross-origin writes are not allowed"}, 403)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self.respond({"error": "Content-Type must be application/json"}, 415)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    return self.respond({"error": "body must be 1..32 MiB"}, 413)
                body = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/import":
                    return self.respond(service.import_report(db_path, body), 201)
                if path == "/api/templates/validate":
                    cases = expand_template(body)
                    return self.respond({"count": len(cases), "cases": cases})
                if path.startswith("/api/runs/") and path.endswith("/archive"):
                    if type(body.get("archived")) is not bool:
                        raise ValueError("archived must be boolean")
                    service.archive_run(db_path, path.split("/")[3], body["archived"])
                    return self.respond({"ok": True})
                self.respond({"error": "not found"}, 404)
            except KeyError as exc:
                self.respond({"error": str(exc)}, 404)
            except (ValueError, TypeError, AttributeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except Exception:
                logging.exception("POST request failed")
                self.respond({"error": "internal server error"}, 500)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--db", default="data/opbench.sqlite3")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("OPBENCH_API_TOKEN", "")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("non-loopback binding requires OPBENCH_API_TOKEN; deploy behind TLS reverse proxy")
    service.initialize(args.db)
    if args.demo and not service.list_runs(args.db):
        from .demo import reports
        for report in reports():
            service.import_report(args.db, report)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.db, token))
    print(f"OpBench listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
