import json
import os
import re
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from utils.runtime_dashboard import (
    load_runtime_dashboard_history,
    load_runtime_dashboard_status,
)
from utils.utils import BASE_DIR, LOG_FILE


DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8787"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_HISTORY_LIMIT = int(os.getenv("DASHBOARD_HISTORY_LIMIT", "240"))
DASHBOARD_EVENT_LIMIT = int(os.getenv("DASHBOARD_EVENT_LIMIT", "24"))

FRONTEND_ROOT = os.path.join(BASE_DIR, "dashboard-ui")
FRONTEND_DIST_ROOT = os.path.join(FRONTEND_ROOT, "dist")

EVENT_PATTERN = re.compile(
    r"交易环境校验完成|paper_ready_ok|Live trading monitor started|"
    r"已恢复最近处理 bar|新bar=|心跳:|执行开仓|执行平仓|执行调仓|"
    r"无明显信号或目标为0|实盘循环异常|未成交|同时多空持仓"
)


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_recent_strategy_events(log_path, limit=DASHBOARD_EVENT_LIMIT):
    if not os.path.exists(log_path):
        return []

    recent_lines = deque(maxlen=max(limit * 6, 120))
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if EVENT_PATTERN.search(line):
                    recent_lines.append(line)
    except Exception:
        return []

    return list(recent_lines)[-limit:]


def build_dashboard_bundle():
    status = load_runtime_dashboard_status()
    history = load_runtime_dashboard_history()
    frontend_index = os.path.join(FRONTEND_DIST_ROOT, "index.html")

    return {
        "generated_at": utc_now_iso(),
        "frontend_built": os.path.isfile(frontend_index),
        "status": status,
        "history": history[-DASHBOARD_HISTORY_LIMIT:],
        "recent_events": read_recent_strategy_events(LOG_FILE, limit=DASHBOARD_EVENT_LIMIT),
    }


def _write_json_response(handler, payload, status=HTTPStatus.OK):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        static_root = FRONTEND_DIST_ROOT if os.path.isfile(os.path.join(FRONTEND_DIST_ROOT, "index.html")) else BASE_DIR
        super().__init__(*args, directory=static_root, **kwargs)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/dashboard":
            return _write_json_response(self, build_dashboard_bundle())

        if path == "/api/health":
            return _write_json_response(self, {"ok": True, "generated_at": utc_now_iso()})

        frontend_ready = os.path.isfile(os.path.join(FRONTEND_DIST_ROOT, "index.html"))
        if not frontend_ready:
            body = (
                "<html><body style='font-family: sans-serif; padding: 24px;'>"
                "<h1>Dashboard UI not built yet</h1>"
                "<p>Run <code>cd dashboard-ui && npm install && npm run build</code> first, "
                "then restart <code>python -m run.dashboard_server</code>.</p>"
                "<p>API endpoints are already available: <code>/api/dashboard</code> and <code>/api/health</code>.</p>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path != "/" and not os.path.exists(os.path.join(FRONTEND_DIST_ROOT, path.lstrip("/"))):
            self.path = "/index.html"

        return super().do_GET()


def main():
    server = ThreadingHTTPServer((DASHBOARD_HOST, DASHBOARD_PORT), DashboardRequestHandler)
    print(f"dashboard_server listening on http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
