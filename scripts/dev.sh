#!/usr/bin/env bash
# Start the DBS KYC document UI and Jina-only API on loopback.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_PORT="${DOCUMENT_API_PORT:-8001}"
FRONTEND_PORT="${DOCUMENT_UI_PORT:-3000}"
PYTHON="$ROOT/backend/.venv/bin/python"
PID_DIR="$ROOT/.dev-pids"
LOG_DIR="$ROOT/.dev-logs"
API_PID_FILE="$PID_DIR/documents-api.pid"
UI_PID_FILE="$PID_DIR/documents-ui.pid"
API_LOG="$LOG_DIR/documents-api.log"
UI_LOG="$LOG_DIR/documents-ui.log"

usage() {
  cat <<EOF
Usage: $0 {start|stop|restart|status}

start    Start the document API and UI in the background
stop     Stop both services
restart  Stop then start both services
status   Show service status
EOF
}

ensure_runtime() {
  if [[ ! -x "$PYTHON" ]]; then
    echo 'Create backend/.venv and install backend/requirements.txt first.' >&2
    exit 1
  fi
  if [[ ! -x "$ROOT/frontend/node_modules/.bin/vite" ]]; then
    echo 'Install frontend dependencies first: cd frontend && npm install' >&2
    exit 1
  fi
  mkdir -p "$PID_DIR" "$LOG_DIR"
}

pid_alive() {
  local file="$1"
  [[ -f "$file" ]] && kill -0 "$(cat "$file")" 2>/dev/null
}

stop_pid() {
  local name="$1"
  local file="$2"
  if pid_alive "$file"; then
    local pid
    pid="$(cat "$file")"
    echo "Stopping $name ($pid)"
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

spawn_detached() {
  local pid_file="$1"
  local log_file="$2"
  local cwd="$3"
  shift 3
  "$PYTHON" - "$pid_file" "$log_file" "$cwd" "$@" <<'PY'
import os
import subprocess
import sys

pid_file, log_file, cwd, *cmd = sys.argv[1:]
os.makedirs(os.path.dirname(pid_file), exist_ok=True)
os.makedirs(os.path.dirname(log_file), exist_ok=True)
log = open(log_file, "ab", buffering=0)
process = subprocess.Popen(
    cmd,
    cwd=cwd,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open(pid_file, "w", encoding="utf-8") as handle:
    handle.write(str(process.pid))
PY
}

start() {
  ensure_runtime
  if pid_alive "$API_PID_FILE" || pid_alive "$UI_PID_FILE"; then
    echo "Document demo already appears to be running."
    status
    exit 0
  fi
  rm -f "$API_PID_FILE" "$UI_PID_FILE"
  spawn_detached "$API_PID_FILE" "$API_LOG" "$ROOT/backend" \
    "$PYTHON" -m uvicorn app.document_app:app --host 127.0.0.1 --port "$BACKEND_PORT"
  spawn_detached "$UI_PID_FILE" "$UI_LOG" "$ROOT/frontend" \
    env VITE_BACKEND_PORT="$BACKEND_PORT" ./node_modules/.bin/vite --host 127.0.0.1 --strictPort --port "$FRONTEND_PORT"
  sleep 1
  if ! pid_alive "$API_PID_FILE" || ! pid_alive "$UI_PID_FILE"; then
    echo "Document demo failed to start. Recent logs:" >&2
    tail -n 40 "$API_LOG" "$UI_LOG" >&2 || true
    stop
    exit 1
  fi
  echo "Document UI: http://127.0.0.1:$FRONTEND_PORT"
  echo "Document API: http://127.0.0.1:$BACKEND_PORT"
  echo "Logs: $API_LOG and $UI_LOG"
}

stop() {
  stop_pid "document UI" "$UI_PID_FILE"
  stop_pid "document API" "$API_PID_FILE"
}

status() {
  if pid_alive "$API_PID_FILE"; then
    echo "Document API running on 127.0.0.1:$BACKEND_PORT (pid $(cat "$API_PID_FILE"))"
  else
    echo "Document API stopped"
    rm -f "$API_PID_FILE"
  fi
  if pid_alive "$UI_PID_FILE"; then
    echo "Document UI running on 127.0.0.1:$FRONTEND_PORT (pid $(cat "$UI_PID_FILE"))"
  else
    echo "Document UI stopped"
    rm -f "$UI_PID_FILE"
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart)
    stop
    start
    ;;
  status) status ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
