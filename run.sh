#!/usr/bin/env bash
# Start CamTrack: builds the frontend if needed, then serves UI + API on http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "Creating Python virtual environment (.venv)…"
  $PY -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r backend/requirements.txt
fi
if [ ! -d frontend/dist ]; then
  echo "Building frontend…"
  (cd frontend && npm install --no-audit --no-fund && npm run build)
fi
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
echo "CamTrack running at http://$HOST:$PORT"
cd backend && exec ../.venv/bin/uvicorn camtrack.api:app --host "$HOST" --port "$PORT"
