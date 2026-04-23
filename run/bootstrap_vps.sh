#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Bootstrap complete."
echo "Next:"
echo "1. Copy .env to this server and keep USE_SERVER=1"
echo "2. Run: PYTHONPATH=. TELEGRAM_ENABLED=0 .venv/bin/python run/check_okx_paper_ready.py"
echo "3. Start with PM2: pm2 start ecosystem.paper.config.js"
