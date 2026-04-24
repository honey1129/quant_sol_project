#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="${APP_NAME:-quant_okx_paper}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/live_trading.log}"
STATE_FILE="${STATE_FILE:-$PROJECT_ROOT/logs/live_trading_state.json}"
LINES="${LINES:-60}"
FOLLOW=0

PATTERN='交易环境校验完成|paper_ready_ok|Live trading monitor started|已恢复最近处理 bar|新bar=|心跳:|执行开仓|执行平仓|执行调仓|无明显信号或目标为0|实盘循环异常|未成交|同时多空持仓'

usage() {
  cat <<'EOF'
Usage:
  bash run/strategy_status_summary.sh [--follow] [--lines N]

Options:
  --follow    Follow strategy status in real time
  --lines N   Show the latest N matching lines, default: 60
  --help      Show this help message

Environment overrides:
  APP_NAME    PM2 app name, default: quant_okx_paper
  LOG_FILE    Strategy log path, default: <project>/logs/live_trading.log
  STATE_FILE  Runtime state path, default: <project>/logs/live_trading_state.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow)
      FOLLOW=1
      shift
      ;;
    --lines)
      [[ $# -ge 2 ]] || {
        echo "[status] ERROR: --lines requires a value" >&2
        exit 1
      }
      LINES="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[status] ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

print_section() {
  printf '\n[%s] %s\n' "status" "$1"
}

print_pm2_summary() {
  if ! command -v pm2 >/dev/null 2>&1; then
    echo "pm2 not installed"
    return
  fi

  pm2 describe "$APP_NAME" 2>/dev/null | grep -E "status|restarts|uptime|script path|exec cwd" || \
    echo "pm2 app $APP_NAME not found"
}

print_state_summary() {
  if [[ ! -f "$STATE_FILE" ]]; then
    echo "state file not found: $STATE_FILE"
    return
  fi

  cat "$STATE_FILE"
}

print_recent_status_lines() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "log file not found: $LOG_FILE"
    return
  fi

  grep -E "$PATTERN" "$LOG_FILE" | tail -n "$LINES" || true
}

follow_status_lines() {
  if [[ ! -f "$LOG_FILE" ]]; then
    echo "log file not found: $LOG_FILE"
    exit 1
  fi

  tail -f "$LOG_FILE" | grep -E --line-buffered "$PATTERN"
}

print_section "Project"
echo "root=$PROJECT_ROOT"
echo "app=$APP_NAME"
echo "log=$LOG_FILE"
echo "state=$STATE_FILE"

print_section "PM2"
print_pm2_summary

print_section "State"
print_state_summary

print_section "Recent Strategy Status"
print_recent_status_lines

if [[ "$FOLLOW" -eq 1 ]]; then
  print_section "Following"
  follow_status_lines
fi
