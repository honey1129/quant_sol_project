#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/root/quant_sol_project}"

SYNC_ENV=0
SYNC_MODELS=0
SKIP_DEPLOY=0

DEPLOY_ARGS=()

log() {
  printf '[remote-deploy] %s\n' "$*"
}

fail() {
  printf '[remote-deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash run/deploy_remote_vps.sh --host <vps-ip-or-host> [options]

Options:
  --host <host>         VPS host or IP, required unless REMOTE_HOST is set
  --user <user>         SSH user, default: root
  --port <port>         SSH port, default: 22
  --remote-dir <path>   Remote project directory, default: /root/quant_sol_project
  --sync-env            Also upload local .env to the VPS
  --sync-models         Also upload local models/ directory to the VPS
  --skip-deploy         Only sync files, do not execute remote deploy_paper_vps.sh
  --skip-check          Forwarded to remote deploy_paper_vps.sh
  --skip-start          Forwarded to remote deploy_paper_vps.sh
  --help                Show this help

Environment overrides:
  REMOTE_HOST
  REMOTE_USER
  REMOTE_PORT
  REMOTE_DIR

Examples:
  bash run/deploy_remote_vps.sh --host 185.214.135.24
  bash run/deploy_remote_vps.sh --host 185.214.135.24 --sync-env --sync-models
  bash run/deploy_remote_vps.sh --host 185.214.135.24 --skip-check

Notes:
  - This script syncs source code via rsync, then runs bash run/deploy_paper_vps.sh remotely.
  - By default it does NOT overwrite the VPS .env or models/. Use --sync-env / --sync-models when needed.
  - Password-auth SSH is fine; ssh/rsync may prompt you interactively.
EOF
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Missing required command: $cmd"
}

shell_quote() {
  printf '%q' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || fail "--host requires a value"
      REMOTE_HOST="$2"
      shift 2
      ;;
    --user)
      [[ $# -ge 2 ]] || fail "--user requires a value"
      REMOTE_USER="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || fail "--port requires a value"
      REMOTE_PORT="$2"
      shift 2
      ;;
    --remote-dir)
      [[ $# -ge 2 ]] || fail "--remote-dir requires a value"
      REMOTE_DIR="$2"
      shift 2
      ;;
    --sync-env)
      SYNC_ENV=1
      shift
      ;;
    --sync-models)
      SYNC_MODELS=1
      shift
      ;;
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --skip-check|--skip-start)
      DEPLOY_ARGS+=("$1")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$REMOTE_HOST" ]] || fail "Missing --host <vps-ip-or-host>"

require_command ssh
require_command rsync

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_BASE=(ssh -p "$REMOTE_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
RSYNC_RSH="ssh -p $REMOTE_PORT -o ServerAliveInterval=30 -o ServerAliveCountMax=3"

run_ssh() {
  "${SSH_BASE[@]}" "$SSH_TARGET" "$@"
}

ensure_remote_dir() {
  log "Ensuring remote directory exists: $REMOTE_DIR"
  run_ssh "mkdir -p $(shell_quote "$REMOTE_DIR")"
}

sync_code() {
  log "Syncing project source to $SSH_TARGET:$REMOTE_DIR"
  RSYNC_RSH="$RSYNC_RSH" rsync -az \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'venv/' \
    --exclude 'env/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.idea/' \
    --exclude '.vscode/' \
    --exclude '.pytest_cache/' \
    --exclude '.cache/' \
    --exclude 'logs/' \
    --exclude 'dashboard-ui/node_modules/' \
    --exclude 'dashboard-ui/dist/' \
    --exclude '.env' \
    --exclude 'models/' \
    "$PROJECT_ROOT/" "$SSH_TARGET:$REMOTE_DIR/"
}

sync_env_if_needed() {
  if [[ "$SYNC_ENV" -ne 1 ]]; then
    return
  fi

  [[ -f "$PROJECT_ROOT/.env" ]] || fail "Local .env not found: $PROJECT_ROOT/.env"
  log "Uploading local .env"
  RSYNC_RSH="$RSYNC_RSH" rsync -az "$PROJECT_ROOT/.env" "$SSH_TARGET:$REMOTE_DIR/.env"
}

sync_models_if_needed() {
  if [[ "$SYNC_MODELS" -ne 1 ]]; then
    return
  fi

  [[ -d "$PROJECT_ROOT/models" ]] || fail "Local models directory not found: $PROJECT_ROOT/models"
  log "Uploading local models/"
  run_ssh "mkdir -p $(shell_quote "$REMOTE_DIR/models")"
  RSYNC_RSH="$RSYNC_RSH" rsync -az "$PROJECT_ROOT/models/" "$SSH_TARGET:$REMOTE_DIR/models/"
}

run_remote_deploy() {
  if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
    log "Skipping remote deploy step"
    return
  fi

  local remote_cmd
  remote_cmd="cd $(shell_quote "$REMOTE_DIR") && bash run/deploy_paper_vps.sh"
  if [[ ${#DEPLOY_ARGS[@]} -gt 0 ]]; then
    local arg
    for arg in "${DEPLOY_ARGS[@]}"; do
      remote_cmd+=" $(shell_quote "$arg")"
    done
  fi

  log "Running remote deploy script"
  run_ssh "$remote_cmd"
}

print_summary() {
  cat <<EOF

[remote-deploy] Done.
[remote-deploy] Remote target: $SSH_TARGET
[remote-deploy] Remote dir:    $REMOTE_DIR

[remote-deploy] Useful follow-up commands:
  ssh -p $REMOTE_PORT $SSH_TARGET "pm2 status"
  ssh -p $REMOTE_PORT $SSH_TARGET "pm2 logs quant_okx_paper --lines 80"
  ssh -p $REMOTE_PORT $SSH_TARGET "pm2 logs quant_okx_dashboard --lines 80"
  ssh -p $REMOTE_PORT $SSH_TARGET "tail -f $REMOTE_DIR/logs/live_trading.log"
EOF
}

main() {
  ensure_remote_dir
  sync_code
  sync_env_if_needed
  sync_models_if_needed
  run_remote_deploy
  print_summary
}

main
