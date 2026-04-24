#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="${APP_NAME:-quant_okx_paper}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TELEGRAM_FLAG="${TELEGRAM_ENABLED:-0}"

DO_GIT_PULL=0
SKIP_CHECK=0
SKIP_START=0

log() {
  printf '[deploy] %s\n' "$*"
}

fail() {
  printf '[deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash run/deploy_paper_vps.sh [--git-pull] [--skip-check] [--skip-start]

Options:
  --git-pull    Fetch and fast-forward the current git branch before deploy
  --skip-check  Skip run/check_okx_paper_ready.py
  --skip-start  Skip PM2 start/reload
  --help        Show this help message

Environment overrides:
  APP_NAME           PM2 app name, default: quant_okx_paper
  ENV_FILE           .env path, default: <project>/.env
  PYTHON_BIN         Python interpreter used to create venv, default: python3
  TELEGRAM_ENABLED   Exported into precheck/runtime, default: 0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --git-pull)
      DO_GIT_PULL=1
      shift
      ;;
    --skip-check)
      SKIP_CHECK=1
      shift
      ;;
    --skip-start)
      SKIP_START=1
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

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Missing required command: $cmd"
}

read_env_value() {
  local key="$1"
  local value
  value="$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 | cut -d= -f2- || true)"
  printf '%s' "$value"
}

ensure_placeholder_free() {
  local key="$1"
  local value="$2"
  case "$value" in
    ""|"你的APIKey"|"你的Secret"|"你的Passphrase"|"YOUR_TG_BOT_TOKEN"|"YOUR_TG_CHAT_ID"|"你的TG_BOT_TOKEN"|"你的TG_CHAT_ID")
      fail "$key is missing or still using placeholder text in $ENV_FILE"
      ;;
  esac
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    return
  fi

  cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
  fail "Created $ENV_FILE from .env.example. Fill in your OKX paper credentials, keep USE_SERVER=1, then rerun."
}

validate_paper_env() {
  local okx_key okx_secret okx_password use_server

  okx_key="$(read_env_value OKX_API_KEY)"
  okx_secret="$(read_env_value OKX_SECRET)"
  okx_password="$(read_env_value OKX_PASSWORD)"
  use_server="$(read_env_value USE_SERVER)"

  ensure_placeholder_free "OKX_API_KEY" "$okx_key"
  ensure_placeholder_free "OKX_SECRET" "$okx_secret"
  ensure_placeholder_free "OKX_PASSWORD" "$okx_password"

  if [[ "$use_server" != "1" ]]; then
    fail "This deploy script is for OKX paper trading only. Set USE_SERVER=1 in $ENV_FILE first."
  fi
}

validate_model_artifacts() {
  local feature_list_path model_paths feature_abs
  feature_list_path="$(read_env_value FEATURE_LIST_PATH)"
  model_paths="$(read_env_value MODEL_PATHS)"

  if [[ -z "$feature_list_path" ]]; then
    feature_list_path="models/feature_list.pkl"
  fi
  if [[ -z "$model_paths" ]]; then
    model_paths="lgb_v1:models/lgb_model.pkl,xgb_v1:models/xgb_model.pkl,rf_v1:models/rf_model.pkl"
  fi

  if [[ "$feature_list_path" = /* ]]; then
    feature_abs="$feature_list_path"
  else
    feature_abs="$PROJECT_ROOT/$feature_list_path"
  fi
  [[ -f "$feature_abs" ]] || fail "Missing feature list file: $feature_abs"

  IFS=',' read -r -a model_entries <<<"$model_paths"
  if [[ ${#model_entries[@]} -eq 0 ]]; then
    fail "MODEL_PATHS is empty in $ENV_FILE"
  fi

  local entry model_path model_abs
  for entry in "${model_entries[@]}"; do
    model_path="${entry#*:}"
    if [[ "$model_path" == "$entry" || -z "$model_path" ]]; then
      fail "Invalid MODEL_PATHS entry: $entry"
    fi
    if [[ "$model_path" = /* ]]; then
      model_abs="$model_path"
    else
      model_abs="$PROJECT_ROOT/$model_path"
    fi
    [[ -f "$model_abs" ]] || fail "Missing model artifact: $model_abs"
  done
}

maybe_git_pull() {
  if [[ "$DO_GIT_PULL" -ne 1 ]]; then
    return
  fi

  require_command git
  git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "Current directory is not a git repo"

  local current_branch
  current_branch="$(git -C "$PROJECT_ROOT" branch --show-current)"
  [[ -n "$current_branch" ]] || fail "Unable to determine current git branch"

  log "Fetching latest code for branch $current_branch"
  git -C "$PROJECT_ROOT" fetch --all --prune
  git -C "$PROJECT_ROOT" pull --ff-only origin "$current_branch"
}

setup_venv() {
  require_command "$PYTHON_BIN"

  if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    log "Creating virtual environment"
    "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv"
  fi

  log "Installing Python dependencies"
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
}

run_precheck() {
  if [[ "$SKIP_CHECK" -eq 1 ]]; then
    log "Skipping paper readiness check"
    return
  fi

  log "Running paper readiness check"
  (
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" TELEGRAM_ENABLED="$TELEGRAM_FLAG" \
      "$PROJECT_ROOT/.venv/bin/python" run/check_okx_paper_ready.py
  )
}

load_nvm_if_needed() {
  if command -v pm2 >/dev/null 2>&1; then
    return
  fi

  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
  fi
}

start_or_reload_pm2() {
  if [[ "$SKIP_START" -eq 1 ]]; then
    log "Skipping PM2 start/reload"
    return
  fi

  load_nvm_if_needed
  require_command pm2

  log "Starting or reloading PM2 app: $APP_NAME"
  (
    cd "$PROJECT_ROOT"
    if pm2 describe "$APP_NAME" >/dev/null 2>&1; then
      PYTHONPATH="$PROJECT_ROOT" TELEGRAM_ENABLED="$TELEGRAM_FLAG" \
        pm2 reload ecosystem.paper.config.js --only "$APP_NAME" --update-env
    else
      PYTHONPATH="$PROJECT_ROOT" TELEGRAM_ENABLED="$TELEGRAM_FLAG" \
        pm2 start ecosystem.paper.config.js --only "$APP_NAME" --update-env
    fi
    pm2 save
    pm2 status "$APP_NAME"
  )
}

print_summary() {
  cat <<EOF

[deploy] Done.
[deploy] Project root: $PROJECT_ROOT
[deploy] Env file:     $ENV_FILE
[deploy] App name:     $APP_NAME

[deploy] Useful commands:
  pm2 logs $APP_NAME
  tail -f $PROJECT_ROOT/logs/live_trading.log
  tail -f $PROJECT_ROOT/logs/scheduler.log
  cat $PROJECT_ROOT/logs/live_trading_state.json
EOF
}

main() {
  cd "$PROJECT_ROOT"
  ensure_env_file
  maybe_git_pull
  validate_paper_env
  validate_model_artifacts
  setup_venv
  mkdir -p "$PROJECT_ROOT/logs"
  run_precheck
  start_or_reload_pm2
  print_summary
}

main
