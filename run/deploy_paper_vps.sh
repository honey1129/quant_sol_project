#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="${APP_NAME:-quant_okx_paper}"
DASHBOARD_APP_NAME="${DASHBOARD_APP_NAME:-quant_okx_dashboard}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TELEGRAM_FLAG="${TELEGRAM_ENABLED:-0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8787}"
PACKAGE_MANAGER=""
PACKAGE_INDEX_REFRESHED=0

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
  DASHBOARD_APP_NAME Dashboard PM2 app name, default: quant_okx_dashboard
  ENV_FILE           .env path, default: <project>/.env
  PYTHON_BIN         Python interpreter used to create venv, default: python3
  TELEGRAM_ENABLED   Exported into precheck/runtime, default: 0
  DASHBOARD_PORT     Dashboard HTTP port, default: 8787

Behavior:
  - Automatically installs missing system packages such as python3-venv,
    python3-pip, nodejs, npm, pm2, git, curl, and build tools when possible.
  - Supports apt-get, dnf, yum, and apk package managers.
  - Requires root or sudo privileges for system package installation.
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

has_root_or_sudo() {
  [[ "${EUID:-$(id -u)}" -eq 0 ]] || command -v sudo >/dev/null 2>&1
}

run_privileged() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    fail "Need root or sudo privileges to install missing system packages"
  fi
}

detect_package_manager() {
  if [[ -n "$PACKAGE_MANAGER" ]]; then
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    PACKAGE_MANAGER="apt-get"
  elif command -v dnf >/dev/null 2>&1; then
    PACKAGE_MANAGER="dnf"
  elif command -v yum >/dev/null 2>&1; then
    PACKAGE_MANAGER="yum"
  elif command -v apk >/dev/null 2>&1; then
    PACKAGE_MANAGER="apk"
  else
    fail "Unsupported package manager. Install python3, python3-venv, pip, nodejs, npm, and pm2 manually."
  fi
}

refresh_package_index() {
  detect_package_manager

  if [[ "$PACKAGE_INDEX_REFRESHED" -eq 1 ]]; then
    return
  fi

  log "Refreshing package index via $PACKAGE_MANAGER"
  case "$PACKAGE_MANAGER" in
    apt-get)
      run_privileged apt-get update
      ;;
    dnf)
      run_privileged dnf makecache -y
      ;;
    yum)
      run_privileged yum makecache -y
      ;;
    apk)
      run_privileged apk update
      ;;
  esac

  PACKAGE_INDEX_REFRESHED=1
}

try_install_packages() {
  local packages=("$@")
  [[ ${#packages[@]} -gt 0 ]] || return 0

  detect_package_manager
  refresh_package_index

  case "$PACKAGE_MANAGER" in
    apt-get)
      DEBIAN_FRONTEND=noninteractive run_privileged apt-get install -y --no-install-recommends "${packages[@]}"
      ;;
    dnf)
      run_privileged dnf install -y "${packages[@]}"
      ;;
    yum)
      run_privileged yum install -y "${packages[@]}"
      ;;
    apk)
      run_privileged apk add --no-cache "${packages[@]}"
      ;;
  esac
}

install_or_fail() {
  local description="$1"
  shift
  log "Installing $description"
  try_install_packages "$@" || fail "Failed to install $description using $PACKAGE_MANAGER"
  hash -r
}

python_version_mm() {
  "$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

ensure_base_system_packages() {
  detect_package_manager
  has_root_or_sudo || fail "Need root or sudo privileges to install missing system packages"

  case "$PACKAGE_MANAGER" in
    apt-get)
      install_or_fail "base system packages" ca-certificates curl git build-essential pkg-config python3 python3-pip
      ;;
    dnf)
      install_or_fail "base system packages" ca-certificates curl git gcc gcc-c++ make pkgconf-pkg-config python3 python3-pip
      ;;
    yum)
      install_or_fail "base system packages" ca-certificates curl git gcc gcc-c++ make pkgconfig python3 python3-pip
      ;;
    apk)
      install_or_fail "base system packages" ca-certificates curl git build-base pkgconf python3 py3-pip
      ;;
  esac
}

ensure_python_venv_support() {
  local py_ver

  require_command "$PYTHON_BIN"
  py_ver="$(python_version_mm)"
  detect_package_manager

  case "$PACKAGE_MANAGER" in
    apt-get)
      if ! "$PYTHON_BIN" -m venv /tmp/codex_venv_probe.$$ >/dev/null 2>&1; then
        rm -rf /tmp/codex_venv_probe.$$
        if ! try_install_packages python3-venv; then
          install_or_fail "Python venv support" "python${py_ver}-venv"
        fi
      else
        rm -rf /tmp/codex_venv_probe.$$
      fi
      ;;
    dnf|yum)
      if ! "$PYTHON_BIN" -m venv /tmp/codex_venv_probe.$$ >/dev/null 2>&1; then
        rm -rf /tmp/codex_venv_probe.$$
        install_or_fail "Python venv support" python3-devel
      else
        rm -rf /tmp/codex_venv_probe.$$
      fi
      ;;
    apk)
      if ! "$PYTHON_BIN" -m venv /tmp/codex_venv_probe.$$ >/dev/null 2>&1; then
        rm -rf /tmp/codex_venv_probe.$$
        install_or_fail "Python venv support" py3-virtualenv
      else
        rm -rf /tmp/codex_venv_probe.$$
      fi
      ;;
  esac
}

dashboard_ui_present() {
  [[ -f "$PROJECT_ROOT/dashboard-ui/package.json" ]]
}

ensure_node() {
  if ! dashboard_ui_present; then
    return
  fi

  detect_package_manager

  if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    case "$PACKAGE_MANAGER" in
      apt-get)
        install_or_fail "Node.js and npm" nodejs npm
        ;;
      dnf)
        install_or_fail "Node.js and npm" nodejs npm
        ;;
      yum)
        install_or_fail "Node.js and npm" nodejs npm
        ;;
      apk)
        install_or_fail "Node.js and npm" nodejs npm
        ;;
      esac
  fi
}

ensure_pm2() {
  if [[ "$SKIP_START" -eq 1 ]]; then
    return
  fi

  if command -v pm2 >/dev/null 2>&1; then
    return
  fi

  ensure_node
  if ! command -v pm2 >/dev/null 2>&1; then
    log "Installing pm2 via npm"
    run_privileged npm install -g pm2 || fail "Failed to install pm2 globally"
    hash -r
  fi
}

ensure_system_dependencies() {
  ensure_base_system_packages
  ensure_python_venv_support
  ensure_node
  ensure_pm2
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

  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]] && ! "$PROJECT_ROOT/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
    log "Detected broken virtual environment without pip, recreating .venv"
    rm -rf "$PROJECT_ROOT/.venv"
  fi

  if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    log "Creating virtual environment"
    "$PYTHON_BIN" -m venv "$PROJECT_ROOT/.venv" || {
      rm -rf "$PROJECT_ROOT/.venv"
      fail "Failed to create virtual environment. Ensure python3-venv is installed and rerun."
    }
  fi

  log "Installing Python dependencies"
  "$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
}

build_dashboard_ui() {
  if ! dashboard_ui_present; then
    log "Dashboard UI source not found, skipping frontend build"
    return
  fi

  require_command node
  require_command npm

  log "Installing dashboard UI dependencies and building static assets"
  (
    cd "$PROJECT_ROOT/dashboard-ui"
    if [[ -f package-lock.json ]]; then
      npm ci
    else
      npm install
    fi
    npm run build
  )
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
  local apps=("$APP_NAME")
  if dashboard_ui_present; then
    apps+=("$DASHBOARD_APP_NAME")
  fi

  if [[ "$SKIP_START" -eq 1 ]]; then
    log "Skipping PM2 start/reload"
    return
  fi

  load_nvm_if_needed
  require_command pm2

  log "Starting or reloading PM2 apps: ${apps[*]}"
  (
    cd "$PROJECT_ROOT"
    local app_name
    for app_name in "${apps[@]}"; do
      if pm2 describe "$app_name" >/dev/null 2>&1; then
        PYTHONPATH="$PROJECT_ROOT" TELEGRAM_ENABLED="$TELEGRAM_FLAG" DASHBOARD_PORT="$DASHBOARD_PORT" \
          pm2 reload ecosystem.paper.config.js --only "$app_name" --update-env
      else
        PYTHONPATH="$PROJECT_ROOT" TELEGRAM_ENABLED="$TELEGRAM_FLAG" DASHBOARD_PORT="$DASHBOARD_PORT" \
          pm2 start ecosystem.paper.config.js --only "$app_name" --update-env
      fi
    done
    pm2 save
    pm2 status
  )
}

print_summary() {
  cat <<EOF

[deploy] Done.
[deploy] Project root: $PROJECT_ROOT
[deploy] Env file:     $ENV_FILE
[deploy] App name:     $APP_NAME
[deploy] Dashboard app: $DASHBOARD_APP_NAME
[deploy] Dashboard URL: http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 127.0.0.1):$DASHBOARD_PORT

[deploy] Useful commands:
  pm2 logs $APP_NAME
  pm2 logs $DASHBOARD_APP_NAME
  tail -f $PROJECT_ROOT/logs/live_trading.log
  tail -f $PROJECT_ROOT/logs/scheduler.log
  cat $PROJECT_ROOT/logs/live_trading_state.json
  curl http://127.0.0.1:$DASHBOARD_PORT/api/health
EOF
}

main() {
  cd "$PROJECT_ROOT"
  ensure_env_file
  ensure_system_dependencies
  maybe_git_pull
  validate_paper_env
  validate_model_artifacts
  setup_venv
  build_dashboard_ui
  mkdir -p "$PROJECT_ROOT/logs"
  run_precheck
  start_or_reload_pm2
  print_summary
}

main
