import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib

from config import config
from utils.utils import BASE_DIR, LOGS_DIR


LOCK_PATH = os.path.join(LOGS_DIR, "model_retrain.lock")
STATE_PATH = os.path.join(LOGS_DIR, "model_retrain_state.json")
BACKUP_ROOT = os.path.join(BASE_DIR, "models", "backups")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def pid_is_running(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def acquire_lock():
    if os.path.exists(LOCK_PATH):
        lock = read_json(LOCK_PATH, {})
        pid = lock.get("pid")
        if pid and pid_is_running(pid):
            raise RuntimeError(f"模型重训已在运行: pid={pid}")
    write_json_atomic(LOCK_PATH, {"pid": os.getpid(), "started_at": utc_now_iso()})


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


def artifact_paths():
    paths = []
    for rel_path in config.MODEL_PATHS.values():
        paths.append(os.path.join(BASE_DIR, rel_path))
    paths.append(os.path.join(BASE_DIR, config.FEATURE_LIST_PATH))
    return sorted(set(paths))


def make_backup(run_id):
    backup_dir = os.path.join(BACKUP_ROOT, f"retrain_{run_id}")
    manifest = []
    os.makedirs(backup_dir, exist_ok=True)

    for src_path in artifact_paths():
        rel_path = os.path.relpath(src_path, BASE_DIR)
        dst_path = os.path.join(backup_dir, rel_path)
        exists = os.path.exists(src_path)
        manifest.append({"path": rel_path, "exists": exists})
        if exists:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

    write_json_atomic(os.path.join(backup_dir, "manifest.json"), manifest)
    return backup_dir, manifest


def restore_backup(backup_dir, manifest):
    for item in manifest:
        rel_path = item["path"]
        dst_path = os.path.join(BASE_DIR, rel_path)
        src_path = os.path.join(backup_dir, rel_path)
        if item.get("exists") and os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
        elif os.path.exists(dst_path):
            os.remove(dst_path)


def prune_backups(keep_count):
    keep_count = max(1, int(keep_count))
    if not os.path.isdir(BACKUP_ROOT):
        return

    backup_dirs = []
    for name in os.listdir(BACKUP_ROOT):
        path = os.path.join(BACKUP_ROOT, name)
        if name.startswith("retrain_") and os.path.isdir(path):
            backup_dirs.append(path)

    backup_dirs.sort(reverse=True)
    for old_path in backup_dirs[keep_count:]:
        shutil.rmtree(old_path, ignore_errors=True)


def validate_artifacts():
    loaded = []
    for path in artifact_paths():
        if not os.path.exists(path):
            raise RuntimeError(f"训练后缺少模型产物: {path}")
        joblib.load(path)
        loaded.append(os.path.relpath(path, BASE_DIR))
    return loaded


def append_log_header(log_file, title):
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"\n===== {title} {utc_now_iso()} =====\n")


def run_subprocess(args, log_file):
    append_log_header(log_file, "subprocess")
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"$ {' '.join(args)}\n")
        file.flush()
        return subprocess.run(
            args,
            cwd=BASE_DIR,
            stdout=file,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def run_backtest_validation(log_file):
    from backtest.backtest import Backtester

    append_log_header(log_file, "backtest_validation")
    base_interval = config.INTERVALS[0] if config.INTERVALS else "5m"
    window = config.WINDOWS.get(base_interval, 1000)
    with open(log_file, "a", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file):
            backtester = Backtester(
                "multi_period",
                window,
                enable_csv_dump=False,
                show_progress=False,
                emit_diagnostics=False,
            )
            summary = backtester.run_backtest()
    if not summary:
        raise RuntimeError("回测验证未返回 summary")

    min_return_pct = float(config.MODEL_RETRAIN_MIN_RETURN_PCT)
    max_drawdown_pct = float(config.MODEL_RETRAIN_MAX_DRAWDOWN_PCT)
    if float(summary["return_pct"]) < min_return_pct:
        raise RuntimeError(
            f"回测收益未达标: return_pct={summary['return_pct']:.2f} < {min_return_pct:.2f}"
        )
    if float(summary["max_drawdown_pct"]) < max_drawdown_pct:
        raise RuntimeError(
            f"回测回撤超限: max_drawdown_pct={summary['max_drawdown_pct']:.2f} < {max_drawdown_pct:.2f}"
        )
    return summary


def write_state(**updates):
    state = read_json(STATE_PATH, {})
    state.update(updates)
    write_json_atomic(STATE_PATH, state)


def retrain_once(*, validate_backtest=None):
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_id = timestamp_id()
    log_file = os.path.join(LOGS_DIR, f"model_retrain_{run_id}.log")
    validate_backtest = (
        bool(config.MODEL_RETRAIN_VALIDATE_BACKTEST)
        if validate_backtest is None
        else bool(validate_backtest)
    )

    acquire_lock()
    backup_dir = None
    manifest = []
    started_at = utc_now_iso()
    write_state(
        last_attempt_at=started_at,
        last_status="running",
        last_log_path=log_file,
    )

    try:
        backup_dir, manifest = make_backup(run_id)
        train_returncode = run_subprocess([sys.executable, "-m", "train.train"], log_file)
        if train_returncode != 0:
            raise RuntimeError(f"训练命令失败: exit_code={train_returncode}")

        loaded_artifacts = validate_artifacts()
        backtest_summary = None
        if validate_backtest:
            backtest_summary = run_backtest_validation(log_file)

        prune_backups(config.MODEL_RETRAIN_KEEP_BACKUPS)
        finished_at = utc_now_iso()
        write_state(
            last_success_at=finished_at,
            last_finished_at=finished_at,
            last_status="success",
            last_log_path=log_file,
            last_backup_path=backup_dir,
            loaded_artifacts=loaded_artifacts,
            backtest_summary=backtest_summary,
        )
        print(f"模型重训成功: log={log_file}")
        if backtest_summary:
            print(
                "回测验证通过: "
                f"return={backtest_summary['return_pct']:.2f}%, "
                f"maxDD={backtest_summary['max_drawdown_pct']:.2f}%"
            )
        return 0
    except Exception as exc:
        if backup_dir and manifest:
            restore_backup(backup_dir, manifest)
        finished_at = utc_now_iso()
        write_state(
            last_finished_at=finished_at,
            last_status="failed",
            last_error=str(exc),
            last_log_path=log_file,
            last_backup_path=backup_dir,
        )
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(f"\nFAILED: {exc}\n")
        print(f"模型重训失败，已回滚旧模型: {exc}")
        print(f"详情日志: {log_file}")
        return 1
    finally:
        release_lock()


def main():
    parser = argparse.ArgumentParser(description="Safely retrain model artifacts from latest market data")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only train and validate model files, without backtest gating",
    )
    args = parser.parse_args()
    return retrain_once(validate_backtest=not args.skip_backtest)


if __name__ == "__main__":
    raise SystemExit(main())
