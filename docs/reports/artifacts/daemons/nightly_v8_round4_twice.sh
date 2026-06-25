#!/usr/bin/env bash
set -uo pipefail

REPO=/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8
OUTPUT_DIR=/home/work/llm_model_full/gpt-oss-20b-seed-v8-round4-section-all-full-ft
CONFIG="$REPO/configs/gpt_oss_20b_seed_v8_round4_section_all_full_ft.json"
VENV_PY=/home/work/dev_data/fine-tuning/.venv-gpt-oss-train/bin/python

LOG_DIR=/home/work/llm_model_full/_nightly_logs
DAEMON_LOG="$LOG_DIR/v8_round4_daemon.log"
mkdir -p "$LOG_DIR"

log_daemon() { echo "[daemon $(date -Is)] $*" >>"$DAEMON_LOG"; }

run_training() {
  local label="$1"
  echo "[nightly] === $label start $(date) ==="
  if [[ -d "$OUTPUT_DIR" ]]; then
    echo "[nightly] removing existing $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
  fi
  (cd "$REPO" && "$VENV_PY" scripts/run_full_ft_gpt_oss_20b.py \
     --config "$CONFIG" --orchestrate) || echo "[nightly] $label exited non-zero ($?)"
  echo "[nightly] === $label end $(date) ==="
}

log_daemon "daemon started (pid=$$)"

while true; do
  now=$(date +%s)
  target=$(date -d "today 18:00" +%s)
  if [[ $now -ge $target ]]; then
    target=$(date -d "tomorrow 18:00" +%s)
  fi
  sleep_sec=$((target - now))
  target_str=$(date -d "@$target" '+%Y-%m-%d %H:%M:%S')
  log_daemon "next session at $target_str, sleeping ${sleep_sec}s (~$((sleep_sec/3600))h $((sleep_sec%3600/60))m)"
  sleep "$sleep_sec"

  SESSION_LOG="$LOG_DIR/v8_round4_twice_$(date +%Y%m%d_%H%M%S).log"
  log_daemon "session starting, log=$SESSION_LOG"
  {
    echo "[nightly] session started at $(date)"
    echo "[nightly] log: $SESSION_LOG"
    run_training "RUN 1"
    run_training "RUN 2"
    run_training "RUN 3"
    echo "[nightly] session done at $(date)"
  } >>"$SESSION_LOG" 2>&1
  log_daemon "session complete, log=$SESSION_LOG"
done
