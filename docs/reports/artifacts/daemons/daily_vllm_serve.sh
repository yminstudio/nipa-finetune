#!/usr/bin/env bash
set -uo pipefail

REPO=/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8
MODEL_PATH=/home/work/llm_model_full/gpt-oss-20b-seed-v8-round4-section-all-full-ft/final-export
SERVED_MODEL_NAME=gpt-oss-20b-zeroin-v8-r4
HOST=127.0.0.1
PORT=8000
TP_SIZE=1
REASONING_PARSER=
VLLM_DISABLE_HARMONY=1
MAX_MODEL_LEN=131072

LOG_DIR=/home/work/llm_model_full/_nightly_logs
DAEMON_LOG="$LOG_DIR/vllm_daemon.log"
mkdir -p "$LOG_DIR"

log_daemon() { echo "[vllm-daemon $(date -Is)] $*" >>"$DAEMON_LOG"; }

stop_vllm() {
  local pids
  pids=$(pgrep -f "vllm serve" || true)
  if [[ -z "$pids" ]]; then
    log_daemon "stop: no vllm process found"
    return 0
  fi
  log_daemon "stop: TERM -> $pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    sleep 2
    pgrep -f "vllm serve" >/dev/null || break
  done
  if pgrep -f "vllm serve" >/dev/null; then
    log_daemon "stop: KILL (TERM timeout)"
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    sleep 2
  fi
  log_daemon "stop: done"
}

start_vllm() {
  local serve_log="$LOG_DIR/vllm_serve_$(date +%Y%m%d_%H%M%S).log"
  if [[ ! -d "$MODEL_PATH" ]] || [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
    log_daemon "start: SKIP — model not ready at $MODEL_PATH"
    return 1
  fi
  log_daemon "start: launching vllm (log=$serve_log)"
  stop_vllm
  (
    MODEL_PATH="$MODEL_PATH" \
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    HOST="$HOST" PORT="$PORT" TP_SIZE="$TP_SIZE" \
    REASONING_PARSER="$REASONING_PARSER" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    VLLM_DISABLE_HARMONY="$VLLM_DISABLE_HARMONY" \
    nohup bash "$REPO/scripts/run_vllm_model_server.sh" >"$serve_log" 2>&1 &
  )
  disown 2>/dev/null || true
  sleep 3
  if pgrep -f "vllm serve" >/dev/null; then
    log_daemon "start: vllm serve process is up"
  else
    log_daemon "start: WARN vllm serve not detected after 3s (check $serve_log)"
  fi
}

log_daemon "daemon started (pid=$$)"

while true; do
  now=$(date +%s)
  today_start=$(date -d "today 12:00" +%s)
  today_1750=$(date -d "today 17:50" +%s)

  if (( now >= today_start && now < today_1750 )); then
    start_ts=$now
    stop_ts=$today_1750
  elif (( now < today_start )); then
    start_ts=$today_start
    stop_ts=$today_1750
  else
    start_ts=$(date -d "tomorrow 12:00" +%s)
    stop_ts=$(date -d "tomorrow 17:50" +%s)
  fi

  if (( start_ts > now )); then
    wait_s=$((start_ts - now))
    log_daemon "sleeping ${wait_s}s until serve start $(date -d "@$start_ts" '+%Y-%m-%d %H:%M:%S')"
    sleep "$wait_s"
  fi

  start_vllm

  now=$(date +%s)
  if (( stop_ts > now )); then
    wait_s=$((stop_ts - now))
    log_daemon "serving; will stop in ${wait_s}s at $(date -d "@$stop_ts" '+%Y-%m-%d %H:%M:%S')"
    sleep "$wait_s"
  fi

  stop_vllm
done
