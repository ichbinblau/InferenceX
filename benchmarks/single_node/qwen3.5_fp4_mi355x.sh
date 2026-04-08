#!/usr/bin/env bash

source "$(dirname "$0")/../benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# Hardcoded Hub cache dir for this benchmark's MXFP4 weights: compare revision vs Hub, then remove and verify.
MXFP4_CACHE_NAME="models--amd--Qwen3.5-397B-A17B-MXFP4"
_hf_hub="${HF_HUB_CACHE:-/mnt/hf_hub_cache}"
MXFP4_CACHE_DIR="${_hf_hub%/}/${MXFP4_CACHE_NAME}"
unset _hf_hub

model_cache_log() {
  echo "[model-cache] $*"
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::notice::model-cache: $*"
  fi
}

# Remote revision on Hub (commit SHA for default branch).
REMOTE_SHA=""
REMOTE_SHA=$(
  python3 - <<'PY' 2>/dev/null || true
import os
import sys
try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit(1)
repo_id = "amd/Qwen3.5-397B-A17B-MXFP4"
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
api = HfApi(token=token) if token else HfApi()
info = api.model_info(repo_id)
sha = getattr(info, "sha", None) or getattr(info, "oid", None) or ""
sys.stdout.write(sha)
PY
)

LOCAL_SHA=""
if [[ -d "$MXFP4_CACHE_DIR" ]]; then
  for ref in "$MXFP4_CACHE_DIR/refs/main" "$MXFP4_CACHE_DIR/refs/heads/main"; do
    if [[ -f "$ref" ]]; then
      LOCAL_SHA=$(tr -d ' \t\n\r' <"$ref")
      break
    fi
  done
  if [[ -z "$LOCAL_SHA" && -d "$MXFP4_CACHE_DIR/snapshots" ]]; then
    LOCAL_SHA=$(find "$MXFP4_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | head -1)
  fi
fi

if [[ -d "$MXFP4_CACHE_DIR" ]]; then
  model_cache_log "pre-delete: cache_dir=$MXFP4_CACHE_DIR"
  model_cache_log "pre-delete: local_revision=${LOCAL_SHA:-<unknown>}"
  if [[ -n "$REMOTE_SHA" ]]; then
    model_cache_log "pre-delete: remote_revision=$REMOTE_SHA (Hub default branch)"
    if [[ -n "$LOCAL_SHA" && "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
      model_cache_log "pre-delete: revision MATCHES Hub (still removing cache per benchmark policy)"
    elif [[ -n "$LOCAL_SHA" ]]; then
      model_cache_log "pre-delete: revision DIFFERS from Hub (local may be stale or partial)"
    else
      model_cache_log "pre-delete: could not read local revision from refs/ or snapshots/"
    fi
  else
    model_cache_log "pre-delete: could not query Hub revision (offline or huggingface_hub); skip remote compare"
  fi
else
  model_cache_log "pre-delete: no cache at $MXFP4_CACHE_DIR (nothing to compare)"
fi

if ! rm -rf "$MXFP4_CACHE_DIR"; then
  model_cache_log "ERROR: rm -rf failed for $MXFP4_CACHE_DIR"
  exit 1
fi

if [[ -e "$MXFP4_CACHE_DIR" ]]; then
  model_cache_log "ERROR: path still exists after rm -rf: $MXFP4_CACHE_DIR"
  exit 1
fi
model_cache_log "post-delete: OK — cleanup SUCCEEDED, verify passed (path gone): $MXFP4_CACHE_DIR"

hf download "$MODEL"

export SGLANG_USE_AITER=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

EVAL_CONTEXT_ARGS=""
if [ "${EVAL_ONLY}" = "true" ]; then
    setup_eval_context
    EVAL_CONTEXT_ARGS="--context-length $EVAL_MAX_MODEL_LEN"
fi
# Start GPU monitoring (power, temperature, clocks every second)
start_gpu_monitor

set -x
python3 -m sglang.launch_server --model-path=$MODEL --trust-remote-code \
--host=0.0.0.0 --port=$PORT \
--tensor-parallel-size=$TP \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--attention-backend aiter \
--mem-fraction-static=0.9 \
--model-loader-extra-config '{"enable_multithread_load": true}' \
--watchdog-timeout 1200 $EVAL_CONTEXT_ARGS > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT"
    append_lm_eval_summary
fi

# Stop GPU monitoring
stop_gpu_monitor
set +x
