#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Local smoke test for: kimik2.5-fp4-mi355x-vllm-disagg
#
# Simulates what the GitHub Actions CI pipeline does:
#   amd-master.yaml config  →  runner (launch_mi355x-amds.sh)
#   →  entry script (kimik2.5_fp4_mi355x_vllm-disagg.sh)
#   →  submit.sh  →  sbatch job.slurm  →  Docker on each node
#
# This is a multi-node Slurm job (3 nodes: 1 prefill co-located with proxy + 2 decode).
# Uses public base image vllm/vllm-openai-rocm:v0.17.1; dependencies are
# installed at container start by setup_deps.sh (sourced from server.sh).
#
# Usage:
#   bash local_test.sh
#   ISL=8192 OSL=1024 CONC_LIST="8 16 32" bash local_test.sh
# ──────────────────────────────────────────────────────────────

# --- Configurable parameters (override via env) ---
#IMAGE="${IMAGE:-aigmkt/vllm-openai-rocm:v0.18.0-1}"
#IMAGE="${IMAGE:-ghcr.io/simondanielsson/vllm-dev:ainic-test-hydra}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036}"
MODEL="${MODEL:-amd/Kimi-K2.5-MXFP4}"
MODEL_NAME="${MODEL_NAME:-Kimi-K2.5-MXFP4}"
PRECISION="${PRECISION:-fp4}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONC_LIST="${CONC_LIST:-8}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
SPEC_DECODING="${SPEC_DECODING:-none}"

# Disagg topology: 1P + 1D = 2 nodes (proxy co-located with first prefill)
PREFILL_NODES="${PREFILL_NODES:-1}"
PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS:-1}"
PREFILL_TP="${PREFILL_TP:-8}"
DECODE_NODES="${DECODE_NODES:-1}"
DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS:-1}"
DECODE_TP="${DECODE_TP:-8}"
PREFILL_ENABLE_EP="${PREFILL_ENABLE_EP:-false}"
PREFILL_ENABLE_DP="${PREFILL_ENABLE_DP:-false}"
DECODE_ENABLE_EP="${DECODE_ENABLE_EP:-true}"
DECODE_ENABLE_DP="${DECODE_ENABLE_DP:-false}"

# Cluster config (NODELIST must be exported so kimik2.5_fp4_mi355x_vllm-disagg.sh → submit.sh see it)
export NODELIST="${NODELIST:-mia1-p01-g06,mia1-p01-g05}"
# HF-cache-style layout lives under models/hub/, so point MODEL_PATH there.
MODEL_PATH="${MODEL_PATH:-/it-share/thshan/models/hub}"

# When set, skip sbatch and reuse the specified existing Slurm allocation.
# The inline run attaches to SLURM_REUSE_JOBID via SLURM_OVERLAP=1.
export SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-7741}"

# export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"

# --- Paths ---
# The InferenceX repo must be on NFS so all Slurm nodes can access it.
# Sync from local: rsync -a --delete ~/InferenceX/ /nfsdata/InferenceX/
INFERENCEX_DIR="${INFERENCEX_DIR:-/it-share/thshan/InferenceX}"

# --- Logs ---
LOG_DIR="/it-share/thshan/InferenceX/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/local_test_kimik2.5_fp4_vllm-disagg_isl${ISL}_osl${OSL}_${TIMESTAMP}.log"

echo "============================================"
echo " Kimi-K2.5 FP4 vLLM Disagg Local Test"
echo "============================================"
echo " Image:          ${IMAGE}"
echo " Model:          ${MODEL}"
echo " ISL:            ${ISL}  OSL: ${OSL}"
echo " Concurrencies:  ${CONC_LIST}"
echo " Prefill:        ${PREFILL_NUM_WORKERS} workers on ${PREFILL_NODES} nodes (TP=${PREFILL_TP})"
echo " Decode:         ${DECODE_NUM_WORKERS} workers on ${DECODE_NODES} nodes (TP=${DECODE_TP})"
echo " Nodes:          ${NODELIST}"
echo " Model path:     ${MODEL_PATH}"
echo " Log:            ${LOG_FILE}"
echo "============================================"

# ──────────────────────────────────────────────────────────────
# Set up the environment as the CI runner would
# (mirrors what launch_mi355x-amds.sh does for IS_MULTINODE=true)
# ──────────────────────────────────────────────────────────────

# Variables that the runner (launch_mi355x-amds.sh) normally sets
export SLURM_PARTITION="${SLURM_PARTITION:-amd-aim}"
if [[ -z "${SLURM_ACCOUNT:-}" ]]; then
    # Prefer an account associated with the selected partition; fall back to
    # a partition-wide/default association and finally to partition name.
    if command -v sacctmgr >/dev/null 2>&1; then
        SLURM_ACCOUNT="$(sacctmgr -nP show assoc user="$(whoami)" format=account,partition 2>/dev/null \
            | awk -F'|' -v part="$SLURM_PARTITION" '$1 != "" && ($2 == part || $2 == "") { print $1; exit }')"
    fi
    export SLURM_ACCOUNT="${SLURM_ACCOUNT:-$SLURM_PARTITION}"
else
    export SLURM_ACCOUNT
fi
export RUNNER_NAME="${RUNNER_NAME:-th-vllm-disagg}"
export GITHUB_WORKSPACE="${INFERENCEX_DIR}"

export MODEL_NAME="${MODEL_NAME}"
export MODEL_PATH="${MODEL_PATH}"
export MODEL_DIR="${MODEL_PATH}"
export GPUS_PER_NODE=8
export IS_MULTINODE=true

# Variables that the CI workflow sets (from amd-master.yaml → benchmark-multinode-tmpl.yml)
export IMAGE="${IMAGE}"
export MODEL="${MODEL}"
export PRECISION="${PRECISION}"
export ISL="${ISL}"
export OSL="${OSL}"
export CONC_LIST="${CONC_LIST}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO}"
export SPEC_DECODING="${SPEC_DECODING}"
export RUN_EVAL="${RUN_EVAL:-false}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export FRAMEWORK="vllm-disagg"
export EXP_NAME="dsr1_isl${ISL}_osl${OSL}"
export CONTAINER_IMAGE="${IMAGE}"

# Prefill / Decode config (from search-space in YAML)
export PREFILL_NODES="${PREFILL_NODES}"
export PREFILL_NUM_WORKERS="${PREFILL_NUM_WORKERS}"
export PREFILL_TP="${PREFILL_TP}"
export PREFILL_EP=1
export PREFILL_DP_ATTN=false
export DECODE_NODES="${DECODE_NODES}"
export DECODE_NUM_WORKERS="${DECODE_NUM_WORKERS}"
export DECODE_TP="${DECODE_TP}"
export DECODE_EP=1
export DECODE_DP_ATTN=false

export BENCHMARK_LOGS_DIR="${LOG_DIR}/benchmark_logs_${TIMESTAMP}"
mkdir -p "$BENCHMARK_LOGS_DIR"

# ──────────────────────────────────────────────────────────────
# Call the entry point (same as what the runner does)
# ──────────────────────────────────────────────────────────────

cd "${INFERENCEX_DIR}/benchmarks/multi_node/amd_utils"

echo ""
echo "Submitting Slurm job..."
set -x
JOB_ID=$(bash "${INFERENCEX_DIR}/benchmarks/multi_node/kimik2.5_fp4_mi355x_vllm-disagg.sh")
RC=$?
set +x

if [[ $RC -ne 0 ]] || [[ -z "$JOB_ID" ]]; then
    echo "ERROR: Failed to submit Slurm job (rc=$RC)"
    exit 1
fi

echo ""
echo "============================================"
echo " Slurm job submitted: ${JOB_ID}"
echo " Logs dir:  ${BENCHMARK_LOGS_DIR}"
echo "============================================"
echo ""

# ──────────────────────────────────────────────────────────────
# Wait for job to complete (same as what the runner does)
# ──────────────────────────────────────────────────────────────

SLURM_LOG="${BENCHMARK_LOGS_DIR}/slurm_job-${JOB_ID}.out"

# In reuse mode, the workload is a plain background bash process; track its PID
# (written by submit.sh) instead of polling squeue (which reports the parent
# allocation, not the workload).
INLINE_PID_FILE="${BENCHMARK_LOGS_DIR}/slurm_job-${JOB_ID}.pid"
REUSE_MODE=0
INLINE_PID=""
if [[ -n "${SLURM_REUSE_JOBID:-}" ]] && [[ -f "$INLINE_PID_FILE" ]]; then
    REUSE_MODE=1
    INLINE_PID=$(cat "$INLINE_PID_FILE")
    echo "Reuse mode: tracking job.slurm pid=${INLINE_PID}"
fi

job_is_alive() {
    if [[ "$REUSE_MODE" -eq 1 ]]; then
        kill -0 "$INLINE_PID" 2>/dev/null
    else
        squeue -u "$(whoami)" --noheader --format='%i' | grep -q "$JOB_ID"
    fi
}

# Wait for log file to appear
echo "Waiting for Slurm log file to appear..."
for i in $(seq 1 60); do
    if [[ -f "$SLURM_LOG" ]]; then
        break
    fi
    if ! job_is_alive; then
        echo "ERROR: Job $JOB_ID disappeared before log file was created"
        [[ "$REUSE_MODE" -eq 0 ]] && scontrol show job "$JOB_ID" 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

if [[ ! -f "$SLURM_LOG" ]]; then
    echo "WARNING: Log file not found after 5 min, tailing anyway..."
fi

# Poll for job completion in background
(
    while job_is_alive; do
        sleep 10
    done
) &
POLL_PID=$!

# Tail the log file until job completes
echo "Tailing Slurm output (Ctrl+C to stop tailing, job continues)..."
echo "──────────────────────────────────────────────"
tail -F -s 2 -n+1 "$SLURM_LOG" --pid=$POLL_PID 2>/dev/null | tee "$LOG_FILE" || true

wait $POLL_PID 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────"
echo " Job ${JOB_ID} completed"
echo " Slurm log:     ${SLURM_LOG}"
echo " Benchmark logs: ${BENCHMARK_LOGS_DIR}"
echo " Local log:      ${LOG_FILE}"
echo "──────────────────────────────────────────────"

# Show final job status
scontrol show job "$JOB_ID" 2>/dev/null | grep -E "JobState|ExitCode|NodeList" || true
