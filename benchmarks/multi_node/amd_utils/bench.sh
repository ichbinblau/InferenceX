#!/bin/bash
# Dual-Engine Disaggregated Benchmark Runner
#
# ENGINE=sglang (default): SGLang benchmark
# ENGINE=vllm:             vLLM benchmark
#
# Produces JSON result files via benchmark_serving.py so that the CI pipeline
# can collect and process results.
#
# Usage: bash bench.sh <n_prefill> <n_decode> <prefill_gpus> <decode_gpus> \
#            <model_dir> <model_name> <log_path> <isl> <osl> \
#            <concurrency_list> <req_rate> <random_range_ratio> <num_prompts_multiplier>

ENGINE="${ENGINE:-sglang-disagg}"

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
model_path=$5
model_name=$6
MODEL_PATH="${MODEL_PATH:-${model_path}/${model_name}}"
# vllm-disagg uses --served-model-name MODEL_NAME; sglang defaults to MODEL_PATH
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    BENCH_MODEL="${MODEL_NAME:-${MODEL_PATH}}"
else
    BENCH_MODEL="${MODEL_PATH}"
fi
log_path=$7

chosen_isl=${8:-1024}
chosen_osl=${9:-1024}
concurrency_list=${10:-"512x1"}
if [[ "$ENGINE" == "vllm-disagg" ]]; then
    chosen_req_rate=${11:-inf}
else
    chosen_req_rate=${11:-1}
fi
random_range_ratio=${12:-0.8}
num_prompts_multiplier=${13:-10}

IFS='x' read -r -a chosen_concurrencies <<< "$concurrency_list"

ROUTER_PORT="${ROUTER_PORT:-30000}"

export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

echo "Config ${chosen_isl}; ${chosen_osl}; ${chosen_concurrencies[0]}; ${chosen_req_rate}"

profile_folder="${log_path}/${ENGINE}_isl_${chosen_isl}_osl_${chosen_osl}"
mkdir -p "$profile_folder"

source "$(dirname "$0")/../../benchmark_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

PORT="${ROUTER_PORT}"
MODEL="${MODEL:-${BENCH_MODEL}}"
DURATION="${DURATION:-1800}"
export MODEL DURATION
RESULT_DIR="${RESULT_DIR:-${profile_folder}}"
AGENTIC_OUTPUT_DIR="${AGENTIC_OUTPUT_DIR:-${REPO_ROOT}}"
export AGENTIC_OUTPUT_DIR
RESULT_FILENAME_BASE="${RESULT_FILENAME:-agentic_bench}"

mkdir -p "$RESULT_DIR"

export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_with_subagents_060826_256k
resolve_trace_source
install_agentic_deps

ANY_FAILED=0
for max_concurrency in "${chosen_concurrencies[@]}"; do

    echo "=========================================="
    echo "Agentic trace replay: conc=$max_concurrency"
    echo "=========================================="

    CONC_RESULT_DIR="$RESULT_DIR/conc${max_concurrency}"
    mkdir -p "$CONC_RESULT_DIR"

    CONC="$max_concurrency"
    USERS="$max_concurrency"
    export CONC USERS
    build_replay_cmd "$CONC_RESULT_DIR"
    echo "$REPLAY_CMD" > "$CONC_RESULT_DIR/benchmark_command.txt"

    set +e
    $REPLAY_CMD 2>&1 | tee "$CONC_RESULT_DIR/benchmark.log"
    REPLAY_RC=${PIPESTATUS[0]}
    set -e

    PER_CONC_RESULT_FILENAME="${RESULT_FILENAME_BASE}_conc${max_concurrency}"
    RESULT_DIR="$CONC_RESULT_DIR" \
        AGENTIC_OUTPUT_DIR="$AGENTIC_OUTPUT_DIR" \
        RESULT_FILENAME="$PER_CONC_RESULT_FILENAME" \
        USERS="$max_concurrency" \
        python3 "$INFMAX_CONTAINER_WORKSPACE/utils/process_agentic_result.py" || {
            echo "WARNING: process_agentic_result.py failed for conc=$max_concurrency" >&2
            ANY_FAILED=1
        }

    python3 "$AGENTIC_DIR/scripts/analyze_benchmark_distributions.py" \
        "$CONC_RESULT_DIR/aiperf_artifacts" -o "$CONC_RESULT_DIR" 2>&1 || true

    # Generate metrics_plots.png from the aiperf artifacts. Best-effort:
    # don't fail the run if plotting has trouble (e.g. matplotlib missing).
    python3 "$INFMAX_CONTAINER_WORKSPACE/utils/generate_aiperf_plots.py" \
        "$CONC_RESULT_DIR" 2>&1 || true

    if [ "$REPLAY_RC" -ne 0 ]; then
        echo "WARNING: agentic trace replay for conc=$max_concurrency exited with code $REPLAY_RC after writing available results" >&2
        ANY_FAILED=1
    fi

    echo "-----------------------------------------"

    if [[ "$ENGINE" == "vllm-disagg" ]]; then
        echo "[BENCH] Cooldown: waiting 10s for idle KV block reaper..."
        sleep 10
    fi
done

export RESULT_FILENAME="$RESULT_FILENAME_BASE"

if [ "$ANY_FAILED" -ne 0 ]; then
    echo "WARNING: at least one conc had a non-zero exit; per-conc result files were still written when possible." >&2
fi
