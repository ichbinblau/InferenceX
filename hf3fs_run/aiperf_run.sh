#!/usr/bin/env bash
set -uo pipefail
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

AIPERF_DIR=/workspace/utils/aiperf
AGENTIC_DIR=/workspace/utils/agentic-benchmark

pip_install() { python3 -m pip install --break-system-packages "$@" 2>/dev/null || python3 -m pip install "$@"; }

if ! python3 -c 'import aiperf' 2>/dev/null; then
    echo "[aiperf] installing deps ..."
    command -v git >/dev/null 2>&1 || { apt-get update && apt-get install -y git; }
    pip_install -q urllib3 requests || true
    [ -f "$AGENTIC_DIR/requirements.txt" ] && pip_install -q -r "$AGENTIC_DIR/requirements.txt" || true
    pip_install -q --ignore-installed -e "$AIPERF_DIR"
    pip_install -q --upgrade "datasets>=4.7.0"
fi

command -v hf >/dev/null 2>&1 || pip_install -q "huggingface_hub[cli]>=0.25.0"
echo "[aiperf] prefetching dataset semianalysisai/cc-traces-weka-with-subagents-052726 ..."
hf download --repo-type dataset "semianalysisai/cc-traces-weka-with-subagents-052726" || echo "[aiperf] WARN: dataset prefetch failed; aiperf will fetch on demand"

for C in 1 2 4; do
    echo "=========================================="
    echo "aiperf agentic replay: conc=$C  duration=900s"
    echo "=========================================="
    RD="/workspace/hf3fs_run/results/conc${C}"
    mkdir -p "$RD/aiperf_artifacts"
    EXTRA=""
    [ "900" -lt 900 ] && EXTRA="--unsafe-override"
    aiperf profile --scenario inferencex-agentx-mvp \
        --url http://localhost:30000 \
        --endpoint /v1/chat/completions --endpoint-type chat --streaming \
        --model /models/DeepSeek-R1-0528-MXFP4-v2 \
        --concurrency "$C" --benchmark-duration 900 --random-seed 42 \
        --failed-request-threshold 0.10 \
        --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
        --use-server-token-count --tokenizer-trust-remote-code \
        --max-context-length 163840 \
        --num-dataset-entries 472 \
        --slice-duration 1.0 \
        --output-artifact-dir "$RD/aiperf_artifacts" \
        $EXTRA \
        --public-dataset semianalysis_cc_traces_weka_with_subagents 2>&1 | tee "$RD/benchmark.log"
done
echo "[aiperf] sweep complete -> /workspace/hf3fs_run/results"
