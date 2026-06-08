#!/usr/bin/env bash
# =============================================================================
# Standalone 1P1D (SGLang disaggregated) launcher with HiCache + HF3FS L3 store,
# driven over ssh + docker run (NO SLURM / no server_sglang.sh integration),
# then runs the agentic aiperf trace-replay benchmark against the PD router.
#
#   Prefill node : mia1-p01-g05  (TP8, HiCache L2 + HF3FS L3)
#   Decode  node : mia1-p01-g06  (TP8, mirrors --page-size 64)
#   Router       : sglang_router --pd-disaggregation on the prefill node :30000
#   Benchmark    : aiperf inferencex-agentx-mvp (Weka subagent traces)
#
# HF3FS notes (see storage_hf3fs.py / hf3fs_client.py in the image):
#   * DeepSeek-R1 is an MLA model -> HiCacheHF3FS REQUIRES a global metadata
#     server (Hf3fsGlobalMetadataClient). We start mini_3fs_metadata_server on
#     the prefill node and point metadata_server_url at it.
#   * No real 3FS/USRBIO hardware here, so we enable the file-backed mock client
#     ({"use_mock_hf3fs_client": true}). Hf3fsMockClient does real pread/pwrite
#     on a sparse ftruncate'd file -> a functional L3 KV store on local disk.
#   * layer_first + kernel IO backend (non-zero-copy path) avoids the
#     host>device pool assert, same as the proven L2-only run.
#
# Usage:
#   bash run_hf3fs_pd_aiperf.sh up        # launch PD + router + servers (no bench)
#   bash run_hf3fs_pd_aiperf.sh bench     # run the aiperf conc sweep (servers must be up)
#   bash run_hf3fs_pd_aiperf.sh all       # up + bench  (default)
#   bash run_hf3fs_pd_aiperf.sh status    # health of metadata/prefill/decode/router
#   bash run_hf3fs_pd_aiperf.sh logs      # tail the in-container logs
#   bash run_hf3fs_pd_aiperf.sh down      # stop & remove the PD containers
#
# Common overrides:
#   CONC="1 2 4"  DURATION=900  bash run_hf3fs_pd_aiperf.sh all
#   PREFILL_NODE=... DECODE_NODE=... IMAGE=... bash run_hf3fs_pd_aiperf.sh up
# =============================================================================
set -euo pipefail

# ── Cluster / nodes ──
PREFILL_NODE="${PREFILL_NODE:-mia1-p01-g05}"
DECODE_NODE="${DECODE_NODE:-mia1-p01-g06}"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

# ── Container / model ──
IMAGE="${IMAGE:-lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260519}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-/it-share/hf_cache}"   # host dir mounted to /models
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528-MXFP4-v2}"
MODEL_CTR="/models/${MODEL_NAME}"                        # path inside container
REPO_DIR="${REPO_DIR:-/it-share/thshan/InferenceX}"      # mounted to /workspace
IBDEVICES="${IBDEVICES:-rdma3,rdma0,rdma2,rdma1,rdma7,rdma4,rdma6,rdma5}"

# ── Ports (host network) ──
SERVER_PORT="${SERVER_PORT:-8000}"      # prefill & decode (different hosts)
ROUTER_PORT="${ROUTER_PORT:-30000}"     # PD router on prefill node
META_PORT="${META_PORT:-18000}"         # hf3fs metadata server on prefill node

# ── Container names ──
PREFILL_CONT="${PREFILL_CONT:-pd_hf3fs_prefill}"
DECODE_CONT="${DECODE_CONT:-pd_hf3fs_decode}"

# ── HiCache / HF3FS knobs ──
HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-125}"          # per-rank L2 host pool (GB)
PAGE_SIZE="${PAGE_SIZE:-64}"
HF3FS_FILE_SIZE="${HF3FS_FILE_SIZE:-107374182800}" # ~100 GiB sparse L3 file
HF3FS_NUMJOBS="${HF3FS_NUMJOBS:-16}"
HF3FS_ENTRIES="${HF3FS_ENTRIES:-8}"
HF3FS_DATA_DIR_CTR="/run_logs/hf3fs_pd"            # = host /tmp/hf3fs_pd (node-local)

# ── MoRI env (from the working mori 1P1D run, job 8196) ──
P_COMBINE_DTYPE="${P_COMBINE_DTYPE:-fp8_direct_cast}"
P_MOE_MAX_INPUT="${P_MOE_MAX_INPUT:-32768}"
P_DISPATCH_TOKENS="${P_DISPATCH_TOKENS:-8192}"
D_COMBINE_DTYPE="${D_COMBINE_DTYPE:-fp8}"
D_MOE_MAX_INPUT="${D_MOE_MAX_INPUT:-2703}"
D_DISPATCH_TOKENS="${D_DISPATCH_TOKENS:-512}"
MORI_RDMA_TC="${MORI_RDMA_TC:-104}"
MORI_RDMA_SL="${MORI_RDMA_SL:-3}"
MORI_IO_TC="${MORI_IO_TC:-104}"
MORI_IO_SL="${MORI_IO_SL:-3}"

# ── Server tunables (match the working 1P1D TP8 run) ──
TP_SIZE="${TP_SIZE:-8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-163840}"
PREFILL_MEM_FRAC="${PREFILL_MEM_FRAC:-0.8}"
DECODE_MEM_FRAC="${DECODE_MEM_FRAC:-0.85}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"

# ── aiperf / benchmark ──
CONC="${CONC:-1 2 4}"
DURATION="${DURATION:-900}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"   # aiperf --max-context-length (filters over-length traces)
WEKA_LOADER="${WEKA_LOADER:-semianalysis_cc_traces_weka_with_subagents}"
WEKA_DATASET="${WEKA_DATASET:-semianalysisai/cc-traces-weka-with-subagents-052726}"
NUM_DATASET_ENTRIES="${NUM_DATASET_ENTRIES:-472}"

# ── Working dir on the shared NFS repo (visible in container at /workspace/hf3fs_run) ──
WORKDIR_HOST="${REPO_DIR}/hf3fs_run"
WORKDIR_CTR="/workspace/hf3fs_run"
RESULT_DIR_CTR="${WORKDIR_CTR}/results"

CUDA_GRAPH_BS="$(seq -s ' ' 1 128)"

log() { printf '\033[1;36m[hf3fs-pd]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[hf3fs-pd ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

resolve_ips() {
    PREFILL_IP="$($SSH "$PREFILL_NODE" 'ip route get 1.1.1.1 | awk "/src/{print \$7}"' 2>/dev/null)"
    DECODE_IP="$($SSH "$DECODE_NODE"  'ip route get 1.1.1.1 | awk "/src/{print \$7}"' 2>/dev/null)"
    [[ -n "$PREFILL_IP" && -n "$DECODE_IP" ]] || die "could not resolve node IPs (prefill=$PREFILL_IP decode=$DECODE_IP)"
    log "prefill=$PREFILL_NODE ($PREFILL_IP)  decode=$DECODE_NODE ($DECODE_IP)"
}

# Generate the per-role inner scripts + hf3fs config onto the shared NFS workdir.
write_inner_scripts() {
    mkdir -p "$WORKDIR_HOST" "${WORKDIR_HOST}/results"

    cat > "${WORKDIR_HOST}/hf3fs_config.json" <<EOF
{
    "file_path_prefix": "${HF3FS_DATA_DIR_CTR}/kvcache",
    "file_size": ${HF3FS_FILE_SIZE},
    "numjobs": ${HF3FS_NUMJOBS},
    "entries": ${HF3FS_ENTRIES},
    "metadata_server_url": "http://${PREFILL_IP}:${META_PORT}"
}
EOF

    # ---- hf3fs metadata server (prefill node only) ----
    cat > "${WORKDIR_HOST}/meta_launch.sh" <<EOF
#!/usr/bin/env bash
set -x
mkdir -p ${HF3FS_DATA_DIR_CTR}
exec python3 -m sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server \\
    --host 0.0.0.0 --port ${META_PORT}
EOF

    # ---- prefill server: HiCache L2 + HF3FS L3 (mock client) ----
    cat > "${WORKDIR_HOST}/prefill_launch.sh" <<EOF
#!/usr/bin/env bash
set -x
mkdir -p ${HF3FS_DATA_DIR_CTR}
export SGLANG_HICACHE_HF3FS_CONFIG_PATH=${WORKDIR_CTR}/hf3fs_config.json
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}:/usr/local/lib/python3.12/dist-packages"
export SGLANG_MORI_COMBINE_DTYPE=${P_COMBINE_DTYPE}
export SGLANG_MORI_MOE_MAX_INPUT_TOKENS=${P_MOE_MAX_INPUT}
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${P_DISPATCH_TOKENS}
export MORI_RDMA_TC=${MORI_RDMA_TC} MORI_RDMA_SL=${MORI_RDMA_SL} MORI_IO_TC=${MORI_IO_TC} MORI_IO_SL=${MORI_IO_SL}
exec python3 -m sglang.launch_server \\
    --model-path ${MODEL_CTR} \\
    --disaggregation-mode prefill \\
    --disaggregation-ib-device ${IBDEVICES} \\
    --host 0.0.0.0 --port ${SERVER_PORT} --trust-remote-code \\
    --tp-size ${TP_SIZE} --decode-log-interval 1000 --log-level warning --watchdog-timeout 3600 \\
    --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 \\
    --attention-backend aiter --disaggregation-transfer-backend mori \\
    --mem-fraction-static ${PREFILL_MEM_FRAC} --max-running-requests ${MAX_RUNNING_REQUESTS} \\
    --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \\
    --cuda-graph-bs ${CUDA_GRAPH_BS} \\
    --context-length ${CONTEXT_LENGTH} \\
    --page-size ${PAGE_SIZE} --enable-hierarchical-cache \\
    --hicache-size ${HICACHE_SIZE_GB} --hicache-io-backend kernel --hicache-mem-layout layer_first \\
    --hicache-write-policy write_through \\
    --hicache-storage-backend hf3fs \\
    --hicache-storage-backend-extra-config '{"use_mock_hf3fs_client": true}' \\
    --enable-metrics --enable-cache-report
EOF

    # ---- decode server: mirrors --page-size, no HiCache (mori -> chunk cache) ----
    cat > "${WORKDIR_HOST}/decode_launch.sh" <<EOF
#!/usr/bin/env bash
set -x
export SGLANG_MORI_COMBINE_DTYPE=${D_COMBINE_DTYPE}
export SGLANG_MORI_MOE_MAX_INPUT_TOKENS=${D_MOE_MAX_INPUT}
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${D_DISPATCH_TOKENS}
export MORI_RDMA_TC=${MORI_RDMA_TC} MORI_RDMA_SL=${MORI_RDMA_SL} MORI_IO_TC=${MORI_IO_TC} MORI_IO_SL=${MORI_IO_SL}
exec python3 -m sglang.launch_server \\
    --model-path ${MODEL_CTR} \\
    --disaggregation-mode decode \\
    --disaggregation-ib-device ${IBDEVICES} \\
    --host 0.0.0.0 --port ${SERVER_PORT} --trust-remote-code \\
    --tp-size ${TP_SIZE} --decode-log-interval 1000 --log-level warning --watchdog-timeout 3600 \\
    --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 \\
    --attention-backend aiter --disaggregation-transfer-backend mori \\
    --mem-fraction-static ${DECODE_MEM_FRAC} --max-running-requests ${MAX_RUNNING_REQUESTS} \\
    --cuda-graph-bs ${CUDA_GRAPH_BS} \\
    --context-length ${CONTEXT_LENGTH} \\
    --prefill-round-robin-balance \\
    --page-size ${PAGE_SIZE}
EOF

    # ---- PD router (prefill node) ----
    cat > "${WORKDIR_HOST}/router_launch.sh" <<EOF
#!/usr/bin/env bash
set -x
exec python3 -m sglang_router.launch_router \\
    --pd-disaggregation \\
    --port ${ROUTER_PORT} \\
    --policy random --prefill-policy random --decode-policy random \\
    --prefill http://${PREFILL_IP}:${SERVER_PORT} \\
    --decode http://${DECODE_IP}:${SERVER_PORT}
EOF

    # ---- aiperf sweep (prefill node) ----
    cat > "${WORKDIR_HOST}/aiperf_run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800

AIPERF_DIR=/workspace/utils/aiperf
AGENTIC_DIR=/workspace/utils/agentic-benchmark

pip_install() { python3 -m pip install --break-system-packages "\$@" 2>/dev/null || python3 -m pip install "\$@"; }

if ! python3 -c 'import aiperf' 2>/dev/null; then
    echo "[aiperf] installing deps ..."
    command -v git >/dev/null 2>&1 || { apt-get update && apt-get install -y git; }
    pip_install -q urllib3 requests || true
    [ -f "\$AGENTIC_DIR/requirements.txt" ] && pip_install -q -r "\$AGENTIC_DIR/requirements.txt" || true
    pip_install -q --ignore-installed -e "\$AIPERF_DIR"
    pip_install -q --upgrade "datasets>=4.7.0"
fi

command -v hf >/dev/null 2>&1 || pip_install -q "huggingface_hub[cli]>=0.25.0"
echo "[aiperf] prefetching dataset ${WEKA_DATASET} ..."
hf download --repo-type dataset "${WEKA_DATASET}" || echo "[aiperf] WARN: dataset prefetch failed; aiperf will fetch on demand"

for C in ${CONC}; do
    echo "=========================================="
    echo "aiperf agentic replay: conc=\$C  duration=${DURATION}s"
    echo "=========================================="
    RD="${RESULT_DIR_CTR}/conc\${C}"
    mkdir -p "\$RD/aiperf_artifacts"
    EXTRA=""
    [ "${DURATION}" -lt 900 ] && EXTRA="--unsafe-override"
    aiperf profile --scenario inferencex-agentx-mvp \\
        --url http://localhost:${ROUTER_PORT} \\
        --endpoint /v1/chat/completions --endpoint-type chat --streaming \\
        --model ${MODEL_CTR} \\
        --concurrency "\$C" --benchmark-duration ${DURATION} --random-seed 42 \\
        --failed-request-threshold 0.10 \\
        --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \\
        --use-server-token-count --tokenizer-trust-remote-code \\
        --max-context-length ${MAX_MODEL_LEN} \\
        --num-dataset-entries ${NUM_DATASET_ENTRIES} \\
        --slice-duration 1.0 \\
        --output-artifact-dir "\$RD/aiperf_artifacts" \\
        \$EXTRA \\
        --public-dataset ${WEKA_LOADER} 2>&1 | tee "\$RD/benchmark.log"
done
echo "[aiperf] sweep complete -> ${RESULT_DIR_CTR}"
EOF

    chmod +x "${WORKDIR_HOST}"/*.sh
    log "wrote inner scripts + hf3fs_config.json to ${WORKDIR_HOST}"
}

docker_run_common=(
    --init --stop-timeout 10
    --device /dev/dri --device /dev/kfd --device /dev/infiniband
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=-1
    --network host --ipc host --group-add video
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged
    -v /sys:/sys
    -v "${MODEL_HOST_DIR}:/models"
    --shm-size 128G
    -v /tmp:/run_logs
    -v "${REPO_DIR}:/workspace"
    --entrypoint=
)

start_container() {
    local node="$1" name="$2"
    log "starting container '$name' on $node ..."
    $SSH "$node" "docker rm -f $name >/dev/null 2>&1 || true; \
        docker run -d --name $name ${docker_run_common[*]} '$IMAGE' sleep infinity >/dev/null" \
        || die "docker run failed on $node"
}

dexec_bg() { $SSH "$1" "docker exec -d $2 bash -lc '$3'"; }
dexec()    { $SSH "$1" "docker exec $2 bash -lc '$3'"; }

wait_http() {  # node port path label
    local node="$1" port="$2" path="$3" label="$4" i
    for ((i=1; i<=180; i++)); do
        if $SSH "$node" "curl -sf -o /dev/null --max-time 3 http://localhost:${port}${path}" 2>/dev/null; then
            log "$label is up (${i}x5s)"; return 0
        fi
        sleep 5
    done
    die "$label did not become healthy on $node:$port$path"
}

cmd_up() {
    resolve_ips
    write_inner_scripts

    start_container "$PREFILL_NODE" "$PREFILL_CONT"
    start_container "$DECODE_NODE"  "$DECODE_CONT"

    # Pre-create the node-local log/data dir; the background execs redirect into
    # it before the inner scripts get a chance to mkdir it themselves.
    $SSH "$PREFILL_NODE" "docker exec $PREFILL_CONT mkdir -p ${HF3FS_DATA_DIR_CTR}"
    $SSH "$DECODE_NODE"  "docker exec $DECODE_CONT  mkdir -p ${HF3FS_DATA_DIR_CTR}"

    log "starting hf3fs metadata server on $PREFILL_NODE ..."
    dexec_bg "$PREFILL_NODE" "$PREFILL_CONT" "bash ${WORKDIR_CTR}/meta_launch.sh > ${HF3FS_DATA_DIR_CTR}/meta.log 2>&1"
    wait_http "$PREFILL_NODE" "$META_PORT" "/docs" "hf3fs metadata server"

    log "launching prefill (HiCache L2 + HF3FS L3) on $PREFILL_NODE ..."
    dexec_bg "$PREFILL_NODE" "$PREFILL_CONT" "bash ${WORKDIR_CTR}/prefill_launch.sh > ${HF3FS_DATA_DIR_CTR}/prefill.log 2>&1"
    log "launching decode on $DECODE_NODE ..."
    dexec_bg "$DECODE_NODE" "$DECODE_CONT" "bash ${WORKDIR_CTR}/decode_launch.sh > ${HF3FS_DATA_DIR_CTR}/decode.log 2>&1"

    wait_http "$PREFILL_NODE" "$SERVER_PORT" "/health" "prefill server"
    wait_http "$DECODE_NODE"  "$SERVER_PORT" "/health" "decode server"

    log "launching PD router on $PREFILL_NODE ..."
    dexec_bg "$PREFILL_NODE" "$PREFILL_CONT" "bash ${WORKDIR_CTR}/router_launch.sh > ${HF3FS_DATA_DIR_CTR}/router.log 2>&1"
    wait_http "$PREFILL_NODE" "$ROUTER_PORT" "/health" "PD router"

    log "verifying HF3FS HiCache wired in prefill log ..."
    $SSH "$PREFILL_NODE" "grep -iE 'HiCacheHF3FS|hf3fs|Hf3fsMockClient|hierarchical' ${HF3FS_DATA_DIR_CTR}/prefill.log | tail -8" || true
    log "PD stack is UP. router=http://${PREFILL_IP}:${ROUTER_PORT}"
}

cmd_bench() {
    resolve_ips
    wait_http "$PREFILL_NODE" "$ROUTER_PORT" "/health" "PD router"
    log "running aiperf sweep (CONC='${CONC}', DURATION=${DURATION}s) ..."
    dexec "$PREFILL_NODE" "$PREFILL_CONT" "bash ${WORKDIR_CTR}/aiperf_run.sh"
    log "results under host ${WORKDIR_HOST}/results"
}

cmd_status() {
    resolve_ips
    for spec in "metadata:$PREFILL_NODE:$META_PORT:/docs" \
                "prefill:$PREFILL_NODE:$SERVER_PORT:/health" \
                "decode:$DECODE_NODE:$SERVER_PORT:/health" \
                "router:$PREFILL_NODE:$ROUTER_PORT:/health"; do
        IFS=: read -r label node port path <<< "$spec"
        if $SSH "$node" "curl -sf -o /dev/null --max-time 3 http://localhost:${port}${path}" 2>/dev/null; then
            printf '  %-9s %-14s :%-6s  UP\n' "$label" "$node" "$port"
        else
            printf '  %-9s %-14s :%-6s  DOWN\n' "$label" "$node" "$port"
        fi
    done
}

cmd_logs() {
    for f in meta prefill router; do
        echo "===== $PREFILL_NODE: ${f}.log (tail) ====="
        $SSH "$PREFILL_NODE" "tail -n 30 ${HF3FS_DATA_DIR_CTR}/${f}.log 2>/dev/null" || true
    done
    echo "===== $DECODE_NODE: decode.log (tail) ====="
    $SSH "$DECODE_NODE" "tail -n 30 ${HF3FS_DATA_DIR_CTR}/decode.log 2>/dev/null" || true
}

cmd_down() {
    log "removing PD containers ..."
    $SSH "$PREFILL_NODE" "docker rm -f $PREFILL_CONT >/dev/null 2>&1 || true"
    $SSH "$DECODE_NODE"  "docker rm -f $DECODE_CONT  >/dev/null 2>&1 || true"
    log "done."
}

case "${1:-all}" in
    render) resolve_ips; write_inner_scripts ;;
    up)     cmd_up ;;
    bench)  cmd_bench ;;
    all)    cmd_up; cmd_bench ;;
    status) cmd_status ;;
    logs)   cmd_logs ;;
    down)   cmd_down ;;
    *) die "unknown command '$1' (use: up | bench | all | status | logs | down)" ;;
esac
