#!/usr/bin/env bash
set -euo pipefail

# Local test: agentic trace replay benchmark against a disaggregated
# SGLang server on MI355X (DeepSeek-R1-0528 MXFP4-v2, 1P1D TP8).
#
# This submits a SLURM job that:
#   1. Starts the disaggregated SGLang server (prefill + decode nodes)
#   2. Runs the agentic trace replay benchmark (aiperf) instead of
#      the fixed-seq-len benchmark_serving.py
#
# Prerequisites:
#   git submodule update --init utils/aiperf
#
# Usage:
#   bash local_test_agentic.sh
#   CONC=8 DURATION=300 bash local_test_agentic.sh        # single conc
#   CONC=1x2x4x8 bash local_test_agentic.sh               # conc sweep ('x'-delimited)
#   KEEP_CONTAINERS=1 bash local_test_agentic.sh          # leave server containers up after run

# ── Cluster / SLURM ──
export SLURM_ACCOUNT="$USER"
export SLURM_PARTITION="compute"
export TIME_LIMIT="08:00:00"
export SLURM_REUSE_JOBID="8429"

# ── Model ──
export MODEL_PATH="/it-share/hf_cache"
export MODEL_NAME="DeepSeek-R1-0528-MXFP4-v2"
# MODEL must match the --model-path the SGLang server uses inside Docker.
# server_sglang.sh launches with --model-path $MODEL_DIR/$MODEL_NAME which
# resolves to /models/DeepSeek-R1-0528-MXFP4-v2 inside the container.  Leave
# MODEL unset here so bench.sh falls back to BENCH_MODEL (= MODEL_PATH inside
# Docker = /models/<MODEL_NAME>).
# export MODEL="amd/DeepSeek-R1-0528-MXFP4-v2"
export MODEL_PREFIX="dsr1"

# ── Prefill context length (temporary) ──
# Cap the prefill server context so over-length agentic prompts are rejected
# instead of reaching the AITER MLA prefill kernel and faulting the GPU
# (HSA_STATUS_ERROR_EXCEPTION). Matches the model's native max (163840).
export PREFILL_CONTEXT_LENGTH="${PREFILL_CONTEXT_LENGTH:-163840}"

# ── Container ──
export CONTAINER_IMAGE="lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260519"
# export CONTAINER_IMAGE="lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260529"


# ── Framework ──
export FRAMEWORK="sglang-disagg"
export PRECISION="fp4"

# ── Runner / job name ──
export RUNNER_NAME="${USER}-dsr1-fp4-agentic-test"
export RESULT_FILENAME="${RUNNER_NAME}"

# ── Agentic benchmark params ──
# DURATION threads through submit.sh → job.slurm → Docker → bench.sh.
# CONC drives the concurrency list: submit.sh passes it as
# BENCH_MAX_CONCURRENCY, which bench.sh splits (on 'x') into loop iterations.
export DURATION="${DURATION:-1800}"
# conc-list from amd-master.yaml: [ 1, 2, 4, 8 ] (delimited by 'x')
CONC="${CONC:-1x2x4x8x16x32}"

# Keep the prefill/decode/router containers running after the benchmark
# finishes (skips job.slurm teardown). Set to 1 to inspect or reuse the
# live servers; default 0 tears them down as usual.
export KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"

# spec-decoding: "none" in amd-master.yaml
export SPEC_DECODING="none"

# ── KV cache offloading (HiCache) ──
# OFFLOADING=none (default) | hicache  (SGLang hierarchical cache; CPU DRAM tier).
# When set to "hicache", server_sglang.sh keeps RadixAttention on (drops
# --disable-radix-cache) and adds --hicache-* flags to BOTH prefill and decode.
# The tunables below are optional overrides; HICACHE_SIZE_GB, when unset, is
# derived from the node-total DRAM budget / TP / host-pool count.
export OFFLOADING="${OFFLOADING:-none}"
export HICACHE_TOTAL_CPU_DRAM_GB="${HICACHE_TOTAL_CPU_DRAM_GB:-1000}"
export HICACHE_HOST_POOL_COUNT="${HICACHE_HOST_POOL_COUNT:-1}"
export HICACHE_PAGE_SIZE="${HICACHE_PAGE_SIZE:-64}"
# Leave layout/io-backend/write-policy unset so server_sglang.sh auto-selects
# the backend-correct combo:
#   mooncake L3 -> page_first_direct + direct + write_through
#   L2-only     -> layer_first      + kernel + write_through_selective
# (page_first_direct asserts host_pool > device_pool and is Mooncake-only; the
#  "direct" IO backend requires a page_first layout, so L2 must use layer_first.)
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-}"
export HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-}"

# ── HiCache L3 storage backend (Mooncake) ──
# HICACHE_STORAGE_BACKEND=""(default, CPU-DRAM L2 only) | "mooncake".
# When "mooncake", server_sglang.sh starts a mooncake_master on node 0 and adds
# --hicache-storage-backend mooncake (+extra-config) to the prefill server.
# Mooncake requires a page_first layout; with HICACHE_IO_BACKEND=direct the
# canonical pairing is page_first_direct (set above).  If unset, server_sglang.sh
# defaults the layout/write-policy to Mooncake-compatible values.  MC_DEVICE/
# MC_MASTER_ADDR default to the auto-detected IBDEVICES / NODE0_ADDR in-container.
#
# HiCache is PREFILL-ONLY by default: the decode server in disaggregation mode
# forces chunk cache unless --disaggregation-decode-enable-radix-cache is used,
# which requires a nixl/mooncake transfer backend (we use mori).  Set
# HICACHE_DECODE=1 only if you switch the transfer backend to nixl/mooncake.
export HICACHE_STORAGE_BACKEND="${HICACHE_STORAGE_BACKEND:-}"
export HICACHE_DECODE="${HICACHE_DECODE:-0}"
# NOTE: these are SHARED nodes. The Mooncake defaults (50061/9003) are often
# already occupied by another user's mooncake_master, which makes our master
# fail to bind its RPC port and the prefill hang connecting to the foreign one.
# Use non-default ports (verified free) to avoid the collision.
export MC_MASTER_PORT="${MC_MASTER_PORT:-58137}"
export MC_METRICS_PORT="${MC_METRICS_PORT:-19003}"
# The image's Mooncake may predate 0.3.8.post1, whose fallback allocator can't
# RDMA-register the store segment (ibv_reg_mr ENOMEM).  MC_UPGRADE=1 upgrades
# mooncake-transfer-engine in-container at launch to enable the RDMA allocator.
export MC_UPGRADE="${MC_UPGRADE:-1}"
# Soften the page_first_direct "host pool must be > device pool" assert into a
# warning so the server starts even when the host L2 pool is sized smaller.
export MC_PATCH_HOSTPOOL="${MC_PATCH_HOSTPOOL:-1}"
export MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
export MC_GLOBAL_SEG="${MC_GLOBAL_SEG:-8gb}"
export MC_DEVICE="${MC_DEVICE:-rdma0}"
export MC_MASTER_ADDR="${MC_MASTER_ADDR:-}"

# ── Disaggregation transfer backend + decode radix cache ──
# The KV transfer engine between prefill and decode.  models.yaml base_flags
# defaults to "mori"; mori forces chunk cache on the decode server.  To run the
# decode server WITH radix cache, use a nixl/mooncake transfer backend and set
# DECODE_RADIX_CACHE=1 (adds --disaggregation-decode-enable-radix-cache).
# This is independent of HiCache/OFFLOADING (no CPU-DRAM offloading needed).
# NOTE: the mooncake transfer engine RDMA-registers HOST staging buffers, which
# collides with the large pinned HiCache L2 host pool (ibv_reg_mr -> ENOMEM) and
# leaves the prefill server unhealthy (/health 503).  mori registers GPU KV
# buffers instead, so it coexists with HiCache L2.  mori forces chunk cache on
# decode (no decode radix), which is fine since HiCache here is prefill-only.
export DISAGG_TRANSFER_BACKEND="${DISAGG_TRANSFER_BACKEND:-mooncake}"
# Enable decode-side radix cache so the cache-aware router has prefix state to
# route on (requires nixl/mooncake transfer backend; mooncake set above). With
# OFFLOADING=none this radix cache lives in VRAM only (no CPU-DRAM offload).
export DECODE_RADIX_CACHE="${DECODE_RADIX_CACHE:-1}"

# ── SGLang PD router policy ──
# cache_aware routes requests to the decode worker with the best prefix-cache
# affinity (radix-tree approximation in the router). Threads through job.slurm
# into server_sglang.sh's sglang_router.launch_router (--policy/--decode-policy).
# Needs >1 decode worker + decode radix cache (DECODE_RADIX_CACHE=1 above) to help.
export ROUTER_POLICY="${ROUTER_POLICY:-random}"

# ── Topology (from amd-master.yaml dsr1-fp4-mi355x-sglang-disagg, 1P1D TP8) ──
PREFILL_NODES=1
PREFILL_WORKERS=1
DECODE_NODES=1
DECODE_WORKERS=1
ISL=1024
OSL=1024
CONCURRENCIES="$CONC"
REQUEST_RATE="inf"

# ── Parallelism (1P1D TP8) ──
#   Prefill: TP=8, EP=1 (no EP), no DP-attn
#   Decode:  TP=8, EP=1 (no EP), no DP-attn, MTP=0
PREFILL_ENABLE_EP=false
PREFILL_ENABLE_DP=false
DECODE_ENABLE_EP=false
DECODE_ENABLE_DP=false
export DECODE_MTP_SIZE=0

PREFILL_TP=8
DECODE_TP=8
RANDOM_RANGE_RATIO=0.8

cd "$(dirname "$0")/benchmarks/multi_node/amd_utils"

bash ./submit.sh \
    "$PREFILL_NODES" "$PREFILL_WORKERS" \
    "$DECODE_NODES"  "$DECODE_WORKERS" \
    "$ISL" "$OSL" "$CONCURRENCIES" "$REQUEST_RATE" \
    "$PREFILL_ENABLE_EP" "$PREFILL_ENABLE_DP" \
    "$DECODE_ENABLE_EP"  "$DECODE_ENABLE_DP" \
    "$PREFILL_TP" "$DECODE_TP" \
    "$RANDOM_RANGE_RATIO" \
    "mia1-p01-g05,mia1-p01-g07"
