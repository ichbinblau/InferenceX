#!/usr/bin/env bash
set -euo pipefail

# Local test: agentic trace replay benchmark against a disaggregated vLLM
# server on MI355X (Kimi-K2.5-MXFP4, 1P2D TP8, decode EP8).
#
# Mirrors local_test_agentic.sh but retargets the SGLang/DeepSeek 1P1D setup
# to the amd-master.yaml `kimik2.5-fp4-mi355x-vllm-disagg` recipe:
#   framework: vllm-disagg, 1 prefill node (TP8, EP1) + 2 decode nodes
#   (TP8, EP8), vLLM external router (consistent_hash) on the prefill node.
#
# This submits a SLURM job that:
#   1. Starts the disaggregated vLLM server (1 prefill + 2 decode nodes)
#   2. Runs the agentic trace replay benchmark (aiperf)
#
# Prerequisites:
#   git submodule update --init utils/aiperf
#
# Usage:
#   bash local_test_kimi_disagg_agentic.sh
#   CONC=8 DURATION=300 bash local_test_kimi_disagg_agentic.sh        # single conc
#   CONC=8x16x32 bash local_test_kimi_disagg_agentic.sh               # conc sweep ('x'-delimited)
#   SLURM_REUSE_JOBID=1234 bash local_test_kimi_disagg_agentic.sh     # reuse a >=3-node allocation
#   KEEP_CONTAINERS=1 bash local_test_kimi_disagg_agentic.sh          # leave containers up after run

# ── Cluster / SLURM ──
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-$USER}"
export SLURM_PARTITION="${SLURM_PARTITION:-amd-aim}"
export TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
# 1P2D needs 3 nodes. Leave SLURM_REUSE_JOBID unset to fresh-submit via sbatch;
# only set it to reuse an existing allocation that has >= 3 nodes.
export SLURM_REUSE_JOBID="${SLURM_REUSE_JOBID:-8271}"

# ── Model (key must match benchmarks/multi_node/amd_utils/models_vllm.yaml) ──
export MODEL_PATH="${MODEL_PATH:-/it-share/hf_cache}"
export MODEL_NAME="${MODEL_NAME:-Kimi-K2.5-MXFP4}"
# Use a flattened/dereferenced snapshot dir so the Kimi remote-code relative
# import (tokenization_kimi.py -> .tool_declaration_ts) resolves. The stock HF
# cache stores the *.py as symlinks into blobs/, which breaks transformers'
# get_relative_imports under trust-remote-code. job.slurm resolves this name
# under MODEL_DIR (=/it-share/hf_cache) and mounts that dir into the container.
export MODEL_DISK_DIR_OVERRIDE="${MODEL_DISK_DIR_OVERRIDE:-models--amd--Kimi-K2.5-MXFP4-flat}"
export MODEL_PREFIX="kimik2.5"
# Leave MODEL unset so bench.sh falls back to the vllm served-model-name
# (= MODEL_NAME for vllm-disagg).

# ── Container (from kimik2.5-fp4-mi355x-vllm-disagg) ──
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-aigmkt/vllm-openai-rocm:nightly-bf610c2f56764e1b30bc6065f4ceace3d6e59036}"

# ── Framework ──
export FRAMEWORK="vllm-disagg"
export PRECISION="fp4"

# ── RDMA devices ──
# The vLLM image's ibv_devinfo does not auto-detect the NICs (unlike the SGLang
# image), so env.sh aborts with "Unable to detect RDMA devices". Set IBDEVICES
# explicitly (matches runners/launch_mi355x-amds.sh) and job.slurm forwards it
# into the container.
export IBDEVICES="${IBDEVICES:-rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7}"

# The stock vLLM ROCm image lacks the AMD Pollara/AINIC "ionic" RDMA userspace
# provider (libionic), so libibverbs/MoRIIO enumerate 0 RDMA devices and mori
# aborts (Assertion `availDevices.size() > 0'). Bind-mount the host AINIC deb
# repo into the container; setup_deps.sh installs libionic from AINIC_DEB_REPO.
export AINIC_DEB_REPO="${AINIC_DEB_REPO:-/opt/ainic-repo}"
export EXTRA_DOCKER_MOUNTS="${EXTRA_DOCKER_MOUNTS:-} -v /opt/amd/ainic/deb-repo:/opt/ainic-repo:ro"

# ── Runner / job name ──
export RUNNER_NAME="${USER}-kimik2.5-fp4-disagg-agentic-test"
export RESULT_FILENAME="${RUNNER_NAME}"

# ── vLLM disaggregation knobs (from the recipe's additional-settings) ──
# MoRIIO connector read mode for the PD KV transfer; submit.sh also defaults
# this to 1 for vllm-disagg, set explicitly here to match the recipe.
export VLLM_MORIIO_CONNECTOR_READ_MODE="${VLLM_MORIIO_CONNECTOR_READ_MODE:-1}"
export PROXY_STREAM_IDLE_TIMEOUT="${PROXY_STREAM_IDLE_TIMEOUT:-300}"
# vLLM external router (job.slurm starts a vllm-router container on node 0).
export ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"

# ── Radix cache (vLLM prefix caching / APC) ──
# The Kimi-K2.5-MXFP4 entry in models_vllm.yaml ships --no-enable-prefix-caching.
# ENABLE_PREFIX_CACHING=1 makes server_vllm.sh strip that opt-out and add
# --enable-prefix-caching to BOTH the prefill and decode servers (radix cache on).
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

# ── Agentic replay context cap ──
# Kimi-K2.5 max_seq_len is 262144. The WEKA agentic corpus has long parent/
# subagent traces; resumed deep into a trace the accumulated context can exceed
# this and the server returns deterministic 4xxs (which aiperf surfaces as
# "Unsupported OpenAI object type: None" and trips --failed-request-threshold,
# cancelling the whole conc level — this is what killed conc32 previously).
# MAX_MODEL_LEN threads into bench.sh's build_replay_cmd as --max-context-length
# so aiperf keeps replay inputs within the window instead of overflowing.
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

# ── Agentic benchmark params ──
export DURATION="${DURATION:-1800}"
# Default agentic sweep (mirrors the kimik2.5 agentic sibling's conc-list,
# trimmed). Override CONC for the full fixed-seq-len sweep (8x16x32x64x...).
CONC="${CONC:-32x64}"

export KEEP_CONTAINERS="${KEEP_CONTAINERS:-0}"
export SPEC_DECODING="none"

# ── Topology (kimik2.5-fp4-mi355x-vllm-disagg, 1P2D) ──
#   Prefill: 1 node,  TP=8, EP=1 (no EP), no DP-attn
#   Decode:  2 nodes, TP=8, EP=8 (expert parallel), no DP-attn
PREFILL_NODES=1
PREFILL_WORKERS=1
DECODE_NODES=1
DECODE_WORKERS=1
ISL=1024
OSL=1024
CONCURRENCIES="$CONC"
REQUEST_RATE="inf"

PREFILL_ENABLE_EP=false
PREFILL_ENABLE_DP=false
DECODE_ENABLE_EP=true
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
    "$RANDOM_RANGE_RATIO"
