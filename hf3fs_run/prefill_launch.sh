#!/usr/bin/env bash
set -x
mkdir -p /run_logs/hf3fs_pd
export SGLANG_HICACHE_HF3FS_CONFIG_PATH=/workspace/hf3fs_run/hf3fs_config.json
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/python3.12/dist-packages"
export SGLANG_MORI_COMBINE_DTYPE=fp8_direct_cast
export SGLANG_MORI_MOE_MAX_INPUT_TOKENS=32768
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192
export MORI_RDMA_TC=104 MORI_RDMA_SL=3 MORI_IO_TC=104 MORI_IO_SL=3
exec python3 -m sglang.launch_server \
    --model-path /models/DeepSeek-R1-0528-MXFP4-v2 \
    --disaggregation-mode prefill \
    --disaggregation-ib-device rdma3,rdma0,rdma2,rdma1,rdma7,rdma4,rdma6,rdma5 \
    --host 0.0.0.0 --port 8000 --trust-remote-code \
    --tp-size 8 --decode-log-interval 1000 --log-level warning --watchdog-timeout 3600 \
    --ep-dispatch-algorithm fake --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 \
    --attention-backend aiter --disaggregation-transfer-backend mori \
    --mem-fraction-static 0.8 --max-running-requests 128 \
    --chunked-prefill-size 16384 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 \
    --context-length 163840 \
    --page-size 64 --enable-hierarchical-cache \
    --hicache-size 125 --hicache-io-backend kernel --hicache-mem-layout layer_first \
    --hicache-write-policy write_through \
    --hicache-storage-backend hf3fs \
    --hicache-storage-backend-extra-config '{"use_mock_hf3fs_client": true}' \
    --enable-metrics --enable-cache-report
