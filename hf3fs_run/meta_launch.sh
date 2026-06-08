#!/usr/bin/env bash
set -x
mkdir -p /run_logs/hf3fs_pd
exec python3 -m sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server \
    --host 0.0.0.0 --port 18000
