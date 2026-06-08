#!/usr/bin/env bash
set -x
exec python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --port 30000 \
    --policy random --prefill-policy random --decode-policy random \
    --prefill http://10.24.112.181:8000 \
    --decode http://10.24.112.182:8000
