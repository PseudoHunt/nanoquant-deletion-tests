#!/bin/bash
# Run one arm of the single-block harness across blocks {0, N/2} and seeds {0,1,2}
set -u
ARM=$1          # tag prefix
VARIANT=$2      # stock | testA | testB | testAB
OPTS=${3:-'{}'}
export HF_HOME=/home/work/hf_cache PYTHONPATH=/home/work/exp/nqx PYTHONUNBUFFERED=1
MID=/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da
cd /home/work/NanoQuant
for B in 0 8; do
  for S in 0 1 2; do
    TAG="${ARM}_b${B}_s${S}"
    if [ -f "/home/work/exp/artifacts/sbh_runs/${TAG}.json" ] && \
       grep -q '"block_post"' "/home/work/exp/artifacts/sbh_runs/${TAG}.json" 2>/dev/null; then
      echo "[skip] $TAG already complete"; continue
    fi
    echo "=== $TAG ==="
    /home/work/venv/bin/python -u /home/work/exp/nqx/sbh.py run \
      --model_id "$MID" --tag "$TAG" --block "$B" --seed "$S" \
      --variant "$VARIANT" --variant_opts "$OPTS" \
      2>&1 | /home/work/venv/bin/python -u /home/work/exp/ts.py > /home/work/exp/logs/${TAG}.log
    tail -1 /home/work/exp/logs/${TAG}.log
  done
done
