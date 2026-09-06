#!/bin/bash
set -u
export HF_HOME=/home/work/hf_cache PYTHONPATH=/home/work/exp/nqx PYTHONUNBUFFERED=1
cd /home/work/NanoQuant
for V in R2 H2; do
  for INIT in conf corner; do
    for LR in 1e-4 3e-4 1e-3 3e-3; do
      TAG="${V}_${INIT}_${LR}"
      if [ -f "/home/work/exp/artifacts/testD2/runs/$TAG.json" ]; then echo "[skip] $TAG"; continue; fi
      echo "=== $TAG $(date -u +%H:%M:%S) ==="
      /home/work/venv/bin/python -u /home/work/exp/nqx/testD2.py --tag "$TAG" \
        --variant "$V" --init "$INIT" --lr "$LR" > /home/work/exp/logs/testD2_$TAG.log 2>&1
      grep -a "\[D2\] FINAL" /home/work/exp/logs/testD2_$TAG.log || tail -3 /home/work/exp/logs/testD2_$TAG.log
    done
  done
done
echo "=== D2 DONE $(date -u) ==="
