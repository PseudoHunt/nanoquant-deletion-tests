#!/bin/bash
set -u
export HF_HOME=/home/work/hf_cache PYTHONPATH=/home/work/exp/nqx PYTHONUNBUFFERED=1
cd /home/work/NanoQuant
cell () {  # tag mode eta
  if [ -f "/home/work/exp/artifacts/testD/runs/$1.json" ]; then echo "[skip] $1"; return; fi
  echo "=== $1 ($2 eta=$3) $(date -u +%H:%M:%S) ==="
  /home/work/venv/bin/python -u /home/work/exp/nqx/testD.py run --tag "$1" --mode "$2" --eta "$3" \
    > /home/work/exp/logs/testD_$1.log 2>&1
  grep -a "\[D\] FINAL" /home/work/exp/logs/testD_$1.log || tail -3 /home/work/exp/logs/testD_$1.log
}
cell ctrl_ste  ste     0
for E in 3e-3 1e-2 3e-2 1e-1; do cell "R_$E" relaxed "$E"; done
for E in 3e-3 1e-2 3e-2 1e-1; do cell "H_$E" hard    "$E"; done
echo "=== TEST D CELLS DONE $(date -u) ==="
