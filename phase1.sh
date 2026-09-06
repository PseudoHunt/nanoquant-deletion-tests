#!/bin/bash
# Phase 1: seed 0 only, all arms, both blocks, the two blocks run concurrently.
# Seeds 1-2 are spent later, only on arms that beat baseline on both blocks.
set -u
cd /home/work/exp
export HF_HOME=/home/work/hf_cache PYTHONPATH=/home/work/exp/nqx PYTHONUNBUFFERED=1
MID=/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da

cell () {  # arm variant opts block seed
  local TAG="$1_b$4_s$5"
  if [ -f "artifacts/sbh_runs/${TAG}.json" ] && grep -q '"block_post"' "artifacts/sbh_runs/${TAG}.json" 2>/dev/null; then
    echo "[skip] $TAG"; return
  fi
  (cd /home/work/NanoQuant && /home/work/venv/bin/python -u /home/work/exp/nqx/sbh.py run \
      --model_id "$MID" --tag "$TAG" --block "$4" --seed "$5" \
      --variant "$2" --variant_opts "$3" 2>&1 \
    | /home/work/venv/bin/python -u /home/work/exp/ts.py > /home/work/exp/logs/${TAG}.log)
}

pair () {  # arm variant opts seed -> both blocks, sequentially
  # Measured: the two blocks run concurrently took 602 s wall against ~640 s
  # sequential -- 6% saved -- and OOM'd the KL stage, which needs a second
  # resident model. Not worth it on a 24 GB card.
  echo "[p1] === $1 seed $4, blocks 0 and 8, start $(date -u +%H:%M:%S) ==="
  local T0=$SECONDS
  cell "$1" "$2" "$3" 0 "$4"
  cell "$1" "$2" "$3" 8 "$4"
  echo "[p1] === $1 seed $4 done in $((SECONDS-T0))s at $(date -u +%H:%M:%S) ==="
}

push () {
  /home/work/venv/bin/python decide.py > results/nanoquant/_sbh_table.txt 2>&1
  git add -A
  git commit -qm "single-block harness: $1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YAQwZnt4tN9kRonGomC5bi" || true
  git push -q origin master || echo "[p1] push failed"
}

pair base  stock  '{}'     0 ; push "phase 1: baseline seed 0"
pair dup   stock  "{}"     0 ; push "phase 1: determinism floor"
pair A     testA  '{}'     0 ; push "phase 1: Test A seed 0"
pair B     testB  '{}'     0 ; push "phase 1: Test B seed 0"
pair AB    testAB '{}'     0 ; push "phase 1: Test A+B seed 0"
echo "[p1] PHASE 1 COMPLETE $(date -u)"
/home/work/venv/bin/python decide.py
