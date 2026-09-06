#!/bin/bash
# Single-block harness matrix: determinism check, then baseline -> A -> A+Change4 -> B -> A+B
set -u
cd /home/work/exp
export HF_HOME=/home/work/hf_cache PYTHONPATH=/home/work/exp/nqx PYTHONUNBUFFERED=1
MID=/home/work/hf_cache/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da

push_arm () {
  /home/work/venv/bin/python collect.py --json > results/nanoquant/_sbh_table.txt 2>&1
  git add -A
  git commit -qm "single-block harness: $1

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YAQwZnt4tN9kRonGomC5bi" || true
  git push -q origin master || echo "[chain] push failed"
}

run_arm () {
  echo "[chain] === arm $1 ($2) start $(date -u +%H:%M:%S) ==="
  ./sbh_arm.sh "$1" "$2" "$3"
  push_arm "arm $1 complete"
  echo "[chain] === arm $1 done $(date -u +%H:%M:%S) ==="
}

run_arm base  stock  '{}'

# determinism gate: identical config, identical seed, fresh process
echo "[chain] === determinism duplicate ==="
cd /home/work/NanoQuant
/home/work/venv/bin/python -u /home/work/exp/nqx/sbh.py run --model_id "$MID" \
  --tag dup_b0_s0 --block 0 --seed 0 --variant stock \
  2>&1 | /home/work/venv/bin/python -u /home/work/exp/ts.py > /home/work/exp/logs/dup_b0_s0.log
cd /home/work/exp
push_arm "determinism duplicate run"

run_arm A     testA  '{}'
run_arm A4    testA  '{"als_rounds":3}'
run_arm B     testB  '{}'
run_arm AB    testAB '{}'
push_arm "matrix complete"
echo "[chain] ALL ARMS DONE $(date -u)"
echo "[chain] === Appendix C config repro run start $(date -u) ==="
/home/work/exp/run_paper_cfg.sh 0 repro_paper_seed0
/home/work/venv/bin/python /home/work/exp/parse_step0.py /home/work/exp/logs/repro_paper_seed0.log \
    /home/work/exp/artifacts/step0_paper_seed0.json > /home/work/exp/logs/parse_paper.txt 2>&1
push_arm "Appendix C config repro run"
echo "[chain] DONE $(date -u)"
