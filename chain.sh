#!/bin/bash
# Wait for the full-model Step 0 run to finish, then walk the single-block
# harness matrix: baseline -> A -> A+Change4 -> B -> A+B, 2 blocks x 3 seeds each.
set -u
cd /home/work/exp
while pgrep -f "nanoquant.main" > /dev/null; do sleep 20; done
echo "[chain] step0 finished at $(date)"
while pgrep -f "sbh.py cache" > /dev/null; do sleep 10; done
echo "[chain] cache finished at $(date)"

run_arm () {
  echo "[chain] === arm $1 ($2 $3) start $(date) ==="
  ./sbh_arm.sh "$1" "$2" "$3"
  /home/work/venv/bin/python collect.py --json > results/nanoquant/_sbh_table.txt 2>&1
  git add -A && git commit -qm "single-block harness: arm $1 complete

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YAQwZnt4tN9kRonGomC5bi" || true
  git push -q origin master || echo "[chain] push failed"
  echo "[chain] === arm $1 done $(date) ==="
}

run_arm base  stock  '{}'
run_arm A     testA  '{}'
run_arm A4    testA  '{"als_rounds":3}'
run_arm B     testB  '{}'
run_arm AB    testAB '{}'
echo "[chain] ALL ARMS DONE $(date)"
