#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
until grep -q FLOOR_DONE $L/floor.log 2>/dev/null; do sleep 20; done
# Exact cyclic sweeps on five WIDELY SPREAD rank blocks, continued from the
# blocks-0..3 state, with a real bf16 checkpoint after each completed block.
# Answers "are blocks 0-3 special?" directly; greedy-lite was dropped because
# top-K recovers only 32%/43% of the exact per-block gain (K=16/32).
python3 -m nqx.gauge.givens descent --blocks 10,20,30,40,49 --passes 1 \
  --mode_desc cyclic --oracle quad_oracle_fp64.pt \
  --init_R /home/work/exp/artifacts/gauge/givens_R_cyclic_b0123.pt \
  --tag _spread > $L/desc_spread.log 2>&1
python3 -m nqx.gauge.arms 4spread > $L/arm4_spread.log 2>&1
echo SPREAD_DONE >> $L/arm4_spread.log
