#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
until grep -q "FINAL held-out" $L/desc_b0123.log 2>/dev/null; do sleep 30; done
python3 -m nqx.gauge.arms 4b0123 > $L/arm4b.log 2>&1

# fp64 H and G, and a like-for-like re-run of block 0 greedy to see if the
# oracle's fp32 accumulation was steering the search
python3 -m nqx.gauge.quad > $L/quad64.log 2>&1
python3 -m nqx.gauge.givens descent --blocks 0 --passes 1 --mode_desc greedy \
  --max_moves 400 --oracle quad_oracle_fp64.pt --tag _fp64 > $L/desc_greedy_fp64.log 2>&1

# the decisive run: every one of the 50 rank blocks, sequentially, with a real
# bf16 checkpoint after each completed 32-coordinate block
ALL=$(python3 -c "print(','.join(str(i) for i in range(50)))")
python3 -m nqx.gauge.givens descent --blocks $ALL --passes 1 --mode_desc cyclic \
  --oracle quad_oracle_fp64.pt --tag _all50 > $L/desc_all50.log 2>&1
python3 -m nqx.gauge.arms 4all50 > $L/arm4_all50.log 2>&1

# then a RANDOM repartition on top of that result: a simultaneous latent
# permutation leaves the model invariant, so this reaches pairs the consecutive
# partition calls "cross-block" without enumerating 1.28M planes
python3 -m nqx.gauge.givens descent --blocks $ALL --passes 1 --mode_desc cyclic \
  --partition random --partition_seed 11 --oracle quad_oracle_fp64.pt \
  --init_R /home/work/exp/artifacts/gauge/givens_R_cyclic_all50.pt \
  --tag _rand1 > $L/desc_rand1.log 2>&1
python3 -m nqx.gauge.arms 4rand1 > $L/arm4_rand1.log 2>&1
echo ALLDONE >> $L/arm4_rand1.log
