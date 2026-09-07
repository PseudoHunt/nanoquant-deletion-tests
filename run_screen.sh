#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
python3 -m nqx.gauge.arms 4b0123 > $L/arm4b.log 2>&1
python3 -m nqx.gauge.quad > $L/quad64.log 2>&1
# calibrate top-K against block 0's known exact greedy result (-0.0418%)
for K in 16 32 64; do
  python3 -m nqx.gauge.givens descent --blocks 0 --mode_desc screen --topk $K \
    --oracle quad_oracle_fp64.pt --tag _k$K --min_gain -1 > $L/screen_k$K.log 2>&1
done
echo KCAL_DONE >> $L/screen_k64.log
