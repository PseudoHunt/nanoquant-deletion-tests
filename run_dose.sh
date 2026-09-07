#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
until grep -q "FINAL held-out" $L/desc_spread.log 2>/dev/null; do sleep 30; done

# baseline Step-3 sign-change mask, for the overlap diagnostic
python3 -m nqx.gauge.arms 0a_r7 > $L/mask.log 2>&1

# dose = 9 blocks (0,1,2,3 + 10,20,30,40,49): 3 Step-3 replicates
python3 -m nqx.gauge.arms rep4:spread:3 > $L/dose09.log 2>&1
# re-score the 4-block dose with replicates too
python3 -m nqx.gauge.arms rep4:b0123:3 > $L/dose04.log 2>&1

# extend to 16 blocks, then 3 replicates
python3 -m nqx.gauge.givens descent --blocks 5,15,25,35,45,12,22 --passes 1 \
  --mode_desc cyclic --oracle quad_oracle_fp64.pt \
  --init_R /home/work/exp/artifacts/gauge/givens_R_cyclic_spread.pt \
  --tag _d16 > $L/desc_d16.log 2>&1
python3 -m nqx.gauge.arms rep4:d16:3 > $L/dose16.log 2>&1
echo DOSE_DONE >> $L/dose16.log
