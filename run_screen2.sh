#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
until grep -q KCAL_DONE $L/screen_k64.log 2>/dev/null; do sleep 20; done
# replay-floor distribution: six re-runs of the identical 0a state through the
# identical common Step 3.  Every significance claim is scaled by this number and
# it was previously estimated from a single pair.
python3 -m nqx.gauge.arms 0a_r1 0a_r2 0a_r3 0a_r4 0a_r5 0a_r6 > $L/floor.log 2>&1
echo FLOOR_DONE >> $L/floor.log
