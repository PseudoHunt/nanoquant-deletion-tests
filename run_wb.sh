#!/bin/bash
cd /home/work/exp; source env.sh
L=/home/work/logs
until grep -q DOSE_DONE $L/dose16.log 2>/dev/null; do sleep 30; done
python3 -m nqx.gauge.wholeblock --nb 8 --sweeps 2 > $L/wholeblock.log 2>&1
python3 -m nqx.gauge.arms rep5:-:3 > $L/wb_arms.log 2>&1
echo WB_DONE >> $L/wb_arms.log
