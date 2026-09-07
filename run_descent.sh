#!/bin/bash
cd /home/work/exp; source env.sh
python3 -m nqx.gauge.givens pilot   --block 0                                    > /home/work/logs/givens_pilot.log 2>&1
python3 -m nqx.gauge.givens descent --blocks 0 --passes 4 --mode_desc cyclic     > /home/work/logs/desc_cyclic.log 2>&1
python3 -m nqx.gauge.givens descent --blocks 0 --passes 1 --mode_desc greedy --max_moves 400 > /home/work/logs/desc_greedy.log 2>&1
python3 -m nqx.gauge.arms 4 4fs 4greedy                                          > /home/work/logs/arm4.log 2>&1
echo ALLDONE >> /home/work/logs/arm4.log
