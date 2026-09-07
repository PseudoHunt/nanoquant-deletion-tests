#!/bin/bash
source env.sh
python3 -m nqx.gauge.givens pilot   --block 0            > /home/work/logs/givens_pilot.log 2>&1
python3 -m nqx.gauge.givens descent --blocks 0 --passes 4 > /home/work/logs/givens_descent.log 2>&1
echo ALLDONE >> /home/work/logs/givens_descent.log
