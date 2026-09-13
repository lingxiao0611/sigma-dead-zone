#!/bin/bash
# 3090 补强链（GPU0 与 fp32 重训共享）：E1 干预 → E2 朴素基线 → E3 无参考响应 → E5 机理探针
set -u
cd /mnt/data/underwater/bpinn_uq
PY=/home/jmuaia/anaconda3/envs/bpinn_uq/bin/python

echo "[suite] start $(date)" > logs/boost_suite.log
$PY code/eval_head_intervention.py   > logs/head_intervention.log   2>&1 && echo "E1 ok"  >> logs/boost_suite.log || echo "E1 FAIL" >> logs/boost_suite.log
$PY code/eval_naive_baselines.py     > logs/naive_baselines.log     2>&1 && echo "E2 ok"  >> logs/boost_suite.log || echo "E2 FAIL" >> logs/boost_suite.log
$PY code/eval_reffree_response.py    > logs/reffree_response.log    2>&1 && echo "E3 ok"  >> logs/boost_suite.log || echo "E3 FAIL" >> logs/boost_suite.log
$PY code/eval_mechanism_probes.py    > logs/mechanism_probes.log    2>&1 && echo "E5 ok"  >> logs/boost_suite.log || echo "E5 FAIL" >> logs/boost_suite.log
touch logs/boost_suite.done
echo "[suite] all done $(date)" >> logs/boost_suite.log
