#!/bin/bash
# 补强套件 A/B/C：切分稳定性 → proxy 敏感性+机理 → U-shape 分层。顺序执行，3090 GPU0。
cd /mnt/data/underwater/bpinn_uq
PY=/home/jmuaia/anaconda3/envs/bpinn_uq/bin/python
CUDA_VISIBLE_DEVICES=0 $PY code/eval_split_stability.py > logs/split_stability.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY code/eval_strat_proxy_evidence.py > logs/strat_proxy_evidence.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY code/eval_ushape_strat.py > logs/ushape_strat.log 2>&1
echo "$(date) robustness suite done" >> logs/robustness_suite.done
