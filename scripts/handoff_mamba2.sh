#!/bin/bash
# 3090：等 Mamba-UIE 评测完成后（handoff_mamba.done），自动跑预算消融
while [ ! -f /mnt/data/underwater/bpinn_uq/logs/handoff_mamba.done ]; do sleep 300; done
sleep 30
cd /mnt/data/underwater/bpinn_uq
/home/jmuaia/anaconda3/envs/bpinn_uq/bin/python code/eval_mambauie_ablation.py > logs/eval_mambauie_ablation.log 2>&1
echo "$(date) mamba ablation finished" >> /mnt/data/underwater/bpinn_uq/logs/handoff_mamba2.done
