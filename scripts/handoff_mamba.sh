#!/bin/bash
# 3090：Mamba-UIE 训练退出 → 自动跑 PSNR/SSIM/MAE 评测
while pgrep -u jmuaia -f "main.py --batchSize 2" > /dev/null; do sleep 300; done
sleep 60
cd /mnt/data/underwater/bpinn_uq
/home/jmuaia/anaconda3/envs/bpinn_uq/bin/python code/eval_mambauie_psnr.py > logs/eval_mambauie.log 2>&1
echo "$(date) Mamba-UIE eval finished" >> /mnt/data/underwater/bpinn_uq/logs/handoff_mamba.done
