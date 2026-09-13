#!/bin/bash
# 4090-B：PUIE-Net 训练退出 → 自动跑同协议评测（GPU1）
while pgrep -u jss -f "puie_net/Train.py" > /dev/null; do sleep 120; done
sleep 60
cd /mnt/data/underwater/bpinn_uq
CUDA_VISIBLE_DEVICES=1 /mnt/datadisk/jss/conda_envs/udwt1/bin/python code/eval_puie_calib.py > logs/eval_puie.log 2>&1
echo "$(date) PUIE eval finished" >> /mnt/data/underwater/bpinn_uq/logs/handoff_puie.done
