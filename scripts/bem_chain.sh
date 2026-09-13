#!/bin/bash
# 4090 GPU0：det sanity 全量核查 → 无论结果如何接着跑 BEM 全量 MC 校准评测
# 全程 setsid 守护，不依赖本地 ssh 连接
cd /mnt/datadisk/underwater/bpinn_uq
CUDA_VISIBLE_DEVICES=0 /mnt/datadisk/jss/conda_envs/udwt1/bin/python code/bem_det_check.py > logs/bem_det_check.log 2>&1
CUDA_VISIBLE_DEVICES=0 /mnt/datadisk/jss/conda_envs/udwt1/bin/python code/eval_bem_calib.py > logs/eval_bem.log 2>&1
echo "$(date) BEM chain done" >> logs/bem_chain.done
