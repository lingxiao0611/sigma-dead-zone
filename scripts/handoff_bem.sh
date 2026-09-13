#!/bin/bash
# 4090-A：BEM 训练退出 → 提取训练日志中的 val PSNR 汇总
while pgrep -u jss -f "basicsr.train" > /dev/null; do sleep 300; done
sleep 30
L=/mnt/data/underwater/bpinn_uq/logs/train_bem.log
OUT=/mnt/data/underwater/bpinn_uq/outputs/bem_val_summary.txt
grep -i "psnr" "$L" | tail -10 > "$OUT" 2>/dev/null
tail -3 "$L" >> "$OUT" 2>/dev/null
echo "$(date) BEM summary extracted" >> /mnt/data/underwater/bpinn_uq/logs/handoff_bem.done
