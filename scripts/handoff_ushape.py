# -*- coding: utf-8 -*-
"""增强②·自动接力：等 U-shape 基线训练结束 → 启动 U-shape EDL 训练。
后台运行（约 60min 等基线），EDL 起来后本脚本退出，EDL 独立训练。
"""
import subprocess, time, os

BASE = "/mnt/data/underwater/bpinn_uq"
LOG = os.path.join(BASE, "logs/handoff_edl.log")


def log(m):
    with open(LOG, "a") as f:
        f.write(time.strftime("[%H:%M:%S] ") + m + "\n")
    print(m, flush=True)


log("HANDOFF 启动：监控 train_ushape_baseline.py ...")

while True:
    r = subprocess.run(["pgrep", "-f", "train_ushape_baseline.py"],
                       capture_output=True, text=True)
    if not r.stdout.strip():
        log("baseline 进程已结束 → 接力启动 EDL")
        break
    time.sleep(30)

with open(os.path.join(BASE, "logs/ushape_edl.log"), "w") as lf:
    p = subprocess.Popen(
        ["/home/jmuaia/anaconda3/envs/bpinn_uq/bin/python",
         os.path.join(BASE, "code/train_ushape_edl.py")],
        stdout=lf, stderr=subprocess.STDOUT)
log(f"EDL 训练已启动 (pid={p.pid})，HANDOFF 退出")
