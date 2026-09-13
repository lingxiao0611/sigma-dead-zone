# -*- coding: utf-8 -*-
"""预算不敏感性消融（Mamba-UIE 部分）：不同训练进度 checkpoint 的 EUVP val 精度。
checkpoint 每 epoch 都有（1..30），取 5 / 15 / 25 / 30。产物 outputs/mambauie_ablation.json"""
import os, sys, json
import torch
import numpy as np
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/mamba_uie")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from net.net import net as MambaNet
from eval_mambauie_psnr import PairDS, psnr_ssim
from torch.utils.data import DataLoader

ROOT = "/mnt/data/underwater/bpinn_uq"
CKPTS = [5, 15, 25, 30]


def main():
    model = MambaNet().cuda().eval()
    vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:1000]
    pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in vp]
    res = {}
    for ep in CKPTS:
        ck = f"{ROOT}/code/mamba_uie/weights_euvp/epoch_{ep}.pth"
        if not os.path.exists(ck):
            print(f"跳过 epoch_{ep}", flush=True)
            continue
        model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
        r = psnr_ssim(model, pairs)
        res[f"epoch_{ep}"] = r
        print(f"[ep{ep}] {r}", flush=True)
    os.makedirs(f"{ROOT}/outputs", exist_ok=True)
    json.dump(res, open(f"{ROOT}/outputs/mambauie_ablation.json", "w"), indent=1)
    print("MAMBA_ABLATION_DONE")


if __name__ == "__main__":
    main()
