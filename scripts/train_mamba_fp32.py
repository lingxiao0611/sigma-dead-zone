# -*- coding: utf-8 -*-
"""Mamba-UIE 重训：作者原配方（fp32 / bs=1 / 30ep / lr 1e-4）。
教训：bf16 混精度导致该模型训练失败（训练集 PSNR 随 epoch 退化 9.1→6.0）。
改进：逐项损失日志（防展示常数 l6 掩盖真实损失）+ 每 epoch val PSNR + 每 epoch 存档。"""
import os, sys, time, json
import torch
import numpy as np
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/mamba_uie")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from net.net import net as MambaNet
from data import get_training_set
from A import get_A
import torch.optim as optim
from net.losses import EdgeLoss
from utils.SSIM_loss import ssim
from utils.UIQM_loss import getUIQM
from eval_mambauie_psnr import PairDS, psnr_ssim
from torch.utils.data import DataLoader

ROOT = "/mnt/data/underwater/bpinn_uq"
OUT_W = f"{ROOT}/code/mamba_uie/weights_fp32"
os.makedirs(OUT_W, exist_ok=True)

train_set = get_training_set(f"{ROOT}/datasets/EUVP_flat/trainA",
                             f"{ROOT}/datasets/EUVP_flat/trainB", 256, True)
loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True,
                                     num_workers=4, pin_memory=True)
vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:1000]
val_pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in vp]

mse = torch.nn.MSELoss().cuda()
Edge = EdgeLoss()
model = MambaNet().cuda()
opt = optim.Adam(model.parameters(), lr=1e-4)
val_ds = PairDS(val_pairs)
hist = []

for epoch in range(1, 31):
    model.train()
    run = {"l1": 0.0, "l3": 0.0, "l6": 0.0}
    t0 = time.time()
    for it, (input, label, _) in enumerate(loader, 1):
        input, label = input.cuda(), label.cuda()
        j, t, tb = model(input)
        a = get_A(input).cuda()
        I_rec = j * t + (1 - tb) * a
        l1 = mse(I_rec, input) + (1 - torch.mean(ssim(I_rec, input)))
        l3 = mse(label, j) + (1 - torch.mean(ssim(label, j)))
        l6 = 1 / (getUIQM(j.cpu()) + 1e-10)     # numpy 只读常数，不参与梯度
        le = Edge(label, j)
        total = l1 + l3 + l6 + 0.05 * le
        opt.zero_grad(); total.backward(); opt.step()
        run["l1"] += float(l1); run["l3"] += float(l3); run["l6"] += float(l6)
        if it % 2000 == 0:
            print(f"ep{ep} it{it}/{len(loader)} l1={float(l1):.4f} l3={float(l3):.4f} "
                  f"l6={float(l6):.1f}", flush=True)
    torch.save(model.state_dict(), f"{OUT_W}/epoch_{epoch}.pth")
    vr = psnr_ssim(model, val_pairs)
    hist.append({"epoch": epoch, "l1": run["l1"]/len(loader), "l3": run["l3"]/len(loader),
                 "l6": run["l6"]/len(loader), "val": vr})
    print(f"== ep{epoch} done: l1={hist[-1]['l1']:.4f} l3={hist[-1]['l3']:.4f} "
          f"valPSNR={vr['psnr']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(hist, open(f"{ROOT}/outputs/mamba_fp32_hist.json", "w"), indent=1)
print("MAMBA_FP32_TRAIN_DONE", flush=True)
