# -*- coding: utf-8 -*-
"""Mamba-UIE fp32/bs=1 原配方 sanity：逐项打印损失，验证 bf16 版训练失败根因。
仅跑 300 迭代（~10 分钟），不保存。"""
import os, sys, time
import torch
import numpy as np
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/mamba_uie")
from net.net import net as MambaNet
from data import get_training_set
from A import get_A
import torch.optim as optim
from net.losses import EdgeLoss
from utils.SSIM_loss import ssim
from utils.UIQM_loss import getUIQM

ROOT = "/mnt/data/underwater/bpinn_uq"
train_set = get_training_set(f"{ROOT}/datasets/EUVP_flat/trainA",
                             f"{ROOT}/datasets/EUVP_flat/trainB", 256, True)
loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
mse = torch.nn.MSELoss().cuda()
Edge = EdgeLoss()
model = MambaNet().cuda()
opt = optim.Adam(model.parameters(), lr=1e-4)
model.train()
it = 0
t0 = time.time()
for epoch in range(2):
    for input, label, _ in loader:
        input, label = input.cuda(), label.cuda()
        j, t, tb = model(input)
        a = get_A(input).cuda()
        I_rec = j * t + (1 - tb) * a
        l1 = mse(I_rec, input).item()
        l3 = mse(label, j).item()
        l_ssim2 = (1 - torch.mean(ssim(label, j))).item()
        l_edge = Edge(label, j).item()
        l6 = 1 / (getUIQM(j.cpu()) + 1e-10)
        total = (mse(I_rec, input) + mse(label, j)
                 + (1 - torch.mean(ssim(I_rec, input))) + (1 - torch.mean(ssim(label, j)))
                 + l6 + 0.05 * Edge(label, j))
        opt.zero_grad(); total.backward(); opt.step()
        it += 1
        if it % 50 == 0:
            print(f"it{it}: l1={l1:.4f} l3={l3:.4f} ssim2={l_ssim2:.4f} "
                  f"edge={l_edge:.4f} l6={l6:.2f} | {time.time()-t0:.0f}s", flush=True)
        if it >= 300:
            print("SANITY_DONE", flush=True); sys.exit(0)
