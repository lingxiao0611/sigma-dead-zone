# -*- coding: utf-8 -*-
"""U-shape Transformer 速度/显存实测（决定增强实验②的训练规模）。"""
import sys, time
import torch
import torch.nn as nn
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from net.Ushape_Trans import Generator
from uq.nig_layer import NormalInverseGammaConvNd


class UshapeDet(nn.Module):
    """确定性版：取多尺度输出里最后一路（全分辨率 256x256）。"""

    def __init__(self):
        super().__init__()
        self.g = Generator()

    def forward(self, x):
        return self.g(x)[-1]


class UshapeNIG(nn.Module):
    """EDL 版：输出层 1:1 替换。"""

    def __init__(self):
        super().__init__()
        g = Generator()
        self.g = g
        # to_rgb(32) = Conv2d(32,3,k1) → NIG 头（自动输出 12 通道）
        self.g.feature_to_rgb[0] = NormalInverseGammaConvNd(
            nn.Conv2d, event_dim=3, in_channels=32, kernel_size=1)

    def forward(self, x):
        out = self.g(x)
        return out[-1]          # 全分辨率那一路


def bench(model, name, bs, iters=12, extract=lambda o: o):
    model = model.cuda()
    model.train()
    x = torch.randn(bs, 3, 256, 256).cuda()
    y = torch.randn(bs, 3, 256, 256).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    l1 = nn.L1Loss()
    # warmup
    for _ in range(3):
        opt.zero_grad(); l1(extract(model(x)), y).backward(); opt.step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad(); l1(extract(model(x)), y).backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print("%-14s bs=%d  %.3f s/iter  %.1f img/s  峰值显存 %.0f MB  参数 %.1fM"
          % (name, bs, dt, bs / dt, mem, n_par), flush=True)
    # 单 epoch 估算（10435 张训练图）
    print("               → 30 epoch 训练耗时估算: %.1f 小时" % (10435 / (bs / dt) * 30 / 3600), flush=True)
    del model, opt, x, y
    torch.cuda.empty_cache()
    return bs / dt


if __name__ == "__main__":
    print("=== U-shape Transformer 实测（256x256, RTX3090）===", flush=True)
    bench(UshapeDet(), "det(基线)", 8)
    bench(UshapeNIG(), "EDL(NIG头)", 8, extract=lambda o: o["loc"])
    bench(UshapeDet(), "det(基线)", 4)
    print("=== done ===", flush=True)
