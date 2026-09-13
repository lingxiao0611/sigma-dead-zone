# -*- coding: utf-8 -*-
"""对照实验：FUnIE + Gaussian-NLL 头（同骨干/同预算/同协议，唯头与损失不同）。

目的：分辨 σ dead zone 属于 (a) NIG 参数化、(b) DER 损失、(c) 任意学习 σ 映射。
与 train_funie_edl.py 逐行对齐：同 seed=42、同 split.json、同 30ep/Adam(1e-4,0.5,0.999)/bs8/翻转增广。
唯一区别 = 末端 6ch（3 loc + 3 logvar），损失 = Gaussian NLL。
"""
import os, sys, json, time, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from pytorch_ssim import ssim

E = "/mnt/data/underwater/bpinn_uq/datasets/EUVP_Dataset/EUVP Dataset/EUVP_Paired"
CATS = ["underwater_dark", "underwater_imagenet", "underwater_scenes(true)"]
CKPT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_gauss"
SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
LOGF = "/mnt/data/underwater/bpinn_uq/logs/funie_gauss_trainlog.jsonl"
EPOCHS, BATCH, LR, W = 30, 8, 1e-4, 4
SEED = 42

os.makedirs(CKPT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def build_pairs():
    pairs = []
    for c in CATS:
        A = os.path.join(E, c, "trainA"); B = os.path.join(E, c, "trainB")
        for f in sorted(os.listdir(A)):
            pb = os.path.join(B, f)
            if os.path.exists(pb):
                pairs.append((os.path.join(A, f), pb))
    return pairs

pairs = build_pairs()
val_set = set(json.load(open(SPLIT))["val_paths"])
val_pairs = [p for p in pairs if p[0] in val_set]
train_pairs = [p for p in pairs if p[0] not in val_set]
assert len(val_pairs) == 1000, f"切分对不上: val={len(val_pairs)}"
print(f"train {len(train_pairs)} | val {len(val_pairs)}", flush=True)

class PairDS(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs, self.train = pairs, train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB"); B = Image.open(b).convert("RGB")
        if A.size != (256, 256): A = A.resize((256, 256))
        if B.size != (256, 256): B = B.resize((256, 256))
        if self.train and random.random() < 0.5:
            A = A.transpose(Image.FLIP_LEFT_RIGHT); B = B.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y

def to01(t): return (t.clamp(-1, 1) + 1) / 2

train_loader = DataLoader(PairDS(train_pairs, True), batch_size=BATCH, shuffle=True,
                          num_workers=W, pin_memory=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(PairDS(val_pairs, False), batch_size=BATCH, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)

class FunieGauss(nn.Module):
    """末端 6ch：3 loc + 3 logvar（高斯头）。σ = exp(0.5·logvar)。"""
    def __init__(self):
        super().__init__()
        g = GeneratorFunieGAN()
        self.down1, self.down2, self.down3 = g.down1, g.down2, g.down3
        self.down4, self.down5 = g.down4, g.down5
        self.up1, self.up2, self.up3, self.up4 = g.up1, g.up2, g.up3, g.up4
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 6, kernel_size=4, padding=1),
        )
    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4)
        u1 = self.up1(d5, d4); u2 = self.up2(u1, d3); u3 = self.up3(u2, d2)
        out = self.final(self.up4(u3, d1))
        loc, logvar = out[:, :3], out[:, 3:].clamp(-8, 8)
        return loc, logvar

model = FunieGauss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
print(f"FunieGauss 参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

def gauss_nll(loc, logvar, y):
    return 0.5 * (logvar + (y - loc) ** 2 / torch.exp(logvar)).mean()

@torch.no_grad()
def evaluate():
    model.eval()
    tp, ts, tn, tsig, n = 0.0, 0.0, 0.0, 0.0, 0
    for xb, yb in val_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        loc, logvar = model(xb)
        loc01, tgt01 = to01(loc), to01(yb)
        mse = ((loc01 - tgt01) ** 2).mean(dim=[1, 2, 3])
        tp += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum().item()
        ts += ssim(loc01, tgt01, size_average=True).item() * xb.size(0)
        tn += gauss_nll(loc, logvar, yb).item() * xb.size(0)
        tsig += torch.exp(0.5 * logvar).mean().item() * xb.size(0)
        n += xb.size(0)
    model.train()
    return tp / n, ts / n, tn / n, tsig / n

best_psnr = -1.0
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); run = 0.0; nb = 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        opt.zero_grad()
        loc, logvar = model(xb)
        loss = gauss_nll(loc, logvar, yb)
        loss.backward(); opt.step()
        run += loss.item(); nb += 1
    vp, vs, vn, vsig = evaluate()
    print(f"== ep{ep}/{EPOCHS} | NLL {run/nb:.3f} | valPSNR {vp:.2f} | valSSIM {vs:.4f} | "
          f"valNLL {vn:.3f} | meanSig {vsig:.4f} | {time.time()-t0:.1f}s", flush=True)
    with open(LOGF, "a") as f: f.write(json.dumps(
        {"epoch": ep, "val_psnr": vp, "val_ssim": vs, "val_nll": vn, "mean_sig": vsig}) + "\n")
    torch.save(model.state_dict(), os.path.join(CKPT, "last.pth"))
    if vp > best_psnr:
        best_psnr = vp
        torch.save(model.state_dict(), os.path.join(CKPT, "best.pth"))
        json.dump({"epoch": ep, "val_psnr": vp, "val_ssim": vs},
                  open(os.path.join(CKPT, "best.json"), "w"))
print(f"TRAIN_DONE best_val_psnr={best_psnr:.2f}", flush=True)
