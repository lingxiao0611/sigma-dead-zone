# -*- coding: utf-8 -*-
"""增强实验②·U-shape Transformer + EDL 证据头（NIG 头 + DERLoss）。

与 U-shape 基线严格可比：同数据、同固定切分（split.json）、同 30 epoch/优化器/增广，
唯一区别 = 末端输出层（feature_to_rgb[0]）换 NIG 证据头（12ch=4*3，无 Tanh，loc 无界，与 FUnIE EDL 对齐），
损失 L1 -> DERLoss（target yb ∈ [-1,1]）。
输出逐像素 {"loc","lmbda","alpha","beta"}；验证集报 PSNR(loc)/SSIM(loc)/NLL/平均证据 alpha。
"""
import os, sys, json, time, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from net.Ushape_Trans import Generator
from uq.nig_layer import NormalInverseGammaConvNd
from uq.nig_dist import NormalInverseGamma
from uq.der_loss import DERLoss
from pytorch_ssim import ssim

E = "/mnt/data/underwater/bpinn_uq/datasets/EUVP_Dataset/EUVP Dataset/EUVP_Paired"
CATS = ["underwater_dark", "underwater_imagenet", "underwater_scenes(true)"]
CKPT = "/mnt/data/underwater/bpinn_uq/ckpt/ushape_edl"
SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
LOGF = "/mnt/data/underwater/bpinn_uq/logs/ushape_edl_trainlog.jsonl"
EPOCHS, BATCH, LR, W = 30, 8, 1e-4, 4
REG_W = 0.01
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
print(f"train {len(train_pairs)} | val {len(val_pairs)} (复用 split.json)", flush=True)


class PairDS(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs, self.train = pairs, train

    def __len__(self):
        return len(self.pairs)

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


def to01(t):
    return (t.clamp(-1, 1) + 1) / 2


train_loader = DataLoader(PairDS(train_pairs, True), batch_size=BATCH, shuffle=True,
                          num_workers=W, pin_memory=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(PairDS(val_pairs, False), batch_size=BATCH, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)


class UshapeNIG(nn.Module):
    def __init__(self):
        super().__init__()
        g = Generator()
        self.g = g
        # feature_to_rgb[0] = to_rgb(32) = Conv2d(32,3,k1) → 替换为 NIG 证据头（自动 12ch=4*3）
        self.g.feature_to_rgb[0] = NormalInverseGammaConvNd(
            nn.Conv2d, event_dim=3, in_channels=32, kernel_size=1)

    def forward(self, x):
        return self.g(x)[-1]        # dict: loc/lmbda/alpha/beta（无 Tanh，loc 无界，与 FUnIE EDL 对齐）


model = UshapeNIG().cuda()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
der = DERLoss(reg_weight=REG_W)
print(f"UshapeNIG 参数量 {n_params:.1f}M | DER reg_weight={REG_W} | 30ep/bs8", flush=True)


@torch.no_grad()
def evaluate():
    model.eval()
    tp, ts, tn, ta, n = 0.0, 0.0, 0.0, 0.0, 0
    for xb, yb in val_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        out = model(xb)
        dist = NormalInverseGamma(out["loc"], out["lmbda"], out["alpha"], out["beta"])
        loc01, tgt01 = to01(out["loc"]), to01(yb)
        mse = ((loc01 - tgt01) ** 2).mean(dim=[1, 2, 3])
        tp += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum().item()
        ts += ssim(loc01, tgt01, size_average=True).item() * xb.size(0)
        tn += (-dist.log_prob(yb)).mean().item() * xb.size(0)
        ta += out["alpha"].mean().item() * xb.size(0)
        n += xb.size(0)
    model.train()
    return tp / n, ts / n, tn / n, ta / n


best_psnr = -1.0
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); run = 0.0; nb = 0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        opt.zero_grad()
        out = model(xb)
        dist = NormalInverseGamma(out["loc"], out["lmbda"], out["alpha"], out["beta"])
        loss = der(dist, yb)
        loss.backward(); opt.step()
        run += loss.item(); nb += 1
    vp, vs, vn, va = evaluate()
    line = {"epoch": ep, "train_der": run / nb, "val_psnr": vp, "val_ssim": vs,
            "val_nll": vn, "val_alpha_mean": va, "sec": round(time.time() - t0, 1)}
    print(f"== ep{ep}/{EPOCHS} | DER {line['train_der']:.3f} | valPSNR {vp:.2f} | "
          f"valSSIM {vs:.4f} | valNLL {vn:.3f} | meanAlpha {va:.2f} | {line['sec']}s", flush=True)
    with open(LOGF, "a") as f:
        f.write(json.dumps(line) + "\n")
    torch.save(model.state_dict(), os.path.join(CKPT, "last.pth"))
    if vp > best_psnr:
        best_psnr = vp
        torch.save(model.state_dict(), os.path.join(CKPT, "best.pth"))
        json.dump({"epoch": ep, "val_psnr": vp, "val_ssim": vs, "val_nll": vn},
                  open(os.path.join(CKPT, "best.json"), "w"))

print(f"USHAPE_EDL_DONE best_val_psnr={best_psnr:.2f}", flush=True)
