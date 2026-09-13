# -*- coding: utf-8 -*-
"""E4：U-shape Transformer Ensemble 成员训练（4090 适配版，flat EUVP 布局）。

- 超参与 train_ushape_baseline.py 完全一致（30ep/bs8/L1/lr1e-4），仅换数据路径适配 4090 的 EUVP_flat
  （文件名 subset__base.jpg）与 split.json 的平铺映射
- seed 由 argv 传入（101..505，与 FUnIE Ensemble 协议一致）
- 用法：python train_ushape_ens4090.py 101
"""
import os, sys, json, time, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/datadisk/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/datadisk/underwater/bpinn_uq/code")
from net.Ushape_Trans import Generator
from pytorch_ssim import ssim

ROOT = "/mnt/datadisk/underwater/bpinn_uq"
E = f"{ROOT}/datasets/EUVP_flat"
SPLIT = f"{ROOT}/ckpt/split.json"
SEED = int(sys.argv[1])
CKPT = f"{ROOT}/ckpt/ushape_ens/seed_{SEED}"
LOGF = f"{ROOT}/logs/ushape_ens_{SEED}_trainlog.jsonl"
EPOCHS, BATCH, LR, W = 30, 8, 1e-4, 4

os.makedirs(CKPT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---- flat 布局配对 + split.json 平铺映射 ----
val_keys = set()
for p in json.load(open(SPLIT))["val_paths"]:
    subset = p.split("/")[-3]
    base = os.path.splitext(os.path.basename(p))[0]
    val_keys.add(f"{subset}__{base}.jpg")

A = os.path.join(E, "trainA"); B = os.path.join(E, "trainB")
VA = os.path.join(E, "valA"); VB = os.path.join(E, "valB")
train_pairs = [(os.path.join(A, f), os.path.join(B, f)) for f in sorted(os.listdir(A))
               if os.path.exists(os.path.join(B, f))]
val_files = sorted(os.listdir(VA))
assert set(val_files) == val_keys, f"valA 与 split.json 键不一致: {len(set(val_files) & val_keys)}/1000"
val_pairs = [(os.path.join(VA, f), os.path.join(VB, f)) for f in val_files]
print(f"[seed {SEED}] train {len(train_pairs)} | val {len(val_pairs)} (flat 复用 split.json)", flush=True)


class PairDS(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs, self.train = pairs, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        A_ = Image.open(a).convert("RGB"); B_ = Image.open(b).convert("RGB")
        if A_.size != (256, 256): A_ = A_.resize((256, 256))
        if B_.size != (256, 256): B_ = B_.resize((256, 256))
        if self.train and random.random() < 0.5:
            A_ = A_.transpose(Image.FLIP_LEFT_RIGHT); B_ = B_.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(A_, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B_, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y


def to01(t):
    return (t.clamp(-1, 1) + 1) / 2


train_loader = DataLoader(PairDS(train_pairs, True), batch_size=BATCH, shuffle=True,
                          num_workers=W, pin_memory=True, drop_last=True, persistent_workers=True)
val_loader = DataLoader(PairDS(val_pairs, False), batch_size=BATCH, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)


class UshapeDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = Generator()

    def forward(self, x):
        return torch.tanh(self.g(x)[-1])


model = UshapeDet().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
l1 = nn.L1Loss()
print(f"[seed {SEED}] UshapeDet L1 监督 30ep/bs8", flush=True)


@torch.no_grad()
def evaluate():
    model.eval()
    tot_p, tot_s, n = 0.0, 0.0, 0
    for xb, yb in val_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        out, tgt = to01(model(xb)), to01(yb)
        mse = ((out - tgt) ** 2).mean(dim=[1, 2, 3])
        tot_p += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum().item()
        tot_s += ssim(out, tgt, size_average=True).item() * xb.size(0)
        n += xb.size(0)
    model.train()
    return tot_p / n, tot_s / n


best_psnr = -1.0
for ep in range(1, EPOCHS + 1):
    model.train(); t0 = time.time(); run = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        opt.zero_grad()
        loss = l1(model(xb), yb)
        loss.backward(); opt.step()
        run += loss.item()
    vp, vs = evaluate()
    line = {"epoch": ep, "train_l1": run / len(train_loader), "val_psnr": vp,
            "val_ssim": vs, "sec": round(time.time() - t0, 1)}
    print(f"== ep{ep}/{EPOCHS} | valPSNR {vp:.2f} | valSSIM {vs:.4f} | {line['sec']}s", flush=True)
    with open(LOGF, "a") as f:
        f.write(json.dumps(line) + "\n")
    torch.save(model.state_dict(), os.path.join(CKPT, "last.pth"))
    if vp > best_psnr:
        best_psnr = vp
        torch.save(model.state_dict(), os.path.join(CKPT, "best.pth"))
        with open(os.path.join(CKPT, "best.json"), "w") as f:
            json.dump({"epoch": ep, "val_psnr": vp, "val_ssim": vs}, f)

print(f"USHAPE_ENS_DONE seed={SEED} best_val_psnr={best_psnr:.2f}", flush=True)
