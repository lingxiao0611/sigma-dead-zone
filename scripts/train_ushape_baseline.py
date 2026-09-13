# -*- coding: utf-8 -*-
"""增强实验②·U-shape Transformer 确定性基线（主干泛化验证）。

- 复用 FUnIE 基线的 EUVP 配对数据管线 + split.json（同数据、同切分、同 30 epoch）
- 唯一区别 = 骨干换成 U-shape Transformer（取多尺度输出最后一路 out[-1]，末端补 Tanh → [-1,1]，与 FUnIE det 对齐）
- 30 epoch / bs=8（实测 ~2.6h @ RTX3090）
- 输出 ckpt/ushape_baseline/；供后续 EDL 档"零精度代价"对比
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
from pytorch_ssim import ssim

E = "/mnt/data/underwater/bpinn_uq/datasets/EUVP_Dataset/EUVP Dataset/EUVP_Paired"
CATS = ["underwater_dark", "underwater_imagenet", "underwater_scenes(true)"]
CKPT = "/mnt/data/underwater/bpinn_uq/ckpt/ushape_baseline"
SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
LOGF = "/mnt/data/underwater/bpinn_uq/logs/ushape_baseline_trainlog.jsonl"
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


class UshapeDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = Generator()

    def forward(self, x):
        return torch.tanh(self.g(x)[-1])     # 全分辨率那一路 + Tanh → [-1,1]


model = UshapeDet().cuda()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
l1 = nn.L1Loss()
print(f"UshapeDet 参数量 {n_params:.1f}M | L1 监督 | 30ep/bs8", flush=True)


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
    print(f"== ep{ep}/{EPOCHS} | trainL1 {line['train_l1']:.4f} | "
          f"valPSNR {vp:.2f} | valSSIM {vs:.4f} | {line['sec']}s", flush=True)
    with open(LOGF, "a") as f:
        f.write(json.dumps(line) + "\n")
    torch.save(model.state_dict(), os.path.join(CKPT, "last.pth"))
    if vp > best_psnr:
        best_psnr = vp
        torch.save(model.state_dict(), os.path.join(CKPT, "best.pth"))
        with open(os.path.join(CKPT, "best.json"), "w") as f:
            json.dump({"epoch": ep, "val_psnr": vp, "val_ssim": vs}, f)

print(f"USHAPE_BASELINE_DONE best_val_psnr={best_psnr:.2f}", flush=True)
