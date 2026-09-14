# -*- coding: utf-8 -*-
"""Stage 3a：EDL 模型在 EUVP 验证集上的校准评测（编排层）。

把逐像素 (y_pred, y_std, y_true) 喂给 uncertainty-toolbox（学术校准指标标准库），
外加手算 z 分覆盖率表（高斯假设下 1/2/3σ 应覆盖 68.3/95.4/99.7%）。
所有量在 [0,1] 图像域计算（σ 从 [-1,1] 域除以 2 换算）。
"""
import os, sys, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image
import uncertainty_toolbox as ut

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
CKPT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_edl/best.pth"
E = "/mnt/data/underwater/bpinn_uq/datasets/EUVP_Dataset/EUVP Dataset/EUVP_Paired"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/edl_calib_euvpval.json"
N_SAMPLE = 300_000
SEED = 42

class FunieNIG(nn.Module):
    def __init__(self):
        super().__init__()
        g = GeneratorFunieGAN()
        self.down1, self.down2, self.down3 = g.down1, g.down2, g.down3
        self.down4, self.down5 = g.down4, g.down5
        self.up1, self.up2, self.up3, self.up4 = g.up1, g.up2, g.up3, g.up4
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)),
            NormalInverseGammaConvNd(nn.Conv2d, event_dim=3, in_channels=64,
                                     kernel_size=4, padding=1))
    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4)
        u1 = self.up1(d5, d4); u2 = self.up2(u1, d3); u3 = self.up3(u2, d2)
        return self.final(self.up4(u3, d1))

class ValDS(torch.utils.data.Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y

val_paths = json.load(open(SPLIT))["val_paths"]
pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths]
loader = DataLoader(ValDS(pairs), batch_size=16, num_workers=4, pin_memory=True)

model = FunieNIG().cuda().eval()
model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))

P, S, T = [], [], []
with torch.no_grad():
    for xb, yb in loader:
        out = model(xb.cuda())
        loc01 = (out["loc"].clamp(-1, 1) + 1) / 2
        tgt01 = (yb.cuda() + 1) / 2
        sig01 = torch.sqrt(out["beta"] / (out["alpha"] - 1)) / 2   # [-1,1]域→[0,1]域
        P.append(loc01.cpu().numpy()); S.append(sig01.cpu().numpy()); T.append(tgt01.cpu().numpy())
P = np.concatenate(P); S = np.concatenate(S); T = np.concatenate(T)
print("val 像素总量:", P.size, flush=True)

rng = np.random.default_rng(SEED)
idx = rng.choice(P.size, size=min(N_SAMPLE, P.size), replace=False)
y_pred, y_std, y_true = P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]

res = ut.get_all_metrics(y_pred, y_std, y_true)

def sanitize(o):   # ndarray→list，嵌套 dict 保持，其余转 float
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [float(x) for x in np.atleast_1d(o)]
    if isinstance(o, (np.floating, float)):
        return float(o)
    return o
res = sanitize(res)

# z 分覆盖率表（逐像素 |err|/σ 落在 kσ 内的比例）
z = np.abs(y_pred - y_true) / np.clip(y_std, 1e-8, None)
res["z_coverage"] = {f"{k}sigma": {"expected": e, "empirical": float((z < k).mean())}
                     for k, e in [(1, 0.6827), (2, 0.9545), (3, 0.9973)]}

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1, default=float))
print("CALIB_DONE")
