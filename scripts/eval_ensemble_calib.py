# -*- coding: utf-8 -*-
"""Stage 3c 收口：Deep Ensemble 档在 EUVP 验证集上的校准评测（编排层）。

与 eval_calibration_edl.py 严格同协议：同样 val 1000 对、同样 256 resize、
同样 [-1,1] 输入域、同样 30 万像素子采样(seed=42)、同样 uncertainty-toolbox 指标。
唯一区别：预测来自 5 个确定性 FUnIE 成员的均值，σ 来自成员间标准差
（[-1,1]域 → [0,1]域同样 ÷2），与 EDL 档的证据 σ 形成同协议对比。
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
from PIL import Image
import uncertainty_toolbox as ut

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
SEEDS = [101, 202, 303, 404, 505]
CKPT_DIR = "/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble"
E = "/mnt/data/underwater/bpinn_uq/datasets/EUVP_Dataset/EUVP Dataset/EUVP_Paired"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/ensemble_calib_euvpval.json"
N_SAMPLE = 300_000
SEED = 42

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

print(f"[ensemble] loading {len(SEEDS)} members ...", flush=True)
models = []
for s in SEEDS:
    m = GeneratorFunieGAN().cuda().eval()
    m.load_state_dict(torch.load(f"{CKPT_DIR}/seed_{s}/best.pth",
                                 map_location="cuda", weights_only=True))
    models.append(m)
print(f"[ensemble] {len(models)} members ready", flush=True)

P, S, T = [], [], []
with torch.no_grad():
    for it, (xb, yb) in enumerate(loader):
        xb = xb.cuda(non_blocking=True)
        # 累加均值/平方和 → 逐像素逐通道 mean & std（ddof=0，与 numpy std 默认一致）
        pred_mean = torch.zeros_like(xb)
        pred_sq = torch.zeros_like(xb)
        for m in models:
            p = m(xb)                      # (B,3,H,W) in [-1,1]
            pred_mean += p
            pred_sq += p * p
        N = len(models)
        pred_mean /= N
        pred_std = torch.sqrt((pred_sq / N - pred_mean ** 2).clamp_min(0))
        pred01 = (pred_mean.clamp(-1, 1) + 1) / 2
        tgt01 = (yb.cuda() + 1) / 2
        sig01 = pred_std / 2               # [-1,1]域 → [0,1]域，与 EDL 档一致
        P.append(pred01.cpu().numpy()); S.append(sig01.cpu().numpy()); T.append(tgt01.cpu().numpy())
        if (it + 1) % 20 == 0:
            print(f"[ensemble] batch {it+1}/{len(loader)}", flush=True)

P = np.concatenate(P); S = np.concatenate(S); T = np.concatenate(T)
print("val 像素总量:", P.size, flush=True)

rng = np.random.default_rng(SEED)
idx = rng.choice(P.size, size=min(N_SAMPLE, P.size), replace=False)
y_pred, y_std, y_true = P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]

res = ut.get_all_metrics(y_pred, y_std, y_true)

def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [float(x) for x in np.atleast_1d(o)]
    if isinstance(o, (np.floating, float)):
        return float(o)
    return o
res = sanitize(res)

z = np.abs(y_pred - y_true) / np.clip(y_std, 1e-8, None)
res["z_coverage"] = {f"{k}sigma": {"expected": e, "empirical": float((z < k).mean())}
                     for k, e in [(1, 0.6827), (2, 0.9545), (3, 0.9973)]}

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1, default=float))
print("ENSEMBLE_CALIB_DONE")
