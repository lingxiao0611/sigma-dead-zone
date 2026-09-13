# -*- coding: utf-8 -*-
"""MC-BN 第三档（Teye et al. 2018 式）：BN 保持 train 模式推理，
随机性来自每次 pass 的随机批组合（BN 批统计量随组成波动）。
T 个 pass → 每图均值=预测、std/2=σ（[-1,1]→[0,1] 与 EDL/ENS 同规）。
协议与 3b 严格一致：EUVP val 1000 固定顺序 cal500/test500、
200k 像素采样 seed42、optimize_recalibration_ratio、同指标集。
零核心算法自写，仅编排层。
"""
import os, sys, json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from PIL import Image
import uncertainty_toolbox as ut

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
CKPT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/best.pth"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/mcbn_recalibration.json"
N_SAMPLE = 200_000
SEED = 42
T_PASSES = 30

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
all_pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[:1000]]
print(f"val {len(all_pairs)} (cal500/test500 按 3b 固定顺序)", flush=True)

model = GeneratorFunieGAN().cuda()
model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))
model.train()  # MC-BN 关键：BN 用批统计量；FUnIE 生成器无 dropout，train() 只影响 BN
n_bn = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
print(f"MC-BN mode: {n_bn} BatchNorm2d layers in train mode, T={T_PASSES}", flush=True)

ds = ValDS(all_pairs)
N = len(ds)
sum_img = torch.zeros(N, 3, 256, 256)
sumsq_img = torch.zeros(N, 3, 256, 256)

g = torch.Generator().manual_seed(SEED)
for t in range(T_PASSES):
    perm = torch.randperm(N, generator=g).tolist()
    loader = DataLoader(Subset(ds, perm), batch_size=16, num_workers=4, pin_memory=True)
    with torch.no_grad():
        for k, (xb, _) in enumerate(loader):
            out = model(xb.cuda())                       # (B,3,256,256) [-1,1]
            orig_idx = perm[k * 16: k * 16 + xb.size(0)]
            o = out.cpu()
            sum_img[orig_idx] += o
            sumsq_img[orig_idx] += o * o
    if (t + 1) % 5 == 0:
        print(f"pass {t+1}/{T_PASSES}", flush=True)

mean = (sum_img / T_PASSES).clamp(-1, 1)
var = (sumsq_img / T_PASSES - (sum_img / T_PASSES) ** 2).clamp_min(0)
std = torch.sqrt(var)
P = ((mean + 1) / 2).numpy().astype(np.float32)
S = (std / 2).numpy().astype(np.float32)                 # [-1,1]域 → [0,1]域
Tgt = np.stack([(np.asarray(Image.open(b).convert("RGB").resize((256, 256)), np.float32) / 255)
                for a, b in all_pairs]).astype(np.float32)
print("MC-BN 前向完成，σ mean:", float(S.mean()), flush=True)

def flat(Pv, Sv, Tv):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(Pv.size, size=min(N_SAMPLE, Pv.size), replace=False)
    return Pv.flatten()[idx], Sv.flatten()[idx], Tv.flatten()[idx]

def zcov(yp, ys, yt):
    z = np.abs(yp - yt) / np.clip(ys, 1e-8, None)
    return {f"{k}sigma": float((z < k).mean()) for k in (1, 2, 3)}

def pick(yp, ys, yt):
    return {"rms_cal": float(ut.root_mean_squared_calibration_error(yp, ys, yt)),
            "ma_cal": float(ut.mean_absolute_calibration_error(yp, ys, yt)),
            "miscal_area": float(ut.miscalibration_area(yp, ys, yt)),
            "mae": float(np.abs(yp - yt).mean()),
            "sharpness": float(ut.sharpness(ys)),
            "nll": float(ut.nll_gaussian(yp, ys, yt)),
            "crps": float(ut.crps_gaussian(yp, ys, yt))}

Pc, Sc, Tc = P[:500], S[:500], Tgt[:500]
ycal, scal, tcal = flat(Pc, Sc, Tc)
ratio = float(ut.optimize_recalibration_ratio(ycal, scal, tcal))
print(f"重校准 ratio = {ratio:.4f}", flush=True)

Pt, St, Tt = P[500:], S[500:], Tgt[500:]
ypred, ystd, ytrue = flat(Pt, St, Tt)
res = {"t_passes": T_PASSES, "ratio": ratio,
       "test_raw": pick(ypred, ystd, ytrue),
       "test_recal": pick(ypred, ystd * ratio, ytrue),
       "z_coverage_raw": zcov(ypred, ystd, ytrue),
       "z_coverage_recal": zcov(ypred, ystd * ratio, ytrue)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("MCBN_DONE")
