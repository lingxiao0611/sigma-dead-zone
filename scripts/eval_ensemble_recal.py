# -*- coding: utf-8 -*-
"""Stage 3c 收口：Deep Ensemble 档 σ 重校准（与 3b 严格同协议）。

- 5 个成员均为标准 GeneratorFunieGAN，取均值+方差当 ensemble 档 (y_pred, y_std)
- 同 3b：split.json val 前 500 为 cal，后 500 为 test
- cal 上 optimize_recalibration_ratio 学 ratio，test 上对比重校准前/后
- 零核心算法自写，仅编排层。
"""
import os, sys, json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
from nets.funiegan import GeneratorFunieGAN
from PIL import Image
import uncertainty_toolbox as ut

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
SEEDS = [101, 202, 303, 404, 505]
CKPT = lambda s: f"/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble/seed_{s}/best.pth"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/ensemble_recalibration.json"
N_SAMPLE = 200_000
SEED = 42

class ValDS(Dataset):
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
cal_pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[:500]]
test_pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
print(f"cal {len(cal_pairs)} / test {len(test_pairs)}", flush=True)

models = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
for m, s in zip(models, SEEDS):
    m.load_state_dict(torch.load(CKPT(s), map_location="cuda", weights_only=True))
print(f"loaded {len(models)} ensemble members", flush=True)

def collect(pairs):
    loader = DataLoader(ValDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models)
    sum_list, sumsq_list, T_list = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            so, sso = None, None
            for m in models:
                out = m(xb.cuda())
                out = out.clamp(-1, 1)
                o = out.cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            sum_list.append(so)
            sumsq_list.append(sso)
            T_list.append(((yb + 1) / 2).numpy())
    S = np.concatenate(sum_list, axis=0)
    SS = np.concatenate(sumsq_list, axis=0)
    T = np.concatenate(T_list, axis=0)
    mean = S / K
    var = (SS - S * S / K) / (K - 1)
    std = np.sqrt(np.clip(var, 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), T.astype(np.float32)

def flat(P, S, T):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P.size, size=min(N_SAMPLE, P.size), replace=False)
    return P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]

Pc, Sc, Tc = collect(cal_pairs)
ycal, scal, tcal = flat(Pc, Sc, Tc)
ratio = float(ut.optimize_recalibration_ratio(ycal, scal, tcal))
print(f"重校准 ratio = {ratio:.4f}", flush=True)

Pt, St, Tt = collect(test_pairs)
ypred, ystd, ytrue = flat(Pt, St, Tt)

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

res = {"ratio": ratio, "test_raw": pick(ypred, ystd, ytrue),
       "test_recal": pick(ypred, ystd * ratio, ytrue),
       "z_coverage_raw": zcov(ypred, ystd, ytrue),
       "z_coverage_recal": zcov(ypred, ystd * ratio, ytrue)}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("ENSEMBLE_RECAL_DONE")
