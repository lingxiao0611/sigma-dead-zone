# -*- coding: utf-8 -*-
"""增强②·Stage 3b：U-shape EDL σ 重校准（EUVP val 拆 cal 500 / test 500）。

与 eval_recalibration.py 同协议，仅把主干从 FUnIE 换成 U-shape Transformer。
cal 上用 toolbox 现成 optimize_recalibration_ratio 学一个 σ 缩放系数，
test 上对比 重校准前 vs 后 的校准指标 + z 覆盖率。
"""
import os, sys, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from net.Ushape_Trans import Generator
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image
import uncertainty_toolbox as ut

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
CKPT = "/mnt/data/underwater/bpinn_uq/ckpt/ushape_edl/best.pth"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/ushape_edl_recalibration.json"
N_SAMPLE = 200_000
SEED = 42


class UshapeNIG(nn.Module):
    def __init__(self):
        super().__init__()
        g = Generator()
        self.g = g
        self.g.feature_to_rgb[0] = NormalInverseGammaConvNd(
            nn.Conv2d, event_dim=3, in_channels=32, kernel_size=1)

    def forward(self, x):
        return self.g(x)[-1]


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


val_paths = json.load(open(SPLIT))["val_paths"]      # 固定顺序
cal_pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[:500]]
test_pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
print(f"cal {len(cal_pairs)} / test {len(test_pairs)}", flush=True)

model = UshapeNIG().cuda().eval()
model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))


def collect(pairs):
    loader = DataLoader(ValDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.cuda())
            P.append(((out["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(out["beta"] / (out["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb.cuda() + 1) / 2).cpu().numpy())
    return np.concatenate(P), np.concatenate(S), np.concatenate(T)


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
    return {f"{k}sigma": float((z < k).mean()) for k, e in [(1, .6827), (2, .9545), (3, .9973)]}


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
print("USHAPE_RECAL_DONE")
