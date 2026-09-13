# -*- coding: utf-8 -*-
"""Gaussian-NLL 头审计（dead-zone 判别实验的评测半场，TU 语义 AURC）。

与 Stage4/EDL 同协议：EUVP test500 + UIEB test445，channel-level flatten，
conf=-σ，regression_aurc。dead-zone 判据 = UIEB 分层 σ/err 响应 + lift + AURC。
输出 outputs/gauss_audit.json。
"""
import os, sys, json, glob
import numpy as np, torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from uq.tu_risk_coverage import regression_aurc
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW, REF = os.path.join(UIEB, "raw/UIEB_raw_reName"), os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT = f"{ROOT}/ckpt/funie_gauss/best.pth"
DEV = "cuda:0"

class FunieGauss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = GeneratorFunieGAN()
        self.down1, self.down2, self.down3 = g.down1, g.down2, g.down3
        self.down4, self.down5 = g.down4, g.down5
        self.up1, self.up2, self.up3, self.up4 = g.up1, g.up2, g.up3, g.up4
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 6, kernel_size=4, padding=1))
    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2)
        d4 = self.down4(d3); d5 = self.down5(d4)
        u1 = self.up1(d5, d4); u2 = self.up2(u1, d3); u3 = self.up3(u2, d2)
        out = self.final(self.up4(u3, d1))
        return out[:, :3], out[:, 3:].clamp(-8, 8)

import torch.nn as nn

def collect(pairs):
    L = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T, G = [], [], [], []
    for xb, yb in L:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        with torch.no_grad():
            loc, logvar = model(xb)
            P.append(((loc.clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.exp(0.5 * logvar) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).cpu().numpy())
            G.append(xb.mean(dim=1, keepdim=True).std(dim=(2, 3))[:, 0].cpu().numpy())
    return (np.concatenate(P).astype(np.float32), np.concatenate(S).astype(np.float32),
            np.concatenate(T).astype(np.float32), np.concatenate(G).astype(np.float32))

class PairDS(torch.utils.data.Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y

def rms_ce(e, s, r): return float(np.sqrt(np.mean((e / (r * s) - 1) ** 2)))

def fit_ratio(e, s):
    """RMS-CE 最小化（Stage3b 同目标）：粗网格+细化。"""
    best, rr = None, None
    for r in np.linspace(0.2, 30, 600):
        v = rms_ce(e, s, r)
        if best is None or v < best: best, rr = v, r
    for r in np.linspace(max(0.01, rr - 0.3), rr + 0.3, 121):
        v = rms_ce(e, s, r)
        if v < best: best, rr = v, r
    return float(rr), float(best)

def img_lift(e, s, frac=0.2):
    ime = e.reshape(e.shape[0], -1).mean(1); ims = s.reshape(s.shape[0], -1).mean(1)
    n = len(ime); k = max(1, int(frac * n))
    return float(ime[np.argsort(-ims)[:k]].mean() / ime[np.argsort(ims)[:k]].mean())

def audit(name, P, S, T, G, cal_mask=None):
    e = np.abs(P - T).reshape(-1); s = S.reshape(-1)
    mae = float(e.mean())
    tu = float(regression_aurc(torch.from_numpy(-s), torch.from_numpy(e)))
    rho = float(spearmanr(s, e)[0])
    out = {"mae": mae, "aurc": tu, "aurc_rand": mae, "sp_sig_err": rho,
           "img_lift": img_lift(np.abs(P - T), S)}
    # 分层（按输入对比度四分位，输入统计无参考）
    gi = G.reshape(G.shape[0], -1).mean(1)
    order = np.argsort(gi); n = len(gi); edges = np.linspace(0, n, 5, dtype=int)
    strata = []
    for b in range(4):
        idx = order[edges[b]:edges[b + 1]]
        eb = np.abs(P - T)[idx].reshape(-1); sb = S[idx].reshape(-1)
        strata.append({"contrast": [float(gi[idx].min()), float(gi[idx].max())],
                       "cov1_raw": float((eb <= sb).mean()),
                       "sig_mean": float(sb.mean()), "err_mean": float(eb.mean())})
    out["strata"] = strata
    out["sig_gradient"] = float(strata[0]["sig_mean"] - strata[3]["sig_mean"])  # >0: 重域 σ 更大
    out["err_gradient"] = float(strata[0]["err_mean"] - strata[3]["err_mean"])
    if cal_mask is not None:
        ec = np.abs(P - T)[cal_mask].reshape(-1); sc = S[cal_mask].reshape(-1)
        et = np.abs(P - T)[~cal_mask].reshape(-1); st = S[~cal_mask].reshape(-1)
        r, cal = fit_ratio(ec, sc)
        out["ratio"] = r; out["rms_cal_fit"] = cal
        out["cov1_raw"] = float((et <= st).mean())
        out["cov1_recal"] = float((et <= r * st).mean())
        out["aurc"] = float(regression_aurc(torch.from_numpy(-st), torch.from_numpy(et)))
        out["aurc_rand"] = float(et.mean())
    print(f"[{name}] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
          for k, v in out.items() if k != 'strata'}), flush=True)
    print(f"[{name}] strata sig_mean: {[round(x['sig_mean'],4) for x in strata]} | "
          f"err_mean: {[round(x['err_mean'],4) for x in strata]}", flush=True)
    return out

model = FunieGauss().cuda()
model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True))
model.eval()
print("gauss model loaded", flush=True)

vp = json.load(open(SPLIT))["val_paths"]
euvp_cal = [(p, p.replace("/trainA/", "/trainB/")) for p in vp[:500]]
euvp_test = [(p, p.replace("/trainA/", "/trainB/")) for p in vp[500:]]
raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
up = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
uieb_test = up[len(up) // 2:]

results = {}
P, S, T, G = collect(euvp_cal + euvp_test)
mask = np.zeros(len(P), bool); mask[:500] = True   # 前 500 = cal
results["EUVP"] = audit("EUVP_test500+cal", P, S, T, G, cal_mask=mask)
P, S, T, G = collect(uieb_test)
results["UIEB"] = audit("UIEB_test445", P, S, T, G)

json.dump(results, open(f"{ROOT}/outputs/gauss_audit.json", "w"), indent=1)
print("GAUSS_AUDIT_DONE", flush=True)
