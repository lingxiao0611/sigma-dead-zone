# -*- coding: utf-8 -*-
"""补强 C：U-shape EDL 主干的分层覆盖率 + lift（主干无关性补全）。
协议同 coverage_stratified：EUVP test500 + UIEB test445；ratio 自算（cal500/cal445）。
输出 outputs/ushape_stratified.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from net.Ushape_Trans import Generator
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image
import uncertainty_toolbox as ut

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT = f"{ROOT}/ckpt/ushape_edl/best.pth"
OUT = f"{ROOT}/outputs/ushape_stratified.json"
K_PIX, SEED = 100_000, 42


class UshapeNIG(nn.Module):
    def __init__(self):
        super().__init__()
        g = Generator()
        self.g = g
        self.g.feature_to_rgb[0] = NormalInverseGammaConvNd(
            nn.Conv2d, event_dim=3, in_channels=32, kernel_size=1)

    def forward(self, x):
        return self.g(x)[-1]


class PairDS(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        arr = np.asarray(A, np.float32) / 255.0
        gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mx, mn = arr.max(axis=2), arr.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0).mean()
        x = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y, torch.tensor([gray.std(), sat, gray.mean()], dtype=torch.float32)


def collect(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=8, num_workers=4, pin_memory=True)
    P, S, T, ST = [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            o = model(xb.cuda())
            P.append(((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb.cuda() + 1) / 2).cpu().numpy()); ST.append(sb.numpy())
    return tuple(np.concatenate(a).astype(np.float32) for a in (P, S, T, ST))


def per_image(P, S, T, ST, ratio=None):
    n = len(P)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P[0].size, size=min(K_PIX, P[0].size), replace=False)
    ip, is_, it = P.reshape(n, -1)[:, idx], S.reshape(n, -1)[:, idx], T.reshape(n, -1)[:, idx]
    err = np.abs(ip - it)
    sig = is_ * ratio if ratio is not None else is_
    z = err / np.clip(sig, 1e-8, None)
    return {"sigma_mean": np.asarray([s.mean() for s in is_]),
            "err_mean": np.asarray([e.mean() for e in err]),
            "cov1": np.asarray([(zz < 1).mean() for zz in z]),
            "contrast": ST[:, 0], "saturation": ST[:, 1], "brightness": ST[:, 2]}


def stratify(d, key="contrast", nb=4):
    v = np.asarray(d[key]); order = np.argsort(v)
    bins = np.array_split(order, nb)
    return [{"stratum": f"S{i}", "n": int(len(b)),
             "err_mean": float(np.asarray(d["err_mean"])[b].mean()),
             "sigma_mean": float(np.asarray(d["sigma_mean"])[b].mean()),
             "cov1": float(np.asarray(d["cov1"])[b].mean())}
            for i, b in enumerate(bins, 1)]


def lift(d):
    o = np.argsort(np.asarray(d["sigma_mean"])); k = max(1, len(o) // 5)
    return float(np.asarray(d["err_mean"])[o[-k:]].mean() /
                 max(np.asarray(d["err_mean"])[o[:k]].mean(), 1e-8))


def ratio_on(cal_P, cal_S, cal_T):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(cal_P.size, size=min(200_000, cal_P.size), replace=False)
    return float(ut.optimize_recalibration_ratio(
        cal_P.flatten()[idx], cal_S.flatten()[idx], cal_T.flatten()[idx]))


def main():
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp_all = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    upairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]

    model = UshapeNIG().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))
    print("ushape loaded", flush=True)

    res = {}
    for dsname, pairs_all, n_cal in [("EUVP_域内", euvp_all, 500), ("UIEB_跨域", upairs, len(upairs) // 2)]:
        Pc, Sc, Tc, _ = collect(model, pairs_all[:n_cal])
        r = ratio_on(Pc, Sc, Tc)
        Pt, St, Tt, ST = collect(model, pairs_all[n_cal:])
        d_raw = per_image(Pt, St, Tt, ST)
        d_rec = per_image(Pt, St, Tt, ST, ratio=r)
        res[dsname] = {"ratio": r, "lift": lift(d_raw),
                       "marginal_cov1": float(d_raw["cov1"].mean()),
                       "marginal_cov1_recal": float(d_rec["cov1"].mean()),
                       "strata_raw": stratify(d_raw), "strata_recal": stratify(d_rec)}
        s = res[dsname]
        print(f"[{dsname}] ratio={r:.3f} 边际1σ={s['marginal_cov1']:.3f}→{s['marginal_cov1_recal']:.3f} "
              f"lift={s['lift']:.2f} 分层raw=" + "/".join(f"{t['cov1']:.3f}" for t in s["strata_raw"]) +
              " recal=" + "/".join(f"{t['cov1']:.3f}" for t in s["strata_recal"]), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("USHAPE_STRAT_DONE")


if __name__ == "__main__":
    main()
