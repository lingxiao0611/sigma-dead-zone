# -*- coding: utf-8 -*-
"""Audit one transfer direction, in-domain and cross-domain, with the paper's metrics.

The paper's transfer axis trains on EUVP and audits on UIEB. This script runs a
direction in either orientation and reports the same quantities, so the two
orientations are directly comparable:

  --cfg uieb_src   train on UIEB (700/95/95), in-domain UIEB, cross-domain EUVP (500/500)
  --cfg euvp_src   train on EUVP (paper split), in-domain EUVP (500/500), cross-domain UIEB (445/445)
  --cfg euvp700    train on 700 EUVP pairs, everything else as euvp_src (size control)

Metric code, granularity, subsample size and seed are copied from eval_rejection.py
and eval_crossdomain_uieb.py so nothing in the comparison depends on how the curve is
built. Orchestration only: the evidential head, the DER loss and the AURC/AUGRC
implementations are the ported upstream ones.
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

R = "/mnt/data/underwater/bpinn_uq"
sys.path.insert(0, R + "/code/funie_gan/PyTorch")
sys.path.insert(0, R + "/code")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from uq.tu_risk_coverage import (regression_aurc, regression_augrc,
                                 regression_risk_at_cov, regression_cov_at_risk,
                                 regression_risk_curve)
from PIL import Image
import uncertainty_toolbox as ut

SEEDS = [101, 202, 303, 404, 505]
N_SAMPLE, SEED = 200_000, 42
COVS = [0.5, 0.7, 0.8, 0.9]

UIEB = R + "/datasets/UIEB_Dataset/UIEB Dataset"
UIEB_RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
UIEB_REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
EUVP_PAIRED = R + "/datasets/EUVP_Dataset/EUVP Dataset/EUVP_Paired"
PAPER_EUVP_SPLIT = R + "/ckpt/funie_baseline/split.json"


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


class PairDS(Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y


def flat(P, S, T):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P.size, size=min(N_SAMPLE, P.size), replace=False)
    return P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]


def collect_edl(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            o = model(xb.cuda())
            P.append(((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy())
    return (np.concatenate(P).astype(np.float32), np.concatenate(S).astype(np.float32),
            np.concatenate(T).astype(np.float32))


def collect_ens(models, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models)
    sl, sql, Tl = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            so, sso = None, None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            sl.append(so); sql.append(sso); Tl.append(((yb + 1) / 2).numpy())
    S = np.concatenate(sl, 0); SS = np.concatenate(sql, 0); T = np.concatenate(Tl, 0)
    mean = S / K
    std = np.sqrt(np.clip((SS - S * S / K) / (K - 1), 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), T.astype(np.float32)


def zcov(yp, ys, yt):
    z = np.abs(yp - yt) / np.clip(ys, 1e-8, None)
    return {("%dsigma" % k): float((z < k).mean()) for k in (1, 2, 3)}


def rc_metrics(yp, ys, yt, ratio=None):
    yp_t = torch.from_numpy(yp); ys_t = torch.from_numpy(ys); yt_t = torch.from_numpy(yt)
    if ratio is not None:
        ys_t = ys_t * ratio
    err = (yp_t - yt_t).abs()
    conf = -ys_t
    curve = regression_risk_curve(conf, err)
    n = curve.size(0)
    grid = np.linspace(0.0, 1.0, 101)
    cov_full = (torch.arange(1, n + 1, dtype=torch.float64) / n).numpy()
    return {
        "aurc": float(regression_aurc(conf, err)),
        "augrc": float(regression_augrc(conf, err)),
        "aurc_random": float(err.mean()),
        "risk_at_100cov": float(err.mean()),
        **{("risk_at_%dcov" % int(c * 100)): float(regression_risk_at_cov(conf, err, c)) for c in COVS},
        "cov_at_5risk": float(regression_cov_at_risk(conf, err, 0.05)),
        "curve_cov": grid.tolist(),
        "curve_risk": np.interp(grid, cov_full, curve.numpy()).tolist(),
    }


def uieb_all_pairs():
    ref = set(os.listdir(UIEB_REF))
    return [(os.path.join(UIEB_RAW, f), os.path.join(UIEB_REF, f))
            for f in sorted(os.listdir(UIEB_RAW)) if f in ref]


def build_cfg(name):
    """Return (source, target, in_cal, in_test, cross_cal, cross_test, ckpt_edl, ckpt_ens, protocol)."""
    if name == "uieb_src":
        s = json.load(open(R + "/ckpt/funie_edl_uieb/split.json"))
        mk = lambda fs: [(os.path.join(UIEB_RAW, f), os.path.join(UIEB_REF, f)) for f in fs]
        vp = json.load(open(PAPER_EUVP_SPLIT))["val_paths"]
        eu = lambda ps: [(p, p.replace("/trainA/", "/trainB/")) for p in ps]
        return ("UIEB", "EUVP", mk(s["split"]["cal"]), mk(s["split"]["test"]),
                eu(vp[:500]), eu(vp[500:]),
                R + "/ckpt/funie_edl_uieb/best.pth",
                lambda sd: R + "/ckpt/funie_ensemble_uieb/seed_%d/best.pth" % sd,
                {"source_train": 700, "source_cal": 95, "source_test": 95,
                 "target_cal": 500, "target_test": 500})

    if name in ("euvp_src", "euvp700"):
        if name == "euvp700":
            root = R + "/datasets/EUVP_700"
            split = R + "/ckpt/funie_edl_euvp700/split.json"
            ced = R + "/ckpt/funie_edl_euvp700/best.pth"
            cef = lambda sd: R + "/ckpt/funie_ensemble_euvp700/seed_%d/best.pth" % sd
            ntr = 700
        else:
            root = EUVP_PAIRED
            split = PAPER_EUVP_SPLIT
            ced = R + "/ckpt/funie_edl/best.pth"
            cef = lambda sd: R + "/ckpt/funie_ensemble/seed_%d/best.pth" % sd
            ntr = 9435
        vp = json.load(open(split))["val_paths"]
        eu = lambda ps: [(p, p.replace("/trainA/", "/trainB/")) for p in ps]
        u = uieb_all_pairs()
        h = len(u) // 2
        return ("EUVP", "UIEB", eu(vp[:500]), eu(vp[500:]), u[:h], u[h:],
                ced, cef,
                {"source_train": ntr, "source_cal": 500, "source_test": 500,
                 "target_cal": h, "target_test": len(u) - h})

    raise SystemExit("unknown cfg " + name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or args.cfg

    src, tgt, in_cal, in_test, cr_cal, cr_test, ced, cef, proto = build_cfg(args.cfg)
    print("direction %s -> %s | in-dom cal/test %d/%d | cross cal/test %d/%d"
          % (src, tgt, len(in_cal), len(in_test), len(cr_cal), len(cr_test)), flush=True)

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(ced, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(cef(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    res = {"source": src, "target": tgt, "protocol": proto}

    src_ratio = {}
    for name, model, col in [("edl", edl, collect_edl), ("ensemble", ens, collect_ens)]:
        Pc, Sc, Tc = col(model, in_cal)
        yc, sc, tc = flat(Pc, Sc, Tc)
        r = float(ut.optimize_recalibration_ratio(yc, sc, tc))
        src_ratio[name] = r
        Pt, St, Tt = col(model, in_test)
        yp, ys, yt = flat(Pt, St, Tt)
        zr, zc = zcov(yp, ys, yt), zcov(yp, ys * r, yt)
        res.setdefault("%s_in_domain" % src, {})[name] = {
            "ratio": r, "mean_sigma": float(St.mean()),
            "z_raw": zr, "z_recal": zc,
            "raw": rc_metrics(yp, ys, yt), "recal": rc_metrics(yp, ys, yt, ratio=r),
        }
        print("[%s in-domain/%s] ratio=%.3f cov1 %.4f -> %.4f | AURC %.5f (random %.5f)"
              % (src, name, r, zr["1sigma"], zc["1sigma"],
                 res["%s_in_domain" % src][name]["raw"]["aurc"],
                 res["%s_in_domain" % src][name]["raw"]["aurc_random"]), flush=True)

    for name, model, col in [("edl", edl, collect_edl), ("ensemble", ens, collect_ens)]:
        Pc, Sc, Tc = col(model, cr_cal)
        yc, sc, tc = flat(Pc, Sc, Tc)
        r_t = float(ut.optimize_recalibration_ratio(yc, sc, tc))
        r_s = src_ratio[name]
        Pt, St, Tt = col(model, cr_test)
        yp, ys, yt = flat(Pt, St, Tt)
        zr = zcov(yp, ys, yt)
        res.setdefault("%s_cross_domain" % tgt, {})[name] = {
            "ratio_source": r_s, "ratio_target": r_t, "inflation": r_t / r_s,
            "mean_sigma": float(St.mean()),
            "z_raw": zr, "z_xfer": zcov(yp, ys * r_s, yt), "z_domain": zcov(yp, ys * r_t, yt),
            "raw": rc_metrics(yp, ys, yt),
            "xfer": rc_metrics(yp, ys, yt, ratio=r_s),
            "domain": rc_metrics(yp, ys, yt, ratio=r_t),
        }
        d = res["%s_cross_domain" % tgt][name]
        print("[%s cross/%s] %s ratio %.3f -> %s ratio %.3f (x%.2f) | AURC %.5f (random %.5f)"
              % (tgt, name, src, r_s, tgt, r_t, r_t / r_s,
                 d["raw"]["aurc"], d["raw"]["aurc_random"]), flush=True)

    out = R + "/outputs/transfer_%s.json" % tag
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)
    print("TRANSFER_DIRECTION_DONE")


if __name__ == "__main__":
    main()
