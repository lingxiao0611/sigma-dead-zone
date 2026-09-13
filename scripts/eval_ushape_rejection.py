# -*- coding: utf-8 -*-
"""增强②·Stage 4：U-shape EDL 拒识部署演示 —— 逐像素风险-覆盖率曲线 + AURC。

与 eval_rejection.py 同协议，仅把主干换成 U-shape Transformer，且只评 EDL 档。
用 TU 移植来的 AURC/RiskAtCov/CovAtRisk（code/uq/tu_risk_coverage.py），
在 ①域内 EUVP test500 ②跨域 UIEB test445 上评 U-shape EDL 的拒识能力。
风险-覆盖率只看 σ 的排序，与 σ 绝对刻度无关（标量 ratio 不改变排序）。
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
from uq.tu_risk_coverage import (regression_aurc, regression_augrc,
                                 regression_risk_at_cov, regression_cov_at_risk,
                                 regression_risk_curve)
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
UIEB = "/mnt/data/underwater/bpinn_uq/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = "/mnt/data/underwater/bpinn_uq/ckpt/ushape_edl/best.pth"
EDL_REC = "/mnt/data/underwater/bpinn_uq/outputs/ushape_edl_recalibration.json"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/ushape_rejection_stage4.json"
FIG = "/mnt/data/underwater/bpinn_uq/outputs/ushape_rejection_curves.png"
N_SAMPLE, SEED = 200_000, 42
COVS = [0.5, 0.7, 0.8, 0.9]


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


def rc_metrics(yp, ys, yt, ratio=None):
    """风险-覆盖率指标。conf = -σ（越大越可信）。"""
    yp_t = torch.from_numpy(yp); ys_t = torch.from_numpy(ys); yt_t = torch.from_numpy(yt)
    if ratio is not None:
        ys_t = ys_t * ratio
    err = (yp_t - yt_t).abs()
    conf = -ys_t
    curve = regression_risk_curve(conf, err)
    n = curve.size(0)
    grid = np.linspace(0.0, 1.0, 101)
    cov_full = (torch.arange(1, n + 1, dtype=torch.float64) / n).numpy()
    curve_np = curve.numpy()
    return {
        "aurc": regression_aurc(conf, err),
        "augrc": regression_augrc(conf, err),
        "risk_at_100cov": float(err.mean()),
        **{f"risk_at_{int(c*100)}cov": regression_risk_at_cov(conf, err, c) for c in COVS},
        "cov_at_5risk": regression_cov_at_risk(conf, err, 0.05),
        "curve_cov": grid.tolist(),
        "curve_risk": np.interp(grid, cov_full, curve_np).tolist(),
    }


def main():
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp_test = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    uieb_pairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb_test = uieb_pairs[len(uieb_pairs) // 2:]
    print(f"EUVP test {len(euvp_test)} | UIEB test {len(uieb_test)}", flush=True)

    edl = UshapeNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    print("model loaded", flush=True)

    r_edl = float(json.load(open(EDL_REC))["ratio"])

    res = {}
    for dsname, pairs in [("EUVP (in-domain)", euvp_test), ("UIEB (cross-domain)", uieb_test)]:
        res[dsname] = {}
        mname, model, col, ratio = "edl", edl, collect_edl, r_edl
        P, S, T = col(model, pairs)
        yp, ys, yt = flat(P, S, T)
        m = rc_metrics(yp, ys, yt)
        # 验证：标量重校准不改变排序 → AURC 应完全一致
        m_chk = rc_metrics(yp, ys, yt, ratio=ratio)
        m["aurc_after_recal"] = m_chk["aurc"]
        m["aurc_scale_invariant"] = bool(abs(m["aurc"] - m_chk["aurc"]) < 1e-8)
        res[dsname][mname] = m
        print(f"[{dsname}/{mname}] AURC={m['aurc']:.5f}  "
              f"AUGRC={m['augrc']:.5f}  MAE(100%cov)={m['risk_at_100cov']:.5f}  "
              f"risk@80%cov={m['risk_at_80cov']:.5f}  "
              f"cov@5%risk={m['cov_at_5risk']:.3f}  "
              f"scale_invariant={m['aurc_scale_invariant']}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dsname in zip(axes, ["EUVP (in-domain)", "UIEB (cross-domain)"]):
        d = res[dsname]["edl"]
        ax.plot(np.array(d["curve_cov"]) * 100, np.array(d["curve_risk"]) * 100,
                label=f"EDL  AURC={d['aurc']*100:.2f}", color="tab:orange", lw=2)
        ax.axhline(res[dsname]["edl"]["risk_at_100cov"] * 100, ls=":", c="gray",
                   label="No-rejection baseline (100% cov)")
        ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Risk = MAE (×100)")
        ax.set_title(dsname); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.invert_xaxis()
    plt.tight_layout(); plt.savefig(FIG, dpi=150)
    print("figure ->", FIG, flush=True)
    print("USHAPE_STAGE4_REJECTION_DONE")


if __name__ == "__main__":
    main()
