# -*- coding: utf-8 -*-
"""补强实验：分层覆盖率 + 每图覆盖率分布 + 图像级风险区分度。

针对两篇 2024 论文的批评做补救：
  [Lind2024, PRL] 不推荐只用 Spearman 评 UQ（只验名次不验数值）
      → 增加图像级"风险区分度 lift"：按 mean σ 排序，最高 20% 图的实际误差 / 最低 20%
  [Sluijterman2024, Neural Networks] 单个测试集的边际覆盖率有根本缺陷
      （过信与欠信会在平均中相互抵消）
      → 按"退化严重度"分层报覆盖率（S1 最严重 → S4 最轻），并报每图覆盖率的分布，
        把"局部崩掉"暴露出来

分层依据只用输入图就能算（对比度/饱和度/亮度），不需要 GT，部署期可用。
协议与 Stage 3b/4 一致：EUVP test500（val_paths[500:]）、UIEB test445（后一半）。
零核心算法自写，仅编排层。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = f"{ROOT}/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"{ROOT}/ckpt/funie_ensemble/seed_{s}/best.pth"
OUT = f"{ROOT}/outputs/coverage_stratified.json"
FIG = f"{ROOT}/outputs/coverage_stratified.png"
K_PIX = 100_000          # 每图随机采样子集（控内存），覆盖率在子图上算
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


def collect_edl(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T, St = [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            o = model(xb.cuda())
            P.append(((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy()); St.append(sb.numpy())
    return tuple(np.concatenate(a).astype(np.float32) for a in (P, S, T, St))


def collect_ens(models, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models); sl, sql, Tl, St = [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            so = sso = None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            sl.append(so); sql.append(sso); Tl.append(((yb + 1) / 2).numpy()); St.append(sb.numpy())
    S = np.concatenate(sl, 0); SS = np.concatenate(sql, 0)
    T = np.concatenate(Tl, 0); ST = np.concatenate(St, 0)
    mean = S / K
    std = np.sqrt(np.clip((SS - S * S / K) / (K - 1), 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), T.astype(np.float32), ST


def spearman(a, b):
    def rank(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), dtype=np.float64)
        r[o] = np.arange(len(v), dtype=np.float64)
        return r
    ra, rb = rank(np.asarray(a, np.float64)), rank(np.asarray(b, np.float64))
    return float(np.corrcoef(ra, rb)[0, 1])


def per_image(P, S, T, ST, ratio=None):
    """返回每图：mean σ、mean 误差、1/2/3σ 覆盖率、退化代理"""
    n = len(P)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P[0].size, size=min(K_PIX, P[0].size), replace=False)
    ip, is_, it = P.reshape(n, -1)[:, idx], S.reshape(n, -1)[:, idx], T.reshape(n, -1)[:, idx]
    err = np.abs(ip - it)
    sig = is_ * ratio if ratio is not None else is_
    z = err / np.clip(sig, 1e-8, None)
    return {
        "sigma_mean": ip.shape[0] and np.asarray([s.mean() for s in is_]),
        "err_mean": np.asarray([e.mean() for e in err]),
        "cov1": np.asarray([(zz < 1).mean() for zz in z]),
        "cov2": np.asarray([(zz < 2).mean() for zz in z]),
        "cov3": np.asarray([(zz < 3).mean() for zz in z]),
        "contrast": ST[:, 0], "saturation": ST[:, 1], "brightness": ST[:, 2],
    }


def stratify(d, key="contrast", nb=4):
    """按退化代理分层：S1 = 对比度最低（退化最重） → S4 最轻"""
    v = d[key]
    order = np.argsort(v)
    bins = np.array_split(order, nb)
    out = []
    for i, b in enumerate(bins, 1):
        out.append({
            "stratum": f"S{i}" + ("(最严重)" if i == 1 else "" if i < nb else "(最轻)"),
            "n": int(len(b)),
            "contrast": float(v[b].mean()),
            "saturation": float(d["saturation"][b].mean()),
            "err_mean": float(d["err_mean"][b].mean()),
            "sigma_mean": float(d["sigma_mean"][b].mean()),
            "cov1": float(d["cov1"][b].mean()),
            "cov2": float(d["cov2"][b].mean()),
            "cov3": float(d["cov3"][b].mean()),
            "cov1_std": float(d["cov1"][b].std()),
            "cov1_p5": float(np.percentile(d["cov1"][b], 5)),
            "cov1_p95": float(np.percentile(d["cov1"][b], 95)),
            "frac_img_cov1_lt40": float((d["cov1"][b] < 0.40).mean()),
        })
    return out


def lift(d):
    """图像级风险区分度：按 mean σ 排序，最高 20% 图 vs 最低 20% 图的平均误差比"""
    o = np.argsort(d["sigma_mean"])
    k = max(1, len(o) // 5)
    lo, hi = o[:k], o[-k:]
    return {"err_top20_sigma": float(d["err_mean"][hi].mean()),
            "err_bottom20_sigma": float(d["err_mean"][lo].mean()),
            "risk_lift": float(d["err_mean"][hi].mean() / max(d["err_mean"][lo].mean(), 1e-8)),
            "spearman_sigma_vs_err": spearman(d["sigma_mean"], d["err_mean"]),
            "spearman_contrast_vs_sigma": spearman(d["contrast"], d["sigma_mean"])}


def main():
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp_test = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    upairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb_test = upairs[len(upairs) // 2:]

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    r_edl = float(json.load(open(f"{ROOT}/outputs/edl_recalibration.json"))["ratio"])
    r_ens = float(json.load(open(f"{ROOT}/outputs/ensemble_recalibration.json"))["ratio"])
    cd = json.load(open(f"{ROOT}/outputs/crossdomain_uieb.json"))
    r_edl_dom = cd.get("edl", {}).get("ratio_domain") or cd.get("edl", {}).get("ratio")
    r_ens_dom = cd.get("ensemble", {}).get("ratio_domain") or cd.get("ensemble", {}).get("ratio")
    print(f"ratios: edl={r_edl:.3f} ens={r_ens:.3f} | domain: edl={r_edl_dom} ens={r_ens_dom}", flush=True)

    res = {}
    for dsname, pairs, rmap in [("EUVP_域内", euvp_test, {"edl": r_edl, "ensemble": r_ens}),
                                ("UIEB_跨域", uieb_test, {"edl": r_edl_dom, "ensemble": r_ens_dom})]:
        res[dsname] = {}
        for mname, model, col in [("edl", edl, collect_edl), ("ensemble", ens, collect_ens)]:
            P, S, T, ST = col(model, pairs)
            d_raw = per_image(P, S, T, ST)
            d_rec = per_image(P, S, T, ST, ratio=rmap.get(mname))
            res[dsname][mname] = {
                "ratio_used": rmap.get(mname),
                "marginal": {"cov1": float(d_raw["cov1"].mean()), "cov2": float(d_raw["cov2"].mean()),
                             "cov3": float(d_raw["cov3"].mean()),
                             "cov1_recal": float(d_rec["cov1"].mean()),
                             "cov1_std_across_imgs": float(d_raw["cov1"].std()),
                             "cov1_p5_across_imgs": float(np.percentile(d_raw["cov1"], 5)),
                             "cov1_p95_across_imgs": float(np.percentile(d_raw["cov1"], 95)),
                             "frac_img_cov1_lt40": float((d_raw["cov1"] < 0.40).mean())},
                "strata_raw": stratify(d_raw),
                "strata_recal": stratify(d_rec),
                "image_level": lift(d_raw),
            }
            s = res[dsname][mname]
            print(f"[{dsname}/{mname}] 边际1σ={s['marginal']['cov1']:.3f} → 重校 {s['marginal']['cov1_recal']:.3f} | "
                  f"分层1σ: " + " ".join(f"{x['stratum'][:2]}={x['cov1']:.3f}" for x in s["strata_raw"]) +
                  f" | lift={s['image_level']['risk_lift']:.2f} ρ(σ,err)={s['image_level']['spearman_sigma_vs_err']:.2f}",
                  flush=True)

    json.dump(res, open(OUT, "w"), indent=1)

    # ---- 图：分层 1σ 覆盖率 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, dsname in zip(axes, ["EUVP_域内", "UIEB_跨域"]):
        x = np.arange(4); w = 0.35
        for j, (mname, lab, c) in enumerate([("edl", "EDL", "tab:orange"), ("ensemble", "Deep Ensemble", "tab:blue")]):
            s = res[dsname][mname]
            raw = [t["cov1"] for t in s["strata_raw"]]
            rec = [t["cov1"] for t in s["strata_recal"]]
            ax.bar(x + j * w - w / 2, raw, w, color=c, alpha=0.45, label=f"{lab} 原始" if j == 0 else f"{lab} 原始")
            ax.bar(x + j * w + w / 2, rec, w, color=c, label=f"{lab} 重校")
        ax.axhline(0.683, ls="--", c="gray", lw=1, label="理想 68.3%")
        ax.set_xticks(x); ax.set_xticklabels(["S1 最严重", "S2", "S3", "S4 最轻"])
        ax.set_ylim(0, 1); ax.set_ylabel("1σ 覆盖率"); ax.set_title(dsname)
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(FIG, dpi=150)
    print("figure ->", FIG, flush=True)
    print("COVERAGE_STRATIFIED_DONE")


if __name__ == "__main__":
    main()
