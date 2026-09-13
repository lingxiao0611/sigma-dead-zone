# -*- coding: utf-8 -*-
"""E2 朴素基线三件套：常数 σ / 输入局部方差 / 复原图局部方差，进同一审计协议。

每个 σ 变体与它配套的复原预测绑定审计：
  edl / ensemble  → 各自模型的预测与误差
  det_var_out / det_var_in / const → 确定性 FUnIE 基线的预测与误差
常数 σ 是"校准指标盲区"的活体证明：标量修复后 coverage 满分，lift=1、AURC 零收益。
AURC 用统一 inline 实现（所有变体同一口径，保证表内可比）。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image
import uncertainty_toolbox as ut

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
SEEDS = [101, 202, 303, 404, 505]
OUT = f"{ROOT}/outputs/naive_baselines.json"
K_PIX, SEED = 100_000, 42
POOL_PIX, BLK = 2_000_000, 8


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
        x = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y, torch.tensor([gray.std()], dtype=torch.float32)


@torch.no_grad()
def block_var(t01):
    """8×8 块内方差（逐块常数，上采样回 256），输入 (B,3,256,256) [0,1]"""
    m = F.avg_pool2d(t01, BLK)
    ms = F.avg_pool2d(t01 * t01, BLK)
    v = (ms - m * m).clamp_min(0).mean(dim=1, keepdim=True)
    return F.interpolate(v, scale_factor=BLK, mode="nearest")   # (B,1,256,256)


@torch.no_grad()
def collect(models, pairs):
    """返回 per-image dict：三个预测的 P、σ 变体的 S、GT、contrast（固定采样 idx）"""
    det, edl, ens = models
    loader = DataLoader(PairDS(pairs), batch_size=8, num_workers=4, pin_memory=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(3 * 256 * 256, size=K_PIX, replace=False)
    R = {k: [] for k in ["P0", "Pe", "Se", "Pn", "Sn", "T", "Vin", "Vout", "C"]}
    for xb, yb, sb in loader:
        xb = xb.cuda(); yb01 = ((yb + 1) / 2)
        p0 = det(xb).clamp(-1, 1); p0 = ((p0 + 1) / 2)
        oe = edl(xb)
        pe = ((oe["loc"].clamp(-1, 1) + 1) / 2)
        se = (torch.sqrt(oe["beta"] / (oe["alpha"] - 1)) / 2)
        so = sso = None
        for m in ens:
            o = m(xb).clamp(-1, 1)
            so = o if so is None else so + o
            sso = o * o if sso is None else sso + o * o
        K = len(ens)
        pn = ((so / K + 1) / 2)
        sn = (torch.sqrt((sso / K - (so / K) ** 2).clamp_min(0)) / 2)
        vin = block_var((xb + 1) / 2)
        vout = block_var(p0)
        for name, t in [("P0", p0), ("Pe", pe), ("Se", se), ("Pn", pn), ("Sn", sn),
                        ("T", yb01), ("Vin", vin), ("Vout", vout)]:
            R[name].append(t.cpu().numpy())
        R["C"].append(sb.numpy())
    out = {k: np.concatenate(v).astype(np.float32) for k, v in R.items()}
    n = out["P0"].shape[0]
    out["err_base"] = np.abs(out["P0"] - out["T"]).reshape(n, -1)[:, idx].astype(np.float32)
    out["err_edl"] = np.abs(out["Pe"] - out["T"]).reshape(n, -1)[:, idx].astype(np.float32)
    out["err_ens"] = np.abs(out["Pn"] - out["T"]).reshape(n, -1)[:, idx].astype(np.float32)
    for k in ["Se", "Sn", "C"]:
        out[k] = out[k].reshape(n, -1)[:, idx] if k != "C" else out[k]
    for k in ["Vin", "Vout"]:
        v = out[k]  # (n,1,256,256)
        out[k] = np.stack([np.tile(v[i].ravel(), 3)[idx] for i in range(n)]).astype(np.float32)
    out["Se"] = out["Se"].astype(np.float32); out["Sn"] = out["Sn"].astype(np.float32)
    out["C"] = out["C"][:, 0] if out["C"].ndim == 2 else out["C"]
    return out


def inline_aurc(sig, err):
    """统一口径 AURC：按 σ 升序保留前 m 像素，累计风险对覆盖取平均"""
    o = np.argsort(sig, kind="mergesort")
    e = err[o]
    cum = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(cum.mean())


def pixel_lift(sig, err):
    o = np.argsort(sig)
    k = max(1, len(o) // 5)
    return float(err[o[-k:]].mean() / max(err[o[:k]].mean(), 1e-8))


def audit_variant(name, sig_test, err_test, d_test, sig_cal, ycal, tcal):
    z = err_test / np.clip(sig_test, 1e-8, None)
    cov1 = float((z < 1).mean())
    try:
        r = float(ut.optimize_recalibration_ratio(
            np.concatenate(ycal), np.concatenate(sig_cal), np.concatenate(tcal)))
    except Exception as ex:
        print("ratio fail", name, ex); r = 1.0
    cov1_rec = float((err_test / np.clip(sig_test * r, 1e-8, None) < 1).mean())
    # 图像级 lift（按图聚合再分层/排序）
    n = sig_test.shape[0]
    img_sig = sig_test.reshape(n, -1).mean(1)
    img_err = err_test.reshape(n, -1).mean(1)
    con = d_test["C"]
    o = np.argsort(img_sig); k = max(1, n // 5)
    img_lift = float(img_err[o[-k:]].mean() / max(img_err[o[:k]].mean(), 1e-8))
    order = np.argsort(con); bins = np.array_split(order, 4)
    strata = [float(((err_test[b] / np.clip(sig_test[b] * r, 1e-8, None)) < 1).mean()) for b in bins]
    # 像素级：池化子采样
    m = sig_test.size
    idx = np.random.default_rng(7).choice(m, size=min(POOL_PIX, m), replace=False)
    pl = pixel_lift(sig_test.ravel()[idx], err_test.ravel()[idx])
    aurc = inline_aurc(sig_test.ravel()[idx], err_test.ravel()[idx])
    return {"ratio": r, "marginal_cov1": cov1, "marginal_cov1_recal": cov1_rec,
            "strata_cov1_recal": strata, "img_lift": img_lift, "pixel_lift": pl, "aurc": aurc}


def main():
    torch.manual_seed(42)
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp_cal = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[:500]]
    euvp_test = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    up = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb_cal, uieb_test = up[:len(up) // 2], up[len(up) // 2:]

    det = GeneratorFunieGAN().cuda().eval()
    det.load_state_dict(torch.load(f"{ROOT}/ckpt/funie_baseline/best.pth", map_location="cuda", weights_only=True))
    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(f"{ROOT}/ckpt/funie_edl/best.pth", map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(f"{ROOT}/ckpt/funie_ensemble/seed_{s}/best.pth",
                                     map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    res = {}
    for ds, calp, testp in [("EUVP_域内", euvp_cal, euvp_test), ("UIEB_跨域", uieb_cal, uieb_test)]:
        print(f"=== {ds} ===", flush=True)
        dt = collect((det, edl, ens), testp)
        dc = collect((det, edl, ens), calp)
        variants = {
            "edl": (dt["Se"], dt["err_edl"], dc["Se"], dc["Pe"], dc["T"]),
            "ensemble": (dt["Sn"], dt["err_ens"], dc["Sn"], dc["Pn"], dc["T"]),
            "det_var_out": (dt["Vout"], dt["err_base"], dc["Vout"], dc["P0"], dc["T"]),
            "det_var_in": (dt["Vin"], dt["err_base"], dc["Vin"], dc["P0"], dc["T"]),
        }
        # 常数 σ：校准集拟合单一 c（用误差均值 std 匹配），预测 = 确定性基线
        c0 = float(np.mean(dc["err_base"]))
        variants["const"] = (np.full_like(dt["err_base"], c0), dt["err_base"],
                             np.full_like(dc["err_base"], c0), dc["P0"], dc["T"])
        res[ds] = {}
        for name, (st, et, sc, yc, tc) in variants.items():
            res[ds][name] = audit_variant(name, st, et, dt, sc, yc, tc)
            print(name, {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v])
                         for k, v in res[ds][name].items()}, flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("NAIVE_BASELINES_DONE")


if __name__ == "__main__":
    main()
