# -*- coding: utf-8 -*-
"""增强实验 A：软拒识门控融合（UIEB 跨域，有 GT）。

问题：EUVP 训练的复原模型在跨域（UIEB）上，部分图/区域可能"越修越糟"。
方案：逐像素把 复原图 与 原图 按不确定度融合 —— 高 σ 像素回退到原图。
    fused = alpha(σ)·P + (1-alpha(σ))·X,  alpha 在 [lo, hi] 线性从 1 降到 0。
门控只用 σ 的"排序/相对大小"（刻度无关），不重校准。
两档对比：EDL σ vs Ensemble σ（预期 ENS 门控有净收益、EDL 门控因 σ 反向失效而无收益）。
协议：UIEB 890 对固定顺序拆 cal 445（网格搜 lo/hi）/ test 445（独立验证）。
指标：逐图 PSNR（[0,1] 域 peak=1）均值 + 最差 10% + 逐图计数。
零核心算法自写，仅编排层。
"""
import os, sys, json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image

UIEB = "/mnt/data/underwater/bpinn_uq/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = "/mnt/data/underwater/bpinn_uq/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble/seed_{s}/best.pth"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/fusion_gate_uieb.json"
N_TUNE = 300_000
RNG = np.random.default_rng(42)


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


def build_pairs():
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    pairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    print(f"UIEB 配对数 {len(pairs)}", flush=True)
    return pairs


# 返回 (N,3,256,256) 的 [0,1] 域数组：P=复原, S=σ, X=原图, T=GT
def collect_edl(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, X, T = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.cuda())
            P.append(((out["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(out["beta"] / (out["alpha"] - 1)) / 2).cpu().numpy())
            X.append(((xb + 1) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy())
    return tuple(np.concatenate(a).astype(np.float32) for a in (P, S, X, T))


def collect_ens(models, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models)
    P, S, X, T = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            so, sso = None, None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            X.append(((xb + 1) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy())
            P.append(so); S.append(sso)
    Ss = np.concatenate(S, axis=0); Sm = np.concatenate(P, axis=0)
    mean = Sm / K
    var = (Ss - Sm * Sm / K) / (K - 1)
    std = np.sqrt(np.clip(var, 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), \
        np.concatenate(X).astype(np.float32), np.concatenate(T).astype(np.float32)


def fuse_alpha(s, lo, hi):
    if hi <= lo:
        return np.ones_like(s)
    a = (hi - s) / (hi - lo)
    return np.clip(a, 0.0, 1.0)


def img_psnr(fused01, gt01):
    """逐图 PSNR: 输入 (N,3,256,256)，返回 (N,)"""
    mse = ((fused01 - gt01) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(1.0 / np.clip(mse, 1e-10, None))


def summarize(psnr, name):
    return {f"{name}_mean": float(psnr.mean()),
            f"{name}_p10": float(np.percentile(psnr, 10)),
            f"{name}_worst": float(psnr.min())}


def main():
    pairs = build_pairs()
    half = len(pairs) // 2
    cal_pairs, test_pairs = pairs[:half], pairs[half:]
    print(f"UIEB cal {len(cal_pairs)} / test {len(test_pairs)}", flush=True)

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    res = {}
    for name, model, col in [("edl", edl, collect_edl),
                             ("ensemble", ens, collect_ens)]:
        Pc, Sc, Xc, Tc = col(model, cal_pairs)
        Pt, St, Xt, Tt = col(model, test_pairs)
        print(f"[{name}] cal P={Pc.shape} σmean={Sc.mean():.4f} | test P={Pt.shape} σmean={St.mean():.4f}", flush=True)

        # ---- 调参：在 cal 上网格搜 (qlo,qhi)（σ 分位数），全局 MSE 最小 ----
        rng = RNG
        idx = rng.choice(Pc.size, size=min(N_TUNE, Pc.size), replace=False)
        sc_, pc_, xc_, tc_ = Sc.flatten()[idx], Pc.flatten()[idx], Xc.flatten()[idx], Tc.flatten()[idx]
        qlo_c, qhi_c = [0.0, 0.5, 0.7, 0.8, 0.9], [0.9, 0.95, 0.99, 1.0]
        best = None
        for qlo in qlo_c:
            for qhi in qhi_c:
                if qhi <= qlo:
                    continue
                lo = float(np.quantile(Sc, qlo)); hi = float(np.quantile(Sc, qhi))
                a = fuse_alpha(sc_, lo, hi)
                fused = a * pc_ + (1 - a) * xc_
                mse = float(((fused - tc_) ** 2).mean())
                if best is None or mse < best[0]:
                    best = (mse, qlo, qhi, lo, hi)
        _, qlo, qhi, lo, hi = best
        print(f"[{name}] best gate: qlo={qlo} qhi={qhi} lo={lo:.5f} hi={hi:.5f} calMSE={_:.6f}", flush=True)

        # ---- 对照①：同回退量的"盲均匀门控"（不看 σ 位置）----
        # 在 cal 全图上算门控的平均 α（总回退量），test 上把这个 α 均匀作用于每个像素。
        # 若 σ 门控 > 均匀门控，说明 σ 的"空间排序"确实带来精度增益。
        A_cal = fuse_alpha(Sc, lo, hi)
        alpha_bar = float(A_cal.mean())
        # ---- 对照②：最优常数融合 w*（不看 σ 也不看位置，盲融合上界）----
        w_cands = np.linspace(0.0, 1.0, 21)
        w_mse = [float((((w * pc_ + (1 - w) * xc_) - tc_) ** 2).mean()) for w in w_cands]
        w_star = float(w_cands[int(np.argmin(w_mse))])

        # ---- test 独立验证：逐图 PSNR ----
        # 1) 门控融合
        A = fuse_alpha(St, lo, hi)
        fused = A * Pt + (1 - A) * Xt
        ps_gate = img_psnr(fused, Tt)
        # 2) 裸复原（无门控）
        ps_naive = img_psnr(Pt, Tt)
        # 3) 原图直接上（下限锚）
        ps_input = img_psnr(Xt, Tt)

        gain = ps_gate - ps_naive
        # 均匀门控（总回退量与 σ 门控一致，但无空间选择性）
        ps_uniform = img_psnr(alpha_bar * Pt + (1 - alpha_bar) * Xt, Tt)
        # 最优常数融合（盲融合上界）
        ps_wstar = img_psnr(w_star * Pt + (1 - w_star) * Xt, Tt)
        res[name] = {
            "gate_params": {"qlo": qlo, "qhi": qhi, "lo": lo, "hi": hi, "cal_global_mse": _,
                            "alpha_bar": alpha_bar, "w_star": w_star},
            "n_test": int(len(test_pairs)),
            **summarize(ps_input, "input"),
            **summarize(ps_naive, "naive"),
            **summarize(ps_gate, "gate"),
            **summarize(ps_uniform, "uniform"),
            **summarize(ps_wstar, "wstar"),
            "gate_minus_naive_mean": float(gain.mean()),
            "gate_minus_uniform_mean": float((ps_gate - ps_uniform).mean()),
            "gate_minus_wstar_mean": float((ps_gate - ps_wstar).mean()),
            "n_gate_improves": int((gain > 0).sum()),
            "frac_gate_improves": float((gain > 0).mean()),
            "harmful_restoration": {  # 裸复原比原图还糟的图
                "n": int((ps_naive < ps_input).sum()),
                "naive_mean_psnr": float(ps_naive[ps_naive < ps_input].mean()) if (ps_naive < ps_input).any() else None,
                "gate_mean_psnr": float(ps_gate[ps_naive < ps_input].mean()) if (ps_naive < ps_input).any() else None,
                "uniform_mean_psnr": float(ps_uniform[ps_naive < ps_input].mean()) if (ps_naive < ps_input).any() else None,
            },
        }
        print(f"[{name}] input {ps_input.mean():.3f} | naive {ps_naive.mean():.3f} | "
              f"gate {ps_gate.mean():.3f} | uniform {ps_uniform.mean():.3f} | wstar {ps_wstar.mean():.3f} | "
              f"gain_vs_naive {gain.mean():+.4f} | gain_vs_uniform {(ps_gate-ps_uniform).mean():+.4f} | "
              f"improved {int((gain > 0).sum())}/{len(test_pairs)}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("FUSION_GATE_DONE")


if __name__ == "__main__":
    main()
