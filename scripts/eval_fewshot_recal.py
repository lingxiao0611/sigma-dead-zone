# -*- coding: utf-8 -*-
"""增强实验①：跨域 per-water 重校准的"少样本"代价曲线。

问题：3d-1 证明重校准 ratio 不能跨域直搬，必须拿目标水体的图标定。
      那到底要多少张目标域图，才能逼近"用满标定集"的效果？

协议：
  - 目标域 UIEB 890 对 → 前 445 当"标定池"，后 445 当测试集（与 3d-1 同切分）
  - 从标定池随机抽 N 张（N = 5/10/20/50/100/200/445），每张固定子采样像素后
    用 toolbox 现成 optimize_recalibration_ratio 学 ratio
  - 学到的 ratio 直接用到测试集 445 上，报 RMS 校准误差（+ z 覆盖）
  - 每个 N 跑 5 个随机 trial，给均值±标准差
  - 基线对照：不重校准(raw) / 直搬 EUVP ratio / 用满 445 张标定(上界)
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
import uncertainty_toolbox as ut

UIEB = "/mnt/data/underwater/bpinn_uq/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = "/mnt/data/underwater/bpinn_uq/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble/seed_{s}/best.pth"
EDL_REC = "/mnt/data/underwater/bpinn_uq/outputs/edl_recalibration.json"
ENS_REC = "/mnt/data/underwater/bpinn_uq/outputs/ensemble_recalibration.json"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/fewshot_recal_uieb.json"
N_SAMPLE, SEED = 200_000, 42
NS = [5, 10, 20, 50, 100, 200, 445]
TRIALS = 5


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


def flat(P, S, T, n=N_SAMPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(P.size, size=min(n, P.size), replace=False)
    return P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]


def collect_edl(model, pairs):
    ld = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T = [], [], []
    with torch.no_grad():
        for xb, yb in ld:
            o = model(xb.cuda())
            P.append(((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy())
    return (np.concatenate(P).astype(np.float32), np.concatenate(S).astype(np.float32),
            np.concatenate(T).astype(np.float32))


def collect_ens(models, pairs):
    ld = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models)
    sl, sql, Tl = [], [], []
    with torch.no_grad():
        for xb, yb in ld:
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


def rms_at(yp, ys, yt, ratio):
    return float(ut.root_mean_squared_calibration_error(yp, ys * ratio, yt))


def z1_at(yp, ys, yt, ratio):
    z = np.abs(yp - yt) / np.clip(ys * ratio, 1e-8, None)
    return float((z < 1).mean())


def main():
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    pairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    half = len(pairs) // 2
    cal_pairs, test_pairs = pairs[:half], pairs[half:]
    print(f"UIEB 标定池 {len(cal_pairs)} / 测试集 {len(test_pairs)}", flush=True)

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    r_edl_eu = float(json.load(open(EDL_REC))["ratio"])
    r_ens_eu = float(json.load(open(ENS_REC))["ratio"])

    res = {}
    for mname, model, col, r_eu in [("edl", edl, collect_edl, r_edl_eu),
                                    ("ensemble", ens, collect_ens, r_ens_eu)]:
        Pc, Sc, Tc = col(model, cal_pairs)
        Pt, St, Tt = col(model, test_pairs)
        yp, ys, yt = flat(Pt, St, Tt)

        # 参考线
        yc_all, sc_all, tc_all = flat(Pc, Sc, Tc)
        r_full = float(ut.optimize_recalibration_ratio(yc_all, sc_all, tc_all))
        ref = {"raw_rms": rms_at(yp, ys, yt, 1.0), "raw_z1": z1_at(yp, ys, yt, 1.0),
               "eu_ratio": r_eu, "eu_rms": rms_at(yp, ys, yt, r_eu),
               "eu_z1": z1_at(yp, ys, yt, r_eu),
               "full_ratio": r_full, "full_rms": rms_at(yp, ys, yt, r_full),
               "full_z1": z1_at(yp, ys, yt, r_full)}
        print(f"[{mname}] raw RMS={ref['raw_rms']:.4f} | EUVP ratio={r_eu:.3f}→{ref['eu_rms']:.4f} "
              f"| 满标定 ratio={r_full:.3f}→{ref['full_rms']:.4f}", flush=True)

        sweep = {}
        for N in NS:
            ratios, rmss, z1s = [], [], []
            for t in range(TRIALS):
                rng = np.random.default_rng(1000 + t)
                img_idx = rng.choice(len(cal_pairs), size=min(N, len(cal_pairs)), replace=False)
                yc, sc, tc = flat(Pc[img_idx], Sc[img_idx], Tc[img_idx], seed=SEED + t)
                rr = float(ut.optimize_recalibration_ratio(yc, sc, tc))
                ratios.append(rr)
                rmss.append(rms_at(yp, ys, yt, rr))
                z1s.append(z1_at(yp, ys, yt, rr))
            sweep[str(N)] = {
                "ratio_mean": float(np.mean(ratios)), "ratio_std": float(np.std(ratios)),
                "rms_mean": float(np.mean(rmss)), "rms_std": float(np.std(rmss)),
                "z1_mean": float(np.mean(z1s)), "z1_std": float(np.std(z1s)),
            }
            print(f"   N={N:4d}  ratio={np.mean(ratios):.3f}±{np.std(ratios):.3f}  "
                  f"RMS={np.mean(rmss):.4f}±{np.std(rmss):.4f}  "
                  f"1σ={np.mean(z1s):.3f}", flush=True)
        res[mname] = {"reference": ref, "sweep": sweep}
        del Pc, Sc, Tc, Pt, St, Tt
        torch.cuda.empty_cache()

    json.dump(res, open(OUT, "w"), indent=1)
    print("FEWSHOT_RECAL_DONE")


if __name__ == "__main__":
    main()
