# -*- coding: utf-8 -*-
"""Stage 3d-1：跨水体域迁移校准（EUVP 训练 → UIEB 890 对，有参考图）。

核心问题：EUVP 上学到的重校准 ratio，能否直接迁移到另一个水体（UIEB）？
协议沿用 3b/3c：UIEB 890 对固定顺序拆 cal 445 / test 445；
对比三档 σ 刻度：原始 / ×EUVP ratio（跨域直搬）/ ×UIEB ratio（域内上界）。
同时报 mean σ 的域间膨胀（EUVP val vs UIEB），看"模型是否知道自己出域了"。
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
import uncertainty_toolbox as ut

UIEB = "/mnt/data/underwater/bpinn_uq/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = "/mnt/data/underwater/bpinn_uq/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble/seed_{s}/best.pth"
EDL_REC = "/mnt/data/underwater/bpinn_uq/outputs/edl_recalibration.json"
ENS_REC = "/mnt/data/underwater/bpinn_uq/outputs/ensemble_recalibration.json"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/crossdomain_uieb.json"
N_SAMPLE = 200_000
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
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y


def build_pairs():
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    pairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    print(f"UIEB 配对数 {len(pairs)} / raw {len(raw)} / ref {len(ref)}", flush=True)
    return pairs


def flat(P, S, T):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P.size, size=min(N_SAMPLE, P.size), replace=False)
    return P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]


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


# ---------------- EDL collect ----------------
def collect_edl(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.cuda())
            P.append(((out["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(out["beta"] / (out["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy())
    return (np.concatenate(P).astype(np.float32), np.concatenate(S).astype(np.float32),
            np.concatenate(T).astype(np.float32))


# ---------------- Ensemble collect ----------------
def collect_ens(models, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models)
    sum_list, sumsq_list, T_list = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            so, sso = None, None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            sum_list.append(so); sumsq_list.append(sso)
            T_list.append(((yb + 1) / 2).numpy())
    S = np.concatenate(sum_list, axis=0); SS = np.concatenate(sumsq_list, axis=0)
    T = np.concatenate(T_list, axis=0)
    mean = S / K
    var = (SS - S * S / K) / (K - 1)
    std = np.sqrt(np.clip(var, 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), T.astype(np.float32)


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

    eu_edl = float(json.load(open(EDL_REC))["ratio"])
    eu_ens = float(json.load(open(ENS_REC))["ratio"])
    print(f"EUVP 学到的 ratio: EDL={eu_edl:.3f}  Ensemble={eu_ens:.3f}", flush=True)

    res = {"eu_ratio": {"edl": eu_edl, "ensemble": eu_ens}}
    for name, model, col, eu in [("edl", edl, collect_edl, eu_edl),
                                 ("ensemble", ens, collect_ens, eu_ens)]:
        Pc, Sc, Tc = col(model, cal_pairs)
        yc, sc, tc = flat(Pc, Sc, Tc)
        ratio_dom = float(ut.optimize_recalibration_ratio(yc, sc, tc))

        Pt, St, Tt = col(model, test_pairs)
        yp, ys, yt = flat(Pt, St, Tt)
        res[name] = {
            "mean_sigma_uieb": float(St.mean()),
            "ratio_domain": ratio_dom,
            "raw": pick(yp, ys, yt),
            "z_raw": zcov(yp, ys, yt),
            "xfer": pick(yp, ys * eu, yt),
            "z_xfer": zcov(yp, ys * eu, yt),
            "domain": pick(yp, ys * ratio_dom, yt),
            "z_domain": zcov(yp, ys * ratio_dom, yt),
        }
        print(f"[{name}] UIEB域内 ratio={ratio_dom:.3f} (EUVP={eu:.3f}) "
              f"meanσ={St.mean():.4f}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("CROSSDOMAIN_UIEB_DONE")


if __name__ == "__main__":
    main()
