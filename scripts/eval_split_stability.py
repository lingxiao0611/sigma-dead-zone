# -*- coding: utf-8 -*-
"""补强 A：重校准 ratio 的切分稳定性。
固定像素缓存（每图 32k 子采样，seed42），K=100 次随机图像级 cal/test 切分，
每次：cal 上优化 ratio → test 上评估 RMS-CE(raw/recal) 与 1σ 覆盖(recal)。
输出 outputs/split_stability.json：每配置 mean±std。
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

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = f"{ROOT}/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"{ROOT}/ckpt/funie_ensemble/seed_{s}/best.pth"
OUT = f"{ROOT}/outputs/split_stability.json"
K_SPLIT, PX_PER_IMG, SEED = 30, 32768, 42


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


def collect_edl(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            o = model(xb.cuda())
            P.append(((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
            T.append(((yb + 1) / 2).numpy())
    return tuple(np.concatenate(a).astype(np.float32) for a in (P, S, T))


def collect_ens(models, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models); sl, sql, Tl = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            so = sso = None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            sl.append(so); sql.append(sso); Tl.append(((yb + 1) / 2).numpy())
    S = np.concatenate(sl, 0); SS = np.concatenate(sql, 0); T = np.concatenate(Tl, 0)
    mean = S / K
    std = np.sqrt(np.clip((SS - S * S / K) / (K - 1), 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), T.astype(np.float32)


def cache_per_image(P, S, T, n_px=PX_PER_IMG, seed=SEED):
    """每图随机固定 n_px 像素 → (n_img, n_px) 矩阵，省内存"""
    n = len(P)
    rng = np.random.default_rng(seed)
    idx = rng.choice(P[0].size, size=min(n_px, P[0].size), replace=False)
    return (P.reshape(n, -1)[:, idx], S.reshape(n, -1)[:, idx], T.reshape(n, -1)[:, idx])


def rms_ce(yp, ys, yt):
    return float(ut.root_mean_squared_calibration_error(yp, ys, yt))


def main():
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths]   # 全部 1000
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    uieb = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    res = {}
    for dsname, pairs, n_cal in [("EUVP", euvp, 500), ("UIEB", uieb, len(uieb) // 2)]:
        res[dsname] = {}
        for mname, model, col in [("edl", edl, collect_edl), ("ensemble", ens, collect_ens)]:
            P, S, T = col(model, pairs)
            Ip, Is, It = cache_per_image(P, S, T)
            n = len(Ip)
            ratios, rms_raw, rms_rec, cov1_rec = [], [], [], []
            for k in range(K_SPLIT):
                rng = np.random.default_rng(1000 + k)
                perm = rng.permutation(n)
                ci, ti = perm[:n_cal], perm[n_cal:]
                r2 = np.random.default_rng(2000 + k)
                ci_pix = r2.choice(Ip[ci].size, size=min(50_000, Ip[ci].size), replace=False)
                ti_pix = r2.choice(Ip[ti].size, size=min(50_000, Ip[ti].size), replace=False)
                ycal = Ip[ci].ravel()[ci_pix]; scal = Is[ci].ravel()[ci_pix]; tcal = It[ci].ravel()[ci_pix]
                ytest = Ip[ti].ravel()[ti_pix]; stest = Is[ti].ravel()[ti_pix]; ttest = It[ti].ravel()[ti_pix]
                try:
                    r = float(ut.optimize_recalibration_ratio(ycal, scal, tcal))
                except Exception as e:
                    print("ratio fail", k, e, flush=True); continue
                ratios.append(r)
                rms_raw.append(rms_ce(ytest, stest, ttest))
                rms_rec.append(rms_ce(ytest, stest * r, ttest))
                z = np.abs(ytest - ttest) / np.clip(stest * r, 1e-8, None)
                cov1_rec.append(float((z < 1).mean()))
                if (k + 1) % 10 == 0:
                    print(f"  split {k+1}/{K_SPLIT}: ratio={r:.3f} ({len(ratios)} ok)", flush=True)
            res[dsname][mname] = {
                "ratio_mean": float(np.mean(ratios)), "ratio_std": float(np.std(ratios)),
                "rms_raw_mean": float(np.mean(rms_raw)), "rms_raw_std": float(np.std(rms_raw)),
                "rms_recal_mean": float(np.mean(rms_rec)), "rms_recal_std": float(np.std(rms_rec)),
                "cov1_recal_mean": float(np.mean(cov1_rec)), "cov1_recal_std": float(np.std(cov1_rec)),
                "n_splits": len(ratios)}
            print(f"[{dsname}/{mname}] ratio={np.mean(ratios):.3f}±{np.std(ratios):.3f}  "
                  f"RMS-CE raw={np.mean(rms_raw):.4f}±{np.std(rms_raw):.4f} → "
                  f"recal={np.mean(rms_rec):.4f}±{np.std(rms_rec):.4f}  "
                  f"cov1(recal)={np.mean(cov1_rec):.3f}±{np.std(cov1_rec):.3f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("SPLIT_STABILITY_DONE")


if __name__ == "__main__":
    main()
