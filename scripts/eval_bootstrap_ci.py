# -*- coding: utf-8 -*-
"""C10: per-claim cluster bootstrap CI（替代 Friedman/Nemenyi 的正确统计学）。

对论文的两个 money claim 做效应级不确定性量化：
  1) 跨域 AURC 差（ENS 0.0974 vs EDL 0.1084）——image-level cluster bootstrap
  2) 图像级 risk lift 差（ENS 1.11 vs EDL 0.87）
bootstrap = 整图重采样（保留像素嵌套结构，与 3.2/2.3 的嵌套论证自洽）。
配对 bootstrap（同一重采样索引用于两方法）以考虑两方法在同一图上的相关。

复用 eval_rejection.py 的模型加载/收集逻辑；逐图收集 err/σ（非展平），
每图固定子采样 4096 像素（seed 固定，跨 bootstrap 重采样共用同一子样以降方差）。

产物：
  outputs/bootstrap_ci.json           —— CI 结果
  outputs/arrays_uieb.npz             —— UIEB 逐图 (err, sigma) 子样（C11 toolbox 复用）
  outputs/arrays_euvp.npz             —— EUVP 逐图 (err, sigma) 子样
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
from uq.tu_risk_coverage import regression_aurc
from PIL import Image

SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
UIEB = "/mnt/data/underwater/bpinn_uq/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = "/mnt/data/underwater/bpinn_uq/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble/seed_{s}/best.pth"
OUT_JSON = "/mnt/data/underwater/bpinn_uq/outputs/bootstrap_ci.json"
PX_PER_IMG = 4096
B_BOOT = 1000
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


def collect_edl_perimg(model, pairs):
    errs, sigs, cons = [], [], []
    for a, b in pairs:
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1)[None].cuda() * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1)[None].cuda() * 2 - 1
        with torch.no_grad():
            o = model(x)
            pred = ((o["loc"].clamp(-1, 1) + 1) / 2)
            sig = (torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2)
        errs.append((pred - (y + 1) / 2).abs()[0].cpu().numpy())   # (3,H,W) y 已转回 [0,1]
        sigs.append(sig[0].cpu().numpy())                          # (3,H,W)
        cons.append(float(x.mean(dim=1, keepdim=True).std().item())) # 输入灰度对比度
    return np.stack(errs), np.stack(sigs), np.array(cons)  # (N,3,H,W),(N,3,H,W),(N,)


def collect_ens_perimg(models, pairs):
    errs, sigs, cons = [], [], []
    K = len(models)
    for a, b in pairs:
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1)[None].cuda() * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1)[None].cuda() * 2 - 1
        with torch.no_grad():
            so, sso = None, None
            for m in models:
                o = m(x).clamp(-1, 1)
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            mean, std = so / K, torch.sqrt(torch.clamp((sso - so * so / K) / (K - 1), 0, None))
        errs.append(((mean + 1) / 2 - (y + 1) / 2).abs()[0].cpu().numpy())  # (3,H,W)
        sigs.append((std / 2)[0].cpu().numpy())                             # (3,H,W)
        cons.append(float(x.mean(dim=1, keepdim=True).std().item()))
    return np.stack(errs), np.stack(sigs), np.array(cons)


def subsample(errs, sigs, rng):
    """每图固定子采样 PX_PER_IMG 像素（跨方法/跨 draw 共用同一索引）。
    保留 3 通道 (N,3,P) float32——同时支持两种估计对象：
      channel-level（Stage4 复现：3 通道各算独立样本）与
      pixel-level（论文 §3.2 部署粒度：3 通道均值后排序）。
    σ 必须 float32——float16 在 σ~0.017 处步长 6e-5，抹掉 1e-5 量级排名差异。"""
    N, C, H, W = sigs.shape
    flat_idx = rng.permutation(H * W)[:PX_PER_IMG]
    return (errs.reshape(N, C, -1)[:, :, flat_idx].astype(np.float32),
            sigs.reshape(N, C, -1)[:, :, flat_idx].astype(np.float32))


def pooled_aurc(err_sub, sig_sub, level="pixel"):
    if level == "pixel":
        e = err_sub.mean(axis=1).reshape(-1)
        s = sig_sub.mean(axis=1).reshape(-1)
    else:  # channel-level：3 通道各算独立样本（Stage4 的估计对象）
        e = err_sub.reshape(-1)
        s = sig_sub.reshape(-1)
    return float(regression_aurc(torch.from_numpy(-s), torch.from_numpy(e)))


def img_lift(err_sub, sig_sub):
    """图像级 lift：按图均 σ 排序，top-20% 图均 err / bottom-20% 图均 err。
    兼容 (N,P) 与 (N,3,P) 两种输入——图级统计量对通道取均值。"""
    ime = err_sub.reshape(err_sub.shape[0], -1).mean(axis=1).astype(np.float32)
    ims = sig_sub.reshape(sig_sub.shape[0], -1).mean(axis=1).astype(np.float32)
    n = len(ime); k = max(1, int(0.2 * n))
    top = ime[np.argsort(-ims)[:k]].mean()
    bot = ime[np.argsort(ims)[:k]].mean()
    return float(top / bot)


def main():
    rng = np.random.default_rng(SEED)
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp_test = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    uieb_pairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb_test = uieb_pairs[len(uieb_pairs) // 2:]

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    out = {}
    for dsname, pairs in [("UIEB_跨域", uieb_test), ("EUVP_域内", euvp_test)]:
        print(f"[{dsname}] collecting per-image arrays...", flush=True)
        e_edl, s_edl, c_vec = collect_edl_perimg(edl, pairs)
        e_ens, s_ens, _ = collect_ens_perimg(ens, pairs)
        sub_edl = subsample(e_edl, s_edl, rng)
        sub_ens = subsample(e_ens, s_ens, rng)
        np.savez_compressed(f"/mnt/data/underwater/bpinn_uq/outputs/arrays_{dsname.split('_')[0]}.npz",
                            err_edl=sub_edl[0], sig_edl=sub_edl[1],
                            err_ens=sub_ens[0], sig_ens=sub_ens[1],
                            contrast=c_vec.astype(np.float32))
        N = sub_edl[0].shape[0]
        idx = np.arange(N)
        aurc_edl, aurc_ens, lift_edl, lift_ens = [], [], [], []
        for b in range(B_BOOT):
            bi = rng.choice(idx, size=N, replace=True)
            aurc_edl.append(pooled_aurc(sub_edl[0][bi], sub_edl[1][bi], level="pixel"))
            aurc_ens.append(pooled_aurc(sub_ens[0][bi], sub_ens[1][bi], level="pixel"))
            lift_edl.append(img_lift(sub_edl[0][bi], sub_edl[1][bi]))
            lift_ens.append(img_lift(sub_ens[0][bi], sub_ens[1][bi]))
            if (b + 1) % 200 == 0:
                print(f"  boot {b+1}/{B_BOOT}", flush=True)

        # channel-level 复现点估计（对齐 Stage4 的估计对象，只算点值不进 bootstrap）
        aurc_edl_ch = pooled_aurc(sub_edl[0], sub_edl[1], level="channel")
        aurc_ens_ch = pooled_aurc(sub_ens[0], sub_ens[1], level="channel")

        def ci(x):
            x = np.array(x)
            return dict(mean=float(x.mean()), lo=float(np.percentile(x, 2.5)),
                        hi=float(np.percentile(x, 97.5)))
        aurc_edl, aurc_ens = np.array(aurc_edl), np.array(aurc_ens)
        lift_edl, lift_ens = np.array(lift_edl), np.array(lift_ens)
        d_aurc = aurc_edl - aurc_ens          # >0 = ENS 更好
        d_lift = lift_ens - lift_edl          # >0 = ENS 更好
        out[dsname] = {
            "n_images": int(N), "px_per_img": PX_PER_IMG, "n_boot": B_BOOT,
            "aurc_edl": ci(aurc_edl), "aurc_ens": ci(aurc_ens),
            "aurc_edl_channel_level_point": aurc_edl_ch,
            "aurc_ens_channel_level_point": aurc_ens_ch,
            "lift_edl": ci(lift_edl), "lift_ens": ci(lift_ens),
            "delta_aurc_edl_minus_ens": dict(mean=float(d_aurc.mean()),
                                             lo=float(np.percentile(d_aurc, 2.5)),
                                             hi=float(np.percentile(d_aurc, 97.5)),
                                             p_gt0=float((d_aurc > 0).mean())),
            "delta_lift_ens_minus_edl": dict(mean=float(d_lift.mean()),
                                             lo=float(np.percentile(d_lift, 2.5)),
                                             hi=float(np.percentile(d_lift, 97.5)),
                                             p_gt0=float((d_lift > 0).mean())),
        }
        print(f"[{dsname}] AURC EDL {out[dsname]['aurc_edl']['mean']:.4f} "
              f"[{out[dsname]['aurc_edl']['lo']:.4f},{out[dsname]['aurc_edl']['hi']:.4f}] | "
              f"ENS {out[dsname]['aurc_ens']['mean']:.4f} "
              f"[{out[dsname]['aurc_ens']['lo']:.4f},{out[dsname]['aurc_ens']['hi']:.4f}] | "
              f"Δ(EDL-ENS) p(>0)={out[dsname]['delta_aurc_edl_minus_ens']['p_gt0']:.3f}", flush=True)
        print(f"[{dsname}] lift EDL {out[dsname]['lift_edl']['mean']:.3f} | "
              f"ENS {out[dsname]['lift_ens']['mean']:.3f} | "
              f"Δ p(>0)={out[dsname]['delta_lift_ens_minus_edl']['p_gt0']:.3f}", flush=True)

    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("BOOTSTRAP_DONE ->", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
