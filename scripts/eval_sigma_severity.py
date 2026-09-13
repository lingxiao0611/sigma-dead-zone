# -*- coding: utf-8 -*-
"""Stage 3d-2：无参考跨域 σ 诊断（拒识依据）。

无 GT 的数据集上测 mean σ，回答两个问题：
  Q1 域漂移是否被 σ 反映？（EUVP 域内 vs UIEB/U45/RUIE 跨域，σ 膨胀排序）
  Q2 σ 是否随退化严重度上升？（RUIE UIQS A~E 质量分级；U45 blue/green/haze 三类）
零核心算法自写，仅编排层。
"""
import os, sys, json, random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image

D = "/mnt/data/underwater/bpinn_uq/datasets"
CKPT_EDL = "/mnt/data/underwater/bpinn_uq/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"/mnt/data/underwater/bpinn_uq/ckpt/funie_ensemble/seed_{s}/best.pth"
SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"
OUT = "/mnt/data/underwater/bpinn_uq/outputs/sigma_severity.json"
N_PER_GROUP = 200          # 每组最多取多少张（控时间）
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


class ImgDS(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        A = Image.open(self.paths[i]).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x


def listdir_imgs(d, n=None):
    if not os.path.isdir(d):
        return []
    fs = sorted([os.path.join(d, f) for f in os.listdir(d)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if n is not None and len(fs) > n:
        r = random.Random(SEED)
        fs = r.sample(fs, n)
    return fs


def sigma_edl(model, paths):
    loader = DataLoader(ImgDS(paths), batch_size=16, num_workers=4, pin_memory=True)
    out = []
    with torch.no_grad():
        for xb in loader:
            o = model(xb.cuda())
            out.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
    return np.concatenate(out)


def sigma_ens(models, paths):
    loader = DataLoader(ImgDS(paths), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models)
    sq, s = [], []
    with torch.no_grad():
        for xb in loader:
            so, sso = None, None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            s.append(so); sq.append(sso)
    S = np.concatenate(s, axis=0); SS = np.concatenate(sq, axis=0)
    var = (SS - S * S / K) / (K - 1)
    return np.sqrt(np.clip(var, 0, None)) / 2


def main():
    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    u45 = f"{D}/underwater-test-dataset-U45-/upload/U45"
    uiqs = f"{D}/RUIE_Dataset/RUIE/UIQS"
    uccs = f"{D}/RUIE_Dataset/RUIE/UCCS"
    eu = f"{D}/UIEB_Dataset/UIEB Dataset/raw/UIEB_raw_reName"

    groups = {}
    groups["EUVP_val(域内基准)"] = sorted(
        json.load(open(SPLIT))["val_paths"])[:N_PER_GROUP]
    groups["UIEB_raw(跨域)"] = listdir_imgs(eu, N_PER_GROUP)
    groups["U45_all"] = listdir_imgs(f"{u45}/U45", N_PER_GROUP)
    for t in ["blue", "green", "haze"]:
        groups[f"U45_{t}"] = listdir_imgs(f"{u45}/{t}", None)
    for g in "ABCDE":
        groups[f"RUIE_UIQS_{g}"] = listdir_imgs(f"{uiqs}/{g}", N_PER_GROUP)
    for c in ["blue", "green", "blue-green"]:
        groups[f"RUIE_UCCS_{c}"] = listdir_imgs(f"{uccs}/{c}", None)

    res = {}
    print(f"{'group':26s} {'n':>5s} {'σ_EDL':>9s} {'σ_ENS':>9s}", flush=True)
    for name, paths in groups.items():
        if not paths:
            print(f"{name:26s}  EMPTY (跳过)", flush=True)
            continue
        se = float(sigma_edl(edl, paths).mean())
        sn = float(sigma_ens(ens, paths).mean())
        res[name] = {"n": len(paths), "sigma_edl": se, "sigma_ens": sn}
        print(f"{name:26s} {len(paths):5d} {se:9.5f} {sn:9.5f}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("SIGMA_SEVERITY_DONE")


if __name__ == "__main__":
    main()
