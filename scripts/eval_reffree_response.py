# -*- coding: utf-8 -*-
"""E3 无参考集 σ 响应性审计：U45（blue/green/haze）+ RUIE-UIQS（A~E 五档）。

死区主张（σ 对退化零响应）不需要 GT——只需要 σ 与输入。
对每个子集报：每类平均 σ、响应比 σ(最重档)/σ(最轻档)、ρ(对比度, σ) 仅作参考。
模型：EDL@FUnIE、Ensemble@FUnIE、EDL@U-shape（可移植头双主干）。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/u_shape_transformer")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from net.Ushape_Trans import Generator
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
SEEDS = [101, 202, 303, 404, 505]
OUT = f"{ROOT}/outputs/reffree_response.json"
U45 = f"{ROOT}/datasets/underwater-test-dataset-U45-/upload/U45"
RUIE = f"{ROOT}/datasets/RUIE_Dataset/RUIE/UIQS"
N_PER_CAT = 200


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


class UshapeNIG(nn.Module):
    def __init__(self):
        super().__init__()
        g = Generator()
        self.g = g
        self.g.feature_to_rgb[0] = NormalInverseGammaConvNd(
            nn.Conv2d, event_dim=3, in_channels=32, kernel_size=1)

    def forward(self, x):
        return self.g(x)[-1]


def load_batch(files):
    xs = []
    for f in files:
        A = Image.open(f).convert("RGB").resize((256, 256))
        xs.append(torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1)
    return torch.stack(xs)


@torch.no_grad()
def sigma_maps(model, x):
    o = model(x.cuda())
    if "loc" in o:
        return (torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).mean(dim=[1, 2, 3]).cpu().numpy()
    return None


@torch.no_grad()
def ens_sigma(models, x):
    so = sso = None
    for m in models:
        o = m(x.cuda()).clamp(-1, 1)
        so = o if so is None else so + o
        sso = o * o if sso is None else sso + o * o
    K = len(models)
    return (torch.sqrt(((sso / K) - (so / K) ** 2).clamp_min(0)) / 2).mean(dim=[1, 2, 3]).cpu().numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    torch.manual_seed(42)
    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(f"{ROOT}/ckpt/funie_edl/best.pth", map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(f"{ROOT}/ckpt/funie_ensemble/seed_{s}/best.pth",
                                     map_location="cuda", weights_only=True))
    ush = UshapeNIG().cuda().eval()
    ush.load_state_dict(torch.load(f"{ROOT}/ckpt/ushape_edl/best.pth", map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    groups = [("U45_blue", sorted(os.listdir(os.path.join(U45, "blue")))[:N_PER_CAT],
               os.path.join(U45, "blue")),
              ("U45_green", sorted(os.listdir(os.path.join(U45, "green")))[:N_PER_CAT],
               os.path.join(U45, "green")),
              ("U45_haze", sorted(os.listdir(os.path.join(U45, "haze")))[:N_PER_CAT],
               os.path.join(U45, "haze"))]
    for g in "ABCDE":
        fs = sorted(os.listdir(os.path.join(RUIE, g)))[:N_PER_CAT]
        groups.append((f"RUIE_{g}", fs, os.path.join(RUIE, g)))

    res = {}
    for name, files, d in groups:
        paths = [os.path.join(d, f) for f in files]
        s_e, s_n, s_u, con = [], [], [], []
        for i in range(0, len(paths), 16):
            x = load_batch(paths[i:i + 16])
            x01 = ((x + 1) / 2).numpy().transpose(0, 2, 3, 1)
            for im in x01:
                g_ = im @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                con.append(float(g_.std()))
            s_e.extend(sigma_maps(edl, x).tolist())
            s_n.extend(ens_sigma(ens, x).tolist())
            s_u.extend(sigma_maps(ush, x).tolist())
        res[name] = {"n": len(paths), "sigma_edl": float(np.mean(s_e)),
                     "sigma_ens": float(np.mean(s_n)), "sigma_ushape_edl": float(np.mean(s_u)),
                     "contrast": float(np.mean(con)),
                     "rho_contrast_edl": spearman(con, s_e),
                     "rho_contrast_ens": spearman(con, s_n),
                     "rho_contrast_ushape": spearman(con, s_u)}
        print(name, {k: round(v, 4) for k, v in res[name].items()}, flush=True)

    # 汇总响应比（最重档 / 最轻档 的平均 σ）
    def ratio(keys, key):
        vals = [res[k][key] for k in keys]
        return float(max(vals) / max(min(vals), 1e-9))
    res["_summary"] = {
        "U45_resp_edl": ratio(["U45_blue", "U45_green", "U45_haze"], "sigma_edl"),
        "U45_resp_ens": ratio(["U45_blue", "U45_green", "U45_haze"], "sigma_ens"),
        "U45_resp_ushape": ratio(["U45_blue", "U45_green", "U45_haze"], "sigma_ushape_edl"),
        "RUIE_resp_edl": float(res["RUIE_E"]["sigma_edl"] / max(res["RUIE_A"]["sigma_edl"], 1e-9)),
        "RUIE_resp_ens": float(res["RUIE_E"]["sigma_ens"] / max(res["RUIE_A"]["sigma_ens"], 1e-9)),
        "RUIE_resp_ushape": float(res["RUIE_E"]["sigma_ushape_edl"] / max(res["RUIE_A"]["sigma_ushape_edl"], 1e-9)),
    }
    print("summary:", {k: round(v, 3) for k, v in res["_summary"].items()}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("REFFREE_RESPONSE_DONE")


if __name__ == "__main__":
    main()
