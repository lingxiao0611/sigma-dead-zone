# -*- coding: utf-8 -*-
"""E5 机理探针：输入梯度敏感度 + 合成退化剂量-响应。

(a) 梯度敏感度：∂σ/∂x 与 ∂α/∂x 的范数，域内(EUVP) vs 跨域(UIEB) 对比——
    若跨域梯度塌缩，"头对跨域输入失去敏感度"有梯度级直接证据。
(b) 合成退化剂量-响应：gamma 连续扫描（仅证明"头有能力表达严重度梯度"这一能力命题，
    不承担因果证明——因果重量在 E1 干预实验）。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/funie_gan/PyTorch")
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from nets.funiegan import GeneratorFunieGAN
from uq.nig_layer import NormalInverseGammaConvNd
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
OUT = f"{ROOT}/outputs/mechanism_probes.json"
N_IMG = 40


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


def load_batch(paths):
    xs = []
    for p in paths:
        A = Image.open(p).convert("RGB").resize((256, 256))
        xs.append(torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1)
    return torch.stack(xs)


@torch.enable_grad()
def grad_norms(model, paths):
    gn_s, gn_a, sg = [], [], []
    for i in range(0, len(paths), 8):
        x = load_batch(paths[i:i + 8]).cuda().requires_grad_(True)
        o = model(x)
        sig = torch.sqrt(o["beta"] / (o["alpha"] - 1))
        for j in range(x.shape[0]):
            gs = torch.autograd.grad(sig[j].mean(), x, retain_graph=True)[0]
            ga = torch.autograd.grad(o["alpha"][j].mean(), x, retain_graph=True)[0]
            gn_s.append(float(gs.abs().mean()))
            gn_a.append(float(ga.abs().mean()))
            sg.append(float(sig[j].mean()))
        x.grad = None
    return {"grad_sigma": float(np.median(gn_s)), "grad_alpha": float(np.median(gn_a)),
            "sigma_mean": float(np.mean(sg))}


@torch.no_grad()
def gamma_dose(model, paths):
    out = []
    for gamma in [0.5, 0.7, 1.0, 1.3, 1.6]:
        ss, aa = [], []
        for i in range(0, len(paths), 8):
            x01 = ((load_batch(paths[i:i + 8]).cuda() + 1) / 2) ** gamma
            o = model(x01 * 2 - 1)
            ss.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).mean().item())
            aa.append(o["alpha"].mean().item())
        out.append({"gamma": gamma, "sigma": float(np.mean(ss)), "alpha": float(np.mean(aa))})
        print("gamma", out[-1], flush=True)
    return out


def main():
    torch.manual_seed(42)
    model = FunieNIG().cuda().eval()
    model.load_state_dict(torch.load(f"{ROOT}/ckpt/funie_edl/best.pth", map_location="cuda", weights_only=True))
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    up = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb = up[len(up) // 2:]
    res = {"grad_euvp": grad_norms(model, [p for p, _ in euvp[:N_IMG]]),
           "grad_uieb": grad_norms(model, [p for p, _ in uieb[:N_IMG]]),
           "gamma_dose_euvp": gamma_dose(model, [p for p, _ in euvp[:100]])}
    print(json.dumps({k: v for k, v in res.items() if k.startswith("grad")}, indent=1), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("MECHANISM_PROBES_DONE")


if __name__ == "__main__":
    main()
