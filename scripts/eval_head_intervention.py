# -*- coding: utf-8 -*-
"""E1 头部干预实验：冻结 FUnIE 主干，只重训 NIG 证据头（DER 损失、超参不动）。

因果问题：跨域死区住在头的映射里（骨干特征充足、头不报），还是骨干特征本身盲？
干预：用 UIEB 校准集（445 / 100 子集）只训头 → 在 UIEB test445 测 lift 与 α/β 梯度是否恢复。
三种结局都可写：恢复=头映射问题；不恢复=骨干特征盲（更深结论）；100 张即恢复=低成本修复。
全部复用 released 配方（DERLoss/超参与 train_funie_edl.py 一致），仅冻结参数，不构成新方法。
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
from uq.nig_dist import NormalInverseGamma
from uq.der_loss import DERLoss
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = f"{ROOT}/ckpt/funie_edl/best.pth"
OUT = f"{ROOT}/outputs/head_intervention.json"
EPOCHS_FULL, EPOCHS_SUB, BS, LR, REG_W = 40, 80, 16, 1e-4, 0.01
K_PIX, SEED = 100_000, 42


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


def uieb_pairs():
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    pairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    return pairs[:len(pairs) // 2], pairs[len(pairs) // 2:]   # cal / test（与主线协议一致）


@torch.no_grad()
def collect(model, pairs):
    """每图：err_mean / sigma_mean / alpha_mean / beta_mean / contrast（固定像素采样）"""
    model.eval()
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(3 * 256 * 256, size=K_PIX, replace=False)
    rows = []
    for xb, yb, sb in loader:
        o = model(xb.cuda())
        P = ((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy()
        S = (torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy()
        A = o["alpha"].mean(dim=[1, 2, 3]).cpu().numpy()
        B = o["beta"].mean(dim=[1, 2, 3]).cpu().numpy()
        T = ((yb + 1) / 2).numpy()
        n = P.shape[0]
        for i in range(n):
            p = P[i].ravel()[idx]; t = T[i].ravel()[idx]; s = S[i].ravel()[idx]
            e = np.abs(p - t)
            rows.append({"err": float(e.mean()), "sig": float(s.mean()),
                         "alpha": float(A[i]), "beta": float(B[i]),
                         "contrast": float(sb[i, 0]), "cov1": float((e / np.clip(s, 1e-8, None) < 1).mean())})
    model.train()
    return rows


def summarize(rows):
    o = np.argsort([r["sig"] for r in rows])
    k = max(1, len(o) // 5)
    lo, hi = o[:k], o[-k:]
    err = np.array([r["err"] for r in rows])
    lift = float(err[hi].mean() / max(err[lo].mean(), 1e-8))
    contrast = np.array([r["contrast"] for r in rows])
    order = np.argsort(contrast); bins = np.array_split(order, 4)
    strata = []
    for b in bins:
        strata.append({"contrast": float(contrast[b].mean()),
                       "err": float(np.mean([rows[i]["err"] for i in b])),
                       "sigma": float(np.mean([rows[i]["sig"] for i in b])),
                       "alpha": float(np.mean([rows[i]["alpha"] for i in b])),
                       "beta": float(np.mean([rows[i]["beta"] for i in b])),
                       "cov1": float(np.mean([rows[i]["cov1"] for i in b]))})
    return {"lift": lift, "marginal_cov1": float(np.mean([r["cov1"] for r in rows])),
            "strata": strata, "n": len(rows)}


def train_head(model, pairs, epochs, tag):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.final.parameters():
        p.requires_grad = True
    head_params = [p for p in model.final.parameters() if p.requires_grad]
    opt = torch.optim.Adam(head_params, lr=LR, betas=(0.5, 0.999))
    der = DERLoss(reg_weight=REG_W)
    loader = DataLoader(PairDS(pairs), batch_size=BS, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    for ep in range(1, epochs + 1):
        run = nb = 0
        for xb, yb, _ in loader:
            opt.zero_grad()
            out = model(xb.cuda())
            dist = NormalInverseGamma(out["loc"], out["lmbda"], out["alpha"], out["beta"])
            loss = der(dist, yb.cuda())
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
        print(f"  [{tag}] ep{ep}/{epochs} der={run/max(nb,1):.4f}", flush=True)
    return model


def main():
    torch.manual_seed(42)
    cal, test = uieb_pairs()
    print(f"UIEB cal={len(cal)} test={len(test)}", flush=True)

    res = {}
    model = FunieNIG().cuda()
    model.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    res["baseline"] = summarize(collect(model, test))
    print("baseline:", {k: res["baseline"][k] for k in ("lift", "marginal_cov1")}, flush=True)

    for tag, sub, ep in [("full445", cal, EPOCHS_FULL), ("sub100", cal[:100], EPOCHS_SUB)]:
        model = FunieNIG().cuda()
        model.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
        model = train_head(model, sub, ep, tag)
        res[tag] = summarize(collect(model, test))
        torch.save(model.state_dict(), f"{ROOT}/ckpt/head_intervention_{tag}.pth")
        print(tag + ":", {k: res[tag][k] for k in ("lift", "marginal_cov1")}, flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("HEAD_INTERVENTION_DONE")


if __name__ == "__main__":
    main()
