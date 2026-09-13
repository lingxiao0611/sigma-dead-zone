# -*- coding: utf-8 -*-
"""补强 B+D：分层 proxy 敏感性（对比度/饱和度/亮度三种分层依据）+ EDL 证据参数机理。
- B：三种输入侧 proxy 分别做四分位分层覆盖率（raw + recal），检验结论是否依赖 proxy 选择
- D：EDL 的 NIG 证据参数（α、β）按退化分层统计——σ 死区的机理证据
协议同 coverage_stratified：EUVP test500 + UIEB test445。
输出 outputs/strat_proxy_evidence.json
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

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
CKPT_EDL = f"{ROOT}/ckpt/funie_edl/best.pth"
SEEDS = [101, 202, 303, 404, 505]
CKPT_ENS = lambda s: f"{ROOT}/ckpt/funie_ensemble/seed_{s}/best.pth"
OUT = f"{ROOT}/outputs/strat_proxy_evidence.json"
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
        mx, mn = arr.max(axis=2), arr.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0).mean()
        x = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1
        return x, y, torch.tensor([gray.std(), sat, gray.mean()], dtype=torch.float32)


def collect_edl(model, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    P, S, T, ST, A, B = [], [], [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            o = model(xb.cuda())
            P.append(((o["loc"].clamp(-1, 1) + 1) / 2).cpu().numpy())
            S.append((torch.sqrt(o["beta"] / (o["alpha"] - 1)) / 2).cpu().numpy())
            A.append(o["alpha"].mean(dim=(1, 2, 3)).cpu().numpy())   # 每图 α 均值
            B.append(o["beta"].mean(dim=(1, 2, 3)).cpu().numpy())    # 每图 β 均值
            T.append(((yb + 1) / 2).numpy()); ST.append(sb.numpy())
    out = [np.concatenate(a).astype(np.float32) for a in (P, S, T, ST, A, B)]
    return out


def collect_ens(models, pairs):
    loader = DataLoader(PairDS(pairs), batch_size=16, num_workers=4, pin_memory=True)
    K = len(models); sl, sql, Tl, St = [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            so = sso = None
            for m in models:
                o = m(xb.cuda()).clamp(-1, 1).cpu().numpy()
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            sl.append(so); sql.append(sso); Tl.append(((yb + 1) / 2).numpy()); St.append(sb.numpy())
    S = np.concatenate(sl, 0); SS = np.concatenate(sql, 0)
    T = np.concatenate(Tl, 0); ST = np.concatenate(St, 0)
    mean = S / K
    std = np.sqrt(np.clip((SS - S * S / K) / (K - 1), 0, None))
    return ((mean + 1) / 2).astype(np.float32), (std / 2).astype(np.float32), T.astype(np.float32), ST


def per_image(P, S, T, ST, ratio=None):
    n = len(P)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P[0].size, size=min(K_PIX, P[0].size), replace=False)
    ip, is_, it = P.reshape(n, -1)[:, idx], S.reshape(n, -1)[:, idx], T.reshape(n, -1)[:, idx]
    err = np.abs(ip - it)
    sig = is_ * ratio if ratio is not None else is_
    z = err / np.clip(sig, 1e-8, None)
    return {"sigma_mean": np.asarray([s.mean() for s in is_]),
            "err_mean": np.asarray([e.mean() for e in err]),
            "cov1": np.asarray([(zz < 1).mean() for zz in z]),
            "cov2": np.asarray([(zz < 2).mean() for zz in z]),
            "cov3": np.asarray([(zz < 3).mean() for zz in z]),
            "contrast": ST[:, 0], "saturation": ST[:, 1], "brightness": ST[:, 2]}


def stratify(d, key, nb=4):
    v = np.asarray(d[key])
    order = np.argsort(v)
    bins = np.array_split(order, nb)
    rows = []
    for i, b in enumerate(bins, 1):
        rows.append({"stratum": f"S{i}", "n": int(len(b)),
                     "err_mean": float(np.asarray(d["err_mean"])[b].mean()),
                     "sigma_mean": float(np.asarray(d["sigma_mean"])[b].mean()),
                     "cov1": float(np.asarray(d["cov1"])[b].mean()),
                     "cov3": float(np.asarray(d["cov3"])[b].mean())})
    return rows


def lift(d):
    o = np.argsort(np.asarray(d["sigma_mean"]))
    k = max(1, len(o) // 5)
    return float(np.asarray(d["err_mean"])[o[-k:]].mean() /
                 max(np.asarray(d["err_mean"])[o[:k]].mean(), 1e-8))


def main():
    val_paths = json.load(open(SPLIT))["val_paths"]
    euvp_test = [(p, p.replace("/trainA/", "/trainB/")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    upairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb_test = upairs[len(upairs) // 2:]

    edl = FunieNIG().cuda().eval()
    edl.load_state_dict(torch.load(CKPT_EDL, map_location="cuda", weights_only=True))
    ens = [GeneratorFunieGAN().cuda().eval() for _ in SEEDS]
    for m, s in zip(ens, SEEDS):
        m.load_state_dict(torch.load(CKPT_ENS(s), map_location="cuda", weights_only=True))
    print("models loaded", flush=True)

    r_edl = float(json.load(open(f"{ROOT}/outputs/edl_recalibration.json"))["ratio"])
    r_ens = float(json.load(open(f"{ROOT}/outputs/ensemble_recalibration.json"))["ratio"])
    cd = json.load(open(f"{ROOT}/outputs/crossdomain_uieb.json"))
    r_edl_dom = cd.get("edl", {}).get("ratio_domain") or cd.get("edl", {}).get("ratio")
    r_ens_dom = cd.get("ensemble", {}).get("ratio_domain") or cd.get("ensemble", {}).get("ratio")

    res = {}
    for dsname, pairs, rmap in [("EUVP_域内", euvp_test, {"edl": r_edl, "ensemble": r_ens}),
                                ("UIEB_跨域", uieb_test, {"edl": r_edl_dom, "ensemble": r_ens_dom})]:
        res[dsname] = {}
        for mname, model, col in [("edl", edl, collect_edl), ("ensemble", ens, collect_ens)]:
            if mname == "edl":
                P, S, T, ST, Aimg, Bimg = col(model, pairs)
            else:
                P, S, T, ST = col(model, pairs)
                Aimg = Bimg = None
            d_raw = per_image(P, S, T, ST)
            d_rec = per_image(P, S, T, ST, ratio=rmap.get(mname))
            entry = {"ratio_used": rmap.get(mname), "lift": lift(d_raw),
                     "strata_by_proxy": {}}
            for key in ["contrast", "saturation", "brightness"]:
                sr = stratify(d_raw, key)
                sc = stratify(d_rec, key)
                entry["strata_by_proxy"][key] = {"raw": sr, "recal": sc,
                                                 "raw_range": round(sr[-1]["cov1"] - sr[0]["cov1"], 4),
                                                 "recal_range": round(sc[-1]["cov1"] - sc[0]["cov1"], 4)}
            if Aimg is not None:
                # D：EDL 证据参数按对比度分层（机理证据）
                v = d_raw["contrast"]; order = np.argsort(v)
                bins = np.array_split(order, 4)
                ev = []
                for i, b in enumerate(bins, 1):
                    ev.append({"stratum": f"S{i}", "n": int(len(b)),
                               "alpha_mean": float(Aimg[b].mean()),
                               "beta_mean": float(Bimg[b].mean()),
                               "sigma_mean": float(np.asarray(d_raw["sigma_mean"])[b].mean()),
                               "err_mean": float(np.asarray(d_raw["err_mean"])[b].mean())})
                entry["edl_evidence_params"] = ev
            res[dsname][mname] = entry
            pr = entry["strata_by_proxy"]
            print(f"[{dsname}/{mname}] lift={entry['lift']:.2f} | 分层cov1极差(重校前→后) "
                  f"contrast={pr['contrast']['raw_range']}→{pr['contrast']['recal_range']} "
                  f"sat={pr['saturation']['raw_range']}→{pr['saturation']['recal_range']} "
                  f"bright={pr['brightness']['raw_range']}→{pr['brightness']['recal_range']}", flush=True)
            if Aimg is not None:
                for e in entry["edl_evidence_params"]:
                    print(f"   {e['stratum']}: alpha={e['alpha_mean']:.2f} beta={e['beta_mean']:.4f} "
                          f"sigma={e['sigma_mean']:.4f} err={e['err_mean']:.4f}", flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("STRAT_PROXY_EVIDENCE_DONE")


if __name__ == "__main__":
    main()
