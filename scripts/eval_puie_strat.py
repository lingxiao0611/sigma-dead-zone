# -*- coding: utf-8 -*-
"""补强 F：PUIE-Net(CVAE) 跨域分层覆盖率 + lift（范式对照，喂给论文 5.5 的范式槽位）。
模型加载与采样协议同 eval_puie_calib.py（先验采样 T=30 → mean/std）；
分层与 lift 协议同 eval_coverage_stratified.py（输入对比度四分位）。
EUVP test500 + UIEB test445，ratio 取 outputs/puie_eval.json。
输出 outputs/puie_stratified.json
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/puie_net")
from PUIENet_MC import mynet
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
W_DIR = f"{ROOT}/code/puie_net/weights_euvp"
OUT = f"{ROOT}/outputs/puie_stratified.json"
T_PASSES = 30
K_PIX, SEED = 100_000, 42


class Args:
    device = "cuda:0"
    batchSize = 1


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
        x = torch.from_numpy(arr).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1)
        return x, y, torch.tensor([gray.std(), sat, gray.mean()], dtype=torch.float32)


def latest_ckpt():
    import glob, re
    cks = glob.glob(f"{W_DIR}/epoch_*.pth")
    return max(cks, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))


def collect(model, pairs, T=T_PASSES):
    loader = DataLoader(PairDS(pairs), batch_size=8, num_workers=4, pin_memory=True)
    P, S, Tg, ST = [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.cuda()
            model.forward(xb, None, training=False)
            so = sso = None
            for _ in range(T):
                o = model.sample(testing=True).clamp(0, 1)
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            mean = so / T
            std = torch.sqrt((sso / T - mean * mean).clamp_min(0))
            P.append(mean.cpu().numpy())
            S.append(std.cpu().numpy())
            Tg.append(yb.numpy()); ST.append(sb.numpy())
    return tuple(np.concatenate(a).astype(np.float32) for a in (P, S, Tg, ST))


def per_image(P, S, T, ST, ratio=None):
    n = len(P)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(P[0].size, size=min(K_PIX, P[0].size), replace=False)
    ip, is_, it = P.reshape(n, -1)[:, idx], S.reshape(n, -1)[:, idx], T.reshape(n, -1)[:, idx]
    err = np.abs(ip - it)
    sig = np.maximum(is_ * ratio if ratio is not None else is_, 1e-6)
    z = err / sig
    return {"sigma_mean": np.asarray([s.mean() for s in is_]),
            "err_mean": np.asarray([e.mean() for e in err]),
            "cov1": np.asarray([(zz < 1).mean() for zz in z]),
            "cov3": np.asarray([(zz < 3).mean() for zz in z]),
            "contrast": ST[:, 0], "saturation": ST[:, 1], "brightness": ST[:, 2]}


def stratify(d, key="contrast", nb=4):
    v = np.asarray(d[key]); order = np.argsort(v)
    bins = np.array_split(order, nb)
    return [{"stratum": f"S{i}", "n": int(len(b)),
             "err_mean": float(np.asarray(d["err_mean"])[b].mean()),
             "sigma_mean": float(np.asarray(d["sigma_mean"])[b].mean()),
             "cov1": float(np.asarray(d["cov1"])[b].mean()),
             "cov3": float(np.asarray(d["cov3"])[b].mean())}
            for i, b in enumerate(bins, 1)]


def lift(d):
    o = np.argsort(np.asarray(d["sigma_mean"])); k = max(1, len(o) // 5)
    return float(np.asarray(d["err_mean"])[o[-k:]].mean() /
                 max(np.asarray(d["err_mean"])[o[:k]].mean(), 1e-8))


def main():
    val_paths = json.load(open(SPLIT))["val_paths"]

    # 4090 上没有 EUVP 原始目录，用扁平路径映射（同 eval_puie_calib.py）
    def to_flat(p, sub):
        subset = p.split("/")[-3]
        base = os.path.splitext(os.path.basename(p))[0]
        return f"{ROOT}/datasets/EUVP_flat/{sub}/{subset}__{base}.jpg"

    euvp_test = [(to_flat(p, "valA"), to_flat(p, "valB")) for p in val_paths[500:]]
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    upairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    uieb_test = upairs[len(upairs) // 2:]

    model = mynet(Args).cuda().eval()
    ck = latest_ckpt()
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))
    print("puie loaded:", ck, flush=True)

    pj = json.load(open(f"{ROOT}/outputs/puie_eval.json"))
    r_eu = pj["euvp_val"]["ratio"]; r_ui = pj["uieb_cross"]["ratio_domain"]

    res = {"ckpt": os.path.basename(ck)}
    for dsname, pairs, r in [("EUVP_域内", euvp_test, r_eu), ("UIEB_跨域", uieb_test, r_ui)]:
        P, S, T, ST = collect(model, pairs)
        d_raw = per_image(P, S, T, ST)
        d_rec = per_image(P, S, T, ST, ratio=r)
        res[dsname] = {"ratio_used": r, "lift": lift(d_raw),
                       "marginal_cov1": float(d_raw["cov1"].mean()),
                       "marginal_cov1_recal": float(d_rec["cov1"].mean()),
                       "strata_raw": stratify(d_raw), "strata_recal": stratify(d_rec)}
        s = res[dsname]
        print(f"[{dsname}] ratio={r:.3f} 边际1σ={s['marginal_cov1']:.3f}→{s['marginal_cov1_recal']:.3f} "
              f"lift={s['lift']:.2f} 分层raw=" + "/".join(f"{t['cov1']:.3f}" for t in s["strata_raw"]) +
              " recal=" + "/".join(f"{t['cov1']:.3f}" for t in s["strata_recal"]), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("PUIE_STRAT_DONE")


if __name__ == "__main__":
    main()
