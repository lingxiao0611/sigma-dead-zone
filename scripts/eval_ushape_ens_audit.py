# -*- coding: utf-8 -*-
"""E4 收尾：ENS@U-shape 审计评测（自包含，脱离训练会话独立运行）。

- 自动 glob ckpt/ushape_ens/seed_*/best.pth 作为集成成员（训练链跑完 5 个成员后应有 5 个）
- 对每个有 GT 的数据集：5 成员前向 → mean=集成预测, std=σ
- 指标：集成 PSNR/SSIM、calibration ratio、1σ 边际覆盖率(raw/recal)、分层覆盖率(按输入对比度)、风险-覆盖率/AURC
- 无 GT 数据集(RUIE/U45) 仅做 σ 对退化代理的响应(Spearman)，失败不影响主结果
输出：outputs/ushape_ens_audit_v2.json + 控制台摘要
"""
import os, sys, json, glob, time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

ROOT = "/mnt/datadisk/underwater/bpinn_uq"
sys.path.insert(0, f"{ROOT}/code/u_shape_transformer")
sys.path.insert(0, f"{ROOT}/code")
from net.Ushape_Trans import Generator  # noqa

CKPT_ROOT = f"{ROOT}/ckpt/ushape_ens"
# 注意：训练链用 CUDA_VISIBLE_DEVICES=1 跑，评测用 GPU0 避免与 monodlgd 冲突
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9


def to01(t):
    return (t.clamp(-1, 1) + 1) / 2


class UshapeDet(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = Generator()

    def forward(self, x):
        return torch.tanh(self.g(x)[-1])


def load_members():
    paths = sorted(glob.glob(f"{CKPT_ROOT}/seed_*/best.pth"))
    assert paths, f"no member ckpt in {CKPT_ROOT}"
    models = []
    for p in paths:
        m = UshapeDet().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"[ens] loaded {len(models)} members: {[os.path.basename(os.path.dirname(x)) for x in paths]}",
          flush=True)
    return models


@torch.no_grad()
def ensemble_forward(models, loader):
    preds, gts, ins = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        stack = torch.stack([to01(m(xb)) for m in models], dim=1)  # (B,5,3,H,W)
        preds.append(stack.cpu().numpy())
        gts.append(to01(yb).numpy())
        # 输入对比度代理：灰度 luminance std（逐图）
        gray = xb.mean(dim=1, keepdim=True)
        ins.append(gray.std(dim=(2, 3)).cpu().numpy()[:, 0])
    preds = np.concatenate(preds, 0)   # (N,5,3,H,W)
    gts = np.concatenate(gts, 0)       # (N,3,H,W)
    ins = np.concatenate(ins, 0)        # (N,)
    mean = preds.mean(axis=1)          # (N,3,H,W)
    std = preds.std(axis=1, ddof=0)    # (N,3,H,W) 集成 σ
    return mean, std, gts, ins


def psnr_ssim(mean, gt):
    mse = ((mean - gt) ** 2).mean(axis=(1, 2, 3))
    psnr = (10 * np.log10(1.0 / np.clip(mse, 1e-10, None))).mean()
    return psnr


def calibration(mean, std, gt):
    err = np.abs(mean - gt)  # (N,3,H,W)
    e2 = (err ** 2).mean()
    s2 = (std ** 2).mean()
    ratio = float(np.sqrt(e2 / s2))
    cov_raw = float((err <= std).mean())
    cov_recal = float((err <= std * ratio).mean())
    rms_err = float(np.sqrt(e2))
    rms_std_raw = float(np.sqrt(s2))
    rms_std_recal = rms_std_raw * ratio
    return dict(ratio=ratio, cov1_raw=cov_raw, cov1_recal=cov_recal,
                rms_err=rms_err, rms_std_raw=rms_std_raw, rms_std_recal=rms_std_recal,
                n_pixels=int(err.size))


def stratified(mean, std, gt, ins, nbins=4):
    err = np.abs(mean - gt)
    order = np.argsort(ins)
    edges = np.linspace(0, len(ins), nbins + 1, dtype=int)
    rows = []
    for b in range(nbins):
        idx = order[edges[b]:edges[b + 1]]
        if len(idx) == 0:
            continue
        e = err[idx]; s = std[idx]
        rows.append(dict(contrast_bin=b,
                         contrast_range=[float(ins[idx].min()), float(ins[idx].max())],
                         cov1_raw=float((e <= s).mean()),
                         cov1_recal=float((e <= s * CAL_RATIO).mean()),
                         n=int(e.size)))
    return rows


def aurc(mean, std, gt):
    err = np.abs(mean - gt).reshape(-1)
    s = std.reshape(-1)
    n = len(err)
    order = np.argsort(s)  # TU 语义：最自信优先保留（conf=-σ 降序）
    risk = err[order]
    cum = np.cumsum(risk) / np.arange(1, n + 1)
    aurc = float(cum.mean())
    aurc_rand = float(err.mean())
    # 80% 覆盖率时的选择性风险
    k = int(0.8 * n)
    risk_at_80 = float(cum[k - 1]) if k > 0 else float(cum[-1])
    return dict(aurc=aurc, aurc_random=aurc_rand,
                risk_at_80cov=risk_at_80,
                rejection_gain_at_80=float(aurc_rand - risk_at_80))


# ---------------- 数据集 ----------------
SPLIT_JSON = f"{ROOT}/ckpt/funie_baseline/split.json"


def _euvp_test_basenames():
    """协议 test500：split.json val_paths 的后 500 个（与 eval_rejection 同口径）。
    flat 命名 = {类别}__{文件名}，如 underwater_dark__271282_00016405.jpg。"""
    vp = json.load(open(SPLIT_JSON))["val_paths"]
    names = set()
    for p in vp[500:]:
        rel = p.split("EUVP_Paired/")[-1]          # underwater_dark/trainA/xxx.jpg
        parts = rel.split("/")
        names.add(f"{parts[0]}__{parts[-1]}")
    return names


class EUVPVal(Dataset):
    def __init__(self, limit=LIMIT):
        va = f"{ROOT}/datasets/EUVP_flat/valA"
        vb = f"{ROOT}/datasets/EUVP_flat/valB"
        test = _euvp_test_basenames()
        fs = sorted(f for f in os.listdir(va) if f in test)[:limit]
        self.pairs = [(os.path.join(va, f), os.path.join(vb, f)) for f in fs
                      if os.path.exists(os.path.join(vb, f))]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB"); B = Image.open(b).convert("RGB")
        if A.size != (256, 256): A = A.resize((256, 256))
        if B.size != (256, 256): B = B.resize((256, 256))
        return (torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1,
                torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1)


class UIEBPaired(Dataset):
    def __init__(self, limit=LIMIT):
        raw = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset/raw/UIEB_raw_reName"
        ref = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset/reference/UIEB_reference_reName"
        fs = sorted([f for f in os.listdir(raw) if f.lower().endswith(('.jpg', '.png'))])
        pairs = [(os.path.join(raw, f), os.path.join(ref, f)) for f in fs
                 if os.path.exists(os.path.join(ref, f))]
        self.pairs = pairs[len(pairs) // 2:][:limit]   # 协议 test445：后一半（与 Stage4 同口径）

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB"); B = Image.open(b).convert("RGB")
        if A.size != (256, 256): A = A.resize((256, 256))
        if B.size != (256, 256): B = B.resize((256, 256))
        return (torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1,
                torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1) * 2 - 1)


def reffree_response(models):
    """无 GT：σ 是否随输入退化严重度单调（用输入对比度作退化代理）。
    返回 Spearman(σ_mean, contrast) 近似（按 bin 均值的相关）。"""
    try:
        d = f"{ROOT}/datasets/RUIE_Dataset/RUIE/UIQS"
        if not os.path.isdir(d):
            return None
        fs = sorted(glob.glob(f"{d}/*/*.jpg"))[:300]
        if not fs:
            return None
        dl = DataLoader(ImgOnly(fs), batch_size=8, num_workers=2)
        sigs, cons = [], []
        for xb in dl:
            xb = xb.to(DEVICE)
            stack = torch.stack([to01(m(xb)) for m in models], dim=1)
            s = stack.std(dim=1, ddof=0).mean(dim=(1, 2, 3)).cpu().numpy()
            g = xb.mean(dim=1, keepdim=True)
            c = g.std(dim=(2, 3))[:, 0, 0].cpu().numpy()
            sigs.append(s); cons.append(c)
        sigs = np.concatenate(sigs); cons = np.concatenate(cons)
        # Spearman ≈ Pearson on ranks
        from scipy.stats import spearmanr
        rho, _ = spearmanr(sigs, cons)
        return float(rho)
    except Exception as e:
        return f"skip:{e}"


class ImgOnly(Dataset):
    def __init__(self, fs):
        self.fs = fs

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, i):
        A = Image.open(self.fs[i]).convert("RGB")
        if A.size != (256, 256): A = A.resize((256, 256))
        return torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1


def run_dataset(name, ds, models):
    global CAL_RATIO
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    mean, std, gt, ins = ensemble_forward(models, loader)
    cal = calibration(mean, std, gt)
    CAL_RATIO = cal["ratio"]
    strat = stratified(mean, std, gt, ins)
    rc = aurc(mean, std, gt)
    ps = psnr_ssim(mean, gt)
    out = dict(dataset=name, psnr=float(ps), **cal, stratified=strat, risk_coverage=rc)
    print(f"[ens:{name}] PSNR={ps:.2f} ratio={cal['ratio']:.3f} "
          f"cov1 raw={cal['cov1_raw']*100:.1f}% recal={cal['cov1_recal']*100:.1f}% "
          f"AURC={rc['aurc']:.4f} (rand {rc['aurc_random']:.4f})", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    models = load_members()
    results = {"method": "ENS@U-shape", "n_members": len(models),
               "device": DEVICE, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    results["EUVP"] = run_dataset("EUVP", EUVPVal(), models)
    results["UIEB"] = run_dataset("UIEB", UIEBPaired(), models)
    try:
        results["reffree_spearman_sigma_vs_contrast"] = reffree_response(models)
    except Exception as e:
        results["reffree_spearman_sigma_vs_contrast"] = f"skip:{e}"
    os.makedirs(f"{ROOT}/outputs", exist_ok=True)
    with open(f"{ROOT}/outputs/ushape_ens_audit_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ens] DONE in {time.time()-t0:.0f}s -> outputs/ushape_ens_audit_v2.json", flush=True)
