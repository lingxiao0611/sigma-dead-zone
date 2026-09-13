# -*- coding: utf-8 -*-
"""BEM(权重 BNN) 同协议评测：MC 采样集成 → mean=预测, std=σ。

与 eval_puie_calib.py 严格同协议：EUVP val 1000 固定顺序 cal500/test500、
像素采样 seed42、optimize_recalibration_ratio；UIEB 890 对做跨域（cal445/test445）。
区别仅在采样源：BEM 贝叶斯卷积层每次 forward 重采权重（deterministic=False），
T 次 forward → mean/std，与 PUIE 先验采样 T 次、Deep Ensemble T 模型同构。

模型/权重：basicbsr 框架 CG_UNet（Network, n_feat=40, stage=1），
ckpt = experiments/CG_UNet_EUVP/best_psnr_23.32_60000.pth（deterministic 模式 val PSNR 23.32）。
附带 sanity：deterministic 模式跑 60 对算 PSNR，对照训练日志的 23.32 防加载错。
"""
import os, sys, json, glob, re
import torch
import numpy as np
from torch.utils.data import DataLoader

BEM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code", "bem")
sys.path.insert(0, BEM_DIR)
from basicsr.archs.UMamba_arch import Network
from basicsr.bayesian.tools import convert2bnn_selective, set_prediction_type
from PIL import Image
import uncertainty_toolbox as ut

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = f"{ROOT}/code/bem/experiments/CG_UNet_EUVP/best_psnr_23.32_60000.pth"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
T_PASSES = 2 if os.environ.get("SMOKE") == "1" else 30
N_SAMPLE = 200_000 if os.environ.get("SMOKE") == "1" else 50_000
SMOKE_N = 60 if os.environ.get("SMOKE") == "1" else None
SEED = 42


class PairDS(torch.utils.data.Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1)
        return x, y


def build_model():
    model = Network(in_channels=3, out_channels=3, n_feat=40,
                    d_state=[1, 1, 1], ssm_ratio=1, mlp_ratio=4, mlp_type="gdmlp",
                    use_pixelshuffle=True, drop_path=0.0, stage=1, num_blocks=[2, 2, 2])
    # 与训练一致：只把带 bayesian=True 标记的 VSS op 内部层转贝叶斯，再加载权重
    convert2bnn_selective(model, {"sigma_init": 0.05, "decay": 0.998, "pretrain": False})
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["params"]
    model.load_state_dict(sd, strict=True)
    return model.cuda().eval()


def psnr_det(model, pairs, n=60):
    """deterministic 模式 PSNR sanity（对照训练日志 23.32）"""
    set_prediction_type(model, deterministic=True)
    loader = DataLoader(PairDS(pairs[:n]), batch_size=8, num_workers=4)
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.cuda())[-1].clamp(0, 1).cpu().numpy()
            for i in range(out.shape[0]):
                p = out[i].transpose(1, 2, 0); t = yb.numpy()[i].transpose(1, 2, 0)
                ps.append(10 * np.log10(1.0 / max(float(((p - t) ** 2).mean()), 1e-10)))
    return float(np.mean(ps))


def collect(model, pairs, T=T_PASSES):
    """MC 采样 T 次 → mean/std，[0,1] 域。每次 forward 贝叶斯权重重采样。"""
    set_prediction_type(model, deterministic=False)
    model.eval()
    loader = DataLoader(PairDS(pairs), batch_size=8, num_workers=4, pin_memory=True)
    P, S, Tg = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.cuda()
            so = sso = None
            for _ in range(T):
                o = model(xb)[-1].clamp(0, 1)
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            mean = so / T
            var = (sso / T - mean * mean).clamp_min(0)
            std = torch.sqrt(var)
            P.append(mean.cpu().numpy()); S.append(std.cpu().numpy())
            Tg.append(yb.numpy())
    P, S, Tg = tuple(np.concatenate(a).astype(np.float32) for a in (P, S, Tg))
    S = np.maximum(S, 1e-6)
    return P, S, Tg


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


def psnr_ssim_all(P, T):
    from skimage.metrics import structural_similarity as ssim_fn
    ps, ss = [], []
    for i in range(P.shape[0]):
        p = P[i].transpose(1, 2, 0); t = T[i].transpose(1, 2, 0)
        mse = float(((p - t) ** 2).mean())
        ps.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        ss.append(ssim_fn(p, t, channel_axis=2, data_range=1.0))
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    model = build_model()
    res = {"smoke": os.environ.get("SMOKE") == "1", "ckpt": os.path.basename(CKPT),
           "det_sanity_psnr60": psnr_det(model, val_pairs_euvp())}
    print(f"det sanity PSNR(60) = {res['det_sanity_psnr60']:.3f} (训练日志 23.32)", flush=True)

    # ---- EUVP val：cal500/test500 同 3b 协议 ----
    pairs = val_pairs_euvp()
    if SMOKE_N:
        pairs = pairs[:SMOKE_N]
        Pc, Sc, Tc = collect(model, pairs[: SMOKE_N // 2])
        Pt, St, Tt = collect(model, pairs[SMOKE_N // 2:])
    else:
        Pc, Sc, Tc = collect(model, pairs[:500])
        Pt, St, Tt = collect(model, pairs[500:])
    ycal, scal, tcal = flat(Pc, Sc, Tc)
    ratio = float(ut.optimize_recalibration_ratio(ycal, scal, tcal))
    ypred, ystd, ytrue = flat(Pt, St, Tt)
    p_s, s_s = psnr_ssim_all(Pt, Tt)
    res["euvp_val"] = {"ratio": ratio, "ckpt_psnr": p_s, "ckpt_ssim": s_s,
                       "test_raw": pick(ypred, ystd, ytrue),
                       "test_recal": pick(ypred, ystd * ratio, ytrue),
                       "z_raw": zcov(ypred, ystd, ytrue),
                       "z_recal": zcov(ypred, ystd * ratio, ytrue),
                       "sigma_mean": float(St.mean())}
    print(f"[EUVP] ratio={ratio:.3f} ckptPSNR={p_s:.3f} SSIM={s_s:.4f} σmean={St.mean():.4f}", flush=True)

    # ---- UIEB 跨域：cal445/test445 ----
    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    upairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    if SMOKE_N:
        upairs = upairs[:SMOKE_N]
    half = len(upairs) // 2
    Uc, Ucs, Uct = collect(model, upairs[:half])
    Ut, Ust, Utt = collect(model, upairs[half:])
    ucal, uscal, utcal = flat(Uc, Ucs, Uct)
    u_ratio = float(ut.optimize_recalibration_ratio(ucal, uscal, utcal))
    uyp, uys, utr = flat(Ut, Ust, Utt)
    u_p, u_s = psnr_ssim_all(Ut, Utt)
    res["uieb_cross"] = {"ratio_domain": u_ratio, "ckpt_psnr": u_p, "ckpt_ssim": u_s,
                         "test_raw": pick(uyp, uys, utr),
                         "z_raw": zcov(uyp, uys, utr),
                         "sigma_mean": float(Ust.mean())}
    print(f"[UIEB] ratio={u_ratio:.3f} ckptPSNR={u_p:.3f} σmean={Ust.mean():.4f}", flush=True)

    json.dump(res, open(f"{ROOT}/outputs/bem_eval.json", "w"), indent=1)
    print("BEM_EVAL_DONE", flush=True)


def val_pairs_euvp():
    vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:1000]
    def to_flat(p, sub):
        subset = p.split("/")[-3]
        base = os.path.splitext(os.path.basename(p))[0]
        return f"{ROOT}/datasets/EUVP_flat/{sub}/{subset}__{base}.jpg"
    return [(to_flat(p, "valA"), to_flat(p, "valB")) for p in vp]


if __name__ == "__main__":
    main()
