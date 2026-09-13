# -*- coding: utf-8 -*-
"""PUIE-Net(CVAE) 同协议评测：先验采样集成 → mean=预测, std/2=σ。

PUIENet_MC.Decoder 的 testing 路径(training=False)用先验分布 rsample，天然随机、
不需要 GT —— 与 Deep Ensemble 协议同构：T 次 sample → mean/std。
协议与 3b 严格一致：EUVP val 1000 固定顺序 cal500/test500、200k 像素采样 seed42、
optimize_recalibration_ratio；UIEB 890 对做跨域（cal445/test445）。
自动选 weights_euvp/ 下 epoch 编号最大的 checkpoint。
"""
import os, sys, json, glob, re
import torch
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/puie_net")
from PUIENet_MC import mynet
from PIL import Image
import uncertainty_toolbox as ut

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_DIR = f"{ROOT}/code/puie_net/weights_euvp"
EUVP = f"{ROOT}/datasets/EUVP_flat"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")
T_PASSES = 2 if os.environ.get("SMOKE") == "1" else 30
N_SAMPLE = 200_000 if os.environ.get("SMOKE") == "1" else 50_000
SMOKE_N = 60 if os.environ.get("SMOKE") == "1" else None
SEED = 42


class Args:
    device = "cuda:0"
    batchSize = 1


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


def latest_ckpt():
    cks = glob.glob(f"{W_DIR}/epoch_*.pth")
    best = max(cks, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    print(f"使用 checkpoint: {best}", flush=True)
    return best


def build_model(ck):
    model = mynet(Args).cuda().eval()
    sd = torch.load(ck, map_location="cuda", weights_only=True)
    model.load_state_dict(sd)
    return model


def collect(model, pairs, T=T_PASSES):
    """先验采样 T 次 → (N,3,256,256) 的 P/S/T，[0,1] 域"""
    loader = DataLoader(PairDS(pairs), batch_size=8, num_workers=4, pin_memory=True)
    P, S, Tg = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.cuda()
            # forward(training=False) 只存 Input；sample(testing=True) 用先验 rsample
            model.forward(xb, None, training=False)
            so, sso = None, None
            for _ in range(T):
                o = model.sample(testing=True).clamp(0, 1)
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            mean = so / T
            var = (sso / T - mean * mean).clamp_min(0)   # 修正: sso 是平方和, 必须除以 T
            std = torch.sqrt(var)
            P.append(mean.cpu().numpy()); S.append(std.cpu().numpy())
            Tg.append(yb.numpy())
    P, S, Tg = tuple(np.concatenate(a).astype(np.float32) for a in (P, S, Tg))
    S = np.maximum(S, 1e-6)   # CVAE 采样可能零方差，toolbox 需要正值
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
    model = build_model(latest_ckpt())
    res = {"smoke": os.environ.get("SMOKE") == "1"}

    # ---- EUVP val：cal500/test500 同 3b 协议（4090 用扁平路径映射） ----
    vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:1000]

    def to_flat(p, sub):
        subset = p.split("/")[-3]
        base = os.path.splitext(os.path.basename(p))[0]
        return f"{ROOT}/datasets/EUVP_flat/{sub}/{subset}__{base}.jpg"

    pairs = [(to_flat(p, "valA"), to_flat(p, "valB")) for p in vp]
    if SMOKE_N:
        pairs = pairs[:SMOKE_N]
        Pc, Sc, Tc = collect(model, pairs[: SMOKE_N // 2])
        Pt, St, Tt = collect(model, pairs[SMOKE_N // 2 :])
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

    json.dump(res, open(f"{ROOT}/outputs/puie_eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("PUIE_EVAL_DONE")


if __name__ == "__main__":
    main()
