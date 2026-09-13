# -*- coding: utf-8 -*-
"""预算不敏感性消融（PUIE-Net 部分）：用不同训练进度的 checkpoint 跑同一套
EUVP 校准评测，验证 UQ 行为（欠自信 ratio≈7、覆盖率形态）不随训练预算剧变。
checkpoint 取 epoch 1 / 9 / 19 / 29（snapshots=2 只存奇数）。
产物 outputs/puie_ablation.json。"""
import os, sys, json
import torch
import numpy as np
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code")
from eval_puie_calib import (build_model, collect, flat, zcov, pick,
                             psnr_ssim_all, W_DIR, ROOT)
import uncertainty_toolbox as ut

CKPTS = [1, 9, 19, 29]
N_SAMPLE = 100_000


def flatN(P, S, T):
    rng = np.random.default_rng(42)
    idx = rng.choice(P.size, size=min(N_SAMPLE, P.size), replace=False)
    return P.flatten()[idx], S.flatten()[idx], T.flatten()[idx]


def main():
    vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:1000]

    def to_flat(p, sub):
        subset = p.split("/")[-3]
        base = os.path.splitext(os.path.basename(p))[0]
        return f"{ROOT}/datasets/EUVP_flat/{sub}/{subset}__{base}.jpg"

    pairs = [(to_flat(p, "valA"), to_flat(p, "valB")) for p in vp]
    res = {}
    for ep in CKPTS:
        ck = f"{W_DIR}/epoch_{ep}.pth"
        if not os.path.exists(ck):
            print(f"跳过 epoch_{ep}（不存在）", flush=True)
            continue
        model = build_model(ck)
        Pc, Sc, Tc = collect(model, pairs[:500])
        Pt, St, Tt = collect(model, pairs[500:])
        ycal, scal, tcal = flatN(Pc, Sc, Tc)
        ratio = float(ut.optimize_recalibration_ratio(ycal, scal, tcal))
        ypred, ystd, ytrue = flatN(Pt, St, Tt)
        p_s, s_s = psnr_ssim_all(Pt, Tt)
        res[f"epoch_{ep}"] = {"ckpt_psnr": p_s, "ckpt_ssim": s_s,
                              "sigma_mean": float(St.mean()), "ratio": ratio,
                              "test_raw": pick(ypred, ystd, ytrue),
                              "z_raw": zcov(ypred, ystd, ytrue)}
        print(f"[ep{ep}] PSNR={p_s:.3f} SSIM={s_s:.4f} σmean={St.mean():.4f} "
              f"ratio={ratio:.3f} 1σ={res[f'epoch_{ep}']['z_raw']['1sigma']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(f"{ROOT}/outputs", exist_ok=True)
    json.dump(res, open(f"{ROOT}/outputs/puie_ablation.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("PUIE_ABLATION_DONE")


if __name__ == "__main__":
    main()
