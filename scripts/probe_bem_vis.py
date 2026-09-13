# -*- coding: utf-8 -*-
"""BEM probe 可视化：det 模式跑 3 张 val 图存 input|output|target 拼图 + 逐图 PSNR，
对照训练日志 val PSNR 23.32，判断是加载问题还是协议差异。"""
import os, sys, json
import torch
import numpy as np
sys.path.insert(0, "/mnt/datadisk/underwater/bpinn_uq/code/bem")
from basicsr.archs.UMamba_arch import Network
from basicsr.bayesian.tools import convert2bnn_selective, set_prediction_type
from PIL import Image

ROOT = "/mnt/datadisk/underwater/bpinn_uq"
CKPT = f"{ROOT}/code/bem/experiments/CG_UNet_EUVP/best_psnr_23.32_60000.pth"
OUT = f"{ROOT}/outputs/bem_probe"
os.makedirs(OUT, exist_ok=True)

model = Network(in_channels=3, out_channels=3, n_feat=40,
                d_state=[1, 1, 1], ssm_ratio=1, mlp_ratio=4, mlp_type="gdmlp",
                use_pixelshuffle=True, drop_path=0.0, stage=1, num_blocks=[2, 2, 2])
convert2bnn_selective(model, {"sigma_init": 0.05, "decay": 0.998, "pretrain": False})
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True)["params"], strict=True)
model = model.cuda().eval()
set_prediction_type(model, deterministic=True)

vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:3]

@torch.no_grad()
def main():
    for i, p in enumerate(vp):
        a = f"{ROOT}/datasets/EUVP_flat/valA/" + p.split("/")[-3] + "__" + os.path.splitext(os.path.basename(p))[0] + ".jpg"
        b = a.replace("/valA/", "/valB/")
        A = np.asarray(Image.open(a).convert("RGB").resize((256, 256)), np.float32) / 255
        B = np.asarray(Image.open(b).convert("RGB").resize((256, 256)), np.float32) / 255
        x = torch.from_numpy(A).permute(2, 0, 1)[None].cuda()
        out = model(x)[-1].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        mse = float(((out - B) ** 2).mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        gap = np.ones((256, 3, 3), np.float32)
        grid = np.concatenate([A, gap, out, gap, B], axis=1)
        Image.fromarray((grid * 255).astype("uint8")).save(f"{OUT}/bem_det_{i}_psnr{psnr:.1f}dB.png")
        print(f"[{i}] det psnr={psnr:.2f}", flush=True)
    print("BEM_PROBE_DONE", flush=True)

main()
