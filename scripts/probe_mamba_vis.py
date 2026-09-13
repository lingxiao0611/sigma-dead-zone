# -*- coding: utf-8 -*-
"""Mamba-UIE probe 可视化：用 weights_euvp（bf16 坏 ckpt）跑 3 张 val 图，
存 input | output | target 横向拼图到 outputs/mamba_probe/，供肉眼判定 checkpoint 好坏。"""
import os, sys, json
import torch
import numpy as np
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/mamba_uie")
from net.net import net as MambaNet
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
OUT = f"{ROOT}/outputs/mamba_probe"
os.makedirs(OUT, exist_ok=True)

ck = f"{ROOT}/code/mamba_uie/weights_euvp/epoch_30.pth"
print(f"probe checkpoint: {ck}", flush=True)
model = MambaNet().cuda().eval()
model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))

vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:3]

@torch.no_grad()
def main():
    for i, p in enumerate(vp):
        a = p.replace("/trainB/", "/trainA/") if "/trainB/" in p else p
        b = p.replace("/trainA/", "/trainB/")
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1)[None].cuda()
        y = np.asarray(B, np.float32) / 255
        j, t, tb = model(x)
        o = j[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        mse = float(((o - y) ** 2).mean())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        gap = np.ones((256, 3, 3), np.float32)
        grid = np.concatenate([np.asarray(A, np.float32) / 255, gap, o, gap, y], axis=1)
        Image.fromarray((grid * 255).astype("uint8")).save(f"{OUT}/probe_{i}_psnr{psnr:.1f}dB.png")
        print(f"[{i}] psnr={psnr:.2f} dB  out_min={o.min():.3f} out_max={o.max():.3f}", flush=True)
    print("PROBE_DONE", flush=True)

main()
