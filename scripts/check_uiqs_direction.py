# -*- coding: utf-8 -*-
"""判定 RUIE UIQS A~E 的质量方向：用无参考图像统计（对比度/饱和度/亮度）。
退化严重的水下图典型特征：对比度低、色偏强。
"""
import os, random
import numpy as np
from PIL import Image

D = "/mnt/data/underwater/bpinn_uq/datasets/RUIE_Dataset/RUIE/UIQS"
N = 120
r = random.Random(0)

print(f"{'grade':6s} {'contrast(gray std)':>19s} {'saturation':>11s} {'brightness':>11s}")
for g in "ABCDE":
    fs = sorted([f for f in os.listdir(os.path.join(D, g))
                 if f.lower().endswith((".jpg", ".png"))])
    fs = r.sample(fs, min(N, len(fs)))
    cs, ss, bs = [], [], []
    for f in fs:
        im = Image.open(os.path.join(D, g, f)).convert("RGB").resize((256, 256))
        a = np.asarray(im, np.float32) / 255.0
        gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        cs.append(gray.std())
        mx, mn = a.max(axis=2), a.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0)
        ss.append(sat.mean())
        bs.append(gray.mean())
    print(f"{g:6s} {np.mean(cs):19.4f} {np.mean(ss):11.4f} {np.mean(bs):11.4f}")
