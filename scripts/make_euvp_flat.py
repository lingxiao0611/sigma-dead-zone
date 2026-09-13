# -*- coding: utf-8 -*-
"""把 EUVP_Paired（按子集分目录）拍平成 mamba_uie 能吃的扁平目录（同一份 split 划分）。

坑：mamba_uie dataset.py 的 is_image_file 只认小写 .png/.jpg/.bmp，
EUVP 文件是 .JPEG（大写）→ 符号链接名一律改成 .jpg 结尾。
配对逻辑 = 两个目录分别 sort 后按 index 配对 → A/B 用完全相同的扁平名即可。
train = 全部配对减去 split.json 的 val 1000（与 FUnIE/U-shape 同一划分）。
"""
import os, json, glob

ROOT = "/mnt/data/underwater/bpinn_uq/datasets"
PAIRED = os.path.join(ROOT, "EUVP_Dataset", "EUVP Dataset", "EUVP_Paired")
FLAT = os.path.join(ROOT, "EUVP_flat")
SPLIT = "/mnt/data/underwater/bpinn_uq/ckpt/funie_baseline/split.json"

val_set = set(json.load(open(SPLIT))["val_paths"])

for sub in ("trainA", "trainB", "valA", "valB"):
    os.makedirs(os.path.join(FLAT, sub), exist_ok=True)

n_train = n_val = 0
skipped = 0
for a in sorted(glob.glob(os.path.join(PAIRED, "*", "trainA", "*"))):
    if not os.path.isfile(a):
        continue
    subset = os.path.basename(os.path.dirname(os.path.dirname(a)))
    fname = os.path.basename(a)
    base, ext = os.path.splitext(fname)
    flat_name = f"{subset}__{base}.jpg"
    b = os.path.join(os.path.dirname(os.path.dirname(a)), "trainB", fname)
    if not os.path.isfile(b):
        skipped += 1
        continue
    target_sub = "val" if a in val_set else "train"
    la = os.path.join(FLAT, f"{target_sub}A", flat_name)
    lb = os.path.join(FLAT, f"{target_sub}B", flat_name)
    if not os.path.lexists(la):
        os.symlink(a, la)
    if not os.path.lexists(lb):
        os.symlink(b, lb)
    if target_sub == "train":
        n_train += 1
    else:
        n_val += 1

print(f"train pairs: {n_train}  val pairs: {n_val}  skipped(no B): {skipped}")
for sub in ("trainA", "trainB", "valA", "valB"):
    print(sub, len(os.listdir(os.path.join(FLAT, sub))))
print("EUVP_FLAT_DONE")
