# -*- coding: utf-8 -*-
"""Mamba-UIE（确定性物理基线）评测：EUVP val 1000 + UIEB 890 的 PSNR/SSIM/MAE。
确定性方法无 σ 档，只做复原精度对照。自动选 weights_euvp/ 最新 checkpoint。
输出 j_out（增强输出，监督信号 label 对齐的就是它）。"""
import os, sys, json, glob, re
import torch
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/mamba_uie")
from net.net import net as MambaNet
from A import get_A
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
W_DIR = f"{ROOT}/code/mamba_uie/weights_euvp"
UIEB = f"{ROOT}/datasets/UIEB_Dataset/UIEB Dataset"
RAW = os.path.join(UIEB, "raw/UIEB_raw_reName")
REF = os.path.join(UIEB, "reference/UIEB_reference_reName")


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
    return max(cks, key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))


@torch.no_grad()
def psnr_ssim(model, pairs):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        ssim_fn = None
    loader = DataLoader(PairDS(pairs), batch_size=8, num_workers=4, pin_memory=True)
    ps, ss, maes = [], [], []
    for xb, yb in loader:
        out = model(xb.cuda())          # (j, t, tb)
        j_out = out[0].clamp(0, 1).cpu().numpy()
        tgt = yb.numpy()
        for i in range(j_out.shape[0]):
            p = j_out[i].transpose(1, 2, 0); t = tgt[i].transpose(1, 2, 0)
            mse = float(((p - t) ** 2).mean())
            ps.append(10 * np.log10(1.0 / max(mse, 1e-10)))
            maes.append(float(np.abs(p - t).mean()))
            if ssim_fn is not None:
                ss.append(ssim_fn(p, t, channel_axis=2, data_range=1.0))
    return {"psnr": float(np.mean(ps)), "mae": float(np.mean(maes)),
            "ssim": float(np.mean(ss)) if ss else None, "n": len(ps)}


def main():
    ck = latest_ckpt()
    print(f"使用 checkpoint: {ck}", flush=True)
    model = MambaNet().cuda().eval()
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))

    res = {"ckpt": os.path.basename(ck)}
    vp = json.load(open(f"{ROOT}/ckpt/funie_baseline/split.json"))["val_paths"][:1000]
    pairs = [(p, p.replace("/trainA/", "/trainB/")) for p in vp]
    res["euvp_val"] = psnr_ssim(model, pairs)
    print(f"[EUVP val] {res['euvp_val']}", flush=True)

    raw = sorted(os.listdir(RAW)); ref = set(os.listdir(REF))
    upairs = [(os.path.join(RAW, f), os.path.join(REF, f)) for f in raw if f in ref]
    res["uieb_cross"] = psnr_ssim(model, upairs)
    print(f"[UIEB cross] {res['uieb_cross']}", flush=True)

    json.dump(res, open(f"{ROOT}/outputs/mambauie_eval.json", "w"), indent=1)
    print("MAMBAMUIE_EVAL_DONE")


if __name__ == "__main__":
    main()
