# -*- coding: utf-8 -*-
"""对拍：同一批图、同一批采样，分别用 eval_puie_calib 的池化口径与
eval_puie_strat 的逐图口径算 1σ 覆盖，定位 11% vs 26% 分歧来源。"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, "/mnt/data/underwater/bpinn_uq/code/puie_net")
from PUIENet_MC import mynet
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT = f"{ROOT}/ckpt/funie_baseline/split.json"


class Args:
    device = "cuda:0"
    batchSize = 1


class DS(torch.utils.data.Dataset):
    def __init__(self, pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        A = Image.open(a).convert("RGB").resize((256, 256))
        B = Image.open(b).convert("RGB").resize((256, 256))
        x = torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1)   # [0,1]
        y = torch.from_numpy(np.asarray(B, np.float32) / 255).permute(2, 0, 1)
        return x, y


def main():
    vp = json.load(open(SPLIT))["val_paths"][500:530]   # 30 张 test 图
    def to_flat(p, sub):
        subset = p.split("/")[-3]
        base = os.path.splitext(os.path.basename(p))[0]
        return f"{ROOT}/datasets/EUVP_flat/{sub}/{subset}__{base}.jpg"
    pairs = [(to_flat(p, "valA"), to_flat(p, "valB")) for p in vp]

    model = mynet(Args).cuda().eval()
    ck = f"{ROOT}/code/puie_net/weights_euvp/epoch_29.pth"
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=True))

    loader = torch.utils.data.DataLoader(DS(pairs), batch_size=8)
    P, S, T = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            model.forward(xb.cuda(), None, training=False)
            so = sso = None
            for _ in range(30):
                o = model.sample(testing=True).clamp(0, 1)
                so = o if so is None else so + o
                sso = o * o if sso is None else sso + o * o
            mean = so / 30
            std = torch.sqrt((sso / 30 - mean * mean).clamp_min(0))
            P.append(mean.cpu().numpy()); S.append(std.cpu().numpy()); T.append(yb.numpy())
    P, S, T = np.concatenate(P).astype(np.float32), np.concatenate(S).astype(np.float32), np.concatenate(T).astype(np.float32)
    S = np.maximum(S, 1e-6)
    print(f"collected {P.shape[0]} imgs, sigma_mean={S.mean():.5f}, err_mean={np.abs(P-T).mean():.5f}", flush=True)

    # 口径1：池化（eval_puie_calib 的 flat 思路，全量不采样）
    z = np.abs(P - T) / S
    print(f"口径1 池化 cov1 = {(z < 1).mean():.4f}", flush=True)

    # 口径2：逐图均值（eval_puie_strat 的思路）
    n = len(P)
    zr = z.reshape(n, -1)
    cov_img = (zr < 1).mean(axis=1)
    print(f"口径2 逐图均值 cov1 = {cov_img.mean():.4f} (std {cov_img.std():.4f})", flush=True)

    # 口径3：池化但只用 5 万随机点（复现 eval_puie_calib 的采样量）
    rng = np.random.default_rng(42)
    idx = rng.choice(z.size, size=50_000, replace=False)
    print(f"口径3 池化5万点 cov1 = {(z.flatten()[idx] < 1).mean():.4f}", flush=True)
    print("RECON_DONE", flush=True)


main()
