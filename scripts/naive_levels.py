# -*- coding: ascii -*-
"""Degradation level of each water body, measured by the identity baseline.

The LSUI cross-domain MAE (0.0414) came out BELOW the in-domain EUVP MAE (0.054)
quoted in the paper, which would mean LSUI is not a harder water body for this
model. Before any of it goes into the paper we need to know how degraded each
source set actually is.

Identity baseline only: no model is loaded, no weights, no training. For each
test split we report MAE(input, reference) and PSNR(input, reference) under the
exact preprocessing the audit uses (resize 256, [0,1], no model).

Read-only.
"""
import json
import os

import numpy as np
from PIL import Image

ROOT = "/mnt/data/underwater/bpinn_uq"
SPLIT_EUVP = ROOT + "/ckpt/funie_baseline/split.json"
UIEB = ROOT + "/datasets/UIEB_Dataset/UIEB Dataset"
UIEB_RAW = UIEB + "/raw/UIEB_raw_reName"
UIEB_REF = UIEB + "/reference/UIEB_reference_reName"
LSUI_RAW = ROOT + "/datasets/LSUI/input"
LSUI_REF = ROOT + "/datasets/LSUI/GT"
LSUI_SPLIT = ROOT + "/outputs/lsui_split.json"
OUT = ROOT + "/outputs/naive_levels.json"


def load(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((256, 256)), np.float32) / 255.0
    return a


def stats(pairs, name):
    maes, psnrs = [], []
    for a, b in pairs:
        x, y = load(a), load(b)
        m = float(np.abs(x - y).mean())
        mse = float(((x - y) ** 2).mean())
        maes.append(m)
        psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    rec = {"set": name, "n": len(pairs),
           "identity_mae": round(float(np.mean(maes)), 5),
           "identity_psnr": round(float(np.mean(psnrs)), 3)}
    print("%-8s n=%3d  identity MAE=%.4f  PSNR=%.2f dB" %
          (name, len(pairs), rec["identity_mae"], rec["identity_psnr"]), flush=True)
    return rec


def main():
    res = {}

    # EUVP held-out val (500), same convention as the audit
    sp = json.load(open(SPLIT_EUVP))
    vp = [p.replace("/trainA/", "/trainB/") for p in sp["val_paths"]]
    res["euvp"] = stats(list(zip(sp["val_paths"], vp)), "EUVP")

    # UIEB, same half-split convention as eval_crossdomain_uieb.py
    raw = sorted(os.listdir(UIEB_RAW))
    ref = set(os.listdir(UIEB_REF))
    pairs = [(os.path.join(UIEB_RAW, f), os.path.join(UIEB_REF, f)) for f in raw if f in ref]
    res["uieb"] = stats(pairs[len(pairs) // 2:], "UIEB")

    # LSUI, the 445 test pairs this run actually used
    ls = json.load(open(LSUI_SPLIT))
    pairs = [(os.path.join(LSUI_RAW, f), os.path.join(LSUI_REF, f)) for f in ls["test"]]
    res["lsui"] = stats(pairs, "LSUI")

    json.dump(res, open(OUT, "w"), indent=1)
    print("WROTE " + OUT, flush=True)


if __name__ == "__main__":
    main()
