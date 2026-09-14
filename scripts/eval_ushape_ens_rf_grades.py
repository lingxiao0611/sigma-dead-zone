# -*- coding: utf-8 -*-
"""Put U-shape ensemble on the same footing as the other three Table VI columns.

Table VI's last row reports rho(sigma, severity) over the five RUIE-UIQS grades.
Three of the four cells come from eval_reffree_response.py, which averages sigma
per grade (200 sorted images each) and correlates those five grade means with the
severity rank. The fourth cell (+0.14) came from eval_ushape_ens_audit.py, which
does something else entirely: a pooled per-image correlation between sigma and a
contrast proxy, on a flat 300-image draw, with an unweighted channel mean instead
of the luminance-weighted one, for a different model. Four differences, so the
cell is not comparable with its neighbours.

This script recomputes the U-shape ensemble cell with eval_reffree_response.py's
protocol verbatim -- same per-grade sampling, same preprocessing, same luminance
proxy -- and reports both the comparable across-grade rho and the old pooled
number so the change is auditable.

    python code/eval_ushape_ens_rf_grades.py            # full, 200 per grade
    N_PER_CAT=10 python code/eval_ushape_ens_rf_grades.py   # smoke test

Writes outputs/ushape_ens_reffree_grades.json.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = "/mnt/datadisk/underwater/bpinn_uq"
sys.path.insert(0, f"{ROOT}/code/u_shape_transformer")
sys.path.insert(0, f"{ROOT}/code")
from net.Ushape_Trans import Generator  # noqa: E402

CKPT_ROOT = f"{ROOT}/ckpt/ushape_ens"
RUIE = f"{ROOT}/datasets/RUIE_Dataset/RUIE/UIQS"
OUT = f"{ROOT}/outputs/ushape_ens_reffree_grades.json"
N_PER_CAT = int(os.environ.get("N_PER_CAT", 200))
BATCH = int(os.environ.get("BATCH", 16))
GRADES = "ABCDE"                       # A = best, E = worst
IMG_EXT = (".jpg", ".jpeg", ".png")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LUMA = np.array([0.299, 0.587, 0.114], np.float32)   # same weights as eval_reffree_response.py


class UshapeDet(nn.Module):
    """Deterministic U-shape generator, tanh-bounded to the same [-1,1] range the
    FUnIE members output, so the ensemble spread is measured on one scale."""

    def __init__(self):
        super().__init__()
        self.g = Generator()

    def forward(self, x):
        return torch.tanh(self.g(x)[-1])


def load_members():
    paths = sorted(glob.glob(f"{CKPT_ROOT}/seed_*/best.pth"))
    assert paths, f"no member checkpoint under {CKPT_ROOT}"
    models = []
    for p in paths:
        m = UshapeDet().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"[ens] {len(models)} members: "
          f"{[os.path.basename(os.path.dirname(x)) for x in paths]}", flush=True)
    return models


def load_batch(paths):
    xs = []
    for f in paths:
        A = Image.open(f).convert("RGB").resize((256, 256))
        xs.append(torch.from_numpy(np.asarray(A, np.float32) / 255).permute(2, 0, 1) * 2 - 1)
    return torch.stack(xs)


@torch.no_grad()
def ens_sigma_and_contrast(models, x):
    """Per-image ensemble sigma (mean over pixels) and luminance-std contrast proxy.

    Mirrors eval_reffree_response.py: clamp each member to [-1,1], take the
    population std across members, halve it to land in [0,1], average over pixels.
    """
    so = sso = None
    for m in models:
        o = m(x).clamp(-1, 1)
        so = o if so is None else so + o
        sso = o * o if sso is None else sso + o * o
    K = len(models)
    var = ((sso / K) - (so / K) ** 2).clamp_min(0)
    sig = (torch.sqrt(var) / 2).mean(dim=[1, 2, 3]).cpu().numpy()
    x01 = ((x + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1)
    con = np.array([float((im @ LUMA).std()) for im in x01], np.float32)
    return sig, con


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    torch.manual_seed(42)
    models = load_members()

    per_grade, pooled_sig, pooled_con = {}, [], []
    for g in GRADES:
        d = os.path.join(RUIE, g)
        allf = sorted(os.listdir(d))
        imgs = [f for f in allf if f.lower().endswith(IMG_EXT)]
        skipped = len(allf) - len(imgs)
        fs = imgs[:N_PER_CAT]
        paths = [os.path.join(d, f) for f in fs]
        sigs, cons = [], []
        for i in range(0, len(paths), BATCH):
            x = load_batch(paths[i:i + BATCH]).to(DEVICE)
            s, c = ens_sigma_and_contrast(models, x)
            sigs.extend(s.tolist())
            cons.extend(c.tolist())
        per_grade[f"RUIE_{g}"] = {
            "n": len(sigs),
            "sigma_ushape_ens": float(np.mean(sigs)),
            "sigma_std": float(np.std(sigs)),
            "contrast": float(np.mean(cons)),
            "non_image_files_skipped": skipped,
        }
        pooled_sig.extend(sigs)
        pooled_con.extend(cons)
        print(f"[grade {g}] n={len(sigs)} sigma={np.mean(sigs):.5f} "
              f"contrast={np.mean(cons):.4f}", flush=True)

    grade_means = [per_grade[f"RUIE_{g}"]["sigma_ushape_ens"] for g in GRADES]
    severity = list(range(len(GRADES)))          # A=0 (best) ... E=4 (worst)
    out = {
        "method": "ENS@U-shape",
        "n_members": len(models),
        "device": DEVICE,
        "protocol": ("eval_reffree_response.py verbatim: first %d sorted image "
                     "filenames per UIQS grade, resize 256, sigma = population std "
                     "across members / 2 averaged over pixels, contrast proxy = "
                     "luminance-weighted (0.299/0.587/0.114) grayscale std" % N_PER_CAT),
        "severity_rank": {g: severity[i] for i, g in enumerate(GRADES)},
        "grades": per_grade,
        "rho_grades_ushape_ens": spearman(severity, grade_means),
        "rho_pooled_sigma_vs_contrast": spearman(pooled_sig, pooled_con),
        "response_ratio_E_over_A": float(grade_means[-1] / max(grade_means[0], 1e-12)),
    }
    os.makedirs(f"{ROOT}/outputs", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print()
    for g in GRADES:
        r = per_grade[f"RUIE_{g}"]
        print("  %s sigma=%.5f contrast=%.4f" % (g, r["sigma_ushape_ens"], r["contrast"]))
    print("across-grade rho(sigma, severity) = %+.3f   <- comparable with Table VI" %
          out["rho_grades_ushape_ens"])
    print("pooled rho(sigma, contrast)       = %+.3f   <- the old +0.14 estimator" %
          out["rho_pooled_sigma_vs_contrast"])
    print("response ratio sigma_E / sigma_A  = %.3f" % out["response_ratio_E_over_A"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
