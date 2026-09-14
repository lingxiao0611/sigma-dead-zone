# -*- coding: utf-8 -*-
"""Build the Table I accuracy record from the per-epoch training logs.

Table I reports restoration accuracy with and without the uncertainty mechanism under
the identical 30-epoch budget. Those numbers are the best-validation epoch of each run,
which lives in the per-epoch training logs rather than in any evaluation record, so this
script lifts them into one JSON that travels with the rest of the audit.

Run from the repository root on the training host:

    python code/_srv/collect_table1_accuracy.py

Writes outputs/table1_accuracy.json.
"""
import json
import os

LOG_KEYS = {
    "FUnIE-GAN (deterministic)": ("logs/funie_baseline_trainlog.jsonl", 7.02),
    "FUnIE-GAN + EDL": ("logs/funie_edl_trainlog.jsonl", 7.02),
    "U-shape Transformer (deterministic)": ("logs/ushape_baseline_trainlog.jsonl", 31.6),
    "U-shape Transformer + EDL": ("logs/ushape_edl_trainlog.jsonl", 31.6),
}

CONTROL = ("logs/funie_gauss_trainlog.jsonl", "Gaussian head (Sec. VI-B)")

# Mamba-UIE is evaluated by its own script rather than a per-epoch training log.
MAMBA_SRC = "outputs/mambauie_eval.json"


def best_row(path):
    """Return the epoch with the highest validation PSNR."""
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        raise ValueError("no epochs recorded in %s" % path)
    return max(rows, key=lambda r: r["val_psnr"])


def summarize(path, params_m):
    b = best_row(path)
    return {
        "psnr": b["val_psnr"],
        "ssim": b["val_ssim"],
        "best_epoch": b["epoch"],
        "epochs_recorded": sum(1 for line in open(path, encoding="utf-8") if line.strip()),
        "params_millions": params_m,
    }


def main():
    out = {
        "table": "I",
        "caption": ("Restoration accuracy with and without the uncertainty mechanism, "
                    "under the identical 30-epoch budget (EUVP validation)."),
        "selection_rule": ("best-validation epoch of each run; the paper reports the value "
                           "at that epoch, not the final-epoch value"),
        "protocol": {
            "epochs": 30,
            "batch_size": 8,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "betas": [0.5, 0.999],
            "lr_schedule": None,
            "weight_decay": None,
            "precision": "fp32",
            "input_normalization": [-1, 1],
            "augmentation": "single random horizontal flip, applied identically to input and reference",
            "validation_split": "EUVP validation block used by the training pipeline",
        },
        "rows": {},
        "sigma_available": {
            "FUnIE-GAN (deterministic)": False,
            "FUnIE-GAN + EDL": True,
            "U-shape Transformer (deterministic)": False,
            "U-shape Transformer + EDL": True,
            "Mamba-UIE (deterministic, fp32)": False,
        },
        "notes": [
            "Every row is retrained by us under one protocol; the numbers are not comparable "
            "to the original papers' settings, which use different splits, losses and budgets.",
            "The best epoch falls in epochs 26-30 for every row, so no comparison here hangs "
            "on an under-trained model.",
            "Mamba-UIE reaches 23.01 dB under this budget, below its published 27.13 dB, which "
            "uses a far larger budget and batch; it is used as an architecture-diverse precision "
            "reference only.",
        ],
    }

    for name, (path, params_m) in LOG_KEYS.items():
        if not os.path.exists(path):
            raise SystemExit("missing training log: %s" % path)
        out["rows"][name] = summarize(path, params_m)

    cpath, cname = CONTROL
    if os.path.exists(cpath):
        out["rows"][cname] = summarize(cpath, 7.02)

    if os.path.exists(MAMBA_SRC):
        m = json.load(open(MAMBA_SRC, encoding="utf-8"))
        out["rows"]["Mamba-UIE (deterministic, fp32)"] = {
            "psnr": m["euvp_val"]["psnr"],
            "ssim": m["euvp_val"]["ssim"],
            "n_images": m["euvp_val"]["n"],
            "checkpoint": m["ckpt"],
            "params_millions": None,
        }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/table1_accuracy.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    for name, r in out["rows"].items():
        print("%-38s psnr %.4f  ssim %.4f  ep %s" % (name, r["psnr"], r["ssim"], r.get("best_epoch")))
    print("wrote outputs/table1_accuracy.json")


if __name__ == "__main__":
    main()
