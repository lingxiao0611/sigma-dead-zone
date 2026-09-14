# -*- coding: utf-8 -*-
"""Assemble the RUIE-UIQS severity-ordering row of Table VI under one estimator.

The row reports rho(sigma, severity) for four arms. Three cells were always
computed as the Spearman correlation between the five per-grade mean sigmas and
the severity rank, in eval_reffree_response.py. The U-shape ensemble cell used a
different quantity -- a pooled per-image correlation against a contrast proxy --
so it was not comparable. eval_ushape_ens_rf_grades.py recomputes that cell under
the shared estimator; this script joins the four so the row has a single source
of truth and the superseded value stays on the record.

    python code/build_ruie_severity_rho.py            # reads outputs/, writes outputs/

Writes outputs/ruie_severity_rho.json.
"""
import json
import os
import sys

import numpy as np

OUT_DIR = os.environ.get("OUT_DIR", "outputs")
GRADES = "ABCDE"


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def arm(rec, key, label, source):
    """Pull one arm's five per-grade sigmas out of a record and score the row."""
    sig = [rec[f"RUIE_{g}"][key] for g in GRADES]
    return {
        "sigma_per_grade": {g: sig[i] for i, g in enumerate(GRADES)},
        "rho": spearman(list(range(len(GRADES))), sig),
        "ratio_E_over_A": float(sig[-1] / sig[0]),
        "source": source,
    }


def main():
    rf = json.load(open(os.path.join(OUT_DIR, "reffree_response.json")))
    ug = json.load(open(os.path.join(OUT_DIR, "ushape_ens_reffree_grades.json")))

    models = {
        "FUnIE+EDL": arm(rf, "sigma_edl", "FUnIE+EDL",
                         "reffree_response.json:RUIE_*/sigma_edl"),
        "U-shape+EDL": arm(rf, "sigma_ushape_edl", "U-shape+EDL",
                           "reffree_response.json:RUIE_*/sigma_ushape_edl"),
        "FUnIE+ENS": arm(rf, "sigma_ens", "FUnIE+ENS",
                         "reffree_response.json:RUIE_*/sigma_ens"),
        "U-shape+ENS": arm(ug["grades"], "sigma_ushape_ens", "U-shape+ENS",
                           "ushape_ens_reffree_grades.json:grades/RUIE_*/sigma_ushape_ens"),
    }
    order = ["FUnIE+EDL", "U-shape+EDL", "FUnIE+ENS", "U-shape+ENS"]

    out = {
        "table": "Table VI, final row: rho(sigma, severity) over RUIE-UIQS grades A-E",
        "estimator": "spearman(severity_rank 0..4, per-grade mean sigma)",
        "severity_rank": {g: i for i, g in enumerate(GRADES)},
        "n_per_grade": 200,
        "models": models,
        "rho_row": {m: models[m]["rho"] for m in order},
        "superseded": {
            "U-shape+ENS": {
                "old_value": 0.14097712196802187,
                "old_source": "ushape_ens_reffree.json:reffree_spearman_sigma_vs_contrast",
                "why": ("pooled per-image rho(sigma, contrast proxy) over a flat "
                        "300-image draw spanning all grades, contrast taken as an "
                        "unweighted channel mean, and a different model from the one "
                        "the column names. Not comparable with the across-grade "
                        "estimator the other three cells use, and its sign convention "
                        "flips with the proxy direction."),
            }
        },
    }

    path = os.path.join(OUT_DIR, "ruie_severity_rho.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print("%-14s %8s %10s" % ("arm", "rho", "ratio E/A"))
    for m in order:
        print("%-14s %+8.3f %10.3f" % (m, models[m]["rho"], models[m]["ratio_E_over_A"]))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
