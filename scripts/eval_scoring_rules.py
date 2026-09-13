# -*- coding: utf-8 -*-
"""Proper scoring rules for the audited sigma maps, before and after the scalar recalibration.

Table III reports NLL and CRPS per pixel. NLL and CRPS treat the reported sigma as a
probability forecast, so unlike coverage they also reward sharpness. Both are read as
raw -> recalibrated, where the recalibration is the single scalar ratio of Sec. V-C.

Sources
  FUnIE-GAN, EDL and deep ensembles : outputs/arrays_{EUVP,UIEB}.npz, scored here.
  FUnIE-GAN ratios                  : outputs/coverage_stratified.json ("ratio_used")
  U-shape Transformer + EDL         : outputs/ushape_edl_recalibration.json, in-domain;
                                      outputs/ushape_crossdomain_uieb.json, cross-domain
  PUIE (CVAE)                       : outputs/puie_eval.json

Sigma floor. A handful of ensemble pixels on UIEB carry sigma exactly 0, which sends the
Gaussian NLL to infinity. We floor sigma at 1e-4; the floor touches at most one pixel in
~6e6, so it does not move the reported means, and we record how many pixels it touched.

Writes outputs/proper_scoring_rules.json.
"""
import json
import os

import numpy as np
from scipy.special import erf

ROOT = os.environ.get("BPINN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs")
FLOOR = 1e-4


def Phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def scores(err, sig):
    """Mean NLL and mean CRPS of a Gaussian forecast, per pixel."""
    z = err / sig
    nll = 0.5 * np.log(2.0 * np.pi * sig ** 2) + 0.5 * z ** 2
    crps = sig * (z * (2.0 * Phi(z) - 1.0) + 2.0 * phi(z) - 1.0 / np.sqrt(np.pi))
    return {"nll": float(nll.mean()), "crps": float(crps.mean())}


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


cs = load("coverage_stratified.json")
record = {"sigma_floor": FLOOR, "paradigms": {}}


def put(paradigm, dataset, raw, recal, note=""):
    record["paradigms"].setdefault(paradigm, {})[dataset] = {
        "nll_raw": raw["nll"], "nll_recal": None if recal is None else recal["nll"],
        "crps_raw": raw["crps"], "crps_recal": None if recal is None else recal["crps"],
        "note": note,
    }


# ---- FUnIE-GAN: scored from the per-pixel arrays ----
for dataset, key in (("EUVP", "EUVP_域内"), ("UIEB", "UIEB_跨域")):
    d = np.load(os.path.join(OUT, "arrays_%s.npz" % dataset))
    for arm, label in (("edl", "EDL (FUnIE)"), ("ensemble", "Ensemble (FUnIE)")):
        err = d["err_" + ("ens" if arm == "ensemble" else arm)].astype(np.float64)
        sig = d["sig_" + ("ens" if arm == "ensemble" else arm)].astype(np.float64)
        n_touched = int((sig < FLOOR).sum())
        floored = np.maximum(sig, FLOOR)
        raw = scores(err, floored)
        recal = scores(err, floored * cs[key][arm]["ratio_used"])
        put(label, dataset, raw, recal,
            "sigma floored on %d of %d pixels" % (n_touched, sig.size))

# ---- U-shape Transformer + EDL: already scored upstream ----
u_euvp = load("ushape_edl_recalibration.json")
put("EDL (U-shape)", "EUVP", u_euvp["test_raw"], u_euvp["test_recal"],
    "its own recalibration, not the FUnIE ratio")
uid = load("ushape_crossdomain_uieb.json")["edl"]
put("EDL (U-shape)", "UIEB", uid["raw"], {"nll": uid["domain"]["nll"], "crps": uid["domain"]["crps"]},
    "cross-domain ratio applied in-domain; no held-out recalibration on UIEB")

# ---- PUIE (CVAE): sigma collapses toward zero, no UIEB recalibration ----
pe = load("puie_eval.json")
put("PUIE (CVAE)", "EUVP", pe["euvp_val"]["test_raw"], pe["euvp_val"]["test_recal"],
    "NLL of order 1e6: sigma collapses on a few pixels")
put("PUIE (CVAE)", "UIEB", pe["uieb_cross"]["test_raw"], None,
    "no recalibration carried out cross-domain")

with open(os.path.join(OUT, "proper_scoring_rules.json"), "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)

print("NLL and CRPS, raw -> recalibrated (lower is better)")
print("%-20s %-6s %18s %18s" % ("paradigm", "set", "NLL", "CRPS"))
for paradigm, per in record["paradigms"].items():
    for dataset, v in per.items():
        f_nll = "%9.3g -> %9.3g" % (v["nll_raw"], v["nll_recal"]) if v["nll_recal"] is not None \
            else "%9.3g -> --" % v["nll_raw"]
        f_crps = "%8.4f -> %8.4f" % (v["crps_raw"], v["crps_recal"]) if v["crps_recal"] is not None \
            else "%8.4f -> --" % v["crps_raw"]
        print("%-20s %-6s %18s %18s" % (paradigm, dataset, f_nll, f_crps))
print("\nwrote", os.path.join(OUT, "proper_scoring_rules.json"))
