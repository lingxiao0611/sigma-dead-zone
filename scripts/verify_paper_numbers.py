# -*- coding: utf-8 -*-
"""Re-derive every headline number in the paper from the released audit records.

Read-only. Each check states the paper's value and the record path it came from, so a
mismatch is caught here rather than by a reviewer. Run from the repository root:

    python code/verify_paper_numbers.py                    # reads records/
    python code/verify_paper_numbers.py --records outputs  # reads the working tree

Exit status is non-zero if any check mismatches, so this can gate a release.
"""
import argparse
import json
import os
import sys

try:
    import numpy as np
except ImportError:  # the array-based figure checks are skipped without it
    np = None

ROWS = []
UNBACKED = []


def load(rec, name):
    path = os.path.join(rec, name)
    if not os.path.exists(path):
        raise SystemExit("record not found: %s" % path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dig(obj, path, default=None):
    """Navigate 'a/b/c' or 'a/1/b'; returns default if any step is missing."""
    cur = obj
    for part in path.split("/"):
        try:
            if isinstance(cur, list):
                cur = cur[int(part)]
            else:
                cur = cur[part]
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    return cur


def chk(label, got, want, tol=5e-4, src=""):
    if got is None:
        ROWS.append((label, "KEY-MISSING", want, "MISSING", src))
        return False
    if isinstance(want, bool):
        ok, shown = bool(got) == want, bool(got)
    elif isinstance(want, str):
        ok, shown = str(got) == want, got
    else:
        ok, shown = abs(float(got) - float(want)) <= tol, round(float(got), 6)
    ROWS.append((label, shown, want, "OK" if ok else "MISMATCH", src))
    return ok


def pct(x):
    """Record stores a fraction; the paper prints a percentage with one decimal."""
    return None if x is None else round(x * 100, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records", help="directory holding the JSON records")
    ap.add_argument("--arrays", default="outputs",
                    help="directory holding arrays_EUVP.npz / arrays_UIEB.npz (optional)")
    a = ap.parse_args()
    rec = a.records

    T1 = load(rec, "table1_accuracy.json")
    EDLR = load(rec, "edl_recalibration.json")
    ENSR = load(rec, "ensemble_recalibration.json")
    XD = load(rec, "crossdomain_uieb.json")
    FS = load(rec, "fewshot_recal_uieb.json")
    COV = load(rec, "coverage_stratified.json")
    REJ = load(rec, "rejection_stage4.json")
    BOOT = load(rec, "bootstrap_ci.json")
    PSR = load(rec, "proper_scoring_rules.json")
    PUIE = load(rec, "puie_eval.json")
    PUIEA = load(rec, "puie_ablation.json")
    PUIS = load(rec, "puie_stratified.json")
    SS = load(rec, "split_stability.json")
    NB = load(rec, "naive_baselines.json")
    HI = load(rec, "head_intervention.json")
    GA = load(rec, "gauss_audit.json")
    RF = load(rec, "reffree_response.json")
    STRAT = load(rec, "strat_proxy_evidence.json")
    FUS = load(rec, "fusion_gate_uieb.json")
    UREJ = load(rec, "ushape_rejection_stage4.json")
    UST = load(rec, "ushape_stratified.json")
    UCD = load(rec, "ushape_crossdomain_uieb.json")
    URE = load(rec, "ushape_edl_recalibration.json")
    UEA = load(rec, "ushape_ens_audit.json")
    UER = load(rec, "ushape_ens_reffree.json")
    UEG = load(rec, "ushape_ens_reffree_grades.json")
    RSR = load(rec, "ruie_severity_rho.json")

    # ---------------------------------------------------------------- Table I
    t1 = T1["rows"]
    s = "table1_accuracy.json"
    chk("T1 FUnIE det PSNR 23.55", t1["FUnIE-GAN (deterministic)"]["psnr"], 23.55, 0.005, s)
    chk("T1 FUnIE det SSIM 0.859", t1["FUnIE-GAN (deterministic)"]["ssim"], 0.859, 5e-4, s)
    chk("T1 FUnIE+EDL PSNR 23.55", t1["FUnIE-GAN + EDL"]["psnr"], 23.55, 0.005, s)
    chk("T1 FUnIE+EDL SSIM 0.860", t1["FUnIE-GAN + EDL"]["ssim"], 0.860, 5e-4, s)
    chk("T1 U-shape det PSNR 22.57", t1["U-shape Transformer (deterministic)"]["psnr"], 22.57, 0.005, s)
    chk("T1 U-shape det SSIM 0.815", t1["U-shape Transformer (deterministic)"]["ssim"], 0.815, 5e-4, s)
    chk("T1 U-shape+EDL PSNR 22.54", t1["U-shape Transformer + EDL"]["psnr"], 22.54, 0.005, s)
    chk("T1 U-shape+EDL SSIM 0.812", t1["U-shape Transformer + EDL"]["ssim"], 0.812, 5e-4, s)
    chk("T1 Mamba PSNR 23.01", t1["Mamba-UIE (deterministic, fp32)"]["psnr"], 23.01, 0.005, s)
    chk("T1 Mamba SSIM 0.876", t1["Mamba-UIE (deterministic, fp32)"]["ssim"], 0.876, 5e-4, s)
    eps = [r.get("best_epoch") for r in t1.values() if r.get("best_epoch")]
    chk("T1 best epoch within 26-30", 26 <= min(eps) and max(eps) <= 30, True, src=s)

    # --------------------------------------------------------------- Table II
    edl = COV["EUVP_域内"]["edl"]["marginal"]
    ens = COV["EUVP_域内"]["ensemble"]["marginal"]
    s = "coverage_stratified.json"
    chk("T2 EDL 1sigma 23.5%", pct(edl["cov1"]), 23.5, 0.05, s)
    chk("T2 EDL 2sigma 44.0%", pct(edl["cov2"]), 44.0, 0.05, s)
    chk("T2 EDL 3sigma 59.5%", pct(edl["cov3"]), 59.5, 0.05, s)
    chk("T2 ENS 1sigma 22.9%", pct(ens["cov1"]), 22.9, 0.05, s)
    chk("T2 ENS 2sigma 42.1%", pct(ens["cov2"]), 42.1, 0.05, s)
    chk("T2 ENS 3sigma 56.7%", pct(ens["cov3"]), 56.7, 0.05, s)
    chk("T2 EDL 5th pct 0.092", edl["cov1_p5_across_imgs"], 0.092, 5e-4, s)
    chk("T2 ENS 5th pct 0.087", ens["cov1_p5_across_imgs"], 0.087, 5e-4, s)
    chk("T2 EDL images<40% 96.2%", pct(edl["frac_img_cov1_lt40"]), 96.2, 0.05, s)
    chk("T2 ENS images<40% 97.0%", pct(ens["frac_img_cov1_lt40"]), 97.0, 0.05, s)
    chk("T2 PUIE 1sigma 11.2%", pct(dig(PUIE, "euvp_val/z_raw/1sigma")), 11.2, 0.05, "puie_eval.json")
    chk("T2 PUIE 2sigma 22.4%", pct(dig(PUIE, "euvp_val/z_raw/2sigma")), 22.4, 0.05, "puie_eval.json")
    chk("T2 PUIE 3sigma 32.7%", pct(dig(PUIE, "euvp_val/z_raw/3sigma")), 32.7, 0.05, "puie_eval.json")
    chk("T2 PUIE mean sigma 0.011", dig(PUIE, "euvp_val/sigma_mean"), 0.011, 5e-4, "puie_eval.json")
    edl_sig = [x["sigma_mean"] for x in COV["EUVP_域内"]["edl"]["strata_raw"]]
    ens_sig = [x["sigma_mean"] for x in COV["EUVP_域内"]["ensemble"]["strata_raw"]]
    chk("T2 EDL mean sigma 0.017", sum(edl_sig) / len(edl_sig), 0.017, 5e-4, s)
    chk("T2 ENS mean sigma 0.017", sum(ens_sig) / len(ens_sig), 0.017, 5e-4, s)
    er = [x["cov1"] for x in COV["EUVP_域内"]["edl"]["strata_raw"]]
    nr = [x["cov1"] for x in COV["EUVP_域内"]["ensemble"]["strata_raw"]]
    chk("T2 EDL stratum S1 0.235", er[0], 0.235, 5e-4, s)
    chk("T2 EDL stratum S4 0.229", er[-1], 0.229, 5e-4, s)
    chk("T2 ENS stratum S1 0.228", nr[0], 0.228, 5e-4, s)
    chk("T2 ENS stratum S4 0.228", nr[-1], 0.228, 5e-4, s)
    frac = [x["frac_img_cov1_lt40"] for x in COV["EUVP_域内"]["edl"]["strata_raw"]] + \
           [x["frac_img_cov1_lt40"] for x in COV["EUVP_域内"]["ensemble"]["strata_raw"]]
    chk("T2 per-stratum range lower bound 94%", round(pct(min(frac))), 94, 0.5, s)

    # -------------------------------------------------------------- Table III
    s = "proper_scoring_rules.json"
    p = PSR["paradigms"]
    chk("T3 EDL/FUnIE EUVP NLL 10.1", dig(p, "EDL (FUnIE)/EUVP/nll_raw"), 10.1, 0.05, s)
    chk("T3 EDL/FUnIE EUVP NLL -> -1.0", dig(p, "EDL (FUnIE)/EUVP/nll_recal"), -1.0, 0.05, s)
    chk("T3 EDL/FUnIE UIEB NLL 48.2", dig(p, "EDL (FUnIE)/UIEB/nll_raw"), 48.2, 0.05, s)
    chk("T3 EDL/FUnIE UIEB CRPS 0.105", dig(p, "EDL (FUnIE)/UIEB/crps_raw"), 0.105, 5e-4, s)
    chk("T3 EDL/FUnIE UIEB CRPS -> 0.085", dig(p, "EDL (FUnIE)/UIEB/crps_recal"), 0.085, 5e-4, s)
    chk("T3 ENS/FUnIE UIEB NLL 46.2", dig(p, "Ensemble (FUnIE)/UIEB/nll_raw"), 46.2, 0.05, s)
    chk("T3 ENS/FUnIE UIEB CRPS 0.099", dig(p, "Ensemble (FUnIE)/UIEB/crps_raw"), 0.099, 5e-4, s)
    chk("T3 ENS/FUnIE UIEB CRPS -> 0.088", dig(p, "Ensemble (FUnIE)/UIEB/crps_recal"), 0.088, 5e-4, s)
    chk("T3 U-shape EUVP NLL 3.7", dig(p, "EDL (U-shape)/EUVP/nll_raw"), 3.7, 0.05, s)
    chk("T3 U-shape UIEB NLL 28.5", dig(p, "EDL (U-shape)/UIEB/nll_raw"), 28.5, 0.05, s)
    chk("T3 U-shape UIEB CRPS 0.111", dig(p, "EDL (U-shape)/UIEB/crps_raw"), 0.111, 5e-4, s)
    chk("T3 U-shape UIEB CRPS -> 0.090", dig(p, "EDL (U-shape)/UIEB/crps_recal"), 0.090, 5e-4, s)
    chk("T3 PUIE EUVP CRPS 0.065", dig(PUIE, "euvp_val/test_raw/crps"), 0.065, 5e-4, "puie_eval.json")
    chk("T3 PUIE EUVP CRPS -> 0.052", dig(PUIE, "euvp_val/test_recal/crps"), 0.052, 5e-4, "puie_eval.json")
    # Sec. V-C body: CRPS prefers EDL to the ensemble cross-domain
    chk("V-C cross CRPS EDL 0.085 < ENS 0.088",
        dig(p, "EDL (FUnIE)/UIEB/crps_recal") < dig(p, "Ensemble (FUnIE)/UIEB/crps_recal"), True, src=s)

    # ------------------------------------------------- Sec. V-C single scalar
    s = "edl_recalibration.json / ensemble_recalibration.json"
    chk("V-C EDL ratio 3.630", EDLR["ratio"], 3.630, 5e-4, s)
    chk("V-C ENS ratio 3.838", ENSR["ratio"], 3.838, 5e-4, s)
    chk("V-C PUIE ratio 7.18", dig(PUIE, "euvp_val/ratio"), 7.18, 5e-3, "puie_eval.json")
    chk("V-C EDL RMS-CE 0.354", dig(EDLR, "test_raw/rms_cal"), 0.354, 5e-4, s)
    chk("V-C EDL RMS-CE -> 0.029", dig(EDLR, "test_recal/rms_cal"), 0.029, 5e-4, s)
    chk("V-C ENS RMS-CE 0.361", dig(ENSR, "test_raw/rms_cal"), 0.361, 5e-4, s)
    chk("V-C ENS RMS-CE -> 0.039", dig(ENSR, "test_recal/rms_cal"), 0.039, 5e-4, s)
    chk("V-C EDL cov1 -> 0.67", dig(EDLR, "z_coverage_recal/1sigma"), 0.67, 5e-3, s)
    chk("V-C ENS cov1 -> 0.66", dig(ENSR, "z_coverage_recal/1sigma"), 0.66, 5e-3, s)
    # ratio is a property of (model, water body), not the draw
    for para, rec, q in (("EDL", SS["EUVP"]["edl"], 3.630), ("ENS", SS["EUVP"]["ensemble"], 3.838)):
        cv = rec["ratio_std"] / rec["ratio_mean"] * 100
        chk("V-C %s ratio CV in 1.3-1.8%%" % para, 1.3 <= cv <= 1.8, True, src="split_stability.json")
        chk("V-C %s quoted within 2%% of 30-split mean" % para,
            abs(q - rec["ratio_mean"]) / rec["ratio_mean"] * 100 <= 2.0, True,
            src="split_stability.json")

    # ----------------------------------------------------- Sec. V-D transfer
    s = "crossdomain_uieb.json / fewshot_recal_uieb.json / coverage_stratified.json"
    chk("V-D EDL cross ratio 7.683", dig(XD, "edl/ratio_domain"), 7.683, 5e-4, s)
    chk("V-D ENS cross ratio 5.661", dig(XD, "ensemble/ratio_domain"), 5.661, 5e-4, s)
    chk("V-D PUIE cross ratio 9.39", dig(PUIE, "uieb_cross/ratio_domain"), 9.39, 5e-3, "puie_eval.json")
    chk("V-D EDL inflation x2.12", dig(XD, "edl/ratio_domain") / EDLR["ratio"], 2.12, 5e-3, s)
    chk("V-D ENS inflation x1.48", dig(XD, "ensemble/ratio_domain") / ENSR["ratio"], 1.48, 5e-3, s)
    chk("V-D PUIE inflation x1.31",
        dig(PUIE, "uieb_cross/ratio_domain") / dig(PUIE, "euvp_val/ratio"), 1.31, 5e-3, "puie_eval.json")
    chk("V-D EDL raw RMS-CE 0.460", dig(FS, "edl/reference/raw_rms"), 0.460, 5e-4, s)
    for n, erow, nrow in ((5, (0.049, 0.016), (0.065, 0.021)),
                          (50, (0.039, 0.011), (0.060, 0.009)),
                          (100, (0.031, 0.004), (0.050, 0.004))):
        g = dig(FS, "edl/sweep/%d/rms_mean" % n)
        st = dig(FS, "edl/sweep/%d/rms_std" % n)
        chk("V-D EDL N=%d rms %s" % (n, erow[0]), g, erow[0], 5e-4, s)
        chk("V-D EDL N=%d std %s" % (n, erow[1]), st, erow[1], 5e-4, s)
        g = dig(FS, "ensemble/sweep/%d/rms_mean" % n)
        st = dig(FS, "ensemble/sweep/%d/rms_std" % n)
        chk("V-D ENS N=%d rms %s" % (n, nrow[0]), g, nrow[0], 5e-4, s)
        chk("V-D ENS N=%d std %s" % (n, nrow[1]), st, nrow[1], 5e-4, s)
    uc, cc = COV["UIEB_跨域"], COV["EUVP_域内"]
    ue = [x["cov1"] for x in uc["edl"]["strata_recal"]]
    un = [x["cov1"] for x in uc["ensemble"]["strata_recal"]]
    chk("V-D EDL recal strata range 0.249", max(ue) - min(ue), 0.249, 5e-4, s)
    chk("V-D ENS recal strata range 0.123", max(un) - min(un), 0.123, 5e-4, s)
    chk("V-D EDL pre-recal strata range 0.080",
        max(x["cov1"] for x in uc["edl"]["strata_raw"]) - min(x["cov1"] for x in uc["edl"]["strata_raw"]),
        0.080, 5e-4, s)
    chk("V-D EUVP recal strata range 0.030",
        max(x["cov1"] for x in cc["ensemble"]["strata_recal"]) -
        min(x["cov1"] for x in cc["ensemble"]["strata_recal"]), 0.030, 5e-4, s)
    for i, want in enumerate((0.556, 0.583, 0.726, 0.804)):
        chk("V-D EDL recal stratum %d %s" % (i + 1, want), ue[i], want, 5e-4, s)
    chk("V-D ENS recal low 0.579", min(un), 0.579, 5e-4, s)
    chk("V-D ENS recal high 0.702", max(un), 0.702, 5e-4, s)

    # -------------------------------------------------------- Sec. V-E dead zone
    s = "coverage_stratified.json / naive_baselines.json"
    chk("V-E EDL img lift in 1.49", dig(NB, "EUVP_域内/edl/img_lift"), 1.49, 5e-3, s)
    chk("V-E ENS img lift in 1.69", dig(NB, "EUVP_域内/ensemble/img_lift"), 1.69, 5e-3, s)
    chk("V-E EDL pixel lift in 2.56", dig(NB, "EUVP_域内/edl/pixel_lift"), 2.56, 5e-3, s)
    chk("V-E ENS pixel lift in 2.23", dig(NB, "EUVP_域内/ensemble/pixel_lift"), 2.23, 5e-3, s)
    chk("V-E EDL img lift cross 0.87", dig(NB, "UIEB_跨域/edl/img_lift"), 0.87, 5e-3, s)
    chk("V-E ENS img lift cross 1.11", dig(NB, "UIEB_跨域/ensemble/img_lift"), 1.11, 5e-3, s)
    chk("V-E EDL pixel lift cross 1.17", dig(NB, "UIEB_跨域/edl/pixel_lift"), 1.17, 5e-3, s)
    chk("V-E ENS pixel lift cross 1.54", dig(NB, "UIEB_跨域/ensemble/pixel_lift"), 1.54, 5e-3, s)
    chk("V-E cross rho -0.11", dig(COV, "UIEB_跨域/edl/image_level/spearman_sigma_vs_err"), -0.11, 5e-3, s)
    chk("V-E / V-F inverted lift CI lo 0.77", dig(BOOT, "UIEB_跨域/lift_edl/lo"), 0.77, 5e-3, "bootstrap_ci.json")
    chk("V-E / V-F inverted lift CI hi 0.97", dig(BOOT, "UIEB_跨域/lift_edl/hi"), 0.97, 5e-3, "bootstrap_ci.json")
    chk("V-E / V-F lift ENS>EDL in 100%",
        dig(BOOT, "UIEB_跨域/delta_lift_ens_minus_edl/p_gt0"), 1.0, 1e-9, "bootstrap_ci.json")
    chk("V-E EDL cov S1 0.087", uc["edl"]["strata_raw"][0]["cov1"], 0.087, 5e-4, s)
    chk("V-E EDL cov S4 0.167", uc["edl"]["strata_raw"][-1]["cov1"], 0.167, 5e-4, s)
    chk("V-E ENS cov S1 0.165", uc["ensemble"]["strata_raw"][0]["cov1"], 0.165, 5e-4, s)
    chk("V-E ENS cov S4 0.177", uc["ensemble"]["strata_raw"][-1]["cov1"], 0.177, 5e-4, s)
    em = [x["err_mean"] for x in uc["edl"]["strata_raw"]]
    sm = [x["sigma_mean"] for x in uc["edl"]["strata_raw"]]
    chk("V-E error S1 0.143", em[0], 0.143, 5e-4, s)
    chk("V-E error S4 0.082", em[-1], 0.082, 5e-4, s)
    chk("V-E error grows 73%", em[0] / em[-1] - 1.0, 0.73, 5e-3, s)
    chk("V-E EDL sigma S1 0.0177", sm[0], 0.0177, 5e-5, s)
    chk("V-E EDL sigma S4 0.0181", sm[-1], 0.0181, 5e-5, s)
    # The sigma response ratios quoted in the Fig. 4/Fig. 6 captions are computed from the
    # per-pixel arrays, whose binning and normalization basis differ from the strata means
    # above, so they are checked in the array section below rather than against the strata.
    chk("V-E RUIE EDL response 0.81",
        RF["RUIE_E"]["sigma_edl"] / RF["RUIE_A"]["sigma_edl"], 0.81, 5e-3, "reffree_response.json")
    chk("V-E RUIE ENS response 1.46",
        RF["RUIE_E"]["sigma_ens"] / RF["RUIE_A"]["sigma_ens"], 1.46, 5e-3, "reffree_response.json")
    for cls, want in (("U45_blue", -0.20), ("U45_green", -0.52), ("U45_haze", -0.58)):
        chk("V-E U45 ENS rho %s" % cls, dig(RF, "%s/rho_contrast_ens" % cls), want, 6e-3,
            "reffree_response.json")
    for cls, want in (("U45_blue", -0.06), ("U45_green", 0.23), ("U45_haze", -0.32)):
        chk("V-E U45 EDL rho %s" % cls, dig(RF, "%s/rho_contrast_edl" % cls), want, 5e-3, "reffree_response.json")

    # ------------------------------------------------- Sec. V-F rejection
    s = "rejection_stage4.json / bootstrap_ci.json / naive_baselines.json / fusion_gate_uieb.json"
    chk("V-F EDL in-domain AURC 0.04016", dig(REJ, "EUVP_域内/edl/aurc"), 0.04016, 1e-5, s)
    chk("V-F ENS in-domain AURC 0.04011", dig(REJ, "EUVP_域内/ensemble/aurc"), 0.04011, 1e-5, s)
    chk("V-F EDL cross AURC 0.1084", dig(REJ, "UIEB_跨域/edl/aurc"), 0.1084, 5e-5, s)
    chk("V-F ENS cross AURC 0.0973", dig(REJ, "UIEB_跨域/ensemble/aurc"), 0.0973, 5e-5, s)
    edla, ensa = dig(REJ, "UIEB_跨域/edl/aurc"), dig(REJ, "UIEB_跨域/ensemble/aurc")
    chk("V-F AURC margin 10.2%", (edla - ensa) / edla * 100, 10.2, 0.05, s)
    chk("V-F delta AURC 0.015",
        dig(BOOT, "UIEB_跨域/delta_aurc_edl_minus_ens/mean"), 0.015, 5e-4, "bootstrap_ci.json")
    chk("V-F delta AURC CI lo 0.010",
        dig(BOOT, "UIEB_跨域/delta_aurc_edl_minus_ens/lo"), 0.010, 5e-4, "bootstrap_ci.json")
    chk("V-F delta AURC CI hi 0.021",
        dig(BOOT, "UIEB_跨域/delta_aurc_edl_minus_ens/hi"), 0.021, 5e-4, "bootstrap_ci.json")
    rem_e = (dig(REJ, "UIEB_跨域/edl/risk_at_100cov") - dig(REJ, "UIEB_跨域/edl/risk_at_80cov")) / \
            dig(REJ, "UIEB_跨域/edl/risk_at_100cov") * 100
    rem_n = (dig(REJ, "UIEB_跨域/ensemble/risk_at_100cov") - dig(REJ, "UIEB_跨域/ensemble/risk_at_80cov")) / \
            dig(REJ, "UIEB_跨域/ensemble/risk_at_100cov") * 100
    chk("V-F EDL error removed @80 2.1%", rem_e, 2.1, 0.05, s)
    chk("V-F ENS error removed @80 6.9%", rem_n, 6.9, 0.05, s)
    chk("V-F rejection value 3.3x", rem_n / rem_e, 3.3, 0.05, s)
    chk("V-F constant sigma EUVP cov1 0.65", dig(NB, "EUVP_域内/const/marginal_cov1"), 0.65, 5e-3, s)
    chk("V-F constant sigma UIEB cov1 0.61", dig(NB, "UIEB_跨域/const/marginal_cov1"), 0.61, 5e-3, s)
    chk("V-F fusion EDL +0.04 dB", dig(FUS, "edl/gate_minus_uniform_mean"), 0.04, 5e-3, "fusion_gate_uieb.json")

    # ------------------------------------------------ Sec. V-G backbone
    s = "ushape_*.json"
    chk("V-G U-shape 1sigma 26.4%", pct(dig(UST, "EUVP_域内/marginal_cov1")), 26.4, 0.05, s)
    chk("V-G U-shape UIEB 1sigma 13.1%", pct(dig(UST, "UIEB_跨域/marginal_cov1")), 13.1, 0.05, s)
    chk("V-G U-shape ratio 3.08", dig(UST, "EUVP_域内/ratio"), 3.08, 5e-3, s)
    chk("V-G U-shape ratio cross 6.67", dig(UCD, "edl/ratio_domain"), 6.67, 5e-3,
        "ushape_crossdomain_uieb.json")
    chk("V-G the two U-shape cross-ratio records agree within 0.02",
        abs(dig(UCD, "edl/ratio_domain") - dig(UST, "UIEB_跨域/ratio")) <= 0.02, True, src=s)
    chk("V-G U-shape inflation x2.17",
        dig(UCD, "edl/ratio_domain") / dig(UST, "EUVP_域内/ratio"), 2.17, 5e-3, s)
    chk("V-G U-shape lift in 1.64", dig(UST, "EUVP_域内/lift"), 1.64, 5e-3, s)
    chk("V-G U-shape lift cross 1.00", dig(UST, "UIEB_跨域/lift"), 1.00, 5e-3, s)
    chk("V-G U-shape AURC in 0.040", dig(UREJ, "EUVP (in-domain)/edl/aurc"), 0.040, 5e-4, s)
    chk("V-G U-shape AURC cross 0.110", dig(UREJ, "UIEB (cross-domain)/edl/aurc"), 0.110, 5e-4, s)
    chk("V-G U-shape ENS 1sigma 27.6%", pct(dig(UEA, "EUVP/cov1_raw")), 27.6, 0.05, "ushape_ens_audit.json")
    chk("V-G U-shape ENS UIEB 1sigma 17.4%", pct(dig(UEA, "UIEB/cov1_raw")), 17.4, 0.05, "ushape_ens_audit.json")
    chk("V-G U-shape ENS ratio in 2.854", dig(UEA, "EUVP/ratio"), 2.854, 5e-4, "ushape_ens_audit.json")
    chk("V-G U-shape ENS ratio cross 3.755", dig(UEA, "UIEB/ratio"), 3.755, 5e-4, "ushape_ens_audit.json")
    chk("V-G U-shape ENS inflation x1.32",
        dig(UEA, "UIEB/ratio") / dig(UEA, "EUVP/ratio"), 1.32, 5e-3, "ushape_ens_audit.json")
    chk("V-G U-shape ENS AURC in 0.0373",
        dig(UEA, "EUVP/risk_coverage/aurc"), 0.0373, 5e-5, "ushape_ens_audit.json")
    chk("V-G U-shape ENS AURC cross 0.0919",
        dig(UEA, "UIEB/risk_coverage/aurc"), 0.0919, 5e-5, "ushape_ens_audit.json")
    chk("V-G U-shape ENS AURC margin 16.7%",
        (dig(UREJ, "UIEB (cross-domain)/edl/aurc") - dig(UEA, "UIEB/risk_coverage/aurc")) /
        dig(UREJ, "UIEB (cross-domain)/edl/aurc") * 100, 16.7, 0.05, s)
    # The U-shape ensemble cell used to be filled by a pooled per-image correlation
    # against a contrast proxy, which is not the estimator the other three cells use.
    # It is now recomputed across grades like the rest; the old number stays checked
    # so the substitution remains auditable.
    chk("V-G U-shape ENS pooled rho +0.14 (superseded cell)",
        dig(UER, "reffree_spearman_sigma_vs_contrast"), 0.14, 5e-3, "ushape_ens_reffree.json")
    chk("V-G U-shape ENS across-grade rho +0.90",
        dig(UEG, "rho_grades_ushape_ens"), 0.90, 5e-3, "ushape_ens_reffree_grades.json")
    chk("V-G U-shape ENS severity ratio E/A 1.18",
        dig(UEG, "response_ratio_E_over_A"), 1.18, 5e-3, "ushape_ens_reffree_grades.json")
    chk("V-G U-shape ENS pooled rho on the new draw is negative",
        dig(UEG, "rho_pooled_sigma_vs_contrast") < 0, True,
        src="ushape_ens_reffree_grades.json")
    # Table VII RUIE rho column: rank correlation between severity grade and mean sigma
    sig_e = [RF["RUIE_%s" % g]["sigma_edl"] for g in "ABCDE"]
    sig_n = [RF["RUIE_%s" % g]["sigma_ens"] for g in "ABCDE"]
    sig_us = [RF["RUIE_%s" % g]["sigma_ushape_edl"] for g in "ABCDE"]

    def spearman(seq):
        n = len(seq)
        order = sorted(range(n), key=lambda i: seq[i])
        rank = [0] * n
        for r, i in enumerate(order):
            rank[i] = r
        d2 = sum((rank[i] - i) ** 2 for i in range(n))
        return 1 - 6 * d2 / (n * (n * n - 1))

    chk("T7 RUIE rho EDL/FUnIE -1.00", spearman(sig_e), -1.00, 5e-3, "reffree_response.json")
    chk("T7 RUIE rho EDL/U-shape -0.40", spearman(sig_us), -0.40, 5e-3, "reffree_response.json")
    chk("T7 RUIE rho ENS/FUnIE +0.70", spearman(sig_n), 0.70, 5e-3, "reffree_response.json")
    sig_ue = [UEG["grades"]["RUIE_%s" % g]["sigma_ushape_ens"] for g in "ABCDE"]
    chk("T7 RUIE rho ENS/U-shape +0.90", spearman(sig_ue), 0.90, 5e-3,
        "ushape_ens_reffree_grades.json")
    # all four cells must now agree with the joined row, which shares one estimator
    chk("T7 row matches ruie_severity_rho.json",
        max(abs(dig(RSR, "rho_row/%s" % m) - spearman(seq)) for m, seq in
            [("FUnIE+EDL", sig_e), ("U-shape+EDL", sig_us), ("FUnIE+ENS", sig_n),
             ("U-shape+ENS", sig_ue)]) <= 1e-9, True, src="ruie_severity_rho.json")

    # --------------------------------------------------------- Sec. VI-B
    s = "head_intervention.json / gauss_audit.json / puie_ablation.json"
    chk("VI-B retrained head lift 0.87 -> 1.17", dig(HI, "full445/lift"), 1.17, 5e-3, s)
    chk("VI-B baseline lift 0.87", dig(HI, "baseline/lift"), 0.87, 5e-3, s)
    chk("VI-B retrained head cov1 0.21", dig(HI, "full445/marginal_cov1"), 0.21, 5e-3, s)
    chk("VI-B Gaussian sigma lightest 0.075", dig(GA, "UIEB/strata/3/sig_mean"), 0.075, 5e-4, s)
    chk("VI-B Gaussian sigma heaviest 0.087", dig(GA, "UIEB/strata/0/sig_mean"), 0.087, 5e-4, s)
    chk("VI-B Gaussian corr +0.13", dig(GA, "UIEB/sp_sig_err"), 0.13, 5e-3, s)
    chk("VI-B Gaussian cross AURC 0.098 < EDL 0.110",
        dig(GA, "UIEB/aurc") < dig(UREJ, "UIEB (cross-domain)/edl/aurc"), True, src=s)
    chk("VI-C PUIE PSNR 18.2 -> 21.2",
        dig(PUIEA, "epoch_29/ckpt_psnr"), 21.2, 0.05, "puie_ablation.json")
    chk("VI-C PUIE PSNR start 18.2", dig(PUIEA, "epoch_1/ckpt_psnr"), 18.2, 0.05, "puie_ablation.json")
    chk("VI-C PUIE sigma 0.039 -> 0.011", dig(PUIEA, "epoch_29/sigma_mean"), 0.011, 5e-4, "puie_ablation.json")
    chk("VI-C PUIE cov1 22.6% -> 11.5%",
        pct(dig(PUIEA, "epoch_29/z_raw/1sigma")), 11.5, 0.05, "puie_ablation.json")
    # Sec. VI-D: error and correction scale together
    chk("VI-D MAE in-domain 0.054", dig(EDLR, "test_raw/mae"), 0.054, 5e-4, s)
    chk("VI-D MAE cross 0.114", dig(XD, "edl/raw/mae"), 0.114, 5e-4, s)

    # --------------------------------------- figure captions, from the arrays
    # The response ratios in the Fig. 4/Fig. 6 captions normalize at the lightest contrast
    # quartile of the per-pixel arrays, a different basis from the strata means used above,
    # and the Fig. 7 ranges come from the evidence parameters. Reproduce them from source.
    au = os.path.join(a.arrays, "arrays_UIEB.npz")
    ae = os.path.join(a.arrays, "arrays_EUVP.npz")
    if np is not None and os.path.exists(au):
        dU = np.load(au)

        def profile(d):
            c = d["contrast"]
            b = np.digitize(c, np.quantile(c, [0.25, 0.5, 0.75]))
            e, se, sn = [], [], []
            for k in range(4):
                m = b == k
                e.append(d["err_edl"][m].mean())
                se.append(d["sig_edl"][m].mean())
                sn.append(d["sig_ens"][m].mean())
            return np.array(e), np.array(se), np.array(sn)

        def pooled(d):
            e = d["err_edl"].ravel()
            edges = np.quantile(e, np.linspace(0, 1, 11))
            idx = np.clip(np.digitize(e, edges) - 1, 0, 9)
            edl = np.array([d["sig_edl"].ravel()[idx == k].mean() for k in range(10)])
            ens = np.array([d["sig_ens"].ravel()[idx == k].mean() for k in range(10)])
            return edl / edl[0], ens / ens[0]

        def spatial_rho(sig, err):
            out = []
            for i in range(sig.shape[0]):
                a_, b_ = sig[i].mean(0).ravel(), err[i].mean(0).ravel()
                if a_.std() < 1e-9 or b_.std() < 1e-9:
                    continue
                out.append(np.corrcoef(a_, b_)[0, 1])
            return np.array(out)

        eU, seU, snU = profile(dU)
        sarr = "arrays_UIEB.npz"
        chk("Fig6 cross error 73.5% (caption reads x1.73)", eU[0] / eU[3], 1.735, 5e-3, sarr)
        chk("Fig6 cross ENS sigma x1.56", snU[0] / snU[3], 1.56, 5e-3, sarr)
        chk("Fig6 cross EDL sigma x0.96", seU[0] / seU[3], 0.96, 5e-3, sarr)
        edlU, ensU = pooled(dU)
        chk("Fig4 cross ENS sigma x1.45", ensU[-1], 1.45, 5e-3, sarr)
        chk("Fig4 cross EDL sigma x1.09", edlU[-1], 1.09, 5e-3, sarr)
        r_e = spatial_rho(dU["sig_edl"], dU["err_edl"])
        r_n = spatial_rho(dU["sig_ens"], dU["err_ens"])
        chk("Fig4b median rho EDL +0.03", np.median(r_e), 0.03, 5e-3, sarr)
        chk("Fig4b median rho ENS +0.14", np.median(r_n), 0.14, 5e-3, sarr)
        chk("Fig4b ENS better on 61.6%", (r_n > r_e[:len(r_n)]).mean() * 100, 61.6, 0.05, sarr)
        if os.path.exists(ae):
            dE = np.load(ae)
            eE, seE, snE = profile(dE)
            edlE, ensE = pooled(dE)
            chk("Fig6 in-domain error near unity", eE[0] / eE[3], 1.04, 0.06, sarr)
            chk("Fig4 in-domain sigma x1.55", max(edlE[-1], ensE[-1]), 1.55, 0.02, sarr)
    else:
        UNBACKED.append("Fig. 4 / Fig. 6 caption ratios and Fig. 7 ranges "
                        "(arrays or numpy not available)")

    bs = dig(HI, "baseline/strata")
    if bs:
        al = [x["alpha"] for x in bs]
        be = [x["beta"] for x in bs]
        chk("Fig7 alpha low 2.63", min(al), 2.63, 5e-3, "head_intervention.json")
        chk("Fig7 alpha high 2.72", max(al), 2.72, 5e-3, "head_intervention.json")
        chk("Fig7 beta low 0.0021", min(be), 0.0021, 5e-5, "head_intervention.json")
        chk("Fig7 beta high 0.0022", max(be), 0.0022, 5e-5, "head_intervention.json")

    # ------------------------------------------------------------------ report
    width = max(len(r[0]) for r in ROWS)
    bad = [r for r in ROWS if r[3] != "OK"]
    for lab, got, want, st, src in ROWS:
        if st != "OK":
            print("%-*s  record=%-14s  paper=%-10s  %-9s  %s" % (width, lab, got, want, st, src))
    print()
    print("checked %d claims: %d OK, %d mismatch/missing" % (len(ROWS), len(ROWS) - len(bad), len(bad)))
    if UNBACKED:
        print()
        print("claims with no machine-readable record (verify by hand):")
        for u in UNBACKED:
            print("  -", u)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
