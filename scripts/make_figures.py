# -*- coding: utf-8 -*-
"""Manuscript figures, final geometry.

Design rules applied here so that LaTeX never rescales a panel:
  * every panel is authored at its printed size (inches) and included at that
    size, so the figure font is exactly the 7 pt below and is identical
    across all panels of the paper;
  * one semantic palette for the whole manuscript
    (blue = ensemble, red = evidential, teal = CVAE, dark = error, gray = reference);
  * the severity axis points the same way in every figure: S1 (heaviest
    degradation) on the left, S4 (lightest) on the right;
  * no top/right spines, no legend frame, Type-42 outlines (vector-safe PDF).

Figures produced (print order = file order)
  fig1_teaser         four-panel teaser (rendered separately, UIEB #634)
  fig2_protocol       the audit protocol, one full-width row
  fig3a_reliability   reliability diagram, in-domain marginal coverage
  fig3b_fewshot       few-shot recalibration on the target water body
  fig4_pixel_response (a) pooled decile response  (b) per-image rho distribution
  fig5_deadzone       (a) stratified coverage  (b,c) risk-coverage
  fig6_response       degradation response profiles
  fig7_evidence       NIG evidence parameters against severity

Data: outputs/*.json (audit records) + outputs/arrays_{EUVP,UIEB}.npz
      (per-pixel sigma and error, 64x64 x 3 channels).
Outputs: outputs/figures/*.{pdf,png} mirrored into latex/figures/.
"""
import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
TEXFIG = os.path.join(ROOT, "latex", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEXFIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
BLUE, RED, TEAL, DARK, GRAY = "#3775BA", "#B64342", "#42949E", "#272727", "#767676"
PALE_RED = "#E9A6A1"

cs = json.load(open(os.path.join(ROOT, "outputs", "coverage_stratified.json"), encoding="utf-8"))
fs = json.load(open(os.path.join(ROOT, "outputs", "fewshot_recal_uieb.json"), encoding="utf-8"))
rj = json.load(open(os.path.join(ROOT, "outputs", "rejection_stage4.json"), encoding="utf-8"))
pe = json.load(open(os.path.join(ROOT, "outputs", "puie_eval.json"), encoding="utf-8"))
hiv = json.load(open(os.path.join(ROOT, "outputs", "head_intervention.json"), encoding="utf-8"))
dE = np.load(os.path.join(ROOT, "outputs", "arrays_EUVP.npz"))
dU = np.load(os.path.join(ROOT, "outputs", "arrays_UIEB.npz"))


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        p = os.path.join(OUT, name + "." + ext)
        fig.savefig(p, bbox_inches="tight", pad_inches=0.02, **kw)
        shutil.copyfile(p, os.path.join(TEXFIG, name + "." + ext))
    plt.close(fig)
    import fitz
    r = fitz.open(os.path.join(OUT, name + ".pdf"))[0].rect
    print("  saved %-22s %6.1f x %6.1f pt  (%.2f x %.2f in)"
          % (name, r.width, r.height, r.width / 72.0, r.height / 72.0))


# =====================================================================
# Fig. 2 -- the audit protocol.
# The paper had no method schematic: Sec. III is four subsections of prose
# and Fig. 1 is a teaser, so a reader had to assemble the flow in his head.
# One row reads left to right, objects -> strata -> metrics -> operations,
# and the bar underneath is the rule the audit outputs.
# Geometry is in fixed data units (0-100) so nothing reflows when the font
# is substituted; LaTeX includes the result 1:1.
# =====================================================================
FC, EC, TC = "#F4F6F8", "#8A93A0", "#333333"


def stage(ax, x0, x1, y0, y1, title, lines, ts=7.0, bs=6.2, dy=7.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                fc=FC, ec=EC, lw=0.7, zorder=2))
    cx = 0.5 * (x0 + x1)
    ax.text(cx, y1 - 4.6, title, ha="center", va="top", fontsize=ts,
            fontweight="bold", color="#1B1F24", zorder=3)
    y = y1 - 4.6 - dy
    for ln in lines:
        ax.text(cx, y, ln, ha="center", va="top", fontsize=bs, color=TC, zorder=3)
        y -= dy


fig = plt.figure(figsize=(7.06, 1.80))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

Y0, Y1, YM = 38.0, 92.0, 0.5 * (38.0 + 92.0)
X = [(0.6, 19.6), (22.4, 41.4), (44.2, 76.2), (79.0, 99.4)]

stage(ax, X[0][0], X[0][1], Y0, Y1, "Audited objects", [
    "per-pixel $\\sigma$ and $|e|$",
    "4 released",
    "mechanisms, 2",
    "backbones",
])
stage(ax, X[1][0], X[1][1], Y0, Y1, "Severity strata", [
    "input-only proxy:",
    "grayscale contrast,",
    "quartiles",
    "S1 (heaviest)",
    "to S4 (lightest)",
])
stage(ax, X[2][0], X[2][1], Y0, Y1, "Three metric families", [
    "Coverage (magnitude):",
    "is $|e| \\leq \\sigma$ as often as claimed?",
    "Ranking (order): are the high-$\\sigma$",
    "pixels the erring ones?",
    "Utility: error removed at retention",
])
stage(ax, X[3][0], X[3][1], Y0, Y1, "Two operations", [
    "Recalibrate: one",
    "scalar $r$ from ~100 refs",
    "Reject: drop the",
    "most-uncertain pixels",
])
for i in range(3):
    ax.annotate("", xy=(X[i + 1][0] - 0.2, YM), xytext=(X[i][1] + 0.2, YM),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=EC,
                                shrinkA=0, shrinkB=0, mutation_scale=7), zorder=1)
ax.annotate("", xy=(89.2, 26.0), xytext=(89.2, Y0 - 0.4),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=EC,
                            shrinkA=0, shrinkB=0, mutation_scale=7), zorder=1)
ax.add_patch(FancyBboxPatch((0.6, 3.0), 98.8, 22.0,
                            boxstyle="round,pad=0,rounding_size=1.6",
                            fc="#E7EAEE", ec=EC, lw=0.7, zorder=2))
ax.text(50.0, 14.0, "Deployment rule: recalibrate per water body, trust rejection "
                    "only to ensembles, never select by marginal coverage",
        ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1B1F24", zorder=3)
save(fig, "fig2_protocol")

# =====================================================================
# Fig. 3(a) -- reliability diagram, in-domain (EUVP test-500).
# The old grouped bar chart could only draw the 1-sigma nominal line and
# repeated Table II; a reliability curve uses the per-pixel arrays to show
# the whole nominal range at once, and the diagonal makes the gap readable.
# =====================================================================
lam = np.linspace(0.76, 3.5, 80)
nominal = 2.0 * norm.cdf(lam) - 1.0


def coverage_curve(err, sig):
    z = np.abs(err) / np.maximum(sig, 1e-12)
    return np.array([(z <= l).mean() for l in lam])


fig, ax = plt.subplots(figsize=(3.85, 1.85))
ax.plot([0, 1], [0, 1], ls="--", lw=0.9, color=GRAY, zorder=1,
        label="ideal")
ax.plot(nominal, coverage_curve(dE["err_edl"], dE["sig_edl"]), color=RED, lw=1.5,
        label="EDL")
ax.plot(nominal, coverage_curve(dE["err_ens"], dE["sig_ens"]), color=BLUE, lw=1.5,
        label="Ensemble")
nom_p = [0.683, 0.954, 0.997]
cov_p = [pe["euvp_val"]["z_raw"]["1sigma"], pe["euvp_val"]["z_raw"]["2sigma"],
         pe["euvp_val"]["z_raw"]["3sigma"]]
ax.plot(nom_p, cov_p, color=TEAL, lw=1.2, ls="-.", marker="D", ms=3,
        label="PUIE")
for l in (0.683, 0.954, 0.997):
    ax.axvline(l, color="0.88", lw=0.6, zorder=0)
# post-recalibration 1-sigma points sit on the diagonal
ax.plot([0.683, 0.683],
        [cs["EUVP_域内"]["edl"]["marginal"]["cov1_recal"],
         cs["EUVP_域内"]["ensemble"]["marginal"]["cov1_recal"]],
        ls="none", marker="o", ms=3.6, mfc="white", mec="0.35", mew=0.9, zorder=5)
ax.annotate("", xy=(0.683, 0.240), xytext=(0.683, 0.612),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35", shrinkA=0, shrinkB=0))
ax.text(0.665, 0.43, "gap 45 pts", fontsize=6.2, color="0.25", rotation=90,
        va="center", ha="right")
ax.annotate("after the scalar fix", xy=(0.687, 0.664), xytext=(0.74, 0.42),
            fontsize=6.2, color="0.35", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="0.55",
                            connectionstyle="arc3,rad=0.25"))
ax.set_xlim(0.55, 1.02)
ax.set_ylim(0.0, 1.0)
ax.set_xticks(nom_p)
ax.set_xticklabels(["1$\\sigma$", "2$\\sigma$", "3$\\sigma$"])
ax.set_xlabel("nominal confidence level")
ax.set_ylabel("empirical marginal coverage")
ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.5, labelspacing=0.25,
          ncol=2, columnspacing=1.0, bbox_to_anchor=(-0.02, 1.04))
save(fig, "fig3a_reliability")

# =====================================================================
# Fig. 3(b) -- few-shot recalibration (log N, mean +/- s.d. over 30 draws).
# =====================================================================
fig, ax = plt.subplots(figsize=(3.90, 1.85))
ax.axvspan(4.0, 20.0, color="0.93", zorder=0)
ax.text(9.0, 0.096, "under-determined", fontsize=6, color="0.35", ha="center", va="top")
for key, lab, c in [("edl", "EDL", RED), ("ensemble", "Ensemble", BLUE)]:
    Ns = sorted(int(k) for k in fs[key]["sweep"])
    m = [fs[key]["sweep"][str(n)]["rms_mean"] for n in Ns]
    sd = [fs[key]["sweep"][str(n)]["rms_std"] for n in Ns]
    ax.errorbar(Ns, m, yerr=sd, marker="o", ms=3.2, capsize=2.0, lw=1.3,
                color=c, label=lab, zorder=3)
    ax.axhline(fs[key]["reference"]["full_rms"], ls=":", lw=0.9, color=c, alpha=0.75)
    ax.plot([], [], ls=":", lw=0.9, color=c, alpha=0.75,
            label="%s, full calib." % ("EDL" if key == "edl" else "Ens."))
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100, 200, 445])
ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "445"], fontsize=6)
ax.set_xlim(4.0, 520)
ax.set_ylim(0.02, 0.10)
ax.set_xlabel("$N$ paired references in the target water body")
ax.set_ylabel("RMS-CE (held-out test)")
ax.legend(loc="upper right", handlelength=1.6, labelspacing=0.28, ncol=1)
save(fig, "fig3b_fewshot")

# =====================================================================
# Fig. 4 -- the dead zone at pixel level.
# (a) pooled decile response: colour = paradigm, marker/linestyle = domain,
#     cross-domain drawn heavier so the headline pair reads first.
# (b) per-image rho distribution; medians marked horizontally.
# =====================================================================
def pooled_response(dd):
    e = dd["err_edl"].ravel()
    se = dd["sig_edl"].ravel()
    sn = dd["sig_ens"].ravel()
    edges = np.quantile(e, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(e, edges) - 1, 0, 9)
    edl = np.array([se[idx == k].mean() for k in range(10)])
    ens = np.array([sn[idx == k].mean() for k in range(10)])
    return edl / edl[0], ens / ens[0]


def spatial_rho(sig, err):
    out = []
    for i in range(sig.shape[0]):
        s = sig[i].mean(0).ravel()
        e = err[i].mean(0).ravel()
        if s.std() < 1e-9 or e.std() < 1e-9:
            continue
        out.append(np.corrcoef(s, e)[0, 1])
    return np.array(out)


edlU, ensU = pooled_response(dU)
edlE, ensE = pooled_response(dE)
xs = np.arange(1, 11)

fig, axes = plt.subplots(1, 2, figsize=(7.12, 1.80))
ax = axes[0]
ax.plot(xs, edlU, color=RED, marker="o", ms=3.0, lw=1.7, label="EDL, cross-domain")
ax.plot(xs, ensU, color=BLUE, marker="o", ms=3.0, lw=1.7, label="Ensemble, cross-domain")
ax.plot(xs, edlE, color=RED, marker="^", ms=2.8, lw=1.1, ls="--", alpha=0.6,
        label="EDL, in-domain")
ax.plot(xs, ensE, color=BLUE, marker="^", ms=2.8, lw=1.1, ls="--", alpha=0.6,
        label="Ensemble, in-domain")
ax.annotate("$\\times$1.45", xy=(9.9, ensU[-1]), fontsize=6.4, color=BLUE,
            ha="right", va="bottom", fontweight="bold")
ax.annotate("$\\times$1.09", xy=(9.9, edlU[-1]), fontsize=6.4, color=RED,
            ha="right", va="top", fontweight="bold")
ax.set_xticks(xs)
ax.set_xlabel("true-error decile of pooled pixels (lowest $\\rightarrow$ highest)")
ax.set_ylabel("$\\sigma$ relative to lowest decile")
ax.set_ylim(0.9, 1.72)
ax.legend(loc="upper left", handlelength=1.7, labelspacing=0.25, fontsize=6)
ax.text(-0.14, 1.05, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

ax = axes[1]
rU_e = spatial_rho(dU["sig_edl"], dU["err_edl"])
rU_n = spatial_rho(dU["sig_ens"], dU["err_ens"])
bins = np.linspace(-0.62, 0.92, 32)
me, mn = np.median(rU_e), np.median(rU_n)
ax.hist(rU_e, bins=bins, color=RED, alpha=0.55,
        label="EDL (median $\\rho=%+.2f$)" % me)
ax.hist(rU_n, bins=bins, color=BLUE, alpha=0.55,
        label="Ensemble (median $\\rho=%+.2f$)" % mn)
ax.axvline(me, color=RED, lw=1.0, ls=":")
ax.axvline(mn, color=BLUE, lw=1.0, ls=":")
ax.set_xlabel("per-image spatial correlation, $\\rho(\\sigma,|e|)$ (UIEB)")
ax.set_ylabel("images")
ax.set_xlim(-0.62, 0.92)
ax.set_ylim(0, 44)
ax.legend(loc="upper left", handlelength=1.5, fontsize=6.5, labelspacing=0.3)
ax.text(-0.16, 1.05, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
plt.tight_layout(w_pad=1.5)
save(fig, "fig4_pixel_response")

# =====================================================================
# Fig. 5 -- the dead zone, cross-domain (UIEB), one full-width row.
# Merged into a single float: three panels across \textwidth keep every label
# at the authored 7 pt, whereas two panels side by side in half-width
# minipages forced a 2.4x downscale and left the legends at ~3 pt.
# =====================================================================
S = ["S1\n(heavy)", "S2", "S3", "S4\n(light)"]
st_edl = cs["UIEB_跨域"]["edl"]
st_ens = cs["UIEB_跨域"]["ensemble"]
x = np.arange(4)
fig = plt.figure(figsize=(6.95, 2.02))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.30,
                      left=0.058, right=0.995, top=0.90, bottom=0.185)

ax = fig.add_subplot(gs[0, 0])
raw = [s["cov1"] for s in st_edl["strata_raw"]]
re_ = [s["cov1"] for s in st_edl["strata_recal"]]
rn = [s["cov1"] for s in st_ens["strata_recal"]]
sem_edl = [s["cov1_std"] / np.sqrt(s["n"]) for s in st_edl["strata_recal"]]
sem_ens = [s["cov1_std"] / np.sqrt(s["n"]) for s in st_ens["strata_recal"]]
ax.axhline(0.683, ls="--", lw=0.9, color=GRAY, label="nominal")
ax.plot(x, raw, color=PALE_RED, ls="--", lw=1.3, marker="v", ms=3.2,
        label="EDL, raw")
ax.errorbar(x, re_, yerr=sem_edl, color=RED, lw=1.7, marker="o", ms=3.6,
            capsize=1.8, label="EDL, rec.")
ax.errorbar(x, rn, yerr=sem_ens, color=BLUE, lw=1.7, marker="s", ms=3.6,
            capsize=1.8, label="Ens., rec.")
ax.annotate("", xy=(3.30, min(re_)), xytext=(3.30, max(re_)),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=RED, shrinkA=0, shrinkB=0))
ax.annotate("", xy=(3.62, min(rn)), xytext=(3.62, max(rn)),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color=BLUE, shrinkA=0, shrinkB=0))
ax.text(2.95, 0.335, "EDL range $0.25$", fontsize=6.2, color=RED, ha="left", va="center")
ax.text(2.95, 0.245, "Ens. range $0.12$", fontsize=6.2, color=BLUE, ha="left", va="center")
ax.set_xticks(x)
ax.set_xticklabels(S)
ax.set_xlim(-0.45, 4.25)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("input-contrast stratum\n(heaviest $\\rightarrow$ lightest)")
ax.set_ylabel("stratified 1$\\sigma$ coverage")
ax.legend(loc="upper left", ncol=2, handlelength=1.4, handletextpad=0.4,
          labelspacing=0.25, columnspacing=0.9, borderaxespad=0.15)
ax.text(-0.24, 1.06, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")

for j, (ds, title) in enumerate(zip(["EUVP_域内", "UIEB_跨域"],
                                    ["EUVP (in-domain)", "UIEB (cross-domain)"])):
    ax = fig.add_subplot(gs[0, j + 1])
    for m, lab, c in [("edl", "EDL", RED), ("ensemble", "Ensemble", BLUE)]:
        d = rj[ds][m]
        ax.plot(np.array(d["curve_cov"]) * 100, np.array(d["curve_risk"]) * 100,
                label="%s (AURC %.3f)" % (lab, d["aurc"]), color=c, lw=1.5)
    base = rj[ds]["edl"]["risk_at_100cov"] * 100
    ax.axhline(base, ls=":", lw=0.9, color=GRAY, label="no rejection")
    ax.set_xlabel("retention (%)")
    if j == 0:
        ax.set_ylabel("risk = pixel MAE (%)")
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.legend(loc="best", handlelength=1.5, labelspacing=0.25, borderaxespad=0.2)
    ax.invert_xaxis()
    ax.text(-0.20, 1.06, "bc"[j], transform=ax.transAxes, fontsize=9, fontweight="bold")
save(fig, "fig5_deadzone")

# =====================================================================
# Fig. 6 -- degradation response profiles.  Solid = cross-domain (UIEB),
# pale dashed = in-domain (EUVP) reference, one legend entry for the family.
# The severity axis runs S1 (heaviest, left) -> S4 (lightest, right), as in
# Fig. 5: profile() bins at the contrast quartiles, so index 0 is S1.
# =====================================================================
def profile(d):
    c = d["contrast"]
    q = np.quantile(c, [0.25, 0.5, 0.75])
    b = np.digitize(c, q)
    e, se, sn = [], [], []
    for s in range(4):
        m = b == s
        e.append(d["err_edl"][m].mean())
        se.append(d["sig_edl"][m].mean())
        sn.append(d["sig_ens"][m].mean())
    return e, se, sn


eU, seU, snU = profile(dU)          # index 0 = S1 (heaviest), 3 = S4 (lightest)
eE, seE, snE = profile(dE)
xs4 = np.arange(4)
fig, ax = plt.subplots(figsize=(3.87, 2.00))
ax.plot(xs4, [v / eU[3] for v in eU], color=DARK, marker="o", ms=3, lw=1.5, label="error")
ax.plot(xs4, [v / snU[3] for v in snU], color=BLUE, marker="s", ms=3, lw=1.5,
        label="Ensemble $\\sigma$")
ax.plot(xs4, [v / seU[3] for v in seU], color=RED, marker="^", ms=3, lw=1.5, label="EDL $\\sigma$")
ax.plot(xs4, [v / eE[3] for v in eE], color=DARK, lw=1.0, ls="--", alpha=0.7)
ax.plot(xs4, [v / snE[3] for v in snE], color=BLUE, lw=1.0, ls="--", alpha=0.7)
ax.plot(xs4, [v / seE[3] for v in seE], color=RED, lw=1.0, ls="--", alpha=0.7)
ax.plot([], [], color="0.35", lw=1.0, ls="--", label="in-domain reference")
ax.axhline(1.0, color="0.8", lw=0.6)
ax.annotate("$\\times$1.73", xy=(0.05, 1.735), fontsize=6.4, ha="left", va="bottom", color=DARK)
ax.annotate("$\\times$1.56", xy=(0.05, 1.555), fontsize=6.4, ha="left", va="bottom", color=BLUE)
ax.annotate("flat ($\\times$0.96)", xy=(1.62, 0.955), xytext=(1.35, 0.795), fontsize=6.4,
            color=RED, ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", lw=0.7, color=RED,
                            connectionstyle="arc3,rad=-0.25"))
ax.set_xticks(xs4)
ax.set_xticklabels(["S1", "S2", "S3", "S4"])
ax.set_xlim(-0.25, 3.25)
ax.set_xlabel("input-contrast quartile   (heavier degradation $\\leftarrow$)")
ax.set_ylabel("value relative to lightest quartile")
ax.set_ylim(0.72, 1.90)
ax.legend(loc="upper right", handlelength=1.7, labelspacing=0.26, borderaxespad=0.2)
save(fig, "fig6_response")

# =====================================================================
# Fig. 7 -- where the blindness sits: the NIG evidence parameters.
# Same severity axis and same normalization as Fig. 6, so the two panels read
# as one argument: the error grows 73% while alpha and beta move by a few
# percent and, crucially, not monotonically in severity -- the residual
# variation cannot encode severity.  Only point estimates exist in the
# records (stratum means, n ~ 111), so no interval is drawn here.
# =====================================================================
hs = hiv["baseline"]["strata"]           # S1 (heaviest) .. S4 (lightest)
al = np.array([s["alpha"] for s in hs])
be = np.array([s["beta"] for s in hs])
er = np.array([s["err"] for s in hs])
def _span(v):
    return 100.0 * (v.max() - v.min()) / v.mean()


print("  fig7 relativised: alpha %s  (span %.1f%%)" % (np.round(al / al[3], 3), _span(al / al[3])))
print("  fig7 relativised: beta  %s  (span %.1f%%)" % (np.round(be / be[3], 3), _span(be / be[3])))
print("  fig7 relativised: error %s  (span %.1f%%)" % (np.round(er / er[3], 3), _span(er / er[3])))

fig, ax = plt.subplots(figsize=(3.87, 1.85))
ax.axhspan(0.95, 1.05, color="0.94", zorder=0)
ax.text(2.05, 1.052, "5% band", fontsize=6, color="0.45", ha="center", va="bottom")
ax.plot(xs4, er / er[3], color=DARK, marker="o", ms=3.2, lw=1.5, label="error")
ax.plot(xs4, al / al[3], color=RED, marker="s", ms=3.2, lw=1.4, label="$\\alpha$ (evidence)")
ax.plot(xs4, be / be[3], color=RED, marker="v", ms=3.2, lw=1.2, ls="--",
        mfc="white", label="$\\beta$ (evidence)")
ax.axhline(1.0, color="0.8", lw=0.6, zorder=0)
ax.annotate("$\\times$1.73", xy=(0.06, 1.742), fontsize=6.4, ha="left", va="bottom",
            color=DARK, fontweight="bold")
ax.annotate("$\\alpha$, $\\beta$: flat and not\nmonotone in severity",
            xy=(2.0, 1.028), xytext=(1.28, 1.155), fontsize=6.4, color=RED,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=RED,
                            connectionstyle="arc3,rad=-0.2"))
ax.set_xticks(xs4)
ax.set_xticklabels(["S1", "S2", "S3", "S4"])
ax.set_xlim(-0.25, 3.25)
ax.set_ylim(0.90, 1.85)
ax.set_xlabel("input-contrast quartile   (heavier degradation $\\leftarrow$)")
ax.set_ylabel("value relative to lightest quartile")
ax.legend(loc="upper right", handlelength=1.7, labelspacing=0.26, borderaxespad=0.2)
save(fig, "fig7_evidence")

print("figures written to", OUT, "and mirrored to", TEXFIG)
