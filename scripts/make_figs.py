# -*- coding: utf-8 -*-
"""Generate Fig.2 / Fig.3 / Fig.4 for the UQ audit paper.
Style: figure-design-expert (Nature-ish): Arial/DejaVu, 8pt, no top/right spines,
consistent paradigm colors across all figures, PDF+PNG 300dpi.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
})
C = {"edl": "#B64342", "ens": "#0F4D92", "puie": "#42949E",
     "nominal": "#4D4D4D", "raw": "#767676"}
NOM = 0.683

def load(p): return json.load(open(os.path.join(ROOT, "outputs", p), encoding="utf-8"))
cov = load("coverage_stratified.json")
few = load("fewshot_recal_uieb.json")
rej = load("rejection_stage4.json")
puie = load("puie_stratified.json")

def svals(entry, key, field):
    return [s[field] for s in entry[key]]

def panel_label(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

# ================= Fig.2 : in-domain calibration audit =================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
ax = axes[0]
paradigms = ["EDL", "Ensemble", "PUIE\n(CVAE)"]
marg = [cov["EUVP_域内"]["edl"]["marginal"]["cov1"],
        cov["EUVP_域内"]["ensemble"]["marginal"]["cov1"],
        puie["EUVP_域内"]["marginal_cov1"]]
bars = ax.bar(paradigms, marg, width=0.55,
              color=[C["edl"], C["ens"], C["puie"]], alpha=0.9)
ax.axhline(NOM, ls="--", lw=1.0, color=C["nominal"])
ax.text(2.45, NOM + 0.015, "nominal 68.3%", ha="right", fontsize=7.5, color=C["nominal"])
for b, v in zip(bars, marg):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_ylabel("marginal 1$\\sigma$ coverage")
ax.set_ylim(0, 0.8)
ax.set_title("In-domain marginal coverage (EUVP)")
panel_label(ax, "a")

ax = axes[1]
eu = cov["EUVP_域内"]
x = range(4); w = 0.19
series = [("edl", "raw", C["edl"], 0.35, "EDL (raw)"),
          ("edl", "recal", C["edl"], 1.0, "EDL (recal.)"),
          ("ensemble", "raw", C["ens"], 0.35, "Ensemble (raw)"),
          ("ensemble", "recal", C["ens"], 1.0, "Ensemble (recal.)")]
for j, (m, state, col, al, lab) in enumerate(series):
    key = "strata_raw" if state == "raw" else "strata_recal"
    vals = svals(eu[m], key, "cov1")
    pos = [xi + (j - 1.5) * w for xi in x]
    ax.bar(pos, vals, width=w * 0.92, color=col, alpha=al, label=lab)
ax.axhline(NOM, ls="--", lw=1.0, color=C["nominal"])
ax.set_xticks(list(x)); ax.set_xticklabels(["S1\n(heavy)", "S2", "S3", "S4\n(mild)"])
ax.set_ylabel("stratified 1$\\sigma$ coverage")
ax.set_ylim(0, 0.85)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, handlelength=1.0,
          columnspacing=0.9, fontsize=7)
ax.set_title("In-domain stratified coverage: flat before and after")
panel_label(ax, "b")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_calibration_audit.pdf"), bbox_inches="tight", pad_inches=0.05)
fig.savefig(os.path.join(OUT, "fig2_calibration_audit.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ================= Fig.3 : few-shot recalibration =================
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for m, col, lab in [("edl", C["edl"], "EDL"), ("ensemble", C["ens"], "Ensemble")]:
    ref = few[m]["reference"]; sw = few[m]["sweep"]
    ns = sorted(int(k) for k in sw)
    rms = [sw[str(n)]["rms_mean"] for n in ns]
    std = [sw[str(n)]["rms_std"] for n in ns]
    ax.errorbar(ns, rms, yerr=std, marker="o", ms=3.5, lw=1.2, color=col, label=lab,
                capsize=2, elinewidth=0.8)
    ax.axhline(ref["full_rms"], ls=":", lw=0.9, color=col, alpha=0.7)
ax.axhline(ref["raw_rms"], ls="--", lw=1.0, color=C["raw"])
ax.set_ylim(0, 0.52)
ax.text(6, ref["raw_rms"] + 0.012, "no recalibration", fontsize=7, color=C["raw"], va="bottom")
ax.text(6, few["edl"]["reference"]["full_rms"] + 0.008, "full calibration set (445)", fontsize=7, color=C["edl"], va="bottom")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 50, 100, 200, 445])
ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "445"])
ax.minorticks_off()
ax.set_xlabel("number of target-domain reference images $N$")
ax.set_ylabel("RMS-CE after recalibration")
ax.set_title("Few-shot recalibration on UIEB (cross-domain)")
ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_fewshot.pdf"), bbox_inches="tight", pad_inches=0.05)
fig.savefig(os.path.join(OUT, "fig3_fewshot.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ================= Fig.4 : the sigma dead zone (4 panels) =================
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
ui = cov["UIEB_跨域"]
strata = ["S1\n(heavy)", "S2", "S3", "S4\n(mild)"]

ax = axes[0]
for m, col, lab in [("edl", C["edl"], "EDL"), ("ensemble", C["ens"], "Ensemble")]:
    ax.plot(range(4), svals(ui[m], "strata_raw", "err_mean"), marker="o", ms=3.5,
            lw=1.2, color=col, label=lab)
ax.set_ylabel("mean per-pixel error")
ax.set_xticks(range(4)); ax.set_xticklabels(strata)
ax.set_title("Error rises with severity")
ax.legend(loc="upper right")
panel_label(ax, "a")

ax = axes[1]
for m, col, lab in [("edl", C["edl"], "EDL"), ("ensemble", C["ens"], "Ensemble")]:
    ax.plot(range(4), svals(ui[m], "strata_raw", "sigma_mean"), marker="s", ms=3.5,
            lw=1.2, color=col, label=lab)
ax.set_ylabel("mean $\\sigma$")
ax.set_xticks(range(4)); ax.set_xticklabels(strata)
ax.set_title("EDL $\\sigma$ stays flat")
panel_label(ax, "b")

ax = axes[2]
x = range(4); w = 0.36
for i, (m, col, lab) in enumerate([("edl", C["edl"], "EDL"), ("ensemble", C["ens"], "Ensemble")]):
    rec = svals(ui[m], "strata_recal", "cov1")
    ax.bar([xi + (i-0.5)*w for xi in x], rec, width=w, color=col, alpha=0.9, label=lab)
ax.axhline(NOM, ls="--", lw=1.0, color=C["nominal"])
ax.text(3.45, NOM + 0.012, "nominal", ha="right", fontsize=7, color=C["nominal"])
ax.set_xticks(list(x)); ax.set_xticklabels(strata)
ax.set_ylabel("recalibrated 1$\\sigma$ coverage")
ax.set_ylim(0, 0.95)
ax.set_title("Recalibration fixes the average")
panel_label(ax, "c")

ax = axes[3]
for m, col, lab in [("edl", C["edl"], "EDL"), ("ensemble", C["ens"], "Ensemble")]:
    cc = rej["UIEB_跨域"][m]["curve_cov"]
    cr = rej["UIEB_跨域"][m]["curve_risk"]
    cc, cr = cc[3:], cr[3:]  # skip the noisy first points at coverage <= 0.02
    ax.plot(cc, cr, lw=1.3, color=col, label=lab)
ax.set_xlabel("coverage (fraction retained)")
ax.set_ylabel("risk (MAE of retained)")
ax.set_title("Rejection: EDL cannot rank")
ax.legend(loc="lower right")
panel_label(ax, "d")
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(OUT, "fig4_dead_zone.pdf"), bbox_inches="tight", pad_inches=0.05)
fig.savefig(os.path.join(OUT, "fig4_dead_zone.png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print("FIGURES_DONE ->", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f, os.path.getsize(os.path.join(OUT, f)))
