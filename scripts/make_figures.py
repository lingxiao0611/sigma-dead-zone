# -*- coding: utf-8 -*-
"""Fig.2/3/4 出图（figure-design-expert 规范：Arial、极简边框、统一语义配色、PDF+PNG）。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)
# 规范：字体栈 / 极简边框 / 无框图例
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})
BLUE, RED, TEAL, GRAY = "#3775BA", "#B64342", "#42949E", "#767676"  # 语义: 方法/EDL基线/次要/参考

cs = json.load(open(os.path.join(ROOT, "outputs", "coverage_stratified.json"), encoding="utf-8"))
fs = json.load(open(os.path.join(ROOT, "outputs", "fewshot_recal_uieb.json"), encoding="utf-8"))
rj = json.load(open(os.path.join(ROOT, "outputs", "rejection_stage4.json"), encoding="utf-8"))
pe = json.load(open(os.path.join(ROOT, "outputs", "puie_eval.json"), encoding="utf-8"))

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

# ---------- Fig.2a: marginal 1/2/3-sigma vs nominal (EUVP test-500) ----------
cov = {"EDL": (cs["EUVP_域内"]["edl"]["marginal"]["cov1"], cs["EUVP_域内"]["edl"]["marginal"]["cov2"], cs["EUVP_域内"]["edl"]["marginal"]["cov3"]),
       "Ensemble": (cs["EUVP_域内"]["ensemble"]["marginal"]["cov1"], cs["EUVP_域内"]["ensemble"]["marginal"]["cov2"], cs["EUVP_域内"]["ensemble"]["marginal"]["cov3"]),
       "PUIE (CVAE)": (pe["euvp_val"]["z_raw"]["1sigma"], pe["euvp_val"]["z_raw"]["2sigma"], pe["euvp_val"]["z_raw"]["3sigma"])}
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(5.4, 3.0))
for i, (nm, c) in enumerate(zip(["1$\\sigma$", "2$\\sigma$", "3$\\sigma$"], ["#B4C0E4", "#7884B4", "#484878"])):
    vals = [cov[k][i] for k in cov]
    ax.bar(x + (i - 1) * w, vals, w, label=f"{nm} (nom {['68.3','95.4','99.7'][i]}%)", color=c)
ax.axhline(0.683, ls="--", lw=0.8, color=GRAY)
ax.set_xticks(x); ax.set_xticklabels(list(cov.keys()))
ax.set_ylabel("Marginal coverage"); ax.set_ylim(0, 1.1)
ax.legend(fontsize=7.5, ncol=3)
pass  # title removed (caption carries the message)
save(fig, "fig2a_marginal")

# ---------- Fig.3: few-shot convergence ----------
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for key, lab, c in [("edl", "EDL", RED), ("ensemble", "Ensemble", BLUE)]:
    Ns = sorted(int(k) for k in fs[key]["sweep"])
    m = [fs[key]["sweep"][str(n)]["rms_mean"] for n in Ns]
    sd = [fs[key]["sweep"][str(n)]["rms_std"] for n in Ns]
    ax.errorbar(Ns, m, yerr=sd, marker="o", ms=3.5, capsize=2.5, lw=1.4, color=c, label=lab)
    ax.axhline(fs[key]["reference"]["full_rms"], ls=":", lw=0.9, color=c, alpha=0.6)
ax.set_xscale("log"); ax.set_xlabel("N paired references in target water body")
ax.set_ylabel("RMS-CE (held-out test)")
pass  # title removed (caption carries the message)
ax.legend(fontsize=8)
save(fig, "fig3_fewshot")

# ---------- Fig.4c: post-recal stratified coverage (shape distortion) ----------
fig, ax = plt.subplots(figsize=(4.8, 3.0))
S = ["S1\n(heavy)", "S2", "S3", "S4\n(light)"]
x = np.arange(4); w = 0.34
ax.bar(x - w/2, [s["cov1"] for s in cs["UIEB_跨域"]["edl"]["strata_raw"]], w,
       label="EDL raw", color="#E9A6A1")
ax.bar(x + w/2, [s["cov1"] for s in cs["UIEB_跨域"]["edl"]["strata_recal"]], w,
       label="EDL recalibrated", color=RED)
ax.plot(x, [s["cov1"] for s in cs["UIEB_跨域"]["ensemble"]["strata_recal"]], "o-", lw=1.6,
        ms=5, color=BLUE, label="Ensemble recalibrated")
ax.axhline(0.683, ls="--", lw=0.9, color=GRAY, label="nominal")
ax.set_xticks(x); ax.set_xticklabels(S); ax.set_ylabel("Stratified 1$\\sigma$ coverage")
ax.set_ylim(0, 1.0); ax.legend(fontsize=7.5)
pass  # title removed (caption carries the message)
save(fig, "fig4c_stratified")

# ---------- Fig.4d: risk-coverage curves ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, ds, title in zip(axes, ["EUVP_域内", "UIEB_跨域"], ["EUVP (in-domain)", "UIEB (cross-domain)"]):
    for m, lab, c in [("edl", "EDL", RED), ("ensemble", "Ensemble", BLUE)]:
        d = rj[ds][m]
        ax.plot(np.array(d["curve_cov"]) * 100, np.array(d["curve_risk"]) * 100,
                label=f"{lab} (AURC {d['aurc']*100:.2f})", color=c, lw=1.6)
    base = rj[ds]["edl"]["risk_at_100cov"] * 100
    ax.axhline(base, ls=":", lw=0.9, color=GRAY, label="no rejection")
    ax.set_xlabel("Retention (%)"); ax.set_ylabel("Risk = pixel MAE (%)")
    ax.set_title(title, fontsize=9.5); ax.legend(fontsize=7.5); ax.invert_xaxis()
plt.tight_layout()
save(fig, "fig4d_riskcov")

print("FIGS_DONE:", sorted(os.listdir(OUT)))
