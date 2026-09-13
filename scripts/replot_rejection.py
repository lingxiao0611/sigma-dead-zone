# -*- coding: utf-8 -*-
"""从 rejection_stage4.json 重画论文用图（无需重跑推理）。
英文标签（服务器无 CJK 字体）；coverage 截 10~100% 避开单像素噪声端。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = "/mnt/data/underwater/bpinn_uq/outputs/"
r = json.load(open(B + "rejection_stage4.json"))
FIG = B + "rejection_curves.png"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
panels = [("EUVP_域内", "In-domain (EUVP test)"), ("UIEB_跨域", "Cross-domain (UIEB test)")]
for ax, (key, title) in zip(axes, panels):
    for m, lab, c, ls in [("edl", "EDL", "tab:orange", "-"),
                          ("ensemble", "Deep Ensemble", "tab:blue", "-")]:
        d = r[key][m]
        cov = np.array(d["curve_cov"]) * 100
        risk = np.array(d["curve_risk"]) * 100
        msk = cov >= 10.0
        ax.plot(cov[msk], risk[msk], ls, color=c, lw=2.2,
                label="%s  (AURC=%.4f)" % (lab, d["aurc"]))
        r80 = d["risk_at_80cov"] * 100
        ax.plot(80, r80, "o", color=c, ms=6)
        ax.annotate("%.1f" % r80, (80, r80), textcoords="offset points",
                    xytext=(6, -12), fontsize=9, color=c)
    base = r[key]["edl"]["risk_at_100cov"] * 100
    ax.axhline(base, ls=":", c="gray", lw=1.2, label="no rejection (100%% cov, MAE=%.2f)" % base)
    ax.set_xlabel("Coverage (%)", fontsize=11)
    ax.set_ylabel("Risk = MAE on kept pixels (%)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(100, 10)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(FIG, dpi=160)
print("replotted ->", FIG)
print("RE_PLOT_DONE")
