import json
from scipy.stats import spearmanr

B = "/mnt/data/underwater/bpinn_uq/outputs/"
r = json.load(open(B + "sigma_severity.json"))
grades = [1, 2, 3, 4, 5]          # A=1(最好) ... E=5(最差)
se = [r[f"RUIE_UIQS_{g}"]["sigma_edl"] for g in "ABCDE"]
sn = [r[f"RUIE_UIQS_{g}"]["sigma_ens"] for g in "ABCDE"]
print("UIQS A→E (退化加重方向)")
print("  EDL  σ:", [round(x, 5) for x in se], " spearman(σ, 严重度) =", round(spearmanr(se, grades)[0], 3))
print("  ENS  σ:", [round(x, 5) for x in sn], " spearman(σ, 严重度) =", round(spearmanr(sn, grades)[0], 3))

print()
base_e = r["EUVP_val(域内基准)"]["sigma_edl"]
base_n = r["EUVP_val(域内基准)"]["sigma_ens"]
print("域漂移 σ 膨胀（相对 EUVP 域内基准）")
for k in ["UIEB_raw(跨域)", "U45_all", "RUIE_UIQS_A", "RUIE_UIQS_E"]:
    if k in r:
        v = r[k]
        print("  %-20s EDL %+.1f%%   ENS %+.1f%%" % (
            k, (v["sigma_edl"] / base_e - 1) * 100, (v["sigma_ens"] / base_n - 1) * 100))
