import json

B = "/mnt/data/underwater/bpinn_uq/outputs/"
edl_eu = json.load(open(B + "edl_recalibration.json"))
ens_eu = json.load(open(B + "ensemble_recalibration.json"))
cd = json.load(open(B + "crossdomain_uieb.json"))

def line(tag, m_raw, m_rec, z_raw, z_rec, mae):
    print("%-28s rms=%.4f→%.4f  1σ=%.3f→%.3f  3σ=%.3f→%.3f  NLL=%.2f→%.2f  MAE=%.4f"
          % (tag, m_raw["rms_cal"], m_rec["rms_cal"], z_raw["1sigma"], z_rec["1sigma"],
             z_raw["3sigma"], z_rec["3sigma"], m_raw["nll"], m_rec["nll"], mae))

print("=" * 120)
print("【A】域内（EUVP cal500 学 ratio → EUVP test500 验证）")
line("EDL  (ratio=%.3f)" % edl_eu["ratio"], edl_eu["test_raw"], edl_eu["test_recal"],
     edl_eu["z_coverage_raw"], edl_eu["z_coverage_recal"], edl_eu["test_raw"]["mae"])
line("ENS  (ratio=%.3f)" % ens_eu["ratio"], ens_eu["test_raw"], ens_eu["test_recal"],
     ens_eu["z_coverage_raw"], ens_eu["z_coverage_recal"], ens_eu["test_raw"]["mae"])

print()
print("【B】跨域（EUVP 训练 → UIEB 890 对；UIEB cal445 学 ratio → UIEB test445 验证）")
for k, nm in [("edl", "EDL"), ("ensemble", "ENS")]:
    d = cd[k]
    print("-- %s : 域内(EUVP) ratio=%.3f  vs  UIEB 域内 ratio=%.3f   (膨胀 %.2f×) ; meanσ(UIEB)=%.4f"
          % (nm, cd["eu_ratio"][k], d["ratio_domain"],
             d["ratio_domain"] / cd["eu_ratio"][k], d["mean_sigma_uieb"]))
    line("   %s UIEB 原始" % nm, d["raw"], d["raw"], d["z_raw"], d["z_raw"], d["raw"]["mae"])
    line("   %s ×EUVP ratio(直搬)" % nm, d["raw"], d["xfer"], d["z_raw"], d["z_xfer"], d["raw"]["mae"])
    line("   %s ×UIEB ratio(上界)" % nm, d["raw"], d["domain"], d["z_raw"], d["z_domain"], d["raw"]["mae"])
print("=" * 120)
