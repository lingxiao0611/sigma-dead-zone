import json

B = "/mnt/data/underwater/bpinn_uq/outputs/"
r = json.load(open(B + "rejection_stage4.json"))

print("=" * 104)
for ds in ["EUVP_域内", "UIEB_跨域"]:
    print("【%s】" % ds)
    print("  %-10s %9s %9s %10s %10s %10s %10s %10s" %
          ("档", "AURC", "AUGRC", "100%cov", "90%cov", "80%cov", "70%cov", "50%cov"))
    for k, nm in [("edl", "EDL"), ("ensemble", "ENS")]:
        d = r[ds][k]
        print("  %-10s %9.5f %9.5f %10.5f %10.5f %10.5f %10.5f %10.5f" % (
            nm, d["aurc"], d["augrc"], d["risk_at_100cov"], d["risk_at_90cov"],
            d["risk_at_80cov"], d["risk_at_70cov"], d["risk_at_50cov"]))
    e, n = r[ds]["edl"], r[ds]["ensemble"]
    print("  -- 拒识收益（相对不拒识 baseline）--")
    for c in ["90", "80", "70", "50"]:
        ke, kn = "risk_at_%scov" % c, "risk_at_%scov" % c
        print("     保留%s%%: EDL %+.1f%%   ENS %+.1f%%" % (
            c, (e[ke] / e["risk_at_100cov"] - 1) * 100,
            (n[kn] / n["risk_at_100cov"] - 1) * 100))
    print("  -- AURC 差距: ENS 比 EDL 低 %.2f%%" % ((1 - n["aurc"] / e["aurc"]) * 100))
    print("  -- cov@5%%risk: EDL %.3f  ENS %.3f" % (e["cov_at_5risk"], n["cov_at_5risk"]))
    print()

print("=" * 104)
print("【尺度不变性自检】AURC(原始σ) vs AURC(σ×ratio)  —— 期望：仅浮点级差异")
for ds in ["EUVP_域内", "UIEB_跨域"]:
    for k, nm in [("edl", "EDL"), ("ensemble", "ENS")]:
        d = r[ds][k]
        print("  %-10s %-10s  raw=%.10f  recal=%.10f  Δ=%.3e" % (
            ds, nm, d["aurc"], d["aurc_after_recal"],
            abs(d["aurc"] - d["aurc_after_recal"])))
