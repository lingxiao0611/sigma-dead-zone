# -*- coding: utf-8 -*-
"""Join the three transfer orientations into one record.

The paper's transfer axis trains on EUVP and audits on UIEB. Three runs share the
same metric code, granularity and seed, and differ only in which water body plays
which role and how much source data is on offer:

  euvp_src   EUVP (10,435 pairs) -> UIEB     the paper's direction, recomputed here
  euvp700    EUVP (700 pairs)    -> UIEB     matched-budget control
  uieb_src   UIEB (700 pairs)    -> EUVP     the reciprocal orientation

euvp_src reproduces the published numbers exactly, which is what licenses reading
its two companions. The reciprocal run is what lets the paper say the inflation is
a property of the shift rather than of one orientation; the matched-budget run is
what bounds the ensemble's advantage to source-data diversity.

    python code/build_transfer_directions.py        # reads outputs/, writes outputs/

Writes outputs/transfer_directions.json.
"""
import json
import os
import sys

OUT_DIR = os.environ.get("OUT_DIR", "outputs")

RUNS = [
    ("euvp_src", "paper direction, full source budget"),
    ("euvp700", "paper direction, source budget matched to UIEB's 700 pairs"),
    ("uieb_src", "reciprocal orientation, UIEB's full paired budget"),
]


def one(tag, note, rec):
    """Slice one run down to what the text quotes."""
    src, tgt = rec["source"], rec["target"]
    ind = rec["%s_in_domain" % src]
    crs = rec["%s_cross_domain" % tgt]
    arms = {}
    for m in ("edl", "ensemble"):
        arms[m] = {
            "ratio_in_domain": ind[m]["ratio"],
            "ratio_cross_domain": crs[m]["ratio_target"],
            "inflation": crs[m]["inflation"],
            "aurc_in_domain": ind[m]["raw"]["aurc"],
            "aurc_cross_domain": crs[m]["raw"]["aurc"],
        }
    return {
        "note": note,
        "source": src,
        "target": tgt,
        "protocol": rec["protocol"],
        "arms": arms,
        "ensemble_advantage_cross_domain": (
            (arms["edl"]["aurc_cross_domain"] - arms["ensemble"]["aurc_cross_domain"])
            / arms["edl"]["aurc_cross_domain"]),
        "ensemble_leads_cross_domain": (
            arms["ensemble"]["aurc_cross_domain"] < arms["edl"]["aurc_cross_domain"]),
    }


def main():
    out = {
        "table": ("Sec. V-D transfer axis, both orientations and a matched-budget "
                  "control; one estimator, one seed, one metric implementation"),
        "runs": {},
    }
    for tag, note in RUNS:
        with open(os.path.join(OUT_DIR, "transfer_%s.json" % tag)) as fh:
            out["runs"][tag] = one(tag, note, json.load(fh))

    # The reciprocal orientation must reproduce the inflation, and the matched-budget
    # control must not, or the text in Sec. V-D would have to be rewritten.
    out["inflation_range"] = {
        "edl": [min(r["arms"]["edl"]["inflation"] for r in out["runs"].values()),
                max(r["arms"]["edl"]["inflation"] for r in out["runs"].values())],
        "ensemble": [min(r["arms"]["ensemble"]["inflation"] for r in out["runs"].values()),
                     max(r["arms"]["ensemble"]["inflation"] for r in out["runs"].values())],
    }
    out["all_orientations_inflate"] = all(
        r["arms"][m]["inflation"] > 1.0 for r in out["runs"].values()
        for m in ("edl", "ensemble"))

    path = os.path.join(OUT_DIR, "transfer_directions.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print("%-10s %-5s %-5s %9s %9s %9s %9s  leads" %
          ("run", "src", "tgt", "EDL inf", "ENS inf", "EDL xAURC", "ENS xAURC"))
    for tag, r in out["runs"].items():
        print("%-10s %-5s %-5s %9.3f %9.3f %9.4f %9.4f  %s" %
              (tag, r["source"], r["target"], r["arms"]["edl"]["inflation"],
               r["arms"]["ensemble"]["inflation"],
               r["arms"]["edl"]["aurc_cross_domain"],
               r["arms"]["ensemble"]["aurc_cross_domain"],
               "ENS" if r["ensemble_leads_cross_domain"] else "EDL"))
    print("all six arm-orientations inflate: %s" % out["all_orientations_inflate"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
