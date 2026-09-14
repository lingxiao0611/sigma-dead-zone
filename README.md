# sigma-dead-zone

Code, protocol, and per-image audit records for

> **The $\sigma$ Dead Zone in Auditing Pixel-wise Uncertainty for Underwater Image Restoration**
> Xiangmiao Yu, Jimei University — *submitted to IEEE TCSVT*

This repository accompanies an **audit** of the uncertainty quantification (UQ) paradigms that
already ship with underwater image restoration methods. It does not propose a new UQ mechanism.
It asks whether the per-pixel uncertainty map $\sigma$ these models emit is calibrated,
correctly ordered, and transferable across water bodies — and it releases the protocol, the
evaluation code, and the raw audit records so that the next $\sigma$ claim in this domain can be
checked rather than believed.

**Headline findings.** Every audited paradigm is overconfident in its home water body, yet a single
scalar restores nominal coverage at zero accuracy cost. Under a change of water body every
correction ratio inflates, and the evidential paradigm loses not only magnitude but *ordering* —
a failure we call the $\sigma$ **dead zone**. Only deep ensembles stay informative on both axes.

---

## What is in this repository

| Path | Contents |
|---|---|
| `scripts/` | Every training and evaluation script used in the paper. |
| `records/` | The JSON audit records behind every number and figure, one file per experiment. |
| `figures/` | The manuscript figures, as vector PDF plus 400 dpi PNG, at the exact size they are printed. File names follow the print order of the paper (`fig2_protocol` is Fig. 2, `fig3a_reliability` and `fig3b_fewshot` are the two panels of Fig. 3, and so on), so a figure can be found from its number without a lookup table. |
| `docs/REPRODUCE.md` | Environment, datasets, upstream models, and the run order. |
| `docs/UPSTREAM_REPOS.md` | The third-party repositories we build on, with what we changed. |
| `docs/TO_FETCH_FROM_SERVER.md` | The few private modules you must copy from your training host. |

**Not included, by design:** datasets (public, see `docs/REPRODUCE.md`), third-party model
implementations (linked upstream, see `docs/UPSTREAM_REPOS.md`), and checkpoints plus the raw
per-pixel arrays (tens of GB — archived separately at `https://doi.org/<ZENODO_DOI>`).

---

## Quick start

```bash
git clone https://github.com/<USER>/sigma-dead-zone && cd sigma-dead-zone
conda create -n sigma_dead_zone python=3.9 -y && conda activate sigma_dead_zone
pip install -r requirements.txt
```

Every script carries a `ROOT` constant near the top pointing at the data root used on our
training host. Repoint it once to your own layout:

```bash
grep -rl '/mnt/data/underwater/bpinn_uq' scripts | xargs sed -i 's#/mnt/data/underwater/bpinn_uq#/your/data/root#g'
```

Then follow `docs/REPRODUCE.md`, which lists the upstream clones to make, what to copy from your
training host, and the order in which the scripts must run.

---

## Script to paper mapping

Every experiment in the paper is reproducible from a single script. Records are written as JSON
into `outputs/` on the machine that runs them.

| Script | Paper artifact |
|---|---|
| `train_ushape_baseline.py`, `train_ushape_edl.py`, `train_ushape_ens4090.py` | Sec. V-A, Table I — accuracy cost of the uncertainty mechanism, U-shape backbone |
| `collect_table1_accuracy.py` | Table I — lifts the best-validation epoch of every run out of the per-epoch training logs |
| `train_funie_gauss.py`, `train_mamba_fp32.py` | Sec. VI-B — Gaussian-head control, Mamba-UIE precision reference |
| `eval_coverage_stratified.py` | Sec. V-C, V-E, Figs. 3–4 — stratified coverage and the dead zone |
| `eval_crossdomain_uieb.py` | Sec. V-D, Table IV — ratio inflation across water bodies |
| `eval_fewshot_recal.py` | Sec. V-D, Fig. 2(b) — few-shot recalibration convergence |
| `eval_recalibration_edl.py`, `eval_ensemble_recal.py` | Sec. V-C, Table II — the single multiplicative scalar fitted on a calibration split |
| `eval_calibration_edl.py`, `eval_ensemble_calib.py` | Table II — uncertainty-toolbox calibration on the EUVP validation block |
| `eval_split_stability.py` | Sec. V-C — scalar stability over 30 random splits |
| `eval_scoring_rules.py` | Sec. V-C, Table III — NLL and CRPS before and after recalibration |
| `eval_rejection.py`, `eval_ushape_rejection.py` | Sec. V-F, Table V — rejection economics |
| `eval_bootstrap_ci.py` | Sec. IV (Statistics) — image-level cluster bootstrap |
| `eval_strat_proxy_evidence.py` | Sec. V-B — the three severity proxies |
| `eval_reffree_response.py` | Sec. V-E — reference-free corroboration (RUIE-UIQS, U45) |
| `eval_head_intervention.py`, `eval_mechanism_probes.py`, `eval_funie_gauss_audit.py` | Sec. VI-B — the three mechanism interventions |
| `eval_naive_baselines.py` | Sec. V-F — the constant-$\sigma$ reductio |
| `eval_recal_methods_comparison.py` | Sec. VI-D — scalar vs. isotonic vs. uncertainty-toolbox recalibration |
| `eval_fusion_gate.py` | Sec. V-F — soft repair by $\sigma$-gated fusion |
| `eval_sigma_severity.py` | Sec. V-E, Fig. 5 — degradation response profiles |
| `eval_puie_*.py`, `eval_mcbn_recal.py`, `eval_bem_calib.py` | Sec. VI-C — the paradigms outside the main comparison |
| `eval_mambauie_*.py`, `eval_mambauie_psnr.py` | Sec. V-A — Mamba-UIE accuracy reference |
| `eval_ushape_calib.py`, `eval_ushape_strat.py`, `eval_ushape_crossdomain.py`, `eval_ushape_recal.py`, `eval_ushape_rejection.py`, `eval_ushape_ens_audit.py` | Sec. V-G, Table VII — backbone robustness |
| `port_risk_coverage.py` | Generates `uq/tu_risk_coverage.py`: TorchUncertainty's AURC/AUGRC core adapted from classification to regression |
| `make_figures.py` | Every figure in the paper, from `records/*.json` and the per-pixel arrays; writes vector PDF plus PNG into `figures/` |
| `verify_paper_numbers.py` | Not a figure or a table: re-derives every headline number in the text from `records/` and reports any drift |

---

## Audit records

`records/` holds the machine-readable output of the audit: 35 JSON files, one per experiment or
per table. These are the files quoted in the "released records" notes in the paper, including the
per-proxy tables behind the stratification robustness check of Sec. V-B. Each file is
self-describing; keys mirror the metric names used in the text.

Every table in the paper has a backing file here, and so does every headline number in the running
text. The tables map to records as follows. Table I is `table1_accuracy.json`. Table II is
`edl_recalibration.json`, `ensemble_recalibration.json` and `puie_eval.json`. Table III is
`proper_scoring_rules.json`. Table IV is `crossdomain_uieb.json` and `fewshot_recal_uieb.json`.
Table V is `rejection_stage4.json` and `bootstrap_ci.json`. Table VI is `ushape_stratified.json`,
`ushape_rejection_stage4.json` and `ushape_ens_audit.json`. Table VII draws on those plus
`gauss_audit.json`, `head_intervention.json` and `reffree_response.json`.

To check that claim rather than trust it, run the verifier from the repository root. It recomputes
181 quantities — every value in the seven tables, the confidence intervals, and the ratios quoted
in the figure captions — from `records/` and prints any disagreement:

```bash
python scripts/verify_paper_numbers.py                       # reads records/
python scripts/verify_paper_numbers.py --records outputs     # reads a working tree instead
python scripts/verify_paper_numbers.py --arrays outputs      # also checks the Fig. 4/6/7 captions
```

It exits non-zero on any mismatch, so it can gate a release. With the arrays present it also
reproduces the response ratios in the Fig. 4 and Fig. 6 captions and the $\alpha$/$\beta$ ranges
of Fig. 7 from the per-pixel data.

Two records are worth calling out because they are easy to mistake for inconsistencies.
`records/ushape_ens_audit.json` is the protocol-corrected U-shape ensemble run; and the U-shape
cross-domain ratio appears in both `ushape_crossdomain_uieb.json` (6.673, the value Table VI
quotes) and `ushape_stratified.json` (6.683), two runs of the same quantity that differ by 0.01.

Raw per-pixel $\sigma$ and error arrays (85 MB and 76 MB compressed, EUVP and UIEB) are archived
at `https://doi.org/<ZENODO_DOI>` rather than committed here.

---

## Upstream credit

This audit evaluates other people's released models; the credit belongs to them.
`docs/UPSTREAM_REPOS.md` lists each upstream repository, its license, and exactly what we changed.
If you use this code, please cite the upstream works as well.

## Citation

```bibtex
@article{yu2026sigmadeadzone,
  title  = {The $\sigma$ Dead Zone in Auditing Pixel-wise Uncertainty for Underwater Image Restoration},
  author = {Yu, Xiangmiao},
  year   = {2026},
  note   = {Submitted to IEEE Transactions on Circuits and Systems for Video Technology}
}
```

## License

MIT for our code (see `LICENSE`). Upstream components keep their own licenses.
