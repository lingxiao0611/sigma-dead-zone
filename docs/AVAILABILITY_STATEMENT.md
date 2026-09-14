# Availability statement — ready to paste

The paper states in four places that the protocol, code, and per-image audit records are released,
and refers to "the released records" for the per-proxy stratification tables. Add the paragraph
below before `\bibliographystyle` in `latex/main.tex` so those claims resolve.

## Once the repository exists

```latex
\section*{Availability of Data and Code}
The audit protocol, all training and evaluation scripts, and the per-image audit records behind
every number and figure in this article are available at
\url{https://github.com/lingxiao0611/sigma-dead-zone} and archived at
\url{https://doi.org/<ZENODO_DOI>}. The datasets are public: EUVP~\cite{islam2020euvp},
UIEB~\cite{li2020underwater}, U45~\cite{li2019fusion}, and RUIE-UIQS~\cite{liu2020real}.
```

Then update the two `released records` references in the body text if you would rather point at the
URL than at the phrase — the current wording is consistent either way:

- Sec. V-B: "per-proxy tables in the released records"
- Sec. V-C: "per-image audit records" appears in the contribution list of Sec. I

## Before you paste

- The repository URL is already filled in: https://github.com/lingxiao0611/sigma-dead-zone
- `records/` now backs every table in the paper and the headline numbers in the running text: 37
  JSON files, each with a producer script in `scripts/`, including Table I's accuracy record
  (`table1_accuracy.json`), the scalar-recalibration pair that Sec. V-C quotes (`edl_recalibration.json`,
  `ensemble_recalibration.json`), the bootstrap intervals (`bootstrap_ci.json`), the Gaussian-head
  control (`gauss_audit.json`), the U-shape ensemble arm (`ushape_ens_audit.json`,
  `ushape_ens_reffree_grades.json`), and the joined severity-ordering row of Table VI
  (`ruie_severity_rho.json`). `scripts/verify_paper_numbers.py` recomputes 202 of those quantities
  from the records and is the check to run before submitting. The per-pixel arrays that the figures
  are drawn from are still only on Zenodo, so keep the wording about "per-image audit records" and do
  not widen it to "every figure".
- Replace `<ZENODO_DOI>` after minting the DOI (Zenodo can mint one straight from a GitHub release).
- Check the section heading against the TCSVT author guide; some IEEE journals expect this as a
  footnote on the first page instead of a standalone section. Either placement is accepted in
  practice, but keep the wording identical to what the code claims.
