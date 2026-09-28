# Paper revision checklist — exposure-fairness fix (EAJ Referee 1)

What changed: **TabPFN only.** Frequency: exposure is now an input feature and the
annualised rate μ is read off a counterfactual `Exposure=1.0` query (was: fit on
`ClaimNb/Exposure` with a silently-ignored `sample_weight`). Severity: TabPFN's
built-in `1_plus_log` target transform with the original-scale predictive mean
(was: fit on `log(AvgSeverity)`, `exp()` the prediction — roughly the median,
biased low). **GLM and XGBoost are unchanged.** This file flags repo-local language to revise and lists
the manuscript claims to re-examine once the corrected VSC numbers land. Nothing
here has been auto-edited — flags only.

## How to regenerate the corrected numbers (VSC / CUDA)

```bash
python scripts/run_q2_frequency.py --skip-baselines --results-tag expo --tabpfn-versions v2_6  # Step 2
python scripts/run_q2_frequency.py --skip-baselines --results-tag expo --tabpfn-versions v3    # Step 4
python scripts/verify_q2_regeneration.py --results-tag expo                                    # baselines unchanged?
python scripts/run_q1_severity.py --skip-baselines --results-tag log1p                         # Step 6
python scripts/verify_q2_regeneration.py --task sev --results-tag log1p                        # baselines unchanged?
```
The corrected runs land in NEW files — `res/results_frequency_expo.csv` +
`res/results_error_frequency_expo.csv` and `res/results_severity_log1p.csv` +
`res/results_error_severity_log1p.csv` (GLM/XGBoost rows copied verbatim); the old
files stay untouched for comparison. To switch the paper tables + Figure 1 over,
point `_CSV_NAMES` in `notebooks/results_tables.ipynb` (cell 1) at the tagged
files instead of the originals and re-run it.

## Repo-local language flagged (regenerated on re-run; not hand-edited)

- **`notebooks/results_tables.ipynb`** — the committed cell OUTPUTS hold OLD
  Strategy-B frequency numbers:
  - cell 5 / cell 19: frequency Poisson-deviance (and RMSE-on-rate) tables.
  - cell 13: Figure 1 — per-fold Poisson deviance vs subsample size (both datasets).
  These are *computed* from `res/`, so re-running after the corrected sweep updates
  them. Severity cells (3, 7, 15, 16) also change: re-run on `res/*_log1p.csv`.
- **`config/experiment_q2_frequency.yaml`** — header framing "Figure 1 — Poisson
  deviance vs. subsample size" and "cost of TabPFN's 10,000-sample ceiling for
  frequency". Structurally still valid; the plotted VALUES change.
- **`README.md` (line 8)** — "comparisons are like-for-like". Now *more* accurate:
  the fix is precisely what makes the frequency comparison like-for-like. No edit
  needed, but the frequency verdict behind the title/abstract may shift.

## Manuscript claims to re-examine (once corrected numbers exist)

- [ ] **Figure 1** — regenerate. TabPFN's frequency curve moves; the GLM/XGBoost
      reference lines do not. Re-read the figure's takeaway text.
- [ ] **Every TabPFN frequency Poisson-deviance table cell** — regenerate (all
      sizes × versions × both datasets).
- [ ] **Frequency exposure-weighted RMSE-on-rate** rows — regenerate.
- [ ] **Abstract / conclusion frequency verdict** — re-verify direction. A
      preliminary ONE-FOLD smoke (beMTPL97, `tabpfn_2000`, v3) moved from ≈0.347
      (old 5-fold mean) to ≈0.652 (corrected fold 0) — i.e. from *below* GLM
      (≈0.555) to *above* it. If the full 5-fold run confirms this, any "TabPFN is
      competitive with / matches GLM on frequency" statement for beMTPL97 likely
      **flips** and must be rewritten. **TO CONFIRM on the VSC.**
- [ ] **"Cost of the 10k-sample ceiling"** narrative for frequency — the cost curve
      changes; re-state.
- [ ] **Exposure-handling methodology** — describe exposure-as-feature +
      counterfactual `Exposure=1.0`, and state the limitation: this is NOT a true
      offset (TabPFN learns the exposure effect rather than having a unit slope on
      `log(Exposure)` imposed). Goes in `EXPERIMENTS.md` (pending).
- [ ] **Remove old "Strategy B" (rate) framing** for frequency wherever the
      manuscript describes how TabPFN was fed exposure.
- [ ] **Severity — TabPFN numbers CHANGE (1_plus_log fix).** The dead
      `sample_weight=ClaimNb` removal alone was number-preserving, but TabPFN
      severity now returns the predictive mean via the built-in `1_plus_log`
      transform instead of exp(E[log y]). Local smoke (beMTPL97 fold 0, v3, 3k
      train): mean prediction 451 → 1,322 vs observed 1,383; Gamma deviance
      3.84 → 2.10. Re-examine every severity claim (Table 1, "competitive on
      severity") once the VSC `_log1p` run lands. GLM/XGBoost severity untouched.

## Not-yet-decided (tracked in CLAUDE.md)

- **Q3 SHAP (frequency)** shares the corrected wrapper. It was not re-run; if
  regenerated, the frequency SHAP array gains an exposure feature and the freq
  deviance shifts. Decide whether Q3 frequency interpretability is in scope.
