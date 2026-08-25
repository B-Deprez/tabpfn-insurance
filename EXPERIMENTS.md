# Experiment Outline
> TabPFN for Insurance Pricing — Letter for EAJ (≤5 pages)
 
---
 
## Research Question
 
Can TabPFN — a tabular foundation model requiring no fitting on the target dataset — compete
with tuned GLMs and GBMs for insurance pricing, and what is the practical cost of its
10,000-sample ceiling for frequency modelling?
 
---
 
## Datasets
 
| Dataset | Source | Task | Rows |
|---|---|---|---|
| `freMTPL2freq` | CASdatasets (R) | Frequency | ~677,991 |
| `freMTPL2sev` | CASdatasets (R) | Severity | ~26,639 |
| `bemtpl` | CASdatasets (R) | Frequency + Severity | ~163,212 |
 
---
 
## Models
 
| Model | Type | Library | Notes |
|---|---|---|---|
| GLM | Baseline | `statsmodels` | Poisson (freq) + Gamma (sev), log link, log-exposure offset. GAM binning done once on full training fold. |
| XGBoost | Baseline | `xgboost` | Fixed hyperparameters from Henckaerts et al. (2021): `max_depth=3`, `n_estimators=500`, `learning_rate=0.01`, `subsample=0.75`. |
| TabPFN | Focal model | `tabpfn` | `TabPFNRegressor`, default config, no tuning. Exposure handled per task (see *Exposure Handling*). Auto-detect device: CUDA → MPS → CPU, in that order. |
 
---
 
## Data Cleaning
 
Follow the standard cleaning steps from Wüthrich & Buser (2021) for `freMTPL2freq`/`freMTPL2sev`
(e.g. removal of records with implausible exposure values). Apply the same principle to `bemtpl`
following Henckaerts et al. (2021). Document any removed records in the loader log.
 
---
 
## Feature Encoding
 
Applied consistently before any model sees the data. All encoding parameters are fit on the
training fold only and applied to the test fold.
 
Encoding decisions are made manually after inspecting `notebooks/explore_categoricals.ipynb`,
which visualises the distribution and mean claim rate per level for every categorical feature.
The decisions are recorded in that notebook and then hard-coded in `config/features.yaml`.
 
| Feature type | Model | Encoding |
|---|---|---|
| Spatial (lat/lon) | All | Keep as continuous numeric features |
| Ordinal categoricals (e.g. bonus-malus bands, age groups, vehicle age) | All | Ordinal/label encoding, respecting the natural order |
| Nominal categoricals (e.g. region, fuel type, coverage type) | GLM | One-hot encoding |
| Nominal categoricals | XGBoost, TabPFN | Label encoding (integer codes) |
| Severity target | TabPFN only | Log-transform `y` before fitting; invert with `exp()` after predicting |
 
**Note for paper:** The absence of GAM-based binning for the GLM is acknowledged as a limitation.
More principled approaches (e.g. PD-clustering via `maidrr`) exist but are outside the scope of
this letter. This simplification applies equally to all models and does not favour any one method.
 
---
 
## Exposure Handling (TabPFN only)
 
TabPFN supports neither a Poisson **offset** (as the GLM uses) nor
**`sample_weight` / `base_margin`** (as XGBoost uses), so exposure cannot enter
the way it does for the baselines. It is handled per task as follows.
 
- **Frequency — exposure as an input feature (counterfactual annualisation).**
  Train on the *bounded* count `ClaimNb` (never the rate `ClaimNb/Exposure`,
  which explodes for tiny exposures), with `Exposure` added as an ordinary input
  feature. Recover the annualised rate μ by a **counterfactual query**: evaluate
  every test policy at a full year of exposure (`Exposure = 1.0`), so the
  expected-count output is already the per-year rate that the exposure-weighted
  Poisson deviance expects (metric unchanged).
 
  > **Limitation (for the paper).** Exposure-as-feature is NOT a true offset:
  > GLM/XGBoost impose a unit slope on `log(Exposure)`, whereas TabPFN *learns*
  > the exposure effect. It is the fairest option available given TabPFN has
  > neither an offset nor `sample_weight`, but it is not identical to an offset.
 
- **Severity — log target, unweighted at estimation.** Response =
  `log(AvgSeverity)` (`AvgSeverity = ClaimAmount / ClaimNb`), inverted with
  `exp()`. TabPFN cannot take `sample_weight`, so — unlike the GLM
  (`var_weights=ClaimNb`) and XGBoost (`sample_weight=ClaimNb`) — TabPFN severity
  is **unweighted at estimation**; the `ClaimNb` weighting legitimately remains at
  **evaluation** (Gamma deviance is weighted by `ClaimNb` for all models).
 
> **Revision note (EAJ Referee 1).** This supersedes the earlier *Strategy B*,
> which fed TabPFN the rate `ClaimNb/Exposure` with `sample_weight=Exposure` for
> frequency. TabPFN ignores `sample_weight`, so that weighting was a silent no-op
> and the frequency comparison was not like-for-like — frequency results must be
> regenerated. **Severity numbers are unaffected** (the `sample_weight` no-op
> applied there too).
---
 
## Cross-Validation Scheme
 
- **5-fold CV**, single repetition, fixed `seed = 42`
- Stratified by claim indicator (`ClaimNb > 0`) using round-robin assignment
- One split object reused by all models
- Assert claim rate per fold within ±0.5pp of overall rate
- Record wall-clock fit + predict time per fold per model
---
 
## Experiments
 
### Q1 — Severity benchmark → Table 1
 
**Question:** Is TabPFN competitive with GLM and XGBoost on severity, with zero tuning?
 
| | |
|---|---|
| **Datasets** | French MTPL severity + Belgian MTPL severity |
| **Models** | GLM, XGBoost, TabPFN |
| **Metric** | Gamma deviance (weighted by `ClaimNb`) |
| **Reporting** | Mean ± std over 5 folds (per-fold) + pooled OOF score |
| **TabPFN note** | French severity (>10k rows): subsample training set to 10k per fold |
| **Output** | **Table 1**: gamma deviance, 2 datasets × 3 models |
 
---
 
### Q2 — Frequency ceiling → Figure 1
 
**Question:** What is the cost of TabPFN's 10,000-sample ceiling for frequency modelling?
 
| | |
|---|---|
| **Dataset** | French MTPL frequency |
| **Models** | TabPFN at subsample sizes {2,000 / 5,000 / 10,000}; GLM and XGBoost on full training fold |
| **Metric** | Poisson deviance (evaluated at rate level, weighted by exposure) |
| **Reporting** | Deviance per subsample size per fold; GLM and XGBoost as horizontal reference lines |
| **Output** | **Figure 1**: Poisson deviance vs. subsample size; baselines as horizontal lines |
 
---
 
### Q3 — Interpretability → Figure 2
 
**Question:** Does TabPFN learn actuarially meaningful feature-response patterns?
 
| | |
|---|---|
| **Dataset** | French MTPL severity |
| **Scope** | Fold 1 only |
| **Method** | SHAP beeswarm for TabPFN (built-in) and XGBoost (`TreeExplainer`); GLM coefficients on same test fold |
| **Output** | **Figure 2**: side-by-side SHAP summary — TabPFN vs. XGBoost vs. GLM coefficients |
 
---
 
## Metrics
 
| Metric | Task | Formula |
|---|---|---|
| Poisson deviance | Frequency | `2 * mean(y * log(y / (e * mu)) - (y - e*mu))` where `mu` = predicted rate |
| Gamma deviance | Severity | `2 * weighted_mean((y - mu)/mu - log(y/mu))` weighted by `ClaimNb` |
 
Both computed per-fold (mean ± std) **and** as a pooled OOF score.
All metric logic lives in `src/utils/metrics.py`.
 
---
 
## Outputs
 
| Item | Content | Script |
|---|---|---|
| `res/results.csv` | All deviance scores, all folds, all models | Appended by all experiment scripts |
| **Table 1** | Gamma deviance: French + Belgian severity × 3 models | `scripts/run_q1_severity.py` |
| **Figure 1** | Poisson deviance vs. subsample size (French frequency) | `scripts/run_q2_frequency.py` |
| **Figure 2** | SHAP beeswarm: TabPFN vs. XGBoost vs. GLM (French severity, fold 1) | `scripts/run_q3_shap.py` |
| `res/shap/` | Raw SHAP arrays and GLM coefficients | `scripts/run_q3_shap.py` |
 
Figures are plotted in `notebooks/` from the saved outputs, not inside scripts.
 
---
 
## Paper Structure (target: 5 pages)
 
| Section | Content | Length |
|---|---|---|
| Abstract | TabPFN, ICL, insurance pricing, competitive on severity, sample ceiling | 6 sentences |
| 1. Introduction | GLM dominance → ML → foundation models → contribution | ~0.6 pages |
| 2. TabPFN and the pricing setup | ICL concept; freq-sev decomposition; exposure strategy B; sample ceiling | ~0.8 pages |
| 3. Empirical study | Two datasets; models; CV scheme; Table 1; Figure 1; Figure 2 | ~1.8 pages |
| 4. Discussion | When to use TabPFN; limitations (ceiling, no offset); future work | ~0.6 pages |
| References | ~12 key references | ~0.3 pages |
 
