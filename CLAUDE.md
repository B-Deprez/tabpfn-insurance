# CLAUDE.md  *(PROPOSED — awaiting approval; rename to `CLAUDE.md` once accepted)*

Guidance for working in the **TabPFN for Insurance Pricing** repository. This file
is the working memory for the reviewer-driven *exposure-fairness* fix (EAJ Referee 1).

> **Why this file exists.** Referee 1 flagged that the frequency comparison is not
> like-for-like: GLM and XGBoost account for policy **exposure** at estimation
> (offset / `base_margin`), but TabPFN was fed unweighted annualised rates
> (`ClaimNb/Exposure`) with `sample_weight=Exposure` — and **TabPFN does not
> support `sample_weight`**, so that weighting was a silent no-op. The frequency
> "Strategy B" is therefore invalid. We fix **only** how TabPFN receives exposure,
> keeping the current deviance-centric framing (Poisson/Gamma deviance + RMSE).

---

## 1. Project conventions (distilled from the code)

**Layout.** `config/` (YAML: data, features, per-experiment), `src/data`
(loaders, contracts, cv, preprocessing), `src/methods` (glm/xgboost/tabpfn
wrappers), `src/utils` (metrics, results, logging), `scripts/run_q{1..4}_*.py`,
`res/` (append-only result CSVs + plots), `notebooks/`.

**Uniform model interface.** Every wrapper exposes the same signature so the
runner scripts call them interchangeably:
```python
model.fit(X_train, y_train, sample_weight=..., log_exposure=..., [fold_seed=...])
mu = model.predict(X_test, log_exposure=...)
```
Unused arguments are accepted "for interface parity" and documented as ignored.

**CV.** One shared stratified 5-fold split (`src/data/cv.py`, `cv_seed=42`,
`cv_folds=5`), round-robin on the claim indicator, reused by all models. Per-fold
claim-rate contract enforced in `contracts.validate_fold_claim_rates`.
**Do not** import the notebook's single random split — keep 5-fold CV.

**Feature handling.** `config/features.yaml` is the authoritative per-feature
encoding record. `encode_features(...)` builds the GLM (one-hot) / tree (label /
ordinal) design matrices; **`get_raw_features(...)` returns raw unencoded columns
for TabPFN** (TabPFN consumes mixed dtypes natively). All encoders are fit on the
**training fold only**.

**Targets / weights** (`preprocessing.get_targets`): freq → `y=ClaimNb`,
`weight=Exposure`, `log_exposure=log(Exposure)`; sev → `y=AvgSeverity`,
`weight=ClaimNb`. `AvgSeverity = ClaimAmount / ClaimNb` (built in the loaders).

**Results schema** (`src/utils/results.py`, **frozen**): one tidy row per
measurement — `timestamp, experiment_id, dataset, model, task, fold, metric,
value, tabpfn_version`. `fold` is an int or the string `"pooled"`.
`append_results` is **append-only** (never overwrites; header written once).
`tabpfn_version` is `""` for GLM/XGBoost, `"v2_6"`/`"v3"` for TabPFN.

**TabPFN config conventions.** Version is selected via
`TabPFNRegressor.create_default_for_version(ModelVersion.V2_6|V3, device=...)`
(`src/methods/tabpfn_model.py`). Device auto-detect: CUDA → MPS → CPU. Large
inputs need `ignore_pretraining_limits=True` (v2.5/v2.6 ceilings 50k/100k rows;
the Q2 sweep skips a version when a subsample exceeds its ceiling). GPU buffers
are released between fits (`_release_gpu`) to avoid CUDA fragmentation.

**Notebook style to borrow** (`notebooks/insurance_claim_modeling.ipynb`) — *style
and technique only, not a runnable dependency*:
- Exposure as an **ordinary input feature**; model the **bounded count**
  (`ClaimNb`), never the rate `ClaimNb/Exposure` (dividing by a tiny exposure
  explodes the tail).
- Recover the annualised rate with a **counterfactual query**:
  `X_te = X_test[features + ["Exposure"]].copy(); X_te["Exposure"] = 1.0`, then
  `predict(X_te)` → μ (expected count at one full year = annualised frequency).
- Clean sklearn-style structure; `n_estimators`, `ignore_pretraining_limits`,
  and target transforms declared explicitly.
- **Out of scope** (do **not** import or let it steer metric choices): Gini,
  Tweedie, Lorenz/calibration, the single-stage track, the notebook's single split.

---

## 2. Exposure handling — how it works TODAY (survey)

| Task | GLM | XGBoost | TabPFN |
|------|-----|---------|--------|
| **Frequency** | `y=ClaimNb`, Poisson log-link, **offset `log(Exposure)`**; predict ÷ exposure → rate | `y=ClaimNb`, `count:poisson`, **`base_margin=log(Exposure)`**; predict ÷ exposure → rate | **INVALID**: `y=ClaimNb/Exposure` (rate), fit on rate; `sample_weight=Exposure` **ignored** (unsupported). Inlined in `run_q2_frequency.py`, not via the wrapper. |
| **Severity** | `y=AvgSeverity`, Gamma log-link, **`var_weights=ClaimNb`** | `y=AvgSeverity`, `reg:gamma`, **`sample_weight=ClaimNb`** | `y=log(AvgSeverity)`, predict `exp(...)` → exp(E[log y]) ≈ median, **biased low** (fixed in Step 6: built-in `1_plus_log`, original-scale mean); **`sample_weight=ClaimNb` passed but ignored** → unweighted at estimation (dead kwarg). |

**Evaluation is identical across models** (this must not change):
- Frequency: `poisson_deviance(y=counts, mu=rate, exposure, sample_weight=exposure)`
  and `exposure_weighted_rmse_rate`. μ is a **rate** (expected count per unit
  exposure).
- Severity: `gamma_deviance(y=AvgSeverity, mu, sample_weight=ClaimNb)` + RMSE/MAE/
  Pearson/Spearman. The `ClaimNb` weighting at **evaluation** is legitimate and stays.

**Asymmetry audit result:** the **only** exposure asymmetry is the known TabPFN
one. GLM and XGBoost both use a proper exposure offset in frequency
(`offset` / `base_margin`) and both weight severity by `ClaimNb`. **No new
asymmetry to fix.** (XGBoost `PoissonXGBoost` also *accepts* `sample_weight` but
ignores it; `base_margin` is the real mechanism — this is intentional, not a bug.)

**Why the fix plugs in unchanged.** After the fix, TabPFN-freq is trained on the
count `ClaimNb` with `Exposure` as a feature, and predicted at `Exposure=1.0`. The
output μ is the expected count at one year = an **annualised rate** — exactly the
scale the existing `poisson_deviance`/`exposure_weighted_rmse_rate` calls expect.
`metrics.py` is untouched.

> **Limitation to record in the paper:** exposure-as-feature is **not a true
> offset**. GLM/XGBoost *impose* a unit slope on `log(Exposure)`; TabPFN *learns*
> the exposure effect from data. It is the fairest option available given TabPFN
> has neither offset nor `sample_weight`, but it is not identical to an offset.

---

## 3. File tiers (governs what may change)

- **FROZEN (behavior):** `src/utils/metrics.py`, `src/methods/glm_model.py`,
  `src/methods/xgboost_model.py`, `src/data/cv.py`, and the results **schema**
  (`RESULT_COLUMNS`). Observable behavior must not change. Additive,
  behavior-preserving edits allowed **only if unavoidable and flagged** (e.g. a new
  optional arg defaulting to current behavior). Any edit that would move an existing
  output is **STOP-and-ask**, not a decision.
- **SHARED (additively editable):** `config/features.yaml`, preprocessing/
  feature-engineering utils (`src/data/preprocessing.py`), and the TabPFN wrapper
  (`src/methods/tabpfn_model.py`). May **add** here (new helper / config key /
  function) when it is the cleanest home. **Must not** alter behavior of paths used
  by GLM/XGBoost (esp. `encode_features`, `get_targets`). Every shared edit is
  called out in the step's diff summary.
- **IN-SCOPE:** the files named in the current step. Edit freely.

---

## 4. Staged change plan (checklist — one step per turn, STOP after each)

- [x] **Step 0 — Survey + snapshot + this file.** DONE. No experiment code touched.
      Baseline saved to `res/_baseline_snapshot.csv` (identity keys + raw value
      string, timestamp dropped).
- [x] **Step 1 — TabPFN frequency exposure refactor.** DONE *(IN-SCOPE:
      `src/methods/tabpfn_model.py`; SHARED-additive: `config/features.yaml`
      key `tabpfn_freq_exposure_feature`, helpers `get_freq_exposure_col` /
      `get_raw_features_freq` in `preprocessing.py`; NEW: `tests/test_tabpfn_freq_exposure.py`).*
      `TabPFNFreq` models the count `ClaimNb` with exposure as an input feature and
      predicts at `Exposure=1.0` → μ. 6/6 tests pass (incl. real-TabPFN v3). No leak
      into GLM/XGBoost; `metrics.py` unchanged; μ plugs into the Poisson-deviance call
      verbatim.
- [~] **Step 2 — Wire into `scripts/run_q2_frequency.py` (TabPFN branch only).**
      CODE COMPLETE, awaiting the VSC run (see D8, D10). TabPFN branch uses
      `get_raw_features_freq` + counterfactual `Exposure=1.0`; new CLI
      `--force` / `--skip-baselines` / `--tabpfn-versions` / `--results-tag`.
      Regeneration = **new tagged files `res/*_expo.csv`** (D10, supersedes D8's
      in-place rewrite), **baselines copied verbatim** (never recomputed); old
      Strategy-B files kept for comparison. Heavy sweep runs on the **VSC (CUDA)** —
      infeasible on the Mac (MPS predict on one beMTPL97 test fold ≈ 9 min). Post-run:
      `verify_q2_regeneration.py --results-tag expo`.
- [x] **Step 3 — Severity cleanup.** DONE *(IN-SCOPE: `scripts/run_q1_severity.py`
      call-site now omits `sample_weight` for TabPFN; `TabPFNSev` docstring notes
      "unweighted at estimation"; NEW `tests/test_tabpfn_sev_unweighted.py`).*
      Log-transform kept. **Provably number-preserving** (test shows TabPFN trains on
      identical inputs with/without the weight; GLM/XGBoost untouched) → **no VSC
      re-run needed for Step 3**; existing severity results stay valid. *(Severity is
      nonetheless re-run for the separate Step 6 fix.)*
- [~] **Step 4 — Re-run v3** and reconcile with v2_6. VSC-blocked: no code; runs on
      the VSC (`--skip-baselines --results-tag expo --tabpfn-versions v3`), then reconcile the corrected
      v2_6 vs v3 numbers once both exist.
- [x] **Step 5 — Docs.** DONE (write-up + flags). `EXPERIMENTS.md` "Exposure
      Handling" rewritten to exposure-as-feature + counterfactual (freq) and
      log-target/unweighted-at-estimation (sev), with the not-a-true-offset
      limitation and an EAJ-Ref-1 revision note; Models-table note updated; the
      accidental Claude.ai UI paste (former lines 1–193) stripped. Repo-local flags +
      manuscript checklist → `PAPER_REVISION_CHECKLIST.md` (notebook freq tables +
      Figure 1 are computed → regenerate on re-run; severity unaffected; preliminary
      smoke suggests beMTPL97 frequency worsens materially, possibly flipping a
      "competitive" claim — TO CONFIRM on VSC). Pre-existing outline staleness (Q2
      datasets/sizes, Q1 10k note, single `results.csv`) left as-is per scope.
      **Remaining = numeric reconciliation after the VSC v2_6/v3 runs (ties to Step 4).**
- [~] **Step 6 — Severity `1_plus_log` fix.** CODE COMPLETE, awaiting the VSC run
      (see D11). *(IN-SCOPE: `TabPFNSev` in `src/methods/tabpfn_model.py` — raw
      `AvgSeverity` + `inference_config={"REGRESSION_Y_PREPROCESS_TRANSFORMS":
      ("1_plus_log",)}`, `predict` returns TabPFN's original-scale mean, no `exp()`;
      SHARED-additive: `_make_regressor(..., **overrides)`, default unchanged for freq;
      `scripts/run_q1_severity.py` gains `--skip-baselines` / `--results-tag`;
      `verify_q2_regeneration.py --task sev`; tests updated.)* Not number-preserving:
      VSC run `--skip-baselines --results-tag log1p` → `res/*_severity_log1p.csv`.

**Invariant enforced by TEST after every step:** GLM/XGBoost statistical values
(excluding `fit_predict_seconds`, which is wall-clock) diff byte-identically
against `res/_baseline_snapshot.csv`, and the Poisson-deviance computation is
unchanged. That guarantee — not a file ban — is what makes the fix trustworthy.

---

## 5. Decision log

- **D1 — Author CLAUDE.md fresh.** No `CLAUDE.md` or `EXPERIMENTS.md` exists in the
  repo or its git history, though `tabpfn_model.py`/`preprocessing.py` reference a
  "CLAUDE.md §Feature Encoding". This file is authored new. *(Open Q1.)*
- **D2 — Exposure is TabPFN-freq-only in `features.yaml`.** `features.yaml` is read
  by BOTH `encode_features` (GLM/XGBoost) and `get_raw_features` (TabPFN). Adding
  `Exposure` to the shared `features` list would leak it into GLM/XGBoost inputs
  and break the frozen baselines. **Plan:** add an **additive, TabPFN-freq-only**
  config key (e.g. per-dataset `tabpfn_freq_extra_features: [Exposure]`) consumed
  by a new freq-specific raw-feature path — never by `encode_features`. Exact shape
  finalized in Step 1.
- **D3 — Model the bounded count, recover rate by counterfactual.** Fit on
  `ClaimNb` with `Exposure` as a feature; predict at `Exposure=1.0`. Matches the
  notebook technique and yields μ on the annualised-rate scale the metrics expect.
- **D4 — Regenerate only the TabPFN frequency rows in Step 2.** GLM/XGBoost rows
  are left exactly as recorded (not recomputed), which makes byte-identity trivially
  true and dodges XGBoost float-reproducibility risk. *(Open Q3.)*
- **D5 — Deviance framing kept; no new metrics.** Gini/Tweedie/Lorenz/calibration/
  single-stage are out of scope per the user's decision.
- **D6 — Exposure feed = config key + `get_raw_features_freq`; wrapper is dual-path.**
  `tabpfn_freq_exposure_feature` names the exposure column (TabPFN-freq only).
  `TabPFNFreq` accepts exposure either already in `X` (the run_q2 path) or via
  `sample_weight` which it injects (the run_q3 SHAP path) — so run_q3 keeps working
  unchanged. `get_raw_features`/`encode_features`/`get_targets` untouched → no leak.
- **D7 — No target transform for the count.** Fit on raw `ClaimNb` counts (small,
  bounded); no rate, no log (unlike severity's log-transform).
- **D8 — Step 2 regeneration: archive+rewrite, baselines verbatim, run on VSC.**
  User chose **A (archive + rewrite in place)** and **preserve GLM/XGBoost verbatim**.
  `--force` archives the two freq CSVs to `res/archive/` and drops stale TabPFN rows
  (keeps GLM/XGBoost lines byte-for-byte); implies `--skip-baselines`. Staged:
  `--force --tabpfn-versions v2_6` (Step 2) then `--skip-baselines --tabpfn-versions v3`
  (Step 4). All heavy runs execute on the **VSC (CUDA)** — see [[vsc-compute]]; the Mac
  (MPS) is only used for small correctness smokes. `verify_q2_regeneration.py` checks
  baselines stayed byte-identical to the `--force` archive.
- **D9 — Q3 SHAP shares the freq wrapper (FLAGGED, not changed).** Since `TabPFNFreq`
  changed, a future `run_q3_shap.py` run would produce corrected freq deviance and a
  SHAP array that now includes an exposure feature. run_q3 was NOT modified or re-run.
  Decide at Step 5 whether Q3 freq interpretability should be regenerated.
- **D10 — Keep old results side by side: tagged output files (supersedes D8's
  in-place rewrite).** User wants the Strategy-B numbers kept for comparison. New
  `--results-tag expo` redirects output to `res/results_frequency_expo.csv` +
  `res/results_error_frequency_expo.csv`; with `--skip-baselines` a missing tagged
  file is seeded with the header + GLM/XGBoost lines of the original, byte for byte
  (an existing one is appended to, so v3 stacks onto v2_6). Original files are never
  touched; `--force` is no longer used by the SLURM jobs (kept for in-place redo).
  A separate file, not a new model label or `experiment_id`: `results_tables.ipynb`
  dedupes on model+version (keep last) and fuses model+version for display, so
  same-file rows would silently overwrite or pool old and new. Switching the paper
  tables = point the notebook's `_CSV_NAMES` at the `_expo` files.
  `verify_q2_regeneration.py --results-tag expo` checks the tagged baselines against
  the originals. VSC's own `res/` held every baseline row twice (88/40 vs 44/20);
  replace it with the Mac copies (snapshot-verified) before stage 1, since the seed
  copies whatever is there.
- **D11 — Severity uses TabPFN's built-in `1_plus_log` (Step 6).** The former
  fit-on-`log(AvgSeverity)` + `exp(predict)` returned exp(E[log y]) — roughly the
  median, below E[y] for right-skewed amounts — so TabPFN severity was biased low
  against GLM/XGBoost (which predict the mean). Now raw `AvgSeverity` with
  `inference_config={"REGRESSION_Y_PREPROCESS_TRANSFORMS": ("1_plus_log",)}` (the
  Prior Labs insurance cookbook setting); TabPFN inverts the whole predictive
  distribution, so `predict` = original-scale mean. Local smoke (beMTPL97 fold 0,
  v3, 3k train / 1.5k test): mean pred 451 → 1,322 vs observed 1,383; Gamma dev
  3.84 → 2.10. Re-run tagged `log1p` (D10 mechanism), old severity files kept.
  Flag: `TabPFNSev.get_shap_values` now explains original-scale predictions (was
  log scale) — Q3 severity SHAP changes if re-run; run_q3 not modified or re-run.

---

## 6. Open questions for the user (raised in Step 0, not yet resolved)

1. **CLAUDE.md / EXPERIMENTS.md don't exist.** Confirm authoring `CLAUDE.md` fresh.
   For Step 5's "EXPERIMENTS.md → Exposure Handling", where should that write-up
   live — a new `EXPERIMENTS.md`, or a section in `README.md`?
2. **Conda envs.** Only `tabpfn-insurance` exists; there are no `tabpfn_v2` /
   `tabpfn_v3` environments. The v2_6/v3 split is the `tabpfn_version` axis inside
   the single env (`create_default_for_version`). Confirm that "re-run in the
   tabpfn_v2 / tabpfn_v3 env" means run the sweep for `tabpfn_version` v2_6 then v3
   in the one env.
3. **No `--force` / regeneration mechanism exists.** `append_results` only ever
   appends, so re-running would duplicate rows. How should Step 2 regenerate the
   invalid Strategy-B TabPFN freq rows — (a) archive+rewrite the two freq CSVs
   preserving GLM/XGBoost rows verbatim and replacing TabPFN rows, (b) add an
   `exposure_mode` column and append the new rows alongside the old, or (c) write a
   fresh results file? (Detailed in Step 2.)
