# SLURM jobs (VSC — wice / gpu_h100)

Batch jobs for the exposure-revision re-runs. One job = one whole script (kept
deliberately simple). All jobs `cd $VSC_DATA/tabpfn/tabpfn_insurance`, activate
the `tabpfn-insurance` conda env, and request 1×H100 + 8 CPUs + 64 GB. Logs land
in `slurm/logs/%x_%j.{out,err}` (job name + job id). Submit from the repo root.

| Script | Runs | Wall-time |
|--------|------|-----------|
| `submit_q1_severity.slurm`  | `run_q1_severity.py --results-tag log1p` (GLM + XGBoost + TabPFN v2_6/v3, both datasets) | 4 h |
| `submit_q2_freq_v2_6.slurm` | `run_q2_frequency.py --results-tag expo --tabpfn-versions v2_6` (GLM + XGBoost + TabPFN v2_6) | 12 h |
| `submit_q2_freq_v3.slurm`   | `run_q2_frequency.py --skip-baselines --results-tag expo --tabpfn-versions v3` | 24 h |
| `submit_q2_freq.slurm`      | `run_q2_frequency.py --results-tag expo` (GLM + XGBoost + TabPFN v2_6/v3 in one job; alternative to the staged pair) | 48 h |

## Submission order

```bash
cd $VSC_DATA/tabpfn/tabpfn_insurance

# Everything runs from scratch (res/ may be empty): GLM/XGBoost are recomputed.
# Q1 severity — independent, submit any time. Writes res/results_severity_log1p.csv
# + res/results_error_severity_log1p.csv.
sbatch slurm/submit_q1_severity.slurm

# Q2 frequency — EITHER one job (GLM + XGBoost + TabPFN v2_6 + v3) ...
sbatch slurm/submit_q2_freq.slurm

# ... OR staged (never both: they write the same res/*_expo.csv files). v2_6 first:
# it creates res/results_frequency_expo.csv + res/results_error_frequency_expo.csv
# with GLM/XGBoost + TabPFN v2_6. v3 then APPENDS onto those files, so it only
# starts if v2_6 succeeds. (--clusters makes --parsable print "ID;wice" -> cut.)
JOBID=$(sbatch --parsable slurm/submit_q2_freq_v2_6.slurm | cut -d';' -f1)
sbatch --dependency=afterok:$JOBID slurm/submit_q2_freq_v3.slurm
```

Afterwards, copy the four tagged CSVs into the Mac's `res/` (next to the old
untagged files) and check the recomputed GLM/XGBoost values match the old ones
(relative tolerance 1e-5; recomputed XGBoost differs by ~1e-6, so not byte-identical):

```bash
python scripts/verify_q2_regeneration.py --results-tag expo
python scripts/verify_q2_regeneration.py --task sev --results-tag log1p
```

## Notes

- **Why v3 gets 24 h / may need more memory.** v3 runs the `"full"` subsample —
  freMTPL2's full training fold is ~490k rows — which is by far the heaviest fit.
  If it OOMs at 64 GB, raise `--mem` to `128g` in `submit_q2_freq_v3.slurm`. v2_6
  is capped at its 100k pretraining ceiling, so its large sizes are skipped
  automatically and it finishes sooner.
- **Restartability.** The two Q2 stages are split so a v3 failure never forces a
  v2_6 redo. Both write via the append-only results path into the `_expo` files.
  To redo a stage-1 or Q1 run, delete its tagged `res/*_expo.csv` /
  `res/*_log1p.csv` files first; otherwise the rerun appends duplicate rows.
- **Copying instead of recomputing baselines.** `--skip-baselines --results-tag TAG`
  seeds a new tagged file with the GLM/XGBoost rows of the untagged file. It now
  stops with an error if the untagged file is missing (e.g. on an empty `res/`),
  instead of writing a file without baselines.
- **Changing cluster/account.** Settings are copied from the previous
  `slurm-tabpfn/` jobs (`--clusters=wice --partition=gpu_h100
  --account=lp_verbekelab`). Edit the `#SBATCH` headers if your allocation differs.
