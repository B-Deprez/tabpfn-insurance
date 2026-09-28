# SLURM jobs (VSC — wice / gpu_h100)

Batch jobs for the exposure-revision re-runs. One job = one whole script (kept
deliberately simple). All jobs `cd $VSC_DATA/tabpfn/tabpfn_insurance`, activate
the `tabpfn-insurance` conda env, and request 1×H100 + 8 CPUs + 64 GB. Logs land
in `slurm/logs/%x_%j.{out,err}` (job name + job id). Submit from the repo root.

| Script | Runs | Wall-time |
|--------|------|-----------|
| `submit_q1_severity.slurm`  | `run_q1_severity.py` (config-driven: both datasets, v2_6 + v3) | 4 h |
| `submit_q2_freq_v2_6.slurm` | `run_q2_frequency.py --skip-baselines --results-tag expo --tabpfn-versions v2_6` | 12 h |
| `submit_q2_freq_v3.slurm`   | `run_q2_frequency.py --skip-baselines --results-tag expo --tabpfn-versions v3` | 24 h |

## Submission order

```bash
cd $VSC_DATA/tabpfn/tabpfn_insurance

# Q1 severity — independent, submit any time.
sbatch slurm/submit_q1_severity.slurm

# Q2 frequency — MUST be staged. v2_6 first: it creates res/results_frequency_expo.csv
# + res/results_error_frequency_expo.csv (GLM/XGBoost rows copied verbatim from the
# original files) and adds the corrected v2_6 rows. v3 then APPENDS onto those files,
# so it only starts if v2_6 succeeds. The original result files are never touched.
JOBID=$(sbatch --parsable slurm/submit_q2_freq_v2_6.slurm)
sbatch --dependency=afterok:$JOBID slurm/submit_q2_freq_v3.slurm

# After both Q2 jobs finish, confirm the baselines were preserved byte-for-byte:
python scripts/verify_q2_regeneration.py --results-tag expo
```

## Notes

- **Why v3 gets 24 h / may need more memory.** v3 runs the `"full"` subsample —
  freMTPL2's full training fold is ~490k rows — which is by far the heaviest fit.
  If it OOMs at 64 GB, raise `--mem` to `128g` in `submit_q2_freq_v3.slurm`. v2_6
  is capped at its 100k pretraining ceiling, so its large sizes are skipped
  automatically and it finishes sooner.
- **Restartability.** The two Q2 stages are split so a v3 failure never forces a
  v2_6 redo. Both write via the append-only results path into the `_expo` files.
  To redo the whole re-run, delete the two `res/*_expo.csv` files first (they are
  re-seeded from the originals); otherwise a rerun appends duplicate TabPFN rows.
- **Baselines are copied, not recomputed.** Check the original files hold each
  GLM/XGBoost row once before stage 1 (44 rows in `results_frequency.csv`, 20 in
  `results_error_frequency.csv`) — the seed copies whatever is there.
- **Changing cluster/account.** Settings are copied from the previous
  `slurm-tabpfn/` jobs (`--clusters=wice --partition=gpu_h100
  --account=lp_verbekelab`). Edit the `#SBATCH` headers if your allocation differs.
