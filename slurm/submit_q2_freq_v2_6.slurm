#!/bin/bash
#SBATCH --job-name=tabpfn_q2_freq_v2_6
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=lp_verbekelab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bruno.deprez@kuleuven.be

# Q2 frequency — STAGE 1 of 2 (TabPFN v2_6).
# --force archives res/results_frequency.csv + res/results_error_frequency.csv to
# res/archive/, keeps the GLM/XGBoost rows byte-for-byte, drops the stale (invalid
# Strategy-B) TabPFN rows, and regenerates TabPFN v2_6 with the exposure-as-feature
# fix. --force implies --skip-baselines (baselines are preserved, never recomputed).
# RUN THIS BEFORE the v3 stage — v3 appends onto the CSVs this job rewrites.

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q2_frequency.py --force --tabpfn-versions v2_6

echo "Finished q2 frequency (v2_6)"
