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
# --results-tag expo writes the exposure-as-feature re-run to NEW files
# res/results_frequency_expo.csv + res/results_error_frequency_expo.csv, seeded with
# the GLM/XGBoost rows of the original files byte-for-byte (baselines preserved,
# never recomputed). The old (Strategy-B) result files are left untouched so the two
# runs can be compared. RUN THIS BEFORE the v3 stage — v3 appends onto these files.

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q2_frequency.py --skip-baselines --results-tag expo --tabpfn-versions v2_6

echo "Finished q2 frequency (v2_6)"
