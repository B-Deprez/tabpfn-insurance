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

# Q2 frequency — STAGE 1 of 2 (GLM + XGBoost + TabPFN v2_6), from scratch.
# --results-tag expo writes to res/results_frequency_expo.csv +
# res/results_error_frequency_expo.csv (named apart from the old Strategy-B files
# kept on the Mac). GLM/XGBoost are recomputed here; TabPFN v2_6 uses the
# exposure-as-feature fix. RUN THIS BEFORE the v3 stage — v3 appends onto these files.

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q2_frequency.py --results-tag expo --tabpfn-versions v2_6

echo "Finished q2 frequency (v2_6)"
