#!/bin/bash
#SBATCH --job-name=tabpfn_q2_freq
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
#SBATCH --time=48:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bruno.deprez@kuleuven.be

# Q2 frequency — from scratch, single job: GLM + XGBoost + TabPFN (v2_6 + v3),
# both datasets. Alternative to the staged submit_q2_freq_v2_6 / _v3 pair; do not
# submit both. TabPFN uses the exposure-as-feature fix. --results-tag expo writes to
# res/results_frequency_expo.csv + res/results_error_frequency_expo.csv (named apart
# from the old Strategy-B files kept on the Mac).
#
# Wall-time = the two staged jobs combined (12 h + 24 h, rounded up); v3's "full"
# pass (freMTPL2's full train fold is ~490k rows) is the heaviest fit.

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q2_frequency.py --results-tag expo

echo "Finished q2 frequency"
