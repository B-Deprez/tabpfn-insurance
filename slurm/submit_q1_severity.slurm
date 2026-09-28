#!/bin/bash
#SBATCH --job-name=tabpfn_q1_severity
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
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bruno.deprez@kuleuven.be

# Q1 severity — from scratch: GLM + XGBoost + TabPFN (v2_6 + v3), both datasets.
# TabPFN severity now uses TabPFN's built-in 1_plus_log target transform and
# returns the predictive mean on the original scale (was: fit on log(y), exp() the
# prediction -> roughly the median, biased low). --results-tag log1p writes to
# res/results_severity_log1p.csv + res/results_error_severity_log1p.csv (named apart
# from the old severity files kept on the Mac).

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q1_severity.py --results-tag log1p

echo "Finished q1 severity"
