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

# Q1 severity — config-driven (no CLI args): both datasets, TabPFN v2_6 + v3.
# The exposure revision left severity provably number-preserving, so this is a
# re-run for a fresh, complete result set rather than a required fix.

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q1_severity.py

echo "Finished q1 severity"
