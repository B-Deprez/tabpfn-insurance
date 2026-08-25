#!/bin/bash
#SBATCH --job-name=tabpfn_q2_freq_v3
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
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bruno.deprez@kuleuven.be

# Q2 frequency — STAGE 2 of 2 (TabPFN v3).
# --skip-baselines appends TabPFN v3 rows onto the CSVs the v2_6 stage rewrote
# (GLM/XGBoost preserved verbatim, no archiving). Submit this AFTER the v2_6 stage,
# chained on its success, e.g.:
#   JOBID=$(sbatch --parsable slurm/submit_q2_freq_v2_6.slurm)
#   sbatch --dependency=afterok:$JOBID slurm/submit_q2_freq_v3.slurm
#
# Heaviest job: v3 runs the "full" pass (freMTPL2's full train fold is ~490k rows).
# Wall-time is set high for this; if the "full" fit OOMs at 64G, raise --mem to 128g.

cd $VSC_DATA/tabpfn/tabpfn_insurance

source ${VSC_DATA}/miniconda3/etc/profile.d/conda.sh
conda activate tabpfn-insurance

which python
python -c "import tabpfn; print('TabPFN version:', tabpfn.__version__)"
nvidia-smi

python -u scripts/run_q2_frequency.py --skip-baselines --tabpfn-versions v3

echo "Finished q2 frequency (v3)"
