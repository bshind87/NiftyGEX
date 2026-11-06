#!/bin/bash
# After hpc/push_data.sh: check every copied file against the Mac's manifests (sizes, rows, SHA-256) on a CPU node.
#   sbatch hpc/verify_transfer.sbatch       (from /home/$USER/niftygex)
#SBATCH --job-name=ngx_verify_transfer
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/niftygex/outputs/hpc/logs/%x_%j.log
set -eo pipefail                            # no -u: module and conda scripts use unset variables
umask 077                                   # the data and results stay private
# Start clean: the login shell may activate another conda env (~/.bashrc) and SLURM copies that environment in.
module purge
module load explorer anaconda3/2024.06
eval "$(conda shell.bash hook)"
for _ in 1 2 3 4 5; do if [ -n "${CONDA_PREFIX:-}" ]; then conda deactivate; fi; done
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
conda activate /home/$USER/envs/niftygex
[ "$(command -v python)" = "/home/$USER/envs/niftygex/bin/python" ] || { echo "python is $(command -v python), not the niftygex env" >&2; exit 2; }
# The anaconda module puts its older libstdc++ on LD_LIBRARY_PATH; the env's own libraries must come first.
export LD_LIBRARY_PATH="/home/$USER/envs/niftygex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/$USER/niftygex
NIFTYGEX_PROFILE=hpc PYTHONPATH=src python scripts/verify_transfer.py
