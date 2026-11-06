#!/bin/bash
# Keep the exploratory GPU jobs submitted (hpc/scheduler.py --loop) from a CPU node; polls every 5 minutes.
#   sbatch hpc/scheduler.sbatch                       all waves except calibration
#   sbatch --export=ALL,WAVES="e3" hpc/scheduler.sbatch
#SBATCH --job-name=ngx_scheduler
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/%u/niftygex/outputs/hpc/logs/%x_%j.log
set -eo pipefail                            # no -u: module and conda scripts use unset variables
umask 077                                   # the data and results stay private
module purge
module load explorer anaconda3/2024.06
source activate /home/$USER/envs/niftygex
cd /home/$USER/niftygex
NIFTYGEX_PROFILE=hpc PYTHONPATH=src python hpc/scheduler.py --loop --interval 300 --waves ${WAVES:-references e1 e3 e2 e2_l72}
