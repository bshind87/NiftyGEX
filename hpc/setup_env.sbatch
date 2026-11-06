#!/bin/bash
# Build the niftygex conda environment on a CPU node, then run the CPU tests (docs/hpc.md, step 3).
#   bash hpc/remote.sh setup      (submits this, then hpc/cuda_check.sbatch on gpu-short once it succeeds)
# Not on a GPU node: an install leaves the GPU idle, and RC cancels idle GPU jobs (11680222 was cancelled after 69 min).
# Set REINSTALL_TORCH=1 to repair an interrupted torch install, CONDA_CLEAN=1 to free home quota afterwards.
#SBATCH --job-name=ngx_setup_env
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/niftygex/outputs/hpc/logs/%x_%j.log
set -eo pipefail                            # no -u: module and conda scripts use unset variables
umask 077                                   # the data and results stay private

ENV_DIR=/home/$USER/envs/niftygex
CODE_DIR=/home/$USER/niftygex
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

# Start clean: the login shell may activate another conda env (~/.bashrc) and SLURM copies that environment in.
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"
for _ in 1 2 3 4 5; do if [ -n "${CONDA_PREFIX:-}" ]; then conda deactivate; fi; done
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR=/scratch/$USER/niftygex/.pip-cache   # keep the ~50 GB home quota for the env

if [ ! -d "$ENV_DIR" ]; then
    conda env create --prefix "$ENV_DIR" --file "$CODE_DIR/env/environment-cuda.yml"
fi
conda activate "$ENV_DIR"
[ "$(command -v python)" = "$ENV_DIR/bin/python" ] || { echo "python is $(command -v python), not the niftygex env" >&2; exit 2; }
# The anaconda module puts its older libstdc++ on LD_LIBRARY_PATH; the env's own libraries must come first.
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pip install ${REINSTALL_TORCH:+--force-reinstall} "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
# conda clean scans the package caches on /home and took over 30 min (job 11682258); run it only when asked.
if [ -n "${CONDA_CLEAN:-}" ]; then conda clean --all --yes; fi
python -c "import torch, polars, numpy, yaml, scipy, sklearn; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"
cd "$CODE_DIR" && NIFTYGEX_PROFILE=hpc PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_nn_models.py tests/test_nn_trainer.py \
    tests/test_exploratory.py tests/test_hpc.py tests/test_intraday_targets.py tests/test_intraday_features.py
echo "environment ready: $ENV_DIR"
