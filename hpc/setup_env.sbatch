#!/bin/bash
# Build the niftygex conda environment on a GPU compute node and check CUDA (docs/hpc.md, step 2).
#   sbatch hpc/setup_env.sbatch            (from /home/$USER/niftygex after hpc/sync_code.sh)
# Settings match hpc/hpc.yaml; override with TORCH_VERSION / TORCH_INDEX if the driver check below fails.
#SBATCH --job-name=ngx_setup_env
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
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
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ ! -d "$ENV_DIR" ]; then
    conda env create --prefix "$ENV_DIR" --file "$CODE_DIR/env/environment-cuda.yml"
fi
conda activate "$ENV_DIR"
[ "$(command -v python)" = "$ENV_DIR/bin/python" ] || { echo "python is $(command -v python), not the niftygex env" >&2; exit 2; }
pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
conda clean --all --yes   # home quota is ~50 GB

python - <<'EOF'
import platform, torch, polars, numpy
assert torch.cuda.is_available(), "torch does not see the GPU: check the driver above, or set TORCH_INDEX to an older CUDA build"
x = torch.randn(256, 256, device="cuda")
assert torch.isfinite(x @ x).all()
print(f"python {platform.python_version()}, torch {torch.__version__} (CUDA {torch.version.cuda}), polars {polars.__version__}, "
      f"numpy {numpy.__version__}, GPU {torch.cuda.get_device_name(0)}")
EOF
cd "$CODE_DIR" && NIFTYGEX_PROFILE=hpc PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_nn_models.py tests/test_nn_trainer.py tests/test_exploratory.py tests/test_hpc.py
echo "environment ready: $ENV_DIR"
