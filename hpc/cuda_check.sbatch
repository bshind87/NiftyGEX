#!/bin/bash
# A short GPU check after hpc/setup_env.sbatch: the driver, torch.cuda and a matmul on a v100-sxm2.
#SBATCH --job-name=ngx_cuda_check
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/%u/niftygex/outputs/hpc/logs/%x_%j.log
set -eo pipefail                            # no -u: module and conda scripts use unset variables
umask 077
# Start clean: the login shell may activate another conda env (~/.bashrc) and SLURM copies that environment in.
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"
for _ in 1 2 3 4 5; do if [ -n "${CONDA_PREFIX:-}" ]; then conda deactivate; fi; done
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
conda activate /home/$USER/envs/niftygex
[ "$(command -v python)" = "/home/$USER/envs/niftygex/bin/python" ] || { echo "python is $(command -v python), not the niftygex env" >&2; exit 2; }
# The anaconda module puts its older libstdc++ on LD_LIBRARY_PATH; the env's own libraries must come first.
export LD_LIBRARY_PATH="/home/$USER/envs/niftygex/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<'PY'
import platform, torch
assert torch.cuda.is_available(), "torch does not see the GPU: set TORCH_INDEX to an older CUDA build and rerun hpc/setup_env.sbatch"
x = torch.randn(2048, 2048, device="cuda")
assert torch.isfinite(x @ x).all()
print(f"python {platform.python_version()}, torch {torch.__version__} (CUDA {torch.version.cuda}), GPU {torch.cuda.get_device_name(0)}, "
      f"compute capability {torch.cuda.get_device_capability(0)}")
PY
