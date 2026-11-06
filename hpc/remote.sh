#!/bin/bash
# From the Mac, over the ssh host alias "explorer" (light commands only; work runs in SLURM jobs):
#   bash hpc/remote.sh sync <commit>          clone/update ~/niftygex at a commit, create the private scratch layout
#   bash hpc/remote.sh setup                  submit the environment build + CUDA check (gpu-short, v100-sxm2)
#   bash hpc/remote.sh verify                 submit the transfer check (short)
#   bash hpc/remote.sh jobs [gpu_type]        write the job scripts (default v100-sxm2)
#   bash hpc/remote.sh scheduler [args...]    one scheduler pass (e.g. --waves calibration, --dry-run)
#   bash hpc/remote.sh loop                   submit the scheduler loop as a CPU job (hpc/scheduler.sbatch)
#   bash hpc/remote.sh status                 queue, accounting and markers
set -euo pipefail
ALIAS=${HPC_ALIAS:-explorer}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# A clean environment on every call: the login shell activates an unrelated conda env from ~/.bashrc.
MODULES='module purge >/dev/null 2>&1; module load explorer anaconda3/2024.06 >/dev/null 2>&1; eval "$(conda shell.bash hook)"; for _ in 1 2 3 4 5; do if [ -n "${CONDA_PREFIX:-}" ]; then conda deactivate; fi; done; unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1'
PY='conda activate ~/envs/niftygex && NIFTYGEX_PROFILE=hpc PYTHONPATH=src python'
command=${1:?usage: remote.sh sync|setup|verify|jobs|scheduler|loop|status}
shift
case "$command" in
    sync) ssh "$ALIAS" "bash -s -- ${1:?commit}" < "$ROOT/hpc/sync_code.sh" ;;
    setup) ssh "$ALIAS" "cd ~/niftygex && umask 077 && sbatch hpc/setup_env.sbatch" ;;
    verify) ssh "$ALIAS" "cd ~/niftygex && umask 077 && sbatch hpc/verify_transfer.sbatch" ;;
    jobs) ssh "$ALIAS" "cd ~/niftygex && umask 077 && $MODULES && $PY hpc/make_jobs.py --gpu-type ${1:-v100-sxm2}" ;;
    scheduler) ssh "$ALIAS" "cd ~/niftygex && umask 077 && $MODULES && $PY hpc/scheduler.py --once $*" ;;
    loop) ssh "$ALIAS" "cd ~/niftygex && umask 077 && sbatch hpc/scheduler.sbatch" ;;
    status) ssh "$ALIAS" "bash -s" < "$ROOT/hpc/status.sh" ;;
    *) echo "unknown command: $command" >&2; exit 2 ;;
esac
