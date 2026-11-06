#!/bin/bash
# From the Mac: copy the cluster's results store, job logs and status markers back (scratch is purged each
# semester), then merge the seeds into the local results store.
#   HPC_USER=<neu_username> bash hpc/pull_results.sh [--no-merge]
set -euo pipefail
HPC_USER=${HPC_USER:?set HPC_USER to your Explorer username}
HOST=${HPC_HOST:-login.explorer.northeastern.edu}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$HPC_USER@$HOST:/scratch/$HPC_USER/niftygex/outputs"
cd "$ROOT"
mkdir -p outputs/hpc_pull
for part in results hpc reports; do     # one call per folder: macOS rsync takes a single remote source
    rsync -avP --exclude 'checkpoints' "$SRC/$part" outputs/hpc_pull/
done
if [ "${1:-}" != "--no-merge" ]; then
    PYTHONPATH=src python scripts/merge_seeds.py --source outputs/hpc_pull/results
fi
