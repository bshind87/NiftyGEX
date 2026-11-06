#!/bin/bash
# From the Mac: copy the cluster's results store, job logs, status markers and reports back (scratch is purged each
# semester), then merge the seeds into the local results store.
#   bash hpc/pull_results.sh [--no-merge]          (ssh host alias "explorer")
set -euo pipefail
ALIAS=${HPC_ALIAS:-explorer}
REMOTE_USER=${HPC_USER:-shinde.b}
SRC="$ALIAS:/scratch/$REMOTE_USER/niftygex/outputs"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p outputs/hpc_pull
for part in results hpc reports; do     # one call per folder: macOS rsync takes a single remote source
    rsync -avP --exclude 'checkpoints' "$SRC/$part" outputs/hpc_pull/
done
if [ "${1:-}" != "--no-merge" ]; then
    PYTHONPATH=src python scripts/merge_seeds.py --source outputs/hpc_pull/results
fi
