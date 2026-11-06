#!/bin/bash
# From the Mac: copy raw data, every processed Parquet stage (including the locally built E3 intraday tables),
# the stage manifests and the reports the eval layer reads to /scratch/$HPC_USER/niftygex on Explorer.
#   HPC_USER=<neu_username> bash hpc/push_data.sh
# Then verify on the cluster: sbatch hpc/verify_transfer.sbatch (row counts and checksums against the manifests).
set -euo pipefail
HPC_USER=${HPC_USER:?set HPC_USER to your Explorer username}
HOST=${HPC_HOST:-login.explorer.northeastern.edu}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$HPC_USER@$HOST:/scratch/$HPC_USER/niftygex"
cd "$ROOT"
rsync -avP --exclude '.*.staging' --exclude 'sample_data' --exclude '.DS_Store' \
    data/raw data/curated data/greeks data/exposure data/features data/targets "$DEST/data/"
rsync -avP --exclude '.DS_Store' outputs/manifests outputs/reports "$DEST/outputs/"
echo "pushed to $DEST"
