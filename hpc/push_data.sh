#!/bin/bash
# From the Mac: copy raw data, every processed Parquet stage (including the E3 intraday tables), the stage manifests
# and the reports the eval layer reads to Explorer scratch, then make it private (go-rwx).
#   bash hpc/push_data.sh                 (ssh host alias "explorer"; see hpc/hpc.yaml)
# Then verify on the cluster: bash hpc/remote.sh verify   (sizes, rows and SHA-256 against the manifests)
set -euo pipefail
ALIAS=${HPC_ALIAS:-explorer}
REMOTE_USER=${HPC_USER:-shinde.b}
SCRATCH=/scratch/$REMOTE_USER/niftygex
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
ssh "$ALIAS" "umask 077 && mkdir -p $SCRATCH/data $SCRATCH/outputs"
rsync -avP --exclude '.*.staging' --exclude 'sample_data' --exclude '.DS_Store' \
    data/raw data/curated data/greeks data/exposure data/features data/targets "$ALIAS:$SCRATCH/data/"
rsync -avP --exclude '.DS_Store' outputs/manifests outputs/reports "$ALIAS:$SCRATCH/outputs/"
ssh "$ALIAS" "chmod -R go-rwx $SCRATCH && ls -ld $SCRATCH $SCRATCH/data $SCRATCH/outputs"
echo "pushed to $ALIAS:$SCRATCH (private)"
