#!/bin/bash
# From the Mac: copy raw data, every processed Parquet stage (including the E3 intraday tables), the stage manifests
# and the reports the eval layer reads to Explorer scratch, then make it private (go-rwx).
#   bash hpc/push_data.sh                 (ssh host alias "explorer-xfer"; see hpc/hpc.yaml and docs/hpc.md)
# Copies go through RC's data-transfer node (xfer.discovery.neu.edu, same /home and /scratch): rsync on the login node
# is cut off after about a minute. Each folder is retried up to 5 times; a dropped copy resumes where it stopped.
# Then verify on the cluster: bash hpc/remote.sh verify   (sizes, rows and SHA-256 against the manifests)
set -uo pipefail
ALIAS=${HPC_XFER:-explorer-xfer}
REMOTE_USER=${HPC_USER:-shinde.b}
SCRATCH=/scratch/$REMOTE_USER/niftygex
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
ssh "$ALIAS" "umask 077 && mkdir -p $SCRATCH/data $SCRATCH/outputs" || exit 1
complete=1
for src in data/raw data/curated data/greeks data/exposure data/features data/targets outputs/manifests outputs/reports; do
    for attempt in 1 2 3 4 5; do
        if rsync -a --partial --exclude '.*.staging' --exclude 'sample_data' --exclude '.DS_Store' \
                -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6" "$src" "$ALIAS:$SCRATCH/$(dirname "$src")/"; then
            echo "$(date +%H:%M:%S) $src copied (attempt $attempt)"
            break
        fi
        echo "$(date +%H:%M:%S) $src: attempt $attempt failed, retrying"
        sleep 10
        [ "$attempt" = 5 ] && complete=0
    done
done
# Always restore privacy, even after a failed copy.
ssh "$ALIAS" "chmod -R go-rwx $SCRATCH && echo \"readable by others: \$(find $SCRATCH -perm /077 | wc -l) entries\""
if [ "$complete" = 1 ]; then echo "pushed to $ALIAS:$SCRATCH (private)"; else echo "push INCOMPLETE: rerun this script" >&2; exit 1; fi
