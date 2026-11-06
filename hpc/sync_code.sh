#!/bin/bash
# On the Explorer login node: clone or update the code at a committed revision and create the scratch layout.
#   bash sync_code.sh <commit>      (light git work only; no computation on the login node)
# The checkout must be clean: every cluster run records its commit and git_dirty.
set -euo pipefail
COMMIT=${1:?usage: sync_code.sh <commit>}
REPO=${NIFTYGEX_REPO:-https://github.com/bshind87/NiftyGEX.git}
CODE_DIR=/home/$USER/niftygex
SCRATCH=/scratch/$USER/niftygex

if [ ! -d "$CODE_DIR/.git" ]; then
    git clone "$REPO" "$CODE_DIR"
fi
cd "$CODE_DIR"
git fetch --quiet origin
git checkout --quiet "$COMMIT"
if [ -n "$(git status --porcelain)" ]; then
    echo "the checkout in $CODE_DIR has local changes; refusing to run from a dirty tree" >&2
    exit 1
fi
mkdir -p "$SCRATCH/data" "$SCRATCH/outputs/hpc/logs" "$SCRATCH/outputs/hpc/status" "$SCRATCH/outputs/hpc/jobs"
echo "code at $(git rev-parse --short HEAD) in $CODE_DIR; scratch layout in $SCRATCH"
