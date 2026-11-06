#!/bin/bash
# Summary of the exploratory cluster jobs: queue, recent accounting, and done / failed / attempt markers.
#   bash hpc/status.sh [days]        (light; fine on the login node)
DAYS=${1:-3}
BASE=/scratch/$USER/niftygex/outputs/hpc
echo "== queue"
squeue -u "$USER" -o "%.10i %.10P %.36j %.8T %.10M %.10l %R"
echo "== last $DAYS days (ngx_ jobs)"
sacct -u "$USER" -S "$(date -d "-$DAYS days" +%F)" --format=JobID%12,JobName%36,Partition%12,State%12,Elapsed,MaxRSS,ExitCode -X \
    | awk 'NR <= 2 || /ngx_/'
echo "== markers"
total=$(ls "$BASE/jobs"/*.sbatch 2>/dev/null | wc -l)
done=$(ls "$BASE/status"/*.done 2>/dev/null | wc -l)
failed=$(ls "$BASE/status"/*.failed 2>/dev/null | wc -l)
echo "jobs $total, done $done, failed $failed"
for f in "$BASE/status"/*.failed; do [ -e "$f" ] && echo "  failed: $(basename "$f" .failed) (exit $(cat "$f"))"; done
grep -h . "$BASE/submissions.tsv" 2>/dev/null | tail -5 | sed 's/^/  submitted: /'
