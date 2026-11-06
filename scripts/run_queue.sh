#!/bin/bash
# Run a queue of commands one after another, keeping the Mac awake.
#
#   scripts/run_queue.sh QUEUE_FILE
#
# QUEUE_FILE: one job per line, "name: command" (blank lines and lines starting with # are skipped), e.g.
#   e3_attention_gru_1h: PYTHONPATH=src python scripts/run_intraday.py --horizon 1 --models attention_gru
#
# Each job runs from the repository root under `caffeinate -dimsu` and `/usr/bin/time -l`, with its output in
# outputs/logs/<name>_<timestamp>.log. The queue records start, end, exit code and peak memory per job, moves on
# to the next job when one fails, and writes outputs/logs/queue_<timestamp>_summary.tsv at the end. Restarting a
# queue reruns its jobs; the run scripts resume from their completed walks.
set -u

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: $0 QUEUE_FILE" >&2
    exit 2
fi
queue=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 2
mkdir -p outputs/logs
stamp=$(date +%Y%m%d_%H%M%S)
summary="outputs/logs/queue_${stamp}_summary.tsv"
printf "job\tstart\tend\tminutes\texit_code\tpeak_rss_gb\tpeak_footprint_gb\tlog\n" > "$summary"
failed=0

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    name=${line%%:*}
    command=${line#*:}
    name=$(echo "$name" | tr -d '[:space:]')
    log="outputs/logs/${name}_$(date +%Y%m%d_%H%M%S).log"
    start=$(date +%s)
    echo "[$(date '+%H:%M:%S')] start $name" | tee -a "$log"
    echo "command:$command" >> "$log"
    caffeinate -dimsu /usr/bin/time -l /bin/bash -c "$command" >> "$log" 2>&1
    code=$?
    end=$(date +%s)
    # /usr/bin/time -l reports bytes: "maximum resident set size" and "peak memory footprint".
    rss=$(awk '/maximum resident set size/ {printf "%.2f", $1 / 1073741824}' "$log" | tail -1)
    footprint=$(awk '/peak memory footprint/ {printf "%.2f", $1 / 1073741824}' "$log" | tail -1)
    minutes=$(awk -v s="$start" -v e="$end" 'BEGIN {printf "%.1f", (e - s) / 60}')
    echo "[$(date '+%H:%M:%S')] end $name: exit $code, $minutes min, peak RSS ${rss:-?} GB, footprint ${footprint:-?} GB" | tee -a "$log"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$(date -r "$start" '+%Y-%m-%d %H:%M:%S')" "$(date -r "$end" '+%Y-%m-%d %H:%M:%S')" \
        "$minutes" "$code" "${rss:-}" "${footprint:-}" "$log" >> "$summary"
    [ "$code" -eq 0 ] || failed=$((failed + 1))
done < "$queue"

echo
column -t -s "$(printf '\t')" "$summary"
echo "summary: $summary ($failed failed)"
[ "$failed" -eq 0 ]
