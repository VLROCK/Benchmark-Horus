#!/usr/bin/env bash
# Submit bounded local array indices; global index is passed separately.
set -euo pipefail
if (( $# != 2 )); then
  echo "Usage: bash scripts/submit_compound_summary.sh MAX_CONCURRENT QOS" >&2
  exit 2
fi
concurrent="$1"
qos="$2"
[[ "$concurrent" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$qos" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
plan=results/evaluation/n2-compound-v1/plans/shards.csv
test -s "$plan"
python_bin="$HOME/miniconda3/envs/horus_theme/bin/python"
test -x "$python_bin"
count="$("$python_bin" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], encoding="utf-8", newline=""))); assert rows; print(len(rows))' "$plan")"
chunk=400
for ((first=0; first<count; first+=chunk)); do
  last=$((first+chunk-1))
  if ((last>=count)); then last=$((count-1)); fi
  local_last=$((last-first))
  job="$(sbatch --parsable --qos="$qos" --array="0-$local_last%$concurrent" \
    --export=ALL,HORUS_ARRAY_OFFSET="$first" \
    scripts/slurm_compound_summary.sh)"
  echo "Compound summary indices $first-$last -> $job"
done
