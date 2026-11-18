#!/usr/bin/env bash
# Submit one secondary-view task CSV using bounded local Slurm array indices.
set -euo pipefail
if (( $# != 4 )); then
  echo "Usage: bash scripts/submit_secondary_views.sh PLAN.csv cpu|bert MAX_CONCURRENT QOS" >&2
  exit 2
fi
plan="$1"
kind="$2"
concurrent="$3"
qos="$4"
test -s "$plan"
test -s results/evaluation/n2-secondary-v1/plans/plan_manifest.json
python_bin="$HOME/miniconda3/envs/horus_theme/bin/python"
test -x "$python_bin"
[[ "$concurrent" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$qos" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
case "$kind" in
  cpu|bert) ;;
  *) echo "Use cpu or bert" >&2; exit 2 ;;
esac
count="$("$python_bin" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], encoding="utf-8", newline=""))); assert rows; assert all((r["model"]=="bertimbau")== (sys.argv[2]=="bert") for r in rows); print(len(rows))' "$plan" "$kind")"
chunk=400
for ((first=0; first<count; first+=chunk)); do
  last=$((first+chunk-1))
  if ((last>=count)); then last=$((count-1)); fi
  local_last=$((last-first))
  if [[ "$kind" == bert ]]; then
    job="$(sbatch --parsable --qos="$qos" --time=2-00:00:00 --gres=gpu:L40S:1 \
      --array="0-$local_last%$concurrent" \
      --export=ALL,HORUS_PLAN="$plan",HORUS_ARRAY_OFFSET="$first" \
      scripts/slurm_secondary_views.sh)"
  else
    job="$(sbatch --parsable --qos="$qos" --time=24:00:00 \
      --array="0-$local_last%$concurrent" \
      --export=ALL,HORUS_PLAN="$plan",HORUS_ARRAY_OFFSET="$first" \
      scripts/slurm_secondary_views.sh)"
  fi
  echo "$plan indices $first-$last -> $job"
done
