#!/usr/bin/env bash
# Submit a precomputed CPU or BERTimbau plan as a bounded Slurm array.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/submit_text_evaluation_plan.sh PLAN.csv cpu|bert MAX_CONCURRENT" >&2
  exit 2
fi
plan="$1"
kind="$2"
concurrent="$3"
test -f "$plan"
test -f results/evaluation/n2-v1/prepared/prepared_manifest.json
[[ "$concurrent" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_CONCURRENT must be positive" >&2; exit 2; }
case "$kind" in
  cpu) slurm_script="scripts/slurm_text_evaluation_cpu.sh" ;;
  bert) slurm_script="scripts/slurm_text_evaluation_bert.sh" ;;
  *) echo "Kind must be cpu or bert" >&2; exit 2 ;;
esac
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export HORUS_PLAN="$plan" HORUS_PLAN_KIND="$kind"
count="$($python_bin - <<'PY'
import csv
import os

with open(os.environ['HORUS_PLAN'], encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit('Empty experiment plan')
neural = [row['model'] == 'bertimbau' for row in rows]
if os.environ['HORUS_PLAN_KIND'] == 'bert' and not all(neural):
    raise SystemExit('BERT plan includes a non-neural task')
if os.environ['HORUS_PLAN_KIND'] == 'cpu' and any(neural):
    raise SystemExit('CPU plan includes a BERTimbau task')
print(len(rows))
PY
)"
chunk_size="${HORUS_ARRAY_CHUNK_SIZE:-400}"
[[ "$chunk_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "HORUS_ARRAY_CHUNK_SIZE must be positive" >&2
  exit 2
}
for ((first = 0; first < count; first += chunk_size)); do
  last=$((first + chunk_size - 1))
  if ((last >= count)); then last=$((count - 1)); fi
  job_id="$(sbatch --parsable --array="${first}-${last}%${concurrent}" \
    --export=ALL,HORUS_PLAN="$plan" "$slurm_script")"
  echo "Submitted $kind tasks ${first}-${last} of $((count - 1)): $job_id"
done

