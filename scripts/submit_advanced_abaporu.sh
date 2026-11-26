#!/usr/bin/env bash
# Usage: bash scripts/submit_advanced_abaporu.sh STAGE cpu|gpu CONCURRENCY
# Completed commands are skipped; incomplete work retains the exact plan.
set -euo pipefail
if (( $# != 3 )); then
  echo "Usage: bash scripts/submit_advanced_abaporu.sh STAGE cpu|gpu CONCURRENCY" >&2
  exit 2
fi
stage="$1"
kind="$2"
concurrent="$3"
source scripts/abaporu_advanced_env.sh
test -x "$HORUS_ADV_PYTHON"
[[ "$stage" =~ ^[a-z][a-z-]*$ ]] || exit 2
[[ "$concurrent" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$kind" in
  cpu|gpu) ;;
  *) echo "Kind must be cpu or gpu" >&2; exit 2 ;;
esac
mkdir -p results/logs results/tmp results/evaluation/n2-advanced-v1/jobs
plan="results/evaluation/n2-advanced-v1/jobs/$stage.csv"
if [[ ! -f "$plan" ]]; then
  args=(jobs --config "$HORUS_ADV_CONFIG" --stage "$stage" --output "$plan"
        --resources "$STANZA_RESOURCES_DIR")
  if [[ -n "${HORUS_SUMMARY_PLAN:-}" ]]; then args+=(--summary-plan "$HORUS_SUMMARY_PLAN"); fi
  "$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.advanced_campaign "${args[@]}"
fi
# Reattaching a terminal must not duplicate already queued/running shards.
if [[ -s "results/tmp/advanced-$stage.jobs" ]]; then
  prior_ids="$(paste -sd, "results/tmp/advanced-$stage.jobs")"
  active="$(squeue -h -j "$prior_ids" -o '%i %T' 2>/dev/null || true)"
  if [[ -n "$active" ]]; then
    echo "Reusing active allocations for $stage:"
    echo "$active"
    exit 0
  fi
fi
partitions=()
qos=()
if [[ -n "${HORUS_ADV_PARTITION:-}" ]]; then partitions=(--partition="$HORUS_ADV_PARTITION"); fi
if [[ -n "${HORUS_ADV_QOS:-}" ]]; then qos=(--qos="$HORUS_ADV_QOS"); fi
while IFS='|' read -r offset indices; do
  [[ -n "$indices" ]] || continue
  job="$(sbatch --parsable "${partitions[@]}" "${qos[@]}" \
    --array="$indices%$concurrent" \
    --export=ALL,HORUS_PLAN="$plan",HORUS_ARRAY_OFFSET="$offset" \
    "scripts/slurm_advanced_${kind}_abaporu.sh")"
  echo "$stage offset=$offset -> $job"
  echo "$job" >> "results/tmp/advanced-$stage.jobs"
done < <("$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.advanced_campaign pending --plan "$plan")
