#!/usr/bin/env bash
# Run inside the repository after ssh abaporu. This script waits between stages.
# A failed stage exits this script (never the interactive parent terminal).
set -euo pipefail
phase="${1:-}"
source scripts/abaporu_advanced_env.sh
root=results/evaluation/n2-advanced-v1
mkdir -p results/logs results/tmp "$root/jobs"
cpu_concurrent="${HORUS_CPU_CONCURRENT:-4}"
gpu_concurrent="${HORUS_GPU_CONCURRENT:-2}"
stage() {
  local name="$1" hardware="$2" concurrent="$3" active ids left
  bash scripts/submit_advanced_abaporu.sh "$name" "$hardware" "$concurrent"
  ids="$(paste -sd, "results/tmp/advanced-$name.jobs" 2>/dev/null || true)"
  while true; do
    left="$("$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.advanced_campaign pending \
      --plan "$root/jobs/$name.csv")"
    if [[ -z "$left" ]]; then echo "Completed stage: $name"; return 0; fi
    active="$(squeue -h -j "$ids" -o '%i %T %R' 2>/dev/null || true)"
    if [[ -z "$active" ]]; then
      echo "Stage $name has unfinished commands and no active allocations." >&2
      sacct -j "$ids" --format=JobID,State,ExitCode,Elapsed >&2 || true
      echo "Inspect results/logs/horus-adv-*; rerun this phase after fixing the cause." >&2
      return 1
    fi
    echo "Waiting: $name"
    echo "$active"
    sleep 30
  done
}
check_detectors() {
  local name="$1" runs="$root/runs"
  if [[ "$name" == common ]]; then runs="$root/common_prefix/runs"; fi
  "$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.advanced_runner check \
    --plan "$root/plans/$name.csv" --output "$runs"
}
case "$phase" in
  install)
    /usr/bin/python3.11 -m venv "$HORUS_ADV_ENV"
    "$HORUS_ADV_PYTHON" -m pip install -e '.[advanced,dev]'
    "$HORUS_ADV_PYTHON" -m pip check
    "$HORUS_ADV_PYTHON" -m pytest -p no:cacheprovider -q \
      tests/test_advanced_campaign.py tests/test_classical_adapted.py \
      tests/test_lm_probability.py tests/test_ghostbuster_adapted.py \
      tests/test_summarize_advanced.py tests/test_robustness_attacks.py \
      tests/test_robustness_runner.py tests/test_robustness_summary.py
    "$HORUS_ADV_PYTHON" -m pip freeze > "$root/environment.freeze.txt"
    ;;
  download)
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    "$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.lm_cache download \
      --config configs/detectors/lm_pairs_v1.yaml --lock "$root/models.lock.json" \
      --cache-dir "$HF_HUB_CACHE"
    "$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.linguistic_cache download \
      --resources "$STANZA_RESOURCES_DIR"
    "$HORUS_ADV_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('neuralmind/bert-base-portuguese-cased',
                  revision='94d69c95f98f7d5b2a8700c420230ae10def0baa',
                  cache_dir=os.environ['HF_HUB_CACHE'],
                  allow_patterns=['config.json', 'vocab.txt', 'tokenizer*.json',
                                  'special_tokens_map.json', 'added_tokens.json'])
PY
    ;;
  prepare) stage prepare cpu 1 ;;
  pilot)
    stage lm-pilot gpu 1
    stage pos-pilot cpu 1
    stage lm-merge-pilot cpu 1
    stage pos-merge-pilot cpu 1
    stage cpu-pilot cpu 1
    echo "Resource pilot ready: $root/cache/pilot/*/shards/*.json and $root/cpu_pilot.json"
    ;;
  cache)
    stage lm-native gpu "$gpu_concurrent"
    stage pos-native cpu "$cpu_concurrent"
    stage lm-merge-native cpu "$cpu_concurrent"
    stage pos-merge-native cpu 1
    ;;
  iid|ood|compound)
    stage "$phase-plan" cpu 1
    stage "$phase" cpu "$cpu_concurrent"
    check_detectors "$phase"
    ;;
  common)
    stage common-prefix cpu 1
    stage lm-common gpu "$gpu_concurrent"
    stage pos-common cpu "$cpu_concurrent"
    stage lm-merge-common cpu "$cpu_concurrent"
    stage pos-merge-common cpu 1
    stage common-plan cpu 1
    stage common cpu "$cpu_concurrent"
    check_detectors common
    stage contrast cpu "$cpu_concurrent"
    ;;
  attacks)
    stage attacks cpu 1
    "$HORUS_ADV_PYTHON" -m horus_benchmark.robustness.prepare check --output "$root/attacks"
    ;;
  attack-cache)
    stage lm-attacks gpu "$gpu_concurrent"
    stage pos-attacks cpu "$cpu_concurrent"
    stage lm-merge-attacks cpu "$cpu_concurrent"
    stage pos-merge-attacks cpu "$cpu_concurrent"
    ;;
  robustness)
    stage robustness-plan cpu 1
    stage robustness-cpu cpu "$cpu_concurrent"
    stage robustness-bert gpu 1
    "$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.robustness_runner check \
      --plan-dir "$root/plans/robustness" --output "$root/robustness/runs"
    stage robustness-summary cpu "$cpu_concurrent"
    stage robustness-merge cpu 1
    ;;
  reports)
    plans=()
    for name in iid ood compound; do
      check_detectors "$name"
      plans+=("$root/plans/$name.csv")
    done
    export HORUS_SUMMARY_PLAN="$root/plans/summary.csv"
    if [[ ! -f "$HORUS_SUMMARY_PLAN" ]]; then
      "$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.summarize_advanced plan \
        --config "$HORUS_ADV_CONFIG" --plan "${plans[@]}" --output "$HORUS_SUMMARY_PLAN"
    fi
    stage summary cpu "$cpu_concurrent"
    stage summary-merge cpu 1
    stage errors cpu 1
    ;;
  status)
    squeue -u "$USER"
    sacct -u "$USER" --starttime "$(date -d '14 days ago' +%F)" \
      --format=JobID,JobName%24,State,ExitCode,Elapsed
    for name in iid ood compound common; do
      if [[ -f "$root/plans/$name.csv" ]]; then check_detectors "$name" || true; fi
    done
    ;;
  *)
    echo "Usage: bash scripts/run_advanced_abaporu.sh install|prepare|download|pilot|cache|iid|ood|compound|common|attacks|attack-cache|robustness|reports|status" >&2
    exit 2
    ;;
esac
