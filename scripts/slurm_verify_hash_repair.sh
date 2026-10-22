#!/usr/bin/env bash
#SBATCH --job-name=horus-hash-check
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/horus-hash-check-%j.out
#SBATCH --error=results/logs/horus-hash-check-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
n1_run="${HORUS_N1_RUN_DIR:?Set HORUS_N1_RUN_DIR to the verified N1 run}"
diagnostics="${HORUS_DIAGNOSTICS_DIR:?Set HORUS_DIAGNOSTICS_DIR to source diagnostics}"
test -s "$n1_run/verification.json"
test -s "$diagnostics/hash_mismatch_records.csv"
output="${HORUS_HASH_CHECK_OUTPUT:-results/normalization/hash-check-${SLURM_JOB_ID:?}}"
test ! -e "$output"
"$python_bin" -m horus_benchmark.data.verify_hash_repair \
  --config configs/experiments/corpus_normalization_n1.yaml \
  --diagnostics-dir "$diagnostics" --n1-run-dir "$n1_run" \
  --output-dir "$output"
printf 'Hash check complete: %s\n' "$output"
