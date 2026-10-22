#!/usr/bin/env bash
#SBATCH --job-name=horus-n2-inspect
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/horus-n2-inspect-%j.out
#SBATCH --error=results/logs/horus-n2-inspect-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
test -s dataset_final_completo/data/all.parquet
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mode="${HORUS_DIAGNOSTIC_MODE:-source}"
case "$mode" in
  source)
    output="${HORUS_DIAGNOSTIC_OUTPUT:-results/normalization/diagnostics-${SLURM_JOB_ID:?}}"
    test ! -e "$output"
    "$python_bin" -m horus_benchmark.data.normalization_diagnostics \
      --config configs/experiments/corpus_normalization_n1.yaml \
      --output-dir "$output" --academic-sample 120 --hash-sample 36
    ;;
  preview)
    n1_run="${HORUS_N1_RUN_DIR:?Set HORUS_N1_RUN_DIR to the verified N1 run}"
    test -s "$n1_run/verification.json"
    output="${HORUS_DIAGNOSTIC_OUTPUT:-results/normalization/n2-preview-${SLURM_JOB_ID:?}}"
    test ! -e "$output"
    "$python_bin" -m horus_benchmark.data.preview_n2 \
      --config configs/experiments/corpus_normalization_n2_preview.yaml \
      --n1-run-dir "$n1_run" --output-dir "$output" --examples-per-kind 80
    ;;
  *)
    echo "HORUS_DIAGNOSTIC_MODE must be source or preview" >&2
    exit 2
    ;;
esac
printf 'Diagnostic complete: %s\n' "$output"
