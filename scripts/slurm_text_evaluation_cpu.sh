#!/usr/bin/env bash
#SBATCH --job-name=horus-text-cpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-text-cpu-%A_%a.out
#SBATCH --error=results/logs/horus-text-cpu-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
: "${HORUS_PLAN:?Set HORUS_PLAN to a CSV produced by benchmark_runner plan}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
array_offset="${HORUS_ARRAY_OFFSET:-0}"
[[ "$array_offset" =~ ^(0|[1-9][0-9]*)$ ]] || {
  echo "HORUS_ARRAY_OFFSET must be a nonnegative decimal integer" >&2
  exit 2
}
task_index=$((array_offset + SLURM_ARRAY_TASK_ID))
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$python_bin" -m horus_benchmark.evaluation.benchmark_runner run-task \
  --config configs/experiments/text_evaluation_n2_v1.yaml \
  --prepared-dir results/evaluation/n2-v1/prepared \
  --plan "$HORUS_PLAN" --task-index "$task_index" \
  --output-root "${HORUS_RUN_ROOT:-results/evaluation/n2-v1/runs}"

