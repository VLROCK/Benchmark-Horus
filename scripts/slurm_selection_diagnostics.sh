#!/usr/bin/env bash
#SBATCH --job-name=horus-select-diag
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-select-diag-%j.out
#SBATCH --error=results/logs/horus-select-diag-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
n2_run="${HORUS_N2_RUN:-results/normalization/full-n2-116035}"
n3_run="${HORUS_N3_RUN:?Set HORUS_N3_RUN to the new synthetic-only N3 run}"
test -s "$n2_run/normalized_corpus.parquet"
test -s "$n3_run/normalized_corpus.parquet"
test -s "$n2_run/verification.json"
test -s "$n3_run/verification.json"
"$python_bin" -m horus_benchmark.audit.selection_diagnostics \
  --config configs/experiments/selection_diagnostics_n2_n3.yaml \
  --n2-run "$n2_run" --n3-run "$n3_run" \
  --output "results/selection/diagnostics-${SLURM_JOB_ID:?}"
