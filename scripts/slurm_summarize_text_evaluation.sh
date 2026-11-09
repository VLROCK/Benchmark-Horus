#!/usr/bin/env bash
#SBATCH --job-name=horus-text-summary
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/horus-text-summary-%j.out
#SBATCH --error=results/logs/horus-text-summary-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
plans="results/evaluation/n2-v1/plans"
"$python_bin" -m horus_benchmark.evaluation.summarize_benchmarks \
  --plans \
    "$plans/iid_cpu.csv" "$plans/iid_bert.csv" \
    "$plans/logo_cpu.csv" "$plans/lodo_cpu.csv" "$plans/loso_cpu.csv" \
    "$plans/logo_bert.csv" "$plans/lodo_bert.csv" "$plans/loso_bert.csv" \
  --runs results/evaluation/n2-v1/runs \
  --output results/evaluation/n2-v1/summary
