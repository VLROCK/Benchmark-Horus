#!/usr/bin/env bash
#SBATCH --job-name=horus-text-prepare
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/horus-text-prepare-%j.out
#SBATCH --error=results/logs/horus-text-prepare-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
output="results/evaluation/n2-v1/prepared"
test ! -e "$output" || { echo "Prepared cache already exists: $output" >&2; exit 2; }
"$python_bin" -m horus_benchmark.evaluation.benchmark_runner prepare \
  --config configs/experiments/text_evaluation_n2_v1.yaml \
  --output "$output" --with-n3s
echo "Selected text cache prepared: $output"

