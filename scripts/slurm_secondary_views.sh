#!/usr/bin/env bash
#SBATCH --job-name=horus-n2-views
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=results/logs/horus-n2-views-%A_%a.out
#SBATCH --error=results/logs/horus-n2-views-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f "$HORUS_PLAN"
python_bin="$HOME/miniconda3/envs/horus_theme/bin/python"
test -x "$python_bin"
task_index=$((HORUS_ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$python_bin" -m horus_benchmark.evaluation.secondary_views run-task \
  --config configs/experiments/secondary_views_n2_v1.yaml \
  --plan "$HORUS_PLAN" --task-index "$task_index" \
  --output-root results/evaluation/n2-secondary-v1/runs
