#!/usr/bin/env bash
#SBATCH --job-name=horus-n2-view-summary
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-n2-view-summary-%A_%a.out
#SBATCH --error=results/logs/horus-n2-view-summary-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="$HOME/miniconda3/envs/horus_theme/bin/python"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
models=(surface_length surface_presentation surface_punctuation surface_all word_lr char4_lr char4_svm human_iforest bertimbau)
model="${models[SLURM_ARRAY_TASK_ID]:?Expected array index 0-8}"
"$python_bin" -m horus_benchmark.evaluation.summarize_secondary_views \
  --config configs/experiments/secondary_views_n2_v1.yaml \
  --plan-dir results/evaluation/n2-secondary-v1/plans \
  --runs results/evaluation/n2-secondary-v1/runs \
  --nuisance-membership results/controls/n2-v1/nuisance/nuisance_membership.parquet \
  --models "$model" \
  --output "results/evaluation/n2-secondary-v1/shards/$model"
