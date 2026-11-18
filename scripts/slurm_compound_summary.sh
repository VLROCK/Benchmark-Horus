#!/usr/bin/env bash
#SBATCH --job-name=horus-n2-joint-summary
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-n2-joint-summary-%A_%a.out
#SBATCH --error=results/logs/horus-n2-joint-summary-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="$HOME/miniconda3/envs/horus_theme/bin/python"
test -x "$python_bin"
task_index=$((HORUS_ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$python_bin" -m horus_benchmark.evaluation.summarize_compound_ood shard \
  --config configs/experiments/compound_ood_summary_n2_v1.yaml \
  --index "$task_index" \
  --output-root results/evaluation/n2-compound-v1/shards
