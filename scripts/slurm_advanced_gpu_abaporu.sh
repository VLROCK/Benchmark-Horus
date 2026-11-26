#!/usr/bin/env bash
#SBATCH --job-name=horus-adv-gpu
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-adv-gpu-%A_%a.out
#SBATCH --error=results/logs/horus-adv-gpu-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Benchmark-Horus root}"
source scripts/abaporu_advanced_env.sh
test -x "$HORUS_ADV_PYTHON"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
nvidia-smi -L
index=$((HORUS_ARRAY_OFFSET + SLURM_ARRAY_TASK_ID))
"$HORUS_ADV_PYTHON" -m horus_benchmark.evaluation.advanced_campaign worker \
  --plan "$HORUS_PLAN" --index "$index"
