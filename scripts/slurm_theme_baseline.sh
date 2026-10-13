#!/usr/bin/env bash
#SBATCH --job-name=horus-theme-tfidf
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/theme-baseline-%j.out
#SBATCH --error=results/logs/theme-baseline-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_eval/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

config="${HORUS_THEME_CONFIG:-experiments/01_quality_audit/human_theme_validation/configs/theme_cluster.yaml}"
test -f "$config"
for split in all train validation; do
    test -s "dataset_final_completo/data/${split}.parquet"
done

baseline_run="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" baseline)"
printf '%s\n' "$baseline_run" > "results/logs/theme-${SLURM_JOB_ID}.baseline_run.path"

predictions="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" baseline-predict --baseline-run "$baseline_run")"
printf '%s\n' "$predictions" > "results/logs/theme-${SLURM_JOB_ID}.baseline_predictions.path"

printf 'Baseline: %s\nPrevisoes: %s\n' "$baseline_run" "$predictions"
