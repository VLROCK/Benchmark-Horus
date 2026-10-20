#!/usr/bin/env bash
#SBATCH --job-name=horus-normalize
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=results/logs/horus-normalize-%j.out
#SBATCH --error=results/logs/horus-normalize-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
test -s dataset_final_completo/data/all.parquet
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

stage="${HORUS_NORMALIZE_STAGE:-n1}"
case "$stage" in
  n1|n2) config="configs/experiments/corpus_normalization_${stage}.yaml" ;;
  *) echo "HORUS_NORMALIZE_STAGE must be n1 or n2" >&2; exit 2 ;;
esac
output="${HORUS_NORMALIZE_OUTPUT:-results/normalization/full-${stage}-${SLURM_JOB_ID:?}}"
test ! -e "$output"
"$python_bin" -m horus_benchmark.data.normalize_corpus --config "$config" --output-dir "$output"
"$python_bin" -m horus_benchmark.data.verify_normalization --config "$config" --run-dir "$output"
"$python_bin" -m horus_benchmark.data.normalization_report --run-dir "$output"
"$python_bin" -m horus_benchmark.data.normalization_review --config "$config" --run-dir "$output"
"$python_bin" -m horus_benchmark.data.prepare_normalized_audit \
  --normalization-config "$config" --run-dir "$output" \
  --base-audit-config configs/experiments/quality_audit_full.yaml
printf 'Normalization completed and verified: %s\n' "$output"
