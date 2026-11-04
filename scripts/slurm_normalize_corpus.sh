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
  n1|n2|n3|n3s) config="configs/experiments/corpus_normalization_${stage}.yaml" ;;
  *) echo "HORUS_NORMALIZE_STAGE must be n1, n2, n3, or n3s" >&2; exit 2 ;;
esac
output="${HORUS_NORMALIZE_OUTPUT:-results/normalization/full-${stage}-${SLURM_JOB_ID:?}}"
test ! -e "$output"
"$python_bin" -m horus_benchmark.data.normalize_corpus --config "$config" --output-dir "$output"
verify_args=()
if [[ "$stage" == "n3s" ]]; then
  n2_run="${HORUS_N2_RUN:-results/normalization/full-n2-116035}"
  test -s "$n2_run/normalized_corpus.parquet"
  test -s "$n2_run/verification.json"
  verify_args=(--baseline-n2-run "$n2_run")
fi
"$python_bin" -m horus_benchmark.data.verify_normalization --config "$config" --run-dir "$output" "${verify_args[@]}"
"$python_bin" -m horus_benchmark.data.normalization_report --run-dir "$output"
review_args=()
if [[ "$stage" == "n3" || "$stage" == "n3s" ]]; then
  review_args=(--per-cell 4 --maximum 720)
fi
"$python_bin" -m horus_benchmark.data.normalization_review \
  --config "$config" --run-dir "$output" "${review_args[@]}"
"$python_bin" -m horus_benchmark.data.prepare_normalized_audit \
  --normalization-config "$config" --run-dir "$output" \
  --base-audit-config configs/experiments/quality_audit_full.yaml
printf 'Normalization completed and verified: %s\n' "$output"
