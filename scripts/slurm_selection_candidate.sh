#!/usr/bin/env bash
#SBATCH --job-name=horus-select-cand
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-select-cand-%j.out
#SBATCH --error=results/logs/horus-select-cand-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

raw="${HORUS_RAW:-dataset_final_completo/data/all.parquet}"
n2="${HORUS_N2_RUN:-results/normalization/full-n2-116035}"
n3="${HORUS_N3_RUN:-results/normalization/full-n3s-126552}"
near="${HORUS_NEAR_PAIRS:-results/audits/20261013T180229080021Z_quality-shortcuts-full-v1_1727096142/tables/near_duplicate_pairs.csv}"
for input in "$raw" "$n2/normalized_corpus.parquet" "$n3/normalized_corpus.parquet" "$near"; do
  test -s "$input" || { echo "Missing input: $input" >&2; exit 2; }
done

"$python_bin" -m horus_benchmark.audit.selection_candidate \
  --config configs/experiments/selection_candidate_n2_v1.yaml \
  --raw "$raw" --n2-run "$n2" --n3-run "$n3" --near-pairs "$near" \
  --output "results/selection/candidates-${SLURM_JOB_ID:?}"
