#!/usr/bin/env bash
#SBATCH --job-name=horus-iid-src-v3
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/horus-iid-src-v3-%j.out
#SBATCH --error=results/logs/horus-iid-src-v3-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

selection="${HORUS_SELECTION_RUN:-results/selection/candidates-126604}"
baseline="${HORUS_IID_BASELINE_RUN:-results/protocols/grouped-iid-layout-127131}"
raw="${HORUS_RAW:-dataset_final_completo/data/all.parquet}"
n2="${HORUS_N2_RUN:-results/normalization/full-n2-116035}"
n3="${HORUS_N3_RUN:-results/normalization/full-n3s-126552}"
for input in "$selection/summary.json" "$selection/candidate_membership.parquet" \
             "$selection/lineage_manifest.parquet" "$selection/dialogue_marker_loss.csv" \
             "$baseline/summary.json" "$baseline/holdout_provenance_seed20261105.csv" \
             "$raw" "$n2/normalized_corpus.parquet" "$n3/normalized_corpus.parquet"; do
  test -s "$input" || { echo "Missing input: $input" >&2; exit 2; }
done

"$python_bin" -m horus_benchmark.audit.refine_grouped_iid \
  --config configs/experiments/grouped_iid_source_balance_v3.yaml \
  --baseline-config configs/experiments/grouped_iid_layout_control_v2.yaml \
  --selection "$selection" --baseline "$baseline" --raw "$raw" \
  --n2-run "$n2" --n3-run "$n3" \
  --output "results/protocols/grouped-iid-source-v3-${SLURM_JOB_ID:?}"
