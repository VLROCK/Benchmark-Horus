#!/usr/bin/env bash
#SBATCH --job-name=horus-view-probes
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/horus-view-probes-%j.out
#SBATCH --error=results/logs/horus-view-probes-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
args=(
  --config configs/experiments/compare_normalization_views.yaml
  --raw dataset_final_completo/data/all.parquet
  --n1-run "${HORUS_N1_RUN:-results/normalization/full-n1-115189}"
  --n2-run "${HORUS_N2_RUN:-results/normalization/full-n2-116035}"
)
if [[ -n "${HORUS_N3_RUN:-}" ]]; then
  test -s "$HORUS_N3_RUN/verification.json"
  args+=(--n3-run "$HORUS_N3_RUN")
  output="results/audits/normalization-view-probes-n3-${SLURM_JOB_ID:?}"
else
  output="results/audits/normalization-view-probes-${SLURM_JOB_ID:?}"
fi
"$python_bin" -m horus_benchmark.audit.compare_normalization_views \
  "${args[@]}" --output "$output"
