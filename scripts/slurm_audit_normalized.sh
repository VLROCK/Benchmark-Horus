#!/usr/bin/env bash
#SBATCH --job-name=horus-audit-norm
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=results/logs/horus-audit-norm-%j.out
#SBATCH --error=results/logs/horus-audit-norm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
normalization_run="${HORUS_NORMALIZE_RUN:?Set the verified normalization run path}"
test -s "$normalization_run/verification.json"
test -s "$normalization_run/audit_experiment.yaml"
audit_run="$("$python_bin" -m horus_benchmark.audit.cli run \
  --config "$normalization_run/audit_experiment.yaml")"
test -s "$audit_run/audit_summary.json"
printf 'Normalized audit: %s\n' "$audit_run"
if [[ -n "${HORUS_ORIGINAL_AUDIT_RUN:-}" ]]; then
  test -s "$HORUS_ORIGINAL_AUDIT_RUN/audit_summary.json"
  comparison="results/audits/comparison-${SLURM_JOB_ID:?}"
  "$python_bin" -m horus_benchmark.audit.cli compare \
    --before "$HORUS_ORIGINAL_AUDIT_RUN" --after "$audit_run" --output "$comparison"
  printf 'Paired comparison: %s\n' "$comparison"
fi
