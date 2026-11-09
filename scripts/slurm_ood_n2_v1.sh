#!/usr/bin/env bash
#SBATCH --job-name=horus-ood-n2
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/horus-ood-n2-%j.out
#SBATCH --error=results/logs/horus-ood-n2-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
output="results/protocols/ood-n2-v1"
test ! -e "$output" || { echo "OOD output already exists: $output" >&2; exit 2; }
"$python_bin" -m horus_benchmark.protocols.ood_manifests build \
  --config configs/experiments/ood_n2_v1.yaml --output "$output"
"$python_bin" -m horus_benchmark.protocols.ood_manifests check --output "$output"
echo "OOD manifests built and checked: $output"

