#!/usr/bin/env bash
#SBATCH --job-name=horus-iid-freeze
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/horus-iid-freeze-%j.out
#SBATCH --error=results/logs/horus-iid-freeze-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from benchmark repository root}"
python_bin="${HORUS_NORMALIZE_PYTHON:-$HOME/miniconda3/envs/horus_theme/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

release="results/releases/horus-n2-iid-v1"
test ! -e "$release" || { echo "Release already exists: $release" >&2; exit 2; }

"$python_bin" -m horus_benchmark.protocols.freeze_iid freeze \
  --config configs/experiments/freeze_grouped_iid_n2_v1.yaml \
  --output "$release"
"$python_bin" -m horus_benchmark.protocols.freeze_iid check \
  --release-dir "$release"
echo "Frozen and checked: $release"
