#!/usr/bin/env bash
#SBATCH --job-name=horus-theme
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/theme-%j.out
#SBATCH --error=results/logs/theme-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
python_bin="${HORUS_PYTHON_BIN:-$HOME/miniconda3/envs/horus_eval/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HORUS_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

config="${HORUS_THEME_CONFIG:-experiments/01_quality_audit/human_theme_validation/configs/theme_cluster.yaml}"
pilot_file="${HORUS_PILOT_PATH_FILE:-results/logs/theme_pilot.path}"
test -f "$config"
test -s "$pilot_file"

for split in all train validation test; do
    test -s "dataset_final_completo/data/${split}.parquet"
done

"$python_bin" - "$config" <<'PY'
import re
import sys

import torch
import transformers
import yaml
from huggingface_hub import hf_hub_download

if sys.version_info < (3, 10):
    raise SystemExit("Este projeto exige Python 3.10 ou superior")
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
name = cfg["model"]["name"]
revision = cfg["model"]["revision"]
if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise SystemExit("Fixe model.revision em um commit SHA de 40 caracteres antes do treino")
for filename in ("config.json", "vocab.txt", "pytorch_model.bin"):
    hf_hub_download(name, filename, revision=revision, local_files_only=True)
if not torch.cuda.is_available():
    raise SystemExit("PyTorch nao encontrou a GPU CUDA alocada")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Runtime: Python {sys.version.split()[0]}, PyTorch {torch.__version__}, Transformers {transformers.__version__}", flush=True)
print(f"Modelo: {name}@{revision}", flush=True)
PY

pilot="$(cat "$pilot_file")"
test -s "$pilot"
printf 'Git: %s\n' "$(git rev-parse HEAD)"
printf 'Piloto: %s\n' "$pilot"

model_run="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" train)"
printf '%s\n' "$model_run" > "results/logs/theme-${SLURM_JOB_ID}.model_run.path"

predictions="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" predict --model-run "$model_run")"
printf '%s\n' "$predictions" > "results/logs/theme-${SLURM_JOB_ID}.predictions.path"

manifest="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" sample --predictions "$predictions" --pilot "$pilot")"
printf '%s\n' "$manifest" > "results/logs/theme-${SLURM_JOB_ID}.sample_manifest.path"

printf 'Modelo: %s\nPrevisoes: %s\nAmostra: %s\n' "$model_run" "$predictions" "$manifest"
