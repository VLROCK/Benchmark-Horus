#!/usr/bin/env bash
#SBATCH --job-name=horus-bert-theme
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=results/logs/horus-bert-theme-%j.out
#SBATCH --error=results/logs/horus-bert-theme-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Benchmark-Horus root}"
python_bin="${HORUS_BERT_PYTHON:?Set an absolute Abaporu Python path with torch, transformers, pandas, pyarrow and sklearn}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HORUS_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
config="${HORUS_LLM_CONFIG:-experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_600.yaml}"
"$python_bin" - "$config" <<'PY'
import sys
import torch
import yaml
import pandas
import pyarrow
import sklearn
import transformers
from huggingface_hub import hf_hub_download
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in BERT allocation")
if not transformers.__version__.startswith("4."):
    raise SystemExit("The BERT pipeline requires Transformers 4.x")
for name in ("all", "train", "validation", "test"):
    from pathlib import Path
    if not Path(f"dataset_final_completo/data/{name}.parquet").is_file():
        raise SystemExit(f"Missing corpus split: {name}")
for name in ("config.json", "vocab.txt", "pytorch_model.bin"):
    hf_hub_download(cfg["model"]["name"], name, revision=cfg["model"]["revision"], local_files_only=True)
print("BERT GPU:", torch.cuda.get_device_name(0))
PY
if [[ -n "${HORUS_BERT_MODEL_RUN:-}" ]]; then
    model_run="$HORUS_BERT_MODEL_RUN"
    test -s "$model_run/run.json"
else
    model_run="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" train)"
fi
printf '%s\n' "$model_run" > "results/logs/horus-bert-theme-${SLURM_JOB_ID}.model_run.path"
predictions="$("$python_bin" -m horus_benchmark.audit.theme_cli --config "$config" predict --model-run "$model_run")"
printf '%s\n' "$predictions" > "results/logs/horus-bert-theme-${SLURM_JOB_ID}.predictions.path"
printf 'BERT predictions: %s\n' "$predictions"
