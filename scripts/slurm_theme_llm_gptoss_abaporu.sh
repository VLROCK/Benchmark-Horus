#!/usr/bin/env bash
#SBATCH --job-name=horus-gptoss-theme
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=results/logs/horus-gptoss-theme-%j.out
#SBATCH --error=results/logs/horus-gptoss-theme-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Benchmark-Horus root}"
test -n "${HORUS_SAMPLE_PATH:-}"
test -s "$HORUS_SAMPLE_PATH"
python_bin="${HORUS_GPTOSS_PYTHON:-/mnt/recod/homes/victor.roquetto/miniconda3/envs/horus_gptoss/bin/python}"
test -x "$python_bin"
config="${HORUS_LLM_CONFIG:-experiments/01_quality_audit/theme_agreement_600/configs/theme_llm_600.yaml}"
output_dir="${HORUS_GPTOSS_OUTPUT:-results/theme_llm_600/inference/gpt_oss_120b}"
repo="${HORUS_HF_HUB:-$HOME/.cache/huggingface/hub}/models--openai--gpt-oss-120b"
revision="$(cat "$repo/refs/main")"
checkpoint="$repo/snapshots/$revision"
test -s "$checkpoint/config.json"
toolchain="${HORUS_TRITON_TOOLCHAIN:-$HOME/tools/triton-toolchain}"
export CC="$toolchain/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$toolchain/bin/x86_64-conda-linux-gnu-g++"
test -x "$CC"
test -x "$CXX"
export PATH="$toolchain/bin:$PATH"
export TRITON_CACHE_DIR="${HORUS_TRITON_CACHE:-$HOME/.cache/triton}"
export HF_HOME="${HORUS_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
unset MODEL_PATH PYTHON_BIN GDN_PREFILL_BACKEND VLLM_USE_DEEP_GEMM
unset VLLM_USE_FLASHINFER_SAMPLER LIMIT_MM_IMAGE LIMIT_MM_AUDIO LIMIT_MM_VIDEO
nvidia-smi -L
printf 'Git: %s\nGPT-OSS checkpoint: %s\n' "$(git rev-parse HEAD)" "$checkpoint"
args=(--config "$config" infer --sample "$HORUS_SAMPLE_PATH" --model-key gpt_oss_120b --checkpoint "$checkpoint" --output-dir "$output_dir")
if [[ -n "${HORUS_SMOKE_LIMIT:-}" ]]; then args+=(--limit "$HORUS_SMOKE_LIMIT"); fi
"$python_bin" -m horus_benchmark.audit.theme_llm_cli "${args[@]}"
