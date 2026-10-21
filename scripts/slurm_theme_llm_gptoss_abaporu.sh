#!/usr/bin/env bash
#SBATCH --job-name=horus-gptoss-theme
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=results/logs/horus-gptoss-theme-%A_%a.out
#SBATCH --error=results/logs/horus-gptoss-theme-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Benchmark-Horus root}"
abaporu_home="${HORUS_ABAPORU_HOME:-/mnt/recod/homes/victor.roquetto}"
python_bin="${HORUS_GPTOSS_PYTHON:-$abaporu_home/miniconda3/envs/horus_gptoss/bin/python}"
test -x "$python_bin"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${HORUS_FULL_INDEX:-}" ]]; then
  resolved="$("$python_bin" -m horus_benchmark.audit.theme_llm_full_shard \
    --index "$HORUS_FULL_INDEX" --shard-id "${SLURM_ARRAY_TASK_ID:?}" --model-key gpt_oss_120b)"
  requests="${resolved%%$'\n'*}"
  output_dir="${resolved#*$'\n'}"
else
  requests="${HORUS_REQUESTS_PATH:?Prepare blind requests first}"
  output_dir="${HORUS_GPTOSS_OUTPUT:-results/theme_llm_600/inference/gpt_oss_120b}"
fi
test -s "$requests"
repo="${HORUS_HF_HUB:-$abaporu_home/.cache/huggingface/hub}/models--openai--gpt-oss-120b"
revision="${HORUS_GPTOSS_REVISION:-$(cat "$repo/refs/main")}"
checkpoint="$repo/snapshots/$revision"
test -s "$checkpoint/config.json"
toolchain="${HORUS_TRITON_TOOLCHAIN:-$abaporu_home/tools/triton-toolchain}"
export CC="$toolchain/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$toolchain/bin/x86_64-conda-linux-gnu-g++"
test -x "$CC"
test -x "$CXX"
export PATH="$toolchain/bin:$PATH"
export TRITON_CACHE_DIR="${HORUS_TRITON_CACHE:-$abaporu_home/.cache/triton}"
export HF_HOME="${HORUS_HF_HOME:-$abaporu_home/.cache/huggingface}"
export HF_HUB_CACHE="${HORUS_HF_HUB:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset MODEL_PATH PYTHON_BIN GDN_PREFILL_BACKEND VLLM_USE_DEEP_GEMM
unset VLLM_USE_FLASHINFER_SAMPLER LIMIT_MM_IMAGE LIMIT_MM_AUDIO LIMIT_MM_VIDEO
nvidia-smi -L
git_revision="unavailable"
if command -v git >/dev/null 2>&1; then
  git_revision="$(git rev-parse HEAD)"
fi
printf 'Git: %s\nGPT-OSS checkpoint: %s\n' "$git_revision" "$checkpoint"
args=(--requests "$requests" --model-key gpt_oss_120b --checkpoint "$checkpoint" --output-dir "$output_dir")
if [[ -n "${HORUS_SMOKE_LIMIT:-}" ]]; then
  args+=(--limit "$HORUS_SMOKE_LIMIT" --limit-strategy "${HORUS_SMOKE_STRATEGY:-first}")
fi
"$python_bin" -m horus_benchmark.audit.theme_llm_worker "${args[@]}"
