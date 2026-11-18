#!/usr/bin/env bash
#SBATCH --job-name=horus-n2-prep
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/horus-n2-prep-%j.out
#SBATCH --error=results/logs/horus-n2-prep-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python_bin="$HOME/miniconda3/envs/horus_theme/bin/python"
test -x "$python_bin"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "HORUS_STAGE=$HORUS_STAGE"
case "$HORUS_STAGE" in
  segments)
    "$python_bin" -m horus_benchmark.protocols.segment_views prepare \
      --config configs/experiments/segment_views_n2_v1.yaml \
      --output results/protocols/n2-segments-v1
    ;;
  segments_check)
    "$python_bin" -m horus_benchmark.protocols.segment_views check \
      --config configs/experiments/segment_views_n2_v1.yaml \
      --output results/protocols/n2-segments-v1
    ;;
  layout)
    "$python_bin" -m horus_benchmark.audit.secondary_controls prepare-layout \
      --config configs/experiments/secondary_controls_n2_v1.yaml \
      --output results/controls/n2-v1/layout
    ;;
  layout_check)
    "$python_bin" -m horus_benchmark.audit.secondary_controls check \
      --config configs/experiments/secondary_controls_n2_v1.yaml \
      --run results/controls/n2-v1/layout
    ;;
  nuisance)
    "$python_bin" -m horus_benchmark.audit.secondary_controls prepare-nuisance \
      --config configs/experiments/secondary_controls_n2_v1.yaml \
      --segments-run results/protocols/n2-segments-v1 \
      --output results/controls/n2-v1/nuisance
    ;;
  nuisance_check)
    "$python_bin" -m horus_benchmark.audit.secondary_controls check \
      --config configs/experiments/secondary_controls_n2_v1.yaml \
      --run results/controls/n2-v1/nuisance
    ;;
  compound_ood)
    "$python_bin" -m horus_benchmark.protocols.compound_ood prepare \
      --config configs/experiments/compound_ood_n2_v1.yaml \
      --output results/protocols/compound-ood-n2-v1
    ;;
  compound_check)
    "$python_bin" -m horus_benchmark.protocols.compound_ood check \
      --output results/protocols/compound-ood-n2-v1
    ;;
  compound_texts)
    "$python_bin" -m horus_benchmark.evaluation.benchmark_runner prepare \
      --config configs/experiments/text_evaluation_compound_n2_v1.yaml \
      --output results/evaluation/n2-compound-v1/prepared
    ;;
  secondary_plan)
    "$python_bin" -m horus_benchmark.evaluation.secondary_views plan \
      --config configs/experiments/secondary_views_n2_v1.yaml \
      --output-dir results/evaluation/n2-secondary-v1/plans
    ;;
  compound_plan)
    mkdir -p results/evaluation/n2-compound-v1/plans
    "$python_bin" -m horus_benchmark.evaluation.benchmark_runner plan \
      --config configs/experiments/text_evaluation_compound_n2_v1.yaml \
      --output results/evaluation/n2-compound-v1/plans/logo_lodo_cpu.csv \
      --protocols logo_lodo \
      --models surface_all word_lr char4_svm human_iforest
    "$python_bin" -m horus_benchmark.evaluation.benchmark_runner plan \
      --config configs/experiments/text_evaluation_compound_n2_v1.yaml \
      --output results/evaluation/n2-compound-v1/plans/logo_lodo_bert.csv \
      --protocols logo_lodo --models bertimbau
    ;;
  compound_summary_plan)
    "$python_bin" -m horus_benchmark.evaluation.summarize_compound_ood plan \
      --config configs/experiments/compound_ood_summary_n2_v1.yaml \
      --output results/evaluation/n2-compound-v1/plans/shards.csv
    ;;
  secondary_merge)
    "$python_bin" -m horus_benchmark.evaluation.summarize_secondary_views merge \
      --shards-root results/evaluation/n2-secondary-v1/shards \
      --output results/evaluation/n2-secondary-v1/summary
    ;;
  compound_merge)
    "$python_bin" -m horus_benchmark.evaluation.summarize_compound_ood merge \
      --config configs/experiments/compound_ood_summary_n2_v1.yaml \
      --shard-root results/evaluation/n2-compound-v1/shards \
      --output results/evaluation/n2-compound-v1/summary
    ;;
  *)
    echo "Unknown HORUS_STAGE; inspect scripts/slurm_secondary_prepare.sh" >&2
    exit 2
    ;;
esac
