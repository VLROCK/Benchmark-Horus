#!/usr/bin/env bash
# Source this after "ssh abaporu"; do not depend on HOME pointing at /mnt.
export HORUS_ABAPORU_HOME="${HORUS_ABAPORU_HOME:-/mnt/recod/homes/victor.roquetto}"
export HORUS_ADV_ENV="${HORUS_ADV_ENV:-$HORUS_ABAPORU_HOME/venvs/horus_advanced}"
export HORUS_ADV_PYTHON="${HORUS_ADV_PYTHON:-$HORUS_ADV_ENV/bin/python}"
export HF_HOME="${HORUS_HF_HOME:-$HORUS_ABAPORU_HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HORUS_HF_HUB:-$HF_HOME/hub}"
export STANZA_RESOURCES_DIR="${HORUS_STANZA_HOME:-$HORUS_ABAPORU_HOME/.cache/stanza_horus}"
export HORUS_ADV_CONFIG="${HORUS_ADV_CONFIG:-configs/experiments/advanced_detectors_n2_v1.yaml}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
