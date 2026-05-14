#!/usr/bin/env bash
# Source this file: `. env/setup_env.sh` (or `source env/setup_env.sh` in bash)
# Sets up the NAVSIM / nuPlan environment variables + activates the venv.
#
# IMPORTANT: paths below must stay aligned with Neymar's data download layout.
# If you change OPENSCENE_DATA_ROOT or the sub-dir convention, also update:
#   - docs/dep_pinning_notes.md
#   - scripts/download_navsim_mini.sh (Neymar owns)

# Resolve repo root robustly whether sourced from bash, zsh, or POSIX sh
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _SETUP_ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _SETUP_ENV_FILE="${(%):-%x}"
else
    # POSIX fallback — assume we're sourced from the repo root or env/
    _SETUP_ENV_FILE="${PWD}/env/setup_env.sh"
    [ -f "${_SETUP_ENV_FILE}" ] || _SETUP_ENV_FILE="${PWD}/setup_env.sh"
fi
_REPO_ROOT="$(cd "$(dirname "${_SETUP_ENV_FILE}")/.." && pwd)"

# --- venv ---
if [ -f "${_REPO_ROOT}/env/py310-navsim-cu128/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${_REPO_ROOT}/env/py310-navsim-cu128/bin/activate"
else
    echo "[setup_env] WARN: venv ${_REPO_ROOT}/env/py310-navsim-cu128 not found"
fi

# --- NAVSIM / nuPlan env vars ---
# Data lives at /home/work/hanjianhua/data — owned by Neymar's download script.
# Layout (per NAVSIM install.md):
#   /home/work/hanjianhua/data/
#     ├── maps/                          <- nuplan maps (v1.0)
#     ├── navsim_logs/{mini,trainval,test,...}/
#     ├── sensor_blobs/{mini,trainval,...}/
#     ├── navhard_two_stage/             <- challenge data (optional now)
#     └── exp/                           <- our experiment outputs (NAVSIM_EXP_ROOT)
#
# NOTE: ping Neymar if his download script lays things out differently.
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/work/hanjianhua/data/maps"
export OPENSCENE_DATA_ROOT="/home/work/hanjianhua/data"
export NAVSIM_EXP_ROOT="/home/work/hanjianhua/data/exp"
export NAVSIM_DEVKIT_ROOT="${_REPO_ROOT}/third_party/navsim"

# Make sure the exp dir exists (NAVSIM writes here on first run)
mkdir -p "${NAVSIM_EXP_ROOT}" 2>/dev/null || true

# --- CUDA / runtime knobs ---
# Blackwell (sm_120) — make sure compiled-on-the-fly kernels know our arch.
export TORCH_CUDA_ARCH_LIST="12.0"
# Avoid OOM-on-fragmentation surprises in long-running training/eval.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- HF token (data download convenience) ---
_TOKENS_ENV="/root/.openclaw/agents/main/workspace/.secrets/tokens.env"
if [ -f "${_TOKENS_ENV}" ]; then
    # shellcheck disable=SC1090
    . "${_TOKENS_ENV}"
fi

echo "[setup_env] venv:           $(command -v python)"
echo "[setup_env] python:         $(python --version 2>&1)"
echo "[setup_env] NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "[setup_env] OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "[setup_env] NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
echo "[setup_env] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "[setup_env] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
