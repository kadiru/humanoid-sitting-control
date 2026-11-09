#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$PROJECT_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/cache/python"
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/IsaacLab/.venv"
export XDG_CACHE_HOME="$PROJECT_DIR/cache/runtime"
export OMNI_KIT_ACCEPT_EULA=yes
export HEADLESS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4
export OMP_NUM_THREADS=8
cd "$PROJECT_DIR/IsaacLab"
exec "$PROJECT_DIR/tools/uv" run --no-sync "$@"
