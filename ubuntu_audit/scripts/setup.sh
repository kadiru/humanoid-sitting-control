#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs tools cache/uv cache/python
exec > >(tee -a logs/setup.log) 2>&1
date -Is
# Fail before downloads when GPU or network access is unavailable.
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv
git ls-remote https://github.com/isaac-sim/IsaacLab.git refs/heads/release/3.0.0
available_kib=$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')
required_gib=70
[[ ! -d IsaacLab/.venv ]] || required_gib=15
if (( available_kib < required_gib * 1024 * 1024 )); then
  echo "Require $required_gib GiB free as a conservative installation/cache budget; inspect storage."
  exit 1
fi
export UV_CACHE_DIR="$PROJECT_DIR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/cache/python"
export UV_INSTALL_DIR="$PROJECT_DIR/tools"
export UV_NO_MODIFY_PATH=1
# Do not inherit another project's environment or Python import paths.
unset VIRTUAL_ENV CONDA_PREFIX PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_PROJECT_ENVIRONMENT="$PROJECT_DIR/IsaacLab/.venv"
export UV_LINK_MODE=hardlink
if [[ ! -x tools/uv ]]; then
  curl --fail --location https://astral.sh/uv/install.sh -o tools/install-uv.sh
  sh tools/install-uv.sh
fi
if [[ ! -d IsaacLab ]]; then
  git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git IsaacLab
fi
cd IsaacLab
git rev-parse HEAD | tee ../logs/isaaclab-commit.txt
git status --short | tee ../logs/isaaclab-status.txt
test -f uv.lock
sha256sum pyproject.toml uv.lock | tee ../logs/dependency-hashes.txt
# Record storage throughout downloads/unpacking, including failure cases.
monitor_storage() {
  while true; do
    date -Is
    df -Pk "$PROJECT_DIR"
    sleep 30
  done
}
monitor_storage > ../logs/storage.log &
storage_monitor_pid=$!
trap 'kill "$storage_monitor_pid" 2>/dev/null || true' EXIT
# Preserve the checkout's indexes, overrides and lock; fail rather than relock.
../tools/uv sync --locked --inexact --python 3.12 --extra isaacsim --extra rsl-rl --extra video
if [[ -f ../sitting_task/pyproject.toml ]]; then
  ../tools/uv pip install --python .venv/bin/python --no-deps --editable ../sitting_task
fi
.venv/bin/python --version | tee ../logs/python-version.txt
../tools/uv pip freeze --python .venv/bin/python > ../logs/resolved-packages.txt
.venv/bin/python - <<'PY'
from importlib.metadata import version
v = version('isaacsim')
print('Isaac Sim:', v)
if v not in ('6.1.0', '6.1.0.0'):
    raise SystemExit('Checkout does not resolve requested Isaac Sim 6.1.0; inspect before proceeding.')
PY
echo 'Installation complete. Inspect this exact revision before choosing validation commands.'
