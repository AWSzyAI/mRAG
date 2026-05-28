#!/usr/bin/env bash
set -euo pipefail

# Usage: bash infra/setup_remote_env.sh [ENV_NAME]
ENV_NAME=${1:-llava}
REPO_DIR=${2:-$PWD}

echo "Setting up remote environment in ${REPO_DIR} (env: ${ENV_NAME})"

pushd "$REPO_DIR" >/dev/null

# Determine environment activation method: conda or venv fallback
ACTIVATE_CMD=""
if command -v conda &>/dev/null; then
  echo "Found conda in PATH. Using conda environment."
  CONDA_BASE=$(conda info --base)
  ACTIVATE_CMD="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
else
  echo "Conda not found in PATH. Falling back to python -m venv at .venv_${ENV_NAME}"
  if [ ! -d ".venv_${ENV_NAME}" ]; then
    set +e
    python3 -m venv ".venv_${ENV_NAME}" 2>/tmp/venv_err || true
    VENV_RET=$?
    set -e
    if [ $VENV_RET -ne 0 ]; then
      echo "python -m venv failed, attempting to install virtualenv via pip --user and create virtualenv instead"
      python3 -m pip install --user virtualenv || true
      python3 -m virtualenv ".venv_${ENV_NAME}" || {
        echo "virtualenv creation also failed. See /tmp/venv_err for details" >&2
      }
    fi
  fi
  ACTIVATE_CMD="source $REPO_DIR/.venv_${ENV_NAME}/bin/activate"
  # If venv activation script missing, fall back to user-site pip installs
  if [ ! -f "$REPO_DIR/.venv_${ENV_NAME}/bin/activate" ]; then
    echo "Virtualenv activation script missing; falling back to pip --user installs (no venv)."
    ACTIVATE_CMD=""
    if ! python3 -m pip --version >/dev/null 2>&1; then
      echo "pip not found. Attempting to bootstrap pip via get-pip.py"
      curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py || true
      python3 /tmp/get-pip.py --user || true
    fi
    python3 -m pip install --user --upgrade pip || true
    if [ -f requirements.txt ]; then
      python3 -m pip install --user -r requirements.txt || true
    fi
    python3 -m pip install --user "torch>=2.4.0" || true
  fi
fi

echo "Ensure submodules and common repos are present..."
git submodule update --init --recursive || true


if command -v conda &>/dev/null; then
  echo "Create conda env from infra/gemma4_env.yml (if not exists)"
  if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda env $ENV_NAME already exists. Skipping creation."
  else
    conda env create -f infra/gemma4_env.yml -n "$ENV_NAME" || {
      echo "conda env create failed; attempting minimal env creation and pip install fallback"
      conda create -y -n "$ENV_NAME" python=3.10 pip
      source $(conda info --base)/etc/profile.d/conda.sh
      conda activate "$ENV_NAME"
      pip install --upgrade pip
      pip install -r requirements.txt || true
    }
  fi
  echo "Activating conda env $ENV_NAME"
  source ${CONDA_BASE}/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"
else
  echo "Using venv at .venv_${ENV_NAME} and installing dependencies via pip"
  # Activate venv and install
  source "$REPO_DIR/.venv_${ENV_NAME}/bin/activate"
  pip install --upgrade pip
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt || true
  fi
fi

echo "Install/upgrade torch suitable for the server."
echo "If you have CUDA, please install the appropriate wheel (example: conda install -c pytorch pytorch=2.4 cudatoolkit=12.1)."
echo "Attempting pip install torch>=2.4 (may require manual wheel for GPU)"
pip install "torch>=2.4.0" || echo "pip install torch failed — please install a wheel matching your CUDA/runtime"

echo "Install MagicLens and LLaVA-NeXT if missing (clone into github/)"
mkdir -p github
pushd github >/dev/null
if [ ! -d LLaVA-NeXT ]; then
  echo "Cloning LLaVA-NeXT"
  git clone https://github.com/robustnessai/LLaVA-NeXT.git LLaVA-NeXT || true
fi
if [ ! -d magiclens ]; then
  echo "Cloning MagicLens"
  git clone https://github.com/example/magiclens.git magiclens || true
fi
popd >/dev/null

echo "Try basic imports to validate environment"
python - <<'PY'
import sys
try:
  import importlib
  importlib.import_module('magiclens')
  print('MagicLens import OK')
except Exception as e:
  print('MagicLens import failed:', e, file=sys.stderr)
try:
  importlib.import_module('llava')
  print('LLaVA-NeXT import OK')
except Exception as e:
  print('LLaVA-NeXT import failed:', e, file=sys.stderr)
PY

echo "Setup script finished."
popd >/dev/null
