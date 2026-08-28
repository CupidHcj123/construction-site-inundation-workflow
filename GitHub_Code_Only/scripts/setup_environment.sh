#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-xiongan}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_DIR}"

echo "[xiongan] project dir: ${PROJECT_DIR}"
echo "[xiongan] env name: ${ENV_NAME}"

CONDA_BIN="${CONDA_BIN:-}"
if [[ -z "${CONDA_BIN}" && -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/miniconda3/bin/conda"
fi
if [[ -z "${CONDA_BIN}" && -x "${HOME}/anaconda3/bin/conda" ]]; then
  CONDA_BIN="${HOME}/anaconda3/bin/conda"
fi
if [[ -z "${CONDA_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi

if [[ -n "${CONDA_BIN}" ]]; then
  echo "[xiongan] conda: ${CONDA_BIN}"
  CONDA_BASE="$("${CONDA_BIN}" info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[xiongan] updating existing conda env..."
    conda env update -n "${ENV_NAME}" -f environment.yml
  else
    echo "[xiongan] creating conda env..."
    conda env create -n "${ENV_NAME}" -f environment.yml
  fi

  echo "[xiongan] running environment check..."
  conda run -n "${ENV_NAME}" python scripts/check_xiongan_env.py
  echo "[xiongan] activate with: conda activate ${ENV_NAME}"
  exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

VENV_DIR=".venv-${ENV_NAME}"
echo "[xiongan] conda not found; creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/check_xiongan_env.py
echo "[xiongan] activate with: source ${PROJECT_DIR}/${VENV_DIR}/bin/activate"
