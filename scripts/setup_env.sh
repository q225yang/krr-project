#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="krr"
ENV_FILE="environment.yml"

echo "==> Loading mamba module"
module load mamba

# Sanity check
if [ ! -f "${ENV_FILE}" ]; then
  echo "ERROR: ${ENV_FILE} not found in current directory: $(pwd)"
  exit 1
fi

echo "==> Checking if env '${ENV_NAME}' exists"
if mamba info --envs | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Env exists. Updating from ${ENV_FILE}"
  mamba env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "==> Env not found. Creating from ${ENV_FILE}"
  mamba env create -f "${ENV_FILE}"
fi

echo "==> Activating env: ${ENV_NAME}"
source activate "${ENV_NAME}"

echo "==> Ensuring spaCy English model is installed (en_core_web_sm)"
python -m spacy download en_core_web_sm

echo "==> Done."
echo "Activate later with: module load mamba && source activate ${ENV_NAME}"