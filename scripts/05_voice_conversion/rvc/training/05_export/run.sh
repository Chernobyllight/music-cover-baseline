#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PATH="/opt/conda/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec /opt/conda/bin/python -m music_cover.rvc_training --stage 05_export "$@"
