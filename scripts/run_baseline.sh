#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/YuE/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${PROJECT_ROOT}/envs/control/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=0
exec "${PROJECT_ROOT}/envs/control/bin/python" -m music_cover.pipeline "$@"
