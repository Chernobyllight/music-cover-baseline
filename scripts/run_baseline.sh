#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(uname -s)" == Darwin ]]; then
    mac_runner="$PROJECT_ROOT/envs/mps-optimized/scripts/run_baseline.sh"
    if [[ -f "$mac_runner" ]]; then
        exec bash "$mac_runner" "$@"
    fi
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/YuE/src${PYTHONPATH:+:${PYTHONPATH}}"
    export PATH="${PROJECT_ROOT}/envs/control/bin:$PATH"
    export PYTORCH_ENABLE_MPS_FALLBACK=0
else
    export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi
exec "${PROJECT_ROOT}/envs/control/bin/python" -m music_cover.pipeline "$@"
