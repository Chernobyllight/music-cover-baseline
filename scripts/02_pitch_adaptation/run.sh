#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${PROJECT_ROOT}/scripts/run_baseline.sh" --stage 02_pitch_adaptation "$@"
