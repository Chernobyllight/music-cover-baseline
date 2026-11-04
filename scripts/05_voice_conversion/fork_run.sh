#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/conda/bin:$PATH"
exec /opt/conda/bin/python "${PROJECT_ROOT}/scripts/fork_voice_run.py" "$@"
