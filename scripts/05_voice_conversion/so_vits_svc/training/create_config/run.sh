#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec /opt/conda/bin/python -m music_cover.svc_config training "$@"
