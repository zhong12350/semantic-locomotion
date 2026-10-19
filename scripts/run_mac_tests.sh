#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
export PYTHONPATH="$ROOT:$REPO/official/rsl_rl:${PYTHONPATH:-}"
python3 -m unittest discover -s "$ROOT/tests" -v
