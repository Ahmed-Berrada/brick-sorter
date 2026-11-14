#!/usr/bin/env bash
# Start the station: loads the models, connects the cameras and the arm (after checking its calibration), then
# serves the control page on :8080. Press START on the page (or pass --start). Ctrl+C stops and holds the arm.
#   scripts/run.sh                 # uses config/local.toml if present
#   scripts/run.sh --step          # wait for NEXT between bricks
#   scripts/run.sh sim             # everything simulated (no hardware)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
UV=(uv run)
[ -f .venv/.xpu ] && UV=(uv run --no-sync)   # keep the Intel GPU PyTorch installed by setup_intel.sh --xpu
CFG=()
[ -f config/local.toml ] && CFG=(-c config/local.toml)
if [ "${1:-}" = sim ]; then shift; exec "${UV[@]}" brick-sorter "${CFG[@]}" sim "$@"; fi
exec "${UV[@]}" brick-sorter "${CFG[@]}" run "$@"
