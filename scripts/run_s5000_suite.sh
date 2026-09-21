#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
SUITE=${1:-standard}
[[ "$SUITE" == smoke || "$SUITE" == standard ]] || { echo 'Usage: bash scripts/run_s5000_suite.sh [smoke|standard]'; exit 1; }
export OPBENCH_MUSA_IMAGE=${OPBENCH_MUSA_IMAGE:-opbench/musa:s5000-arch31}
docker image inspect "$OPBENCH_MUSA_IMAGE" >/dev/null
NAME="s5000-arch31-${SUITE}-$(date +%Y%m%d-%H%M%S)"
echo "Native arch31 backend; addbmm uses labelled GPU composition; CPU reference checks enabled."
bash scripts/run_musa_container.sh -m opbench.runner \
  --template "templates/$SUITE.json" --device musa:0 --backend-module torch_musa \
  --backend-version 'torch_musa 2.5.0+aed8b42 rebuilt arch31; host SDK 4.3.8; driver 3.3.8-server' \
  --addbmm-implementation composite --verify-reference \
  --warmup 10 --iterations 50 --name "$NAME" --output "results/$NAME.json"
