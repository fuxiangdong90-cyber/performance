#!/usr/bin/env bash
# Validated on physical GPU 0 of an MTT S5000 host; see docs/musa-testing.md.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${OPBENCH_MUSA_IMAGE:-registry.mthreads.com/mcconline/musa-pytorch-release-public@sha256:58e39c5bd27eaae4df479aa0c36db6fb2afcd4390f11e658db606a4d71248d9f}
SDK=${OPBENCH_MUSA_SDK:-/usr/local/musa}
DRIVER=${OPBENCH_MUSA_DRIVER:-/usr/lib/x86_64-linux-gnu/libmusa.so.1}
GPU_NODE=${OPBENCH_MUSA_DEVICE_NODE:-/dev/mtgpu.0}
CARD=${OPBENCH_MUSA_DRM_CARD:-/dev/dri/card1}
RENDER=${OPBENCH_MUSA_DRM_RENDER:-/dev/dri/renderD128}
VISIBLE=${OPBENCH_MUSA_VISIBLE_DEVICES:-0}
for path in "$SDK/lib" "$DRIVER" "$GPU_NODE" "$CARD" "$RENDER"; do
  [[ -e "$path" ]] || { echo "Required runtime path is missing: $path" >&2; exit 1; }
done
args=(--rm --network none --cap-drop ALL --security-opt no-new-privileges --shm-size 1g
  --device "$GPU_NODE" --device "$CARD" --device "$RENDER"
  -v "$ROOT:/workspace" -w /workspace -v "$SDK:/opt/opbench-sdk:ro"
  -e LD_LIBRARY_PATH=/opt/opbench-driver:/opt/opbench-sdk/lib:/usr/local/musa/lib
  -e "MUSA_VISIBLE_DEVICES=$VISIBLE" --entrypoint python3)
# The vendor image contains driver placeholders. Some libraries use dlopen with
# an absolute filename, so both loader aliases and absolute paths are required.
for alias in libmusa.so libmusa.so.1 libmusa.so.4; do
  args+=(-v "$DRIVER:/opt/opbench-driver/$alias:ro"
         -v "$DRIVER:/usr/lib/x86_64-linux-gnu/$alias:ro")
done
exec docker run "${args[@]}" "$IMAGE" "$@"
