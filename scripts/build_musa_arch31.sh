#!/usr/bin/env bash
# Requires the matching vendor torch_musa and vendor-patched PyTorch source.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SOURCE=${OPBENCH_TORCH_MUSA_SOURCE:-$ROOT/vendor-source/home/torch_musa}
PYTORCH=${OPBENCH_PYTORCH_SOURCE:-$ROOT/vendor-source/home/pytorch}
SDK=${OPBENCH_MUSA_SDK:-/usr/local/musa}
DRIVER=${OPBENCH_MUSA_DRIVER:-/usr/lib/x86_64-linux-gnu/libmusa.so.1}
BASE=${OPBENCH_MUSA_BUILD_BASE:-registry.mthreads.com/mcconline/musa-pytorch-release-public@sha256:58e39c5bd27eaae4df479aa0c36db6fb2afcd4390f11e658db606a4d71248d9f}
OUTPUT=${OPBENCH_MUSA_OUTPUT_IMAGE:-opbench/musa:s5000-arch31}
[[ -f "$SOURCE/setup.py" && -f "$PYTORCH/aten/src/ATen/native/native_functions.yaml" ]] || {
  echo 'Matching vendor sources are required; see docs/musa-testing.md' >&2; exit 1;
}
grep -q '_scaled_dot_product_attention_flash_musa' "$PYTORCH/aten/src/ATen/native/native_functions.yaml" || {
  echo 'PyTorch source is missing the matching vendor patches' >&2; exit 1;
}
args=(--rm --network none --cap-drop ALL --security-opt no-new-privileges
  -v "$SOURCE:/workspace/vendor-source/home/torch_musa"
  -v "$PYTORCH:/workspace/vendor-source/home/pytorch"
  -v "$SDK:/opt/opbench-sdk:ro"
  -w /workspace/vendor-source/home/torch_musa --entrypoint /bin/bash
  -e MUSA_HOME=/opt/opbench-sdk -e TORCH_MUSA_ARCH_LIST=31
  -e "MAX_JOBS=${MAX_JOBS:-32}" -e BUILD_TEST=0 -e USE_KINETO=0 -e USE_MCCL=1
  -e PYTORCH_REPO_PATH=/workspace/vendor-source/home/pytorch
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e CPLUS_INCLUDE_PATH=/usr/local/musa/include
  -e PATH=/opt/opbench-sdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  -e LD_LIBRARY_PATH=/opt/opbench-driver:/opt/opbench-sdk/lib:/usr/local/musa/lib)
for alias in libmusa.so libmusa.so.1 libmusa.so.4; do
  args+=(-v "$DRIVER:/opt/opbench-driver/$alias:ro"
         -v "$DRIVER:/usr/lib/x86_64-linux-gnu/$alias:ro")
done
docker run "${args[@]}" "$BASE" -c 'python setup.py bdist_wheel'
CONTEXT=$(mktemp -d /tmp/opbench-musa-image.XXXXXX)
trap 'rm -rf -- "$CONTEXT"' EXIT
cp "$SOURCE"/dist/torch_musa*.whl "$CONTEXT/"
docker build --network none --build-arg "BASE_IMAGE=$BASE" -f "$ROOT/Dockerfile.musa" -t "$OUTPUT" "$CONTEXT"
echo "Built $OUTPUT. Verify get_arch_list() before benchmarking."
