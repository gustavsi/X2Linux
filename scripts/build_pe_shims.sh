#!/usr/bin/env bash
set -euo pipefail

PROFILE="debug"
CARGO_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
  PROFILE="release"
  CARGO_FLAG="--release"
fi

echo "[x2linux] Cross-compiling PE shim DLLs (x86_64-pc-windows-gnu, ${PROFILE})..."
cargo build --workspace ${CARGO_FLAG} \
  --manifest-path crates/x2l-pe-shims/Cargo.toml \
  --target x86_64-pc-windows-gnu

SRC_DIR="crates/x2l-pe-shims/target/x86_64-pc-windows-gnu/${PROFILE}"
OUT_DIR="target/pe-shims/${PROFILE}"
mkdir -p "${OUT_DIR}"

for dll in kernelx.dll xgameruntime.dll d3d12_xs.dll; do
  cp -f "${SRC_DIR}/${dll}" "${OUT_DIR}/${dll}"
done

echo "[x2linux] Done. Staged DLLs in ${OUT_DIR}/:"
ls -la "${OUT_DIR}"/*.dll
