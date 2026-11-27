#!/usr/bin/env bash
set -euo pipefail

PROFILE="debug"
CARGO_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
  PROFILE="release"
  CARGO_FLAG="--release"
fi

echo "[x2linux] Cross-compiling PE shim DLLs (x86_64-pc-windows-gnu, ${PROFILE})..."
cargo build ${CARGO_FLAG} \
  --manifest-path crates/x2l-pe-shims/Cargo.toml \
  --target x86_64-pc-windows-gnu

echo "[x2linux] Done. Artifacts in crates/x2l-pe-shims/target/x86_64-pc-windows-gnu/${PROFILE}/"
