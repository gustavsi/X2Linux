#!/usr/bin/env bash
set -euo pipefail

PROFILE="debug"
CARGO_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
  PROFILE="release"
  CARGO_FLAG="--release"
fi

echo "[x2linux] Cross-compiling synthetic console fixture x2l-testtitle.exe (x86_64-pc-windows-gnu, ${PROFILE})..."
cargo build ${CARGO_FLAG} \
  --manifest-path crates/x2l-testtitle/Cargo.toml \
  --target x86_64-pc-windows-gnu

echo "[x2linux] Done. Artifact: crates/x2l-testtitle/target/x86_64-pc-windows-gnu/${PROFILE}/x2l-testtitle.exe"
