#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <title_dir> [--snapshot <file> | --verify <file>]"
  exit 1
fi

TITLE_DIR="$1"
MODE="${2:---snapshot}"
OUT_FILE="${3:-/tmp/x2l_title_tree.sha256}"

compute_tree() {
  (cd "$TITLE_DIR" && find . -type f -print0 | sort -z | xargs -0 sha256sum)
}

if [[ "$MODE" == "--snapshot" ]]; then
  compute_tree > "$OUT_FILE"
  echo "[x2linux] Snapshot saved to $OUT_FILE ($(wc -l < "$OUT_FILE") files)"
elif [[ "$MODE" == "--verify" ]]; then
  compute_tree | diff -u "$OUT_FILE" -
  echo "[x2linux] PASS: Title directory is byte-for-byte identical."
else
  echo "Unknown mode: $MODE"
  exit 1
fi
