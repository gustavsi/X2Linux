#!/usr/bin/env bash
set -euo pipefail

echo "[x2linux] Downloading official GNU GPL v3.0 text from gnu.org..."
curl -fsSL https://www.gnu.org/licenses/gpl-3.0.txt -o LICENSE
echo "[x2linux] Wrote LICENSE ($(wc -l < LICENSE) lines)."
