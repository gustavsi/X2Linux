# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M0 Bootstrap Complete**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` remains strictly excluded).
- Workspace: All 14 Linux workspace crates + 2 excluded PE-side crates (`x2l-pe-shims`, `x2l-testtitle`) scaffolded with working M0 foundations (`xtask`, `x2l-common`, `x2l-cli`, `x2l-vfs-overlay`).

## Next Engineering Tasks (M0 → M1)
1. Run `cargo xtask install-hooks` on every developer clone to activate the pre-commit asset-policy guard.
2. Verify read-only isolation end-to-end with `./scripts/verify_readonly.sh` against a synthetic temp fixture mounted via `x2l-vfs-overlay`.
3. Import generic `x2l-formats`, `x2l-pipeline`, and `x2l-resolver` modules from audited `GPL-3.0-or-later` upstreams with GPL-3.0 Section 5 headers and commit SHAs logged in `NOTICE` and `THIRD_PARTY.md`.
4. Stand up `x2l-pe-shims` (`x86_64-pc-windows-gnu`) ↔ `libx2l_core.so` via `__wine_unix_call` and boot `x2l-testtitle` under Wine + `/dev/ntsync`.
