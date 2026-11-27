# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M1.1 Complete (`x2l-resolver`, `x2linux inspect`, synthetic `x2l-testtitle.exe`, and live `x2l-vfs-overlay` immutability verification)**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` remains strictly excluded).
- Verified M1.1 Deliverables:
  - `crates/x2l-resolver`: bounds-checked read-only PE32+ (AMD64) parser (adapted from `xcl-dev/xcl` `6089122a1c96dff16e93b6fdf04c8ed20cfb6102` with GPL-3.0 §5 provenance header), regular + delay-load imports, ordinal imports, static TLS & relocation detection, and data-driven import policy (`data/aliases.toml`, `data/ordinals/*.toml`).
  - `crates/x2l-cli`: `x2linux inspect <exe> [--json] [--aliases <path>] [--ordinals-dir <path>]` wired strictly read-only.
  - `crates/x2l-testtitle`: cross-compiled `x86_64-pc-windows-gnu` synthetic console fixture declaring `kernelx.dll` (named + ordinal `#999`), `xgameruntime.dll`, `d3d12_xs.dll`, and static TLS (`./scripts/build_testtitle.sh`).
  - `crates/x2l-vfs-overlay`: real unprivileged `CLONE_NEWUSER | CLONE_NEWNS` + `MS_PRIVATE | MS_REC` Kernel OverlayFS mounting with automatic `fuse-overlayfs` fallback and end-to-end SHA-256 read-only `lowerdir` immutability integration test (`readonly_immutability`).

## Next Engineering Tasks (M1.2)
1. Wire `x2l-pe-shims` (`x86_64-pc-windows-gnu`) ↔ `libx2l_core.so` via `__wine_unix_call` with dynamic x86-64 auto-stub thunks logging caller RVA into `report.json`.
2. Wire `x2linux run` to mount `x2l-vfs-overlay`, launch Wine + `/dev/ntsync`, and execute `x2l-testtitle.exe` end-to-end.
3. Import generic `x2l-formats` and `x2l-pipeline` crates from audited `GPL-3.0-or-later` upstreams with GPL-3.0 Section 5 headers logged in `NOTICE` and `THIRD_PARTY.md`.
