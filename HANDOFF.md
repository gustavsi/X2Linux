# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M1.2 Complete — Wine/`ntsync` Launcher, In-Process PE Shims ↔ `libx2l_core.so` Bridge, Dynamic Auto-Stubbing, Offline HLE (`kernelx`, `XGameRuntime`, Null `d3d12_xs`), and End-to-End `x2l-testtitle.exe` Boot**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` remains strictly excluded).
- Verified M1.2 Deliverables:
  - **Policy Cleanup:** `api-ms-win-core-synch-` → `system:kernelbase`, `userenv.dll` recognized as a system DLL, and `[implemented]` table populated in `data/aliases.toml` (zero false-positive stubs on `x2l-testtitle.exe`).
  - **Version-Independent In-Process Bridge:** `LD_PRELOAD` `libx2l_core.so` exports `x2l_bridge_dispatch` (`extern "win64"`) via `.init_array` (`X2L_BRIDGE_FN`) with `OP_HANDSHAKE` magic verification and `.fini_array` + `DLL_PROCESS_DETACH` `report.json` flush.
  - **Console PE Shims (`crates/x2l-pe-shims`):** Cross-compiled `kernelx.dll` (with `kernelx.def` NONAME ordinal `#999` auto-stub), `xgameruntime.dll` (offline-only, `XStore` returns `E_NOTIMPL`), and `d3d12_xs.dll` (Null M1 `ID3D12DeviceX` with W^X x86-64 Win64 vtable probe thunks), staged into `target/pe-shims/<profile>/`.
  - **End-to-End `x2linux run`:** Mounts `lowerdir` read-only via `x2l-vfs-overlay` (`KernelOverlayFs` → `fuse-overlayfs`), configures isolated Wine prefix (`G:`, `S:`, `P:`, `T:`), enables `NTSYNC=1` when `/dev/ntsync` is present, boots `x2l-testtitle.exe`, verifies `report.json` (`kernelx.dll#999` + `ID3D12DeviceX[45]`) and 100% SHA-256 `lowerdir` immutability.

## Next Engineering Tasks (M1.3)
1. Port generic `x2l-formats` (GPU texture detiling, parallel zlib/BCPack decompression, Opus/XMA2 audio decoding) from audited `GPL-3.0-or-later` upstreams (`Marvisak/VIctory` / `xcl-dev/xcl`) with GPL-3.0 §5 provenance headers logged in `NOTICE` and `THIRD_PARTY.md`.
2. Port `x2l-pipeline` content-addressed XDG shader/pipeline disk cache (`~/.cache/x2linux/<title>/spirv/`).
3. Begin M2 `x2l-rdna2-spirv` direct `rspirv` binary SPIR-V 1.6 emitter + `spirv-val` validation.
