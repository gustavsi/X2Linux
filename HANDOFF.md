# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M1.3 Complete — `x2l-formats` (BCPack, Parallel Zlib, RDNA2 Texture Detiling, XMA2/Opus) & `x2l-pipeline` (XDG Content-Addressed Atomic Cache)**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` remains strictly excluded).
- Verified M1.3 Deliverables:
  - **`crates/x2l-formats`:** Adapted `bcpack.rs` (+ pure-Rust `bcpack_tables.rs`), `texture.rs` (RDNA2/GFX10.3 addrlib tiling/detiling), and `xma.rs` + `xma/tables.rs` (pure-Rust XMA2 PCM decoder) from `Marvisak/VIctory` (`63bcca43c6a00a758510398fa43096ce7e4f7df0`) with GPL-3.0 §5 provenance headers; added bounded single-stream & parallel multi-chunk `zlib` decompression (`zlib.rs`) and RFC 6716 `opus` framing + PipeWire-ready PCM helpers (`opus.rs`).
  - **`crates/x2l-pipeline`:** Adapted `Converter` trait and atomic write-then-rename content-addressed `Cache` from `Marvisak/VIctory` (`63bcca43c6a00a758510398fa43096ce7e4f7df0`), wired directly to `x2l_common::XdgPaths` (`$XDG_CACHE_HOME/x2linux/<title>/` and `.../spirv/`) with pure-Rust SHA-256 keys, `sync_all`, and repository-write refusal guard.
  - **Provenance:** Recorded exact commit SHA and adapted modules in `NOTICE` and `THIRD_PARTY.md`.

## Next Engineering Tasks (M2.1)
1. Implement `crates/x2l-rdna2-spirv` front-end: console shader container unpacker + RDNA2 (GFX10.3) instruction decoder (`x2linux disasm <shader>`).
2. Wire first direct `rspirv` binary SPIR-V 1.6 compute shader lowering path with `spirv-val` validation tests (`cargo test -p x2l-rdna2-spirv --features spirv-val`) and `x2l-pipeline` XDG SPIR-V cache integration.
3. Expand `x2l-gpu-vk` from Null M1 COM device toward direct Vulkan 1.3/1.4 (`ash`) device/queue initialization and compute dispatch verification on Mesa RADV/ANV/NVK.
