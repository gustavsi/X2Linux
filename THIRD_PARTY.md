# Third-Party Components and References

X2Linux is licensed under **GPL-3.0-or-later** (`LICENSE`). Every file adapted from an upstream project must preserve original copyright headers, include a dated modification notice (GPL-3.0 Section 5), and record its commit SHA in `NOTICE`.

| Component | Source | License | Usage & Provenance |
|---|---|---|---|
| **xcl-dev/xcl** *(generic modules only; Windows PE loader, VFS hooks, HLSL/DXIL emitters, & single-archive probes excluded)* | https://github.com/xcl-dev/xcl | GPL-3.0-or-later *(relicensed at `06f90c1986f43698d0405eff3c2ab0af230e1ed9`; imported `6089122a1c96dff16e93b6fdf04c8ed20cfb6102`)* | GPL-3.0 Rust Series X|S compatibility layer: `crates/x2l-resolver/src/pe.rs` adapted from `crates/xcl/src/pe.rs` (2026-11-27); `crates/x2l-rdna2-spirv/src/{container,decode}.rs` informed by `crates/xcl/src/gpu/{xbox,rdna2_xbox,rdna2}.rs` (2026-11-27); reference for resolver, PE inspection, `report.json`, RDNA2 ISA tables, HLE, and synthetic test title. |
| **Marvisak/VIctory** *(generic crates only; `victory-game` & `victory-loader` excluded)* | https://github.com/Marvisak/VIctory | GPL-3.0-or-later *(imported `63bcca43c6a00a758510398fa43096ce7e4f7df0`)* | GPL-3.0 Rust Series X|S compatibility layer: `crates/x2l-formats` (`bcpack.rs`, `bcpack_tables.rs`, `texture.rs`, `xma.rs`, `xma/tables.rs`) and `crates/x2l-pipeline` (`lib.rs`, `cache.rs`) adapted on 2026-11-27; `crates/x2l-rdna2-spirv/src/{container,decode}.rs` informed by `crates/victory-shader/src/{dxbc,native,gfx10}`; `crates/x2l-gpu-vk/src/com.rs` console vtable slot mappings & `PresentX` plane layout informed by `crates/victory-gpu/src/{device,queue,list,pipeline}.rs`. |
| **pablogonz12/Scarlight** *(excluding `victory-game`, `victory-loader`, & DXC/vkd3d-proton bundling)* | https://github.com/pablogonz12/Scarlight | GPL-3.0-or-later *(audited `HEAD` `7e556ea08ee076ca3b4c46f65a1a6984a955d15a`)* | GPL-3.0 Linux-focused Series X|S compatibility layer: Steam compatibility tool packaging patterns, wave64-on-wave32 analyses, ISA progress tooling. |
| **WinDurango** | https://github.com/WinDurango/WinDurango | LGPL-3.0 | Compatible for porting `kernelx` and XMA logic into Rust with preserved notices. |
| **AMD Machine-Readable ISA & LLVM AMDGPU** | AMD GPUOpen / LLVM | MIT / Apache-2.0 WITH LLVM-exception | RDNA2 instruction decoder tables, GFX10 format reference, and ISA test vectors. |
| **Mesa ACO / RADV / addrlib** | https://mesa3d.org | MIT | RDNA2 semantics, GFX10 swizzle tables, and Linux Vulkan driver reference. |
| **libHttpClient** | https://github.com/microsoft/libHttpClient | MIT | Public reference for `XTaskQueue` / `XAsync`. |
| **DirectX-Headers** | https://github.com/microsoft/DirectX-Headers | MIT | Public base D3D12 COM layouts & enums. |
| **SPIRV-Tools (`spirv-val`)** | https://github.com/KhronosGroup/SPIRV-Tools | Apache-2.0 | Binary SPIR-V 1.6 validation in CI and debug builds. |
| **`ash` / `rspirv`** | crates.io | MIT OR Apache-2.0 | Low-level Vulkan 1.3/1.4 bindings & binary SPIR-V assembly. |
| **FFmpeg (`libavcodec`)** | https://ffmpeg.org | LGPL-2.1+ | Dynamic linking / safe Rust reference for Opus & XMA2 decoding. |
| **PipeWire / SDL3** | pipewire.org / libsdl.org | MIT / Zlib | Native Linux audio output, WSI, and gamepad input. |
| **boykopovar/AnyPS5** | https://github.com/boykopovar/AnyPS5 | **GPL-2.0-only** | **STRICTLY EXCLUDED — ZERO CODE COPIED (legally incompatible with GPL-3.0).** |
