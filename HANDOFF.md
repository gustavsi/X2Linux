# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M2.1 Complete — Console Shader Container Unpacker, RDNA2 Disassembler (`x2linux disasm` / `x2linux shaders`), and Phase-1 Direct Binary SPIR-V 1.6 Emitter (`rspirv` + `spirv-val`)**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` remains strictly excluded).
- Verified M2.1 Deliverables:
  - **`crates/x2l-rdna2-spirv/src/container.rs`:** Checked, zero-panic console shader container classifier and unpacker (`DXBC -> DXIL(NOBC) -> XDAT -> XBCH v3 (zlib via x2l-formats) -> FSSM v0x101`, standalone `XBCH`/`FSSM`, `DXIL`, `ELF`, raw RDNA2), with bounded decompression caps, stage-table extraction, `wave32` flag detection, embedded constant-pool segmentation, and SHA-256 content hashing.
  - **`crates/x2l-rdna2-spirv/src/decode.rs`:** Checked RDNA2 (GFX10.3) + console `gfx10.1` (`v_mac_f32`, `v_mad_f32`, `v_madmk_f32`, `v_madak_f32`) instruction decoder & disassembler across SOPP, SOPC, SOP1, SOPK, SOP2, VOP1, VOPC, VOP2, and VOP3, enforcing hardware `SRC == 255` literal rules (including unused VOP3 source slots), mandatory-literal VOP2 opcodes, SDWA/DPP16/DPP8 extension words, and reachable CFG disassembly across multi-`s_endpgm` programs.
  - **`crates/x2l-rdna2-spirv/src/spirv.rs`:** Clean-room direct binary SPIR-V 1.6 compute emitter via `rspirv` (zero HLSL/DXIL/DXC/`vkd3d-proton`). Supports Wave32 and Wave64, Vulkan 1.3 `StorageBuffer` wave-state ABI (`WAVE_STATE_STRIDE = 66_072`), structured `OpLoopMerge` + `OpSwitch` basic-block dispatch, EXEC-masked VGPR writes, exact `v_mul_legacy_f32` (`0 * NaN/Inf = +0.0`), `NoContraction` non-fused `gfx10.1` MAD/MAC, `GLSL.std.450` FMA, subgroup ballot `v_cmp_*`, `v_cndmask_b32`, `Rdna2SpirvConverter` (`x2l-pipeline` XDG SPIR-V cache integration), and `spirv-val --target-env vulkan1.3` validation (`--features spirv-val`).
  - **`crates/x2l-cli`:** Wired `x2linux disasm <file> [--spirv-out <path>]` and `x2linux shaders <dir> [--recomp] [--validate-spirv] [--out <dir>]`.
  - **CI & Provenance:** Added mandatory `cargo test -p x2l-rdna2-spirv --features spirv-val` step to `.github/workflows/ci.yml` and recorded upstream references in `NOTICE` and `THIRD_PARTY.md`.

## Next Engineering Tasks (M2.2)
1. Build a native Linux Vulkan 1.3/1.4 (`ash`) GPU Compute Instruction Test Runner in `x2l-gpu-vk` that executes `x2l-rdna2-spirv` compute shaders on real Mesa RADV/ANV/NVK (or Lavapipe in CI) against the `WAVE_STATE_STRIDE` (`66_072`-byte) storage-buffer ABI, verifying exact bit-for-bit SGPR, VGPR, EXEC, VCC, and SCC results across Wave32/Wave64, inactive lanes, `v_mul_legacy_f32` NaN/Inf edge cases, and branches.
2. Introduce a structured SSA IR + control-flow structurizer layer in `x2l-rdna2-spirv` (extending into SMEM/VMEM/DS/EXP lowering, descriptor-buffer resource bindings, and graphics stages VS/PS/MS/AS).
3. Wire `x2l-gpu-vk` Vulkan 1.3/1.4 device, command queue, timeline semaphore, and pipeline cache initialization behind `x2linux run --gpu vulkan`.
