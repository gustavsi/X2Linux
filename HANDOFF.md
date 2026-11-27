# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M2.2 Complete — Vulkan 1.3 Compute GPU Instruction Test Runner (`--features gpu-tests`), RDNA2 Test Assembler, and Verified Bit-Exact Execution of EXEC Divergence, VOP3 Modifiers, Cross-Lane Ops, and Buffer Memory Loads/Stores**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` and `xcl`'s `gpu/desc.rs` / HLSL/DXIL emitters remain strictly excluded).
- Verified M2.2 Deliverables:
  - **`crates/x2l-rdna2-spirv/src/gpu_runner.rs` (`--features gpu-tests`):** Self-contained Vulkan 1.3 (`ash`) compute test harness (`GpuRunner` + 66,072-byte `WaveState` ABI + 3-binding `state_in`/`state_out`/`memory_buf` storage-buffer layout). Supports hardware GPU auto-selection, `X2L_VK_DEVICE` override, `VK_EXT_subgroup_size_control` (`requiredSubgroupSize` pinning), automatic `WorkgroupShared` fallback when `maxSubgroupSize < wave_width` (e.g., Lavapipe), thread-safe serialized queue/pool dispatch, and `spirv-val --target-env vulkan1.3` pre-validation.
  - **`crates/x2l-rdna2-spirv/src/encode.rs`:** Clean-room RDNA2 (GFX10.3) + console `gfx10.1` instruction encoder & `Asm` label builder covering SOPP/SOP1/SOP2/SOPK/SOPC, VOP1/VOP2/VOPC/VOP3, cross-lane ops, DS swizzle, DPP16/DPP8, SMEM, MUBUF, GLOBAL, and clean-room AMD RDNA2 128-bit V# buffer descriptor builder (`make_vsharp`).
  - **`crates/x2l-rdna2-spirv/src/decode.rs` & `src/spirv.rs`:** Expanded decoding & direct `rspirv` binary SPIR-V 1.6 lowering for:
    - 32-bit & 64-bit EXEC/VCC mask manipulation (`s_and/or/xor/andn2_saveexec_b32/b64`, `s_and/or/xor/andn2_b64`, `s_not_b64`, `s_mov_b64`, `s_cselect_b64`, `vccz`/`execz`, `s_cbranch_execz/nz`, `s_cbranch_vccz/nz`).
    - VOP3 `abs`/`neg`/`omod`/DX10 `clamp`, bit-exact `v_min_f32`/`v_max_f32` (NaN & signed zero), safe saturating `v_cvt_i32_f32`/`v_cvt_u32_f32`, `v_rcp_f32`/`v_rsq_f32`/`v_sqrt_f32`.
    - Cross-lane ops: `v_readfirstlane_b32`, `v_readlane_b32`, `v_writelane_b32`, `ds_swizzle_b32` (bitmode/quad/rotate/FFT), `v_permlane16_b32`/`v_permlanex16_b32`, DPP16 & DPP8 — verified bit-identical across both `CrossLaneMode::Subgroup` and `CrossLaneMode::WorkgroupShared`.
    - Memory ops: `s_load_dword*`, `s_buffer_load_dword*`, `buffer_load_dword*`, `buffer_store_dword*`, `global_load_dword*`, `global_store_dword*` with clean-room AMD RDNA2 V# 128-bit descriptor decoding, 64-bit carry-safe address math (zero `Capability::Int64`), and `OpArrayLength` + AMD Table 35 out-of-bounds clamping/drop guards.
  - **`crates/x2l-rdna2-spirv/tests/gpu_instruction_tests.rs` & CI:**
    - Host hardware verified: **AMD Radeon RX 9060 XT (RADV GFX1200)** (`Vulkan 1.4.354`, `Mesa 26.2.7`, `subgroup=64 [32..64]`, `denorm_f32=true`) — 5/5 GPU test suites passed (Wave32 & Wave64 + native `Subgroup == WorkgroupShared` parity) in ~0.11s.
    - Software Vulkan verified: **llvmpipe / Lavapipe** (`LLVM 22.1.8`, `subgroup=8 [8..8]`, `denorm_f32=false`) — 5/5 GPU test suites passed (Wave32 & Wave64 via `WorkgroupShared`) in ~1.0s.
    - Wired `mesa-vulkan-drivers` and `cargo test -p x2l-rdna2-spirv --features spirv-val,gpu-tests` in `.github/workflows/ci.yml`.

## Next Engineering Tasks (M2.3)
1. Adapt `Marvisak/VIctory`'s clean GPL-3.0 `victory-shader` SSA IR (`ir.rs`) and lane-mask control-flow structurizer (`structure.rs`, `io.rs`, `resource.rs`) onto X2Linux's direct `rspirv` binary SPIR-V 1.6 backend (replacing any DXIL/HLSL writer completely).
2. Extend direct SPIR-V 1.6 emission from compute-only verification into graphics shader stages (`VS`, `PS`, `CS`, `MS`/`VK_EXT_mesh_shader`) with `VK_EXT_descriptor_buffer` / `VK_KHR_buffer_device_address` bindings, `IMAGE` (`T#`/`S#`) sampling, and `EXP` / `v_interp_*` lowering.
3. Wire `x2l-gpu-vk` Vulkan 1.3/1.4 device, command queue, timeline semaphore, and XDG pipeline cache initialization behind `x2linux run --gpu vulkan`.
