# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M2.3 Complete — Graphics-Stage Direct Binary SPIR-V 1.6 Lowering (`VS`, `PS`, `CS` LDS, `image_*` Sampling, and Live-Register Pruning) Verified by Headless Offscreen Vulkan 1.3 Dynamic Rendering on RADV + Lavapipe**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5` and `xcl`'s `gpu/desc.rs` / HLSL/DXIL emitters remain strictly excluded).
- Verified M2.3 Deliverables:
  - **`crates/x2l-rdna2-spirv/src/container.rs`:** Bounds-checked `ISG1`/`OSG1` (32-byte) and `ISGN`/`OSGN` (24-byte) DXBC signature parser (`SignatureElement`, `StageSignature`, `parse_dxbc_signature`) populated on `ParsedContainer`.
  - **`crates/x2l-rdna2-spirv/src/decode.rs` & `src/encode.rs`:** `llvm-mc -mcpu=gfx1030`-verified GFX10.3 encodings for `EXP` (`pos0..4`, `param0..31`, `mrt0..7`, `mrtz`, `null`, `COMPR`/`DONE`/`VM`), 1-DWORD `VINTRP` (`v_interp_p1_f32`/`p2`/`mov`), `MIMG` (`image_load`, `image_store`, `image_sample`, `image_sample_l`, plus NSA multi-word addresses), corrected `DS` `op[25:18]` / `gds[17]` (`ds_write_b32`, `ds_write2_b32`, `ds_read_b32`, `ds_read2_b32`, `ds_swizzle_b32`), and `s_swappc_b64`.
  - **`crates/x2l-rdna2-spirv/src/graphics.rs` & `src/spirv.rs`:**
    - Direct binary SPIR-V 1.6 lowering for `ShaderStage::Vertex` (`BuiltIn Position`, `ISG1` `Location` inputs, `s_swappc_b64` fetch bypass, `exp paramN`, optional Y-flip) and `ShaderStage::Pixel` (`OriginUpperLeft`, `VINTRP`, `MIMG` separated `OpTypeImage` + `OpTypeSampler` + `OpSampledImage`, `COMPR=1` `UnpackHalf2x16` `exp mrtN`, `DepthReplacing` `exp mrtz`, and SPIR-V 1.6 `OpDemoteToHelperInvocation` pixel discard).
    - Lightweight live-register / dead-register pruning pass (`analyze_live_registers`) allocating `Function` variables only for live SGPRs/VGPRs (no dead 256-VGPR/128-SGPR banks in graphics stages).
    - Compute LDS lowering (`Workgroup u32[4096]` = 16 KiB) with bounds/alignment guards for `ds_read_b32`/`ds_read2_b32`/`ds_write_b32`/`ds_write2_b32` + `s_barrier`.
  - **`crates/x2l-rdna2-spirv/src/gpu_runner.rs` & `tests/gpu_graphics_tests.rs`:**
    - Headless Vulkan 1.3 Dynamic Rendering (`VK_KHR_dynamic_rendering` + `Synchronization2`) offscreen renderer supporting `R8G8B8A8_UNORM` color, `D32_SFLOAT` depth, sampled 2D textures, and multi-draw passes.
    - Verified with `VK_LAYER_KHRONOS_validation` (**0 validation errors**) across **14 unit + 4 offscreen graphics/LDS + 5 compute ISA tests (23/23)** on both:
      - **AMD Radeon RX 9060 XT (RADV GFX1200, Mesa 26.2.7, Vulkan 1.4.354):** colored triangle + `COMPR=1` f16x2 magenta triangle, 4-quadrant `image_sample`/`image_sample_l` checkerboard quad, half-quad `OpDemoteToHelperInvocation` discard + `exp mrtz` custom depth occlusion, and Wave32/Wave64 compute LDS reversed-lane exchange.
      - **llvmpipe / Lavapipe (LLVM 22.1.8, Mesa 26.2.7):** identical 23/23 pixel-exact & bit-exact pass.
  - **`crates/x2l-cli`:** Enhanced `x2linux shaders` with stage breakdown (`VS`/`PS`/`CS`/etc.) and graphics-stage SPIR-V 1.6 recompilation + `spirv-val` validation.

## Next Engineering Tasks (M2.4)
1. Implement real Vulkan 1.3/1.4 (`ash`) device, graphics/compute queues, `VK_KHR_timeline_semaphore`, `VK_EXT_descriptor_buffer` / `VK_KHR_buffer_device_address`, and Linux `userfaultfd` write-protect unified memory synchronization in `crates/x2l-gpu-vk`.
2. Wire `x2l-gpu-vk` behind `x2l-core`'s `D3D12.X` COM vtable dispatch (`ID3D12DeviceX`, `ID3D12CommandQueueX`, `ID3D12GraphicsCommandListX`, `ID3D12PipelineState`, swapchain/present) so `x2l-testtitle.exe` creates a real `D3D12.X` device and renders a verified frame on Linux Vulkan (`--gpu vulkan`).
3. Add `VK_EXT_mesh_shader` (`MS`/`AS`) lowering and `VK_EXT_graphics_pipeline_library` XDG disk pipeline caching in `x2l-pipeline` + `x2l-gpu-vk`.
