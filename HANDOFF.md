# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M2.4 Complete — Direct `D3D12.X` → Vulkan 1.3/1.4 Backend (`crates/x2l-gpu-vk`), Linux 3-Tier `userfaultfd` + `VK_KHR_buffer_device_address` Unified Memory Manager, and End-to-End Rendered Frame from `x2l-testtitle.exe` via `d3d12_xs.dll`**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5`, Windows D3D12 host backends, `victory-game`, `victory-loader`, and `xcl`'s HLSL/DXIL emitters remain strictly excluded).
- Verified M2.4 Deliverables:
  - **`crates/x2l-gpu-vk/src/memory.rs`:** Linux 3-Tier Unified Memory Manager:
    - **Tier 1 (`HostExternal`):** Zero-copy `VK_EXT_external_memory_host` (`HOST_ALLOCATION_EXT`) + `VK_KHR_buffer_device_address` aligned to `lcm(4096, minImportedHostPointerAlignment)`.
    - **Tier 2 (`UserfaultfdWp` / `ExplicitStaging`):** Unprivileged `userfaultfd(UFFD_USER_MODE_ONLY | O_CLOEXEC | O_NONBLOCK)` negotiating `wp_async_pagemap_scan` (`UFFD_FEATURE_WP_ASYNC` + `/proc/self/pagemap` `PAGEMAP_SCAN`), `wp_sync_fault_thread`, or 4 KiB page-diff `explicit_dirty_tracking` fallback; adjacent dirty 4 KiB pages coalesce into minimal `vkCmdCopyBuffer` spans with `Synchronization2` barriers and zero spurious re-uploads on unmodified pages.
    - **Tier 3 (`GpuVaRegistry`):** O(log N) `BTreeMap` interval registry resolving guest GPU virtual addresses to `(vk::Buffer, offset, vk::DeviceAddress, AllocationId)`.
  - **`crates/x2l-gpu-vk/src/vulkan.rs` & `src/com.rs`:**
    - Direct Vulkan 1.3/1.4 (`ash`) backend using `VK_KHR_dynamic_rendering`, `VK_KHR_synchronization2`, `VK_KHR_timeline_semaphore`, `VK_KHR_buffer_device_address`, and optional `VK_EXT_external_memory_host` + `VK_EXT_descriptor_buffer`.
    - `CreateGraphicsPipelineState` compiles RDNA2 VS+PS directly to binary SPIR-V 1.6 via `x2l-rdna2-spirv` and caches modules under `$XDG_CACHE_HOME/x2linux/<title>/spirv/` via `x2l-pipeline`.
    - Authentic Win64 `extern "win64"` COM vtables for `ID3D12DeviceX`, `ID3D12CommandQueueX` (legacy console `Signal@14`, `PresentX@30`), `ID3D12CommandAllocator`, `ID3D12GraphicsCommandListX` (`OMSetRenderTargets@50`, `ClearRenderTargetView@52`), `ID3D12Fence` (1:1 `VK_KHR_timeline_semaphore`), `ID3D12Resource`, `ID3D12DescriptorHeap` (MSVC struct-return ABI), and `ID3D12PipelineState`, preserving W^X spare-slot probe thunks (including `ID3D12DeviceX[45]`) and keeping `gpu.backend = "null"` 100% intact.
    - Dedicated SUD-free native Linux GPU worker thread (`start_native_gpu_worker` / `exec_native`) spawned in `libx2l_core.so`'s `.init_array` before Wine enables `PR_SET_SYSCALL_USER_DISPATCH`, isolating `userfaultfd`, `PAGEMAP_SCAN`, and Vulkan driver ioctls from Wine SUD with a zero-allocation atomic mailbox.
    - Headless offscreen `R8G8B8A8_UNORM` rendering + `PresentX` readback + optional P6 PPM output via `X2L_FRAME_DUMP`.
  - **`crates/x2l-common`, `crates/x2l-cli`, `crates/x2l-core`, `crates/x2l-testtitle`, & `.github/workflows/ci.yml`:**
    - Added `GpuTelemetry` to `report.json` (`backend`, `device_name`, `driver_info`, `external_memory_host`, `descriptor_buffer`, `userfaultfd_mode`, `shaders_compiled`, `frames_presented`).
    - Wired `x2linux run --gpu <null|vulkan>` with XDG SPIR-V cache and `X2L_FRAME_DUMP` passthrough.
    - Extended `x2l-testtitle.exe` (`--render-vulkan` / `X2L_TEST_RENDER=1` / `X2L_GPU_BACKEND=vulkan`) to draw a verified 64×64 yellow triangle on a dark blue background (`corner(4,4)=(0,0,64)`, `center(32,32)=(255,255,0)`) via real `d3d12_xs.dll` COM vtables while keeping `verify_readonly.sh` PASS and logging `kernelx.dll#999` + `ID3D12DeviceX[45]`.
    - Verified with `VK_LAYER_KHRONOS_validation` (**0 validation errors**) on both:
      - **AMD Radeon RX 9060 XT (RADV GFX1200, Mesa 26.2.7, Vulkan 1.4.354):** `external_memory_host=true`, `descriptor_buffer=true`, `userfaultfd_mode=wp_async_pagemap_scan`, `shaders_compiled=2`, `frames_presented=1`, pixel-exact PPM verification + `verify_readonly.sh` PASS.
      - **llvmpipe / Lavapipe (LLVM 22.1.8, Mesa 26.2.7, Vulkan 1.4.354):** identical pixel-exact pass and CI coverage for both `--gpu null` and `--gpu vulkan`.

## Next Engineering Tasks (M2.5 / M3)
1. **M2.5 — `VK_EXT_mesh_shader` (`MS`/`AS`) & `VK_EXT_graphics_pipeline_library`:** Add RDNA2 NGG/Mesh/Amplification shader lowering in `x2l-rdna2-spirv` and pipeline-library disk caching + descriptor-buffer root-signature binding in `x2l-gpu-vk`.
2. **M2.6 — Real-Title Read-Only `inspect` / `shaders` Telemetry Loop:** Expand `D3D12.X` COM surface coverage (compute pipelines, root signatures, CBV/SRV/UAV descriptor buffers, depth/stencil targets, upload/copy queues) driven by sanitized `report.json` telemetry.
3. **M3 — Native PipeWire Audio & SDL3 Input/WSI:** Wire `crates/x2l-audio` (PipeWire + SDL3 fallback + XMA2/Opus decode) and `crates/x2l-input` (SDL3/evdev gamepad + optional windowed Vulkan swapchain present).
