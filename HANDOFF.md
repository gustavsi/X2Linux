# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M3.1 Complete — Complete Generic GDK PE Shim Surface (`d3d12_xs` Xbox Entrypoints, `xmem.dll` Physical/Scatter Memory, `xg_xs.dll` Texture Computer, `XCRDAPI.dll` DirectStorage, `GameInput.dll` Ordinal `#1`, Audio Co-Processor Shims `AcpHal`/`xapu`/`chad`, PSM AppNotify, and Expanded API-Set Aliases)**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5`, Windows D3D12 host backends, `victory-game`, `victory-loader`, and `xcl`'s HLSL/DXIL emitters remain strictly excluded).
- Verified M3.1 Deliverables:
  - **Expanded Resolver Policy & Ordinal Databases (`data/aliases.toml`, `data/ordinals/{gameinput,xaudio2_9}.toml`, `crates/x2l-resolver`):**
    - Added all standard Windows Core API-set prefixes (`api-ms-win-core-{com,datetime,debug,errorhandling,handle,heap,interlocked,libraryloader,localization,namedpipe,processenvironment,processthreads,processtopology,profile,rtlsupport,string,sysinfo,timezone,windowserrorreporting}-*`, `ext-ms-win-rtcore-ntuser-*`), Wine system fallbacks (`xaudio2_9.dll`, `winhttp.dll`, `mfplat.dll`, `mfreadwrite.dll`), and full `[implemented]` tables for all console HLE DLLs.
    - Verified host Wine `xaudio2_9.dll` (`/usr/lib64/wine-wow64/wine/x86_64-windows/xaudio2_9.dll`) already exports `XAudio2Create` at ordinal `#1` (ordinal base `1`, index `0`, RVA `0x1F990`), so `xaudio2_9.dll` safely resolves via `system:xaudio2_9` (`SystemFallback`) with `data/ordinals/xaudio2_9.toml`.
    - `data/ordinals/gameinput.toml` maps `gameinput.dll!#1 -> GameInputCreate` (`Implemented`).
  - **`xmem.dll` Linux Physical & Scatter Memory Arena (`crates/x2l-hle-kernelx/src/xmem.rs`, `crates/x2l-pe-shims/xmem`):**
    - 13 exports (`XMemAllocDefault`, `XMemFreeDefault`, `XMemVirtualAlloc`, `XMemVirtualFree`, `XMemVirtualQuery`, `XMemAllocatePhysicalPages`, `XMemFreePhysicalPages`, `XMemMapPhysicalPages`, `XMemMapPhysicalPagesScatter`, `XMemUnmapPhysicalPages`, `XMemGetWorkingSetStatistics`, `XMemAllocateTransferMemory`, `XMemFreeTransferMemory`).
    - Backed by SUD-free native worker `memfd_create` pools + `mmap(MAP_SHARED | MAP_FIXED)` aliasing so contiguous and scatter mappings share physical backing pages across multiple virtual addresses, with 64-byte `XMemBasicInformation` (48-byte Win64 prefix + flags/tag) and 16-byte `XMemGetWorkingSetStatistics` prefix.
  - **`d3d12_xs.dll` Xbox Entrypoints (`crates/x2l-gpu-vk/src/d3d12x_ext.rs`, `crates/x2l-pe-shims/d3d12_xs`):**
    - Added 4-arg `D3D12XboxCreateDevice(adapter, pParameters, riid, ppvDevice)`, `D3D12ConfigureVirtualMemoryX`, and `D3D12SerializeVersionedRootSignature` returning a ref-counted Win64 COM `ID3DBlob` wrapping a valid 68-byte DXBC `RTS0` container.
    - Kept `gpu.backend = "null"` and the direct Vulkan 1.3/1.4 rendering pipeline 100% intact.
  - **`xg_xs.dll`, `XCRDAPI.dll`, `GameInput.dll`, `AcpHal.dll`, `xapu.dll`, `chad.dll`, & PSM AppNotify (`crates/x2l-core/src/gdk_com.rs`, `crates/x2l-audio/src/coproc.rs`, `crates/x2l-input/src/gameinput.rs`, `crates/x2l-pe-shims/*`):**
    - `xg_xs.dll`: `XGSuggestSwizzleMode` + `XGCreateTextureComputer` (`IXGTextureAddressComputer` with non-QI slot 0=`AddRef`, 1=`Release`, 2=`GetResourceLayout` (`0x1B40` bytes), backed by `x2l-formats::texture`).
    - `XCRDAPI.dll`: `DStorageGetFactory` → `IDStorageFactoryX`, `IDStorageQueueX` (`0x48`-byte `DSTORAGE_REQUEST`, raw/zlib/BCPack decompression), `IDStorageFileX` (G:/S:/Z: console path translation), and `IDStorageStatusArrayX`.
    - `GameInput.dll`: `.def` ordinal `#1` (`GameInputCreate @1`) → `IGameInput` v0 offline lifecycle (`GAMEINPUT_E_READING_NOT_FOUND`, monotonic timestamps, callback tokens).
    - `AcpHal.dll` (9), `xapu.dll` (5), `chad.dll` (5): APU heap/VA↔APU address mapping, `IAcpHal`, XAPU command ring, and DSP stream activation.
    - `api-ms-win-core-psm-appnotify-l1-1-0/1.dll` (4): registration cookie lifecycle; also staged as `twinapi.appcore.dll` in `target/pe-shims/` and symlinked into the isolated Wine prefix's `drive_c/windows/system32/` alongside `gameinput.dll` so Wine's `apisetschema` and builtin DLL resolution route cleanly through our native shims.
  - **Verification:**
    - All 13 staged PE shim DLLs cross-compile cleanly via `./scripts/build_pe_shims.sh`.
    - `x2l-testtitle.exe` passes 100% of checks across `--gpu null`, **Host RADV (AMD Radeon RX 9060 XT, Mesa 26.2.7, Vulkan 1.4.354)**, and **Lavapipe (`X2L_VK_DEVICE=llvmpipe`)**, with pixel-exact 64×64 triangle verification (`corner(4,4)=(0,0,64)`, `center(32,32)=(255,255,0)`), `verify_readonly.sh` PASS, 0 Vulkan validation errors, and `report.json` recording `kernelx.dll#999` + `ID3D12DeviceX[45]`.

## Next Engineering Tasks (M3.2+)
1. **M3.2 — Interactive Presentation & Native Linux Audio/Input:** Wire SDL3 Wayland/X11 Vulkan swapchain presentation (`PresentX` → live window), PipeWire audio playback (SDL3 fallback) + XMA2/Opus streaming, and live SDL3/evdev gamepad input into `IGameInput`.
2. **M3.3 — `VK_EXT_mesh_shader` (`MS`/`AS`), `VK_EXT_graphics_pipeline_library`, & Expanded D3D12.X COM Surface:** Add RDNA2 NGG/Mesh/Amplification shader lowering in `x2l-rdna2-spirv`, compute pipelines, root-signature descriptor-buffer bindings, depth/stencil targets, and upload/copy queues driven by sanitized `report.json` telemetry.
