# X2Linux — Engineering Handoff & Verification Guide

## Current Status (2026-11-27)
- Stage: **M3.2 Complete — Native Linux Vulkan WSI Swapchain Window (Wayland/X11), PipeWire Audio Output (`x2l-audio`), Linux `evdev` Gamepad Input (`x2l-input` → `IGameInput`), and Safe `VirtualAlloc` Console Flag-Stripping**
- License: **GPL-3.0-or-later** (aligned with `xcl-dev/xcl`, `Marvisak/VIctory`, and `pablogonz12/Scarlight`; `boykopovar/AnyPS5`, Windows D3D12 host backends, `victory-game`, `victory-loader`, and `xcl`'s HLSL/DXIL emitters remain strictly excluded). Zero new Cargo dependencies or C bindgen build requirements were added in M3.2.
- Verified M3.2 Deliverables:
  - **Safe `VirtualAlloc` Console Flag-Stripping (`crates/x2l-hle-kernelx/src/valloc_flags.rs`, `crates/x2l-pe-shims/{shim-common,kernelx,xmem}`, `data/aliases.toml`):**
    - Verified empirically on Wine 11.21 that `apisetschema` resolves `api-ms-win-core-memory-l1-1-*` directly to `kernelbase.dll` before hitting disk (bypassing `system32` overrides), and overriding `kernelbase.dll` prevents startup.
    - Implemented single-source-of-truth console bitmask sanitization (`MEM_GRAPHICS` `0x10000000`, `MEM_TITLE` `0x40000000`, `MEM_4MB_PAGES` `0x80000000`, `MEM_64K_PAGES` `0x20400000`, and full `0x0007F800` `PAGE_GRAPHICS_*` mask) preserving standalone Win32 `MEM_PHYSICAL`, zero protection, and `VirtualAllocFromApp` non-executable enforcement (`ERROR_INVALID_PARAMETER = 87`).
    - Added 6 named exports to `kernelx.dll` (`VirtualAlloc`, `VirtualAllocEx`, `VirtualAllocFromApp`, `VirtualFree`, `VirtualQuery`, `VirtualQueryEx`) plus a loader-lock-safe startup PE32+ IAT patcher in `shim-common` patching only `VirtualAlloc`/`VirtualAllocEx`/`VirtualAllocFromApp` IAT slots in the main executable before TLS callbacks and `main`.
  - **Native Linux Vulkan WSI Swapchain Presentation (`crates/x2l-gpu-vk/src/wsi.rs`, `vulkan.rs`, `x2l-cli`, `x2l-common`, `x2l-core`):**
    - Pure-Rust runtime `dlopen` for Wayland (`libwayland-client.so.0` + static `xdg-shell` `wl_interface`/`wl_message` tables) and X11 fallback (`libxcb.so.1`), running on X2Linux's SUD-free native GPU worker thread.
    - Full `VkSwapchainKHR` lifecycle with `Synchronization2` `vkCmdBlitImage2` presentation, per-swapchain-image `render_finished` semaphores, optional `VK_EXT_swapchain_maintenance1` present fences, resize/`SUBOPTIMAL`/`OUT_OF_DATE` recreation, and orderly teardown during `OP_FLUSH_REPORT` (`DllMain(DLL_PROCESS_DETACH)` — before Linux `exit()` `.fini_array`).
    - Added `--window`, `--headless`, and `--frames <N>` flags to `x2linux run`, `[gpu].present_mode` in `x2linux.toml`, and `present_surface` (`"wayland"`, `"x11"`, `"headless"`) in `report.json`.
  - **Real PipeWire Audio Sink (`crates/x2l-audio/src/pipewire.rs`):**
    - Wait-free SPSC `AudioRingBuffer` (`AtomicU32` bit-packed `f32` samples) accepting interleaved `f32`, `x2l_formats::opus::PcmBuffer`, and XMA2 `i16` frames with mono→stereo upmix, linear resampling, and additive stereo mixing.
    - Runtime `dlopen("libpipewire-0.3.so.0")` `pw_thread_loop` + `pw_stream` F32LE 48 kHz stereo output (initialized SUD-free via `exec_native` and verified on live host `/run/user/1000/pipewire-0`) with automatic deterministic `NullAudioSink` fallback in headless CI; wired to `xapu` audio frame submission in `x2l-core`.
  - **Real Linux `evdev` Gamepad Polling (`crates/x2l-input/src/evdev.rs` → `IGameInput`):**
    - Pure-Rust `/dev/input/event*` scanner and non-blocking poller (`EVIOCGBIT`/`EVIOCGABS`/`EVIOCGNAME` ioctls) mapping Linux gamepad buttons, D-pad/hats, triggers, and deadzone-normalized Y-inverted sticks onto public GDK `GameInputGamepadState`.
    - Wired SUD-free (`exec_native`) into `IGameInput::GetCurrentReading` returning live `ComGameInputReading` (`IGameInputReading::GetGamepadState` at slot 22) when a controller is present, or `GAMEINPUT_E_READING_NOT_FOUND` cleanly in headless CI.
  - **Verification:**
    - `x2l-testtitle.exe` passes 100% of checks across `--gpu null`, `--gpu vulkan --headless` on **Host RADV (`AMD Radeon RX 9060 XT`, Mesa 26.2.7, Vulkan 1.4.354)** and **Lavapipe (`X2L_VK_DEVICE=llvmpipe`)** (`corner(4,4)=(0,0,64)`, `center(32,32)=(255,255,0)`), and `--gpu vulkan --window --frames 5` on **Host RADV** with `VK_LAYER_KHRONOS_validation` (**0 validation errors**, `present_surface="wayland"`, `frames_presented=5`), `verify_readonly.sh` PASS, and `cargo deny check licenses advisories` PASS.

## Next Engineering Tasks (M3.3+)
1. **M3.3 — `VK_EXT_mesh_shader` (`MS`/`AS`), `VK_EXT_descriptor_buffer` Bindless/Root-Signature Binding, & Compute/Copy Queues:** Add RDNA2 NGG/Mesh/Amplification shader lowering in `x2l-rdna2-spirv`, root-signature descriptor-buffer bindings, `VK_EXT_graphics_pipeline_library` disk caching, compute pipelines, depth/stencil targets, and async copy/compute queues in `x2l-gpu-vk`.
2. **M3.4 — Real-World Telemetry-Driven HLE & Shader Corpus Expansion:** Use sanitized `report.json` outputs from `x2linux inspect` / `shaders` / `run` to prioritize additional `D3D12.X` COM slots, `kernelx`/`XGameRuntime` APIs, and RDNA2 ISA instructions.
