# X2Linux — Xbox Series X|S → Linux Native Compatibility Layer (Design Specification)

Date: 2026-11-27 · Status: M0 Bootstrap · License: **GPL-3.0-or-later**

## 1. Goal and Premise
Run unpacked Xbox Series X|S (GDK / Scarlett) software natively on Linux x86-64 without CPU emulation. Guest x86-64 instructions execute directly on the host CPU while X2Linux supplies clean-room console libraries (`kernelx`, `XGameRuntime`, `d3d12_xs`, `XAudio2`, `GameInput`, `DirectStorage`) backed directly by Linux kernel primitives (`ntsync`, OverlayFS, `userfaultfd`, `io_uring`), Vulkan 1.3/1.4, direct binary SPIR-V compilation, PipeWire, and SDL3.

## 2. Goals, Non-Goals, Anti-Goals
- **Goals:** General-purpose Series X|S compatibility; data-driven import resolver & automatic stubbing with `report.json` telemetry; kernel-enforced read-only title directory via OverlayFS; direct `D3D12.X → Vulkan 1.3/1.4` and `RDNA2 ISA → SSA IR → Binary SPIR-V 1.6` translation; Steam/Proton compatibility tool packaging.
- **Non-Goals:** XVC/MSIXVC package decryption or DRM bypass; Xbox Live online services (XSAPI/multiplayer); Xbox One (D3D11.X/ERA); Windows/macOS hosts; CPU instruction emulation.
- **Anti-Goals:** Any modification or deletion in the original title folder; silent stubs; NDA or leaked SDK material; faking `XStore` licenses; title-specific hacks or proprietary archive fingerprints.

## 3. Target Host Requirements
- Linux x86-64, kernel 6.14+ (`ntsync` + unprivileged userns + `userfaultfd` + `io_uring`)
- CPU: x86-64-v3 (`AVX2`, `FMA`, `F16C`, `BMI1/2`, `LZCNT`, `MOVBE`)
- GPU: Vulkan 1.3/1.4 with `bufferDeviceAddress`, `descriptorBuffer`, `meshShader`, `graphicsPipelineLibrary`, `externalMemoryHost`, `timelineSemaphore` (Mesa RADV / ANV / NVK or NVIDIA 565+)
- RAM/VRAM: 16 GB RAM + 8 GB VRAM min; 32 GB RAM + 16 GB VRAM recommended

## 4. Architecture Selection
**Wine/Proton (`ntsync`) + Thin PE Shims (`x2l-pe-shims`) ↔ Native Linux Unix-call `.so` Core (`libx2l_core.so`) + Kernel OverlayFS (`x2l-vfs-overlay`) + Direct Vulkan 1.3/1.4 (`x2l-gpu-vk`) & Direct Binary SPIR-V 1.6 (`x2l-rdna2-spirv`).**

## 5. Subsystems
- **5.1 PE Bootstrap & Resolver (`x2l-resolver`):** `x2linux inspect`; runtime x86-64 stub thunks (`mov r10, info; jmp dispatch`); data-driven `data/aliases.toml` and `data/ordinals/*.toml`.
- **5.2 Kernel OverlayFS VFS (`x2l-vfs-overlay`):** Unprivileged `CLONE_NEWUSER | CLONE_NEWNS` mounting `lowerdir=<title>` (RO) + `upperdir=~/.local/share/x2linux/<title>/overlay` (RW), with `fuse-overlayfs` fallback.
- **5.3 `kernelx` & Offline `XGameRuntime` (`x2l-hle-kernelx`, `x2l-hle-xgr`):** Flag-stripped `VirtualAlloc*`, `XMemAlloc*`, `TitleMemoryStatus`, thread naming & affinity; offline `XTaskQueue`/`XAsync`, single local `XUser`, `XPackage`, `XGameSave` (`saves/`), `XPersistentLocalStorage` (`pls/`), offline `XStore`.
- **5.4 Null `d3d12_xs` (M1):** Dummy COM objects + spare-slot vtable probes + raw shader capture into `~/.cache/x2linux/<title>/shaders_raw/`.
- **5.5 Direct `D3D12.X` → Vulkan 1.3/1.4 (`x2l-gpu-vk`, M2):** `VK_KHR_buffer_device_address` + `VK_EXT_external_memory_host` + `userfaultfd` WP dirty-page sync; `VK_EXT_descriptor_buffer`; `VK_EXT_mesh_shader`; `VK_EXT_graphics_pipeline_library`; `VK_KHR_timeline_semaphore`.
- **5.6 Direct RDNA2 → SPIR-V Compiler (`x2l-rdna2-spirv`, M2):** Combines GPL-3.0-or-later console shader container parsing, RDNA2 decoding, SSA IR, and lane-mask structurization, lowering directly to binary SPIR-V 1.6 via `rspirv` + `spirv-val` and persistent XDG disk caching (`x2l-pipeline`) across large shader corpora.
- **5.7 Formats, Audio (M3) & Input/DirectStorage (M4):** `x2l-formats` (parallel zlib, BCPack decompression, texture detiling, compressed audio containers: Opus/XMA2); native PipeWire (SDL3 fallback) audio (`x2l-audio`); SDL3 Gamepad/evdev (`x2l-input`); Linux `io_uring` DirectStorage.

## 6. Licensing & Governance
**GPL-3.0-or-later.** Compatible with `xcl-dev/xcl` (`GPL-3.0-or-later`, post-relicense), `Marvisak/VIctory` (`GPL-3.0-or-later`, excluding `victory-game`/`victory-loader`), `pablogonz12/Scarlight` (`GPL-3.0-or-later`), `WinDurango` (`LGPL-3.0`), `Apache-2.0` (`SPIRV-Tools`), and MIT/BSD/Zlib/LGPL-2.1+. Code copying from `GPL-2.0-only` projects (`boykopovar/AnyPS5`) is strictly forbidden.
