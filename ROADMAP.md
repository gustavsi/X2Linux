# X2Linux Roadmap (GPL-3.0 Fast-Track)

## M0 — Repo Bootstrap, GPL-3.0 Guardrails & OverlayFS Sandbox (Week 1)
- GPL-3.0-or-later Cargo workspace + `deny.toml` (`GPL-2.0-only` ban) + `xtask` asset-policy pre-commit hooks
- `x2l-common` (XDG paths, `x2linux.toml`, `report.json` schema) and `x2l-cli` (`x2linux`)
- `x2l-vfs-overlay` (unprivileged userns Kernel OverlayFS + `fuse-overlayfs` fallback + SHA-256 read-only integrity tests)
- `x2l-resolver` (`victory-pe` + `xcl-dev/xcl` resolver/`inspect`), `x2l-formats` (`victory-formats`), `x2l-pipeline` (`victory-pipeline`)
- `x2l-testtitle` synthetic console executable

## M1 — Wine/`ntsync` Bridge, Auto-Stubs & Offline HLE (Weeks 2–3)
- Wine/`ntsync` launcher + Unix-call bridge (`x2l-pe-shims` ↔ `libx2l_core.so`)
- Dynamic x86-64 auto-stub thunks + `data/ordinals/*.toml` + `report.json`
- `kernelx` + offline `XGameRuntime` (adapted from `xcl-dev/xcl`, `victory-hle`, and `WinDurango` LGPL-3.0 logic)
- Null `d3d12_xs` with spare-slot vtable probes & raw shader capture

## M2 — Direct SPIR-V Backend & Direct Vulkan 1.3/1.4 (Weeks 4–12)
- Attach direct `rspirv` binary SPIR-V 1.6 backend to SSA IR + lane-mask structurizer (`x2l-rdna2-spirv`)
- Validate via `spirv-val` and Vulkan compute GPU instruction test runner on Mesa RADV/ANV/NVK
- `x2l-gpu-vk` (D3D12.X front-end → Vulkan 1.3/1.4: `VK_KHR_buffer_device_address`, `VK_EXT_descriptor_buffer`, `VK_EXT_mesh_shader`, `VK_EXT_graphics_pipeline_library`, `VK_EXT_external_memory_host`, `VK_KHR_timeline_semaphore`, `userfaultfd` WP unified memory, SDL3 Wayland/X11 swapchain)

## M3 — Native Linux Audio (Weeks 13–14)
- `XAudio2` → PipeWire (SDL3 fallback) fed by `x2l-formats` Opus/XMA2 PCM + FFmpeg fallback

## M4 — Input, `io_uring` DirectStorage & Steam Compatibility Tool (Week 15)
- `GameInput` / `XInput` → SDL3 Gamepad / `evdev`
- `DirectStorage` → Linux `io_uring` + `x2l-formats` (zlib/BCPack)
- Steam compatibility tool packaging (adapted from `pablogonz12/Scarlight` without DXC/vkd3d-proton bundles)
