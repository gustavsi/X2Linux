# X2Linux — Architecture Deep Dive

## 1. Dual-Side Wine Unix-Call Split
- **`crates/x2l-pe-shims` (`x86_64-pc-windows-gnu`):** Minimal `extern "win64"` DLLs (`kernelx.dll`, `xgameruntime.dll`, `d3d12_xs.dll`, `xaudio2_9.dll`, `gameinput.dll`, `dstorage.dll`) + dynamic x86-64 stub thunk generator invoking `libx2l_core.so` via `__wine_unix_call`.
- **`crates/x2l-core` (`x86_64-unknown-linux-gnu` → `libx2l_core.so`):** All heavy runtime systems running natively on Linux (`ash` Vulkan 1.3/1.4, `rspirv`, `userfaultfd`, `io_uring`, PipeWire, FFmpeg, SDL3).

## 2. Kernel OverlayFS Mount Sequence (`x2l-vfs-overlay`)
1. `unshare(CLONE_NEWUSER | CLONE_NEWNS)`
2. Configure `uid_map` / `gid_map`
3. `mount("overlay", merged, "overlay", 0, "lowerdir=<title>,upperdir=<data>/overlay,workdir=<data>/work")` (fallback: `fuse-overlayfs`)
4. Map Wine prefix drives (`G:` → `merged`, `S:` → `saves`, `P:` → `pls`, `T:` → `temp`)

## 3. Three-Tier Unified Memory Manager (`x2l-gpu-vk`)
- **Tier 1 (Zero-Copy / ReBAR):** `VK_EXT_external_memory_host` + `VK_KHR_buffer_device_address` on large-BAR GPUs.
- **Tier 2 (`userfaultfd` WP Sync):** Write-protect CPU shadow pages (`UFFDIO_REGISTER_MODE_WP`), collect dirty 4 KB ranges, DMA-sync before dependent GPU submits.
- **Tier 3 (GPU VA Lookup Table):** Compact device-address range table for shaders dereferencing raw console GPU pointers.

## 4. Direct RDNA2 → SPIR-V Pipeline (`x2l-rdna2-spirv`)
1. Front-end (adapted from GPL-3.0 `victory-shader` + `xcl-dev/xcl` ISA tables/CFG/SSA analyses): console shader container unpacker, RDNA2 decoder, SSA IR lifter, lane-mask structurizer.
2. Direct binary SPIR-V 1.6 backend (`rspirv`): lowers SSA IR + structured CFG (`OpSelectionMerge`/`OpLoopMerge`), VS/PS/CS/MS (`VK_EXT_mesh_shader`), and `VK_EXT_subgroup_size_control` directly to binary SPIR-V validated by `spirv-val`.
3. Content-addressed caching (`x2l-pipeline`) under `~/.cache/x2linux/<title>/spirv/` (respecting `$XDG_CACHE_HOME`).
