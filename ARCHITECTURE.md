# X2Linux — Architecture Deep Dive

## 1. Dual-Side In-Process Linux Bridge (`x2l-pe-shims` ↔ `libx2l_core.so`)
- **`crates/x2l-pe-shims` (`x86_64-pc-windows-gnu`):** Ultra-thin `extern "win64"` DLLs (`kernelx.dll`, `xgameruntime.dll`, `d3d12_xs.dll`, expanding in M3/M4 to `xaudio2_9.dll`, `gameinput.dll`, `dstorage.dll`) + dynamic W^X x86-64 auto-stub thunk generator (`x2l-shim-common`).
- **`crates/x2l-core` (`x86_64-unknown-linux-gnu` → `libx2l_core.so`):** Loaded in-process into Wine via `LD_PRELOAD`.
  - **Version-Independent ABI Contract:** On ELF initialization (`.init_array`), when `X2L_BRIDGE_ENABLED=1`, `libx2l_core.so` publishes the process-local ASLR address of its single `extern "win64"` dispatch entrypoint (`x2l_bridge_dispatch(opcode, a0..a5) -> u64`) via `setenv("X2L_BRIDGE_FN", "0x...", 1)`.
  - **Handshake & Segment Isolation:** On first call, PE shims read `X2L_BRIDGE_FN` via `GetEnvironmentVariableA`, invoke `OP_HANDSHAKE` and verify `X2L_BRIDGE_MAGIC` (`0x5832_4C5F_0001_0001`). Because Linux x86-64 uses `%fs` for native pthread TLS while Wine uses `%gs` for the Win64 TEB, direct `extern "win64"` calls run natively inside the shared 64-bit address space without Wine-internal header lock-in—enabling zero-copy M2 Vulkan (`VK_EXT_external_memory_host`) and `userfaultfd` integration.
  - **Clean Telemetry Flush:** PE shims invoke `OP_FLUSH_REPORT` on `DLL_PROCESS_DETACH` and `libx2l_core.so` flushes again in `.fini_array` only when guest activity occurred (preventing helper Wine processes such as `wineserver` from clobbering `report.json`).

## 2. Kernel OverlayFS Mount Sequence (`x2l-vfs-overlay`)
1. Pre-flight checks (`Threads == 1`, `max_user_namespaces != 0`, `apparmor_restrict_unprivileged_userns != 1`) ensure `Auto` mode never traps the process in a restricted user namespace before trying `fuse-overlayfs`.
2. `unshare(CLONE_NEWUSER | CLONE_NEWNS)`
3. Configure `/proc/self/setgroups` (`deny`), `uid_map`, `gid_map`, and `MS_REC | MS_PRIVATE` root propagation
4. `mount("overlay", merged, "overlay", MS_NODEV | MS_NOSUID, "lowerdir=<title>,upperdir=<data>/overlay,workdir=<data>/work")` (fallback: `fuse-overlayfs`)
5. Map isolated Wine prefix drives (`G:` → `merged`, `S:` → `saves`, `P:` → `pls`, `T:` → `temp`)

## 3. Three-Tier Unified Memory Manager (`x2l-gpu-vk`)
- **M1 Null Device:** `D3D12CreateDevice` returns a ref-counted Win64 COM `NullD3D12DeviceX` whose slots `3..128` are W^X-protected 21-byte x86-64 tail-call thunks capturing `[rsp]` caller RIP and recording `(ID3D12DeviceX, slot_index, caller_rip)` into `report.json`.
- **Tier 1 (Zero-Copy / ReBAR):** `VK_EXT_external_memory_host` + `VK_KHR_buffer_device_address` on large-BAR GPUs.
- **Tier 2 (`userfaultfd` WP Sync):** Write-protect CPU shadow pages (`UFFDIO_REGISTER_MODE_WP`), collect dirty 4 KB ranges, DMA-sync before dependent GPU submits.
- **Tier 3 (GPU VA Lookup Table):** Compact device-address range table for shaders dereferencing raw console GPU pointers.

## 4. Direct RDNA2 → SPIR-V Pipeline (`x2l-rdna2-spirv`)
1. Front-end (adapted from GPL-3.0 `victory-shader` + `xcl-dev/xcl` ISA tables/CFG/SSA analyses): console shader container unpacker, RDNA2 decoder, SSA IR lifter, lane-mask structurizer.
2. Direct binary SPIR-V 1.6 backend (`rspirv`): lowers SSA IR + structured CFG (`OpSelectionMerge`/`OpLoopMerge`), VS/PS/CS/MS (`VK_EXT_mesh_shader`), and `VK_EXT_subgroup_size_control` directly to binary SPIR-V validated by `spirv-val`.
3. Content-addressed caching (`x2l-pipeline`) under `~/.cache/x2linux/<title>/spirv/` (respecting `$XDG_CACHE_HOME`).
