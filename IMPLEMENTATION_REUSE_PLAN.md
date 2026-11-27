# X2Linux — GPL-3.0 Code-Reuse & Fast-Track Implementation Plan

Date: 2026-11-27 · License: **GPL-3.0-or-later**

All three upstream compatibility-layer repositories audited below are licensed under **GPL-3.0-or-later** (verified via root `Cargo.toml`, member manifests, and verbatim GNU GPL v3.0 `LICENSE` files). Accordingly, X2Linux uses **`GPL-3.0-or-later`** across the workspace.

## 1. Audited Upstream Reference Snapshot

| Upstream Repository | Audited `HEAD` Commit SHA | Verified License | Technical Role |
|---|---|---|---|
| `xcl-dev/xcl` | `6089122a1c96dff16e93b6fdf04c8ed20cfb6102` *(post-GPL-3.0 relicense `06f90c1986f43698d0405eff3c2ab0af230e1ed9`)* | `GPL-3.0-or-later` | GPL-3.0 Rust Series X|S compatibility layer (resolver, diagnostics, ISA tables, console runtime HLE, synthetic test title) |
| `Marvisak/VIctory` | `63bcca43c6a00a758510398fa43096ce7e4f7df0` | `GPL-3.0-or-later` | GPL-3.0 Rust Series X|S compatibility layer (generic shader front-end, formats, pipeline cache, HLE, PE parser) |
| `pablogonz12/Scarlight` | `7e556ea08ee076ca3b4c46f65a1a6984a955d15a` | `GPL-3.0-or-later` | GPL-3.0 Linux-focused fork of `VIctory` (Steam compatibility tool packaging, wave64-on-wave32 analyses, ISA progress scripts) |
| `WinDurango/WinDurango` | *(tracked on first import)* | `LGPL-3.0` | Compatible reference for `kernelx` and XMA behavior |

## 2. Verified Reusable Paths by Upstream Repository

### A. `xcl-dev/xcl` (`6089122a1c96dff16e93b6fdf04c8ed20cfb6102`, `GPL-3.0-or-later`)
- **Import resolution, inspection, & telemetry (`x2l-resolver`, `x2l-common`, `x2l-cli`):**
  - `crates/xcl/src/resolver.rs`, `crates/xcl/src/inspect.rs`, `crates/xcl/src/report.rs`, `crates/xcl/src/stubs.rs`, `crates/xcl/src/manifest.rs`, `crates/xcl/data/stubdb.toml`
- **RDNA2 ISA & console shader/tiling foundations (`x2l-rdna2-spirv`, `x2l-formats`):**
  - `crates/xcl/src/gpu/xbox.rs`, `crates/xcl/src/gpu/rdna2.rs`, `crates/xcl/src/gpu/rdna2_spec.rs`, `crates/xcl/src/gpu/rdna2_xbox.rs`
  - `crates/xcl/src/gpu/tiling.rs`, `crates/xcl/src/gpu/tiling_tables.rs`
  - `crates/xcl/src/gpu/recomp/cfg.rs`, `crates/xcl/src/gpu/recomp/ssa.rs`, `crates/xcl/src/gpu/recomp/upper_lanes.rs`, `crates/xcl/src/gpu/recomp/x64_masks.rs`
  - Codegen scripts: `scripts/gen_rdna2.py`, `scripts/gen_tiling.py`, `scripts/gen_d3d12_layout.py`, `scripts/gen_gdk_table.py`, `scripts/gen_bcpack_victory_tables.py`
- **Console HLE & formats (`x2l-hle-kernelx`, `x2l-hle-xgr`, `x2l-formats`, `x2l-gpu-vk`):**
  - `crates/xcl/src/hle/kernelx/`, `crates/xcl/src/hle/xgameruntime/`, `crates/xcl/src/hle/d3d12x/`, `crates/xcl/src/hle/bcpack/`, `crates/xcl/src/hle/xma2/` *(LGPL-2.1+ FFmpeg port)*, `crates/xcl/src/hle/dstorage/`
- **Synthetic test fixtures (`crates/x2l-testtitle`):**
  - `tests/testtitle/`, `tests/testgamedll/`, `tests/data/llvm-amdgpu/` *(Apache-2.0 WITH LLVM-exception)*

### B. `Marvisak/VIctory` (`63bcca43c6a00a758510398fa43096ce7e4f7df0`, `GPL-3.0-or-later`)
- `crates/victory-shader` (`src/gfx10/`, `src/ir.rs`, `src/structure.rs`, `src/io.rs`, `src/range.rs`, `src/resource.rs`, `src/native.rs`, `src/dxbc.rs`) → `crates/x2l-rdna2-spirv` *(console shader container parser, RDNA2 decoder, SSA IR, lane-mask structurizer; attach X2Linux's new direct `rspirv` binary SPIR-V 1.6 emitter)*
- `crates/victory-formats` (`src/bcpack.rs`, `src/bcpack-tables.bin`, `src/texture.rs`, `src/texture/`, `src/xma.rs`, `src/xma/tables.rs`) → `crates/x2l-formats`
- `crates/victory-pipeline` (`src/cache.rs`) → `crates/x2l-pipeline` (XDG content-addressed cache)
- `crates/victory-pe` (`src/image.rs`, `src/implib.rs`, `src/layout.rs`) → `crates/x2l-resolver`
- `crates/victory-core`, `crates/victory-hle` (`src/registry.rs`, `src/runtime.rs`, `src/modules/`), `crates/victory-gpu` *(console D3D12.X front-end / descriptor / fence / arena models only; emit direct Vulkan 1.3/1.4 via `ash`)*, `apps/victory` *(CLI patterns)*, `xtask` *(asset-policy hook patterns)*

### C. `pablogonz12/Scarlight` (`7e556ea08ee076ca3b4c46f65a1a6984a955d15a`, `GPL-3.0-or-later`)
- **Steam & Linux packaging:** `tools/steam-compat/` (`install.sh`, `package.sh`, `BASE`, compatibility tool launch script patterns) and `tools/scarlight-launcher/` (`examples/profile.env`, desktop entry template) — omitting any DXC or `vkd3d-proton` runtime bundling.
- **Wave64/shader analysis & ISA tooling:** `crates/victory-shader/src/emit/` analysis passes (`wave.rs`, `own_bits.rs`, `group_wide.rs`, `valu_clamp.rs`, `clamp.rs`) and `tools/progress/` (`gen_isa_forms.py`, `rdna2_isa_forms.txt`, `shader_scan.py`). *(Note: `Scarlight` does not contain a native Rust `rspirv` SPIR-V backend; its shader emitter targets DXIL bitcode for external translation, which X2Linux intentionally replaces with direct `rspirv` binary SPIR-V 1.6.)*

## 3. Audit Results & Strict DO NOT PORT List

### Legal Audit of `boykopovar/AnyPS5` (`GPL-2.0-only`) — STRICTLY FORBIDDEN
- `boykopovar/AnyPS5` is `GPL-2.0-only` and **incompatible with GPL-3.0**. NEVER copy or derive code from `AnyPS5`.
- Audit of `xcl-dev/xcl` (`6089122a1c96dff16e93b6fdf04c8ed20cfb6102`):
  - `xcl`'s `THIRD_PARTY.md` records `AnyPS5` as *"reference only; no code taken"*.
  - Source audit confirmed **zero `GPL-2.0-only` code-origin headers** in `crates/xcl/src/`. However, four files contain reference comments citing `AnyPS5` hardware behavior notes:
    - `crates/xcl/src/gpu/desc.rs` (line 5)
    - `crates/xcl/src/gpu/dxil/lower_image.rs` (line 1381)
    - `crates/xcl/src/gpu/recomp/emit.rs` (lines 2480, 2890, 3491)
    - `crates/xcl/src/gpu/recomp/tests.rs` (lines 9563, 9599)
  - **Policy:** **DO NOT PORT** `crates/xcl/src/gpu/recomp/emit.rs`, `crates/xcl/src/gpu/recomp/dxc.rs`, or `crates/xcl/src/gpu/dxil/*` (these are HLSL/DXIL emitters incompatible with X2Linux's direct SPIR-V architecture anyway). For descriptor layouts (`desc.rs`) and tests, derive independently from public AMD GFX10/RDNA2 ISA documentation and Mesa MIT headers.

### Architecture & Neutrality Exclusions — DO NOT PORT
- **Single-title / proprietary-archive code:** `VIctory`/`Scarlight` `crates/victory-game/` and single-archive probe files in `xcl`.
- **Windows manual PE/TLS loaders:** `VIctory`/`Scarlight` `crates/victory-loader/` and `xcl`'s `crates/xcl/src/loader.rs`, `tls.rs`, `eh.rs`. Wine/Proton + kernel `ntsync` owns the NT floor.
- **User-space VFS file-API detours:** `xcl`'s `crates/xcl/src/vfs_hooks.rs` and `hooks.rs`. Linux Kernel OverlayFS (`x2l-vfs-overlay`) enforces read-only isolation at the kernel level.
- **HLSL / DXIL / Windows DXC / `vkd3d-proton` paths:** `xcl`'s `gpu/recomp/emit.rs`, `gpu/recomp/dxc.rs`, `gpu/dxil/`, `VIctory`/`Scarlight`'s `victory-shader/src/emit/bitcode*` DXIL writer, and `Scarlight`'s `vkd3d-proton`/DXC bundling inside `tools/steam-compat/`.

## 4. Fast-Track Phases
1. **Phase 1 (Week 1):** GPL-3.0-or-later workspace, `deny.toml` (`GPL-2.0-only` ban), `xtask` asset-policy hooks, `x2l-common` (XDG + `x2linux.toml` + `report.json`), `x2l-cli`, `x2l-vfs-overlay` (kernel OverlayFS + SHA-256 read-only test), `x2l-formats`, `x2l-pipeline`, `x2l-resolver`.
2. **Phase 2 (Weeks 2–3):** Thin `x86_64-pc-windows-gnu` PE shims + dynamic stub thunks ↔ `libx2l_core.so` via `__wine_unix_call` + `ntsync`; `x2l-testtitle` M1 boot verification.
3. **Phase 3 (Weeks 4–7):** Attach direct `rspirv` binary SPIR-V 1.6 emitter onto SSA IR + lane-mask structurizer (`x2l-rdna2-spirv`); validate with `spirv-val` and Vulkan compute GPU tests on Mesa RADV/ANV/NVK (VS/PS/CS/`VK_EXT_mesh_shader`).
4. **Phase 4 (Weeks 8–12):** Direct Vulkan 1.3/1.4 device (`ash`, `x2l-gpu-vk`) + `VK_KHR_buffer_device_address` + `VK_EXT_descriptor_buffer` + `VK_EXT_external_memory_host` + `VK_EXT_graphics_pipeline_library` + `VK_KHR_timeline_semaphore` + Linux `userfaultfd` WP dirty-page sync.
5. **Phase 5 (Weeks 13–15):** Native PipeWire (SDL3 fallback) audio, SDL3 Gamepad / `evdev` input, Linux `io_uring` DirectStorage, Steam compatibility tool packaging.
