# AGENTS.md — X2Linux Engineering & Clean-Room Rules

X2Linux is a native Linux x86-64 compatibility layer for unpacked Xbox Series X|S (GDK / Scarlett) software, licensed under **GPL-3.0-or-later**.

## Build, Lint & Test Commands

Run these verbatim from the repository root before claiming any task is complete:

```bash
# 1. Format & lint native Linux Rust workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# 2. Run workspace tests (includes OverlayFS read-only & XDG/report schema tests)
cargo test --workspace

# 3. Verify CLI help output
cargo run -p x2l-cli -- --help

# 4. Verify strict GPL-3.0-or-later license compliance (blocks GPL-2.0-only)
cargo deny check licenses

# 5. Cross-compile thin PE-side shim DLLs (required if touching x2l-pe-shims)
./scripts/build_pe_shims.sh

# 6. Validate generated SPIR-V modules (required if touching x2l-rdna2-spirv)
cargo test -p x2l-rdna2-spirv --features spirv-val
```

## Non-Negotiable Legal, Neutrality & Clean-Room Boundaries

1. **Absolute Neutrality Rule (Zero Commercial Title, Studio, or Engine Mentions):**
   - Never mention **any** commercial game title, franchise, abbreviation, character/location name, publisher, studio, proprietary game engine, or single-title archive/shader fingerprint anywhere in this repository (code, comments, docs, tests, fixtures, CI logs, or commit messages).
   - Use strictly generic terminology: *"unpacked GDK/Scarlett software"*, *"console titles"*, *"archive files"*, *"compressed audio containers"*, and *"console shader containers"*.
   - Describe upstream projects strictly by technical role; never quote commercial title names from upstream documentation.
2. **License is `GPL-3.0-or-later` (`GPL-2.0-only` is forbidden):**
   - **Allowed upstream code sources:** `xcl-dev/xcl` (`GPL-3.0-or-later`, post-relicense), `Marvisak/VIctory` (`GPL-3.0-or-later`), `pablogonz12/Scarlight` (`GPL-3.0-or-later`), `WinDurango` (`LGPL-3.0`), and MIT/BSD/Zlib/ISC/LGPL-2.1+/Apache-2.0/GPL-2.0-or-later crates.
   - **Forbidden (`GPL-2.0-only`):** NEVER copy, port, or derive code from `boykopovar/AnyPS5` (`GPL-2.0-only`) or any other `GPL-2.0-only` repository — combining `GPL-2.0-only` code with GPL-3.0 is legally incompatible.
   - **Mandatory GPL-3.0 Section 5 provenance header:** Any file adapted from `xcl-dev/xcl`, `Marvisak/VIctory`, `pablogonz12/Scarlight`, or `WinDurango` MUST preserve original copyright notices, include a dated modification notice, and be recorded in `THIRD_PARTY.md` and `NOTICE`.
3. **Never import `victory-game` or single-title hacks:**
   - X2Linux is strictly general-purpose across unpacked GDK titles. Reject title-specific archive IDs, single-archive probes, hardcoded offsets, or title-specific workarounds.
4. **Zero leaked SDKs, commercial assets, keys, or DRM code:**
   - Never copy headers, structs, or constants from NDA/leaked Microsoft GDK/XDK SDKs. Use only public `docs.xbox.com`, `learn.microsoft.com`, MIT `DirectX-Headers`/`libHttpClient`, public AMD ISA specs, and sanitized `report.json` telemetry.
   - Never commit commercial binaries, shader dumps, encryption keys, or XVC/MSIXVC decryption / `XStore` license spoofing. Install and keep `cargo xtask install-hooks` active.

## Linux-First Architecture Invariants

- **Target:** Linux x86-64 only (kernel 6.14+ with `/dev/ntsync`, unprivileged userns, `userfaultfd`, `io_uring`, Vulkan 1.3/1.4).
- **Never modify the user's software folder:** Copy-on-write isolation is enforced at the Linux kernel level by `x2l-vfs-overlay` (unprivileged `CLONE_NEWUSER | CLONE_NEWNS` + Kernel OverlayFS, `fuse-overlayfs` fallback). Do NOT add Windows-style user-space file API detour hooks (`retour`/Detours or `xcl`'s VFS hooks).
- **Direct Vulkan + Direct SPIR-V only:** Never introduce an HLSL text emitter, DXIL emitter, Windows `dxcompiler.dll`/`dxil.dll`, or a runtime `vkd3d-proton` dependency. Emit binary SPIR-V 1.6 via `rspirv` straight from SSA IR (`x2l-rdna2-spirv`) and target Vulkan 1.3/1.4 (`ash`) in `x2l-gpu-vk`.
- **Wine/Proton (`ntsync`) owns the NT floor:** Do NOT port Windows manual PE/TLS loaders (`xcl`'s PE loader or `victory-loader`). Keep `crates/x2l-pe-shims` (`x86_64-pc-windows-gnu`) ultra-thin (`extern "win64"` + dynamic x86-64 auto-stub thunks calling `libx2l_core.so` via `__wine_unix_call`); put all heavy HLE, memory (`userfaultfd`), Vulkan, PipeWire, and SDL3 logic in `libx2l_core.so`.
- **No silent stubs:** Every unimplemented console export, unknown ordinal, unmapped path, or unmapped `D3D12.X` vtable slot must log caller RVA and increment counters in `report.json`.

## Workspace Map (Where Details Live)

- `docs/specs/2026-11-27-x2linux-design.md` — Master Linux architecture specification.
- `IMPLEMENTATION_REUSE_PLAN.md` — Pinned 3-repo GPL-3.0 reuse map & DO NOT PORT audit (`xcl-dev/xcl`, `Marvisak/VIctory`, `pablogonz12/Scarlight`, `WinDurango`).
- `xtask/` — Repo automation including `cargo xtask install-hooks` asset-policy pre-commit guard.
- `crates/x2l-common` — XDG paths (`~/.config/x2linux`, `~/.local/share/x2linux`, `~/.cache/x2linux`), `x2linux.toml` config, and `report.json` schema.
- `crates/x2l-cli` — `x2linux` CLI (`inspect`, `run`, `disasm`, `shaders`).
- `crates/x2l-vfs-overlay` — Unprivileged userns Kernel OverlayFS + `fuse-overlayfs` fallback.
- `crates/x2l-resolver` — PE32+ import inspection, API-set aliases, ordinal DB, dynamic stub dispatch.
- `crates/x2l-formats` — Texture detiling, parallel zlib/BCPack decompression, Opus/XMA2 decoding.
- `crates/x2l-pipeline` — Content-addressed XDG shader & pipeline cache.
- `crates/x2l-hle-kernelx` / `crates/x2l-hle-xgr` — Console `kernelx` and offline `XGameRuntime` HLE.
- `crates/x2l-rdna2-spirv` — Console shader container parser, RDNA2 decoder, SSA IR, structurizer, and direct `rspirv` binary SPIR-V 1.6 backend.
- `crates/x2l-gpu-vk` — Direct `D3D12.X` → Vulkan 1.3/1.4 (`VK_KHR_buffer_device_address`, `VK_EXT_descriptor_buffer`, `VK_EXT_mesh_shader`, `VK_EXT_graphics_pipeline_library`, `VK_EXT_external_memory_host`, `VK_KHR_timeline_semaphore`, `userfaultfd` WP dirty-page tracking).
- `crates/x2l-audio` / `crates/x2l-input` — Native PipeWire (SDL3 fallback) audio & SDL3/evdev gamepad input.
- `crates/x2l-core` — Native Linux shared library (`libx2l_core.so`) entry point for `__wine_unix_call`.
- `crates/x2l-pe-shims` / `crates/x2l-testtitle` — Excluded PE-side crates cross-compiled with `--target x86_64-pc-windows-gnu`.

## Scope & Approval Boundaries

- **Autonomous:** Editing code/tests, running `cargo fmt`, `cargo clippy`, `cargo test`, `cargo deny check licenses`, `./scripts/build_pe_shims.sh`.
- **Ask human first:** Adding new Cargo dependencies, changing license/security policies, modifying `deny.toml`, running commands requiring `sudo` or system package installs, or pushing to remote branches.
