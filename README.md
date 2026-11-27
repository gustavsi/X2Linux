# X2Linux — Xbox Series X|S → Linux Compatibility Layer

**License:** GPL-3.0-or-later · **Status:** Pre-alpha (M0 Bootstrap) · **Target:** Linux x86-64 (Vulkan 1.3/1.4 + Kernel `ntsync`)

Run unpacked **Xbox Series X|S (GDK / Scarlett)** software natively on Linux x86-64 **without CPU emulation**.

**X2Linux** is architected from first principles for modern Linux hosts—combining Wine/Proton (`ntsync`), kernel OverlayFS, direct Vulkan 1.3/1.4, and a direct binary SPIR-V shader pipeline. Under **GPL-3.0-or-later**, X2Linux builds cleanly alongside three GPL-3.0-or-later upstream compatibility-layer projects: [xcl-dev/xcl](https://github.com/xcl-dev/xcl) (resolver, diagnostics, RDNA2 ISA tables, and console runtime analysis), [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (platform-generic Rust shader front-end, compression/texture/audio formats, and caching crates), and [pablogonz12/Scarlight](https://github.com/pablogonz12/Scarlight) (Linux Steam compatibility tool packaging and wave-lowering analyses), plus compatible `LGPL-3.0` references from [WinDurango](https://github.com/WinDurango/WinDurango).

## Why Linux-Native?

| Capability | Windows-Host Layers | **X2Linux** |
|---|---|---|
| **CPU Execution** | Native x86-64 | Native x86-64 (zero CPU emulation) |
| **OS / NT / Sync** | Custom PE loader + TEB/TLS carrier hacks | Wine/Proton + Linux kernel **`ntsync`** (6.14+) |
| **Read-Only Title Folder** | User-space file API hooking | **Kernel OverlayFS** in an unprivileged mount namespace (`x2l-vfs-overlay`) |
| **Graphics Translation** | `D3D12.X` → PC `D3D12` (→ `vkd3d-proton`) | **Direct `D3D12.X` → Vulkan 1.3/1.4** (`ash`, `x2l-gpu-vk`) |
| **Shader Pipeline** | `RDNA2` → HLSL/DXIL → Windows DXC | **Direct `RDNA2` → SSA IR → Binary SPIR-V 1.6** (`rspirv` + `spirv-val`, `x2l-rdna2-spirv`) |
| **Unified Memory** | Software staging over D3D12 heaps | `VK_KHR_buffer_device_address` + `VK_EXT_external_memory_host` + `userfaultfd` |
| **Formats & Compression** | Title-by-title or stubbed | Parallel zlib, **BCPack**, GPU texture detiling, Opus & XMA2 (`x2l-formats`) |
| **Audio & Input** | Windows XAudio2.9 / GameInput | Native **PipeWire** (SDL3 fallback) + FFmpeg/codecs + **SDL3 / evdev** |

## Quickstart

### Arch Linux / CachyOS
```bash
sudo pacman -S --needed rustup clang lld mingw-w64-gcc spirv-tools \
  vulkan-headers vulkan-icd-loader vulkan-validation-layers \
  pipewire ffmpeg sdl3 fuse-overlayfs wine
```

### Fedora
```bash
sudo dnf install rust cargo clang lld mingw64-gcc spirv-tools \
  vulkan-headers vulkan-loader-devel vulkan-validation-layers \
  pipewire-devel ffmpeg-free-devel SDL3-devel fuse-overlayfs wine
```

### Debian 14+ (Forky / Sid)
```bash
sudo apt update && sudo apt install -y rustc cargo clang lld gcc-mingw-w64-x86-64 \
  spirv-tools libvulkan-dev vulkan-validationlayers libpipewire-0.3-dev \
  libavcodec-dev libsdl3-dev fuse-overlayfs ntsync-autoload
```

### Initial Repo Setup & Build
```bash
# Install pre-commit asset-policy guard
cargo xtask install-hooks

# Install Windows GNU target strictly for thin PE-side shim DLLs & synthetic test title
rustup target add x86_64-pc-windows-gnu

# Build & verify native Linux workspace
cargo build --release --workspace
./scripts/build_pe_shims.sh --release
cargo test --workspace
cargo deny check licenses
```

### Usage
```bash
./target/release/x2linux inspect /path/to/unpacked/title/Title.exe
./target/release/x2linux run /path/to/unpacked/title/
./target/release/x2linux disasm shader.bin
./target/release/x2linux shaders /path/to/shaders/ --recomp --validate-spirv
```

Per-title configuration lives under `~/.config/x2linux/<title>/x2linux.toml` (see `x2linux.example.toml`), writable overlay data under `~/.local/share/x2linux/<title>/`, and compiled shader caches under `~/.cache/x2linux/<title>/`.

## Documentation
- Master Spec: `docs/specs/2026-11-27-x2linux-design.md`
- Architecture: `ARCHITECTURE.md`
- GPL-3.0 Code Reuse Plan: `IMPLEMENTATION_REUSE_PLAN.md`
- Roadmap & Handoff: `ROADMAP.md`, `HANDOFF.md`
- AI Agent Instructions: `AGENTS.md`, `CLAUDE.md`
- Clean-Room & Licensing: `CONTRIBUTING.md`, `THIRD_PARTY.md`, `NOTICE`
- Security Policy: `SECURITY.md`

## Legal & Clean-Room Notice
X2Linux distributes no commercial software files, keys, firmware, or console SDKs and performs no DRM/XVC decryption. Licensed under **GPL-3.0-or-later**.

Linux® is the registered trademark of Linus Torvalds in the U.S. and other countries. X2Linux is not affiliated with Microsoft, AMD, or the Linux Foundation.
