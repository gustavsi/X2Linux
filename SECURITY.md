# Security Policy

**Project:** X2Linux (Xbox Series X|S → Linux Compatibility Layer)  
**License:** GPL-3.0-or-later

## 1. Threat Model & Security Boundaries

1. **Strict Read-Only Immutability of the User's Software Directory:** No bug, path traversal, symlink escape, or crafted filename may modify, truncate, or delete any file inside the mounted `lowerdir`.
2. **Host Process Memory Safety (`libx2l_core.so`):** Rust crates parsing external binaries (`x2l-resolver`, `x2l-formats`, `x2l-rdna2-spirv`) treat all offsets, sizes, zlib/BCPack lengths, and vtable indices as untrusted.
3. **Privilege & Namespace Hygiene:** `x2linux` must never require `root` privileges or setuid binaries. Unprivileged user/mount namespace setup (`x2l-vfs-overlay`) must never leak writable mounts outside the target prefix.
4. **Sanitized Telemetry (`report.json`):** Diagnostic reports must not leak host environment secrets or proprietary binary payloads.

## 2. Supported Versions

| Version / Branch | Supported |
|---|---|
| `main` (active development) | Yes |
| Older commits / unmerged forks | No |

## 3. Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Use **GitHub Private Vulnerability Reporting** (`Security` tab → `Report a vulnerability`) as the sole official disclosure channel for this repository.

Include:
- Affected crate, Linux distro, kernel version (`uname -r`), Vulkan/Mesa version, and commit SHA.
- A minimal **synthetic** proof-of-concept (crafted byte slice, unit test, or malformed synthetic container).

**Strict Attachment Prohibition:** NEVER attach commercial executables, console archive files, decrypted assets, keys, firmware dumps, or NDA SDK files. Such reports will be deleted immediately.

## 4. Out of Scope

- DRM / XVC / MSIXVC decryption, anti-tamper bypasses, or `XStore` license spoofing.
- Standard guest crashes caused by unimplemented console APIs (submit a regular issue with sanitized `report.json`).
- Upstream Wine, Proton, Mesa, FFmpeg, or Linux kernel bugs (report directly upstream).

## 5. Developer Hygiene Controls

- Install local pre-commit asset guard: `cargo xtask install-hooks`.
- Automated CI checks: `cargo deny check licenses` and `cargo deny check advisories`.
- Every `unsafe` block must include a `// SAFETY:` comment documenting all upheld invariants.
