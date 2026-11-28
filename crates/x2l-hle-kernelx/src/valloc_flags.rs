// Console `VirtualAlloc` / `VirtualAllocEx` / `VirtualAllocFromApp` flag definitions & sanitizer
// informed by xcl-dev/xcl (commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102,
// crates/xcl/src/hle/kernelx.rs), Marvisak/VIctory (commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-hle/src/modules/xmem.rs), and
// public Microsoft Win32 `winnt.h` headers.
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: dependency-free console flag sanitizer shared between
// `x2l-hle-kernelx` and `x2l-pe-shims/shim-common`.

/// Console graphics-visible virtual memory flag (`MEM_GRAPHICS`).
pub const MEM_GRAPHICS: u32 = 0x1000_0000;
/// Console title-partition virtual memory flag (`MEM_TITLE`).
pub const MEM_TITLE: u32 = 0x4000_0000;
/// Console 4 MiB page selector (`MEM_4MB_PAGES`).
pub const MEM_4MB_PAGES: u32 = 0x8000_0000;
/// Win32 / console large-page flag (`MEM_LARGE_PAGES`).
pub const MEM_LARGE_PAGES: u32 = 0x2000_0000;
/// Win32 physical memory reserve flag (`MEM_PHYSICAL`).
pub const MEM_PHYSICAL: u32 = 0x0040_0000;
/// Composite 64 KiB page selector (`MEM_LARGE_PAGES | MEM_PHYSICAL`).
pub const MEM_64K_PAGES: u32 = MEM_LARGE_PAGES | MEM_PHYSICAL;

/// Public `PAGE_GRAPHICS_*` protection bits (`0x0007_F800`).
pub const PAGE_GRAPHICS_NOACCESS: u32 = 0x0000_0800;
pub const PAGE_GRAPHICS_READONLY: u32 = 0x0000_1000;
pub const PAGE_GRAPHICS_READWRITE: u32 = 0x0000_2000;
pub const PAGE_GRAPHICS_EXECUTE: u32 = 0x0000_4000;
pub const PAGE_GRAPHICS_EXECUTE_READ: u32 = 0x0000_8000;
pub const PAGE_GRAPHICS_EXECUTE_READWRITE: u32 = 0x0001_0000;
pub const PAGE_GRAPHICS_COHERENT: u32 = 0x0002_0000;
pub const PAGE_GRAPHICS_NOCACHE: u32 = 0x0004_0000;

/// Full GPU protection modifier bitmask stripped before forwarding to Wine.
pub const PAGE_GRAPHICS_MASK: u32 = PAGE_GRAPHICS_NOACCESS
    | PAGE_GRAPHICS_READONLY
    | PAGE_GRAPHICS_READWRITE
    | PAGE_GRAPHICS_EXECUTE
    | PAGE_GRAPHICS_EXECUTE_READ
    | PAGE_GRAPHICS_EXECUTE_READWRITE
    | PAGE_GRAPHICS_COHERENT
    | PAGE_GRAPHICS_NOCACHE;

/// Console-only allocation marker bits that are never valid Win32 `VirtualAlloc` flags.
pub const CONSOLE_ONLY_ALLOC_MARKERS: u32 = MEM_GRAPHICS | MEM_TITLE | MEM_4MB_PAGES;

pub const MEM_RESERVE_PLACEHOLDER: u32 = 0x0004_0000;
pub const MEM_REPLACE_PLACEHOLDER: u32 = 0x0000_4000;
pub const MEM_COALESCE_PLACEHOLDERS: u32 = 0x0000_0001;
pub const MEM_PRESERVE_PLACEHOLDER: u32 = 0x0000_0002;

pub const PLACEHOLDER_ALLOC_FLAGS: u32 = MEM_RESERVE_PLACEHOLDER | MEM_REPLACE_PLACEHOLDER;
/// Bit in the arena allocation bridge's options word, above the extended-parameter count.
pub const WIN32_VALLOC_FROM_APP: u64 = 1 << 32;

/// Win32 one-shot page access trap modifier.
pub const PAGE_GUARD: u32 = 0x100;

/// Returns `true` when the CPU base protection byte requests execute permission
/// (`PAGE_EXECUTE*` = `0x10`, `0x20`, `0x40`, `0x80`).
pub const fn is_executable_cpu_protect(protect: u32) -> bool {
    (protect & 0x0000_00F0) != 0
}

/// Returns whether the current Win32 protection permits ordinary CPU reads.
pub const fn is_cpu_readable_protect(protect: u32) -> bool {
    protect & PAGE_GUARD == 0 && matches!(protect & 0xff, 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80)
}

/// Sanitize console-only `VirtualAlloc*` allocation and protection flags for Wine's NT virtual
/// memory manager while preserving valid Win32 flags and CPU protection semantics.
///
/// - Always strips verified console-only markers (`MEM_GRAPHICS`, `MEM_TITLE`, `MEM_4MB_PAGES`)
///   and the full `PAGE_GRAPHICS_*` mask (`0x0007_F800`).
/// - In explicit console context (`console_context == true`, e.g., `kernelx.dll` exports) or when
///   any console marker is present, strips `MEM_64K_PAGES` (`0x2040_0000`) as a unit and
///   `MEM_LARGE_PAGES` (`0x2000_0000`).
/// - For unmarked system calls (`console_context == false`), preserves standalone `MEM_PHYSICAL`
///   and 2 MiB-aligned `MEM_64K_PAGES`, while stripping sub-2 MiB `MEM_64K_PAGES` that Wine
///   rejects with `STATUS_INVALID_PARAMETER`.
/// - Preserves CPU protection bits verbatim (including `0`, never silently turning `0` into
///   `PAGE_READWRITE`).
pub const fn sanitize_virtual_alloc_flags(
    alloc_type: u32,
    protect: u32,
    size: usize,
    console_context: bool,
) -> (u32, u32) {
    let has_console_marker =
        (alloc_type & CONSOLE_ONLY_ALLOC_MARKERS) != 0 || (protect & PAGE_GRAPHICS_MASK) != 0;

    let mut clean_alloc = alloc_type & !CONSOLE_ONLY_ALLOC_MARKERS;

    if console_context || has_console_marker {
        if (clean_alloc & MEM_64K_PAGES) == MEM_64K_PAGES {
            clean_alloc &= !MEM_64K_PAGES;
        }
        if (clean_alloc & MEM_LARGE_PAGES) != 0 {
            clean_alloc &= !MEM_LARGE_PAGES;
        }
    } else if (clean_alloc & MEM_64K_PAGES) == MEM_64K_PAGES && size < 2 * 1024 * 1024 {
        clean_alloc &= !MEM_64K_PAGES;
    }

    let clean_protect = protect & !PAGE_GRAPHICS_MASK;
    (clean_alloc, clean_protect)
}
