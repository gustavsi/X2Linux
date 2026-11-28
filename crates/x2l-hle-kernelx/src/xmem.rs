// Win64 ABI signatures, attribute bitfields, and 64-byte `XMemVirtualQuery` / working-set
// layouts informed by Marvisak/VIctory (https://github.com/Marvisak/VIctory, commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-hle/src/modules/xmem.rs) and
// xcl-dev/xcl (https://github.com/xcl-dev/xcl, commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/kernelx.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27 (updated 2026-11-28): Linux `memfd_create` +
// `mmap(MAP_SHARED | MAP_FIXED)` physical page pools, console-compliant `< 44-bit` aligned
// low-VA reservations (`MAP_FIXED_NOREPLACE | MAP_NORESERVE`), sparse 4 KiB extent tracking,
// Win32 `VirtualAlloc`/`VirtualFree`/`VirtualQuery` interop, and budget accounting.

//! Linux `memfd` + `mmap` implementation of all 13 console `xmem.dll` memory operations:
//! `XMemAlloc`, `XMemAllocDefault`, `XMemFree`, `XMemFreeDefault`,
//! `XMemAllocatePhysicalPages`, `XMemFreePhysicalPages`, `XMemMapPhysicalPages`,
//! `XMemMapPhysicalPagesScatter`, `XMemVirtualAlloc`, `XMemVirtualQuery`,
//! `XMemTransferMemory`, `XMemGetWorkingSetStatistics`, and `XMemIsToolsMemAvailable`.

use crate::title_memory_budget_bytes;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::mem::{offset_of, size_of};
use std::ptr;
use std::sync::Mutex;
use x2l_common::DeviceType;

/// Default physical page granule (64 KiB) matching console GDK physical pages.
pub const GRANULE_4K: usize = 4 * 1024;
pub const GRANULE_64K: usize = 64 * 1024;
pub const GRANULE_2M: usize = 2 * 1024 * 1024;
pub const GRANULE_4M: usize = 4 * 1024 * 1024;

/// Console low-VA search window: `[1 TiB .. 16 TiB)` (strictly `< 44-bit`).
pub const CONSOLE_VA_START: usize = 0x0000_0100_0000_0000;
pub const CONSOLE_VA_LIMIT: usize = 1usize << 44;

// Standard Win32 & console memory state / type / protect constants
pub const MEM_COMMIT: u32 = 0x0000_1000;
pub const MEM_RESERVE: u32 = 0x0000_2000;
pub const MEM_DECOMMIT: u32 = 0x0000_4000;
pub const MEM_RELEASE: u32 = 0x0000_8000;
pub const MEM_FREE: u32 = 0x0001_0000;
pub const MEM_PRIVATE: u32 = 0x0002_0000;
pub const MEM_MAPPED: u32 = 0x0004_0000;
pub const MEM_LARGE_PAGES: u32 = 0x2000_0000;
pub const MEM_64K_PAGES: u32 = 0x0040_0000;
pub const MEM_4MB_PAGES: u32 = 0x8000_0000;

pub const PAGE_NOACCESS: u32 = 0x01;
pub const PAGE_READONLY: u32 = 0x02;
pub const PAGE_READWRITE: u32 = 0x04;
pub const PAGE_EXECUTE_READ: u32 = 0x20;
pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;

pub const ERROR_INVALID_PARAMETER: u32 = 87;
pub const ERROR_INVALID_ADDRESS: u32 = 487;
pub const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;

/// Outcome of routing a Win32 `VirtualAlloc`/`VirtualFree` call through `XMemArena`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaOpResult<T> {
    /// Address does not belong to any `XMemArena` reservation; caller should fall through to Wine.
    NotOwned,
    /// Address belongs to `XMemArena` and the operation succeeded.
    Ok(T),
    /// Address belongs to `XMemArena` and failed with the given Win32 `GetLastError` code.
    Error(u32),
}

/// 64-byte `XMemVirtualQuery` output structure:
/// exact 48-byte Win64 `MEMORY_BASIC_INFORMATION` prefix followed by the 16-byte XMem tail.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XMemBasicInformation {
    pub base_address: u64,       // @0
    pub allocation_base: u64,    // @8
    pub allocation_protect: u32, // @16
    pub partition_id: u16,       // @20
    pub _pad0: u16,              // @22
    pub region_size: u64,        // @24
    pub state: u32,              // @32
    pub protect: u32,            // @36
    pub mem_type: u32,           // @40
    pub _pad1: u32,              // @44
    pub flags: u64,              // @48 (bit 0 = CPU committed, bit 1 = GPU accessible)
    pub tag_code: u32,           // @56
    pub tag: [u8; 4],            // @60
}

/// Common 16-byte prefix of `XMemGetWorkingSetStatistics` (safe for all observed caller buffer sizes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XMemWorkingSetPrefix {
    pub title_limit_bytes: u64, // @0
    pub title_used_bytes: u64,  // @8
}

// Compile-time Win64 ABI size and offset assertions.
const _: () = assert!(size_of::<XMemBasicInformation>() == 64);
const _: () = assert!(offset_of!(XMemBasicInformation, base_address) == 0);
const _: () = assert!(offset_of!(XMemBasicInformation, allocation_base) == 8);
const _: () = assert!(offset_of!(XMemBasicInformation, allocation_protect) == 16);
const _: () = assert!(offset_of!(XMemBasicInformation, partition_id) == 20);
const _: () = assert!(offset_of!(XMemBasicInformation, region_size) == 24);
const _: () = assert!(offset_of!(XMemBasicInformation, state) == 32);
const _: () = assert!(offset_of!(XMemBasicInformation, protect) == 36);
const _: () = assert!(offset_of!(XMemBasicInformation, mem_type) == 40);
const _: () = assert!(offset_of!(XMemBasicInformation, flags) == 48);
const _: () = assert!(offset_of!(XMemBasicInformation, tag_code) == 56);
const _: () = assert!(offset_of!(XMemBasicInformation, tag) == 60);
const _: () = assert!(size_of::<XMemWorkingSetPrefix>() == 16);

// ============================================================================
// Direct Linux libc FFI (Wine SUD-safe when executed on the native worker)
// ============================================================================

mod sys {
    use super::*;

    pub const MFD_CLOEXEC: c_uint = 0x0001;
    pub const FALLOC_FL_KEEP_SIZE: c_int = 0x01;
    pub const FALLOC_FL_PUNCH_HOLE: c_int = 0x02;

    pub const PROT_NONE: c_int = 0x0;
    pub const PROT_READ: c_int = 0x1;
    pub const PROT_WRITE: c_int = 0x2;
    pub const PROT_EXEC: c_int = 0x4;

    pub const MAP_SHARED: c_int = 0x01;
    pub const MAP_PRIVATE: c_int = 0x02;
    pub const MAP_FIXED: c_int = 0x10;
    pub const MAP_ANONYMOUS: c_int = 0x20;
    pub const MAP_NORESERVE: c_int = 0x4000;
    pub const MAP_FIXED_NOREPLACE: c_int = 0x10_0000;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    unsafe extern "C" {
        pub fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
        pub fn ftruncate(fd: c_int, length: i64) -> c_int;
        pub fn fallocate(fd: c_int, mode: c_int, offset: i64, len: i64) -> c_int;
        pub fn close(fd: c_int) -> c_int;
        pub fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> c_int;
        pub fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
    }

    pub fn win_protect_to_linux(protect: u32) -> c_int {
        let base = protect & 0xFF;
        match base {
            PAGE_NOACCESS => PROT_NONE,
            PAGE_READONLY => PROT_READ,
            PAGE_READWRITE | 0x08 => PROT_READ | PROT_WRITE,
            0x10 => PROT_EXEC,
            PAGE_EXECUTE_READ => PROT_READ | PROT_EXEC,
            PAGE_EXECUTE_READWRITE | 0x80 => PROT_READ | PROT_WRITE | PROT_EXEC,
            _ => PROT_READ | PROT_WRITE,
        }
    }
}

/// Supported backing physical page granule pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PageGranule {
    Size4K,
    Size64K,
    Size2M,
    Size4M,
}

impl PageGranule {
    pub const fn bytes(self) -> usize {
        match self {
            Self::Size4K => GRANULE_4K,
            Self::Size64K => GRANULE_64K,
            Self::Size2M => GRANULE_2M,
            Self::Size4M => GRANULE_4M,
        }
    }

    /// Number of 4 KiB PFN units occupied by one page of this granule.
    pub const fn pfn_step(self) -> usize {
        self.bytes() / GRANULE_4K
    }
}

struct MemfdPool {
    granule: PageGranule,
    fd: c_int,
    slot_count: usize,
    free_slots: Vec<usize>,
}

impl MemfdPool {
    fn new(granule: PageGranule) -> Option<Self> {
        const NAME: &[u8] = b"x2l-xmem-phys\0";
        // SAFETY: `NAME` is a static NUL-terminated string; `MFD_CLOEXEC` is a valid flag.
        let fd = unsafe { sys::memfd_create(NAME.as_ptr().cast::<c_char>(), sys::MFD_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        Some(Self {
            granule,
            fd,
            slot_count: 0,
            free_slots: Vec::new(),
        })
    }

    fn alloc_slot(&mut self) -> Option<usize> {
        let g = self.granule.bytes();
        if let Some(slot) = self.free_slots.pop() {
            let off = (slot as u64).checked_mul(g as u64)?;
            let off_i64 = i64::try_from(off).ok()?;
            let len_i64 = i64::try_from(g).ok()?;
            // Ensure zeroed pages on reuse via punch-hole + fallocate.
            // SAFETY: `self.fd` is an owned memfd covering `[off, off + g)`.
            unsafe {
                let _ = sys::fallocate(
                    self.fd,
                    sys::FALLOC_FL_PUNCH_HOLE | sys::FALLOC_FL_KEEP_SIZE,
                    off_i64,
                    len_i64,
                );
                let _ = sys::fallocate(self.fd, 0, off_i64, len_i64);
            }
            return Some(slot);
        }

        let slot = self.slot_count;
        let new_count = slot.checked_add(1)?;
        let new_len_u64 = (new_count as u64).checked_mul(g as u64)?;
        let new_len = i64::try_from(new_len_u64).ok()?;
        let off = i64::try_from((slot as u64).checked_mul(g as u64)?).ok()?;
        let len_i64 = i64::try_from(g).ok()?;

        // SAFETY: `self.fd` is a valid owned memfd; growing by one granule is checked against i64::MAX.
        let rc = unsafe { sys::ftruncate(self.fd, new_len) };
        if rc != 0 {
            return None;
        }
        // Preallocate backing pages so later writes never SIGBUS.
        // SAFETY: `[off, off + g)` lies within the newly truncated memfd.
        let _ = unsafe { sys::fallocate(self.fd, 0, off, len_i64) };
        self.slot_count = new_count;
        Some(slot)
    }

    fn recycle_slot(&mut self, slot: usize) {
        let g = self.granule.bytes();
        if let (Ok(off), Ok(len)) = (
            i64::try_from((slot as u64).saturating_mul(g as u64)),
            i64::try_from(g),
        ) {
            // SAFETY: `self.fd` is owned and `[off, off + len)` has zero remaining mapped aliases.
            let rc = unsafe {
                sys::fallocate(
                    self.fd,
                    sys::FALLOC_FL_PUNCH_HOLE | sys::FALLOC_FL_KEEP_SIZE,
                    off,
                    len,
                )
            };
            if rc == 0 {
                self.free_slots.push(slot);
            }
        }
    }
}

impl Drop for MemfdPool {
    fn drop(&mut self) {
        if self.fd >= 0 {
            // SAFETY: `MemfdPool` uniquely owns `self.fd`.
            unsafe {
                let _ = sys::close(self.fd);
            }
            self.fd = -1;
        }
    }
}

struct PhysPageRecord {
    granule: PageGranule,
    slot: usize,
    live: bool,
    aliases: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtentKind {
    Private { protect: u32 },
    Physical { alias_id: u64, protect: u32 },
}

impl ExtentKind {
    fn state_type_protect(self) -> (u32, u32, u32) {
        match self {
            Self::Private { protect } => (MEM_COMMIT, MEM_PRIVATE, protect),
            Self::Physical { protect, .. } => (MEM_COMMIT, MEM_MAPPED, protect),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageExtent {
    end: usize,
    kind: ExtentKind,
}

struct VirtualReservation {
    base: usize,
    size: usize,
    granule: PageGranule,
    alloc_protect: u32,
    gpu_accessible: bool,
    /// Sparse 4 KiB-aligned committed/mapped extents keyed by start VA. Gaps are `MEM_RESERVE`.
    extents: BTreeMap<usize, PageExtent>,
}

impl VirtualReservation {
    fn end(&self) -> usize {
        self.base.saturating_add(self.size)
    }
}

struct ArenaInner {
    device: DeviceType,
    budget_limit: u64,
    heap_allocs: BTreeMap<usize, Layout>,
    heap_bytes_used: u64,
    phys_bytes_used: u64,
    private_bytes_used: u64,
    pools: BTreeMap<PageGranule, MemfdPool>,
    phys_pages: BTreeMap<usize, PhysPageRecord>,
    next_pfn_4k: usize,
    next_alias_id: u64,
    /// Maps each installed physical mapping `alias_id -> (pfn, remaining_mapped_bytes)`.
    alias_surviving_bytes: BTreeMap<u64, (usize, usize)>,
    reservations: BTreeMap<usize, VirtualReservation>,
}

/// Read current mapped VA intervals `[start, end)` from `/proc/self/maps` as a search hint.
fn read_proc_self_maps_ranges() -> Vec<(usize, usize)> {
    let Ok(text) = std::fs::read_to_string("/proc/self/maps") else {
        return Vec::new();
    };
    let mut ranges = Vec::new();
    for line in text.lines() {
        let Some(range_part) = line.split_whitespace().next() else {
            continue;
        };
        let Some((s_hex, e_hex)) = range_part.split_once('-') else {
            continue;
        };
        if let (Ok(s), Ok(e)) = (
            usize::from_str_radix(s_hex, 16),
            usize::from_str_radix(e_hex, 16),
        ) && e > s
        {
            ranges.push((s, e));
        }
    }
    ranges.sort_unstable();
    ranges
}

/// Find a free `< 44-bit` aligned hole of length `size` aligned to `align` and claim it atomically
/// via `MAP_FIXED_NOREPLACE | MAP_NORESERVE`.
fn reserve_low_va_hole(size: usize, align: usize) -> Option<usize> {
    let align = align.max(GRANULE_64K);
    let maps = read_proc_self_maps_ranges();

    let try_claim = |cand: usize| -> bool {
        // SAFETY: `MAP_FIXED_NOREPLACE | MAP_NORESERVE` atomically claims `[cand, cand + size)`
        // as `PROT_NONE` without disturbing any existing mapping or charging physical RAM.
        let mapped = unsafe {
            sys::mmap(
                cand as *mut c_void,
                size,
                sys::PROT_NONE,
                sys::MAP_PRIVATE
                    | sys::MAP_ANONYMOUS
                    | sys::MAP_FIXED_NOREPLACE
                    | sys::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if mapped == sys::MAP_FAILED {
            false
        } else if (mapped as usize) != cand {
            // SAFETY: Defensive cleanup if a kernel ignored `MAP_FIXED_NOREPLACE`.
            unsafe {
                let _ = sys::munmap(mapped, size);
            }
            false
        } else {
            true
        }
    };

    // Walk gaps in `[CONSOLE_VA_START .. CONSOLE_VA_LIMIT)`.
    let mut cursor = round_up_usize(CONSOLE_VA_START, align)?;
    for &(ms, me) in &maps {
        if me <= cursor {
            continue;
        }
        if ms >= CONSOLE_VA_LIMIT {
            break;
        }
        while let Some(end) = cursor.checked_add(size)
            && end <= ms
            && end <= CONSOLE_VA_LIMIT
        {
            if try_claim(cursor) {
                return Some(cursor);
            }
            // Concurrent collision — step forward past the conflicting region.
            cursor = cursor.checked_add(align.max(GRANULE_4M))?;
        }
        if me > cursor {
            cursor = round_up_usize(me, align)?;
        }
    }

    // Trailing region after last `/proc/self/maps` entry below `CONSOLE_VA_LIMIT`.
    while let Some(end) = cursor.checked_add(size)
        && end <= CONSOLE_VA_LIMIT
    {
        if try_claim(cursor) {
            return Some(cursor);
        }
        cursor = cursor.checked_add(align.max(GRANULE_4M))?;
    }

    None
}

impl ArenaInner {
    fn new(device: DeviceType) -> Self {
        Self {
            device,
            budget_limit: title_memory_budget_bytes(device),
            heap_allocs: BTreeMap::new(),
            heap_bytes_used: 0,
            phys_bytes_used: 0,
            private_bytes_used: 0,
            pools: BTreeMap::new(),
            phys_pages: BTreeMap::new(),
            // Start at PFN 0x10 (64 KiB aligned, never 0, stays within 21-bit `pfn >> 4` packing).
            next_pfn_4k: 0x10,
            next_alias_id: 1,
            alias_surviving_bytes: BTreeMap::new(),
            reservations: BTreeMap::new(),
        }
    }

    fn total_used_bytes(&self) -> u64 {
        self.heap_bytes_used
            .saturating_add(self.phys_bytes_used)
            .saturating_add(self.private_bytes_used)
    }

    fn ensure_pool(&mut self, granule: PageGranule) -> Option<&mut MemfdPool> {
        if let std::collections::btree_map::Entry::Vacant(e) = self.pools.entry(granule) {
            let pool = MemfdPool::new(granule)?;
            e.insert(pool);
        }
        self.pools.get_mut(&granule)
    }

    fn next_aligned_pfn(&mut self, granule: PageGranule) -> Option<usize> {
        let step = granule.pfn_step();
        let aligned = round_up_usize(self.next_pfn_4k, step)?;
        // Preserve 21-bit packing for 64 KiB PFNs (`pfn >> 4` < 2^21).
        if granule == PageGranule::Size64K && (aligned >> 4) >= (1usize << 21) {
            return None;
        }
        self.next_pfn_4k = aligned.checked_add(step)?;
        Some(aligned)
    }

    fn find_reservation_key(&self, addr: usize) -> Option<usize> {
        let (&base, res) = self.reservations.range(..=addr).next_back()?;
        if addr < res.end() { Some(base) } else { None }
    }

    fn next_reservation_after(&self, addr: usize) -> Option<usize> {
        self.reservations
            .range((std::ops::Bound::Excluded(addr), std::ops::Bound::Unbounded))
            .next()
            .map(|(&base, _)| base)
    }

    fn reserve_va_range(
        &mut self,
        requested: usize,
        size: usize,
        granule: PageGranule,
        alloc_protect: u32,
        gpu_accessible: bool,
    ) -> Option<usize> {
        let g = granule.bytes();
        let aligned_size = round_up_usize(size.max(1), g)?;

        let base = if requested != 0 {
            if !requested.is_multiple_of(g) {
                return None;
            }
            let req_end = requested.checked_add(aligned_size)?;
            if req_end > CONSOLE_VA_LIMIT {
                return None;
            }
            if let Some(existing_key) = self.find_reservation_key(requested) {
                let existing = &self.reservations[&existing_key];
                if req_end <= existing.end() {
                    return Some(requested);
                }
                return None;
            }
            // SAFETY: `MAP_FIXED_NOREPLACE | MAP_NORESERVE` atomically fails without overwriting
            // any Wine mapping or committing physical RAM.
            let mapped = unsafe {
                sys::mmap(
                    requested as *mut c_void,
                    aligned_size,
                    sys::PROT_NONE,
                    sys::MAP_PRIVATE
                        | sys::MAP_ANONYMOUS
                        | sys::MAP_FIXED_NOREPLACE
                        | sys::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if mapped == sys::MAP_FAILED || (mapped as usize) != requested {
                return None;
            }
            requested
        } else {
            reserve_low_va_hole(aligned_size, g)?
        };

        self.reservations.insert(
            base,
            VirtualReservation {
                base,
                size: aligned_size,
                granule,
                alloc_protect,
                gpu_accessible,
                extents: BTreeMap::new(),
            },
        );
        Some(base)
    }

    fn decrement_pfn_alias(&mut self, pfn: usize) {
        let mut recycle: Option<(PageGranule, usize)> = None;
        if let Some(rec) = self.phys_pages.get_mut(&pfn) {
            rec.aliases = rec.aliases.saturating_sub(1);
            if !rec.live && rec.aliases == 0 {
                recycle = Some((rec.granule, rec.slot));
            }
        }
        if let Some((granule, slot)) = recycle {
            self.phys_pages.remove(&pfn);
            self.phys_bytes_used = self.phys_bytes_used.saturating_sub(granule.bytes() as u64);
            if let Some(pool) = self.pools.get_mut(&granule) {
                pool.recycle_slot(slot);
            }
        }
    }

    fn drop_extent_bytes(&mut self, kind: ExtentKind, bytes: usize) {
        match kind {
            ExtentKind::Private { .. } => {
                self.private_bytes_used = self.private_bytes_used.saturating_sub(bytes as u64);
            }
            ExtentKind::Physical { alias_id, .. } => {
                let mut retired_pfn = None;
                if let Some((pfn, rem)) = self.alias_surviving_bytes.get_mut(&alias_id) {
                    *rem = rem.saturating_sub(bytes);
                    if *rem == 0 {
                        retired_pfn = Some(*pfn);
                    }
                }
                if let Some(pfn) = retired_pfn {
                    self.alias_surviving_bytes.remove(&alias_id);
                    self.decrement_pfn_alias(pfn);
                }
            }
        }
    }

    /// Clear metadata in `[start, end)` inside `res_base` (splitting boundary extents cleanly)
    /// and update private/physical accounting. Does NOT touch OS page tables.
    fn clear_extents_range(&mut self, res_base: usize, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let mut removed: Vec<(ExtentKind, usize)> = Vec::new();
        let mut reinsert: Vec<(usize, PageExtent)> = Vec::new();

        if let Some(res) = self.reservations.get_mut(&res_base) {
            // Find the first extent that could overlap `[start, end)`.
            let first_key = res
                .extents
                .range(..=start)
                .next_back()
                .filter(|(_, ext)| ext.end > start)
                .map(|(&k, _)| k);
            let search_start = first_key.unwrap_or(start);

            let keys: Vec<usize> = res
                .extents
                .range(search_start..end)
                .map(|(&k, _)| k)
                .collect();

            for k in keys {
                if let Some(ext) = res.extents.remove(&k) {
                    let ov_start = k.max(start);
                    let ov_end = ext.end.min(end);
                    if ov_start < ov_end {
                        removed.push((ext.kind, ov_end - ov_start));
                    }
                    if k < start {
                        reinsert.push((
                            k,
                            PageExtent {
                                end: start,
                                kind: ext.kind,
                            },
                        ));
                    }
                    if ext.end > end {
                        reinsert.push((
                            end,
                            PageExtent {
                                end: ext.end,
                                kind: ext.kind,
                            },
                        ));
                    }
                }
            }
            for (k, v) in reinsert {
                res.extents.insert(k, v);
            }
        }

        for (kind, bytes) in removed {
            self.drop_extent_bytes(kind, bytes);
        }
    }

    /// Coalesce adjacent `Private` extents with matching protection around `res_base`.
    fn coalesce_around(&mut self, res_base: usize, start: usize) {
        let Some(res) = self.reservations.get_mut(&res_base) else {
            return;
        };
        // Merge with predecessor if contiguous and identical Private protect.
        let mut cur_start = start;
        if let Some((&prev_start, &prev_ext)) = res.extents.range(..cur_start).next_back()
            && prev_ext.end == cur_start
            && let Some(&cur_ext) = res.extents.get(&cur_start)
            && matches!(prev_ext.kind, ExtentKind::Private { .. })
            && prev_ext.kind == cur_ext.kind
        {
            res.extents.remove(&cur_start);
            if let Some(p) = res.extents.get_mut(&prev_start) {
                p.end = cur_ext.end;
            }
            cur_start = prev_start;
        }
        // Merge with successor if contiguous and identical Private protect.
        if let Some(&cur_ext) = res.extents.get(&cur_start)
            && let Some(&next_ext) = res.extents.get(&cur_ext.end)
            && matches!(cur_ext.kind, ExtentKind::Private { .. })
            && cur_ext.kind == next_ext.kind
        {
            let next_key = cur_ext.end;
            res.extents.remove(&next_key);
            if let Some(c) = res.extents.get_mut(&cur_start) {
                c.end = next_ext.end;
            }
        }
    }

    /// Commit `[start, end)` (`4 KiB`-aligned, strictly inside `res_base`) with `clean_prot`.
    /// Preserves existing physical memfd aliases while updating their protection in place.
    fn commit_range(&mut self, res_base: usize, start: usize, end: usize, clean_prot: u32) -> bool {
        if start >= end {
            return true;
        }
        let Some(res) = self.reservations.get(&res_base) else {
            return false;
        };
        if start < res.base || end > res.end() {
            return false;
        }
        let linux_prot = sys::win_protect_to_linux(clean_prot);
        // SAFETY: `[start, end)` lies strictly within an arena-owned reservation.
        let rc = unsafe { sys::mprotect(start as *mut c_void, end - start, linux_prot) };
        if rc != 0 {
            return false;
        }

        // Split any boundary extents at `start` and `end`, update Physical extents in place,
        // and fill gaps with Private { protect: clean_prot }.
        let mut newly_private_bytes = 0usize;
        if let Some(res) = self.reservations.get_mut(&res_base) {
            // Split extent straddling `start`
            if let Some((&k, &ext)) = res.extents.range(..start).next_back()
                && ext.end > start
            {
                res.extents.get_mut(&k).expect("present").end = start;
                res.extents.insert(
                    start,
                    PageExtent {
                        end: ext.end,
                        kind: ext.kind,
                    },
                );
            }
            // Split extent straddling `end`
            if let Some((&k, &ext)) = res.extents.range(..end).next_back()
                && ext.end > end
            {
                res.extents.get_mut(&k).expect("present").end = end;
                res.extents.insert(
                    end,
                    PageExtent {
                        end: ext.end,
                        kind: ext.kind,
                    },
                );
            }

            let existing_keys: Vec<(usize, PageExtent)> = res
                .extents
                .range(start..end)
                .map(|(&k, &v)| (k, v))
                .collect();

            let mut cursor = start;
            let mut to_insert = Vec::new();
            for (k, ext) in existing_keys {
                if k > cursor {
                    newly_private_bytes = newly_private_bytes.saturating_add(k - cursor);
                    to_insert.push((
                        cursor,
                        PageExtent {
                            end: k,
                            kind: ExtentKind::Private {
                                protect: clean_prot,
                            },
                        },
                    ));
                }
                if let Some(entry) = res.extents.get_mut(&k) {
                    match &mut entry.kind {
                        ExtentKind::Private { protect } | ExtentKind::Physical { protect, .. } => {
                            *protect = clean_prot;
                        }
                    }
                }
                cursor = ext.end;
            }
            if cursor < end {
                newly_private_bytes = newly_private_bytes.saturating_add(end - cursor);
                to_insert.push((
                    cursor,
                    PageExtent {
                        end,
                        kind: ExtentKind::Private {
                            protect: clean_prot,
                        },
                    },
                ));
            }
            for (k, v) in to_insert {
                res.extents.insert(k, v);
            }
        }

        self.private_bytes_used = self
            .private_bytes_used
            .saturating_add(newly_private_bytes as u64);
        self.coalesce_around(res_base, start);
        if end > start {
            // Also try coalescing at the tail boundary.
            if let Some((&last_k, _)) = self
                .reservations
                .get(&res_base)
                .and_then(|r| r.extents.range(..end).next_back())
            {
                self.coalesce_around(res_base, last_k);
            }
        }
        true
    }

    /// Decommit `[start, end)` (`4 KiB`-aligned, strictly inside `res_base`), atomically dropping
    /// private pages or shared memfd aliases back to anonymous `PROT_NONE` (zero-filled on recommit).
    fn decommit_range(&mut self, res_base: usize, start: usize, end: usize) -> bool {
        if start >= end {
            return true;
        }
        let Some(res) = self.reservations.get(&res_base) else {
            return false;
        };
        if start < res.base || end > res.end() {
            return false;
        }
        // SAFETY: `[start, end)` lies inside an arena-owned reservation; replacing with
        // `MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE` `PROT_NONE` atomically drops
        // dirty private pages and any shared memfd alias.
        let rc = unsafe {
            sys::mmap(
                start as *mut c_void,
                end - start,
                sys::PROT_NONE,
                sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_FIXED | sys::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if rc == sys::MAP_FAILED {
            return false;
        }
        self.clear_extents_range(res_base, start, end);
        true
    }

    /// Install a `MAP_SHARED | MAP_FIXED` physical alias at `[va, va + g)` inside `res_base`.
    fn install_physical_alias(
        &mut self,
        res_base: usize,
        va: usize,
        g: usize,
        pfn: usize,
        pool_fd: c_int,
        file_off: i64,
    ) -> bool {
        // SAFETY: `[va, va + g)` is inside an arena-owned reservation and `pool_fd` covers
        // `[file_off, file_off + g)`.
        let mapped = unsafe {
            sys::mmap(
                va as *mut c_void,
                g,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_SHARED | sys::MAP_FIXED,
                pool_fd,
                file_off,
            )
        };
        if mapped == sys::MAP_FAILED || (mapped as usize) != va {
            return false;
        }
        // Retain new PFN reference BEFORE dropping old extents so self-remap cannot recycle early.
        let alias_id = self.next_alias_id;
        self.next_alias_id = self.next_alias_id.saturating_add(1);
        if let Some(rec) = self.phys_pages.get_mut(&pfn) {
            rec.aliases += 1;
        }
        self.alias_surviving_bytes.insert(alias_id, (pfn, g));

        self.clear_extents_range(res_base, va, va + g);
        if let Some(res) = self.reservations.get_mut(&res_base) {
            res.extents.insert(
                va,
                PageExtent {
                    end: va + g,
                    kind: ExtentKind::Physical {
                        alias_id,
                        protect: PAGE_READWRITE,
                    },
                },
            );
        }
        true
    }
}

impl Drop for ArenaInner {
    fn drop(&mut self) {
        let reservations = std::mem::take(&mut self.reservations);
        for (_, res) in reservations {
            // SAFETY: Each `VirtualReservation` uniquely owns `[res.base, res.base + res.size)`.
            unsafe {
                let _ = sys::munmap(res.base as *mut c_void, res.size);
            }
        }
        let allocs = std::mem::take(&mut self.heap_allocs);
        for (addr, layout) in allocs {
            // SAFETY: Each entry came from `alloc_zeroed(layout)` and has not been freed.
            unsafe {
                dealloc(addr as *mut u8, layout);
            }
        }
        // `self.pools` drops cleanly after all `MAP_SHARED` views have been unmapped.
    }
}

/// Thread-safe Linux `xmem.dll` backing arena managing heap, physical page pools, and scatter VAs.
pub struct XMemArena {
    inner: Mutex<ArenaInner>,
}

impl XMemArena {
    /// Construct a fresh `XMemArena` sized to the configured console hardware profile.
    pub fn new(device: DeviceType) -> Self {
        Self {
            inner: Mutex::new(ArenaInner::new(device)),
        }
    }

    /// Return the next reservation base strictly greater than `addr` (for clamping Wine `MEM_FREE`).
    pub fn next_reservation_after(&self, addr: usize) -> Option<usize> {
        self.inner.lock().ok()?.next_reservation_after(addr)
    }

    /// `XMemAlloc` / `XMemAllocDefault`: zero-initialized heap allocation decoding alignment log2
    /// from attribute bits 24..=28 (minimum 16-byte alignment, capped to 64 KiB).
    pub fn alloc(&self, size: usize, attributes: u64) -> *mut u8 {
        let alloc_size = size.max(1);
        let align_log2 = ((attributes >> 24) & 0x1F) as u32;
        let align = if align_log2 == 0 {
            16usize
        } else {
            1usize.checked_shl(align_log2.min(16)).unwrap_or(16).max(16)
        };
        let Ok(layout) = Layout::from_size_align(alloc_size, align) else {
            return ptr::null_mut();
        };
        // SAFETY: `layout` has non-zero size and verified power-of-two alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return ptr;
        }
        if let Ok(mut guard) = self.inner.lock() {
            guard.heap_allocs.insert(ptr as usize, layout);
            guard.heap_bytes_used = guard.heap_bytes_used.saturating_add(alloc_size as u64);
        }
        ptr
    }

    /// `XMemFree` / `XMemFreeDefault`: release a heap allocation previously returned by [`Self::alloc`].
    pub fn free(&self, addr: usize, _attributes: u64) -> bool {
        if addr == 0 {
            return true;
        }
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        let Some(layout) = guard.heap_allocs.remove(&addr) else {
            return false;
        };
        guard.heap_bytes_used = guard.heap_bytes_used.saturating_sub(layout.size() as u64);
        // SAFETY: `addr` was returned by `alloc_zeroed(layout)` and removed so it cannot double-free.
        unsafe { dealloc(addr as *mut u8, layout) };
        true
    }

    /// `XMemAllocatePhysicalPages`: allocate `count` physical pages from the granule's `memfd` pool
    /// (default 64 KiB granule) and return their unique 4 KiB-unit PFNs.
    pub fn allocate_physical_pages(
        &self,
        flags: u64,
        count: usize,
        granule_override: Option<PageGranule>,
    ) -> Option<Vec<usize>> {
        if count == 0 {
            return Some(Vec::new());
        }
        let granule = granule_override.unwrap_or_else(|| decode_phys_granule(flags));
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let needed_bytes = (count as u64).checked_mul(granule.bytes() as u64)?;
        if guard.total_used_bytes().saturating_add(needed_bytes) > guard.budget_limit {
            return None;
        }

        let mut allocated: Vec<(usize, usize)> = Vec::with_capacity(count);
        for _ in 0..count {
            let Some(pfn) = guard.next_aligned_pfn(granule) else {
                rollback_phys_alloc(&mut guard, granule, &allocated);
                return None;
            };
            let Some(slot) = guard.ensure_pool(granule).and_then(|p| p.alloc_slot()) else {
                rollback_phys_alloc(&mut guard, granule, &allocated);
                return None;
            };
            allocated.push((pfn, slot));
        }

        let mut pfns = Vec::with_capacity(count);
        for (pfn, slot) in allocated {
            guard.phys_pages.insert(
                pfn,
                PhysPageRecord {
                    granule,
                    slot,
                    live: true,
                    aliases: 0,
                },
            );
            guard.phys_bytes_used = guard.phys_bytes_used.saturating_add(granule.bytes() as u64);
            pfns.push(pfn);
        }
        Some(pfns)
    }

    /// `XMemFreePhysicalPages`: deduplicate and retire physical page IDs. Any active mapped aliases
    /// remain backed until unmapped, at which point the backing slot is hole-punched and recycled.
    pub fn free_physical_pages(&self, pfns: &[usize]) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        let mut unique = BTreeSet::new();
        for &pfn in pfns {
            if pfn != 0 {
                unique.insert(pfn);
            }
        }
        for pfn in unique {
            let mut recycle: Option<(PageGranule, usize)> = None;
            if let Some(rec) = guard.phys_pages.get_mut(&pfn) {
                rec.live = false;
                if rec.aliases == 0 {
                    recycle = Some((rec.granule, rec.slot));
                }
            }
            if let Some((granule, slot)) = recycle {
                guard.phys_pages.remove(&pfn);
                guard.phys_bytes_used =
                    guard.phys_bytes_used.saturating_sub(granule.bytes() as u64);
                if let Some(pool) = guard.pools.get_mut(&granule) {
                    pool.recycle_slot(slot);
                }
            }
        }
        true
    }

    /// `XMemMapPhysicalPages`: map `pfns` contiguously starting at `base` (or reserve a fresh
    /// `< 44-bit` aligned VA range when `base == 0`). Passing `pfns == None` unmaps back to `PROT_NONE`.
    pub fn map_physical_pages(
        &self,
        base: usize,
        count: usize,
        pfns: Option<&[usize]>,
    ) -> Option<usize> {
        if count == 0 {
            return (base != 0).then_some(base);
        }
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };

        let Some(pfn_slice) = pfns else {
            // Unmap `[base, base + count * granule)` back to `PROT_NONE`.
            let res_key = guard.find_reservation_key(base)?;
            let (res_base, g, res_end) = {
                let res = &guard.reservations[&res_key];
                (res.base, res.granule.bytes(), res.end())
            };
            let span = count.checked_mul(g)?;
            let end = base.checked_add(span)?;
            if !(base - res_base).is_multiple_of(GRANULE_4K) || end > res_end {
                return None;
            }
            if !guard.decommit_range(res_base, base, end) {
                return None;
            }
            return Some(base);
        };

        if pfn_slice.len() != count {
            return None;
        }

        // Validate all PFNs first and determine uniform granule.
        let mut granule: Option<PageGranule> = None;
        let mut targets: Vec<(usize, PageGranule, usize)> = Vec::with_capacity(count);
        for &pfn in pfn_slice {
            if pfn == 0 {
                targets.push((0, granule.unwrap_or(PageGranule::Size64K), 0));
                continue;
            }
            let rec = guard.phys_pages.get(&pfn)?;
            if !rec.live {
                return None;
            }
            match granule {
                Some(g) if g != rec.granule => return None,
                None => granule = Some(rec.granule),
                _ => {}
            }
            targets.push((pfn, rec.granule, rec.slot));
        }
        let granule = granule.unwrap_or(PageGranule::Size64K);
        let g = granule.bytes();
        let total_bytes = count.checked_mul(g)?;

        let created_new = if base == 0 {
            true
        } else {
            guard.find_reservation_key(base).is_none()
        };
        let mapped_base =
            guard.reserve_va_range(base, total_bytes, granule, PAGE_READWRITE, true)?;
        let res_key = guard.find_reservation_key(mapped_base)?;
        let (res_base, res_end) = {
            let res = &guard.reservations[&res_key];
            (res.base, res.end())
        };
        if !(mapped_base - res_base).is_multiple_of(g)
            || mapped_base.checked_add(total_bytes)? > res_end
        {
            return None;
        }

        let pool_fd = guard.pools.get(&granule)?.fd;

        // Map each page transactionally using `MAP_SHARED | MAP_FIXED`.
        for (i, &(pfn, _, slot)) in targets.iter().enumerate() {
            let va = mapped_base + i * g;
            if pfn == 0 {
                if !guard.decommit_range(res_base, va, va + g) {
                    if created_new && let Some(res) = guard.reservations.remove(&res_base) {
                        // SAFETY: Newly reserved range being rolled back.
                        unsafe {
                            let _ = sys::munmap(res.base as *mut c_void, res.size);
                        }
                    }
                    return None;
                }
                continue;
            }
            let file_off = i64::try_from((slot as u64).checked_mul(g as u64)?).ok()?;
            if !guard.install_physical_alias(res_base, va, g, pfn, pool_fd, file_off) {
                if created_new {
                    let _ = guard.decommit_range(res_base, mapped_base, va);
                    if let Some(res) = guard.reservations.remove(&res_base) {
                        // SAFETY: Roll back newly reserved VA range on failure.
                        unsafe {
                            let _ = sys::munmap(res.base as *mut c_void, res.size);
                        }
                    }
                }
                return None;
            }
        }

        Some(mapped_base)
    }

    /// `XMemMapPhysicalPagesScatter`: map each `pfns[i]` onto its corresponding caller-supplied
    /// reserved virtual address `addresses[i]` (with PFN `0` restoring `PROT_NONE`).
    /// Supports mapping 64 KiB physical pages inside 64 KiB, 2 MiB, or 4 MiB aligned VA arenas.
    pub fn map_physical_pages_scatter(&self, addresses: &[usize], pfns: &[usize]) -> bool {
        if addresses.len() != pfns.len() {
            return false;
        }
        if addresses.is_empty() {
            return true;
        }
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };

        // Pre-validate every `(va, pfn)` target before mutating any page table mapping.
        struct ScatterOp {
            res_base: usize,
            va: usize,
            pfn: usize,
            pool_fd: c_int,
            file_off: i64,
            granule_bytes: usize,
        }
        let mut ops = Vec::with_capacity(addresses.len());
        for (&va, &pfn) in addresses.iter().zip(pfns.iter()) {
            let Some(res_key) = guard.find_reservation_key(va) else {
                return false;
            };
            let (res_base, res_granule, res_end) = {
                let res = &guard.reservations[&res_key];
                (res.base, res.granule, res.end())
            };
            if pfn == 0 {
                let g = res_granule.bytes().min(GRANULE_64K);
                if !(va - res_base).is_multiple_of(g)
                    || va.checked_add(g).is_none_or(|e| e > res_end)
                {
                    return false;
                }
                ops.push(ScatterOp {
                    res_base,
                    va,
                    pfn: 0,
                    pool_fd: -1,
                    file_off: 0,
                    granule_bytes: g,
                });
            } else {
                let Some(rec) = guard.phys_pages.get(&pfn) else {
                    return false;
                };
                if !rec.live {
                    return false;
                }
                let g = rec.granule.bytes();
                if !(va - res_base).is_multiple_of(g)
                    || va.checked_add(g).is_none_or(|e| e > res_end)
                {
                    return false;
                }
                let Some(pool) = guard.pools.get(&rec.granule) else {
                    return false;
                };
                let Some(off_u64) = (rec.slot as u64).checked_mul(g as u64) else {
                    return false;
                };
                let Ok(file_off) = i64::try_from(off_u64) else {
                    return false;
                };
                ops.push(ScatterOp {
                    res_base,
                    va,
                    pfn,
                    pool_fd: pool.fd,
                    file_off,
                    granule_bytes: g,
                });
            }
        }

        for op in ops {
            if op.pfn == 0 {
                if !guard.decommit_range(op.res_base, op.va, op.va + op.granule_bytes) {
                    return false;
                }
                continue;
            }
            if !guard.install_physical_alias(
                op.res_base,
                op.va,
                op.granule_bytes,
                op.pfn,
                op.pool_fd,
                op.file_off,
            ) {
                return false;
            }
        }

        true
    }

    /// `XMemVirtualAlloc`: reserve and/or commit arena-owned virtual address space in the console
    /// `< 44-bit` low-VA window with granule alignment (`64 KiB`, `2 MiB`, or `4 MiB` for
    /// `MEM_4MB_PAGES`).
    pub fn virtual_alloc(
        &self,
        addr: usize,
        size: usize,
        alloc_type: u32,
        flags: u64,
        protect: u32,
    ) -> Option<usize> {
        if size == 0 {
            return None;
        }
        let granule = decode_virtual_granule(alloc_type, flags);
        let g = granule.bytes();
        let clean_prot = if protect & 0xFF == 0 {
            PAGE_READWRITE
        } else {
            protect & 0xFF
        };

        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };

        let is_reserve = (alloc_type & MEM_RESERVE) != 0 || addr == 0;
        let is_commit = (alloc_type & MEM_COMMIT) != 0;

        if is_reserve {
            let aligned_size = round_up_usize(size, g)?;
            let base = guard.reserve_va_range(addr, aligned_size, granule, clean_prot, true)?;
            if is_commit && !guard.commit_range(base, base, base + aligned_size, clean_prot) {
                return None;
            }
            Some(base)
        } else if is_commit {
            let res_key = guard.find_reservation_key(addr)?;
            let start = round_down_usize(addr, GRANULE_4K);
            let end = round_up_usize(addr.checked_add(size)?, GRANULE_4K)?;
            if !guard.commit_range(res_key, start, end, clean_prot) {
                return None;
            }
            Some(start)
        } else {
            None
        }
    }

    /// Route a Win32 `VirtualAlloc`/`VirtualAllocEx`/`VirtualAllocFromApp` call into `XMemArena`
    /// when `addr` lies inside an arena-owned reservation. Returns `ArenaOpResult::NotOwned` for
    /// non-arena addresses so Wine handles standard Win32 allocations untouched.
    pub fn try_win32_virtual_alloc(
        &self,
        addr: usize,
        size: usize,
        alloc_type: u32,
        protect: u32,
    ) -> ArenaOpResult<usize> {
        if addr == 0 {
            return ArenaOpResult::NotOwned;
        }
        let Ok(mut guard) = self.inner.lock() else {
            return ArenaOpResult::NotOwned;
        };
        let Some(res_key) = guard.find_reservation_key(addr) else {
            return ArenaOpResult::NotOwned;
        };
        if size == 0 {
            return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
        }
        // Re-reserving inside an already-reserved arena range is invalid per Win32 semantics.
        if (alloc_type & MEM_RESERVE) != 0 {
            return ArenaOpResult::Error(ERROR_INVALID_ADDRESS);
        }
        if (alloc_type & MEM_COMMIT) == 0 {
            return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
        }
        let start = round_down_usize(addr, GRANULE_4K);
        let Some(end) = addr
            .checked_add(size)
            .and_then(|e| round_up_usize(e, GRANULE_4K))
        else {
            return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
        };
        if end > guard.reservations[&res_key].end() {
            return ArenaOpResult::Error(ERROR_INVALID_ADDRESS);
        }
        let clean_prot = if protect & 0xFF == 0 {
            PAGE_READWRITE
        } else {
            protect & 0xFF
        };
        if guard.commit_range(res_key, start, end, clean_prot) {
            ArenaOpResult::Ok(start)
        } else {
            ArenaOpResult::Error(ERROR_NOT_ENOUGH_MEMORY)
        }
    }

    /// Route a Win32 `VirtualFree` call into `XMemArena` when `addr` lies inside an arena
    /// reservation. Enforces exact Win32 release (`addr == base && size == 0`) and 4 KiB-rounded
    /// decommit semantics.
    pub fn try_win32_virtual_free(
        &self,
        addr: usize,
        size: usize,
        free_type: u32,
    ) -> ArenaOpResult<()> {
        if addr == 0 {
            return ArenaOpResult::NotOwned;
        }
        let Ok(mut guard) = self.inner.lock() else {
            return ArenaOpResult::NotOwned;
        };
        let Some(res_key) = guard.find_reservation_key(addr) else {
            return ArenaOpResult::NotOwned;
        };

        let is_release = (free_type & MEM_RELEASE) != 0;
        let is_decommit = (free_type & MEM_DECOMMIT) != 0;
        if is_release == is_decommit {
            return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
        }

        if is_release {
            if addr != res_key || size != 0 {
                return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
            }
            let Some(res) = guard.reservations.remove(&res_key) else {
                return ArenaOpResult::Error(ERROR_INVALID_ADDRESS);
            };
            for (&start, &ext) in &res.extents {
                guard.drop_extent_bytes(ext.kind, ext.end - start);
            }
            // SAFETY: `res` uniquely owns `[res.base, res.base + res.size)`.
            unsafe {
                let _ = sys::munmap(res.base as *mut c_void, res.size);
            }
            return ArenaOpResult::Ok(());
        }

        // MEM_DECOMMIT
        let (res_base, res_end) = {
            let res = &guard.reservations[&res_key];
            (res.base, res.end())
        };
        let (start, end) = if size == 0 {
            if addr != res_base {
                return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
            }
            (res_base, res_end)
        } else {
            let s = round_down_usize(addr, GRANULE_4K);
            let Some(e) = addr
                .checked_add(size)
                .and_then(|x| round_up_usize(x, GRANULE_4K))
            else {
                return ArenaOpResult::Error(ERROR_INVALID_PARAMETER);
            };
            if e > res_end {
                return ArenaOpResult::Error(ERROR_INVALID_ADDRESS);
            }
            (s, e)
        };
        if guard.decommit_range(res_base, start, end) {
            ArenaOpResult::Ok(())
        } else {
            ArenaOpResult::Error(ERROR_INVALID_ADDRESS)
        }
    }

    /// Release or decommit an arena-owned virtual reservation. Returns `true` if `addr` belonged
    /// to this arena and was released/decommitted.
    pub fn virtual_free(&self, addr: usize, size: usize, free_type: u32) -> bool {
        matches!(
            self.try_win32_virtual_free(addr, size, free_type),
            ArenaOpResult::Ok(())
        )
    }

    /// `XMemVirtualQuery` / Win32 `VirtualQuery`: return the 64-byte `XMemBasicInformation`
    /// (whose first 48 bytes are exact Win64 `MEMORY_BASIC_INFORMATION`) describing the contiguous
    /// forward run from the 4 KiB-aligned page containing `addr` inside an arena reservation.
    pub fn virtual_query(&self, addr: usize) -> Option<XMemBasicInformation> {
        let Ok(guard) = self.inner.lock() else {
            return None;
        };
        let res_key = guard.find_reservation_key(addr)?;
        let res = &guard.reservations[&res_key];
        let page_base = round_down_usize(addr, GRANULE_4K);
        let res_end = res.end();

        // Check if `page_base` lies inside an active extent or in a reserved gap.
        let covering = res
            .extents
            .range(..=page_base)
            .next_back()
            .filter(|(_, ext)| ext.end > page_base)
            .map(|(&k, &ext)| (k, ext));

        let (state, mem_type, protect, region_end) = if let Some((_, first_ext)) = covering {
            let (st, mt, pr) = first_ext.kind.state_type_protect();
            let mut cur_end = first_ext.end;
            // Extend forward across contiguous extents with identical `(state, mem_type, protect)`.
            while let Some(&next_ext) = res.extents.get(&cur_end) {
                let (nst, nmt, npr) = next_ext.kind.state_type_protect();
                if nst == st && nmt == mt && npr == pr {
                    cur_end = next_ext.end;
                } else {
                    break;
                }
            }
            (st, mt, pr, cur_end)
        } else {
            // Gap (`MEM_RESERVE`, `MEM_PRIVATE`, `protect = 0`) up to next extent or `res_end`.
            let next_start = res
                .extents
                .range(page_base..)
                .next()
                .map(|(&k, _)| k)
                .unwrap_or(res_end);
            (MEM_RESERVE, MEM_PRIVATE, 0, next_start)
        };

        let region_size = (region_end - page_base) as u64;

        let mut flags = 0u64;
        if state == MEM_COMMIT {
            flags |= 1 << 0; // CPU-committed
        }
        if res.gpu_accessible {
            flags |= 1 << 1; // GPU-accessible
        }

        Some(XMemBasicInformation {
            base_address: page_base as u64,
            allocation_base: res.base as u64,
            allocation_protect: res.alloc_protect,
            partition_id: 0,
            _pad0: 0,
            region_size,
            state,
            protect,
            mem_type,
            _pad1: 0,
            flags,
            tag_code: 0,
            tag: [0; 4],
        })
    }

    /// `XMemTransferMemory`: adjust title memory budget (0 bytes always succeeds; direction 0
    /// transfers rounded 2 MiB increments from title to system, rejecting over-budget/under-used).
    pub fn transfer_memory(&self, direction: u32, size: usize) -> bool {
        if size == 0 {
            return true;
        }
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        let Some(rounded) = round_up_usize(size, GRANULE_2M) else {
            return false;
        };
        let delta = rounded as u64;
        let max_budget = title_memory_budget_bytes(guard.device);
        match direction {
            0 => {
                // Title -> System: reduce title limit while keeping limit >= currently used.
                let Some(new_limit) = guard.budget_limit.checked_sub(delta) else {
                    return false;
                };
                if new_limit < guard.total_used_bytes() {
                    return false;
                }
                guard.budget_limit = new_limit;
                true
            }
            1 => {
                // System -> Title: restore up to standard hardware budget.
                guard.budget_limit = guard.budget_limit.saturating_add(delta).min(max_budget);
                true
            }
            _ => false,
        }
    }

    /// `XMemGetWorkingSetStatistics`: return the common 16-byte `(title_limit, title_used)` prefix.
    pub fn working_set_statistics(&self) -> XMemWorkingSetPrefix {
        let Ok(guard) = self.inner.lock() else {
            return XMemWorkingSetPrefix::default();
        };
        XMemWorkingSetPrefix {
            title_limit_bytes: guard.budget_limit,
            title_used_bytes: guard.total_used_bytes(),
        }
    }

    /// `XMemIsToolsMemAvailable`: returns `false` (`0`) for retail console profiles.
    pub const fn is_tools_mem_available(&self) -> bool {
        false
    }
}

fn rollback_phys_alloc(guard: &mut ArenaInner, granule: PageGranule, allocated: &[(usize, usize)]) {
    if let Some(pool) = guard.pools.get_mut(&granule) {
        for &(_, slot) in allocated {
            pool.recycle_slot(slot);
        }
    }
}

fn decode_phys_granule(_flags: u64) -> PageGranule {
    // Publicly evidenced default for `XMemAllocatePhysicalPages` is 64 KiB.
    PageGranule::Size64K
}

fn decode_virtual_granule(alloc_type: u32, flags: u64) -> PageGranule {
    if (alloc_type & MEM_4MB_PAGES) != 0 || ((flags >> 14) & 0x3) == 2 {
        PageGranule::Size4M
    } else if (alloc_type & (MEM_LARGE_PAGES | MEM_64K_PAGES)) == (MEM_LARGE_PAGES | MEM_64K_PAGES)
    {
        PageGranule::Size64K
    } else if (alloc_type & MEM_LARGE_PAGES) != 0 {
        PageGranule::Size2M
    } else {
        PageGranule::Size64K
    }
}

fn round_down_usize(value: usize, align: usize) -> usize {
    if align == 0 {
        value
    } else {
        value - (value % align)
    }
}

fn round_up_usize(value: usize, align: usize) -> Option<usize> {
    if align == 0 {
        return Some(value);
    }
    let rem = value % align;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(align - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xmem_heap_alloc_alignment_and_zero_init() {
        let arena = XMemArena::new(DeviceType::SeriesX);
        // Alignment log2 = 12 (4096 B) encoded in bits 24..=28.
        let attrs = 12u64 << 24;
        let ptr = arena.alloc(128, attrs);
        assert!(!ptr.is_null());
        assert_eq!((ptr as usize) % 4096, 0);
        // SAFETY: `ptr` points to 128 zero-initialized bytes owned by this test.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, 128) };
        assert!(slice.iter().all(|&b| b == 0));
        slice[0] = 0xAB;
        slice[127] = 0xCD;
        assert_eq!(arena.working_set_statistics().title_used_bytes, 128);
        assert!(arena.free(ptr as usize, attrs));
        assert!(!arena.free(ptr as usize, attrs));
        assert_eq!(arena.working_set_statistics().title_used_bytes, 0);
    }

    #[test]
    fn physical_pages_contiguous_scatter_alias_coherency_and_virtual_query() {
        let arena = XMemArena::new(DeviceType::SeriesX);

        // Allocate 3 physical 64 KiB pages.
        let pfns = arena
            .allocate_physical_pages(0, 3, None)
            .expect("allocate 3 physical pages");
        assert_eq!(pfns.len(), 3);
        assert_eq!(
            arena.working_set_statistics().title_used_bytes,
            (3 * GRANULE_64K) as u64
        );

        // 1. Contiguous map in reversed PFN order [pfn2, pfn0].
        let va1 = arena
            .map_physical_pages(0, 2, Some(&[pfns[2], pfns[0]]))
            .expect("contiguous map");
        assert_eq!(va1 % GRANULE_64K, 0);
        assert!(va1 < CONSOLE_VA_LIMIT);

        // Write distinct markers through va1.
        // SAFETY: `va1` maps two 64 KiB RW pages backed by `memfd`.
        unsafe {
            ptr::write_volatile(va1 as *mut u32, 0x1122_3344);
            ptr::write_volatile((va1 + GRANULE_64K + 16) as *mut u32, 0xAABB_CCDD);
        }

        // 2. Reserve a 2-page VA range via XMemVirtualAlloc and scatter-map [pfn0, pfn2] (aliased!).
        let va2 = arena
            .virtual_alloc(0, 2 * GRANULE_64K, MEM_RESERVE, 0, PAGE_READWRITE)
            .expect("reserve scatter VA");
        assert!(va2 < CONSOLE_VA_LIMIT);
        assert!(arena.map_physical_pages_scatter(&[va2, va2 + GRANULE_64K], &[pfns[0], pfns[2]]));

        // Verify bidirectional kernel page-table alias coherency!
        // SAFETY: `va2` aliases `pfns[0]` at slot 0 and `pfns[2]` at slot 1.
        unsafe {
            assert_eq!(
                ptr::read_volatile((va2 + 16) as *const u32),
                0xAABB_CCDD,
                "pfn[0] alias must see va1's write"
            );
            assert_eq!(
                ptr::read_volatile((va2 + GRANULE_64K) as *const u32),
                0x1122_3344,
                "pfn[2] alias must see va1's write"
            );
            // Write back through va2 and verify immediately at va1!
            ptr::write_volatile((va2 + 32) as *mut u32, 0xDEAD_BEEF);
            assert_eq!(
                ptr::read_volatile((va1 + GRANULE_64K + 32) as *const u32),
                0xDEAD_BEEF,
                "bidirectional alias write must be visible at va1"
            );
        }

        // 3. XMemVirtualQuery on va2 returns exact 64-byte descriptor.
        let q = arena
            .virtual_query(va2 + 4096)
            .expect("virtual query inside va2");
        assert_eq!(q.base_address, (va2 + 4096) as u64);
        assert_eq!(q.allocation_base, va2 as u64);
        assert_eq!(q.region_size, (2 * GRANULE_64K - 4096) as u64);
        assert_eq!(q.state, MEM_COMMIT);
        assert_eq!(q.mem_type, MEM_MAPPED);
        assert_eq!(q.protect, PAGE_READWRITE);
        assert_eq!(q.flags, 0b11);

        // 4. Freeing PFNs while aliased keeps aliases live until unmapped/released.
        assert!(arena.free_physical_pages(&pfns));
        // SAFETY: Aliases remain valid while mapped.
        unsafe {
            assert_eq!(ptr::read_volatile((va2 + 32) as *const u32), 0xDEAD_BEEF);
        }

        assert!(arena.virtual_free(va1, 0, MEM_RELEASE));
        assert!(arena.virtual_free(va2, 0, MEM_RELEASE));
        assert_eq!(arena.working_set_statistics().title_used_bytes, 0);
    }

    #[test]
    fn reserves_24gib_4mb_aligned_low_va_and_supports_win32_commit_query_scatter_and_release() {
        let arena = XMemArena::new(DeviceType::SeriesX);
        let size_24g: usize = 24 * 1024 * 1024 * 1024;
        let base = arena
            .virtual_alloc(0, size_24g, MEM_RESERVE | MEM_4MB_PAGES, 0, PAGE_NOACCESS)
            .expect("reserve 24 GiB 4 MiB-aligned low-VA arena");

        assert!(base < CONSOLE_VA_LIMIT, "base 0x{base:x} must be < 2^44");
        assert!(
            base.checked_add(size_24g)
                .is_some_and(|end| end <= CONSOLE_VA_LIMIT),
            "entire 24 GiB reservation must stay below 2^44"
        );
        assert_eq!(base % GRANULE_4M, 0, "base must be 4 MiB aligned");
        // Reserve-only must NOT charge committed memory budget.
        assert_eq!(arena.working_set_statistics().title_used_bytes, 0);

        // Sub-range Win32 commit of 64 KiB at +4 MiB offset.
        let sub = base + GRANULE_4M;
        assert_eq!(
            arena.try_win32_virtual_alloc(sub, GRANULE_64K, MEM_COMMIT, PAGE_READWRITE),
            ArenaOpResult::Ok(sub)
        );
        assert_eq!(
            arena.working_set_statistics().title_used_bytes,
            GRANULE_64K as u64
        );

        // Write & read through committed sub-range.
        // SAFETY: `[sub, sub + 64 KiB)` was just committed as `PAGE_READWRITE`.
        unsafe {
            ptr::write_volatile(sub as *mut u64, 0xCAFE_BABE_1234_5678);
            assert_eq!(ptr::read_volatile(sub as *const u64), 0xCAFE_BABE_1234_5678);
        }

        // Query committed sub-range and surrounding reserved regions.
        let q_sub = arena.virtual_query(sub).expect("query committed sub-range");
        assert_eq!(q_sub.base_address, sub as u64);
        assert_eq!(q_sub.allocation_base, base as u64);
        assert_eq!(q_sub.region_size, GRANULE_64K as u64);
        assert_eq!(q_sub.state, MEM_COMMIT);
        assert_eq!(q_sub.mem_type, MEM_PRIVATE);
        assert_eq!(q_sub.protect, PAGE_READWRITE);

        let q_pre = arena
            .virtual_query(base)
            .expect("query leading reserved gap");
        assert_eq!(q_pre.state, MEM_RESERVE);
        assert_eq!(q_pre.region_size, GRANULE_4M as u64);

        // Scatter-map a 64 KiB physical page into an adjacent 64 KiB slot inside the 24 GiB arena.
        let pfns = arena
            .allocate_physical_pages(0, 1, None)
            .expect("allocate 1 64 KiB physical page");
        let phys_va = sub + GRANULE_64K;
        assert!(arena.map_physical_pages_scatter(&[phys_va], &pfns));
        // SAFETY: `phys_va` maps a valid 64 KiB RW memfd page.
        unsafe {
            ptr::write_volatile(phys_va as *mut u32, 0x55AA_77EE);
            assert_eq!(ptr::read_volatile(phys_va as *const u32), 0x55AA_77EE);
        }

        // Decommit the private 64 KiB sub-range, recommit it, and verify zero-fill + adjacent alias intact.
        assert_eq!(
            arena.try_win32_virtual_free(sub, GRANULE_64K, MEM_DECOMMIT),
            ArenaOpResult::Ok(())
        );
        assert_eq!(
            arena.try_win32_virtual_alloc(sub, GRANULE_64K, MEM_COMMIT, PAGE_READWRITE),
            ArenaOpResult::Ok(sub)
        );
        // SAFETY: Both `sub` (recommitted zeroed) and `phys_va` (intact alias) are mapped RW.
        unsafe {
            assert_eq!(
                ptr::read_volatile(sub as *const u64),
                0,
                "recommitted pages must be zero-filled"
            );
            assert_eq!(
                ptr::read_volatile(phys_va as *const u32),
                0x55AA_77EE,
                "adjacent physical alias must survive private decommit"
            );
        }

        // Win32 MEM_RELEASE rejects non-base address or non-zero size.
        assert_eq!(
            arena.try_win32_virtual_free(sub, 0, MEM_RELEASE),
            ArenaOpResult::Error(ERROR_INVALID_PARAMETER)
        );
        assert_eq!(
            arena.try_win32_virtual_free(base, GRANULE_64K, MEM_RELEASE),
            ArenaOpResult::Error(ERROR_INVALID_PARAMETER)
        );

        // Clean release + PFN retirement returns used memory to 0.
        assert!(arena.free_physical_pages(&pfns));
        assert_eq!(
            arena.try_win32_virtual_free(base, 0, MEM_RELEASE),
            ArenaOpResult::Ok(())
        );
        assert_eq!(arena.working_set_statistics().title_used_bytes, 0);
    }

    #[test]
    fn transfer_memory_and_tools_mem_accounting() {
        let arena = XMemArena::new(DeviceType::SeriesX);
        assert!(!arena.is_tools_mem_available());
        let initial = arena.working_set_statistics();
        assert_eq!(initial.title_limit_bytes, 13_500_000_000);
        assert!(arena.transfer_memory(0, 0));
        assert!(arena.transfer_memory(0, 4 * 1024 * 1024));
        assert_eq!(
            arena.working_set_statistics().title_limit_bytes,
            13_500_000_000 - 4 * 1024 * 1024
        );
        // Reject transferring more than remaining budget.
        assert!(!arena.transfer_memory(0, 20_000_000_000));
    }
}
