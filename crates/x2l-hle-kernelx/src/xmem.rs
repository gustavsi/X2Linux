// Win64 ABI signatures, attribute bitfields, and 64-byte `XMemVirtualQuery` / working-set
// layouts informed by Marvisak/VIctory (https://github.com/Marvisak/VIctory, commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-hle/src/modules/xmem.rs) and
// xcl-dev/xcl (https://github.com/xcl-dev/xcl, commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/kernelx.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: Linux `memfd_create` + `mmap(MAP_SHARED | MAP_FIXED)`
// physical page pools, transactional scatter/contiguous aliasing, and budget accounting.

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

// Standard Win32 memory state / type / protect constants
pub const MEM_COMMIT: u32 = 0x0000_1000;
pub const MEM_RESERVE: u32 = 0x0000_2000;
pub const MEM_DECOMMIT: u32 = 0x0000_4000;
pub const MEM_RELEASE: u32 = 0x0000_8000;
pub const MEM_FREE: u32 = 0x0001_0000;
pub const MEM_PRIVATE: u32 = 0x0002_0000;
pub const MEM_MAPPED: u32 = 0x0004_0000;

pub const PAGE_NOACCESS: u32 = 0x01;
pub const PAGE_READONLY: u32 = 0x02;
pub const PAGE_READWRITE: u32 = 0x04;
pub const PAGE_EXECUTE_READ: u32 = 0x20;
pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;

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
enum PageMappingKind {
    Reserved,
    PrivateCommitted,
    PhysicalMapped { pfn: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageSlotState {
    kind: PageMappingKind,
    protect: u32,
}

struct VirtualReservation {
    base: usize,
    size: usize,
    granule: PageGranule,
    alloc_protect: u32,
    gpu_accessible: bool,
    slots: Vec<PageSlotState>,
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
    reservations: BTreeMap<usize, VirtualReservation>,
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
        let slot_count = aligned_size / g;

        let base = if requested != 0 {
            if !requested.is_multiple_of(g) {
                return None;
            }
            if let Some(existing_key) = self.find_reservation_key(requested) {
                let existing = &self.reservations[&existing_key];
                if requested.checked_add(aligned_size)? <= existing.end() {
                    return Some(requested);
                }
                return None;
            }
            // SAFETY: `MAP_FIXED_NOREPLACE` atomically fails without overwriting any Wine mapping.
            let mapped = unsafe {
                sys::mmap(
                    requested as *mut c_void,
                    aligned_size,
                    sys::PROT_NONE,
                    sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_FIXED_NOREPLACE,
                    -1,
                    0,
                )
            };
            if mapped == sys::MAP_FAILED || (mapped as usize) != requested {
                return None;
            }
            requested
        } else {
            // Over-reserve to guarantee `g`-byte alignment, then trim leading/trailing slack.
            let over = aligned_size.checked_add(g.saturating_sub(GRANULE_4K))?;
            // SAFETY: Anonymous private `PROT_NONE` reservation chosen safely by the Linux kernel.
            let raw = unsafe {
                sys::mmap(
                    ptr::null_mut(),
                    over,
                    sys::PROT_NONE,
                    sys::MAP_PRIVATE | sys::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if raw == sys::MAP_FAILED || raw.is_null() {
                return None;
            }
            let raw_addr = raw as usize;
            let aligned_addr = round_up_usize(raw_addr, g)?;
            let lead = aligned_addr - raw_addr;
            let trail = (raw_addr + over) - (aligned_addr + aligned_size);
            if lead > 0 {
                // SAFETY: `[raw_addr, aligned_addr)` is unused leading slack from our fresh `mmap`.
                unsafe {
                    let _ = sys::munmap(raw, lead);
                }
            }
            if trail > 0 {
                // SAFETY: `[aligned_addr + aligned_size, raw_addr + over)` is unused trailing slack.
                unsafe {
                    let _ = sys::munmap((aligned_addr + aligned_size) as *mut c_void, trail);
                }
            }
            aligned_addr
        };

        self.reservations.insert(
            base,
            VirtualReservation {
                base,
                size: aligned_size,
                granule,
                alloc_protect,
                gpu_accessible,
                slots: vec![
                    PageSlotState {
                        kind: PageMappingKind::Reserved,
                        protect: PAGE_NOACCESS,
                    };
                    slot_count
                ],
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

    fn unmap_slot_to_placeholder(
        &mut self,
        res_base: usize,
        slot_idx: usize,
    ) -> Option<PageSlotState> {
        let (addr, g, old_state) = {
            let res = self.reservations.get(&res_base)?;
            let g = res.granule.bytes();
            let old = *res.slots.get(slot_idx)?;
            (res.base + slot_idx * g, g, old)
        };
        // SAFETY: `[addr, addr + g)` is inside an arena-owned reservation; replacing with
        // `MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED` `PROT_NONE` atomically drops any shared alias.
        let rc = unsafe {
            sys::mmap(
                addr as *mut c_void,
                g,
                sys::PROT_NONE,
                sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_FIXED,
                -1,
                0,
            )
        };
        if rc == sys::MAP_FAILED {
            return None;
        }
        match old_state.kind {
            PageMappingKind::PhysicalMapped { pfn } => self.decrement_pfn_alias(pfn),
            PageMappingKind::PrivateCommitted => {
                self.private_bytes_used = self.private_bytes_used.saturating_sub(g as u64);
            }
            PageMappingKind::Reserved => {}
        }
        if let Some(res) = self.reservations.get_mut(&res_base) {
            res.slots[slot_idx] = PageSlotState {
                kind: PageMappingKind::Reserved,
                protect: PAGE_NOACCESS,
            };
        }
        Some(old_state)
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
    /// aligned VA range when `base == 0`). Passing `pfns == None` unmaps back to `PROT_NONE`.
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
            let (res_base, g, total_slots) = {
                let res = &guard.reservations[&res_key];
                (res.base, res.granule.bytes(), res.slots.len())
            };
            if !(base - res_base).is_multiple_of(g) {
                return None;
            }
            let start_slot = (base - res_base) / g;
            if start_slot.checked_add(count)? > total_slots {
                return None;
            }
            for i in 0..count {
                guard.unmap_slot_to_placeholder(res_base, start_slot + i)?;
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
        let (res_base, res_granule, res_slots) = {
            let res = &guard.reservations[&res_key];
            (res.base, res.granule, res.slots.len())
        };
        if res_granule != granule || !(mapped_base - res_base).is_multiple_of(g) {
            return None;
        }
        let start_slot = (mapped_base - res_base) / g;
        if start_slot.checked_add(count)? > res_slots {
            return None;
        }

        let pool_fd = guard.pools.get(&granule)?.fd;

        // Map each slot transactionally using `MAP_SHARED | MAP_FIXED`.
        for (i, &(pfn, _, slot)) in targets.iter().enumerate() {
            let va = mapped_base + i * g;
            let slot_idx = start_slot + i;
            if pfn == 0 {
                if guard
                    .unmap_slot_to_placeholder(res_base, slot_idx)
                    .is_none()
                {
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
            // First drop any prior alias on this slot.
            let old_state = guard.reservations[&res_base].slots[slot_idx];
            // SAFETY: `[va, va + g)` is inside an arena-owned reservation and `pool_fd` covers
            // `[file_off, file_off + g)`.
            let ptr = unsafe {
                sys::mmap(
                    va as *mut c_void,
                    g,
                    sys::PROT_READ | sys::PROT_WRITE,
                    sys::MAP_SHARED | sys::MAP_FIXED,
                    pool_fd,
                    file_off,
                )
            };
            if ptr == sys::MAP_FAILED || (ptr as usize) != va {
                if created_new {
                    for prev_i in 0..i {
                        let _ = guard.unmap_slot_to_placeholder(res_base, start_slot + prev_i);
                    }
                    if let Some(res) = guard.reservations.remove(&res_base) {
                        // SAFETY: Roll back newly reserved VA range on failure.
                        unsafe {
                            let _ = sys::munmap(res.base as *mut c_void, res.size);
                        }
                    }
                }
                return None;
            }
            match old_state.kind {
                PageMappingKind::PhysicalMapped { pfn: old_pfn } => {
                    guard.decrement_pfn_alias(old_pfn);
                }
                PageMappingKind::PrivateCommitted => {
                    guard.private_bytes_used = guard.private_bytes_used.saturating_sub(g as u64);
                }
                PageMappingKind::Reserved => {}
            }
            if let Some(rec) = guard.phys_pages.get_mut(&pfn) {
                rec.aliases += 1;
            }
            if let Some(res) = guard.reservations.get_mut(&res_base) {
                res.slots[slot_idx] = PageSlotState {
                    kind: PageMappingKind::PhysicalMapped { pfn },
                    protect: PAGE_READWRITE,
                };
            }
        }

        Some(mapped_base)
    }

    /// `XMemMapPhysicalPagesScatter`: map each `pfns[i]` onto its corresponding caller-supplied
    /// reserved virtual address `addresses[i]` (with PFN `0` restoring `PROT_NONE`).
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
            slot_idx: usize,
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
            let (res_base, res_granule, slot_count) = {
                let res = &guard.reservations[&res_key];
                (res.base, res.granule, res.slots.len())
            };
            let g = res_granule.bytes();
            if !(va - res_base).is_multiple_of(g) {
                return false;
            }
            let slot_idx = (va - res_base) / g;
            if slot_idx >= slot_count {
                return false;
            }
            if pfn == 0 {
                ops.push(ScatterOp {
                    res_base,
                    slot_idx,
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
                if !rec.live || rec.granule != res_granule {
                    return false;
                }
                let Some(pool) = guard.pools.get(&res_granule) else {
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
                    slot_idx,
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
                if guard
                    .unmap_slot_to_placeholder(op.res_base, op.slot_idx)
                    .is_none()
                {
                    return false;
                }
                continue;
            }
            let old_state = guard.reservations[&op.res_base].slots[op.slot_idx];
            // SAFETY: `[op.va, op.va + op.granule_bytes)` is inside an arena-owned reservation
            // and `op.pool_fd` covers `[op.file_off, op.file_off + op.granule_bytes)`.
            let ptr = unsafe {
                sys::mmap(
                    op.va as *mut c_void,
                    op.granule_bytes,
                    sys::PROT_READ | sys::PROT_WRITE,
                    sys::MAP_SHARED | sys::MAP_FIXED,
                    op.pool_fd,
                    op.file_off,
                )
            };
            if ptr == sys::MAP_FAILED || (ptr as usize) != op.va {
                return false;
            }
            match old_state.kind {
                PageMappingKind::PhysicalMapped { pfn: old_pfn } => {
                    guard.decrement_pfn_alias(old_pfn);
                }
                PageMappingKind::PrivateCommitted => {
                    guard.private_bytes_used = guard
                        .private_bytes_used
                        .saturating_sub(op.granule_bytes as u64);
                }
                PageMappingKind::Reserved => {}
            }
            if let Some(rec) = guard.phys_pages.get_mut(&op.pfn) {
                rec.aliases += 1;
            }
            if let Some(res) = guard.reservations.get_mut(&op.res_base) {
                res.slots[op.slot_idx] = PageSlotState {
                    kind: PageMappingKind::PhysicalMapped { pfn: op.pfn },
                    protect: PAGE_READWRITE,
                };
            }
        }

        true
    }

    /// `XMemVirtualAlloc`: reserve and/or commit arena-owned virtual address space with console
    /// granule selection (default 64 KiB; `0x2040_0000` -> 64 KiB, `0x2000_0000` -> 2 MiB).
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
        let aligned_size = round_up_usize(size, g)?;
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

        let base = if is_reserve {
            guard.reserve_va_range(addr, aligned_size, granule, clean_prot, true)?
        } else {
            // Commit inside an existing arena reservation.
            let _ = guard.find_reservation_key(addr)?;
            addr
        };

        if is_commit {
            let res_key = guard.find_reservation_key(base)?;
            let (res_base, res_g, slot_count) = {
                let res = &guard.reservations[&res_key];
                (res.base, res.granule.bytes(), res.slots.len())
            };
            if !(base - res_base).is_multiple_of(res_g) {
                return None;
            }
            let start_slot = (base - res_base) / res_g;
            let num_slots = aligned_size.div_ceil(res_g);
            if start_slot.checked_add(num_slots)? > slot_count {
                return None;
            }
            let linux_prot = sys::win_protect_to_linux(clean_prot);
            // SAFETY: `[base, base + num_slots * res_g)` is inside an arena-owned reservation.
            let rc = unsafe { sys::mprotect(base as *mut c_void, num_slots * res_g, linux_prot) };
            if rc != 0 {
                return None;
            }
            let mut newly_committed = 0u64;
            if let Some(res) = guard.reservations.get_mut(&res_base) {
                for slot in &mut res.slots[start_slot..start_slot + num_slots] {
                    if slot.kind == PageMappingKind::Reserved {
                        newly_committed = newly_committed.saturating_add(res_g as u64);
                    }
                    *slot = PageSlotState {
                        kind: PageMappingKind::PrivateCommitted,
                        protect: clean_prot,
                    };
                }
            }
            guard.private_bytes_used = guard.private_bytes_used.saturating_add(newly_committed);
        }

        Some(base)
    }

    /// Release or decommit an arena-owned virtual reservation. Returns `true` if `addr` belonged
    /// to this arena and was released/decommitted.
    pub fn virtual_free(&self, addr: usize, size: usize, free_type: u32) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        let Some(res_key) = guard.find_reservation_key(addr) else {
            return false;
        };
        if (free_type & MEM_RELEASE) != 0 {
            let Some(res) = guard.reservations.remove(&res_key) else {
                return false;
            };
            let g = res.granule.bytes() as u64;
            for slot in &res.slots {
                match slot.kind {
                    PageMappingKind::PhysicalMapped { pfn } => guard.decrement_pfn_alias(pfn),
                    PageMappingKind::PrivateCommitted => {
                        guard.private_bytes_used = guard.private_bytes_used.saturating_sub(g);
                    }
                    PageMappingKind::Reserved => {}
                }
            }
            // SAFETY: `res` uniquely owns `[res.base, res.base + res.size)`.
            unsafe {
                let _ = sys::munmap(res.base as *mut c_void, res.size);
            }
            return true;
        }
        if (free_type & MEM_DECOMMIT) != 0 {
            let (res_base, g, slot_count) = {
                let res = &guard.reservations[&res_key];
                (res.base, res.granule.bytes(), res.slots.len())
            };
            let start_slot = (addr - res_base) / g;
            let count = if size == 0 {
                slot_count - start_slot
            } else {
                size.div_ceil(g).min(slot_count - start_slot)
            };
            for i in 0..count {
                let _ = guard.unmap_slot_to_placeholder(res_base, start_slot + i);
            }
            return true;
        }
        false
    }

    /// `XMemVirtualQuery`: return the 64-byte `XMemBasicInformation` describing the contiguous
    /// forward run from the 4 KiB-aligned page containing `addr` inside an arena reservation.
    pub fn virtual_query(&self, addr: usize) -> Option<XMemBasicInformation> {
        let Ok(guard) = self.inner.lock() else {
            return None;
        };
        let res_key = guard.find_reservation_key(addr)?;
        let res = &guard.reservations[&res_key];
        let g = res.granule.bytes();
        let page_base = addr & !(GRANULE_4K - 1);
        let first_slot = (addr - res.base) / g;
        let first = res.slots[first_slot];

        let state_of = |k: PageMappingKind| -> (u32, u32) {
            match k {
                PageMappingKind::Reserved => (MEM_RESERVE, MEM_PRIVATE),
                PageMappingKind::PrivateCommitted => (MEM_COMMIT, MEM_PRIVATE),
                PageMappingKind::PhysicalMapped { .. } => (MEM_COMMIT, MEM_MAPPED),
            }
        };
        let (state, mem_type) = state_of(first.kind);

        let mut end_slot = first_slot + 1;
        while end_slot < res.slots.len() {
            let s = res.slots[end_slot];
            let (st, mt) = state_of(s.kind);
            if st == state && mt == mem_type && s.protect == first.protect {
                end_slot += 1;
            } else {
                break;
            }
        }
        let region_end = res.base + end_slot * g;
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
            protect: if state == MEM_COMMIT {
                first.protect
            } else {
                0
            },
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
    const MEM_LARGE_PAGES: u32 = 0x2000_0000;
    const MEM_64K_PAGES: u32 = 0x0040_0000;
    if (alloc_type & (MEM_LARGE_PAGES | MEM_64K_PAGES)) == (MEM_LARGE_PAGES | MEM_64K_PAGES) {
        PageGranule::Size64K
    } else if (alloc_type & MEM_LARGE_PAGES) != 0 {
        PageGranule::Size2M
    } else if ((flags >> 14) & 0x3) == 2 {
        PageGranule::Size4M
    } else {
        PageGranule::Size64K
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
