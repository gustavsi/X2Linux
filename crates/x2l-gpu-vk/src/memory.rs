// Clean-room Linux 3-Tier Unified Memory Manager for X2Linux.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Linux 3-Tier Unified Memory Manager for `D3D12.X` → Vulkan 1.3/1.4:
//!
//! - **Tier 1 (`HostExternal`):** Zero-copy host-imported allocations via
//!   `VK_EXT_external_memory_host` + `VK_KHR_buffer_device_address` when advertised by the ICD
//!   and properly aligned to `minImportedHostPointerAlignment`.
//! - **Tier 2 (`UserfaultfdWp` / `ExplicitStaging`):** Unprivileged Linux `userfaultfd`
//!   write-protect dirty-page tracking (`UFFD_USER_MODE_ONLY | O_CLOEXEC | O_NONBLOCK`) with
//!   automatic negotiation of:
//!   1. `WpAsyncPagemapScan` (`UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED` +
//!      `/proc/self/pagemap` `PAGEMAP_SCAN`),
//!   2. `WpSyncFaultThread` (dedicated `poll`/`read` fault worker + `UFFDIO_WRITEPROTECT`),
//!   3. `ExplicitDirtyTracking` (4 KiB page-granularity shadow comparison fallback when a
//!      restricted kernel denies `userfaultfd` or WP features).
//! - **Tier 3 (`GpuVaRegistry`):** O(log N) `BTreeMap` interval registry translating guest GPU
//!   virtual addresses to `(vk::Buffer, offset, vk::DeviceAddress, AllocationId)`.
//!
//! All Linux syscalls route through `libc` wrapper functions (never raw inline `syscall`
//! instructions) so the memory manager remains safe under Wine's Syscall User Dispatch (SUD).

use ash::vk;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void};
use std::mem::size_of;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use thiserror::Error;
use tracing::{debug, warn};

/// Default Linux x86-64 base page size (4 KiB).
pub const PAGE_SIZE: usize = 4096;

/// Opaque identifier for an allocation tracked by [`UnifiedMemoryManager`].
pub type AllocationId = u64;

/// Errors raised by [`UnifiedMemoryManager`] and [`GpuVaRegistry`].
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("allocation size must be non-zero")]
    ZeroSize,
    #[error("integer overflow while computing address or alignment")]
    ArithmeticOverflow,
    #[error("guest virtual address range [0x{base:x}, 0x{end:x}) overlaps an existing mapping")]
    OverlappingRange { base: u64, end: u64 },
    #[error("allocation {0} was not found")]
    NotFound(AllocationId),
    #[error("Linux OS call '{op}' failed with errno {errno}")]
    Os { op: &'static str, errno: i32 },
    #[error("Vulkan call '{op}' failed with VkResult({code})")]
    Vulkan { op: &'static str, code: i32 },
    #[error("no compatible Vulkan memory type found for {purpose}")]
    NoMemoryType { purpose: &'static str },
    #[error("internal lock poisoned")]
    LockPoisoned,
}

impl MemoryError {
    fn from_vk(op: &'static str, err: vk::Result) -> Self {
        Self::Vulkan {
            op,
            code: err.as_raw(),
        }
    }
}

/// Active dirty-page tracking mode negotiated with the host Linux kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserfaultfdMode {
    /// Async write-protect via `UFFD_FEATURE_WP_ASYNC` + `/proc/self/pagemap` `PAGEMAP_SCAN`.
    WpAsyncPagemapScan,
    /// Synchronous write-protect fault thread via `UFFD_EVENT_PAGEFAULT` + `UFFDIO_WRITEPROTECT`.
    WpSyncFaultThread,
    /// Portable 4 KiB page-diff fallback when `userfaultfd` or WP is unavailable.
    ExplicitDirtyTracking,
}

impl UserfaultfdMode {
    /// Stable string tag used in `report.json` telemetry.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WpAsyncPagemapScan => "wp_async_pagemap_scan",
            Self::WpSyncFaultThread => "wp_sync_fault_thread",
            Self::ExplicitDirtyTracking => "explicit_dirty_tracking",
        }
    }
}

/// Concrete storage tier selected for an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Tier 1: zero-copy `VK_EXT_external_memory_host` + `VK_KHR_buffer_device_address`.
    HostExternal,
    /// Tier 2a: `userfaultfd` write-protect tracked guest memory + staged `vkCmdCopyBuffer`.
    UserfaultfdWp,
    /// Tier 2b: 4 KiB page-diff tracked guest memory + staged `vkCmdCopyBuffer`.
    ExplicitStaging,
}

/// Policy controlling tier selection for [`UnifiedMemoryManager::allocate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationPolicy {
    /// Prefer Tier 1 `HostExternal` when available, otherwise Tier 2 dirty-tracked staging.
    #[default]
    Auto,
    /// Prefer Tier 1 `HostExternal` (falls back to Tier 2 if ICD cannot import host memory).
    PreferHostExternal,
    /// Force Tier 2 dirty-page tracking (`UserfaultfdWp` if kernel supports it, else `ExplicitStaging`).
    ForceDirtyTracked,
    /// Force Tier 2 `ExplicitStaging` (4 KiB shadow diff without `userfaultfd`).
    ForceExplicitStaging,
}

/// Summary statistics returned by [`UnifiedMemoryManager::flush_dirty`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadStats {
    pub dirty_pages: usize,
    pub copy_regions: usize,
    pub bytes_uploaded: u64,
    pub tier: MemoryTier,
    pub mode: UserfaultfdMode,
}

/// Host memory capabilities negotiated during initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryCapabilities {
    pub external_memory_host: bool,
    pub min_imported_host_pointer_alignment: u64,
    pub buffer_device_address: bool,
    pub userfaultfd_mode: UserfaultfdMode,
}

/// Interval metadata stored in [`GpuVaRegistry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuVaEntry {
    pub end: u64,
    pub buffer: vk::Buffer,
    pub device_address: vk::DeviceAddress,
    pub allocation_id: AllocationId,
}

/// Result of looking up a guest GPU virtual address in [`GpuVaRegistry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuVaResolution {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub device_address: vk::DeviceAddress,
    pub allocation_id: AllocationId,
}

/// Tier 3 O(log N) interval registry translating guest GPU virtual addresses to host Vulkan buffers.
#[derive(Debug, Default)]
pub struct GpuVaRegistry {
    ranges: BTreeMap<u64, GpuVaEntry>,
}

impl GpuVaRegistry {
    /// Create an empty GPU virtual-address registry.
    pub fn new() -> Self {
        Self {
            ranges: BTreeMap::new(),
        }
    }

    /// Number of active mapped intervals.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the registry has no mapped intervals.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Register `[base, base + len)` with strict non-overlap and overflow checking.
    pub fn register(
        &mut self,
        base: u64,
        len: u64,
        buffer: vk::Buffer,
        device_address: vk::DeviceAddress,
        allocation_id: AllocationId,
    ) -> Result<(), MemoryError> {
        if len == 0 {
            return Err(MemoryError::ZeroSize);
        }
        let end = base
            .checked_add(len)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        if device_address != 0 {
            device_address
                .checked_add(len)
                .ok_or(MemoryError::ArithmeticOverflow)?;
        }

        if let Some((&_pred_base, pred)) = self.ranges.range(..=base).next_back()
            && pred.end > base
        {
            return Err(MemoryError::OverlappingRange { base, end });
        }
        if let Some((&succ_base, _)) = self.ranges.range(base..).next()
            && succ_base < end
        {
            return Err(MemoryError::OverlappingRange { base, end });
        }

        self.ranges.insert(
            base,
            GpuVaEntry {
                end,
                buffer,
                device_address,
                allocation_id,
            },
        );
        Ok(())
    }

    /// Remove the mapping rooted at `base` owned by `allocation_id`.
    pub fn unregister(
        &mut self,
        base: u64,
        allocation_id: AllocationId,
    ) -> Result<(), MemoryError> {
        match self.ranges.get(&base) {
            Some(entry) if entry.allocation_id == allocation_id => {
                self.ranges.remove(&base);
                Ok(())
            }
            _ => Err(MemoryError::NotFound(allocation_id)),
        }
    }

    /// Resolve a guest GPU virtual address in O(log N) time.
    pub fn lookup(&self, guest_va: u64) -> Option<GpuVaResolution> {
        let (&base, entry) = self.ranges.range(..=guest_va).next_back()?;
        if guest_va >= entry.end {
            return None;
        }
        let offset = guest_va - base;
        let device_address = if entry.device_address != 0 {
            entry.device_address.checked_add(offset)?
        } else {
            0
        };
        Some(GpuVaResolution {
            buffer: entry.buffer,
            offset,
            device_address,
            allocation_id: entry.allocation_id,
        })
    }
}

/// Coalesce dirty 4 KiB page indices into minimal contiguous [`vk::BufferCopy`] spans
/// clamped to `logical_len`.
pub fn coalesce_dirty_pages(
    dirty_pages: &[usize],
    logical_len: usize,
    page_size: usize,
) -> Vec<vk::BufferCopy> {
    if dirty_pages.is_empty() || logical_len == 0 || page_size == 0 {
        return Vec::new();
    }
    let mut sorted = dirty_pages.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut regions = Vec::new();
    let mut run_start = sorted[0];
    let mut run_end = run_start + 1;

    let push_run = |regions: &mut Vec<vk::BufferCopy>, first: usize, end: usize| {
        let byte_start = first.saturating_mul(page_size);
        if byte_start >= logical_len {
            return;
        }
        let byte_end = end.saturating_mul(page_size).min(logical_len);
        let size = (byte_end - byte_start) as u64;
        if size > 0 {
            regions.push(vk::BufferCopy {
                src_offset: byte_start as u64,
                dst_offset: byte_start as u64,
                size,
            });
        }
    };

    for &page in &sorted[1..] {
        if page == run_end {
            run_end += 1;
        } else {
            push_run(&mut regions, run_start, run_end);
            run_start = page;
            run_end = page + 1;
        }
    }
    push_run(&mut regions, run_start, run_end);
    regions
}

// ============================================================================
// Linux x86-64 libc wrappers & userfaultfd / PAGEMAP_SCAN ABI
// ============================================================================

#[allow(dead_code)]
mod sys {
    use super::*;

    pub const SYS_USERFAULTFD: c_long = 323;

    pub const O_RDONLY: c_int = 0;
    pub const O_NONBLOCK: c_int = 0x800;
    pub const O_CLOEXEC: c_int = 0x80000;
    pub const EFD_NONBLOCK: c_int = 0x800;
    pub const EFD_CLOEXEC: c_int = 0x80000;

    pub const EINTR: c_int = 4;
    pub const EAGAIN: c_int = 11;

    pub const PROT_READ: c_int = 0x1;
    pub const PROT_WRITE: c_int = 0x2;
    pub const MAP_PRIVATE: c_int = 0x02;
    pub const MAP_ANONYMOUS: c_int = 0x20;
    pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
    pub const MADV_NOHUGEPAGE: c_int = 15;
    pub const SC_PAGESIZE: c_int = 30;
    pub const POLLIN: i16 = 0x0001;

    pub const UFFD_API: u64 = 0xAA;
    pub const UFFD_USER_MODE_ONLY: c_int = 1;
    pub const UFFD_FEATURE_PAGEFAULT_FLAG_WP: u64 = 1 << 0;
    pub const UFFD_FEATURE_WP_UNPOPULATED: u64 = 1 << 13;
    pub const UFFD_FEATURE_WP_ASYNC: u64 = 1 << 15;

    pub const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
    pub const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1 << 0;
    pub const UFFDIO_WRITEPROTECT_BIT: u64 = 1 << 6;

    pub const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
    pub const UFFD_PAGEFAULT_FLAG_WP: u64 = 1 << 1;

    pub const UFFDIO_API: c_ulong = 0xc018_aa3f;
    pub const UFFDIO_REGISTER: c_ulong = 0xc020_aa00;
    pub const UFFDIO_UNREGISTER: c_ulong = 0x8010_aa01;
    pub const UFFDIO_WRITEPROTECT: c_ulong = 0xc018_aa06;

    pub const PAGEMAP_SCAN: c_ulong = 0xc060_6610;
    pub const PM_SCAN_WP_MATCHING: u64 = 1 << 0;
    pub const PM_SCAN_CHECK_WPASYNC: u64 = 1 << 1;
    pub const PAGE_IS_WRITTEN: u64 = 1 << 1;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct UffdioApi {
        pub api: u64,
        pub features: u64,
        pub ioctls: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct UffdioRange {
        pub start: u64,
        pub len: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct UffdioRegister {
        pub range: UffdioRange,
        pub mode: u64,
        pub ioctls: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct UffdioWriteprotect {
        pub range: UffdioRange,
        pub mode: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct PmScanArg {
        pub size: u64,
        pub flags: u64,
        pub start: u64,
        pub end: u64,
        pub walk_end: u64,
        pub vec: u64,
        pub vec_len: u64,
        pub max_pages: u64,
        pub category_inverted: u64,
        pub category_mask: u64,
        pub category_anyof_mask: u64,
        pub return_mask: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct PageRegion {
        pub start: u64,
        pub end: u64,
        pub categories: u64,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, Default)]
    pub struct PollFd {
        pub fd: c_int,
        pub events: i16,
        pub revents: i16,
    }

    // Compile-time Linux x86-64 ABI size assertions.
    const _: () = assert!(size_of::<UffdioApi>() == 24);
    const _: () = assert!(size_of::<UffdioRange>() == 16);
    const _: () = assert!(size_of::<UffdioRegister>() == 32);
    const _: () = assert!(size_of::<UffdioWriteprotect>() == 24);
    const _: () = assert!(size_of::<PmScanArg>() == 96);
    const _: () = assert!(size_of::<PageRegion>() == 24);

    unsafe extern "C" {
        pub fn syscall(number: c_long, ...) -> c_long;
        pub fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
        pub fn open(path: *const c_char, flags: c_int, ...) -> c_int;
        pub fn close(fd: c_int) -> c_int;
        pub fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
        pub fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
        pub fn mmap(
            addr: *mut c_void,
            length: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: isize,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, length: usize) -> c_int;
        pub fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
        pub fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
        pub fn eventfd(initval: c_uint, flags: c_int) -> c_int;
        pub fn sysconf(name: c_int) -> c_long;
        pub fn __errno_location() -> *mut c_int;
    }

    pub fn last_errno() -> i32 {
        // SAFETY: `__errno_location` returns a valid thread-local `c_int*` in glibc/musl.
        unsafe { *__errno_location() }
    }

    pub fn host_page_size() -> usize {
        // SAFETY: `sysconf(_SC_PAGESIZE)` is a read-only POSIX query.
        let val = unsafe { sysconf(SC_PAGESIZE) };
        if val > 0 { val as usize } else { PAGE_SIZE }
    }

    pub fn open_userfaultfd() -> Result<c_int, i32> {
        let flags: c_long = (UFFD_USER_MODE_ONLY | O_CLOEXEC | O_NONBLOCK) as c_long;
        // SAFETY: Routes through libc's `syscall` wrapper (Wine SUD-safe) with valid flags.
        let fd = unsafe { syscall(SYS_USERFAULTFD, flags) };
        if fd < 0 {
            Err(last_errno())
        } else {
            Ok(fd as c_int)
        }
    }

    pub fn uffd_api(fd: c_int, features: u64) -> Result<UffdioApi, i32> {
        let mut api = UffdioApi {
            api: UFFD_API,
            features,
            ioctls: 0,
        };
        // SAFETY: `api` is an initialized `#[repr(C)]` `uffdio_api` buffer owned for this ioctl.
        let rc = unsafe { ioctl(fd, UFFDIO_API, &mut api as *mut UffdioApi) };
        if rc < 0 { Err(last_errno()) } else { Ok(api) }
    }

    pub fn uffd_register_wp(fd: c_int, start: u64, len: u64) -> Result<u64, i32> {
        let mut reg = UffdioRegister {
            range: UffdioRange { start, len },
            mode: UFFDIO_REGISTER_MODE_WP,
            ioctls: 0,
        };
        // SAFETY: `reg` is an initialized `uffdio_register` for a page-aligned live mapping.
        let rc = unsafe { ioctl(fd, UFFDIO_REGISTER, &mut reg as *mut UffdioRegister) };
        if rc < 0 {
            Err(last_errno())
        } else {
            Ok(reg.ioctls)
        }
    }

    pub fn uffd_unregister(fd: c_int, start: u64, len: u64) -> Result<(), i32> {
        let mut range = UffdioRange { start, len };
        // SAFETY: `range` is an initialized `uffdio_range` previously registered on `fd`.
        let rc = unsafe { ioctl(fd, UFFDIO_UNREGISTER, &mut range as *mut UffdioRange) };
        if rc < 0 { Err(last_errno()) } else { Ok(()) }
    }

    pub fn uffd_writeprotect(fd: c_int, start: u64, len: u64, enable: bool) -> Result<(), i32> {
        let mut wp = UffdioWriteprotect {
            range: UffdioRange { start, len },
            mode: if enable {
                UFFDIO_WRITEPROTECT_MODE_WP
            } else {
                0
            },
        };
        // SAFETY: `wp` is an initialized `uffdio_writeprotect` over a WP-registered range.
        let rc = unsafe { ioctl(fd, UFFDIO_WRITEPROTECT, &mut wp as *mut UffdioWriteprotect) };
        if rc < 0 { Err(last_errno()) } else { Ok(()) }
    }

    pub fn open_self_pagemap() -> Result<c_int, i32> {
        const PATH: &[u8] = b"/proc/self/pagemap\0";
        // SAFETY: `PATH` is a static NUL-terminated ASCII string.
        let fd = unsafe { open(PATH.as_ptr().cast::<c_char>(), O_RDONLY | O_CLOEXEC) };
        if fd < 0 { Err(last_errno()) } else { Ok(fd) }
    }

    pub fn pagemap_scan_wp_async(
        pagemap_fd: c_int,
        start: u64,
        end: u64,
        out_pages: &mut Vec<usize>,
        base: u64,
        page_size: usize,
    ) -> Result<(), i32> {
        let mut cursor = start;
        let mut regions = [PageRegion::default(); 64];

        while cursor < end {
            let mut arg = PmScanArg {
                size: size_of::<PmScanArg>() as u64,
                flags: PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC,
                start: cursor,
                end,
                walk_end: 0,
                vec: regions.as_mut_ptr() as u64,
                vec_len: regions.len() as u64,
                max_pages: 0,
                category_inverted: 0,
                category_mask: PAGE_IS_WRITTEN,
                category_anyof_mask: 0,
                return_mask: PAGE_IS_WRITTEN,
            };
            // SAFETY: `arg` and `regions` are live, initialized ABI buffers; `regions.len()` bounds writes.
            let count = unsafe { ioctl(pagemap_fd, PAGEMAP_SCAN, &mut arg as *mut PmScanArg) };
            if count < 0 {
                return Err(last_errno());
            }
            let n = (count as usize).min(regions.len());
            for region in &regions[..n] {
                if region.start < base || region.end <= region.start || region.end > end {
                    continue;
                }
                let first_page = ((region.start - base) as usize) / page_size;
                let end_page = ((region.end - base) as usize) / page_size;
                for p in first_page..end_page {
                    out_pages.push(p);
                }
            }
            if arg.walk_end <= cursor {
                break;
            }
            cursor = arg.walk_end;
        }
        Ok(())
    }
}

/// Owned anonymous Linux `mmap` allocation supporting arbitrary power-of-two alignment.
struct OwnedMmap {
    raw_ptr: NonNull<u8>,
    raw_len: usize,
    aligned_ptr: NonNull<u8>,
    usable_len: usize,
}

// SAFETY: `OwnedMmap` owns a private anonymous virtual mapping exclusively until dropped.
unsafe impl Send for OwnedMmap {}

impl OwnedMmap {
    fn new(usable_len: usize, alignment: usize) -> Result<Self, MemoryError> {
        if usable_len == 0 {
            return Err(MemoryError::ZeroSize);
        }
        let page_size = sys::host_page_size();
        let align = alignment.max(page_size);
        let extra = align.saturating_sub(page_size);
        let raw_len = usable_len
            .checked_add(extra)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        if raw_len > isize::MAX as usize {
            return Err(MemoryError::ArithmeticOverflow);
        }

        // SAFETY: Anonymous private RW mapping of `raw_len` bytes via libc `mmap`.
        let ptr = unsafe {
            sys::mmap(
                ptr::null_mut(),
                raw_len,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_PRIVATE | sys::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == sys::MAP_FAILED || ptr.is_null() {
            return Err(MemoryError::Os {
                op: "mmap",
                errno: sys::last_errno(),
            });
        }

        // Hint base 4 KiB pages so WP granularity stays 4 KiB.
        // SAFETY: `[ptr, ptr + raw_len)` is a live mapping returned by `mmap`.
        let _ = unsafe { sys::madvise(ptr, raw_len, sys::MADV_NOHUGEPAGE) };

        let raw_addr = ptr as usize;
        let rem = raw_addr % align;
        let aligned_addr = if rem == 0 {
            raw_addr
        } else {
            raw_addr
                .checked_add(align - rem)
                .ok_or(MemoryError::ArithmeticOverflow)?
        };
        debug_assert!(aligned_addr + usable_len <= raw_addr + raw_len);

        let raw_ptr = NonNull::new(ptr.cast::<u8>()).ok_or(MemoryError::ArithmeticOverflow)?;
        let aligned_ptr =
            NonNull::new(aligned_addr as *mut u8).ok_or(MemoryError::ArithmeticOverflow)?;

        Ok(Self {
            raw_ptr,
            raw_len,
            aligned_ptr,
            usable_len,
        })
    }

    fn as_ptr(&self) -> NonNull<u8> {
        self.aligned_ptr
    }

    fn as_addr(&self) -> u64 {
        self.aligned_ptr.as_ptr() as usize as u64
    }

    fn usable_len(&self) -> usize {
        self.usable_len
    }
}

impl Drop for OwnedMmap {
    fn drop(&mut self) {
        // SAFETY: `raw_ptr` and `raw_len` are the exact live arguments from `mmap` and no
        // Vulkan external memory or UFFD registration remains active at drop time.
        let _ = unsafe { sys::munmap(self.raw_ptr.as_ptr().cast::<c_void>(), self.raw_len) };
    }
}

/// Owned POSIX file descriptor closed automatically on drop.
struct OwnedFd(c_int);

impl OwnedFd {
    fn raw(&self) -> c_int {
        self.0
    }
}

impl Drop for OwnedFd {
    fn drop(&mut self) {
        if self.0 >= 0 {
            // SAFETY: `OwnedFd` uniquely owns this open file descriptor.
            let _ = unsafe { sys::close(self.0) };
            self.0 = -1;
        }
    }
}

/// Probe the host Linux kernel for the best supported `userfaultfd` write-protect tracking mode.
///
/// Always succeeds and degrades cleanly:
/// `WpAsyncPagemapScan` -> `WpSyncFaultThread` -> `ExplicitDirtyTracking`.
pub fn probe_userfaultfd_mode() -> UserfaultfdMode {
    let page_size = sys::host_page_size();
    let probe_fd = match sys::open_userfaultfd() {
        Ok(fd) => OwnedFd(fd),
        Err(errno) => {
            debug!(
                errno,
                "userfaultfd(UFFD_USER_MODE_ONLY) unavailable; using explicit dirty tracking"
            );
            return UserfaultfdMode::ExplicitDirtyTracking;
        }
    };

    let api = match sys::uffd_api(probe_fd.raw(), 0) {
        Ok(api) => api,
        Err(errno) => {
            debug!(
                errno,
                "UFFDIO_API probe failed; using explicit dirty tracking"
            );
            return UserfaultfdMode::ExplicitDirtyTracking;
        }
    };
    drop(probe_fd);

    let async_bits = sys::UFFD_FEATURE_WP_ASYNC
        | sys::UFFD_FEATURE_WP_UNPOPULATED
        | sys::UFFD_FEATURE_PAGEFAULT_FLAG_WP;
    if (api.features & async_bits) == async_bits && verify_wp_async_cycle(page_size) {
        return UserfaultfdMode::WpAsyncPagemapScan;
    }

    if (api.features & sys::UFFD_FEATURE_PAGEFAULT_FLAG_WP) != 0 {
        return UserfaultfdMode::WpSyncFaultThread;
    }

    UserfaultfdMode::ExplicitDirtyTracking
}

fn verify_wp_async_cycle(page_size: usize) -> bool {
    let Ok(uffd_raw) = sys::open_userfaultfd() else {
        return false;
    };
    let uffd = OwnedFd(uffd_raw);
    if sys::uffd_api(
        uffd.raw(),
        sys::UFFD_FEATURE_WP_ASYNC | sys::UFFD_FEATURE_WP_UNPOPULATED,
    )
    .is_err()
    {
        return false;
    }
    let Ok(pagemap_raw) = sys::open_self_pagemap() else {
        return false;
    };
    let pagemap = OwnedFd(pagemap_raw);

    let Ok(scratch) = OwnedMmap::new(page_size * 2, page_size) else {
        return false;
    };
    let base = scratch.as_addr();
    let len = scratch.usable_len() as u64;

    let Ok(ioctls) = sys::uffd_register_wp(uffd.raw(), base, len) else {
        return false;
    };
    if (ioctls & sys::UFFDIO_WRITEPROTECT_BIT) == 0 {
        let _ = sys::uffd_unregister(uffd.raw(), base, len);
        return false;
    }

    let ok = (|| -> bool {
        if sys::uffd_writeprotect(uffd.raw(), base, len, true).is_err() {
            return false;
        }
        // SAFETY: `scratch` owns 2 writable anonymous pages; writing byte 0 triggers WP-async.
        unsafe {
            ptr::write_volatile(scratch.as_ptr().as_ptr(), 0xA5);
        }
        let mut dirty = Vec::new();
        if sys::pagemap_scan_wp_async(pagemap.raw(), base, base + len, &mut dirty, base, page_size)
            .is_err()
        {
            return false;
        }
        if dirty != [0] {
            return false;
        }
        // Immediate rescan must return 0 dirty pages because PM_SCAN_WP_MATCHING re-armed WP.
        dirty.clear();
        if sys::pagemap_scan_wp_async(pagemap.raw(), base, base + len, &mut dirty, base, page_size)
            .is_err()
        {
            return false;
        }
        dirty.is_empty()
    })();

    let _ = sys::uffd_unregister(uffd.raw(), base, len);
    ok
}

/// Standalone dirty-page tracker owning a page-aligned anonymous mapping and its tracking backend.
pub struct DirtyTracker {
    mode: UserfaultfdMode,
    page_size: usize,
    logical_len: usize,
    mapping: OwnedMmap,
    snapshot: Vec<u8>,
    backend: TrackerBackend,
}

enum TrackerBackend {
    WpAsync {
        uffd: OwnedFd,
        pagemap: OwnedFd,
        registered: bool,
    },
    WpSync {
        uffd: OwnedFd,
        wake_efd: OwnedFd,
        dirty_pages: Arc<Mutex<Vec<usize>>>,
        worker: Option<JoinHandle<()>>,
        registered: bool,
    },
    Explicit,
}

impl DirtyTracker {
    /// Allocate a `logical_len`-byte tracked guest buffer using the requested mode (degrading
    /// safely to `ExplicitDirtyTracking` if kernel registration fails).
    pub fn new(logical_len: usize, requested_mode: UserfaultfdMode) -> Result<Self, MemoryError> {
        if logical_len == 0 {
            return Err(MemoryError::ZeroSize);
        }
        let page_size = sys::host_page_size();
        let mapped_len = round_up_usize(logical_len, page_size)?;
        let mapping = OwnedMmap::new(mapped_len, page_size)?;
        let snapshot = vec![0u8; logical_len];

        let (mode, backend) = match requested_mode {
            UserfaultfdMode::WpAsyncPagemapScan => {
                match Self::init_wp_async(mapping.as_addr(), mapped_len as u64) {
                    Ok(b) => (UserfaultfdMode::WpAsyncPagemapScan, b),
                    Err(e) => {
                        warn!(
                            error = %e,
                            "WpAsyncPagemapScan registration failed; falling back to ExplicitDirtyTracking"
                        );
                        (
                            UserfaultfdMode::ExplicitDirtyTracking,
                            TrackerBackend::Explicit,
                        )
                    }
                }
            }
            UserfaultfdMode::WpSyncFaultThread => match Self::init_wp_sync(&mapping, page_size) {
                Ok(b) => (UserfaultfdMode::WpSyncFaultThread, b),
                Err(e) => {
                    warn!(
                        error = %e,
                        "WpSyncFaultThread registration failed; falling back to ExplicitDirtyTracking"
                    );
                    (
                        UserfaultfdMode::ExplicitDirtyTracking,
                        TrackerBackend::Explicit,
                    )
                }
            },
            UserfaultfdMode::ExplicitDirtyTracking => (
                UserfaultfdMode::ExplicitDirtyTracking,
                TrackerBackend::Explicit,
            ),
        };

        Ok(Self {
            mode,
            page_size,
            logical_len,
            mapping,
            snapshot,
            backend,
        })
    }

    fn init_wp_async(base: u64, len: u64) -> Result<TrackerBackend, MemoryError> {
        let uffd = OwnedFd(sys::open_userfaultfd().map_err(|errno| MemoryError::Os {
            op: "userfaultfd",
            errno,
        })?);
        sys::uffd_api(
            uffd.raw(),
            sys::UFFD_FEATURE_WP_ASYNC | sys::UFFD_FEATURE_WP_UNPOPULATED,
        )
        .map_err(|errno| MemoryError::Os {
            op: "UFFDIO_API(WP_ASYNC)",
            errno,
        })?;
        let pagemap = OwnedFd(sys::open_self_pagemap().map_err(|errno| MemoryError::Os {
            op: "open(/proc/self/pagemap)",
            errno,
        })?);
        let ioctls =
            sys::uffd_register_wp(uffd.raw(), base, len).map_err(|errno| MemoryError::Os {
                op: "UFFDIO_REGISTER(WP)",
                errno,
            })?;
        if (ioctls & sys::UFFDIO_WRITEPROTECT_BIT) == 0 {
            let _ = sys::uffd_unregister(uffd.raw(), base, len);
            return Err(MemoryError::Os {
                op: "UFFDIO_REGISTER missing WP ioctl",
                errno: 95,
            });
        }
        if let Err(errno) = sys::uffd_writeprotect(uffd.raw(), base, len, true) {
            let _ = sys::uffd_unregister(uffd.raw(), base, len);
            return Err(MemoryError::Os {
                op: "UFFDIO_WRITEPROTECT(arm)",
                errno,
            });
        }
        Ok(TrackerBackend::WpAsync {
            uffd,
            pagemap,
            registered: true,
        })
    }

    fn init_wp_sync(mapping: &OwnedMmap, page_size: usize) -> Result<TrackerBackend, MemoryError> {
        let base = mapping.as_addr();
        let len = mapping.usable_len() as u64;

        let uffd = OwnedFd(sys::open_userfaultfd().map_err(|errno| MemoryError::Os {
            op: "userfaultfd",
            errno,
        })?);
        let api =
            sys::uffd_api(uffd.raw(), sys::UFFD_FEATURE_PAGEFAULT_FLAG_WP).map_err(|errno| {
                MemoryError::Os {
                    op: "UFFDIO_API(WP_SYNC)",
                    errno,
                }
            })?;

        // Prefault zero pages so synchronous WP works even without WP_UNPOPULATED.
        if (api.features & sys::UFFD_FEATURE_WP_UNPOPULATED) == 0 {
            let mut offset = 0usize;
            while offset < mapping.usable_len() {
                // SAFETY: `offset < mapping.usable_len()` is within the owned RW anonymous mapping.
                unsafe {
                    ptr::write_volatile(mapping.as_ptr().as_ptr().add(offset), 0);
                }
                offset += page_size;
            }
        }

        // SAFETY: `eventfd` via libc creates an owned shutdown-signal fd.
        let efd_raw = unsafe { sys::eventfd(0, sys::EFD_CLOEXEC | sys::EFD_NONBLOCK) };
        if efd_raw < 0 {
            return Err(MemoryError::Os {
                op: "eventfd",
                errno: sys::last_errno(),
            });
        }
        let wake_efd = OwnedFd(efd_raw);

        let ioctls =
            sys::uffd_register_wp(uffd.raw(), base, len).map_err(|errno| MemoryError::Os {
                op: "UFFDIO_REGISTER(WP_SYNC)",
                errno,
            })?;
        if (ioctls & sys::UFFDIO_WRITEPROTECT_BIT) == 0 {
            let _ = sys::uffd_unregister(uffd.raw(), base, len);
            return Err(MemoryError::Os {
                op: "UFFDIO_REGISTER missing WP ioctl",
                errno: 95,
            });
        }
        if let Err(errno) = sys::uffd_writeprotect(uffd.raw(), base, len, true) {
            let _ = sys::uffd_unregister(uffd.raw(), base, len);
            return Err(MemoryError::Os {
                op: "UFFDIO_WRITEPROTECT(arm)",
                errno,
            });
        }

        let dirty_pages = Arc::new(Mutex::new(Vec::<usize>::new()));
        let dirty_clone = Arc::clone(&dirty_pages);
        let uffd_fd = uffd.raw();
        let wake_fd = wake_efd.raw();

        let worker = thread::Builder::new()
            .name("x2l-uffd-wp".into())
            .spawn(move || {
                sync_wp_worker(uffd_fd, wake_fd, base, len, page_size, dirty_clone);
            })
            .map_err(|_| MemoryError::Os {
                op: "thread::spawn(uffd-wp)",
                errno: sys::last_errno(),
            })?;

        Ok(TrackerBackend::WpSync {
            uffd,
            wake_efd,
            dirty_pages,
            worker: Some(worker),
            registered: true,
        })
    }

    /// Active tracking mode for this buffer.
    pub fn mode(&self) -> UserfaultfdMode {
        self.mode
    }

    /// Logical byte length of the tracked buffer.
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    /// Raw host pointer into the tracked anonymous mapping.
    pub fn as_ptr(&self) -> NonNull<u8> {
        self.mapping.as_ptr()
    }

    /// Host page size used by this tracker.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Detect modified 4 KiB page indices since the last committed snapshot.
    ///
    /// # Safety
    /// Caller must ensure no concurrent guest writes are racing with this scan.
    pub unsafe fn collect_dirty_pages(&mut self) -> Result<(Vec<usize>, Vec<usize>), MemoryError> {
        let base = self.mapping.as_addr();
        let mapped_len = self.mapping.usable_len() as u64;
        let page_size = self.page_size;

        let mut candidates = Vec::new();
        let mut fallback_full_scan = false;

        match &mut self.backend {
            TrackerBackend::WpAsync {
                pagemap,
                registered,
                ..
            } if *registered => {
                if let Err(errno) = sys::pagemap_scan_wp_async(
                    pagemap.raw(),
                    base,
                    base + mapped_len,
                    &mut candidates,
                    base,
                    page_size,
                ) {
                    warn!(
                        errno,
                        "PAGEMAP_SCAN failed; retiring WP registration and scanning snapshot"
                    );
                    *registered = false;
                    self.mode = UserfaultfdMode::ExplicitDirtyTracking;
                    fallback_full_scan = true;
                }
            }
            TrackerBackend::WpSync {
                dirty_pages,
                registered,
                ..
            } if *registered => {
                let mut guard = dirty_pages.lock().map_err(|_| MemoryError::LockPoisoned)?;
                candidates.append(&mut *guard);
            }
            _ => {
                fallback_full_scan = true;
            }
        }

        // SAFETY: Caller guarantees guest writers are quiescent; `[as_ptr, logical_len)` is owned.
        let live =
            unsafe { std::slice::from_raw_parts(self.mapping.as_ptr().as_ptr(), self.logical_len) };

        let mut changed = Vec::new();
        if fallback_full_scan {
            let num_pages = self.logical_len.div_ceil(page_size);
            for p in 0..num_pages {
                let start = p * page_size;
                let end = (start + page_size).min(self.logical_len);
                if live[start..end] != self.snapshot[start..end] {
                    changed.push(p);
                }
            }
            return Ok((changed, Vec::new()));
        }

        candidates.sort_unstable();
        candidates.dedup();
        for &p in &candidates {
            let start = p.saturating_mul(page_size);
            if start >= self.logical_len {
                continue;
            }
            let end = (start + page_size).min(self.logical_len);
            if live[start..end] != self.snapshot[start..end] {
                changed.push(p);
            }
        }

        Ok((changed, candidates))
    }

    /// Commit synced dirty pages into the shadow snapshot and re-arm sync WP pages.
    ///
    /// # Safety
    /// Caller must ensure guest writers are quiescent while copying synced pages to the snapshot.
    pub unsafe fn commit_synced(
        &mut self,
        dirty_pages: &[usize],
        sync_rearm_candidates: &[usize],
    ) -> Result<(), MemoryError> {
        let page_size = self.page_size;
        // SAFETY: Caller guarantees quiescence; mapping owns `logical_len` live bytes.
        let live =
            unsafe { std::slice::from_raw_parts(self.mapping.as_ptr().as_ptr(), self.logical_len) };
        for &p in dirty_pages {
            let start = p.saturating_mul(page_size);
            if start >= self.logical_len {
                continue;
            }
            let end = (start + page_size).min(self.logical_len);
            self.snapshot[start..end].copy_from_slice(&live[start..end]);
        }

        if let TrackerBackend::WpSync {
            uffd, registered, ..
        } = &mut self.backend
            && *registered
            && !sync_rearm_candidates.is_empty()
        {
            let base = self.mapping.as_addr();
            for region in
                coalesce_dirty_pages(sync_rearm_candidates, self.mapping.usable_len(), page_size)
            {
                let addr = base + region.dst_offset;
                let len = round_up_u64(region.size, page_size as u64)?;
                if let Err(errno) = sys::uffd_writeprotect(uffd.raw(), addr, len, true) {
                    warn!(
                        errno,
                        "UFFDIO_WRITEPROTECT re-arm failed; disabling sync WP"
                    );
                    *registered = false;
                    self.mode = UserfaultfdMode::ExplicitDirtyTracking;
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for DirtyTracker {
    fn drop(&mut self) {
        let base = self.mapping.as_addr();
        let len = self.mapping.usable_len() as u64;
        match &mut self.backend {
            TrackerBackend::WpAsync {
                uffd, registered, ..
            } => {
                if *registered {
                    let _ = sys::uffd_unregister(uffd.raw(), base, len);
                    *registered = false;
                }
            }
            TrackerBackend::WpSync {
                uffd,
                wake_efd,
                worker,
                registered,
                ..
            } => {
                let signal: u64 = 1;
                // SAFETY: Writing an 8-byte counter to an owned `eventfd` wakes the worker.
                let _ = unsafe {
                    sys::write(
                        wake_efd.raw(),
                        (&signal as *const u64).cast::<c_void>(),
                        size_of::<u64>(),
                    )
                };
                if let Some(handle) = worker.take() {
                    let _ = handle.join();
                }
                if *registered {
                    let _ = sys::uffd_unregister(uffd.raw(), base, len);
                    *registered = false;
                }
            }
            TrackerBackend::Explicit => {}
        }
        // `self.mapping` drops AFTER `uffd_unregister` and worker join.
    }
}

fn sync_wp_worker(
    uffd_fd: c_int,
    wake_fd: c_int,
    base: u64,
    len: u64,
    page_size: usize,
    dirty_pages: Arc<Mutex<Vec<usize>>>,
) {
    let mut fds = [
        sys::PollFd {
            fd: uffd_fd,
            events: sys::POLLIN,
            revents: 0,
        },
        sys::PollFd {
            fd: wake_fd,
            events: sys::POLLIN,
            revents: 0,
        },
    ];
    let mut msg = [0u8; 32];

    loop {
        fds[0].revents = 0;
        fds[1].revents = 0;
        // SAFETY: `fds` is a valid 2-element `pollfd` array referencing live owned fds.
        let rc = unsafe { sys::poll(fds.as_mut_ptr(), 2, -1) };
        if rc < 0 {
            if sys::last_errno() == sys::EINTR {
                continue;
            }
            break;
        }
        if (fds[1].revents & sys::POLLIN) != 0 {
            break;
        }
        if (fds[0].revents & sys::POLLIN) != 0 {
            loop {
                // SAFETY: `msg` is a 32-byte buffer matching `sizeof(struct uffd_msg)`.
                let n = unsafe { sys::read(uffd_fd, msg.as_mut_ptr().cast::<c_void>(), msg.len()) };
                if n < 0 {
                    let err = sys::last_errno();
                    if err == sys::EINTR {
                        continue;
                    }
                    if err == sys::EAGAIN {
                        break;
                    }
                    return;
                }
                if (n as usize) < 32 {
                    break;
                }
                let event = msg[0];
                if event == sys::UFFD_EVENT_PAGEFAULT {
                    let flags = u64::from_ne_bytes(msg[8..16].try_into().unwrap_or([0; 8]));
                    let addr = u64::from_ne_bytes(msg[16..24].try_into().unwrap_or([0; 8]));
                    if (flags & sys::UFFD_PAGEFAULT_FLAG_WP) != 0
                        && addr >= base
                        && addr < base + len
                    {
                        let page_addr = addr & !((page_size as u64) - 1);
                        let page_idx = ((page_addr - base) as usize) / page_size;
                        if let Ok(mut guard) = dirty_pages.lock() {
                            guard.push(page_idx);
                        }
                        let _ = sys::uffd_writeprotect(uffd_fd, page_addr, page_size as u64, false);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Vulkan 3-Tier UnifiedMemoryManager
// ============================================================================

struct AllocationRecord {
    guest_base: u64,
    logical_len: usize,
    tier: MemoryTier,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
    backing: AllocationBacking,
}

enum AllocationBacking {
    HostExternal {
        memory: vk::DeviceMemory,
        mapping: OwnedMmap,
    },
    Staged {
        dst_memory: vk::DeviceMemory,
        staging_buffer: vk::Buffer,
        staging_memory: vk::DeviceMemory,
        staging_mapped: NonNull<u8>,
        staging_coherent: bool,
        tracker: DirtyTracker,
        has_prior_gpu_use: bool,
    },
}

// SAFETY: All Vulkan handles and raw pointers inside `AllocationRecord` are exclusively owned
// by `UnifiedMemoryManager` and accessed under `&mut self` or external synchronization.
unsafe impl Send for AllocationRecord {}

/// Linux 3-Tier Unified Memory Manager backed by Vulkan 1.3/1.4 (`ash`).
pub struct UnifiedMemoryManager {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    external_host_ext: Option<ash::ext::external_memory_host::Device>,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    non_coherent_atom_size: u64,
    caps: HostMemoryCapabilities,
    registry: GpuVaRegistry,
    allocations: BTreeMap<AllocationId, AllocationRecord>,
    next_id: AllocationId,
    next_auto_va: u64,
}

impl UnifiedMemoryManager {
    /// Construct a new [`UnifiedMemoryManager`] bound to an active Vulkan 1.3/1.4 device & queue.
    ///
    /// # Safety
    /// `instance`, `physical_device`, `device`, and `queue` must be valid, live Vulkan handles
    /// created with Vulkan 1.3 (`synchronization2`) and `bufferDeviceAddress` if `buffer_device_address`
    /// is true. This manager must be dropped before `device` is destroyed.
    pub unsafe fn new(
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        queue: vk::Queue,
        queue_family_index: u32,
        external_memory_host_enabled: bool,
        buffer_device_address_enabled: bool,
    ) -> Result<Self, MemoryError> {
        // SAFETY: `physical_device` is a live handle owned by `instance`.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let mut ext_host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default();
        if external_memory_host_enabled {
            props2 = props2.push_next(&mut ext_host_props);
        }
        // SAFETY: `props2` and `ext_host_props` remain alive on the stack for the duration of the call.
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };

        let non_coherent_atom_size = props2.properties.limits.non_coherent_atom_size.max(1);
        let min_align = if external_memory_host_enabled {
            ext_host_props.min_imported_host_pointer_alignment.max(1)
        } else {
            0
        };
        let external_host_ext = if external_memory_host_enabled {
            Some(ash::ext::external_memory_host::Device::new(
                &instance, &device,
            ))
        } else {
            None
        };

        let userfaultfd_mode = probe_userfaultfd_mode();
        let caps = HostMemoryCapabilities {
            external_memory_host: external_memory_host_enabled && min_align > 0,
            min_imported_host_pointer_alignment: min_align,
            buffer_device_address: buffer_device_address_enabled,
            userfaultfd_mode,
        };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        // SAFETY: `device` is live and `queue_family_index` belongs to `physical_device`.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| MemoryError::from_vk("create_command_pool", e))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `command_pool` was just created on `device`.
        let command_buffer = match unsafe { device.allocate_command_buffers(&alloc_info) } {
            Ok(list) => list[0],
            Err(e) => {
                // SAFETY: `command_pool` is unused and owned locally on error.
                unsafe { device.destroy_command_pool(command_pool, None) };
                return Err(MemoryError::from_vk("allocate_command_buffers", e));
            }
        };

        let fence_info = vk::FenceCreateInfo::default();
        // SAFETY: `device` is live.
        let fence = match unsafe { device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(e) => {
                // SAFETY: `command_pool` owns `command_buffer` and neither has been submitted.
                unsafe { device.destroy_command_pool(command_pool, None) };
                return Err(MemoryError::from_vk("create_fence", e));
            }
        };

        Ok(Self {
            instance,
            physical_device,
            device,
            queue,
            command_pool,
            command_buffer,
            fence,
            external_host_ext,
            mem_props,
            non_coherent_atom_size,
            caps,
            registry: GpuVaRegistry::new(),
            allocations: BTreeMap::new(),
            next_id: 1,
            next_auto_va: 0x0000_0001_0000_0000,
        })
    }

    /// Return the negotiated Vulkan + Linux memory capabilities.
    pub fn capabilities(&self) -> HostMemoryCapabilities {
        self.caps
    }

    /// Borrow the Tier 3 GPU VA interval registry.
    pub fn registry(&self) -> &GpuVaRegistry {
        &self.registry
    }

    /// Number of currently live allocations.
    pub fn live_allocations(&self) -> usize {
        self.allocations.len()
    }

    /// Reserve a fresh non-overlapping synthetic guest GPU virtual address aligned to 64 KiB.
    pub fn reserve_guest_va(&mut self, len: u64) -> Result<u64, MemoryError> {
        let aligned_len = round_up_u64(len.max(1), 65536)?;
        let base = self.next_auto_va;
        self.next_auto_va = base
            .checked_add(aligned_len)
            .ok_or(MemoryError::ArithmeticOverflow)?;
        Ok(base)
    }

    /// Allocate a `len`-byte guest-accessible GPU buffer mapped at `guest_base`.
    pub fn allocate(
        &mut self,
        guest_base: u64,
        len: usize,
        policy: AllocationPolicy,
    ) -> Result<AllocationId, MemoryError> {
        if len == 0 {
            return Err(MemoryError::ZeroSize);
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(MemoryError::ArithmeticOverflow)?;

        let try_host_external = matches!(
            policy,
            AllocationPolicy::Auto | AllocationPolicy::PreferHostExternal
        ) && self.caps.external_memory_host;

        let record = if try_host_external {
            match self.try_allocate_host_external(id, guest_base, len) {
                Ok(Some(rec)) => rec,
                Ok(None) => {
                    self.allocate_staged(id, guest_base, len, self.caps.userfaultfd_mode)?
                }
                Err(e) => {
                    warn!(error = %e, "HostExternal allocation failed; falling back to staged Tier 2");
                    self.allocate_staged(id, guest_base, len, self.caps.userfaultfd_mode)?
                }
            }
        } else {
            let mode = match policy {
                AllocationPolicy::ForceExplicitStaging => UserfaultfdMode::ExplicitDirtyTracking,
                _ => self.caps.userfaultfd_mode,
            };
            self.allocate_staged(id, guest_base, len, mode)?
        };

        if let Err(err) = self.registry.register(
            guest_base,
            len as u64,
            record.buffer,
            record.device_address,
            id,
        ) {
            self.destroy_record(record);
            return Err(err);
        }

        self.allocations.insert(id, record);
        Ok(id)
    }

    /// Return the storage tier chosen for `id`.
    pub fn allocation_tier(&self, id: AllocationId) -> Result<MemoryTier, MemoryError> {
        self.allocations
            .get(&id)
            .map(|r| r.tier)
            .ok_or(MemoryError::NotFound(id))
    }

    /// Return the guest GPU virtual address base for `id`.
    pub fn guest_base(&self, id: AllocationId) -> Result<u64, MemoryError> {
        self.allocations
            .get(&id)
            .map(|r| r.guest_base)
            .ok_or(MemoryError::NotFound(id))
    }

    /// Return the logical byte length of `id`.
    pub fn allocation_len(&self, id: AllocationId) -> Result<usize, MemoryError> {
        self.allocations
            .get(&id)
            .map(|r| r.logical_len)
            .ok_or(MemoryError::NotFound(id))
    }

    /// Return the underlying Vulkan buffer and BDA for `id`.
    pub fn allocation_vulkan(
        &self,
        id: AllocationId,
    ) -> Result<(vk::Buffer, vk::DeviceAddress), MemoryError> {
        let rec = self.allocations.get(&id).ok_or(MemoryError::NotFound(id))?;
        Ok((rec.buffer, rec.device_address))
    }

    /// Synchronize GPU writes to `id` back into the guest CPU mapping (`Map()` coherence).
    ///
    /// No-op for `HostExternal` zero-copy allocations (already coherent with host memory).
    /// For `Staged` allocations, reads back the GPU buffer and copies into the tracked guest
    /// buffer while refreshing the dirty-page snapshot.
    pub fn sync_gpu_to_guest(&mut self, id: AllocationId) -> Result<(), MemoryError> {
        let is_staged = matches!(
            self.allocations
                .get(&id)
                .ok_or(MemoryError::NotFound(id))?
                .backing,
            AllocationBacking::Staged { .. }
        );
        if !is_staged {
            return Ok(());
        }
        let data = self.readback_buffer(id)?;
        let Some(rec) = self.allocations.get_mut(&id) else {
            return Err(MemoryError::NotFound(id));
        };
        if let AllocationBacking::Staged { tracker, .. } = &mut rec.backing {
            let len = data.len().min(rec.logical_len);
            // SAFETY: Guest mapping is valid for `logical_len` bytes; we write GPU readback
            // bytes and immediately commit them to the snapshot so `flush_dirty` won't treat
            // GPU-written pages as dirty CPU writes.
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), tracker.as_ptr().as_ptr(), len);
                let (dirty, candidates) = tracker.collect_dirty_pages()?;
                tracker.commit_synced(&dirty, &candidates)?;
            }
        }
        Ok(())
    }

    /// Return the writable guest CPU pointer and logical byte length for `id`.
    ///
    /// # Safety
    /// The returned pointer remains valid only until `free_allocation(id)` or dropping the
    /// manager. Callers must quiesce guest writes before calling [`Self::flush_dirty`].
    pub unsafe fn guest_ptr(&self, id: AllocationId) -> Result<(NonNull<u8>, usize), MemoryError> {
        let rec = self.allocations.get(&id).ok_or(MemoryError::NotFound(id))?;
        let ptr = match &rec.backing {
            AllocationBacking::HostExternal { mapping, .. } => mapping.as_ptr(),
            AllocationBacking::Staged { tracker, .. } => tracker.as_ptr(),
        };
        Ok((ptr, rec.logical_len))
    }

    /// Synchronize dirty 4 KiB pages for `id` into its GPU buffer via batched `vkCmdCopyBuffer`.
    ///
    /// # Safety
    /// Caller must ensure guest writers to `id` are quiescent for the duration of this call.
    pub unsafe fn flush_dirty(&mut self, id: AllocationId) -> Result<UploadStats, MemoryError> {
        let Some(rec) = self.allocations.get_mut(&id) else {
            return Err(MemoryError::NotFound(id));
        };

        let logical_len = rec.logical_len;
        let dst_buffer = rec.buffer;
        let tier = rec.tier;

        match &mut rec.backing {
            AllocationBacking::HostExternal { .. } => Ok(UploadStats {
                dirty_pages: 0,
                copy_regions: 0,
                bytes_uploaded: 0,
                tier,
                mode: self.caps.userfaultfd_mode,
            }),
            AllocationBacking::Staged {
                staging_buffer,
                staging_memory,
                staging_mapped,
                staging_coherent,
                tracker,
                has_prior_gpu_use,
                ..
            } => {
                // SAFETY: Caller guarantees guest writers are quiescent.
                let (dirty_pages, candidates) = unsafe { tracker.collect_dirty_pages()? };
                let active_mode = tracker.mode();
                let regions = coalesce_dirty_pages(&dirty_pages, logical_len, tracker.page_size());

                if regions.is_empty() {
                    if !candidates.is_empty() {
                        // SAFETY: Caller guarantees quiescence; re-arm sync WP pages whose contents
                        // matched the snapshot so future writes still fault.
                        unsafe { tracker.commit_synced(&[], &candidates)? };
                    }
                    return Ok(UploadStats {
                        dirty_pages: 0,
                        copy_regions: 0,
                        bytes_uploaded: 0,
                        tier,
                        mode: active_mode,
                    });
                }

                let mut bytes_uploaded = 0u64;
                for region in &regions {
                    let off = region.src_offset as usize;
                    let sz = region.size as usize;
                    // SAFETY: `off + sz <= logical_len` by `coalesce_dirty_pages`; both buffers
                    // are non-overlapping owned allocations of at least `logical_len` bytes.
                    unsafe {
                        ptr::copy_nonoverlapping(
                            tracker.as_ptr().as_ptr().add(off),
                            staging_mapped.as_ptr().add(off),
                            sz,
                        );
                    }
                    bytes_uploaded += region.size;
                }

                if !*staging_coherent {
                    let atom = self.non_coherent_atom_size;
                    let flush_ranges: Vec<vk::MappedMemoryRange<'_>> = regions
                        .iter()
                        .map(|r| {
                            let start = (r.src_offset / atom) * atom;
                            let end_raw = r.src_offset + r.size;
                            let end = end_raw.div_ceil(atom) * atom;
                            vk::MappedMemoryRange::default()
                                .memory(*staging_memory)
                                .offset(start)
                                .size(end - start)
                        })
                        .collect();
                    // SAFETY: `staging_memory` is persistently mapped and ranges are atom-aligned.
                    unsafe { self.device.flush_mapped_memory_ranges(&flush_ranges) }
                        .map_err(|e| MemoryError::from_vk("flush_mapped_memory_ranges", e))?;
                }

                let staging_buf = *staging_buffer;
                let prior = *has_prior_gpu_use;
                submit_buffer_copies(
                    &self.device,
                    self.queue,
                    self.command_buffer,
                    self.fence,
                    staging_buf,
                    dst_buffer,
                    &regions,
                    prior,
                )?;
                *has_prior_gpu_use = true;

                // SAFETY: Upload completed (fence signaled); commit snapshot and re-arm WP.
                unsafe { tracker.commit_synced(&dirty_pages, &candidates)? };

                Ok(UploadStats {
                    dirty_pages: dirty_pages.len(),
                    copy_regions: regions.len(),
                    bytes_uploaded,
                    tier,
                    mode: active_mode,
                })
            }
        }
    }

    /// Read back the current GPU contents of `id` into a newly allocated host `Vec<u8>`.
    ///
    /// Useful for headless readback and GPU verification tests.
    pub fn readback_buffer(&mut self, id: AllocationId) -> Result<Vec<u8>, MemoryError> {
        let rec = self.allocations.get(&id).ok_or(MemoryError::NotFound(id))?;
        let len = rec.logical_len;
        let src_buffer = rec.buffer;

        let (rb_buf, rb_mem, rb_ptr, rb_coherent) = create_host_staging_buffer(
            &self.device,
            &self.mem_props,
            len as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;

        let copy_result = (|| -> Result<Vec<u8>, MemoryError> {
            let region = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: len as u64,
            }];
            submit_buffer_copies(
                &self.device,
                self.queue,
                self.command_buffer,
                self.fence,
                src_buffer,
                rb_buf,
                &region,
                true,
            )?;

            if !rb_coherent {
                let range = [vk::MappedMemoryRange::default()
                    .memory(rb_mem)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)];
                // SAFETY: `rb_mem` is mapped and idle after fence wait.
                unsafe { self.device.invalidate_mapped_memory_ranges(&range) }
                    .map_err(|e| MemoryError::from_vk("invalidate_mapped_memory_ranges", e))?;
            }

            let mut out = vec![0u8; len];
            // SAFETY: GPU copy completed; `rb_ptr` is mapped for at least `len` bytes.
            unsafe {
                ptr::copy_nonoverlapping(rb_ptr.as_ptr(), out.as_mut_ptr(), len);
            }
            Ok(out)
        })();

        // SAFETY: Fence wait finished (or failed before reuse); unmap, destroy buffer, free memory.
        unsafe {
            self.device.unmap_memory(rb_mem);
            self.device.destroy_buffer(rb_buf, None);
            self.device.free_memory(rb_mem, None);
        }

        copy_result
    }

    /// Unregister and destroy allocation `id` cleanly in strict Vulkan + Linux teardown order.
    ///
    /// # Safety
    /// Caller must ensure no live guest pointers or unretired command buffers still reference `id`.
    pub unsafe fn free_allocation(&mut self, id: AllocationId) -> Result<(), MemoryError> {
        let rec = self
            .allocations
            .remove(&id)
            .ok_or(MemoryError::NotFound(id))?;
        let _ = self.registry.unregister(rec.guest_base, id);
        // SAFETY: Ensure GPU is idle before destroying Vulkan handles and unmapping Linux pages.
        let _ = unsafe { self.device.queue_wait_idle(self.queue) };
        self.destroy_record(rec);
        Ok(())
    }

    fn buffer_usage_flags(&self) -> vk::BufferUsageFlags {
        let mut usage = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        if self.caps.buffer_device_address {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
        usage
    }

    fn try_allocate_host_external(
        &self,
        _id: AllocationId,
        guest_base: u64,
        len: usize,
    ) -> Result<Option<AllocationRecord>, MemoryError> {
        let Some(ext) = &self.external_host_ext else {
            return Ok(None);
        };
        let usage = self.buffer_usage_flags();
        let ext_info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut ext_props = vk::ExternalBufferProperties::default();
        // SAFETY: `physical_device` is live; `ext_info` and `ext_props` are stack-owned.
        unsafe {
            self.instance
                .get_physical_device_external_buffer_properties(
                    self.physical_device,
                    &ext_info,
                    &mut ext_props,
                );
        }
        let features = ext_props
            .external_memory_properties
            .external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        {
            return Ok(None);
        }

        let mut ext_buf_ci = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let buf_ci = vk::BufferCreateInfo::default()
            .size(len as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_buf_ci);
        // SAFETY: `ext_buf_ci` outlives `create_buffer`.
        let buffer = unsafe { self.device.create_buffer(&buf_ci, None) }
            .map_err(|e| MemoryError::from_vk("create_buffer(host_external)", e))?;

        let req_info = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mem_req = {
            let mut req2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
            // SAFETY: `buffer` is live and `dedicated` stays alive across the call.
            unsafe {
                self.device
                    .get_buffer_memory_requirements2(&req_info, &mut req2);
            }
            req2.memory_requirements
        };
        if dedicated.requires_dedicated_allocation == vk::TRUE {
            // SAFETY: `buffer` is unused.
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Ok(None);
        }

        let page_size = sys::host_page_size();
        let min_align = self.caps.min_imported_host_pointer_alignment as usize;
        let align = lcm_usize(page_size, min_align)?;
        let alloc_size = round_up_usize(mem_req.size as usize, align)?;

        let mapping = match OwnedMmap::new(alloc_size, align) {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: `buffer` is unused.
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };

        let mut host_props = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: `mapping.as_ptr()` is aligned to `minImportedHostPointerAlignment` and live.
        let res = unsafe {
            (ext.fp().get_memory_host_pointer_properties_ext)(
                self.device.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                mapping.as_ptr().as_ptr().cast::<c_void>(),
                &mut host_props,
            )
        };
        if res != vk::Result::SUCCESS {
            // SAFETY: `buffer` is unused; `mapping` drops cleanly afterwards.
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Ok(None);
        }

        let compatible_bits = host_props.memory_type_bits & mem_req.memory_type_bits;
        let Some(mem_type_index) = find_memory_type(
            &self.mem_props,
            compatible_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            // SAFETY: `buffer` is unused.
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Ok(None);
        };

        let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(mapping.as_ptr().as_ptr().cast::<c_void>());
        let mut alloc_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(if self.caps.buffer_device_address {
                vk::MemoryAllocateFlags::DEVICE_ADDRESS
            } else {
                vk::MemoryAllocateFlags::empty()
            });
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(alloc_size as u64)
            .memory_type_index(mem_type_index)
            .push_next(&mut import_info)
            .push_next(&mut alloc_flags);

        // SAFETY: `import_info`, `alloc_flags`, and `mapping` remain live; `alloc_size` is aligned.
        let memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(mem) => mem,
            Err(_) => {
                // SAFETY: `buffer` is unused.
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Ok(None);
            }
        };

        // SAFETY: `buffer` and `memory` are live and offset 0 meets alignment requirements.
        if let Err(e) = unsafe { self.device.bind_buffer_memory(buffer, memory, 0) } {
            // SAFETY: Tear down in strict order: destroy buffer -> free memory -> drop mapping.
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            return Err(MemoryError::from_vk("bind_buffer_memory(host_external)", e));
        }

        let device_address = query_bda(&self.device, buffer, self.caps.buffer_device_address);

        Ok(Some(AllocationRecord {
            guest_base,
            logical_len: len,
            tier: MemoryTier::HostExternal,
            buffer,
            device_address,
            backing: AllocationBacking::HostExternal { memory, mapping },
        }))
    }

    fn allocate_staged(
        &self,
        _id: AllocationId,
        guest_base: u64,
        len: usize,
        requested_mode: UserfaultfdMode,
    ) -> Result<AllocationRecord, MemoryError> {
        let tracker = DirtyTracker::new(len, requested_mode)?;
        let tier = match tracker.mode() {
            UserfaultfdMode::ExplicitDirtyTracking => MemoryTier::ExplicitStaging,
            UserfaultfdMode::WpAsyncPagemapScan | UserfaultfdMode::WpSyncFaultThread => {
                MemoryTier::UserfaultfdWp
            }
        };

        let usage = self.buffer_usage_flags();
        let buf_ci = vk::BufferCreateInfo::default()
            .size(len as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `device` is live.
        let dst_buffer = unsafe { self.device.create_buffer(&buf_ci, None) }
            .map_err(|e| MemoryError::from_vk("create_buffer(staged_dst)", e))?;

        // SAFETY: `dst_buffer` is live.
        let req = unsafe { self.device.get_buffer_memory_requirements(dst_buffer) };
        let mem_type = find_memory_type(
            &self.mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type(
                &self.mem_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or(MemoryError::NoMemoryType {
            purpose: "staged destination buffer",
        })?;

        let mut alloc_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(if self.caps.buffer_device_address {
                vk::MemoryAllocateFlags::DEVICE_ADDRESS
            } else {
                vk::MemoryAllocateFlags::empty()
            });
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type)
            .push_next(&mut alloc_flags);
        // SAFETY: `alloc_flags` outlives `allocate_memory`.
        let dst_memory = match unsafe { self.device.allocate_memory(&alloc_info, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: `dst_buffer` is unused.
                unsafe { self.device.destroy_buffer(dst_buffer, None) };
                return Err(MemoryError::from_vk("allocate_memory(staged_dst)", e));
            }
        };
        // SAFETY: `dst_buffer` and `dst_memory` are live.
        if let Err(e) = unsafe { self.device.bind_buffer_memory(dst_buffer, dst_memory, 0) } {
            // SAFETY: Unused handles torn down cleanly.
            unsafe {
                self.device.destroy_buffer(dst_buffer, None);
                self.device.free_memory(dst_memory, None);
            }
            return Err(MemoryError::from_vk("bind_buffer_memory(staged_dst)", e));
        }

        let (staging_buffer, staging_memory, staging_mapped, staging_coherent) =
            match create_host_staging_buffer(
                &self.device,
                &self.mem_props,
                len as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
            ) {
                Ok(s) => s,
                Err(e) => {
                    // SAFETY: Unused handles torn down cleanly.
                    unsafe {
                        self.device.destroy_buffer(dst_buffer, None);
                        self.device.free_memory(dst_memory, None);
                    }
                    return Err(e);
                }
            };

        // Zero-initialize the destination once before exposing it so unmodified pages match the
        // zeroed anonymous guest mapping.
        // SAFETY: `staging_mapped` owns `len` bytes of host-visible staging memory.
        unsafe { ptr::write_bytes(staging_mapped.as_ptr(), 0, len) };
        if !staging_coherent {
            let range = [vk::MappedMemoryRange::default()
                .memory(staging_memory)
                .offset(0)
                .size(vk::WHOLE_SIZE)];
            // SAFETY: `staging_memory` is mapped.
            let _ = unsafe { self.device.flush_mapped_memory_ranges(&range) };
        }
        let init_region = [vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: len as u64,
        }];
        if let Err(e) = submit_buffer_copies(
            &self.device,
            self.queue,
            self.command_buffer,
            self.fence,
            staging_buffer,
            dst_buffer,
            &init_region,
            false,
        ) {
            // SAFETY: All created Vulkan handles are idle and torn down in order.
            unsafe {
                self.device.unmap_memory(staging_memory);
                self.device.destroy_buffer(staging_buffer, None);
                self.device.free_memory(staging_memory, None);
                self.device.destroy_buffer(dst_buffer, None);
                self.device.free_memory(dst_memory, None);
            }
            return Err(e);
        }

        let device_address = query_bda(&self.device, dst_buffer, self.caps.buffer_device_address);

        Ok(AllocationRecord {
            guest_base,
            logical_len: len,
            tier,
            buffer: dst_buffer,
            device_address,
            backing: AllocationBacking::Staged {
                dst_memory,
                staging_buffer,
                staging_memory,
                staging_mapped,
                staging_coherent,
                tracker,
                has_prior_gpu_use: true,
            },
        })
    }

    fn destroy_record(&self, rec: AllocationRecord) {
        match rec.backing {
            AllocationBacking::HostExternal { memory, mapping } => {
                // SAFETY: GPU work using `rec.buffer` has completed; destroy buffer and free
                // imported memory BEFORE dropping `mapping` (`munmap`).
                unsafe {
                    self.device.destroy_buffer(rec.buffer, None);
                    self.device.free_memory(memory, None);
                }
                drop(mapping);
            }
            AllocationBacking::Staged {
                dst_memory,
                staging_buffer,
                staging_memory,
                tracker,
                ..
            } => {
                // SAFETY: GPU work completed; unmap staging, destroy buffers, free device memory,
                // then drop `tracker` (which stops worker, unregisters UFFD, and unmaps pages).
                unsafe {
                    self.device.unmap_memory(staging_memory);
                    self.device.destroy_buffer(staging_buffer, None);
                    self.device.free_memory(staging_memory, None);
                    self.device.destroy_buffer(rec.buffer, None);
                    self.device.free_memory(dst_memory, None);
                }
                drop(tracker);
            }
        }
    }
}

impl Drop for UnifiedMemoryManager {
    fn drop(&mut self) {
        // SAFETY: Wait for all submitted copies to retire before destroying Vulkan handles.
        let _ = unsafe { self.device.queue_wait_idle(self.queue) };
        let records = std::mem::take(&mut self.allocations);
        for (id, rec) in records {
            let _ = self.registry.unregister(rec.guest_base, id);
            self.destroy_record(rec);
        }
        // SAFETY: All allocations and submitted commands are retired.
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

fn create_host_staging_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, NonNull<u8>, bool), MemoryError> {
    let ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `device` is live.
    let buffer = unsafe { device.create_buffer(&ci, None) }
        .map_err(|e| MemoryError::from_vk("create_buffer(staging)", e))?;
    // SAFETY: `buffer` is live.
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };

    let (mem_type, coherent) = if let Some(idx) = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ) {
        (idx, true)
    } else if let Some(idx) = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
    ) {
        (idx, false)
    } else {
        // SAFETY: `buffer` is unused.
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(MemoryError::NoMemoryType {
            purpose: "host-visible staging buffer",
        });
    };

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    // SAFETY: `device` is live.
    let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(m) => m,
        Err(e) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(MemoryError::from_vk("allocate_memory(staging)", e));
        }
    };
    // SAFETY: `buffer` and `memory` are live.
    if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        // SAFETY: Unused handles torn down cleanly.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(MemoryError::from_vk("bind_buffer_memory(staging)", e));
    }

    // SAFETY: `memory` is `HOST_VISIBLE` and unmapped.
    let mapped = match unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) } {
        Ok(ptr) => ptr,
        Err(e) => {
            // SAFETY: Unused handles torn down cleanly.
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(MemoryError::from_vk("map_memory(staging)", e));
        }
    };
    let non_null = NonNull::new(mapped.cast::<u8>()).ok_or(MemoryError::ArithmeticOverflow)?;
    Ok((buffer, memory, non_null, coherent))
}

#[allow(clippy::too_many_arguments)]
fn submit_buffer_copies(
    device: &ash::Device,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    regions: &[vk::BufferCopy],
    prior_gpu_use: bool,
) -> Result<(), MemoryError> {
    if regions.is_empty() {
        return Ok(());
    }
    // SAFETY: `cmd` and `fence` are exclusively owned by `UnifiedMemoryManager` and idle.
    unsafe {
        device
            .reset_fences(&[fence])
            .map_err(|e| MemoryError::from_vk("reset_fences", e))?;
        device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            .map_err(|e| MemoryError::from_vk("reset_command_buffer", e))?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(|e| MemoryError::from_vk("begin_command_buffer", e))?;

        if prior_gpu_use {
            let pre_barriers: Vec<vk::BufferMemoryBarrier2<'_>> = regions
                .iter()
                .map(|r| {
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(
                            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(dst_buffer)
                        .offset(r.dst_offset)
                        .size(r.size)
                })
                .collect();
            let dep = vk::DependencyInfo::default().buffer_memory_barriers(&pre_barriers);
            device.cmd_pipeline_barrier2(cmd, &dep);
        }

        device.cmd_copy_buffer(cmd, src_buffer, dst_buffer, regions);

        let post_barriers: Vec<vk::BufferMemoryBarrier2<'_>> = regions
            .iter()
            .map(|r| {
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(dst_buffer)
                    .offset(r.dst_offset)
                    .size(r.size)
            })
            .collect();
        let dep = vk::DependencyInfo::default().buffer_memory_barriers(&post_barriers);
        device.cmd_pipeline_barrier2(cmd, &dep);

        device
            .end_command_buffer(cmd)
            .map_err(|e| MemoryError::from_vk("end_command_buffer", e))?;

        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submits = [vk::SubmitInfo2::default().command_buffer_infos(&cmd_infos)];
        device
            .queue_submit2(queue, &submits, fence)
            .map_err(|e| MemoryError::from_vk("queue_submit2", e))?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| MemoryError::from_vk("wait_for_fences", e))?;
    }
    Ok(())
}

fn query_bda(device: &ash::Device, buffer: vk::Buffer, enabled: bool) -> vk::DeviceAddress {
    if !enabled {
        return 0;
    }
    let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: `buffer` was created with `SHADER_DEVICE_ADDRESS` and bound to `DEVICE_ADDRESS` memory.
    unsafe { device.get_buffer_device_address(&info) }
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1u32 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}

fn round_up_usize(value: usize, align: usize) -> Result<usize, MemoryError> {
    if align == 0 {
        return Ok(value);
    }
    let rem = value % align;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(align - rem)
            .ok_or(MemoryError::ArithmeticOverflow)
    }
}

fn round_up_u64(value: u64, align: u64) -> Result<u64, MemoryError> {
    if align == 0 {
        return Ok(value);
    }
    let rem = value % align;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(align - rem)
            .ok_or(MemoryError::ArithmeticOverflow)
    }
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

fn lcm_usize(a: usize, b: usize) -> Result<usize, MemoryError> {
    let g = gcd_usize(a, b);
    (a / g)
        .checked_mul(b)
        .ok_or(MemoryError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_va_registry_lookup_and_overlap_guards() {
        let mut reg = GpuVaRegistry::new();
        let buf_a = vk::Buffer::null();
        reg.register(0x1000_0000, 0x4000, buf_a, 0xABCD_0000, 1)
            .expect("register A");
        reg.register(0x1000_4000, 0x2000, buf_a, 0xABCD_8000, 2)
            .expect("register adjacent B");

        // Reject overlaps at start, middle, and enclosing span.
        assert!(reg.register(0x1000_3000, 0x1000, buf_a, 0, 3).is_err());
        assert!(reg.register(0x0FFF_F000, 0x2000, buf_a, 0, 4).is_err());
        assert!(reg.register(0x1000_0000, 0, buf_a, 0, 5).is_err());

        let r0 = reg.lookup(0x1000_0000).expect("lookup start");
        assert_eq!(r0.offset, 0);
        assert_eq!(r0.device_address, 0xABCD_0000);
        assert_eq!(r0.allocation_id, 1);

        let r1 = reg.lookup(0x1000_1234).expect("lookup interior");
        assert_eq!(r1.offset, 0x1234);
        assert_eq!(r1.device_address, 0xABCD_1234);

        let r2 = reg.lookup(0x1000_4000).expect("lookup B start");
        assert_eq!(r2.allocation_id, 2);
        assert_eq!(r2.offset, 0);
        assert_eq!(r2.device_address, 0xABCD_8000);

        assert!(reg.lookup(0x0FFF_FFFF).is_none());
        assert!(reg.lookup(0x1000_6000).is_none());

        reg.unregister(0x1000_0000, 1).expect("unregister A");
        assert!(reg.lookup(0x1000_1234).is_none());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn coalesces_adjacent_dirty_pages_and_clamps_tail() {
        let regions = coalesce_dirty_pages(&[5, 2, 1, 2, 6], 6 * 4096 + 512, 4096);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].src_offset, 4096);
        assert_eq!(regions[0].dst_offset, 4096);
        assert_eq!(regions[0].size, 8192);
        assert_eq!(regions[1].src_offset, 5 * 4096);
        assert_eq!(regions[1].dst_offset, 5 * 4096);
        assert_eq!(regions[1].size, 4096 + 512);
    }

    #[test]
    fn dirty_tracker_detects_writes_and_skips_unmodified_pages() {
        let mode = probe_userfaultfd_mode();
        let mut tracker = DirtyTracker::new(4 * PAGE_SIZE, mode).expect("create tracker");

        // Fresh buffer must have zero dirty pages.
        // SAFETY: Single-threaded test with no concurrent writers.
        let (dirty0, cand0) = unsafe { tracker.collect_dirty_pages().expect("scan 0") };
        assert!(dirty0.is_empty());
        // SAFETY: Single-threaded test.
        unsafe { tracker.commit_synced(&dirty0, &cand0).expect("commit 0") };

        // Write pages 1 and 2 only.
        // SAFETY: Offsets lie within the 4-page allocation.
        unsafe {
            ptr::write_volatile(tracker.as_ptr().as_ptr().add(PAGE_SIZE + 16), 0x42);
            ptr::write_volatile(tracker.as_ptr().as_ptr().add(2 * PAGE_SIZE + 32), 0x99);
        }
        // SAFETY: Writers quiescent.
        let (dirty1, cand1) = unsafe { tracker.collect_dirty_pages().expect("scan 1") };
        assert_eq!(dirty1, vec![1, 2]);
        // SAFETY: Writers quiescent.
        unsafe { tracker.commit_synced(&dirty1, &cand1).expect("commit 1") };

        // Immediate rescan with no writes must report 0 dirty pages.
        // SAFETY: Writers quiescent.
        let (dirty2, cand2) = unsafe { tracker.collect_dirty_pages().expect("scan 2") };
        assert!(dirty2.is_empty());
        // SAFETY: Writers quiescent.
        unsafe { tracker.commit_synced(&dirty2, &cand2).expect("commit 2") };

        // Rewrite page 3 only.
        // SAFETY: Offset lies within page 3.
        unsafe {
            ptr::write_volatile(tracker.as_ptr().as_ptr().add(3 * PAGE_SIZE + 8), 0xFE);
        }
        // SAFETY: Writers quiescent.
        let (dirty3, cand3) = unsafe { tracker.collect_dirty_pages().expect("scan 3") };
        assert_eq!(dirty3, vec![3]);
        // SAFETY: Writers quiescent.
        unsafe { tracker.commit_synced(&dirty3, &cand3).expect("commit 3") };
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use std::ffi::CStr;

    struct TestVkContext {
        _entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
        queue: vk::Queue,
        queue_family_index: u32,
        external_memory_host: bool,
        buffer_device_address: bool,
        device_name: String,
    }

    impl TestVkContext {
        fn new() -> Self {
            // SAFETY: Loads system `libvulkan.so.1`.
            let entry = unsafe { ash::Entry::load().expect("load libvulkan") };
            let app_info =
                vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
            let inst_ci = vk::InstanceCreateInfo::default().application_info(&app_info);
            // SAFETY: Valid Vulkan 1.3 instance create info.
            let instance = unsafe { entry.create_instance(&inst_ci, None) }
                .expect("create Vulkan 1.3 instance");

            // SAFETY: `instance` is live.
            let pdevs = unsafe { instance.enumerate_physical_devices() }
                .expect("enumerate physical devices");
            assert!(!pdevs.is_empty(), "no Vulkan physical devices found");

            let preferred = std::env::var("X2L_VK_DEVICE").ok();
            let physical_device = pdevs
                .iter()
                .copied()
                .find(|&pd| {
                    let props = unsafe { instance.get_physical_device_properties(pd) };
                    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                        .to_string_lossy()
                        .to_lowercase();
                    if let Some(ref want) = preferred {
                        name.contains(&want.to_lowercase())
                    } else {
                        props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                            || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                    }
                })
                .unwrap_or(pdevs[0]);

            // SAFETY: `physical_device` is live.
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            // SAFETY: `device_name` is a NUL-terminated Vulkan C string.
            let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            // SAFETY: `physical_device` is live.
            let qf_props =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let queue_family_index = qf_props
                .iter()
                .enumerate()
                .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(i, _)| i as u32)
                .expect("graphics queue family");

            // SAFETY: `physical_device` is live.
            let exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .expect("enumerate device extensions");
            let has_ext_host = exts.iter().any(|e| {
                // SAFETY: `extension_name` is a NUL-terminated Vulkan C string.
                let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                n == ash::ext::external_memory_host::NAME
            });

            let mut enabled_exts = Vec::new();
            if has_ext_host {
                enabled_exts.push(ash::ext::external_memory_host::NAME.as_ptr());
            }

            let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .timeline_semaphore(true);
            let mut feat13 = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true);

            let priorities = [1.0f32];
            let q_ci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let dev_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&q_ci)
                .enabled_extension_names(&enabled_exts)
                .push_next(&mut feat12)
                .push_next(&mut feat13);

            // SAFETY: `dev_ci` and its `pNext` chain are live on the stack.
            let device = unsafe { instance.create_device(physical_device, &dev_ci, None) }
                .expect("create Vulkan 1.3 device");
            // SAFETY: Queue 0 of `queue_family_index` was requested in `dev_ci`.
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

            Self {
                _entry: entry,
                instance,
                physical_device,
                device,
                queue,
                queue_family_index,
                external_memory_host: has_ext_host,
                buffer_device_address: true,
                device_name,
            }
        }
    }

    impl Drop for TestVkContext {
        fn drop(&mut self) {
            // SAFETY: All dependent objects (`UnifiedMemoryManager`) are dropped before `TestVkContext`.
            unsafe {
                let _ = self.device.device_wait_idle();
                self.device.destroy_device(None);
                self.instance.destroy_instance(None);
            }
        }
    }

    #[test]
    fn gpu_userfaultfd_dirty_sync_no_spurious_reupload_and_teardown() {
        let ctx = TestVkContext::new();
        // SAFETY: `ctx` outlives `mgr`.
        let mut mgr = unsafe {
            UnifiedMemoryManager::new(
                ctx.instance.clone(),
                ctx.physical_device,
                ctx.device.clone(),
                ctx.queue,
                ctx.queue_family_index,
                ctx.external_memory_host,
                ctx.buffer_device_address,
            )
            .expect("create UnifiedMemoryManager")
        };

        let caps = mgr.capabilities();
        println!(
            "[x2l-gpu-vk::memory] device='{}' ext_host={} min_align={} bda={} uffd_mode={}",
            ctx.device_name,
            caps.external_memory_host,
            caps.min_imported_host_pointer_alignment,
            caps.buffer_device_address,
            caps.userfaultfd_mode.as_str()
        );

        // 1. Tier 2 userfaultfd-tracked allocation (4 pages = 16 KiB).
        let va = mgr.reserve_guest_va(16384).expect("reserve va");
        let id = mgr
            .allocate(va, 4 * PAGE_SIZE, AllocationPolicy::ForceDirtyTracked)
            .expect("allocate dirty-tracked");
        let (_buf, bda) = mgr.allocation_vulkan(id).expect("vulkan handles");
        assert_ne!(bda, 0, "BDA must be non-zero");

        let resolved = mgr.registry().lookup(va + 4096 + 16).expect("VA lookup");
        assert_eq!(resolved.allocation_id, id);
        assert_eq!(resolved.offset, 4096 + 16);
        assert_eq!(resolved.device_address, bda + 4096 + 16);

        // SAFETY: Single-threaded test owns `id`.
        let (ptr, len) = unsafe { mgr.guest_ptr(id).expect("guest_ptr") };
        assert_eq!(len, 4 * PAGE_SIZE);

        // Write pages 1 and 2 (adjacent -> should coalesce into 1 copy region of 8 KiB).
        // SAFETY: Offsets lie inside the 16 KiB guest mapping.
        unsafe {
            ptr::write_volatile(ptr.as_ptr().add(PAGE_SIZE + 4), 0xDE);
            ptr::write_volatile(ptr.as_ptr().add(PAGE_SIZE + 5), 0xAD);
            ptr::write_volatile(ptr.as_ptr().add(2 * PAGE_SIZE + 8), 0xBE);
            ptr::write_volatile(ptr.as_ptr().add(2 * PAGE_SIZE + 9), 0xEF);
        }

        // SAFETY: Writers are quiescent.
        let stats1 = unsafe { mgr.flush_dirty(id).expect("flush 1") };
        assert_eq!(stats1.dirty_pages, 2);
        assert_eq!(stats1.copy_regions, 1);
        assert_eq!(stats1.bytes_uploaded, 8192);

        let gpu_bytes = mgr.readback_buffer(id).expect("readback 1");
        assert_eq!(gpu_bytes[0], 0);
        assert_eq!(gpu_bytes[PAGE_SIZE + 4], 0xDE);
        assert_eq!(gpu_bytes[PAGE_SIZE + 5], 0xAD);
        assert_eq!(gpu_bytes[2 * PAGE_SIZE + 8], 0xBE);
        assert_eq!(gpu_bytes[2 * PAGE_SIZE + 9], 0xEF);
        assert_eq!(gpu_bytes[3 * PAGE_SIZE], 0);

        // 2. Immediate second flush with no writes must upload 0 bytes (no spurious re-upload).
        // SAFETY: Writers are quiescent.
        let stats2 = unsafe { mgr.flush_dirty(id).expect("flush 2") };
        assert_eq!(stats2.dirty_pages, 0);
        assert_eq!(stats2.copy_regions, 0);
        assert_eq!(stats2.bytes_uploaded, 0);

        // 3. Modify page 3 only -> exactly 1 dirty page (4 KiB).
        // SAFETY: Offset lies inside page 3.
        unsafe {
            ptr::write_volatile(ptr.as_ptr().add(3 * PAGE_SIZE + 12), 0x77);
        }
        // SAFETY: Writers are quiescent.
        let stats3 = unsafe { mgr.flush_dirty(id).expect("flush 3") };
        assert_eq!(stats3.dirty_pages, 1);
        assert_eq!(stats3.copy_regions, 1);
        assert_eq!(stats3.bytes_uploaded, 4096);

        let gpu_bytes3 = mgr.readback_buffer(id).expect("readback 3");
        assert_eq!(gpu_bytes3[3 * PAGE_SIZE + 12], 0x77);
        assert_eq!(gpu_bytes3[PAGE_SIZE + 4], 0xDE);

        // 4. Clean teardown.
        // SAFETY: No guest pointers remain in use.
        unsafe { mgr.free_allocation(id).expect("free dirty-tracked") };
        assert_eq!(mgr.live_allocations(), 0);
        assert!(mgr.registry().is_empty());

        // 5. Tier 1 HostExternal zero-copy allocation (when advertised by the ICD).
        if caps.external_memory_host {
            let va2 = mgr.reserve_guest_va(8192).expect("reserve va2");
            let id2 = mgr
                .allocate(va2, 8192, AllocationPolicy::PreferHostExternal)
                .expect("allocate host external");
            assert_eq!(
                mgr.allocation_tier(id2).expect("tier"),
                MemoryTier::HostExternal
            );
            let (_buf2, bda2) = mgr.allocation_vulkan(id2).expect("bda2");
            assert_ne!(bda2, 0);
            // SAFETY: Single-threaded test owns `id2`.
            let (ptr2, _) = unsafe { mgr.guest_ptr(id2).expect("guest_ptr2") };
            // SAFETY: Offset 64 lies within 8192-byte host-imported mapping.
            unsafe {
                ptr::write_volatile(ptr2.as_ptr().add(64), 0x5A);
            }
            let rb2 = mgr.readback_buffer(id2).expect("readback host external");
            assert_eq!(rb2[64], 0x5A);
            // SAFETY: No pointers remain in use.
            unsafe { mgr.free_allocation(id2).expect("free host external") };
        }

        // 6. Tier 2 ExplicitStaging fallback test.
        let va3 = mgr.reserve_guest_va(8192).expect("reserve va3");
        let id3 = mgr
            .allocate(va3, 8192, AllocationPolicy::ForceExplicitStaging)
            .expect("allocate explicit");
        assert_eq!(
            mgr.allocation_tier(id3).expect("tier3"),
            MemoryTier::ExplicitStaging
        );
        // SAFETY: Single-threaded test owns `id3`.
        let (ptr3, _) = unsafe { mgr.guest_ptr(id3).expect("guest_ptr3") };
        // SAFETY: Offset lies inside page 1.
        unsafe {
            ptr::write_volatile(ptr3.as_ptr().add(PAGE_SIZE + 3), 0xC3);
        }
        // SAFETY: Writers quiescent.
        let st_exp1 = unsafe { mgr.flush_dirty(id3).expect("explicit flush 1") };
        assert_eq!(st_exp1.dirty_pages, 1);
        assert_eq!(st_exp1.bytes_uploaded, 4096);
        // SAFETY: Writers quiescent.
        let st_exp2 = unsafe { mgr.flush_dirty(id3).expect("explicit flush 2") };
        assert_eq!(st_exp2.dirty_pages, 0);
        let rb3 = mgr.readback_buffer(id3).expect("readback explicit");
        assert_eq!(rb3[PAGE_SIZE + 3], 0xC3);
        // SAFETY: No pointers remain in use.
        unsafe { mgr.free_allocation(id3).expect("free explicit") };

        drop(mgr);
    }
}
