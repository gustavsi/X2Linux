//! Console `kernelx` HLE: console identity, memory budgets, flag-stripped allocations, and `TitleMemoryStatus`.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::BTreeMap;
use std::sync::Mutex;
use x2l_common::DeviceType;

/// Console type returned by `kernelx!GetConsoleType` (public Scarlett/Lockhart IDs).
pub const CONSOLE_TYPE_SERIES_S: u32 = 0x0006;
pub const CONSOLE_TYPE_SERIES_X: u32 = 0x0007;

/// Public `XSystemDeviceType` values from `docs.xbox.com`.
pub const XSYSTEM_DEVICE_SERIES_S: u32 = 3;
pub const XSYSTEM_DEVICE_SERIES_X: u32 = 4;

/// Console-only VirtualAlloc flags stripped before forwarding to host memory allocation.
pub const CONSOLE_ONLY_ALLOC_FLAGS: u32 = 0x1000_0000 | 0x4000_0000 | 0x0040_0000;
/// Console-only page protection bits stripped before forwarding.
pub const CONSOLE_ONLY_PROTECT_FLAGS: u32 = 0x0000_0800 | 0x0000_1000 | 0x0000_2000;

/// Console `TITLEMEMORYSTATUS` layout (public `docs.xbox.com` structure).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TitleMemoryStatus {
    pub dw_length: u32,
    pub dw_reserved: u32,
    pub ull_total_mem: u64,
    pub ull_avail_mem: u64,
    pub ull_legacy_used: u64,
    pub ull_legacy_peak: u64,
    pub ull_legacy_avail: u64,
    pub ull_title_used: u64,
    pub ull_title_avail: u64,
    pub ull_legacy_page_faults: u64,
    pub ull_title_page_faults: u64,
}

/// Reported title memory budget in bytes for the selected console profile.
pub const fn title_memory_budget_bytes(device: DeviceType) -> u64 {
    match device {
        DeviceType::SeriesX => 13_500_000_000,
        DeviceType::SeriesS => 8_000_000_000,
    }
}

/// Return the console type ID for `kernelx!GetConsoleType`.
pub const fn console_type_id(device: DeviceType) -> u32 {
    match device {
        DeviceType::SeriesX => CONSOLE_TYPE_SERIES_X,
        DeviceType::SeriesS => CONSOLE_TYPE_SERIES_S,
    }
}

/// Return the `XSystemDeviceType` enum value for `XSystemGetDeviceType`.
pub const fn xsystem_device_type(device: DeviceType) -> u32 {
    match device {
        DeviceType::SeriesX => XSYSTEM_DEVICE_SERIES_X,
        DeviceType::SeriesS => XSYSTEM_DEVICE_SERIES_S,
    }
}

/// Strip console-only `VirtualAlloc` allocation and protection flags.
pub const fn strip_virtual_alloc_flags(alloc_type: u32, protect: u32) -> (u32, u32) {
    let clean_alloc = alloc_type & !CONSOLE_ONLY_ALLOC_FLAGS;
    let clean_protect = protect & !CONSOLE_ONLY_PROTECT_FLAGS;
    (
        clean_alloc,
        if clean_protect == 0 {
            0x04
        } else {
            clean_protect
        },
    )
}

/// Stateful `kernelx` HLE runtime tracking console profile and `XMemAlloc` allocations.
#[derive(Debug)]
pub struct KernelxState {
    device: DeviceType,
    allocations: Mutex<BTreeMap<usize, Layout>>,
    bytes_used: Mutex<u64>,
}

impl KernelxState {
    pub fn new(device: DeviceType) -> Self {
        Self {
            device,
            allocations: Mutex::new(BTreeMap::new()),
            bytes_used: Mutex::new(0),
        }
    }

    pub fn console_type(&self) -> u32 {
        console_type_id(self.device)
    }

    pub fn device_type(&self) -> u32 {
        xsystem_device_type(self.device)
    }

    /// Allocate zero-initialized, 16-byte aligned memory for `kernelx!XMemAlloc` (stripping
    /// console-only attribute bits).
    pub fn xmem_alloc(&self, size: usize, _attributes: u64) -> *mut u8 {
        let alloc_size = size.max(1);
        let Ok(layout) = Layout::from_size_align(alloc_size, 16) else {
            return std::ptr::null_mut();
        };
        // SAFETY: `layout` has non-zero size and power-of-two alignment 16.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return ptr;
        }
        if let Ok(mut map) = self.allocations.lock() {
            map.insert(ptr as usize, layout);
        }
        if let Ok(mut used) = self.bytes_used.lock() {
            *used = used.saturating_add(alloc_size as u64);
        }
        ptr
    }

    /// Free an allocation previously returned by [`Self::xmem_alloc`].
    ///
    /// Returns `false` safely without touching memory if `addr` was not allocated by this table.
    pub fn xmem_free(&self, addr: usize, _attributes: u64) -> bool {
        if addr == 0 {
            return true;
        }
        let removed = self
            .allocations
            .lock()
            .ok()
            .and_then(|mut map| map.remove(&addr));
        let Some(layout) = removed else {
            return false;
        };
        if let Ok(mut used) = self.bytes_used.lock() {
            *used = used.saturating_sub(layout.size() as u64);
        }
        // SAFETY: `addr` was returned by `alloc_zeroed(layout)` in `xmem_alloc` and removed
        // from the tracking table so it cannot be double-freed.
        unsafe { dealloc(addr as *mut u8, layout) };
        true
    }

    /// Populate a `TitleMemoryStatus` snapshot reflecting the configured console budget.
    pub fn title_memory_status(&self) -> TitleMemoryStatus {
        let total = title_memory_budget_bytes(self.device);
        let used = self.bytes_used.lock().map(|g| *g).unwrap_or(0);
        let avail = total.saturating_sub(used);
        TitleMemoryStatus {
            dw_length: std::mem::size_of::<TitleMemoryStatus>() as u32,
            dw_reserved: 0,
            ull_total_mem: total,
            ull_avail_mem: avail,
            ull_legacy_used: 0,
            ull_legacy_peak: 0,
            ull_legacy_avail: 0,
            ull_title_used: used,
            ull_title_avail: avail,
            ull_legacy_page_faults: 0,
            ull_title_page_faults: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xmem_alloc_free_and_memory_status() {
        let state = KernelxState::new(DeviceType::SeriesX);
        assert_eq!(state.console_type(), CONSOLE_TYPE_SERIES_X);
        assert_eq!(state.device_type(), XSYSTEM_DEVICE_SERIES_X);

        let ptr = state.xmem_alloc(4096, 0xFFFF_0000);
        assert!(!ptr.is_null());
        let status = state.title_memory_status();
        assert_eq!(status.ull_total_mem, 13_500_000_000);
        assert_eq!(status.ull_title_used, 4096);
        assert!(state.xmem_free(ptr as usize, 0));
        assert!(!state.xmem_free(ptr as usize, 0));
        assert_eq!(state.title_memory_status().ull_title_used, 0);
    }

    #[test]
    fn strips_console_virtual_alloc_flags() {
        let (alloc_t, prot) = strip_virtual_alloc_flags(0x1000_3000, 0x0000_1004);
        assert_eq!(alloc_t, 0x3000);
        assert_eq!(prot, 0x04);
    }
}
