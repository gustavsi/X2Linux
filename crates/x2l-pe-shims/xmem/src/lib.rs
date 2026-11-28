//! Thin Win64 PE shim for `xmem.dll` forwarding all 13 console memory exports into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    GetModuleHandleA, OP_XMEM_ALLOC_DEFAULT, OP_XMEM_ALLOC_PHYSICAL_PAGES, OP_XMEM_FREE_DEFAULT,
    OP_XMEM_FREE_PHYSICAL_PAGES, OP_XMEM_GET_WORKING_SET_STATS, OP_XMEM_IS_TOOLS_MEM_AVAILABLE,
    OP_XMEM_MAP_PHYSICAL_PAGES, OP_XMEM_MAP_PHYSICAL_PAGES_SCATTER, OP_XMEM_TRANSFER_MEMORY,
    OP_XMEM_VIRTUAL_ALLOC, OP_XMEM_VIRTUAL_QUERY, call_bridge, handle_dll_main,
    install_valloc_iat_patch,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    if reason == 1 {
        // Only install the IAT patch from `xmem.dll` if `kernelx.dll` is not loaded in this image.
        // SAFETY: `GetModuleHandleA` checks already-loaded modules without invoking `LoadLibrary`.
        if unsafe { GetModuleHandleA(b"kernelx.dll\0".as_ptr()) } == 0 {
            install_valloc_iat_patch();
        }
    }
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void {
    XMemAllocDefault(size, attrs)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemAllocDefault(size: usize, attrs: u64) -> *mut c_void {
    // SAFETY: Bridge allocates zeroed memory inside this process address space.
    let addr =
        unsafe { call_bridge(OP_XMEM_ALLOC_DEFAULT, size as u64, attrs, 0, 0, 0, 0).unwrap_or(0) };
    addr as usize as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemFree(ptr: *mut c_void, attrs: u64) {
    XMemFreeDefault(ptr, attrs);
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemFreeDefault(ptr: *mut c_void, attrs: u64) {
    // SAFETY: Frees pointer previously allocated by `XMemAlloc[Default]`.
    let _ = unsafe { call_bridge(OP_XMEM_FREE_DEFAULT, ptr as usize as u64, attrs, 0, 0, 0, 0) };
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    out_pfns: *mut usize,
) -> i32 {
    // SAFETY: `count` and `out_pfns` are caller-provided in/out pointers.
    unsafe {
        call_bridge(
            OP_XMEM_ALLOC_PHYSICAL_PAGES,
            flags,
            count as usize as u64,
            out_pfns as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemFreePhysicalPages(count: usize, pfns: *const usize) -> i32 {
    // SAFETY: `pfns` points to `count` PFNs allocated by `XMemAllocatePhysicalPages`.
    unsafe {
        call_bridge(
            OP_XMEM_FREE_PHYSICAL_PAGES,
            count as u64,
            pfns as usize as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemMapPhysicalPages(
    base: *mut c_void,
    count: usize,
    pfns: *const usize,
) -> *mut c_void {
    // SAFETY: `pfns` is either null (unmap) or a readable `count`-element PFN array.
    let addr = unsafe {
        call_bridge(
            OP_XMEM_MAP_PHYSICAL_PAGES,
            base as usize as u64,
            count as u64,
            pfns as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0)
    };
    addr as usize as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    count: usize,
    pfns: *const usize,
) -> i32 {
    // SAFETY: `addresses` and `pfns` are readable `count`-element arrays.
    unsafe {
        call_bridge(
            OP_XMEM_MAP_PHYSICAL_PAGES_SCATTER,
            addresses as usize as u64,
            count as u64,
            pfns as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    alloc_type: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    // SAFETY: Bridge reserves/commits arena-owned Linux virtual memory on SUD-free native worker.
    let addr = unsafe {
        call_bridge(
            OP_XMEM_VIRTUAL_ALLOC,
            base as usize as u64,
            size as u64,
            u64::from(alloc_type),
            flags,
            u64::from(protect),
            0,
        )
        .unwrap_or(0)
    };
    addr as usize as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemVirtualQuery(
    process: *mut c_void,
    addr: *const c_void,
    out_info: *mut c_void,
) -> i32 {
    // SAFETY: `out_info` is the caller's writable 64-byte `XMemBasicInformation` buffer.
    unsafe {
        call_bridge(
            OP_XMEM_VIRTUAL_QUERY,
            process as usize as u64,
            addr as usize as u64,
            out_info as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemTransferMemory(direction: u32, size: usize) -> i32 {
    // SAFETY: Pure integer budget adjustment via bridge.
    unsafe {
        call_bridge(
            OP_XMEM_TRANSFER_MEMORY,
            u64::from(direction),
            size as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemGetWorkingSetStatistics(flags: u32, out_buf: *mut u8) -> i32 {
    // SAFETY: `out_buf` is a caller-supplied writable buffer of at least 16 bytes.
    unsafe {
        call_bridge(
            OP_XMEM_GET_WORKING_SET_STATS,
            u64::from(flags),
            out_buf as usize as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemIsToolsMemAvailable() -> i32 {
    // SAFETY: Query-only bridge call.
    unsafe { call_bridge(OP_XMEM_IS_TOOLS_MEM_AVAILABLE, 0, 0, 0, 0, 0, 0).unwrap_or(0) as i32 }
}
