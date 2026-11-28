//! Thin Win64 PE shim for `kernelx.dll` forwarding to `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    OP_KERNELX_GET_CONSOLE_TYPE, OP_KERNELX_GET_DEVICE_TYPE, OP_KERNELX_TITLE_MEMORY_STATUS,
    OP_KERNELX_XMEM_ALLOC, OP_KERNELX_XMEM_FREE, OP_XMEM_VIRTUAL_ALLOC, call_bridge,
    console_virtual_alloc, console_virtual_alloc_ex, console_virtual_alloc_from_app,
    console_virtual_free, console_virtual_free_ex, console_virtual_query, console_virtual_query_ex,
    handle_dll_main, record_unknown_ordinal, route_virtual_alloc_2,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn GetConsoleType() -> u32 {
    // SAFETY: `OP_KERNELX_GET_CONSOLE_TYPE` takes no pointers.
    unsafe { call_bridge(OP_KERNELX_GET_CONSOLE_TYPE, 0, 0, 0, 0, 0, 0).unwrap_or(0x0007) as u32 }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetDeviceType() -> u32 {
    // SAFETY: `OP_KERNELX_GET_DEVICE_TYPE` takes no pointers.
    unsafe { call_bridge(OP_KERNELX_GET_DEVICE_TYPE, 0, 0, 0, 0, 0, 0).unwrap_or(4) as u32 }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    // SAFETY: `OP_KERNELX_XMEM_ALLOC` allocates zeroed 16-byte aligned memory in `libx2l_core.so`.
    let addr = unsafe {
        call_bridge(OP_KERNELX_XMEM_ALLOC, size as u64, attributes, 0, 0, 0, 0).unwrap_or(0)
    };
    addr as usize as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemFree(ptr: *mut c_void, attributes: u64) {
    // SAFETY: `ptr` was allocated by `XMemAlloc` (null is a safe no-op).
    let _ = unsafe {
        call_bridge(
            OP_KERNELX_XMEM_FREE,
            ptr as usize as u64,
            attributes,
            0,
            0,
            0,
            0,
        )
    };
}

#[unsafe(no_mangle)]
pub extern "win64" fn TitleMemoryStatus(out_buf: *mut c_void) -> i32 {
    // SAFETY: `out_buf` is the caller's writable `TITLEMEMORYSTATUS` struct pointer.
    unsafe {
        call_bridge(
            OP_KERNELX_TITLE_MEMORY_STATUS,
            out_buf as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XMemVirtualAlloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    attributes: u64,
    protect: u32,
) -> *mut c_void {
    // SAFETY: Routes directly to `XMemArena` for console `< 44-bit` aligned reservations & commits.
    let res = unsafe {
        call_bridge(
            OP_XMEM_VIRTUAL_ALLOC,
            addr as usize as u64,
            size as u64,
            u64::from(alloc_type),
            attributes,
            u64::from(protect),
            0,
        )
        .unwrap_or(0)
    };
    res as usize as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "win64" fn x2l_kernelx_virtual_alloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    console_virtual_alloc(addr, size, alloc_type, protect)
}

#[unsafe(no_mangle)]
pub extern "win64" fn x2l_kernelx_virtual_alloc_ex(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    console_virtual_alloc_ex(process, addr, size, alloc_type, protect)
}

#[unsafe(no_mangle)]
pub extern "win64" fn x2l_kernelx_virtual_alloc_from_app(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    console_virtual_alloc_from_app(addr, size, alloc_type, protect)
}

#[unsafe(no_mangle)]
pub extern "win64" fn VirtualAlloc2(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
    parameters: *mut c_void,
    parameter_count: u32,
) -> *mut c_void {
    route_virtual_alloc_2(
        process,
        addr,
        size,
        alloc_type,
        protect,
        parameters,
        parameter_count,
        false,
        true,
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn VirtualAlloc2FromApp(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
    parameters: *mut c_void,
    parameter_count: u32,
) -> *mut c_void {
    route_virtual_alloc_2(
        process,
        addr,
        size,
        alloc_type,
        protect,
        parameters,
        parameter_count,
        true,
        true,
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn x2l_kernelx_virtual_free(
    addr: *mut c_void,
    size: usize,
    free_type: u32,
) -> i32 {
    console_virtual_free(addr, size, free_type)
}

#[unsafe(no_mangle)]
pub extern "win64" fn VirtualFreeEx(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    free_type: u32,
) -> i32 {
    console_virtual_free_ex(process, addr, size, free_type)
}

#[unsafe(no_mangle)]
pub extern "win64" fn x2l_kernelx_virtual_query(
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> usize {
    console_virtual_query(addr, buf, len)
}

#[unsafe(no_mangle)]
pub extern "win64" fn x2l_kernelx_virtual_query_ex(
    process: *mut c_void,
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> usize {
    console_virtual_query_ex(process, addr, buf, len)
}

/// Auto-stubbed unknown ordinal `#999` (exported NONAME via `kernelx.def`).
#[unsafe(no_mangle)]
pub extern "win64" fn x2l_ord_stub_999() -> u64 {
    record_unknown_ordinal("kernelx.dll", 999, 0)
}
