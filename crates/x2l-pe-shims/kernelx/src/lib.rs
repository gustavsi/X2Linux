//! Thin Win64 PE shim for `kernelx.dll` forwarding to `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    OP_KERNELX_GET_CONSOLE_TYPE, OP_KERNELX_GET_DEVICE_TYPE, OP_KERNELX_TITLE_MEMORY_STATUS,
    OP_KERNELX_VIRTUAL_ALLOC, OP_KERNELX_XMEM_ALLOC, OP_KERNELX_XMEM_FREE, VirtualAlloc,
    call_bridge, handle_dll_main, record_unknown_ordinal,
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
    _attributes: u64,
    protect: u32,
) -> *mut c_void {
    // Strip console-only memory/protection flags via `libx2l_core.so`, then forward to Wine `VirtualAlloc`.
    // SAFETY: `OP_KERNELX_VIRTUAL_ALLOC` computes sanitized flags without dereferencing `addr`.
    let packed = unsafe {
        call_bridge(
            OP_KERNELX_VIRTUAL_ALLOC,
            addr as usize as u64,
            size as u64,
            u64::from(alloc_type),
            u64::from(protect),
            0,
            0,
        )
        .unwrap_or(((u64::from(alloc_type)) << 32) | u64::from(protect))
    };
    let clean_alloc = (packed >> 32) as u32;
    let clean_protect = (packed & 0xFFFF_FFFF) as u32;
    // SAFETY: Forwards sanitized Win32 allocation flags to Wine's `kernel32!VirtualAlloc`.
    unsafe { VirtualAlloc(addr, size, clean_alloc, clean_protect) }
}

/// Auto-stubbed unknown ordinal `#999` (exported NONAME via `kernelx.def`).
#[unsafe(no_mangle)]
pub extern "win64" fn x2l_ord_stub_999() -> u64 {
    record_unknown_ordinal("kernelx.dll", 999, 0)
}
