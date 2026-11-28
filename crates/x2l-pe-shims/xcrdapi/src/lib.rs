//! Thin Win64 PE shim for `XCRDAPI.dll` (DirectStorage) forwarding into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_DSTORAGE_GET_FACTORY, call_bridge, capture_caller_rva, handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn DStorageGetFactory(riid: *const c_void, ppv: *mut *mut c_void) -> i32 {
    let caller_rva = capture_caller_rva(1);
    // SAFETY: `ppv` is the caller's writable `IDStorageFactoryX**` output pointer.
    unsafe {
        call_bridge(
            OP_DSTORAGE_GET_FACTORY,
            ppv as usize as u64,
            riid as usize as u64,
            caller_rva,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}
