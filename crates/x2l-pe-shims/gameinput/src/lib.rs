//! Thin Win64 PE shim for `GameInput.dll` exporting ordinal `#1` (`GameInputCreate`).

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_GAMEINPUT_CREATE, call_bridge, capture_caller_rva, handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn GameInputCreate(out_gameinput: *mut *mut c_void) -> i32 {
    let caller_rva = capture_caller_rva(1);
    // SAFETY: `out_gameinput` is the caller's writable `IGameInput**` output pointer.
    unsafe {
        call_bridge(
            OP_GAMEINPUT_CREATE,
            out_gameinput as usize as u64,
            caller_rva,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}
