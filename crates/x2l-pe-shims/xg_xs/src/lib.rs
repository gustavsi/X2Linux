//! Thin Win64 PE shim for `xg_xs.dll` forwarding texture tiling queries into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_XG_CREATE_TEXTURE_COMPUTER, OP_XG_SUGGEST_SWIZZLE_MODE, call_bridge,
    handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGCreateTextureComputer(
    desc: *const c_void,
    out_computer: *mut *mut c_void,
) -> i32 {
    // SAFETY: `desc` points to a 56-byte `D3D12_RESOURCE_DESC`; `out_computer` is writable.
    unsafe {
        call_bridge(
            OP_XG_CREATE_TEXTURE_COMPUTER,
            desc as usize as u64,
            out_computer as usize as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGSuggestSwizzleMode(
    desc: *const c_void,
    flags: u32,
    out_layout: *mut u32,
) -> i32 {
    // SAFETY: `desc` is readable and `out_layout` is writable.
    unsafe {
        call_bridge(
            OP_XG_SUGGEST_SWIZZLE_MODE,
            desc as usize as u64,
            u64::from(flags),
            out_layout as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}
