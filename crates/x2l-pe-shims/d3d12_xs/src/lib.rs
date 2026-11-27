//! Thin Win64 PE shim for `d3d12_xs.dll` forwarding into `libx2l_core.so` (`x2l-gpu-vk`).

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_D3D12X_CREATE_DEVICE, call_bridge, capture_caller_rva, handle_dll_main,
    record_stub,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn D3D12CreateDevice(
    _adapter: *mut c_void,
    _min_feature_level: u32,
    _riid: *const c_void,
    out_device: *mut *mut c_void,
) -> i32 {
    let caller_rva = capture_caller_rva(1);
    // SAFETY: `out_device` is the caller's writable COM `ID3D12DeviceX**` output pointer.
    unsafe {
        call_bridge(
            OP_D3D12X_CREATE_DEVICE,
            out_device as usize as u64,
            caller_rva,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn D3D12GetDebugInterface(
    _riid: *const c_void,
    _out_debug: *mut *mut c_void,
) -> i32 {
    record_stub(
        "d3d12_xs.dll",
        "D3D12GetDebugInterface",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}
