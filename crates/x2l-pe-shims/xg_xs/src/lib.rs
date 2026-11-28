//! Thin Win64 PE shim for `xg_xs.dll` forwarding texture tiling queries into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_XG_COMPUTE_OPTIMAL_TILE_MODE, OP_XG_CREATE_TEXTURE_COMPUTER,
    OP_XG_CREATE_TYPED_COMPUTER, OP_XG_SUGGEST_SWIZZLE_MODE, call_bridge, handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

fn create_typed(dimension: u32, desc: *const c_void, out_computer: *mut *mut c_void) -> i32 {
    // SAFETY: The export passes its descriptor and writable COM output to a synchronous call.
    unsafe {
        call_bridge(
            OP_XG_CREATE_TYPED_COMPUTER,
            u64::from(dimension),
            desc as u64,
            out_computer as u64,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGCreateTexture2DComputer(
    desc: *const c_void,
    out_computer: *mut *mut c_void,
) -> i32 {
    create_typed(3, desc, out_computer)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGCreateTexture3DComputer(
    desc: *const c_void,
    out_computer: *mut *mut c_void,
) -> i32 {
    create_typed(4, desc, out_computer)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGComputeOptimalTileMode(
    dimension: u32,
    format: u32,
    width: u32,
    height: u32,
    depth_or_array_size: u32,
    samples: u32,
    bind_flags: u32,
) -> u32 {
    XGComputeOptimalSwizzleMode(
        dimension,
        format,
        width,
        height,
        depth_or_array_size,
        samples,
        bind_flags,
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGComputeOptimalSwizzleMode(
    dimension: u32,
    format: u32,
    width: u32,
    height: u32,
    depth_or_array_size: u32,
    samples: u32,
    bind_flags: u32,
) -> u32 {
    let args = [
        dimension,
        format,
        width,
        height,
        depth_or_array_size,
        samples,
        bind_flags,
    ];
    // SAFETY: The synchronous bridge reads the seven live argument words.
    unsafe {
        call_bridge(
            OP_XG_COMPUTE_OPTIMAL_TILE_MODE,
            args.as_ptr() as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(u64::from(u32::MAX)) as u32
    }
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
