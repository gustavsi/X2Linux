//! Thin Win64 PE shim for offline `xgameruntime.dll` forwarding to `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, HRESULT_S_OK, OP_XGR_GAMESAVE_ROOT, OP_XGR_INITIALIZE,
    OP_XGR_PACKAGE_MOUNT_ROOT, OP_XGR_STORE_OFFLINE, OP_XGR_UNINITIALIZE, OP_XGR_USER_GET_GAMERTAG,
    call_bridge, handle_dll_main, record_stub,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameRuntimeInitialize() -> i32 {
    // SAFETY: `OP_XGR_INITIALIZE` takes no pointers.
    unsafe { call_bridge(OP_XGR_INITIALIZE, 0, 0, 0, 0, 0, 0).unwrap_or(0) as i32 }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameRuntimeUninitialize() {
    // SAFETY: `OP_XGR_UNINITIALIZE` takes no pointers.
    let _ = unsafe { call_bridge(OP_XGR_UNINITIALIZE, 0, 0, 0, 0, 0, 0) };
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetGamertag(
    _user: *mut c_void,
    _component: u32,
    buf_size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    // SAFETY: Caller supplies writable `buf` of length `buf_size` and optional `used` out-pointer.
    let hr = unsafe {
        call_bridge(
            OP_XGR_USER_GET_GAMERTAG,
            buf as usize as u64,
            buf_size as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    };
    if hr == HRESULT_S_OK && !buf.is_null() && !used.is_null() {
        // SAFETY: `OP_XGR_USER_GET_GAMERTAG` guarantees NUL termination within `buf_size`.
        unsafe {
            let slice = std::slice::from_raw_parts(buf, buf_size);
            let len = slice.iter().position(|&b| b == 0).unwrap_or(0);
            *used = len + 1;
        }
    }
    hr
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserAddAsync(_options: u32, _async_block: *mut c_void) -> i32 {
    record_stub(
        "xgameruntime.dll",
        "XUserAddAsync",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetCurrentProcessPackageIdentifier(
    buf_size: usize,
    buf: *mut u8,
) -> i32 {
    // SAFETY: Caller supplies writable `buf` of length `buf_size`.
    unsafe {
        call_bridge(
            OP_XGR_PACKAGE_MOUNT_ROOT,
            buf as usize as u64,
            buf_size as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageMountWithUiAsync(
    _package_id: *const u8,
    _async_block: *mut c_void,
) -> i32 {
    record_stub(
        "xgameruntime.dll",
        "XPackageMountWithUiAsync",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveInitializeProviderAsync(
    _user: *mut c_void,
    _scid: *const u8,
    _sync_on_demand: bool,
    _async_block: *mut c_void,
) -> i32 {
    let mut scratch = [0u8; 260];
    // SAFETY: Ensures saves directory is created under `$XDG_DATA_HOME/x2linux/<title>/saves/`.
    let _ = unsafe {
        call_bridge(
            OP_XGR_GAMESAVE_ROOT,
            scratch.as_mut_ptr() as usize as u64,
            scratch.len() as u64,
            0,
            0,
            0,
            0,
        )
    };
    record_stub(
        "xgameruntime.dll",
        "XGameSaveInitializeProviderAsync",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreCreateContext(_user: *mut c_void, _out_ctx: *mut *mut c_void) -> i32 {
    // SAFETY: Offline-only; never fakes commercial licenses.
    unsafe {
        call_bridge(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0).unwrap_or(HRESULT_E_NOTIMPL as u64)
            as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreQueryGameLicenseAsync(
    _ctx: *mut c_void,
    _async_block: *mut c_void,
) -> i32 {
    // SAFETY: Offline-only; never fakes commercial licenses.
    unsafe {
        call_bridge(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0).unwrap_or(HRESULT_E_NOTIMPL as u64)
            as i32
    }
}
