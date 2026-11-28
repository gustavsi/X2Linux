//! Thin Win64 PE shim for `api-ms-win-core-psm-appnotify-l1-1-0.dll` & `l1-1-1.dll` (4 exports).

use std::ffi::c_void;
use x2l_shim_common::{
    OP_PSM_REGISTER_APP_STATE, OP_PSM_REGISTER_CONSTRAINED, OP_PSM_UNREGISTER_APP_STATE,
    OP_PSM_UNREGISTER_CONSTRAINED, call_bridge, handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn RegisterAppStateChangeNotification(
    callback: *const c_void,
    context: *mut c_void,
    out_registration: *mut *mut c_void,
) -> u32 {
    // SAFETY: `out_registration` is the caller's writable cookie output pointer.
    unsafe {
        call_bridge(
            OP_PSM_REGISTER_APP_STATE,
            callback as usize as u64,
            context as usize as u64,
            out_registration as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0) as u32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn UnregisterAppStateChangeNotification(registration: *mut c_void) {
    // SAFETY: Unregisters cookie returned by `RegisterAppStateChangeNotification`.
    let _ = unsafe {
        call_bridge(
            OP_PSM_UNREGISTER_APP_STATE,
            registration as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
    };
}

#[unsafe(no_mangle)]
pub extern "win64" fn RegisterAppConstrainedChangeNotification(
    callback: *const c_void,
    context: *mut c_void,
    out_registration: *mut *mut c_void,
) -> u32 {
    // SAFETY: `out_registration` is the caller's writable cookie output pointer.
    unsafe {
        call_bridge(
            OP_PSM_REGISTER_CONSTRAINED,
            callback as usize as u64,
            context as usize as u64,
            out_registration as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0) as u32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn UnregisterAppConstrainedChangeNotification(registration: *mut c_void) {
    // SAFETY: Unregisters cookie returned by `RegisterAppConstrainedChangeNotification`.
    let _ = unsafe {
        call_bridge(
            OP_PSM_UNREGISTER_CONSTRAINED,
            registration as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
    };
}
