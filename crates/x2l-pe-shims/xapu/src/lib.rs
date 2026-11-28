//! Thin Win64 PE shim for `xapu.dll` (5 exports) forwarding into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_XAPU_CONNECT, OP_XAPU_DEQUEUE_RESULT, OP_XAPU_DISCONNECT,
    OP_XAPU_ENQUEUE_COMMAND, OP_XAPU_GET_RESULT_COUNT, call_bridge, handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XApuConnect(
    input: *const c_void,
    output: *mut c_void,
    out_conn: *mut *mut c_void,
) -> i32 {
    // SAFETY: `out_conn` is the caller's writable connection handle output pointer.
    unsafe {
        call_bridge(
            OP_XAPU_CONNECT,
            input as usize as u64,
            output as usize as u64,
            out_conn as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XApuDisconnect(conn: *mut c_void) -> i32 {
    // SAFETY: Disconnects handle previously returned by `XApuConnect`.
    unsafe {
        call_bridge(OP_XAPU_DISCONNECT, conn as usize as u64, 0, 0, 0, 0, 0)
            .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XApuEnqueueCommand(
    conn: *mut c_void,
    cmd: *const c_void,
    out_ticket: *mut u32,
) -> i32 {
    // SAFETY: `conn` is a valid connection handle and `out_ticket` is optional writable output.
    unsafe {
        call_bridge(
            OP_XAPU_ENQUEUE_COMMAND,
            conn as usize as u64,
            cmd as usize as u64,
            out_ticket as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XApuDequeueResult(conn: *mut c_void, out_result: *mut c_void) -> i32 {
    // SAFETY: `conn` is a valid connection handle and `out_result` is optional writable output.
    unsafe {
        call_bridge(
            OP_XAPU_DEQUEUE_RESULT,
            conn as usize as u64,
            out_result as usize as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XApuGetResultCount(conn: *mut c_void, out_count: *mut u32) -> u32 {
    // SAFETY: `conn` is a valid connection handle and `out_count` is optional writable output.
    unsafe {
        call_bridge(
            OP_XAPU_GET_RESULT_COUNT,
            conn as usize as u64,
            out_count as usize as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as u32
    }
}
