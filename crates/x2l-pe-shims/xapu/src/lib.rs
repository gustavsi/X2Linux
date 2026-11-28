//! Thin Win64 PE shim for `xapu.dll` (5 exports) forwarding into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_XAPU_CONNECT_DECODER, OP_XAPU_DISCONNECT_DECODER, OP_XAPU_POP,
    OP_XAPU_RESULT_COUNTS, OP_XAPU_SUBMIT, call_bridge, handle_dll_main,
};

#[link(name = "kernel32")]
// SAFETY: The declarations match the Win32 Kernel32 signatures.
unsafe extern "system" {
    fn CreateEventW(
        attributes: *const c_void,
        manual: i32,
        initial: i32,
        name: *const u16,
    ) -> *mut c_void;
    fn SetEvent(event: *mut c_void) -> i32;
    fn CloseHandle(handle: *mut c_void) -> i32;
}

extern "win64" fn signal_event(event: usize) {
    // SAFETY: The native queue holds the event open until all submissions complete.
    unsafe { SetEvent(event as *mut c_void) };
}

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
/// # Safety
/// Input and output must supply the XApu descriptors when non-null.
pub unsafe extern "win64" fn XApuConnect(
    input: *const c_void,
    output: *mut c_void,
    out_conn: *mut *mut c_void,
) -> i32 {
    let mut event = std::ptr::null_mut();
    if !input.is_null() && !output.is_null() {
        // SAFETY: The XApu input descriptor has options in its sixth 32-bit word.
        let options = unsafe { input.cast::<u32>().add(5).read_unaligned() };
        if options & 8 != 0 {
            // SAFETY: An unnamed, initially unsignalled, auto-reset Win32 event.
            event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
            if event.is_null() {
                return 0x8007_000eu32 as i32;
            }
        }
    }
    // SAFETY: `out_conn` is the caller's writable connection handle output pointer.
    let hr = unsafe {
        call_bridge(
            OP_XAPU_CONNECT_DECODER,
            input as usize as u64,
            output as usize as u64,
            out_conn as usize as u64,
            event as u64,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    };
    if hr < 0 && !event.is_null() {
        // SAFETY: The connection did not take ownership of the newly opened event.
        unsafe { CloseHandle(event) };
    }
    hr
}

#[unsafe(no_mangle)]
pub extern "win64" fn XApuDisconnect(conn: *mut c_void) -> i32 {
    let mut event = 0usize;
    // SAFETY: Disconnects handle previously returned by `XApuConnect`.
    let hr = unsafe {
        call_bridge(
            OP_XAPU_DISCONNECT_DECODER,
            conn as u64,
            (&mut event as *mut usize) as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    };
    if event != 0 {
        // SAFETY: The bridge returned ownership of this event after disconnecting the queue.
        unsafe { CloseHandle(event as *mut c_void) };
    }
    hr
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
            OP_XAPU_SUBMIT,
            conn as usize as u64,
            cmd as usize as u64,
            out_ticket as usize as u64,
            signal_event as *const () as u64,
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
            OP_XAPU_POP,
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
            OP_XAPU_RESULT_COUNTS,
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
