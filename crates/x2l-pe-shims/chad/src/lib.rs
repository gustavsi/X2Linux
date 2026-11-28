//! Thin Win64 PE shim for `chad.dll` (5 DSP stream exports) forwarding into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_CHAD_ACTIVATE_STATUS, OP_CHAD_CONNECT_STREAM_LIMIT, OP_CHAD_DEACTIVATE,
    OP_CHAD_DECOMPRESS, OP_CHAD_DISCONNECT, OP_CHAD_SUBMIT_STATUS, call_bridge, handle_dll_main,
    record_stub,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDspConnectWithMaximumStreamLimit(
    ctx: *mut c_void,
    max_streams: u32,
    arg2: u32,
    arg3: u32,
    arg4: u32,
    out_dsp: *mut *mut c_void,
) -> i32 {
    // SAFETY: `out_dsp` is the caller's writable DSP handle output pointer.
    unsafe {
        call_bridge(
            OP_CHAD_CONNECT_STREAM_LIMIT,
            ctx as usize as u64,
            u64::from(max_streams),
            u64::from(arg2),
            u64::from(arg3),
            u64::from(arg4),
            out_dsp as usize as u64,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDspDisconnect(dsp: *mut c_void) -> i32 {
    // SAFETY: Disconnects handle previously returned by `XDspConnectWithMaximumStreamLimit`.
    unsafe {
        call_bridge(OP_CHAD_DISCONNECT, dsp as usize as u64, 0, 0, 0, 0, 0)
            .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDspActivate(
    dsp: *mut c_void,
    stream_desc: *const c_void,
    out_status: *mut *const c_void,
    out_stream: *mut *mut c_void,
) -> i32 {
    // SAFETY: `out_stream` is the caller's writable DSP stream handle output pointer.
    unsafe {
        call_bridge(
            OP_CHAD_ACTIVATE_STATUS,
            dsp as usize as u64,
            stream_desc as usize as u64,
            out_status as usize as u64,
            out_stream as usize as u64,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[link(name = "kernel32")]
// SAFETY: SetEvent uses the Win32 Kernel32 signature.
unsafe extern "system" {
    fn SetEvent(event: *mut c_void) -> i32;
}

/// X2Linux decompression command with synchronous stream completion.
#[repr(C)]
pub struct ChadDecompressCommand {
    pub codec: u32,
    pub source_len: u32,
    pub source: *const u8,
    pub destination_len: u32,
    pub destination: *mut u8,
    pub bytes_written: *mut u32,
}

/// # Safety
/// The command and its buffers remain live through the call. The optional event is a Win32 handle.
#[unsafe(no_mangle)]
pub unsafe extern "win64" fn X2LChadDecompress(
    stream: *mut c_void,
    command: *const ChadDecompressCommand,
    event: *mut c_void,
    out_ticket: *mut u32,
) -> i32 {
    let mut ticket = 0;
    // SAFETY: The caller supplies a live descriptor and the bridge writes the local ticket.
    let hr = unsafe {
        call_bridge(
            OP_CHAD_DECOMPRESS,
            stream as u64,
            command as u64,
            (&mut ticket as *mut u32) as u64,
            0,
            0,
            0,
        )
    }
    .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32;
    if !out_ticket.is_null() {
        // SAFETY: The caller provides an optional writable ticket.
        unsafe { out_ticket.write_unaligned(ticket) };
    }
    if ticket != 0 && !event.is_null() {
        // SAFETY: The non-null handle belongs to the caller and is live through this call.
        unsafe { SetEvent(event) };
    }
    hr
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDspDeactivate(stream: *mut c_void) -> i32 {
    // SAFETY: Deactivates handle previously returned by `XDspActivate`.
    unsafe {
        call_bridge(OP_CHAD_DEACTIVATE, stream as usize as u64, 0, 0, 0, 0, 0)
            .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDspSubmitCommand(
    stream: *mut c_void,
    cmd: *const c_void,
    out_ticket: *mut u32,
) -> i32 {
    // SAFETY: `stream` is a live DSP stream handle and `out_ticket` is optional writable output.
    let hr = unsafe {
        call_bridge(
            OP_CHAD_SUBMIT_STATUS,
            stream as usize as u64,
            cmd as usize as u64,
            out_ticket as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    };
    if hr == HRESULT_E_NOTIMPL {
        record_stub("chad.dll", "XDspSubmitCommand", hr as u32 as u64);
    }
    hr
}
