//! Thin Win64 PE shim for `AcpHal.dll` (9 audio co-processor exports) forwarding into `libx2l_core.so`.

use std::ffi::c_void;
use x2l_shim_common::{
    HRESULT_E_NOTIMPL, OP_ACPHAL_ALLOC_SHAPE_CONTEXTS, OP_ACPHAL_CREATE,
    OP_ACPHAL_RELEASE_SHAPE_CONTEXTS, OP_APU_ALLOC, OP_APU_CREATE_HEAP, OP_APU_FREE,
    OP_APU_IS_VIRTUAL_ADDRESS_VALID, OP_APU_MAP_APU_ADDRESS, OP_APU_MAP_VIRTUAL_ADDRESS,
    call_bridge, handle_dll_main,
};

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn AcpHalCreate(out_acp: *mut *mut c_void) -> i32 {
    // SAFETY: `out_acp` is the caller's writable `IAcpHal**` output pointer.
    unsafe {
        call_bridge(OP_ACPHAL_CREATE, out_acp as usize as u64, 0, 0, 0, 0, 0)
            .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn AcpHalAllocateShapeContexts(contexts: *mut c_void) -> i32 {
    // SAFETY: Bridge marks shape contexts active.
    unsafe {
        call_bridge(
            OP_ACPHAL_ALLOC_SHAPE_CONTEXTS,
            contexts as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn AcpHalReleaseShapeContexts() -> i32 {
    // SAFETY: Bridge releases shape contexts.
    unsafe {
        call_bridge(OP_ACPHAL_RELEASE_SHAPE_CONTEXTS, 0, 0, 0, 0, 0, 0)
            .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn ApuCreateHeap(cached_bytes: u32, noncached_bytes: u32) -> i32 {
    // SAFETY: Bridge initializes APU heap parameters.
    unsafe {
        call_bridge(
            OP_APU_CREATE_HEAP,
            u64::from(cached_bytes),
            u64::from(noncached_bytes),
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn ApuAlloc(
    out_va: *mut *mut c_void,
    out_apu_addr: *mut u32,
    size: u32,
    alignment: u32,
    flags: u32,
) -> i32 {
    // SAFETY: `out_va` and optional `out_apu_addr` are caller-provided output pointers.
    unsafe {
        call_bridge(
            OP_APU_ALLOC,
            out_va as usize as u64,
            out_apu_addr as usize as u64,
            u64::from(size),
            u64::from(alignment),
            u64::from(flags),
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn ApuFree(va: *mut c_void) -> i32 {
    // SAFETY: Frees pointer previously returned by `ApuAlloc`.
    unsafe { call_bridge(OP_APU_FREE, va as usize as u64, 0, 0, 0, 0, 0).unwrap_or(0) as i32 }
}

#[unsafe(no_mangle)]
pub extern "win64" fn ApuMapApuAddress(apu_addr: u32) -> *mut c_void {
    // SAFETY: Returns mapped virtual address owned by the APU heap table.
    let addr = unsafe {
        call_bridge(OP_APU_MAP_APU_ADDRESS, u64::from(apu_addr), 0, 0, 0, 0, 0).unwrap_or(0)
    };
    addr as usize as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "win64" fn ApuMapVirtualAddress(va: *const c_void) -> u32 {
    // SAFETY: Pure lookup in APU heap address map.
    unsafe {
        call_bridge(
            OP_APU_MAP_VIRTUAL_ADDRESS,
            va as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as u32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn ApuIsVirtualAddressValid(va: *const c_void, size: u32) -> i32 {
    // SAFETY: Pure range validity check in APU heap table.
    unsafe {
        call_bridge(
            OP_APU_IS_VIRTUAL_ADDRESS_VALID,
            va as usize as u64,
            u64::from(size),
            0,
            0,
            0,
            0,
        )
        .unwrap_or(0) as i32
    }
}
