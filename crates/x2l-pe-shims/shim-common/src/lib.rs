//! Shared Win64 PE-side bridge client, caller RVA capture, and dynamic W^X x86-64 auto-stub
//! thunk generator for `crates/x2l-pe-shims`.

use std::ffi::c_void;
use std::sync::OnceLock;

/// Magic value returned by `OP_HANDSHAKE` confirming ABI v1 compatibility with `libx2l_core.so`.
pub const X2L_BRIDGE_MAGIC: u64 = 0x5832_4C5F_0001_0001;

pub const HRESULT_S_OK: i32 = 0;
pub const HRESULT_E_NOTIMPL: i32 = -2147467263; // 0x80004001

// Shared opcodes matching `x2l_common::bridge`.
pub const OP_HANDSHAKE: u32 = 0x0001;
pub const OP_FLUSH_REPORT: u32 = 0x0002;
pub const OP_RECORD_STUB: u32 = 0x0010;
pub const OP_RECORD_UNKNOWN_ORDINAL: u32 = 0x0011;
pub const OP_RECORD_VTABLE_PROBE: u32 = 0x0013;

pub const OP_KERNELX_GET_CONSOLE_TYPE: u32 = 0x0100;
pub const OP_KERNELX_GET_DEVICE_TYPE: u32 = 0x0101;
pub const OP_KERNELX_XMEM_ALLOC: u32 = 0x0102;
pub const OP_KERNELX_XMEM_FREE: u32 = 0x0103;
pub const OP_KERNELX_TITLE_MEMORY_STATUS: u32 = 0x0104;
pub const OP_KERNELX_VIRTUAL_ALLOC: u32 = 0x0105;

pub const OP_XMEM_ALLOC_DEFAULT: u32 = 0x0110;
pub const OP_XMEM_FREE_DEFAULT: u32 = 0x0111;
pub const OP_XMEM_ALLOC_PHYSICAL_PAGES: u32 = 0x0112;
pub const OP_XMEM_FREE_PHYSICAL_PAGES: u32 = 0x0113;
pub const OP_XMEM_MAP_PHYSICAL_PAGES: u32 = 0x0114;
pub const OP_XMEM_MAP_PHYSICAL_PAGES_SCATTER: u32 = 0x0115;
pub const OP_XMEM_VIRTUAL_ALLOC: u32 = 0x0116;
pub const OP_XMEM_VIRTUAL_QUERY: u32 = 0x0117;
pub const OP_XMEM_TRANSFER_MEMORY: u32 = 0x0118;
pub const OP_XMEM_GET_WORKING_SET_STATS: u32 = 0x0119;
pub const OP_XMEM_IS_TOOLS_MEM_AVAILABLE: u32 = 0x011A;

pub const OP_XGR_INITIALIZE: u32 = 0x0200;
pub const OP_XGR_UNINITIALIZE: u32 = 0x0201;
pub const OP_XGR_USER_GET_GAMERTAG: u32 = 0x0202;
pub const OP_XGR_PACKAGE_MOUNT_ROOT: u32 = 0x0203;
pub const OP_XGR_GAMESAVE_ROOT: u32 = 0x0204;
pub const OP_XGR_STORE_OFFLINE: u32 = 0x0205;

pub const OP_D3D12X_CREATE_DEVICE: u32 = 0x0300;
pub const OP_D3D12X_XBOX_CREATE_DEVICE: u32 = 0x0302;
pub const OP_D3D12X_CONFIGURE_VIRTUAL_MEMORY: u32 = 0x0303;
pub const OP_D3D12X_SERIALIZE_VERSIONED_ROOT_SIG: u32 = 0x0304;

pub const OP_XG_CREATE_TEXTURE_COMPUTER: u32 = 0x0400;
pub const OP_XG_SUGGEST_SWIZZLE_MODE: u32 = 0x0401;

pub const OP_DSTORAGE_GET_FACTORY: u32 = 0x0500;

pub const OP_GAMEINPUT_CREATE: u32 = 0x0600;

pub const OP_ACPHAL_CREATE: u32 = 0x0700;
pub const OP_ACPHAL_ALLOC_SHAPE_CONTEXTS: u32 = 0x0701;
pub const OP_ACPHAL_RELEASE_SHAPE_CONTEXTS: u32 = 0x0702;
pub const OP_APU_CREATE_HEAP: u32 = 0x0703;
pub const OP_APU_ALLOC: u32 = 0x0704;
pub const OP_APU_FREE: u32 = 0x0705;
pub const OP_APU_MAP_APU_ADDRESS: u32 = 0x0706;
pub const OP_APU_MAP_VIRTUAL_ADDRESS: u32 = 0x0707;
pub const OP_APU_IS_VIRTUAL_ADDRESS_VALID: u32 = 0x0708;

pub const OP_XAPU_CONNECT: u32 = 0x0710;
pub const OP_XAPU_DISCONNECT: u32 = 0x0711;
pub const OP_XAPU_ENQUEUE_COMMAND: u32 = 0x0712;
pub const OP_XAPU_DEQUEUE_RESULT: u32 = 0x0713;
pub const OP_XAPU_GET_RESULT_COUNT: u32 = 0x0714;

pub const OP_CHAD_CONNECT: u32 = 0x0720;
pub const OP_CHAD_DISCONNECT: u32 = 0x0721;
pub const OP_CHAD_ACTIVATE: u32 = 0x0722;
pub const OP_CHAD_DEACTIVATE: u32 = 0x0723;
pub const OP_CHAD_SUBMIT_COMMAND: u32 = 0x0724;

pub const OP_PSM_REGISTER_APP_STATE: u32 = 0x0800;
pub const OP_PSM_UNREGISTER_APP_STATE: u32 = 0x0801;
pub const OP_PSM_REGISTER_CONSTRAINED: u32 = 0x0802;
pub const OP_PSM_UNREGISTER_CONSTRAINED: u32 = 0x0803;

/// Single version-independent `extern "win64"` dispatch pointer published by `libx2l_core.so`.
pub type BridgeFn = unsafe extern "win64" fn(
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) -> u64;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetEnvironmentVariableA(name: *const u8, buf: *mut u8, size: u32) -> u32;
    fn GetModuleHandleA(module_name: *const u8) -> usize;
    fn RtlCaptureStackBackTrace(
        frames_to_skip: u32,
        frames_to_capture: u32,
        back_trace: *mut *mut c_void,
        back_trace_hash: *mut u32,
    ) -> u16;
    pub fn VirtualAlloc(
        addr: *mut c_void,
        size: usize,
        alloc_type: u32,
        protect: u32,
    ) -> *mut c_void;
    fn VirtualProtect(
        addr: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;

static BRIDGE_FN: OnceLock<Option<BridgeFn>> = OnceLock::new();

fn resolve_bridge() -> Option<BridgeFn> {
    *BRIDGE_FN.get_or_init(|| {
        let mut buf = [0u8; 64];
        // SAFETY: Valid NUL-terminated ASCII name and stack-allocated writable buffer.
        let len = unsafe {
            GetEnvironmentVariableA(
                b"X2L_BRIDGE_FN\0".as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as u32,
            )
        } as usize;
        if len == 0 || len >= buf.len() {
            return None;
        }
        let s = std::str::from_utf8(&buf[..len]).ok()?.trim();
        let hex = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
        let addr = usize::from_str_radix(hex, 16).ok()?;
        if addr == 0 {
            return None;
        }
        // SAFETY: `X2L_BRIDGE_FN` is published in this process's `.init_array` by `libx2l_core.so`
        // pointing directly to `x2l_bridge_dispatch` (`extern "win64"`). We pass the primary
        // executable's ImageBase (`GetModuleHandleA(NULL)`) and verify the ABI magic immediately.
        let exe_base = unsafe { GetModuleHandleA(std::ptr::null()) } as u64;
        let func: BridgeFn = unsafe { std::mem::transmute(addr) };
        let magic = unsafe { func(OP_HANDSHAKE, exe_base, 0, 0, 0, 0, 0) };
        (magic == X2L_BRIDGE_MAGIC).then_some(func)
    })
}

/// Call into in-process `libx2l_core.so` via the `extern "win64"` bridge.
///
/// # Safety
/// Caller must pass valid arguments matching `opcode`.
pub unsafe fn call_bridge(
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) -> Option<u64> {
    let func = resolve_bridge()?;
    // SAFETY: Validated by `OP_HANDSHAKE` magic and caller guarantees opcode argument invariants.
    Some(unsafe { func(opcode, a0, a1, a2, a3, a4, a5) })
}

/// Capture the caller's RVA relative to the primary executable's `ImageBase` (or raw RIP if
/// called from another module).
#[inline(never)]
pub fn capture_caller_rva(frames_to_skip: u32) -> u64 {
    let mut frames = [std::ptr::null_mut::<c_void>(); 8];
    // SAFETY: `frames` is a valid writable stack array; `GetModuleHandleA(NULL)` returns the
    // primary executable's ImageBase in this process.
    let captured = unsafe {
        RtlCaptureStackBackTrace(
            frames_to_skip,
            frames.len() as u32,
            frames.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    } as usize;
    if captured == 0 {
        return 0;
    }
    let exe_base = unsafe { GetModuleHandleA(std::ptr::null()) };
    if exe_base != 0 {
        for &frame in &frames[..captured] {
            let rip = frame as usize;
            if rip >= exe_base && (rip - exe_base) < 0x1000_0000 {
                return (rip - exe_base) as u64;
            }
        }
    }
    frames[0] as usize as u64
}

/// Log and count an unimplemented named export, returning `default_ret`.
#[inline(never)]
pub fn record_stub(dll: &str, symbol: &str, default_ret: u64) -> u64 {
    let caller_rva = capture_caller_rva(2);
    // SAFETY: `dll` and `symbol` are valid UTF-8 slices alive for the duration of the call.
    unsafe {
        call_bridge(
            OP_RECORD_STUB,
            dll.as_ptr() as usize as u64,
            dll.len() as u64,
            symbol.as_ptr() as usize as u64,
            symbol.len() as u64,
            caller_rva,
            default_ret,
        )
        .unwrap_or(default_ret)
    }
}

/// Log and count an unknown ordinal export, returning `default_ret`.
#[inline(never)]
pub fn record_unknown_ordinal(dll: &str, ordinal: u16, default_ret: u64) -> u64 {
    let caller_rva = capture_caller_rva(2);
    // SAFETY: `dll` is a valid UTF-8 slice alive for the duration of the call.
    unsafe {
        call_bridge(
            OP_RECORD_UNKNOWN_ORDINAL,
            dll.as_ptr() as usize as u64,
            dll.len() as u64,
            u64::from(ordinal),
            caller_rva,
            default_ret,
            0,
        )
        .unwrap_or(default_ret)
    }
}

/// Flush telemetry on `DLL_PROCESS_DETACH` (`reason == 0`).
pub fn handle_dll_main(reason: u32) -> i32 {
    if reason == 0 {
        // SAFETY: `OP_FLUSH_REPORT` takes no pointers.
        let _ = unsafe { call_bridge(OP_FLUSH_REPORT, 0, 0, 0, 0, 0, 0) };
    }
    1
}

/// Descriptor owned by a dynamically emitted Win64 auto-stub thunk.
#[repr(C)]
pub struct StubThunkInfo {
    pub dll: &'static str,
    pub symbol: Option<&'static str>,
    pub ordinal: u16,
    pub default_ret: u64,
}

extern "win64" fn dynamic_stub_dispatch(info: *const StubThunkInfo, caller_rip: u64) -> u64 {
    if info.is_null() {
        return 0;
    }
    // SAFETY: `info` points to a `'static` `StubThunkInfo` embedded in the thunk.
    let info = unsafe { &*info };
    let exe_base = unsafe { GetModuleHandleA(std::ptr::null()) } as u64;
    let caller_rva =
        if exe_base != 0 && caller_rip >= exe_base && (caller_rip - exe_base) < 0x1000_0000 {
            caller_rip - exe_base
        } else {
            caller_rip
        };
    match info.symbol {
        Some(sym) => unsafe {
            call_bridge(
                OP_RECORD_STUB,
                info.dll.as_ptr() as usize as u64,
                info.dll.len() as u64,
                sym.as_ptr() as usize as u64,
                sym.len() as u64,
                caller_rva,
                info.default_ret,
            )
            .unwrap_or(info.default_ret)
        },
        None => unsafe {
            call_bridge(
                OP_RECORD_UNKNOWN_ORDINAL,
                info.dll.as_ptr() as usize as u64,
                info.dll.len() as u64,
                u64::from(info.ordinal),
                caller_rva,
                info.default_ret,
                0,
            )
            .unwrap_or(info.default_ret)
        },
    }
}

/// Dynamically emit a W^X-protected 26-byte x86-64 Win64 tail-call auto-stub thunk:
/// ```asm
/// mov rdx, [rsp]          ; 48 8B 14 24
/// mov rcx, <info_ptr>     ; 48 B9 <u64>
/// mov rax, <dispatch_fn>  ; 48 B8 <u64>
/// jmp rax                 ; FF E0
/// ```
pub fn alloc_dynamic_stub_thunk(info: &'static StubThunkInfo) -> Option<usize> {
    // SAFETY: Allocate RW page, encode tail-call thunk, then flip strictly to RX (W^X).
    unsafe {
        let mem = VirtualAlloc(
            std::ptr::null_mut(),
            4096,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        if mem.is_null() {
            return None;
        }
        let buf = std::slice::from_raw_parts_mut(mem as *mut u8, 32);
        buf.fill(0xCC);
        buf[0..4].copy_from_slice(&[0x48, 0x8B, 0x14, 0x24]);
        buf[4..6].copy_from_slice(&[0x48, 0xB9]);
        buf[6..14].copy_from_slice(&(info as *const StubThunkInfo as usize as u64).to_le_bytes());
        buf[14..16].copy_from_slice(&[0x48, 0xB8]);
        buf[16..24]
            .copy_from_slice(&(dynamic_stub_dispatch as *const () as usize as u64).to_le_bytes());
        buf[24..26].copy_from_slice(&[0xFF, 0xE0]);

        let mut old_prot = 0u32;
        if VirtualProtect(mem, 4096, PAGE_EXECUTE_READ, &mut old_prot) == 0 {
            return None;
        }
        Some(mem as usize)
    }
}
