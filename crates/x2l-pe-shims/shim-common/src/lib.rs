//! Shared Win64 PE-side bridge client, caller RVA capture, arena-aware `VirtualAlloc`/`Free`/`Query`
//! + `GetProcAddress` + `OutputDebugStringA/W` PE32+ IAT patcher, fatal VEH context capture,
//! and dynamic W^X x86-64 auto-stub thunk generator for `crates/x2l-pe-shims`.

#[path = "../../../x2l-common/src/bridge/exception.rs"]
mod exception;
#[path = "../../../x2l-hle-kernelx/src/valloc_flags.rs"]
pub mod valloc_flags;

use exception::{CrashSnapshot, CrashSnapshotV2};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Magic value returned by `OP_HANDSHAKE` confirming ABI v1 compatibility with `libx2l_core.so`.
pub const X2L_BRIDGE_MAGIC: u64 = 0x5832_4C5F_0001_0001;

pub const HRESULT_S_OK: i32 = 0;
pub const HRESULT_E_NOTIMPL: i32 = -2147467263; // 0x80004001
pub const HRESULT_E_INVALIDARG: i32 = -2147024809; // 0x80070057
pub const HRESULT_E_POINTER: i32 = -2147467261; // 0x80004003
pub const HRESULT_E_ABORT: i32 = -2147467260; // 0x80004004
pub const HRESULT_E_PENDING: i32 = -2147483638; // 0x8000000A

// Shared opcodes matching `x2l_common::bridge`.
pub const OP_HANDSHAKE: u32 = 0x0001;
pub const OP_FLUSH_REPORT: u32 = 0x0002;
pub const OP_RECORD_STUB: u32 = 0x0010;
pub const OP_RECORD_UNKNOWN_ORDINAL: u32 = 0x0011;
pub const OP_RECORD_VTABLE_PROBE: u32 = 0x0013;
pub const OP_TRACE_HLE_LINE: u32 = 0x0014;
pub const OP_RECORD_EXCEPTION: u32 = 0x0015;
pub const OP_RECORD_DEBUG_STRING: u32 = 0x0016;
pub const OP_RECORD_MISSING_PROC: u32 = 0x0017;

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
pub const OP_XMEM_TRY_WIN32_VALLOC: u32 = 0x011B;
pub const OP_XMEM_TRY_WIN32_VFREE: u32 = 0x011C;
pub const OP_XMEM_TRY_WIN32_VQUERY: u32 = 0x011D;

pub const OP_XGR_INITIALIZE: u32 = 0x0200;
pub const OP_XGR_UNINITIALIZE: u32 = 0x0201;
pub const OP_XGR_USER_GET_GAMERTAG: u32 = 0x0202;
pub const OP_XGR_PACKAGE_MOUNT_ROOT: u32 = 0x0203;
pub const OP_XGR_GAMESAVE_ROOT: u32 = 0x0204;
pub const OP_XGR_STORE_OFFLINE: u32 = 0x0205;
pub const OP_XGR_GET_DEVICE_TYPE: u32 = 0x0206;
pub const OP_XGR_GET_TITLE_ID: u32 = 0x0207;
pub const OP_XGR_GET_CONSOLE_ID: u32 = 0x0208;
pub const OP_XGR_GET_SANDBOX_ID: u32 = 0x0209;
pub const OP_XGR_GET_APP_DEVICE_ID: u32 = 0x020A;
pub const OP_XGR_PLS_GET_PATH: u32 = 0x020B;
pub const OP_XGR_PLS_GET_SPACE_INFO: u32 = 0x020C;
pub const OP_XGR_GAMESAVE_WRITE_BLOB: u32 = 0x020D;
pub const OP_XGR_GAMESAVE_READ_BLOB: u32 = 0x020E;
pub const OP_XGR_GAMESAVE_ENUM_BLOBS: u32 = 0x020F;

pub const OP_D3D12X_CREATE_DEVICE: u32 = 0x0300;
pub const OP_D3D12X_XBOX_CREATE_DEVICE: u32 = 0x0302;
pub const OP_D3D12X_CONFIGURE_VIRTUAL_MEMORY: u32 = 0x0303;
pub const OP_D3D12X_SERIALIZE_VERSIONED_ROOT_SIG: u32 = 0x0304;

pub const OP_XG_CREATE_TEXTURE_COMPUTER: u32 = 0x0400;
pub const OP_XG_SUGGEST_SWIZZLE_MODE: u32 = 0x0401;
pub const OP_XG_CREATE_TYPED_COMPUTER: u32 = 0x0402;
pub const OP_XG_COMPUTE_OPTIMAL_TILE_MODE: u32 = 0x0403;

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
pub const OP_CHAD_DECOMPRESS: u32 = 0x0725;
pub const OP_XAPU_CONNECT_DECODER: u32 = 0x0715;
pub const OP_XAPU_DISCONNECT_DECODER: u32 = 0x0716;
pub const OP_XAPU_SUBMIT: u32 = 0x0717;
pub const OP_XAPU_POP: u32 = 0x0718;
pub const OP_XAPU_RESULT_COUNTS: u32 = 0x0719;
pub const OP_CHAD_CONNECT_STREAM_LIMIT: u32 = 0x0726;
pub const OP_CHAD_ACTIVATE_STATUS: u32 = 0x0727;
pub const OP_CHAD_SUBMIT_STATUS: u32 = 0x0728;

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
    caller_rva: u64,
) -> u64;

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetEnvironmentVariableA(name: *const u8, buf: *mut u8, size: u32) -> u32;
    fn SetEnvironmentVariableA(name: *const u8, value: *const u8) -> i32;
    pub fn GetModuleHandleA(module_name: *const u8) -> usize;
    fn GetModuleHandleExA(flags: u32, module_name: *const c_void, out_mod: *mut usize) -> i32;
    fn GetModuleFileNameA(module: usize, buf: *mut u8, size: u32) -> u32;
    fn GetProcAddress(module: usize, proc_name: *const u8) -> usize;
    fn GetCurrentProcess() -> *mut c_void;
    fn GetCurrentProcessId() -> u32;
    fn GetProcessId(process: *mut c_void) -> u32;
    fn GetLastError() -> u32;
    fn SetLastError(err_code: u32);
    fn OutputDebugStringA(msg: *const u8);
    fn OutputDebugStringW(msg: *const u16);
    fn AddVectoredExceptionHandler(
        first: u32,
        handler: unsafe extern "win64" fn(*mut ExceptionPointers) -> i32,
    ) -> *mut c_void;
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
    pub fn VirtualAllocEx(
        process: *mut c_void,
        addr: *mut c_void,
        size: usize,
        alloc_type: u32,
        protect: u32,
    ) -> *mut c_void;
    pub fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
    pub fn VirtualQuery(addr: *const c_void, buf: *mut c_void, len: usize) -> usize;
    pub fn VirtualQueryEx(
        process: *mut c_void,
        addr: *const c_void,
        buf: *mut c_void,
        len: usize,
    ) -> usize;
    fn ReadProcessMemory(
        process: *mut c_void,
        base_address: *const c_void,
        buffer: *mut c_void,
        size: usize,
        bytes_read: *mut usize,
    ) -> i32;
    fn VirtualProtect(
        addr: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

const ERROR_INVALID_PARAMETER: u32 = 87;
const ERROR_PROC_NOT_FOUND: u32 = 127;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
const GET_MODULE_HANDLE_EX_FLAG_PIN: u32 = 0x0000_0001;
const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

// Win64 x86-64 CONTEXT flag masks (exact Windows/Wine ABI).
const CONTEXT_AMD64: u32 = 0x0010_0000;
const CONTEXT_CONTROL: u32 = CONTEXT_AMD64 | 0x0001;
const CONTEXT_INTEGER: u32 = CONTEXT_AMD64 | 0x0002;

/// Exact Win64 x86-64 `CONTEXT` prefix up through `Rip` (`0x100` bytes).
#[repr(C)]
struct Win64ContextPrefix {
    _pad0: [u8; 0x30],
    context_flags: u32, // @0x30
    _pad1: [u8; 0x44],
    rax: u64, // @0x78
    rcx: u64, // @0x80
    rdx: u64, // @0x88
    rbx: u64, // @0x90
    rsp: u64, // @0x98
    rbp: u64, // @0xA0
    rsi: u64, // @0xA8
    rdi: u64, // @0xB0
    r8: u64,  // @0xB8
    r9: u64,  // @0xC0
    _pad2: [u8; 0x30],
    rip: u64, // @0xF8
}

const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, context_flags) == 0x30);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, rax) == 0x78);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, rcx) == 0x80);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, rdx) == 0x88);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, rsp) == 0x98);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, r8) == 0xB8);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, r9) == 0xC0);
const _: () = assert!(std::mem::offset_of!(Win64ContextPrefix, rip) == 0xF8);

/// Win64 `MEMORY_BASIC_INFORMATION` returned by `VirtualQuery`.
#[repr(C)]
#[derive(Default)]
struct MemoryBasicInformation {
    base_address: usize,
    allocation_base: usize,
    allocation_protect: u32,
    partition_id: u16,
    _pad0: u16,
    region_size: usize,
    state: u32,
    protect: u32,
    mem_type: u32,
    _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<MemoryBasicInformation>() == 48);

fn capture_rip_bytes(rip: usize, bytes: &mut [u8; 16]) -> u32 {
    let addr = rip as *const c_void;
    let mut mbi = MemoryBasicInformation::default();
    let mbi_size = std::mem::size_of::<MemoryBasicInformation>();
    // SAFETY: VirtualQuery accepts an arbitrary address and a writable buffer of mbi_size bytes.
    if unsafe { VirtualQuery(addr, std::ptr::from_mut(&mut mbi).cast(), mbi_size) } != mbi_size {
        return 0;
    }
    if mbi.state != MEM_COMMIT || !valloc_flags::is_cpu_readable_protect(mbi.protect) {
        return 0;
    }
    let Some(offset) = rip.checked_sub(mbi.base_address) else {
        return 0;
    };
    let len = mbi.region_size.saturating_sub(offset).min(bytes.len());
    if len == 0 {
        return 0;
    }
    let mut copied = 0;
    // SAFETY: The destination and count are writable stack storage. ReadProcessMemory handles
    // races with changes to the source mapping without a raw pointer read by the VEH.
    if unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            addr,
            bytes.as_mut_ptr().cast(),
            len,
            &mut copied,
        )
    } != 0
    {
        copied.min(len) as u32
    } else {
        0
    }
}

#[repr(C)]
struct ExceptionRecord {
    exception_code: u32,
    exception_flags: u32,
    exception_record: *mut ExceptionRecord,
    exception_address: *mut c_void,
    number_parameters: u32,
    _pad: u32,
    exception_information: [usize; 15],
}

#[repr(C)]
struct ExceptionPointers {
    exception_record: *mut ExceptionRecord,
    context_record: *mut c_void,
}

static TRACE_HLE_ENABLED: OnceLock<bool> = OnceLock::new();
static VEH_INSTALLED: AtomicBool = AtomicBool::new(false);
static VEH_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
static CACHED_EXE_BASE: AtomicUsize = AtomicUsize::new(0);
static CACHED_EXE_SIZE: AtomicUsize = AtomicUsize::new(0);

/// Returns `true` when `X2L_TRACE_HLE=1` is enabled for opt-in line-buffered HLE tracing.
pub fn trace_hle_enabled() -> bool {
    *TRACE_HLE_ENABLED.get_or_init(|| {
        let mut buf = [0u8; 16];
        // SAFETY: NUL-terminated ASCII name and writable stack buffer.
        let len = unsafe {
            GetEnvironmentVariableA(
                b"X2L_TRACE_HLE\0".as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as u32,
            )
        } as usize;
        if len == 0 || len >= buf.len() {
            return false;
        }
        std::str::from_utf8(&buf[..len]).is_ok_and(|s| s == "1" || s.eq_ignore_ascii_case("true"))
    })
}

/// Emit a line-buffered `[HLE]` trace entry into `logs/x2linux.log` when `X2L_TRACE_HLE=1`.
pub fn trace_hle_line(msg: &str) {
    if !trace_hle_enabled() {
        return;
    }
    // SAFETY: `msg` is a live UTF-8 slice for the duration of the synchronous bridge call.
    let _ = unsafe {
        call_bridge_raw(
            OP_TRACE_HLE_LINE,
            msg.as_ptr() as usize as u64,
            msg.len() as u64,
            0,
            0,
            0,
            0,
            0,
        )
    };
}

unsafe extern "win64" fn x2l_veh_handler(info: *mut ExceptionPointers) -> i32 {
    if info.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: Windows/Wine passes a valid `EXCEPTION_POINTERS` to registered VEH handlers.
    let (rec, ctx_ptr) = unsafe { ((*info).exception_record, (*info).context_record) };
    if rec.is_null() {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // SAFETY: `rec` is valid for the duration of the exception callback.
    let (code, addr) = unsafe { ((*rec).exception_code, (*rec).exception_address as usize) };

    // Only log & flush on fatal crash codes; never interfere with normal C++/Rust SEH unwinds.
    let is_fatal = matches!(
        code,
        0x8000_0003 // EXCEPTION_BREAKPOINT
        | 0xC000_0005 // EXCEPTION_ACCESS_VIOLATION
        | 0xC000_001D // EXCEPTION_ILLEGAL_INSTRUCTION
        | 0xC000_0094 // EXCEPTION_INT_DIVIDE_BY_ZERO
        | 0xC000_00FD // EXCEPTION_STACK_OVERFLOW
    );
    if !is_fatal {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Nonblocking reentrancy guard so a nested fault inside the handler never recurses.
    if VEH_IN_FLIGHT.swap(true, Ordering::AcqRel) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // SAFETY: GetLastError reads this thread's Win32 error status.
    let last_error = unsafe { GetLastError() };
    let exe_base = CACHED_EXE_BASE.load(Ordering::Acquire);
    let exe_size = CACHED_EXE_SIZE.load(Ordering::Acquire);

    let mut mod_base: usize = 0;
    let mut name_buf = [0u8; 260];
    let mut mod_name: &str = "<unknown>";
    let mut rva: u64 = addr as u64;

    if addr_in_module(addr, exe_base, exe_size) {
        rva = (addr - exe_base) as u64;
        mod_base = exe_base;
    }

    if code != 0xC000_00FD {
        // SAFETY: Query module containing `addr` without incrementing its reference count.
        if unsafe {
            GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                addr as *const c_void,
                &mut mod_base,
            )
        } != 0
            && mod_base != 0
        {
            rva = (addr.saturating_sub(mod_base)) as u64;
            // SAFETY: `mod_base` is a valid module handle and `name_buf` is writable.
            let n = unsafe {
                GetModuleFileNameA(mod_base, name_buf.as_mut_ptr(), name_buf.len() as u32)
            } as usize;
            if n > 0
                && n < name_buf.len()
                && let Ok(full) = std::str::from_utf8(&name_buf[..n])
            {
                mod_name = full.rsplit(['\\', '/']).next().unwrap_or(full);
            }
        }
    }

    // Extract Win64 registers only when the corresponding full `ContextFlags` mask is valid.
    let mut has_control = 0u32;
    let mut has_integer = 0u32;
    let (mut rip, mut rsp, mut rax, mut rcx, mut rdx, mut r8, mut r9) = (0u64, 0, 0, 0, 0, 0, 0);
    if !ctx_ptr.is_null() {
        // SAFETY: `ctx_ptr` points to a Wine/Win64 `CONTEXT` with verified ABI offsets.
        let ctx = unsafe { &*(ctx_ptr.cast::<Win64ContextPrefix>()) };
        if (ctx.context_flags & CONTEXT_CONTROL) == CONTEXT_CONTROL {
            has_control = 1;
            rip = ctx.rip;
            rsp = ctx.rsp;
        }
        if (ctx.context_flags & CONTEXT_INTEGER) == CONTEXT_INTEGER {
            has_integer = 1;
            rax = ctx.rax;
            rcx = ctx.rcx;
            rdx = ctx.rdx;
            r8 = ctx.r8;
            r9 = ctx.r9;
        }
    }

    // Capture up to 8 guest-module return RVAs (skip stack walk on stack overflow).
    let mut bt_rvas = [0u64; 8];
    let mut bt_count = 0usize;
    if addr_in_module(addr, exe_base, exe_size) {
        bt_rvas[0] = (addr - exe_base) as u64;
        bt_count = 1;
    }
    if code != 0xC000_00FD && exe_base != 0 && exe_size != 0 {
        let mut raw_frames = [std::ptr::null_mut::<c_void>(); 32];
        // SAFETY: Stack array `raw_frames` is writable; skip stack walk on `0xC000_00FD`.
        let captured = unsafe {
            RtlCaptureStackBackTrace(
                1,
                raw_frames.len() as u32,
                raw_frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        } as usize;
        for &frame in &raw_frames[..captured.min(raw_frames.len())] {
            if bt_count >= bt_rvas.len() {
                break;
            }
            let pc = frame as usize;
            if addr_in_module(pc, exe_base, exe_size) {
                let frame_rva = (pc - exe_base) as u64;
                if !bt_rvas[..bt_count].contains(&frame_rva) {
                    bt_rvas[bt_count] = frame_rva;
                    bt_count += 1;
                }
            }
        }
    }

    let base = CrashSnapshot {
        code,
        has_control,
        has_integer,
        bt_count: bt_count as u32,
        address: addr as u64,
        rva,
        module_ptr: mod_name.as_ptr() as usize as u64,
        module_len: mod_name.len() as u64,
        rip,
        rsp,
        rax,
        rcx,
        rdx,
        r8,
        r9,
        bt_rvas,
    };
    let mut snap = CrashSnapshotV2 {
        base,
        ..CrashSnapshotV2::default()
    };
    // SAFETY: The fixed record is readable and two parameters must be present before use.
    if code == 0xC000_0005 && unsafe { (*rec).number_parameters } >= 2 {
        // SAFETY: The live exception record advertises both access-violation parameters.
        let (access, address) = unsafe {
            (
                (*rec).exception_information[0],
                (*rec).exception_information[1],
            )
        };
        snap.has_fault = 1;
        snap.fault_access = access as u64;
        snap.fault_address = address as u64;
    }
    if has_control != 0 {
        snap.rip_bytes_len = capture_rip_bytes(rip as usize, &mut snap.rip_bytes);
    }

    // SAFETY: `snap` and `mod_name` remain alive on the stack for the synchronous bridge call.
    let _ = unsafe {
        call_bridge_raw(
            OP_RECORD_EXCEPTION,
            (&snap as *const CrashSnapshotV2) as usize as u64,
            0,
            0,
            0,
            0,
            std::mem::size_of::<CrashSnapshotV2>() as u64,
            rva,
        )
    };

    // SAFETY: Restore the current thread's error status before continuing exception dispatch.
    unsafe { SetLastError(last_error) };
    VEH_IN_FLIGHT.store(false, Ordering::Release);
    // Always continue searching so guest SEH/C++ exception handlers remain unaffected.
    EXCEPTION_CONTINUE_SEARCH
}

/// Install a non-invasive first-chance Vectored Exception Handler once per process.
pub fn install_crash_veh() {
    if VEH_INSTALLED.swap(true, Ordering::AcqRel) {
        return;
    }
    // SAFETY: `x2l_veh_handler` is a static `extern "win64"` function returning `EXCEPTION_CONTINUE_SEARCH`.
    let _ = unsafe { AddVectoredExceptionHandler(1, x2l_veh_handler) };
}

type VirtualAllocFn = unsafe extern "win64" fn(*mut c_void, usize, u32, u32) -> *mut c_void;
type VirtualAllocExFn =
    unsafe extern "win64" fn(*mut c_void, *mut c_void, usize, u32, u32) -> *mut c_void;
type VirtualFreeFn = unsafe extern "win64" fn(*mut c_void, usize, u32) -> i32;
type VirtualQueryFn = unsafe extern "win64" fn(*const c_void, *mut c_void, usize) -> usize;
type VirtualQueryExFn =
    unsafe extern "win64" fn(*mut c_void, *const c_void, *mut c_void, usize) -> usize;
type GetProcAddressFn = unsafe extern "win64" fn(usize, *const u8) -> usize;
type OutputDebugStringAFn = unsafe extern "win64" fn(*const u8);
type OutputDebugStringWFn = unsafe extern "win64" fn(*const u16);

static IAT_PATCH_INSTALLED: AtomicBool = AtomicBool::new(false);
static ORIG_VIRTUAL_ALLOC: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_ALLOC_EX: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_ALLOC_FROM_APP: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_FREE: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_QUERY: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_QUERY_EX: AtomicUsize = AtomicUsize::new(0);
static ORIG_GET_PROC_ADDRESS: AtomicUsize = AtomicUsize::new(0);
static ORIG_OUTPUT_DEBUG_STRING_A: AtomicUsize = AtomicUsize::new(0);
static ORIG_OUTPUT_DEBUG_STRING_W: AtomicUsize = AtomicUsize::new(0);

/// Known console shim DLLs whose unresolved `GetProcAddress` lookups are always logged to
/// `report.json` (`missing_imports` / `unknown_ordinals`).
const CONSOLE_SHIM_DLLS: &[(&str, &[u8])] = &[
    ("kernelx.dll", b"kernelx.dll\0"),
    ("xmem.dll", b"xmem.dll\0"),
    ("xgameruntime.dll", b"xgameruntime.dll\0"),
    ("d3d12_xs.dll", b"d3d12_xs.dll\0"),
    ("d3d12_x.dll", b"d3d12_x.dll\0"),
    ("xg_xs.dll", b"xg_xs.dll\0"),
    ("xcrdapi.dll", b"xcrdapi.dll\0"),
    ("dstorage.dll", b"dstorage.dll\0"),
    ("gameinput.dll", b"gameinput.dll\0"),
    ("acphal.dll", b"acphal.dll\0"),
    ("xapu.dll", b"xapu.dll\0"),
    ("chad.dll", b"chad.dll\0"),
    (
        "api-ms-win-core-psm-appnotify-l1-1-0.dll",
        b"api-ms-win-core-psm-appnotify-l1-1-0.dll\0",
    ),
    (
        "api-ms-win-core-psm-appnotify-l1-1-1.dll",
        b"api-ms-win-core-psm-appnotify-l1-1-1.dll\0",
    ),
    ("twinapi.appcore.dll", b"twinapi.appcore.dll\0"),
];

fn identify_console_shim_module(hmodule: usize) -> Option<&'static str> {
    if hmodule == 0 {
        return None;
    }
    for &(label, cstr) in CONSOLE_SHIM_DLLS {
        // SAFETY: `GetModuleHandleA` checks already-loaded modules without invoking `LoadLibrary`.
        let loaded = unsafe { GetModuleHandleA(cstr.as_ptr()) };
        if loaded != 0 && loaded == hmodule {
            return Some(label);
        }
    }
    None
}

fn is_current_process_handle(process: *mut c_void) -> bool {
    let raw = process as isize;
    if raw == -1 {
        return true; // Pseudo-handle `GetCurrentProcess()`
    }
    if process.is_null() {
        return false;
    }
    // SAFETY: Valid Win32 query comparing process handle PID against current PID.
    unsafe {
        let target_pid = GetProcessId(process);
        target_pid != 0 && target_pid == GetCurrentProcessId()
    }
}

fn resolve_wine_kernelbase_proc(name: &[u8]) -> usize {
    // SAFETY: NUL-terminated ASCII module/proc names passed to Win32 loader lookup (no LoadLibrary).
    unsafe {
        let kb = GetModuleHandleA(b"kernelbase.dll\0".as_ptr());
        if kb != 0 {
            let p = GetProcAddress(kb, name.as_ptr());
            if p != 0 {
                return p;
            }
        }
        let k32 = GetModuleHandleA(b"kernel32.dll\0".as_ptr());
        if k32 != 0 {
            return GetProcAddress(k32, name.as_ptr());
        }
        0
    }
}

fn try_xmem_virtual_alloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> Option<*mut c_void> {
    if addr.is_null() {
        return None;
    }
    let caller_rva = if protect & valloc_flags::PAGE_GUARD != 0 {
        capture_caller_rva(3)
    } else {
        0
    };
    let mut out_val: u64 = 0;
    // SAFETY: `out_val` is a writable stack slot for `OP_XMEM_TRY_WIN32_VALLOC`.
    let rc = unsafe {
        call_bridge_raw(
            OP_XMEM_TRY_WIN32_VALLOC,
            addr as usize as u64,
            size as u64,
            u64::from(alloc_type),
            u64::from(protect),
            (&mut out_val as *mut u64) as usize as u64,
            0,
            caller_rva,
        )
        .unwrap_or(0)
    };
    match rc {
        1 => Some(out_val as usize as *mut c_void),
        2 => {
            // SAFETY: Sets thread-local Win32 last error returned by `XMemArena`.
            unsafe { SetLastError(out_val as u32) };
            Some(std::ptr::null_mut())
        }
        _ => None, // NotOwned -> fall through to Wine
    }
}

fn try_xmem_virtual_free(addr: *mut c_void, size: usize, free_type: u32) -> Option<i32> {
    if addr.is_null() {
        return None;
    }
    let mut err: u32 = 0;
    // SAFETY: `err` is a writable stack slot for `OP_XMEM_TRY_WIN32_VFREE`.
    let rc = unsafe {
        call_bridge_raw(
            OP_XMEM_TRY_WIN32_VFREE,
            addr as usize as u64,
            size as u64,
            u64::from(free_type),
            (&mut err as *mut u32) as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0)
    };
    match rc {
        1 => Some(1),
        2 => {
            // SAFETY: Sets thread-local Win32 last error returned by `XMemArena`.
            unsafe { SetLastError(err) };
            Some(0)
        }
        _ => None,
    }
}

fn try_xmem_virtual_query(
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> (Option<usize>, usize) {
    let mut next_res: u64 = 0;
    // SAFETY: `buf` is caller's `MEMORY_BASIC_INFORMATION` output buffer and `next_res` is on stack.
    let rc = unsafe {
        call_bridge_raw(
            OP_XMEM_TRY_WIN32_VQUERY,
            addr as usize as u64,
            buf as usize as u64,
            len as u64,
            (&mut next_res as *mut u64) as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(0)
    };
    match rc {
        48 => (Some(48), next_res as usize),
        2 => {
            // SAFETY: Sets Win32 `ERROR_INVALID_PARAMETER` when caller buffer < 48 bytes.
            unsafe { SetLastError(ERROR_INVALID_PARAMETER) };
            (Some(0), next_res as usize)
        }
        _ => (None, next_res as usize),
    }
}

/// If Wine returned `MEM_FREE` whose span crosses an `XMemArena` reservation, clamp `RegionSize`.
fn clamp_wine_free_mbi(buf: *mut c_void, ret_len: usize, next_res_base: usize) {
    if ret_len < 48 || buf.is_null() || next_res_base == 0 {
        return;
    }
    // SAFETY: Win64 `MEMORY_BASIC_INFORMATION`: base@0 (u64), region_size@24 (u64), state@32 (u32).
    unsafe {
        let p = buf.cast::<u8>();
        let state = u32::from_le_bytes(*(p.add(32) as *const [u8; 4]));
        if state == 0x0001_0000 {
            // MEM_FREE
            let base = u64::from_le_bytes(*(p as *const [u8; 8])) as usize;
            let region_size = u64::from_le_bytes(*(p.add(24) as *const [u8; 8])) as usize;
            if next_res_base > base && base.saturating_add(region_size) > next_res_base {
                let clamped = (next_res_base - base) as u64;
                std::ptr::copy_nonoverlapping(clamped.to_le_bytes().as_ptr(), p.add(24), 8);
            }
        }
    }
}

fn call_wine_virtual_alloc(
    saved: &AtomicUsize,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let mut ptr = saved.load(Ordering::Acquire);
    if ptr == 0 {
        ptr = resolve_wine_kernelbase_proc(b"VirtualAlloc\0");
        if ptr != 0 {
            let _ = saved.compare_exchange(0, ptr, Ordering::Release, Ordering::Relaxed);
        }
    }
    if ptr != 0 {
        // SAFETY: `ptr` is the validated Wine `kernelbase`/`kernel32` `VirtualAlloc` entrypoint.
        unsafe {
            let f: VirtualAllocFn = std::mem::transmute(ptr);
            f(addr, size, alloc_type, protect)
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!VirtualAlloc` import.
        unsafe { VirtualAlloc(addr, size, alloc_type, protect) }
    }
}

fn call_wine_virtual_alloc_ex(
    saved: &AtomicUsize,
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let mut ptr = saved.load(Ordering::Acquire);
    if ptr == 0 {
        ptr = resolve_wine_kernelbase_proc(b"VirtualAllocEx\0");
        if ptr != 0 {
            let _ = saved.compare_exchange(0, ptr, Ordering::Release, Ordering::Relaxed);
        }
    }
    if ptr != 0 {
        // SAFETY: `ptr` is the validated Wine `kernelbase`/`kernel32` `VirtualAllocEx` entrypoint.
        unsafe {
            let f: VirtualAllocExFn = std::mem::transmute(ptr);
            f(process, addr, size, alloc_type, protect)
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!VirtualAllocEx` import.
        unsafe { VirtualAllocEx(process, addr, size, alloc_type, protect) }
    }
}

fn call_wine_virtual_alloc_from_app(
    saved: &AtomicUsize,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    if valloc_flags::is_executable_cpu_protect(protect) {
        // SAFETY: Sets thread-local Win32 last error to `ERROR_INVALID_PARAMETER` (87).
        unsafe { SetLastError(ERROR_INVALID_PARAMETER) };
        return std::ptr::null_mut();
    }
    let mut ptr = saved.load(Ordering::Acquire);
    if ptr == 0 {
        ptr = resolve_wine_kernelbase_proc(b"VirtualAllocFromApp\0");
        if ptr != 0 {
            let _ = saved.compare_exchange(0, ptr, Ordering::Release, Ordering::Relaxed);
        }
    }
    if ptr != 0 {
        // SAFETY: `ptr` is the validated Wine `kernelbase!VirtualAllocFromApp` entrypoint.
        unsafe {
            let f: VirtualAllocFn = std::mem::transmute(ptr);
            f(addr, size, alloc_type, protect)
        }
    } else {
        call_wine_virtual_alloc(&ORIG_VIRTUAL_ALLOC, addr, size, alloc_type, protect)
    }
}

fn call_wine_virtual_free(addr: *mut c_void, size: usize, free_type: u32) -> i32 {
    let ptr = ORIG_VIRTUAL_FREE.load(Ordering::Acquire);
    if ptr != 0 {
        // SAFETY: `ptr` is the saved Wine `VirtualFree` entrypoint.
        unsafe {
            let f: VirtualFreeFn = std::mem::transmute(ptr);
            f(addr, size, free_type)
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!VirtualFree` import.
        unsafe { VirtualFree(addr, size, free_type) }
    }
}

fn call_wine_virtual_query(addr: *const c_void, buf: *mut c_void, len: usize) -> usize {
    let ptr = ORIG_VIRTUAL_QUERY.load(Ordering::Acquire);
    if ptr != 0 {
        // SAFETY: `ptr` is the saved Wine `VirtualQuery` entrypoint.
        unsafe {
            let f: VirtualQueryFn = std::mem::transmute(ptr);
            f(addr, buf, len)
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!VirtualQuery` import.
        unsafe { VirtualQuery(addr, buf, len) }
    }
}

fn call_wine_virtual_query_ex(
    process: *mut c_void,
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> usize {
    let ptr = ORIG_VIRTUAL_QUERY_EX.load(Ordering::Acquire);
    if ptr != 0 {
        // SAFETY: `ptr` is the saved Wine `VirtualQueryEx` entrypoint.
        unsafe {
            let f: VirtualQueryExFn = std::mem::transmute(ptr);
            f(process, addr, buf, len)
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!VirtualQueryEx` import.
        unsafe { VirtualQueryEx(process, addr, buf, len) }
    }
}

/// Query Wine-managed addresses and fill the console-specific tail.
///
/// # Safety
/// A non-null `out` must point to a writable 64-byte XMem query result.
unsafe fn query_wine_xmem(process: *mut c_void, addr: *const c_void, out: *mut c_void) -> u64 {
    #[repr(C)]
    struct XMemInformation {
        mbi: MemoryBasicInformation,
        flags: u64,
        tag_code: u32,
        tag: [u8; 4],
    }
    const _: () = assert!(std::mem::size_of::<XMemInformation>() == 64);

    let mut mbi = MemoryBasicInformation::default();
    if call_wine_virtual_query_ex(
        process,
        addr,
        std::ptr::from_mut(&mut mbi).cast(),
        std::mem::size_of::<MemoryBasicInformation>(),
    ) != std::mem::size_of::<MemoryBasicInformation>()
    {
        return 0;
    }
    let flags =
        u64::from(mbi.state == MEM_COMMIT && valloc_flags::is_cpu_readable_protect(mbi.protect));
    let info = XMemInformation {
        mbi,
        flags,
        tag_code: 0,
        tag: [0; 4],
    };
    // SAFETY: The XMem caller supplies writable storage for the 64-byte result.
    unsafe { out.cast::<XMemInformation>().write_unaligned(info) };
    1
}

/// IAT replacement thunk for system-imported `VirtualAlloc`.
extern "win64" fn iat_thunk_virtual_alloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, false);
    let ret = try_xmem_virtual_alloc(addr, size, clean_alloc, clean_protect).unwrap_or_else(|| {
        call_wine_virtual_alloc(&ORIG_VIRTUAL_ALLOC, addr, size, clean_alloc, clean_protect)
    });
    if trace_hle_enabled() {
        let rva = capture_caller_rva(2);
        trace_hle_line(&format!(
            "iat!VirtualAlloc(addr=0x{:x}, size=0x{size:x}, alloc=0x{alloc_type:08x}->0x{clean_alloc:08x}, prot=0x{protect:08x}->0x{clean_protect:08x}) -> 0x{:x} @ rva=0x{rva:08x}",
            addr as usize, ret as usize
        ));
    }
    ret
}

/// IAT replacement thunk for system-imported `VirtualAllocEx`.
extern "win64" fn iat_thunk_virtual_alloc_ex(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, false);
    let ret = if is_current_process_handle(process) {
        try_xmem_virtual_alloc(addr, size, clean_alloc, clean_protect)
    } else {
        None
    }
    .unwrap_or_else(|| {
        call_wine_virtual_alloc_ex(
            &ORIG_VIRTUAL_ALLOC_EX,
            process,
            addr,
            size,
            clean_alloc,
            clean_protect,
        )
    });
    if trace_hle_enabled() {
        let rva = capture_caller_rva(2);
        trace_hle_line(&format!(
            "iat!VirtualAllocEx(addr=0x{:x}, size=0x{size:x}, alloc=0x{alloc_type:08x}->0x{clean_alloc:08x}, prot=0x{protect:08x}->0x{clean_protect:08x}) -> 0x{:x} @ rva=0x{rva:08x}",
            addr as usize, ret as usize
        ));
    }
    ret
}

/// IAT replacement thunk for system-imported `VirtualAllocFromApp`.
extern "win64" fn iat_thunk_virtual_alloc_from_app(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, false);
    let ret = if !valloc_flags::is_executable_cpu_protect(clean_protect) {
        try_xmem_virtual_alloc(addr, size, clean_alloc, clean_protect)
    } else {
        None
    }
    .unwrap_or_else(|| {
        call_wine_virtual_alloc_from_app(
            &ORIG_VIRTUAL_ALLOC_FROM_APP,
            addr,
            size,
            clean_alloc,
            clean_protect,
        )
    });
    if trace_hle_enabled() {
        let rva = capture_caller_rva(2);
        trace_hle_line(&format!(
            "iat!VirtualAllocFromApp(addr=0x{:x}, size=0x{size:x}, alloc=0x{alloc_type:08x}->0x{clean_alloc:08x}, prot=0x{protect:08x}->0x{clean_protect:08x}) -> 0x{:x} @ rva=0x{rva:08x}",
            addr as usize, ret as usize
        ));
    }
    ret
}

/// IAT replacement thunk for `VirtualFree` (routes `XMemArena`-owned addresses to `XMemArena`).
extern "win64" fn iat_thunk_virtual_free(addr: *mut c_void, size: usize, free_type: u32) -> i32 {
    console_virtual_free(addr, size, free_type)
}

/// IAT replacement thunk for `VirtualQuery` (routes `XMemArena`-owned addresses to `XMemArena`).
extern "win64" fn iat_thunk_virtual_query(
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> usize {
    console_virtual_query(addr, buf, len)
}

/// IAT replacement thunk for `VirtualQueryEx` (routes `XMemArena`-owned addresses to `XMemArena`).
extern "win64" fn iat_thunk_virtual_query_ex(
    process: *mut c_void,
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> usize {
    console_virtual_query_ex(process, addr, buf, len)
}

/// IAT replacement thunk for `GetProcAddress`: forwards transparently to Wine; whenever a lookup
/// returns `NULL` on a known console shim DLL (or when `X2L_TRACE_HLE=1`), logs `[MISSING_PROC]`
/// and records it into `report.json` before returning `NULL`.
extern "win64" fn iat_thunk_get_proc_address(hmodule: usize, proc_name: *const u8) -> usize {
    let orig = ORIG_GET_PROC_ADDRESS.load(Ordering::Acquire);
    let resolved = if orig != 0 {
        // SAFETY: `orig` is the saved Wine `kernelbase`/`kernel32!GetProcAddress` entrypoint.
        unsafe {
            let f: GetProcAddressFn = std::mem::transmute(orig);
            f(hmodule, proc_name)
        }
    } else {
        // SAFETY: Direct fallback to our own untouched `kernel32!GetProcAddress` import.
        unsafe { GetProcAddress(hmodule, proc_name) }
    };

    if resolved != 0 || proc_name.is_null() {
        return resolved;
    }

    // Preserve Win32 last error (`ERROR_PROC_NOT_FOUND`) across telemetry recording.
    // SAFETY: Reads thread-local Win32 last error.
    let saved_err = unsafe { GetLastError() };
    let shim_name = identify_console_shim_module(hmodule);
    if shim_name.is_some() || trace_hle_enabled() {
        let mut dll_buf = [0u8; 260];
        let dll_label: &str = if let Some(known) = shim_name {
            known
        } else {
            // SAFETY: Query loaded module basename safely.
            let n =
                unsafe { GetModuleFileNameA(hmodule, dll_buf.as_mut_ptr(), dll_buf.len() as u32) }
                    as usize;
            if n > 0
                && n < dll_buf.len()
                && let Ok(full) = std::str::from_utf8(&dll_buf[..n])
            {
                full.rsplit(['\\', '/']).next().unwrap_or("<unknown>")
            } else {
                "<unknown>"
            }
        };

        let raw_ptr = proc_name as usize;
        let caller_rva = capture_caller_rva(2);
        if raw_ptr <= 0xFFFF {
            // Ordinal lookup (`IS_INTRESOURCE`) — never dereference `proc_name`!
            // SAFETY: `dll_label` is valid UTF-8 on stack/static; ordinal passed in `a5`.
            let _ = unsafe {
                call_bridge_raw(
                    OP_RECORD_MISSING_PROC,
                    dll_label.as_ptr() as usize as u64,
                    dll_label.len() as u64,
                    0,
                    0,
                    caller_rva,
                    raw_ptr as u64,
                    caller_rva,
                )
            };
        } else {
            // Bounded C-string scan (max 256 bytes).
            let mut sym_len = 0usize;
            // SAFETY: `proc_name` is a non-ordinal C-string pointer supplied by the caller.
            unsafe {
                while sym_len < 256 && *proc_name.add(sym_len) != 0 {
                    sym_len += 1;
                }
            }
            // SAFETY: `[proc_name .. proc_name + sym_len]` is readable ASCII/UTF-8.
            let _ = unsafe {
                call_bridge_raw(
                    OP_RECORD_MISSING_PROC,
                    dll_label.as_ptr() as usize as u64,
                    dll_label.len() as u64,
                    raw_ptr as u64,
                    sym_len as u64,
                    caller_rva,
                    0,
                    caller_rva,
                )
            };
        }
    }

    // SAFETY: Restore exact Win32 last error expected by the caller.
    unsafe {
        SetLastError(if saved_err != 0 {
            saved_err
        } else {
            ERROR_PROC_NOT_FOUND
        });
    }
    0
}

/// IAT replacement thunk for `OutputDebugStringA`: captures bounded text into `libx2l_core.so`
/// (logging `[DEBUGSTR]` under `--trace-hle` and retaining the last 16 entries for crash reports),
/// preserves incoming `LastError`, and forwards cleanly to Wine.
extern "win64" fn iat_thunk_output_debug_string_a(msg: *const u8) {
    // SAFETY: Preserve incoming Win32 last error across hook telemetry.
    let saved_err = unsafe { GetLastError() };
    if !msg.is_null() {
        let mut len = 0usize;
        // SAFETY: Caller passes a NUL-terminated ANSI string; bounded to 512 bytes.
        unsafe {
            while len < 512 && *msg.add(len) != 0 {
                len += 1;
            }
        }
        if len > 0 {
            // SAFETY: `[msg .. msg + len]` is readable for the synchronous bridge call.
            let _ = unsafe {
                call_bridge_raw(
                    OP_RECORD_DEBUG_STRING,
                    msg as usize as u64,
                    len as u64,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            };
        }
    }
    // SAFETY: Restore incoming last error before forwarding to original Wine function.
    unsafe { SetLastError(saved_err) };
    let orig = ORIG_OUTPUT_DEBUG_STRING_A.load(Ordering::Acquire);
    if orig != 0 {
        // SAFETY: `orig` is the saved Wine `OutputDebugStringA` entrypoint.
        unsafe {
            let f: OutputDebugStringAFn = std::mem::transmute(orig);
            f(msg);
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!OutputDebugStringA` import.
        unsafe { OutputDebugStringA(msg) };
    }
}

/// IAT replacement thunk for `OutputDebugStringW`: decodes bounded UTF-16 to UTF-8, sends to
/// `libx2l_core.so`, preserves incoming `LastError`, and forwards cleanly to Wine.
extern "win64" fn iat_thunk_output_debug_string_w(msg: *const u16) {
    // SAFETY: Preserve incoming Win32 last error across hook telemetry.
    let saved_err = unsafe { GetLastError() };
    if !msg.is_null() {
        let mut len = 0usize;
        // SAFETY: Caller passes a NUL-terminated UTF-16 string; bounded to 512 code units.
        unsafe {
            while len < 512 && *msg.add(len) != 0 {
                len += 1;
            }
        }
        if len > 0 {
            // SAFETY: `[msg .. msg + len]` is readable UTF-16 code units.
            let utf16 = unsafe { std::slice::from_raw_parts(msg, len) };
            let utf8 = String::from_utf16_lossy(utf16);
            // SAFETY: `utf8` lives on the stack across the synchronous bridge call.
            let _ = unsafe {
                call_bridge_raw(
                    OP_RECORD_DEBUG_STRING,
                    utf8.as_ptr() as usize as u64,
                    utf8.len() as u64,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            };
        }
    }
    // SAFETY: Restore incoming last error before forwarding to original Wine function.
    unsafe { SetLastError(saved_err) };
    let orig = ORIG_OUTPUT_DEBUG_STRING_W.load(Ordering::Acquire);
    if orig != 0 {
        // SAFETY: `orig` is the saved Wine `OutputDebugStringW` entrypoint.
        unsafe {
            let f: OutputDebugStringWFn = std::mem::transmute(orig);
            f(msg);
        }
    } else {
        // SAFETY: Direct fallback to our own `kernel32!OutputDebugStringW` import.
        unsafe { OutputDebugStringW(msg) };
    }
}

/// Explicit console-context `kernelx!VirtualAlloc` routing arena addresses to `XMemArena` and
/// stripping console flags before forwarding to Wine.
pub fn console_virtual_alloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, true);
    let ret = try_xmem_virtual_alloc(addr, size, clean_alloc, clean_protect).unwrap_or_else(|| {
        call_wine_virtual_alloc(&ORIG_VIRTUAL_ALLOC, addr, size, clean_alloc, clean_protect)
    });
    if trace_hle_enabled() {
        let rva = capture_caller_rva(2);
        trace_hle_line(&format!(
            "kernelx!VirtualAlloc(addr=0x{:x}, size=0x{size:x}, alloc=0x{alloc_type:08x}->0x{clean_alloc:08x}, prot=0x{protect:08x}->0x{clean_protect:08x}) -> 0x{:x} @ rva=0x{rva:08x}",
            addr as usize, ret as usize
        ));
    }
    ret
}

/// Explicit console-context `kernelx!VirtualAllocEx` routing arena addresses to `XMemArena` and
/// stripping console flags before forwarding to Wine.
pub fn console_virtual_alloc_ex(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, true);
    let ret = if is_current_process_handle(process) {
        try_xmem_virtual_alloc(addr, size, clean_alloc, clean_protect)
    } else {
        None
    }
    .unwrap_or_else(|| {
        call_wine_virtual_alloc_ex(
            &ORIG_VIRTUAL_ALLOC_EX,
            process,
            addr,
            size,
            clean_alloc,
            clean_protect,
        )
    });
    if trace_hle_enabled() {
        let rva = capture_caller_rva(2);
        trace_hle_line(&format!(
            "kernelx!VirtualAllocEx(addr=0x{:x}, size=0x{size:x}, alloc=0x{alloc_type:08x}->0x{clean_alloc:08x}, prot=0x{protect:08x}->0x{clean_protect:08x}) -> 0x{:x} @ rva=0x{rva:08x}",
            addr as usize, ret as usize
        ));
    }
    ret
}

/// Explicit console-context `kernelx!VirtualAllocFromApp` stripping console flags and enforcing
/// non-executable CPU protection before routing arena/Wine allocations.
pub fn console_virtual_alloc_from_app(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, true);
    let ret = if !valloc_flags::is_executable_cpu_protect(clean_protect) {
        try_xmem_virtual_alloc(addr, size, clean_alloc, clean_protect)
    } else {
        None
    }
    .unwrap_or_else(|| {
        call_wine_virtual_alloc_from_app(
            &ORIG_VIRTUAL_ALLOC_FROM_APP,
            addr,
            size,
            clean_alloc,
            clean_protect,
        )
    });
    if trace_hle_enabled() {
        let rva = capture_caller_rva(2);
        trace_hle_line(&format!(
            "kernelx!VirtualAllocFromApp(addr=0x{:x}, size=0x{size:x}, alloc=0x{alloc_type:08x}->0x{clean_alloc:08x}, prot=0x{protect:08x}->0x{clean_protect:08x}) -> 0x{:x} @ rva=0x{rva:08x}",
            addr as usize, ret as usize
        ));
    }
    ret
}

/// Arena-aware `VirtualFree` shared between IAT thunks and `kernelx.dll`.
pub fn console_virtual_free(addr: *mut c_void, size: usize, free_type: u32) -> i32 {
    try_xmem_virtual_free(addr, size, free_type)
        .unwrap_or_else(|| call_wine_virtual_free(addr, size, free_type))
}

/// Arena-aware `VirtualQuery` shared between IAT thunks and `kernelx.dll`.
pub fn console_virtual_query(addr: *const c_void, buf: *mut c_void, len: usize) -> usize {
    let (owned, next_res) = try_xmem_virtual_query(addr, buf, len);
    if let Some(res) = owned {
        return res;
    }
    let wine_ret = call_wine_virtual_query(addr, buf, len);
    clamp_wine_free_mbi(buf, wine_ret, next_res);
    wine_ret
}

/// Arena-aware `VirtualQueryEx` shared between IAT thunks and `kernelx.dll`.
pub fn console_virtual_query_ex(
    process: *mut c_void,
    addr: *const c_void,
    buf: *mut c_void,
    len: usize,
) -> usize {
    if is_current_process_handle(process) {
        let (owned, next_res) = try_xmem_virtual_query(addr, buf, len);
        if let Some(res) = owned {
            return res;
        }
        let wine_ret = call_wine_virtual_query_ex(process, addr, buf, len);
        clamp_wine_free_mbi(buf, wine_ret, next_res);
        wine_ret
    } else {
        call_wine_virtual_query_ex(process, addr, buf, len)
    }
}

fn pe_image_size(base: usize) -> Option<usize> {
    if base == 0 {
        return None;
    }
    // SAFETY: `base` is an in-process PE module base returned by `GetModuleHandleA`.
    unsafe {
        let p = base as *const u8;
        if u16::from_le_bytes([*p, *p.add(1)]) != 0x5A4D {
            return None;
        }
        let e_lfanew =
            u32::from_le_bytes([*p.add(0x3C), *p.add(0x3D), *p.add(0x3E), *p.add(0x3F)]) as usize;
        if !(0x40..=0x10000).contains(&e_lfanew) {
            return None;
        }
        let nt = p.add(e_lfanew);
        if u32::from_le_bytes([*nt, *nt.add(1), *nt.add(2), *nt.add(3)]) != 0x0000_4550 {
            return None;
        }
        let opt = nt.add(24);
        if u16::from_le_bytes([*opt, *opt.add(1)]) != 0x020B {
            return None; // PE32+ only
        }
        let size_of_image =
            u32::from_le_bytes([*opt.add(56), *opt.add(57), *opt.add(58), *opt.add(59)]) as usize;
        (size_of_image >= e_lfanew + 264).then_some(size_of_image)
    }
}

fn read_cstr_bounded<'a>(base: usize, image_size: usize, rva: usize) -> Option<&'a [u8]> {
    if rva == 0 || rva >= image_size {
        return None;
    }
    let max_len = (image_size - rva).min(256);
    // SAFETY: `[base + rva .. base + rva + max_len]` is strictly within the mapped PE image.
    let slice = unsafe { std::slice::from_raw_parts((base + rva) as *const u8, max_len) };
    let end = slice.iter().position(|&b| b == 0)?;
    Some(&slice[..end])
}

fn is_target_win32_hook_dll(name: &[u8]) -> bool {
    let mut lower = [0u8; 64];
    if name.len() > lower.len() {
        return false;
    }
    for (i, &b) in name.iter().enumerate() {
        lower[i] = b.to_ascii_lowercase();
    }
    let s = &lower[..name.len()];
    s == b"kernel32.dll"
        || s == b"kernelbase.dll"
        || s.starts_with(b"api-ms-win-core-memory-l1-1-")
        || s.starts_with(b"api-ms-win-core-libraryloader-l1-")
        || s.starts_with(b"api-ms-win-core-debug-l1-")
}

fn addr_in_module(addr: usize, mod_base: usize, mod_size: usize) -> bool {
    mod_base != 0 && addr >= mod_base && addr < mod_base.saturating_add(mod_size)
}

/// Idempotently install main-executable PE32+ IAT hooks (`VirtualAlloc*`, `VirtualFree`,
/// `VirtualQuery*`, `GetProcAddress`, `OutputDebugStringA/W`) and the fatal VEH crash guard
/// once per process from any loaded console shim's `DllMain(DLL_PROCESS_ATTACH)`.
///
/// Safe under Wine's loader lock: performs zero `LoadLibrary` calls, zero filesystem I/O, and
/// zero native bridge calls. Pins the owning shim DLL in memory so installed thunks stay valid
/// for process lifetime even if a guest dynamically unloads a shim.
pub fn install_valloc_iat_patch() {
    if IAT_PATCH_INSTALLED.swap(true, Ordering::AcqRel) {
        return;
    }

    // Only activate when launched under X2Linux (`X2L_BRIDGE_ENABLED=1`).
    let mut flag_buf = [0u8; 8];
    // SAFETY: Valid NUL-terminated env name and stack buffer.
    let flag_len = unsafe {
        GetEnvironmentVariableA(
            b"X2L_BRIDGE_ENABLED\0".as_ptr(),
            flag_buf.as_mut_ptr(),
            flag_buf.len() as u32,
        )
    } as usize;
    if flag_len != 1 || flag_buf[0] != b'1' {
        return;
    }

    // Process-wide single-shot guard across all statically linked console PE shims.
    let mut installed_buf = [0u8; 8];
    // SAFETY: PEB environment block is process-wide and safe under Wine's loader lock.
    let installed_len = unsafe {
        GetEnvironmentVariableA(
            b"X2L_MAIN_HOOKS_INSTALLED\0".as_ptr(),
            installed_buf.as_mut_ptr(),
            installed_buf.len() as u32,
        )
    } as usize;
    if installed_len == 1 && installed_buf[0] == b'1' {
        return;
    }
    // SAFETY: Mark process-wide hook installation complete before patching.
    unsafe {
        let _ = SetEnvironmentVariableA(b"X2L_MAIN_HOOKS_INSTALLED\0".as_ptr(), b"1\0".as_ptr());
    }

    // Pin the console shim DLL owning these thunks for process lifetime.
    let mut pinned_mod: usize = 0;
    // SAFETY: `iat_thunk_virtual_alloc` resides inside this DLL image.
    unsafe {
        let _ = GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
            iat_thunk_virtual_alloc as *const c_void,
            &mut pinned_mod,
        );
    }

    // SAFETY: `GetModuleHandleA` reads already-loaded module handles without acquiring loader IO.
    // SAFETY: `GetModuleHandleA(NULL)` returns the current process primary executable ImageBase.
    let exe_base = unsafe { GetModuleHandleA(std::ptr::null()) };
    let Some(exe_size) = pe_image_size(exe_base) else {
        return;
    };
    CACHED_EXE_BASE.store(exe_base, Ordering::Release);
    CACHED_EXE_SIZE.store(exe_size, Ordering::Release);

    install_crash_veh();

    // SAFETY: Read Wine `kernel32` & `kernelbase` ranges so we only replace genuine Wine targets.
    let k32_base = unsafe { GetModuleHandleA(b"kernel32.dll\0".as_ptr()) };
    let k32_size = pe_image_size(k32_base).unwrap_or(0);
    let kb_base = unsafe { GetModuleHandleA(b"kernelbase.dll\0".as_ptr()) };
    let kb_size = pe_image_size(kb_base).unwrap_or(0);

    // SAFETY: All offsets are bounds-checked against `exe_size` before dereferencing.
    unsafe {
        let p = exe_base as *const u8;
        let e_lfanew =
            u32::from_le_bytes([*p.add(0x3C), *p.add(0x3D), *p.add(0x3E), *p.add(0x3F)]) as usize;
        let opt = p.add(e_lfanew + 24);
        let num_rvas =
            u32::from_le_bytes([*opt.add(108), *opt.add(109), *opt.add(110), *opt.add(111)])
                as usize;
        if num_rvas <= 1 {
            return;
        }
        // DataDirectory[1] = Import Table (at OptionalHeader + 112 + 8 = 120)
        let dd_imp = opt.add(120);
        let imp_rva =
            u32::from_le_bytes([*dd_imp, *dd_imp.add(1), *dd_imp.add(2), *dd_imp.add(3)]) as usize;
        let imp_size = u32::from_le_bytes([
            *dd_imp.add(4),
            *dd_imp.add(5),
            *dd_imp.add(6),
            *dd_imp.add(7),
        ]) as usize;
        if imp_rva == 0 || imp_size < 20 || imp_rva + imp_size > exe_size {
            return;
        }

        let mut desc_off = imp_rva;
        while desc_off + 20 <= imp_rva + imp_size {
            let d = p.add(desc_off);
            let orig_first_thunk =
                u32::from_le_bytes([*d, *d.add(1), *d.add(2), *d.add(3)]) as usize;
            let name_rva =
                u32::from_le_bytes([*d.add(12), *d.add(13), *d.add(14), *d.add(15)]) as usize;
            let first_thunk =
                u32::from_le_bytes([*d.add(16), *d.add(17), *d.add(18), *d.add(19)]) as usize;
            if orig_first_thunk == 0 && name_rva == 0 && first_thunk == 0 {
                break;
            }
            desc_off += 20;

            let Some(dll_name) = read_cstr_bounded(exe_base, exe_size, name_rva) else {
                continue;
            };
            if !is_target_win32_hook_dll(dll_name) || orig_first_thunk == 0 || first_thunk == 0 {
                continue;
            }

            let mut idx = 0usize;
            loop {
                let ilt_rva = orig_first_thunk.saturating_add(idx * 8);
                let iat_rva = first_thunk.saturating_add(idx * 8);
                if ilt_rva + 8 > exe_size || iat_rva + 8 > exe_size {
                    break;
                }
                let ilt_entry = u64::from_le_bytes(*(p.add(ilt_rva) as *const [u8; 8]));
                if ilt_entry == 0 {
                    break;
                }
                idx += 1;
                if (ilt_entry & (1u64 << 63)) != 0 {
                    continue; // Ordinal import
                }
                let hint_name_rva = (ilt_entry & 0x7FFF_FFFF) as usize;
                let Some(sym_name) =
                    read_cstr_bounded(exe_base, exe_size, hint_name_rva.saturating_add(2))
                else {
                    continue;
                };

                let (saved, replacement): (&AtomicUsize, usize) = match sym_name {
                    b"VirtualAlloc" => (
                        &ORIG_VIRTUAL_ALLOC,
                        iat_thunk_virtual_alloc as *const () as usize,
                    ),
                    b"VirtualAllocEx" => (
                        &ORIG_VIRTUAL_ALLOC_EX,
                        iat_thunk_virtual_alloc_ex as *const () as usize,
                    ),
                    b"VirtualAllocFromApp" => (
                        &ORIG_VIRTUAL_ALLOC_FROM_APP,
                        iat_thunk_virtual_alloc_from_app as *const () as usize,
                    ),
                    b"VirtualFree" => (
                        &ORIG_VIRTUAL_FREE,
                        iat_thunk_virtual_free as *const () as usize,
                    ),
                    b"VirtualQuery" => (
                        &ORIG_VIRTUAL_QUERY,
                        iat_thunk_virtual_query as *const () as usize,
                    ),
                    b"VirtualQueryEx" => (
                        &ORIG_VIRTUAL_QUERY_EX,
                        iat_thunk_virtual_query_ex as *const () as usize,
                    ),
                    b"GetProcAddress" => (
                        &ORIG_GET_PROC_ADDRESS,
                        iat_thunk_get_proc_address as *const () as usize,
                    ),
                    b"OutputDebugStringA" => (
                        &ORIG_OUTPUT_DEBUG_STRING_A,
                        iat_thunk_output_debug_string_a as *const () as usize,
                    ),
                    b"OutputDebugStringW" => (
                        &ORIG_OUTPUT_DEBUG_STRING_W,
                        iat_thunk_output_debug_string_w as *const () as usize,
                    ),
                    _ => continue,
                };

                let slot_ptr = (exe_base + iat_rva) as *mut usize;
                let current = *slot_ptr;
                let is_wine_target = addr_in_module(current, k32_base, k32_size)
                    || addr_in_module(current, kb_base, kb_size);
                if current == 0 || current == replacement || !is_wine_target {
                    continue;
                }

                // Publish original Wine entrypoint BEFORE patching the IAT slot.
                let _ = saved.compare_exchange(0, current, Ordering::Release, Ordering::Relaxed);

                let mut old_prot = 0u32;
                if VirtualProtect(
                    slot_ptr.cast::<c_void>(),
                    std::mem::size_of::<usize>(),
                    PAGE_READWRITE,
                    &mut old_prot,
                ) != 0
                {
                    std::ptr::write_volatile(slot_ptr, replacement);
                    let mut ignored = 0u32;
                    let _ = VirtualProtect(
                        slot_ptr.cast::<c_void>(),
                        std::mem::size_of::<usize>(),
                        old_prot,
                        &mut ignored,
                    );
                }
            }
        }
    }
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
        // SAFETY: `GetModuleHandleA(NULL)` returns the primary executable's ImageBase.
        let exe_base = unsafe { GetModuleHandleA(std::ptr::null()) } as u64;
        // SAFETY: `X2L_BRIDGE_FN` is published in `.init_array` by `libx2l_core.so` pointing to `x2l_bridge_dispatch`.
        let func: BridgeFn = unsafe { std::mem::transmute(addr) };
        // SAFETY: `OP_HANDSHAKE` takes `exe_base` and returns `X2L_BRIDGE_MAGIC` to validate the bridge ABI.
        let magic = unsafe { func(OP_HANDSHAKE, exe_base, 0, 0, 0, 0, 0, 0) };
        (magic == X2L_BRIDGE_MAGIC).then_some(func)
    })
}

/// Low-level bridge call taking an explicit `caller_rva` (used internally for trace/VEH opcodes).
///
/// # Safety
/// Caller must pass valid arguments matching `opcode`.
#[allow(clippy::too_many_arguments)]
unsafe fn call_bridge_raw(
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    caller_rva: u64,
) -> Option<u64> {
    let func = resolve_bridge()?;
    // SAFETY: Validated by `OP_HANDSHAKE` magic and caller guarantees opcode argument invariants.
    Some(unsafe { func(opcode, a0, a1, a2, a3, a4, a5, caller_rva) })
}

/// Call into in-process `libx2l_core.so` via the `extern "win64"` bridge (automatically capturing
/// caller RVA when `X2L_TRACE_HLE=1` is enabled).
///
/// # Safety
/// Caller must pass valid arguments matching `opcode`.
#[inline(never)]
pub unsafe fn call_bridge(
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) -> Option<u64> {
    let unsupported_guard =
        opcode == OP_XMEM_VIRTUAL_ALLOC && a4 as u32 & valloc_flags::PAGE_GUARD != 0;
    let caller_rva = if unsupported_guard
        || trace_hle_enabled()
            && !matches!(
                opcode,
                OP_HANDSHAKE
                    | OP_FLUSH_REPORT
                    | OP_TRACE_HLE_LINE
                    | OP_RECORD_EXCEPTION
                    | OP_RECORD_DEBUG_STRING
                    | OP_RECORD_MISSING_PROC
            ) {
        capture_caller_rva(2)
    } else {
        0
    };
    if opcode == OP_XMEM_VIRTUAL_QUERY {
        if a2 == 0 {
            // SAFETY: Set Win32 error on the calling thread for the null output argument.
            unsafe { SetLastError(ERROR_INVALID_PARAMETER) };
            return Some(0);
        }
        let process = if a0 == 0 {
            // SAFETY: GetCurrentProcess returns the current process pseudo-handle.
            unsafe { GetCurrentProcess() }
        } else {
            a0 as *mut c_void
        };
        if is_current_process_handle(process) {
            // SAFETY: Forward the caller's arguments to the core's arena query.
            let hit = unsafe { call_bridge_raw(opcode, a0, a1, a2, a3, a4, a5, caller_rva) };
            if hit.is_some_and(|ret| ret != 0) {
                return hit;
            }
        }
        // SAFETY: The output pointer was checked for null and points to the caller's result.
        return Some(unsafe { query_wine_xmem(process, a1 as *const c_void, a2 as *mut c_void) });
    }
    // SAFETY: Forwards validated arguments to `call_bridge_raw`.
    unsafe { call_bridge_raw(opcode, a0, a1, a2, a3, a4, a5, caller_rva) }
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
    // SAFETY: `GetModuleHandleA(NULL)` returns the current process primary executable ImageBase.
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
        call_bridge_raw(
            OP_RECORD_STUB,
            dll.as_ptr() as usize as u64,
            dll.len() as u64,
            symbol.as_ptr() as usize as u64,
            symbol.len() as u64,
            caller_rva,
            default_ret,
            caller_rva,
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
        call_bridge_raw(
            OP_RECORD_UNKNOWN_ORDINAL,
            dll.as_ptr() as usize as u64,
            dll.len() as u64,
            u64::from(ordinal),
            caller_rva,
            default_ret,
            0,
            caller_rva,
        )
        .unwrap_or(default_ret)
    }
}

/// Common `DllMain` lifecycle handler used by every console shim DLL:
/// - On `DLL_PROCESS_ATTACH` (`reason == 1`): idempotently installs the main-image IAT hooks & VEH.
/// - On `DLL_PROCESS_DETACH` (`reason == 0`): flushes telemetry & orderly WSI teardown.
pub fn handle_dll_main(reason: u32) -> i32 {
    if reason == 1 {
        install_valloc_iat_patch();
    } else if reason == 0 {
        // SAFETY: `OP_FLUSH_REPORT` takes no pointers.
        let _ = unsafe { call_bridge_raw(OP_FLUSH_REPORT, 0, 0, 0, 0, 0, 0, 0) };
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
        // SAFETY: `info.dll` and `sym` are `'static` UTF-8 slices passed to `OP_RECORD_STUB`.
        Some(sym) => unsafe {
            call_bridge_raw(
                OP_RECORD_STUB,
                info.dll.as_ptr() as usize as u64,
                info.dll.len() as u64,
                sym.as_ptr() as usize as u64,
                sym.len() as u64,
                caller_rva,
                info.default_ret,
                caller_rva,
            )
            .unwrap_or(info.default_ret)
        },
        // SAFETY: `info.dll` is a `'static` UTF-8 slice passed to `OP_RECORD_UNKNOWN_ORDINAL`.
        None => unsafe {
            call_bridge_raw(
                OP_RECORD_UNKNOWN_ORDINAL,
                info.dll.as_ptr() as usize as u64,
                info.dll.len() as u64,
                u64::from(info.ordinal),
                caller_rva,
                info.default_ret,
                0,
                caller_rva,
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
