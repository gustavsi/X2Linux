//! Shared Win64 PE-side bridge client, caller RVA capture, safe `VirtualAlloc` console flag
//! sanitizer + PE32+ IAT patcher, and dynamic W^X x86-64 auto-stub thunk generator for
//! `crates/x2l-pe-shims`.

#[path = "../../../x2l-hle-kernelx/src/valloc_flags.rs"]
pub mod valloc_flags;

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    pub fn GetModuleHandleA(module_name: *const u8) -> usize;
    fn GetProcAddress(module: usize, proc_name: *const u8) -> usize;
    fn SetLastError(err_code: u32);
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
    fn VirtualProtect(
        addr: *mut c_void,
        size: usize,
        new_protect: u32,
        old_protect: *mut u32,
    ) -> i32;
}

const ERROR_INVALID_PARAMETER: u32 = 87;

type VirtualAllocFn = unsafe extern "win64" fn(*mut c_void, usize, u32, u32) -> *mut c_void;
type VirtualAllocExFn =
    unsafe extern "win64" fn(*mut c_void, *mut c_void, usize, u32, u32) -> *mut c_void;

static IAT_PATCH_INSTALLED: AtomicBool = AtomicBool::new(false);
static ORIG_VIRTUAL_ALLOC: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_ALLOC_EX: AtomicUsize = AtomicUsize::new(0);
static ORIG_VIRTUAL_ALLOC_FROM_APP: AtomicUsize = AtomicUsize::new(0);

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

/// IAT replacement thunk for system-imported `VirtualAlloc`.
extern "win64" fn iat_thunk_virtual_alloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, false);
    call_wine_virtual_alloc(&ORIG_VIRTUAL_ALLOC, addr, size, clean_alloc, clean_protect)
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
    call_wine_virtual_alloc_ex(
        &ORIG_VIRTUAL_ALLOC_EX,
        process,
        addr,
        size,
        clean_alloc,
        clean_protect,
    )
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
    call_wine_virtual_alloc_from_app(
        &ORIG_VIRTUAL_ALLOC_FROM_APP,
        addr,
        size,
        clean_alloc,
        clean_protect,
    )
}

/// Explicit console-context `kernelx!VirtualAlloc` stripping console flags and forwarding to Wine.
pub fn console_virtual_alloc(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, true);
    call_wine_virtual_alloc(&ORIG_VIRTUAL_ALLOC, addr, size, clean_alloc, clean_protect)
}

/// Explicit console-context `kernelx!VirtualAllocEx` stripping console flags and forwarding to Wine.
pub fn console_virtual_alloc_ex(
    process: *mut c_void,
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, true);
    call_wine_virtual_alloc_ex(
        &ORIG_VIRTUAL_ALLOC_EX,
        process,
        addr,
        size,
        clean_alloc,
        clean_protect,
    )
}

/// Explicit console-context `kernelx!VirtualAllocFromApp` stripping console flags and enforcing
/// non-executable CPU protection before forwarding to Wine.
pub fn console_virtual_alloc_from_app(
    addr: *mut c_void,
    size: usize,
    alloc_type: u32,
    protect: u32,
) -> *mut c_void {
    let (clean_alloc, clean_protect) =
        valloc_flags::sanitize_virtual_alloc_flags(alloc_type, protect, size, true);
    call_wine_virtual_alloc_from_app(
        &ORIG_VIRTUAL_ALLOC_FROM_APP,
        addr,
        size,
        clean_alloc,
        clean_protect,
    )
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

fn is_target_memory_dll(name: &[u8]) -> bool {
    let mut lower = [0u8; 64];
    if name.len() > lower.len() {
        return false;
    }
    for (i, &b) in name.iter().enumerate() {
        lower[i] = b.to_ascii_lowercase();
    }
    let s = &lower[..name.len()];
    s == b"kernel32.dll" || s == b"kernelbase.dll" || s.starts_with(b"api-ms-win-core-memory-l1-1-")
}

fn addr_in_module(addr: usize, mod_base: usize, mod_size: usize) -> bool {
    mod_base != 0 && addr >= mod_base && addr < mod_base.saturating_add(mod_size)
}

/// Patch only `VirtualAlloc`, `VirtualAllocEx`, and `VirtualAllocFromApp` IAT slots in the main
/// executable during `DllMain(DLL_PROCESS_ATTACH)` (before TLS callbacks and `main`).
///
/// Safe under Wine's loader lock: performs zero `LoadLibrary` calls, zero filesystem I/O, and
/// zero native bridge calls; only patches slots currently pointing inside Wine's `kernel32.dll`
/// or `kernelbase.dll`.
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

    // SAFETY: `GetModuleHandleA` reads already-loaded module handles without acquiring loader IO.
    let exe_base = unsafe { GetModuleHandleA(std::ptr::null()) };
    let Some(exe_size) = pe_image_size(exe_base) else {
        return;
    };

    // SAFETY: Read already-loaded Wine `kernel32.dll` / `kernelbase.dll` module ranges.
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
            if !is_target_memory_dll(dll_name) || orig_first_thunk == 0 || first_thunk == 0 {
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
                    _ => continue,
                };

                let slot_ptr = (exe_base + iat_rva) as *mut usize;
                let current = *slot_ptr;
                if current == 0 || current == replacement {
                    continue;
                }
                if !addr_in_module(current, k32_base, k32_size)
                    && !addr_in_module(current, kb_base, kb_size)
                {
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
