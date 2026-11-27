//! Native Linux core runtime shared library (`libx2l_core.so`).
//!
//! Loaded in-process into Wine via `LD_PRELOAD`. On ELF initialization (`.init_array`), when
//! `X2L_BRIDGE_ENABLED=1`, exports the process-local address of [`x2l_bridge_dispatch`]
//! (`extern "win64"`) through the `X2L_BRIDGE_FN` environment variable so thin Win64 PE shims
//! (`kernelx.dll`, `xgameruntime.dll`, `d3d12_xs.dll`) invoke native Linux Rust HLE, Vulkan,
//! and telemetry directly in-process with zero Wine-internal header coupling.

use std::ffi::{CString, c_char, c_int, c_void};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use x2l_common::bridge::*;
use x2l_common::{DeviceType, HostRuntimeMetadata, TelemetryReport};
use x2l_gpu_vk::{
    NULL_DEVICE_VTABLE_SLOTS, NullD3D12DeviceX, build_null_device_vtable, capture_raw_shader_blob,
    create_null_device,
};
use x2l_hle_kernelx::{KernelxState, TitleMemoryStatus, strip_virtual_alloc_flags};
use x2l_hle_xgr::XgrState;

/// ABI version shared between `x2l-pe-shims` and `libx2l_core.so`.
pub const X2L_UNIX_CALL_ABI_VERSION: u32 = 1;

unsafe extern "C" {
    fn setenv(name: *const c_char, value: *const c_char, overwrite: c_int) -> c_int;
}

/// Process-global runtime state owned by `libx2l_core.so`.
struct CoreRuntime {
    kernelx: KernelxState,
    xgr: XgrState,
    report: Mutex<TelemetryReport>,
    report_path: Option<PathBuf>,
    log_path: Option<PathBuf>,
    shaders_raw_dir: Option<PathBuf>,
    null_device_vtbl: OnceLock<&'static [usize; NULL_DEVICE_VTABLE_SLOTS]>,
}

static RUNTIME: OnceLock<CoreRuntime> = OnceLock::new();
static HAS_GUEST_ACTIVITY: AtomicBool = AtomicBool::new(false);
static GUEST_IMAGE_BASE: AtomicU64 = AtomicU64::new(0);

fn runtime() -> &'static CoreRuntime {
    RUNTIME.get_or_init(CoreRuntime::from_env)
}

impl CoreRuntime {
    fn from_env() -> Self {
        let title_id = std::env::var("X2L_TITLE_ID").unwrap_or_else(|_| "unknown-title".into());
        let device_str = std::env::var("X2L_DEVICE_TYPE").unwrap_or_else(|_| "series_x".into());
        let device = match device_str.as_str() {
            "series_s" => DeviceType::SeriesS,
            _ => DeviceType::SeriesX,
        };
        let gamertag = std::env::var("X2L_GAMERTAG").unwrap_or_else(|_| "LinuxPlayer".into());
        let saves_dir = std::env::var("X2L_SAVES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("x2linux-saves"));
        let report_path = std::env::var("X2L_REPORT_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let log_path = std::env::var("X2L_LOG_PATH")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let shaders_raw_dir = std::env::var("X2L_SHADERS_RAW_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let mut report = TelemetryReport::new(title_id);
        report.host = HostRuntimeMetadata {
            kernel_release: read_kernel_release(),
            ntsync_available: std::env::var("X2L_NTSYNC_AVAILABLE")
                .map(|v| v == "1" || v == "true")
                .unwrap_or_else(|_| std::path::Path::new("/dev/ntsync").exists()),
            ntsync_enabled: std::env::var("X2L_NTSYNC_ENABLED")
                .map(|v| v == "1" || v == "true")
                .unwrap_or(false),
            overlayfs_mode: std::env::var("X2L_OVERLAY_BACKEND")
                .unwrap_or_else(|_| "unknown".into()),
            wine_binary: std::env::var("X2L_WINE_BINARY").unwrap_or_else(|_| "wine".into()),
            device_type: device_str,
        };

        Self {
            kernelx: KernelxState::new(device),
            xgr: XgrState::new(gamertag, saves_dir),
            report: Mutex::new(report),
            report_path,
            log_path,
            shaders_raw_dir,
            null_device_vtbl: OnceLock::new(),
        }
    }

    fn log_line(&self, line: &str) {
        if let Some(path) = &self.log_path {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(file, "{line}");
            }
        }
    }

    fn flush_report(&self) {
        let Some(path) = &self.report_path else {
            return;
        };
        let Ok(guard) = self.report.lock() else {
            return;
        };
        let Ok(json) = guard.to_json_pretty() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, json);
    }

    fn null_vtable(&self) -> Option<&'static [usize; NULL_DEVICE_VTABLE_SLOTS]> {
        if let Some(v) = self.null_device_vtbl.get() {
            return Some(*v);
        }
        let built = build_null_device_vtable(d3d12_vtable_probe_callback).ok()?;
        let _ = self.null_device_vtbl.set(built);
        Some(built)
    }
}

fn read_kernel_release() -> String {
    fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "linux".into())
}

extern "win64" fn d3d12_vtable_probe_callback(slot_index: u32, caller_rip: u64) -> u64 {
    let rt = runtime();
    HAS_GUEST_ACTIVITY.store(true, Ordering::SeqCst);
    let base = GUEST_IMAGE_BASE.load(Ordering::Acquire);
    let caller_rva = if base != 0 && caller_rip >= base && (caller_rip - base) < 0x1000_0000 {
        caller_rip - base
    } else {
        caller_rip
    };
    if let Ok(mut rep) = rt.report.lock() {
        rep.record_unmapped_vtable_slot("ID3D12DeviceX", slot_index, caller_rva);
    }
    rt.log_line(&format!(
        "[D3D12X_VTABLE] ID3D12DeviceX[{slot_index}] caller_rva=0x{caller_rva:08x} -> S_OK"
    ));
    HRESULT_S_OK
}

/// Read a guest UTF-8 string slice bounded safely to 4096 bytes.
///
/// # Safety
/// `ptr` must point to at least `len` readable bytes in the shared guest/host address space.
unsafe fn read_guest_str(ptr: u64, len: u64) -> String {
    if ptr == 0 || len == 0 {
        return String::new();
    }
    let clamped = (len as usize).min(4096);
    // SAFETY: Caller passes a valid readable slice from the in-process PE shim.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, clamped) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Write a NUL-terminated UTF-8 string into a caller-supplied buffer.
///
/// # Safety
/// `dst` must point to `cap` writable bytes in the shared address space.
unsafe fn write_guest_cstr(dst: u64, cap: u64, value: &str) -> u64 {
    if dst == 0 || cap == 0 {
        return HRESULT_E_INVALIDARG;
    }
    let cap = cap as usize;
    let bytes = value.as_bytes();
    let copy_len = bytes.len().min(cap.saturating_sub(1));
    // SAFETY: Caller passes a valid writable output buffer of size `cap`.
    let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, cap) };
    out[..copy_len].copy_from_slice(&bytes[..copy_len]);
    out[copy_len] = 0;
    HRESULT_S_OK
}

/// Single version-independent `extern "win64"` dispatch entrypoint called directly in-process
/// by `crates/x2l-pe-shims` DLLs (`kernelx.dll`, `xgameruntime.dll`, `d3d12_xs.dll`).
///
/// # Safety
/// Pointer arguments (`a0..a5`) must be valid for the duration of the call as defined by each
/// `OP_*` contract in `x2l_common::bridge`.
#[unsafe(no_mangle)]
pub unsafe extern "win64" fn x2l_bridge_dispatch(
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) -> u64 {
    if opcode == OP_HANDSHAKE {
        HAS_GUEST_ACTIVITY.store(true, Ordering::SeqCst);
        if a0 != 0 {
            GUEST_IMAGE_BASE.store(a0, Ordering::Release);
        }
        let rt = runtime();
        rt.log_line("[BRIDGE] PE shim handshake OK (ABI v1)");
        return X2L_BRIDGE_MAGIC;
    }

    HAS_GUEST_ACTIVITY.store(true, Ordering::SeqCst);
    let rt = runtime();

    match opcode {
        OP_FLUSH_REPORT => {
            rt.flush_report();
            HRESULT_S_OK
        }
        OP_RECORD_STUB => {
            // SAFETY: PE shim passes valid static UTF-8 slices for DLL and symbol names.
            let dll = unsafe { read_guest_str(a0, a1) };
            let sym = unsafe { read_guest_str(a2, a3) };
            let caller_rva = a4;
            let default_ret = a5;
            if let Ok(mut rep) = rt.report.lock() {
                rep.record_missing_import(&dll, &sym, caller_rva);
            }
            rt.log_line(&format!(
                "[STUB] {dll}!{sym} caller_rva=0x{caller_rva:08x} -> 0x{default_ret:x}"
            ));
            default_ret
        }
        OP_RECORD_UNKNOWN_ORDINAL => {
            // SAFETY: PE shim passes valid static UTF-8 slice for DLL name.
            let dll = unsafe { read_guest_str(a0, a1) };
            let ordinal = a2 as u16;
            let caller_rva = a3;
            let default_ret = a4;
            if let Ok(mut rep) = rt.report.lock() {
                rep.record_unknown_ordinal(&dll, ordinal, caller_rva);
            }
            rt.log_line(&format!(
                "[STUB] {dll}!#{ordinal} caller_rva=0x{caller_rva:08x} -> 0x{default_ret:x}"
            ));
            default_ret
        }
        OP_RECORD_UNMAPPED_PATH => {
            // SAFETY: PE shim passes valid UTF-8 slice for path.
            let path = unsafe { read_guest_str(a0, a1) };
            let caller_rva = a2;
            if let Ok(mut rep) = rt.report.lock() {
                rep.record_unmapped_path(&path, caller_rva);
            }
            rt.log_line(&format!(
                "[UNMAPPED_PATH] '{path}' caller_rva=0x{caller_rva:08x}"
            ));
            0
        }
        OP_RECORD_VTABLE_PROBE => {
            // SAFETY: PE shim passes valid static UTF-8 slice for interface name.
            let iface = unsafe { read_guest_str(a0, a1) };
            let slot_index = a2 as u32;
            let caller_rva = a3;
            let default_ret = a4;
            if let Ok(mut rep) = rt.report.lock() {
                rep.record_unmapped_vtable_slot(&iface, slot_index, caller_rva);
            }
            rt.log_line(&format!(
                "[VTABLE_PROBE] {iface}[{slot_index}] caller_rva=0x{caller_rva:08x} -> 0x{default_ret:x}"
            ));
            default_ret
        }
        OP_KERNELX_GET_CONSOLE_TYPE => u64::from(rt.kernelx.console_type()),
        OP_KERNELX_GET_DEVICE_TYPE => u64::from(rt.kernelx.device_type()),
        OP_KERNELX_XMEM_ALLOC => rt.kernelx.xmem_alloc(a0 as usize, a1) as usize as u64,
        OP_KERNELX_XMEM_FREE => u64::from(rt.kernelx.xmem_free(a0 as usize, a1)),
        OP_KERNELX_TITLE_MEMORY_STATUS => {
            if a0 == 0 {
                return 0;
            }
            let status = rt.kernelx.title_memory_status();
            // SAFETY: Caller passes a writable `TitleMemoryStatus` pointer from guest code.
            unsafe {
                std::ptr::write_unaligned(a0 as *mut TitleMemoryStatus, status);
            }
            1
        }
        OP_KERNELX_VIRTUAL_ALLOC => {
            let (clean_alloc, clean_protect) = strip_virtual_alloc_flags(a2 as u32, a3 as u32);
            ((u64::from(clean_alloc)) << 32) | u64::from(clean_protect)
        }
        OP_XGR_INITIALIZE => {
            rt.log_line("[XGR] XGameRuntimeInitialize -> S_OK");
            rt.xgr.initialize()
        }
        OP_XGR_UNINITIALIZE => {
            rt.log_line("[XGR] XGameRuntimeUninitialize");
            rt.xgr.uninitialize();
            HRESULT_S_OK
        }
        OP_XGR_USER_GET_GAMERTAG => {
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, rt.xgr.gamertag()) }
        }
        OP_XGR_PACKAGE_MOUNT_ROOT => {
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, rt.xgr.package_mount_root()) }
        }
        OP_XGR_GAMESAVE_ROOT => {
            let path = rt.xgr.saves_dir().to_string_lossy().into_owned();
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, &path) }
        }
        OP_XGR_STORE_OFFLINE => {
            rt.log_line("[XGR] XStore offline -> E_NOTIMPL (no license spoofing)");
            rt.xgr.store_offline_result()
        }
        OP_D3D12X_CREATE_DEVICE => {
            let out_device_ptr = a0 as *mut *mut NullD3D12DeviceX;
            if out_device_ptr.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            let Some(vtbl) = rt.null_vtable() else {
                return HRESULT_E_NOTIMPL;
            };
            let dev = create_null_device(vtbl);
            // SAFETY: Caller supplied non-null writable COM output pointer `ppDevice`.
            unsafe {
                *out_device_ptr = dev;
            }
            rt.log_line(&format!(
                "[D3D12X] D3D12CreateDevice (null M1) -> S_OK (device=0x{:x})",
                dev as usize
            ));
            HRESULT_S_OK
        }
        OP_D3D12X_CAPTURE_SHADER => {
            if a0 == 0 || a1 == 0 {
                return HRESULT_E_INVALIDARG;
            }
            if let Some(dir) = &rt.shaders_raw_dir {
                let len = (a1 as usize).min(64 * 1024 * 1024);
                // SAFETY: Caller passes readable shader blob `(a0, len)`.
                let bytes = unsafe { std::slice::from_raw_parts(a0 as *const u8, len) };
                if let Ok(path) = capture_raw_shader_blob(dir, bytes) {
                    rt.log_line(&format!("[D3D12X] Captured raw shader: {}", path.display()));
                }
            }
            HRESULT_S_OK
        }
        _ => HRESULT_E_NOTIMPL,
    }
}

/// ELF `.init_array` constructor: publishes the ASLR-resolved address of [`x2l_bridge_dispatch`]
/// into `X2L_BRIDGE_FN` when launched under `x2linux run` (`X2L_BRIDGE_ENABLED=1`).
extern "C" fn x2l_core_ctor() {
    if std::env::var(ENV_BRIDGE_ENABLED).as_deref() != Ok("1") {
        return;
    }
    // Pre-initialize runtime state (and W^X vtable thunks) during ELF init before Wine enables
    // Syscall User Dispatch (SUD) on guest PE threads.
    let rt = runtime();
    let _ = rt.null_vtable();

    let addr = x2l_bridge_dispatch as *const c_void as usize;
    if let (Ok(k), Ok(v)) = (
        CString::new(ENV_BRIDGE_FN),
        CString::new(format!("0x{addr:x}")),
    ) {
        // SAFETY: Runs single-threaded inside ld-linux `.init_array` before `main()` starts,
        // updating process `environ` so Wine propagates `X2L_BRIDGE_FN` into the PE process.
        unsafe {
            setenv(k.as_ptr(), v.as_ptr(), 1);
        }
    }
}

/// ELF `.fini_array` destructor: flushes `report.json` when the guest process exits (and skips
/// unrelated Wine helper processes that never touched the X2Linux bridge).
extern "C" fn x2l_core_dtor() {
    if !HAS_GUEST_ACTIVITY.load(Ordering::SeqCst) {
        return;
    }
    if let Some(rt) = RUNTIME.get() {
        rt.flush_report();
    }
}

#[used]
#[unsafe(link_section = ".init_array")]
static INIT_ARRAY: extern "C" fn() = x2l_core_ctor;

#[used]
#[unsafe(link_section = ".fini_array")]
static FINI_ARRAY: extern "C" fn() = x2l_core_dtor;

/// C-ABI entry point returning the core runtime bridge ABI version.
#[unsafe(no_mangle)]
pub extern "C" fn x2l_core_abi_version() -> u32 {
    X2L_UNIX_CALL_ABI_VERSION
}

/// C-ABI entry point returning the `report.json` telemetry schema version.
#[unsafe(no_mangle)]
pub extern "C" fn x2l_core_report_schema_version() -> u32 {
    x2l_common::report::REPORT_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_dispatch_handshake_kernelx_xgr_and_null_d3d12x() {
        // SAFETY: Test exercises valid opcodes and stack-owned output pointers in-process.
        unsafe {
            assert_eq!(
                x2l_bridge_dispatch(OP_HANDSHAKE, 0, 0, 0, 0, 0, 0),
                X2L_BRIDGE_MAGIC
            );
            assert_eq!(
                x2l_bridge_dispatch(OP_KERNELX_GET_CONSOLE_TYPE, 0, 0, 0, 0, 0, 0),
                u64::from(x2l_hle_kernelx::CONSOLE_TYPE_SERIES_X)
            );
            let ptr = x2l_bridge_dispatch(OP_KERNELX_XMEM_ALLOC, 256, 0, 0, 0, 0, 0);
            assert_ne!(ptr, 0);
            assert_eq!(
                x2l_bridge_dispatch(OP_KERNELX_XMEM_FREE, ptr, 0, 0, 0, 0, 0),
                1
            );
            assert_eq!(
                x2l_bridge_dispatch(OP_XGR_INITIALIZE, 0, 0, 0, 0, 0, 0),
                HRESULT_S_OK
            );
            assert_eq!(
                x2l_bridge_dispatch(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0),
                HRESULT_E_NOTIMPL
            );
            let mut dev: *mut NullD3D12DeviceX = std::ptr::null_mut();
            assert_eq!(
                x2l_bridge_dispatch(
                    OP_D3D12X_CREATE_DEVICE,
                    (&mut dev as *mut *mut NullD3D12DeviceX) as usize as u64,
                    0,
                    0,
                    0,
                    0,
                    0
                ),
                HRESULT_S_OK
            );
            assert!(!dev.is_null());
            let release: extern "win64" fn(*mut NullD3D12DeviceX) -> u32 =
                std::mem::transmute(*(*dev).vtbl.add(2));
            assert_eq!(release(dev), 0);
        }
    }
}
