//! Native Linux core runtime shared library (`libx2l_core.so`).
//!
//! Loaded in-process into Wine via `LD_PRELOAD`. On ELF initialization (`.init_array`), when
//! `X2L_BRIDGE_ENABLED=1`, exports the process-local address of [`x2l_bridge_dispatch`]
//! (`extern "win64"`) through the `X2L_BRIDGE_FN` environment variable so thin Win64 PE shims
//! (`kernelx.dll`, `xgameruntime.dll`, `d3d12_xs.dll`) invoke native Linux Rust HLE, Vulkan,
//! and telemetry directly in-process with zero Wine-internal header coupling.

mod gdk_com;

use gdk_com::{
    AppNotifyRegistry, OP_CHAD_ACTIVATE_STATUS, OP_CHAD_CONNECT_STREAM_LIMIT, OP_CHAD_DECOMPRESS,
    OP_CHAD_SUBMIT_STATUS, OP_XAPU_CONNECT_DECODER, OP_XAPU_DISCONNECT_DECODER, OP_XAPU_POP,
    OP_XAPU_RESULT_COUNTS, OP_XAPU_SUBMIT, OP_XG_COMPUTE_OPTIMAL_TILE_MODE,
    OP_XG_CREATE_TYPED_COMPUTER, acphal_create, dstorage_get_factory, gameinput_create,
    xg_compute_optimal_tile_mode, xg_create_texture_computer, xg_create_typed_computer,
    xg_suggest_swizzle_mode,
};
use std::collections::VecDeque;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use x2l_audio::{AudioCoprocState, AudioOutput, DEFAULT_SAMPLE_RATE_HZ};
use x2l_common::bridge::*;
use x2l_common::{
    DeviceType, ExceptionRegisters, GpuTelemetry, HostRuntimeMetadata, TelemetryReport,
};
use x2l_gpu_vk::com::{
    D3D12ResourceDesc, create_vulkan_com_device, exec_native, start_native_gpu_worker,
};
use x2l_gpu_vk::d3d12x_ext::{
    D3D12ConfigureVirtualMemoryXParams, D3D12VersionedRootSignatureDesc,
    configure_virtual_memory_x, serialize_versioned_root_signature,
};
use x2l_gpu_vk::vulkan::{VulkanBackend, VulkanBackendConfig};
use x2l_gpu_vk::{
    NULL_DEVICE_VTABLE_SLOTS, build_null_device_vtable, capture_raw_shader_blob, create_null_device,
};
use x2l_hle_kernelx::xmem::{ArenaOpResult, XMemArena, XMemBasicInformation, XMemWorkingSetPrefix};
use x2l_hle_kernelx::{KernelxState, TitleMemoryStatus, strip_virtual_alloc_flags};
use x2l_hle_xgr::{PersistentLocalStorageSpaceInfo, XgrState};
use x2l_input::GameInputState;

/// ABI version shared between `x2l-pe-shims` and `libx2l_core.so`.
pub const X2L_UNIX_CALL_ABI_VERSION: u32 = 1;

unsafe extern "C" {
    fn setenv(name: *const c_char, value: *const c_char, overwrite: c_int) -> c_int;
}

/// Process-global runtime state owned by `libx2l_core.so`.
struct CoreRuntime {
    kernelx: KernelxState,
    xmem: Arc<XMemArena>,
    xgr: Arc<XgrState>,
    audio_coproc: AudioCoprocState,
    audio_output: OnceLock<AudioOutput>,
    gameinput: Arc<GameInputState>,
    appnotify: AppNotifyRegistry,
    report: Mutex<TelemetryReport>,
    debug_strings: Mutex<VecDeque<String>>,
    report_path: Option<PathBuf>,
    log_path: Option<PathBuf>,
    shaders_raw_dir: Option<PathBuf>,
    merged_dir: Option<PathBuf>,
    saves_dir: PathBuf,
    gpu_backend_mode: String,
    trace_hle: bool,
    null_device_vtbl: OnceLock<&'static [usize; NULL_DEVICE_VTABLE_SLOTS]>,
    vulkan_backend: OnceLock<Arc<VulkanBackend>>,
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
        let merged_dir = std::env::var("X2L_MERGED_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::current_dir().ok());

        let gpu_backend_mode = std::env::var(ENV_GPU_BACKEND)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "null".into());
        let trace_hle = std::env::var(ENV_TRACE_HLE)
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        let title_slug = title_id.clone();
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
            network_mode: std::env::var("X2L_NETWORK_MODE")
                .unwrap_or_else(|_| "unrestricted".into()),
        };
        report.gpu = GpuTelemetry {
            backend: gpu_backend_mode.clone(),
            present_surface: "headless".to_owned(),
            ..Default::default()
        };

        Self {
            kernelx: KernelxState::new(device),
            xmem: Arc::new(XMemArena::new(device)),
            xgr: Arc::new(XgrState::with_device_and_title(
                gamertag,
                saves_dir.clone(),
                device,
                &title_slug,
            )),
            audio_coproc: AudioCoprocState::new(),
            audio_output: OnceLock::new(),
            gameinput: Arc::new(GameInputState::new()),
            appnotify: AppNotifyRegistry::new(),
            report: Mutex::new(report),
            debug_strings: Mutex::new(VecDeque::with_capacity(16)),
            report_path,
            log_path,
            shaders_raw_dir,
            merged_dir,
            saves_dir,
            gpu_backend_mode,
            trace_hle,
            null_device_vtbl: OnceLock::new(),
            vulkan_backend: OnceLock::new(),
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
        let Ok(mut guard) = self.report.lock() else {
            return;
        };
        if let Some(vk) = self.vulkan_backend.get() {
            let snap = vk.telemetry_snapshot();
            guard.gpu = GpuTelemetry {
                backend: snap.backend,
                present_surface: snap.present_surface,
                device_name: snap.device_name,
                driver_info: snap.driver_info,
                external_memory_host: snap.external_memory_host,
                descriptor_buffer: snap.descriptor_buffer,
                userfaultfd_mode: snap.userfaultfd_mode,
                shaders_compiled: snap.shaders_compiled,
                frames_presented: snap.frames_presented,
            };
        }
        let Ok(json) = guard.to_json_pretty() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, json);
    }

    fn vulkan_backend(&self) -> Result<Arc<VulkanBackend>, String> {
        if let Some(vk) = self.vulkan_backend.get() {
            return Ok(Arc::clone(vk));
        }
        let vk = exec_native(|| {
            let cfg = VulkanBackendConfig::from_env();
            VulkanBackend::new(cfg)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        })?;
        let _ = self.vulkan_backend.set(Arc::clone(&vk));
        Ok(vk)
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
    let (iface, decoded_slot) = x2l_gpu_vk::decode_d3d12_vtable_probe(slot_index);
    if let Ok(mut rep) = rt.report.lock() {
        rep.record_unmapped_vtable_slot(iface, decoded_slot, caller_rva);
    }
    rt.log_line(&format!(
        "[D3D12X_VTABLE] {iface}[{decoded_slot}] caller_rva=0x{caller_rva:08x} -> S_OK"
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

/// Fixed-size `#[repr(C)]` fatal VEH crash snapshot passed by `shim-common` in `OP_RECORD_EXCEPTION`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CrashSnapshot {
    code: u32,
    has_control: u32,
    has_integer: u32,
    bt_count: u32,
    address: u64,
    rva: u64,
    module_ptr: u64,
    module_len: u64,
    rip: u64,
    rsp: u64,
    rax: u64,
    rcx: u64,
    rdx: u64,
    r8: u64,
    r9: u64,
    bt_rvas: [u64; 8],
}

/// Bridge payload for `OP_XGR_GAMESAVE_WRITE_BLOB`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GameSaveWriteReq {
    container_ptr: u64,
    container_len: u64,
    blob_ptr: u64,
    blob_len: u64,
    data_ptr: u64,
    data_len: u64,
}

/// Bridge payload for `OP_XGR_GAMESAVE_READ_BLOB`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GameSaveReadReq {
    container_ptr: u64,
    container_len: u64,
    blob_ptr: u64,
    blob_len: u64,
    out_ptr: u64,
    out_cap: u64,
    out_size: u64,
}

/// Bridge payload for `OP_XGR_GAMESAVE_ENUM_BLOBS`.
/// Writes NUL-separated `name\0` + 4-byte LE size entries into `(out_ptr, out_cap)` and sets `out_count`/`out_bytes`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GameSaveEnumReq {
    container_ptr: u64,
    container_len: u64,
    prefix_ptr: u64,
    prefix_len: u64,
    out_ptr: u64,
    out_cap: u64,
    out_count: u64,
    out_bytes: u64,
}

/// Single version-independent `extern "win64"` dispatch entrypoint called directly in-process
/// by `crates/x2l-pe-shims` DLLs (`kernelx.dll`, `xgameruntime.dll`, `d3d12_xs.dll`).
///
/// # Safety
/// Pointer arguments (`a0..a5`) must be valid for the duration of the call as defined by each
/// `OP_*` contract in `x2l_common::bridge`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "win64" fn x2l_bridge_dispatch(
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
    caller_rva: u64,
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

    if opcode == OP_TRACE_HLE_LINE {
        // SAFETY: PE shim passes valid UTF-8 trace line `(a0, a1)`.
        let line = unsafe { read_guest_str(a0, a1) };
        rt.log_line(&format!("[HLE] {line}"));
        return HRESULT_S_OK;
    }
    if opcode == OP_RECORD_DEBUG_STRING {
        // SAFETY: PE shim passes valid UTF-8 debug string `(a0, a1)`.
        let raw = unsafe { read_guest_str(a0, a1) };
        let trimmed = raw.trim_end_matches(['\r', '\n']).to_owned();
        if !trimmed.is_empty() {
            if let Ok(mut ring) = rt.debug_strings.lock() {
                if ring.len() >= 16 {
                    ring.pop_front();
                }
                ring.push_back(trimmed.clone());
            }
            if rt.trace_hle {
                rt.log_line(&format!("[DEBUGSTR] {trimmed}"));
            }
        }
        return HRESULT_S_OK;
    }
    if opcode == OP_RECORD_MISSING_PROC {
        // SAFETY: PE shim passes valid UTF-8 DLL and symbol slices.
        let dll = unsafe { read_guest_str(a0, a1) };
        let sym = unsafe { read_guest_str(a2, a3) };
        let rva = a4;
        let ordinal = a5 as u16;
        if ordinal != 0 && sym.is_empty() {
            if let Ok(mut rep) = rt.report.lock() {
                rep.record_unknown_ordinal(&dll, ordinal, rva);
            }
            rt.log_line(&format!(
                "[MISSING_PROC] {dll}!#{ordinal} @ rva=0x{rva:08x}"
            ));
        } else {
            if let Ok(mut rep) = rt.report.lock() {
                rep.record_missing_import(&dll, &sym, rva);
            }
            rt.log_line(&format!("[MISSING_PROC] {dll}!{sym} @ rva=0x{rva:08x}"));
        }
        return HRESULT_S_OK;
    }
    if opcode == OP_RECORD_EXCEPTION {
        let (code, addr, rva, module, regs, bt_slice) = if a0 != 0 && a1 == 0 && a2 == 0 {
            // New M3.4 fixed-size `CrashSnapshot` path (`a0 = *const CrashSnapshot`).
            // SAFETY: `shim-common` passes a stack-allocated readable `CrashSnapshot`.
            let snap = unsafe { *(a0 as *const CrashSnapshot) };
            // SAFETY: `snap.module_ptr`/`snap.module_len` point to a stack UTF-8 slice.
            let m = unsafe { read_guest_str(snap.module_ptr, snap.module_len) };
            let regs = if snap.has_control != 0 || snap.has_integer != 0 {
                Some(ExceptionRegisters {
                    rip: (snap.has_control != 0).then(|| format!("0x{:016x}", snap.rip)),
                    rsp: (snap.has_control != 0).then(|| format!("0x{:016x}", snap.rsp)),
                    rax: (snap.has_integer != 0).then(|| format!("0x{:016x}", snap.rax)),
                    rcx: (snap.has_integer != 0).then(|| format!("0x{:016x}", snap.rcx)),
                    rdx: (snap.has_integer != 0).then(|| format!("0x{:016x}", snap.rdx)),
                    r8: (snap.has_integer != 0).then(|| format!("0x{:016x}", snap.r8)),
                    r9: (snap.has_integer != 0).then(|| format!("0x{:016x}", snap.r9)),
                })
            } else {
                None
            };
            let bt_len = (snap.bt_count as usize).min(snap.bt_rvas.len());
            (
                snap.code,
                snap.address,
                snap.rva,
                m,
                regs,
                snap.bt_rvas[..bt_len].to_vec(),
            )
        } else {
            // Legacy M3.3 scalar-arg fallback.
            // SAFETY: PE shim passes valid UTF-8 module name `(a3, a4)`.
            let m = unsafe { read_guest_str(a3, a4) };
            (a0 as u32, a1, a2, m, None, Vec::new())
        };
        let debug_tail: Vec<String> = rt
            .debug_strings
            .lock()
            .map(|g| g.iter().cloned().collect())
            .unwrap_or_default();
        if let Ok(mut rep) = rt.report.lock() {
            rep.record_exception_detailed(code, addr, rva, &module, regs, &bt_slice, &debug_tail);
        }
        rt.log_line(&format!(
            "[EXCEPTION] code=0x{code:08x} module={module} addr=0x{addr:016x} rva=0x{rva:08x} bt_frames={}",
            bt_slice.len()
        ));
        rt.flush_report();
        return HRESULT_S_OK;
    }

    let ret = dispatch_bridge_op(rt, opcode, a0, a1, a2, a3, a4, a5);
    if rt.trace_hle && opcode != OP_FLUSH_REPORT {
        rt.log_line(&format!(
            "[HLE] {}(a0=0x{a0:x}, a1=0x{a1:x}, a2=0x{a2:x}, a3=0x{a3:x}) -> ret=0x{ret:x} @ rva=0x{caller_rva:08x}",
            bridge_opcode_name(opcode)
        ));
    }
    ret
}

fn bridge_opcode_name(opcode: u32) -> &'static str {
    match opcode {
        OP_HANDSHAKE => "bridge!Handshake",
        OP_FLUSH_REPORT => "bridge!FlushReport",
        OP_RECORD_STUB => "bridge!RecordStub",
        OP_RECORD_UNKNOWN_ORDINAL => "bridge!RecordUnknownOrdinal",
        OP_RECORD_UNMAPPED_PATH => "bridge!RecordUnmappedPath",
        OP_RECORD_VTABLE_PROBE => "bridge!RecordVtableProbe",
        OP_KERNELX_GET_CONSOLE_TYPE => "kernelx!GetConsoleType",
        OP_KERNELX_GET_DEVICE_TYPE => "kernelx!XSystemGetDeviceType",
        OP_KERNELX_XMEM_ALLOC => "kernelx!XMemAlloc",
        OP_KERNELX_XMEM_FREE => "kernelx!XMemFree",
        OP_KERNELX_TITLE_MEMORY_STATUS => "kernelx!TitleMemoryStatus",
        OP_KERNELX_VIRTUAL_ALLOC => "kernelx!XMemVirtualAlloc",
        OP_XMEM_ALLOC_DEFAULT => "xmem!XMemAllocDefault",
        OP_XMEM_FREE_DEFAULT => "xmem!XMemFreeDefault",
        OP_XMEM_ALLOC_PHYSICAL_PAGES => "xmem!XMemAllocatePhysicalPages",
        OP_XMEM_FREE_PHYSICAL_PAGES => "xmem!XMemFreePhysicalPages",
        OP_XMEM_MAP_PHYSICAL_PAGES => "xmem!XMemMapPhysicalPages",
        OP_XMEM_MAP_PHYSICAL_PAGES_SCATTER => "xmem!XMemMapPhysicalPagesScatter",
        OP_XMEM_VIRTUAL_ALLOC => "xmem!XMemVirtualAlloc",
        OP_XMEM_VIRTUAL_QUERY => "xmem!XMemVirtualQuery",
        OP_XMEM_TRANSFER_MEMORY => "xmem!XMemTransferMemory",
        OP_XMEM_GET_WORKING_SET_STATS => "xmem!XMemGetWorkingSetStatistics",
        OP_XMEM_IS_TOOLS_MEM_AVAILABLE => "xmem!XMemIsToolsMemAvailable",
        OP_XMEM_TRY_WIN32_VALLOC => "xmem!Win32VirtualAlloc",
        OP_XMEM_TRY_WIN32_VFREE => "xmem!Win32VirtualFree",
        OP_XMEM_TRY_WIN32_VQUERY => "xmem!Win32VirtualQuery",
        OP_XGR_INITIALIZE => "xgameruntime!XGameRuntimeInitialize",
        OP_XGR_UNINITIALIZE => "xgameruntime!XGameRuntimeUninitialize",
        OP_XGR_USER_GET_GAMERTAG => "xgameruntime!XUserGetGamertag",
        OP_XGR_PACKAGE_MOUNT_ROOT => "xgameruntime!XPackageMountRoot",
        OP_XGR_GAMESAVE_ROOT => "xgameruntime!XGameSaveRoot",
        OP_XGR_STORE_OFFLINE => "xgameruntime!XStoreOffline",
        OP_XGR_GET_DEVICE_TYPE => "xgameruntime!XSystemGetDeviceType",
        OP_XGR_GET_TITLE_ID => "xgameruntime!XGameGetXboxTitleId",
        OP_XGR_GET_CONSOLE_ID => "xgameruntime!XSystemGetConsoleId",
        OP_XGR_GET_SANDBOX_ID => "xgameruntime!XSystemGetXboxLiveSandboxId",
        OP_XGR_GET_APP_DEVICE_ID => "xgameruntime!XSystemGetAppSpecificDeviceId",
        OP_XGR_PLS_GET_PATH => "xgameruntime!XPersistentLocalStorageGetPath",
        OP_XGR_PLS_GET_SPACE_INFO => "xgameruntime!XPersistentLocalStorageGetSpaceInfo",
        OP_XGR_GAMESAVE_WRITE_BLOB => "xgameruntime!XGameSaveWriteBlob",
        OP_XGR_GAMESAVE_READ_BLOB => "xgameruntime!XGameSaveReadBlob",
        OP_XGR_GAMESAVE_ENUM_BLOBS => "xgameruntime!XGameSaveEnumBlobs",
        OP_D3D12X_CREATE_DEVICE => "d3d12_xs!D3D12CreateDevice",
        OP_D3D12X_CAPTURE_SHADER => "d3d12_xs!CaptureShader",
        OP_D3D12X_XBOX_CREATE_DEVICE => "d3d12_xs!D3D12XboxCreateDevice",
        OP_D3D12X_CONFIGURE_VIRTUAL_MEMORY => "d3d12_xs!D3D12ConfigureVirtualMemoryX",
        OP_D3D12X_SERIALIZE_VERSIONED_ROOT_SIG => "d3d12_xs!D3D12SerializeVersionedRootSignature",
        OP_XG_CREATE_TEXTURE_COMPUTER => "xg_xs!XGCreateTextureComputer",
        OP_XG_SUGGEST_SWIZZLE_MODE => "xg_xs!XGSuggestSwizzleMode",
        OP_XG_CREATE_TYPED_COMPUTER => "xg_xs!XGCreateTypedComputer",
        OP_XG_COMPUTE_OPTIMAL_TILE_MODE => "xg_xs!XGComputeOptimalTileMode",
        OP_DSTORAGE_GET_FACTORY => "XCRDAPI!DStorageGetFactory",
        OP_GAMEINPUT_CREATE => "GameInput!GameInputCreate",
        OP_ACPHAL_CREATE => "AcpHal!AcpHalCreate",
        OP_ACPHAL_ALLOC_SHAPE_CONTEXTS => "AcpHal!AcpHalAllocateShapeContexts",
        OP_ACPHAL_RELEASE_SHAPE_CONTEXTS => "AcpHal!AcpHalReleaseShapeContexts",
        OP_APU_CREATE_HEAP => "AcpHal!ApuCreateHeap",
        OP_APU_ALLOC => "AcpHal!ApuAlloc",
        OP_APU_FREE => "AcpHal!ApuFree",
        OP_APU_MAP_APU_ADDRESS => "AcpHal!ApuMapApuAddress",
        OP_APU_MAP_VIRTUAL_ADDRESS => "AcpHal!ApuMapVirtualAddress",
        OP_APU_IS_VIRTUAL_ADDRESS_VALID => "AcpHal!ApuIsVirtualAddressValid",
        OP_XAPU_CONNECT => "xapu!XApuConnect",
        OP_XAPU_DISCONNECT => "xapu!XApuDisconnect",
        OP_XAPU_ENQUEUE_COMMAND => "xapu!XApuEnqueueCommand",
        OP_XAPU_DEQUEUE_RESULT => "xapu!XApuDequeueResult",
        OP_XAPU_GET_RESULT_COUNT => "xapu!XApuGetResultCount",
        OP_CHAD_CONNECT => "chad!XDspConnectWithMaximumStreamLimit",
        OP_CHAD_DISCONNECT => "chad!XDspDisconnect",
        OP_CHAD_ACTIVATE => "chad!XDspActivate",
        OP_CHAD_DEACTIVATE => "chad!XDspDeactivate",
        OP_CHAD_SUBMIT_COMMAND => "chad!XDspSubmitCommand",
        OP_CHAD_DECOMPRESS => "chad!X2LChadDecompress",
        OP_XAPU_CONNECT_DECODER => "xapu!XApuConnect",
        OP_XAPU_DISCONNECT_DECODER => "xapu!XApuDisconnect",
        OP_XAPU_SUBMIT => "xapu!XApuEnqueueCommand",
        OP_XAPU_POP => "xapu!XApuDequeueResult",
        OP_XAPU_RESULT_COUNTS => "xapu!XApuGetResultCount",
        OP_CHAD_CONNECT_STREAM_LIMIT => "chad!XDspConnectWithMaximumStreamLimit",
        OP_CHAD_ACTIVATE_STATUS => "chad!XDspActivate",
        OP_CHAD_SUBMIT_STATUS => "chad!XDspSubmitCommand",
        OP_PSM_REGISTER_APP_STATE => "appnotify!RegisterAppStateChangeNotification",
        OP_PSM_UNREGISTER_APP_STATE => "appnotify!UnregisterAppStateChangeNotification",
        OP_PSM_REGISTER_CONSTRAINED => "appnotify!RegisterAppConstrainedChangeNotification",
        OP_PSM_UNREGISTER_CONSTRAINED => "appnotify!UnregisterAppConstrainedChangeNotification",
        _ => "bridge!UnknownOp",
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_bridge_op(
    rt: &CoreRuntime,
    opcode: u32,
    a0: u64,
    a1: u64,
    a2: u64,
    a3: u64,
    a4: u64,
    a5: u64,
) -> u64 {
    match opcode {
        OP_FLUSH_REPORT => {
            rt.flush_report();
            if let Some(vk) = rt.vulkan_backend.get() {
                let vk = Arc::clone(vk);
                exec_native(move || vk.shutdown_wsi());
            }
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
        // ---- xmem.dll (Linux memfd/mmap physical & scatter arena on SUD-free native worker) ----
        OP_XMEM_ALLOC_DEFAULT => rt.xmem.alloc(a0 as usize, a1) as usize as u64,
        OP_XMEM_FREE_DEFAULT => u64::from(rt.xmem.free(a0 as usize, a1)),
        OP_XMEM_ALLOC_PHYSICAL_PAGES => {
            let flags = a0;
            let count_ptr = a1 as *mut usize;
            let out_pfns_ptr = a2 as *mut usize;
            if count_ptr.is_null() || out_pfns_ptr.is_null() {
                return 0;
            }
            // SAFETY: Caller passes readable/writable `count_ptr` and `out_pfns_ptr` array.
            let count = unsafe { *count_ptr };
            let arena = Arc::clone(&rt.xmem);
            match exec_native(move || arena.allocate_physical_pages(flags, count, None)) {
                Some(pfns) => {
                    // SAFETY: Caller allocated space for at least `count` `usize` PFNs.
                    unsafe {
                        let out = std::slice::from_raw_parts_mut(out_pfns_ptr, pfns.len());
                        out.copy_from_slice(&pfns);
                        *count_ptr = pfns.len();
                    }
                    1
                }
                None => {
                    // SAFETY: Zero caller count on failure per GDK contract.
                    unsafe { *count_ptr = 0 };
                    0
                }
            }
        }
        OP_XMEM_FREE_PHYSICAL_PAGES => {
            let count = a0 as usize;
            let pfns_ptr = a1 as *const usize;
            if count == 0 {
                return 1;
            }
            if pfns_ptr.is_null() {
                return 0;
            }
            // SAFETY: Caller passes readable `count`-element PFN slice.
            let pfns = unsafe { std::slice::from_raw_parts(pfns_ptr, count) }.to_vec();
            let arena = Arc::clone(&rt.xmem);
            u64::from(exec_native(move || arena.free_physical_pages(&pfns)))
        }
        OP_XMEM_MAP_PHYSICAL_PAGES => {
            let base = a0 as usize;
            let count = a1 as usize;
            let pfns_opt = if a2 == 0 {
                None
            } else {
                // SAFETY: Caller passes readable `count`-element PFN slice.
                Some(unsafe { std::slice::from_raw_parts(a2 as *const usize, count) }.to_vec())
            };
            let arena = Arc::clone(&rt.xmem);
            exec_native(move || arena.map_physical_pages(base, count, pfns_opt.as_deref()))
                .unwrap_or(0) as u64
        }
        OP_XMEM_MAP_PHYSICAL_PAGES_SCATTER => {
            let addrs_ptr = a0 as *const usize;
            let count = a1 as usize;
            let pfns_ptr = a2 as *const usize;
            if count == 0 {
                return 1;
            }
            if addrs_ptr.is_null() || pfns_ptr.is_null() {
                return 0;
            }
            // SAFETY: Caller passes readable `count`-element address and PFN arrays.
            let addrs = unsafe { std::slice::from_raw_parts(addrs_ptr, count) }.to_vec();
            let pfns = unsafe { std::slice::from_raw_parts(pfns_ptr, count) }.to_vec();
            let arena = Arc::clone(&rt.xmem);
            u64::from(exec_native(move || {
                arena.map_physical_pages_scatter(&addrs, &pfns)
            }))
        }
        OP_XMEM_VIRTUAL_ALLOC => {
            let arena = Arc::clone(&rt.xmem);
            let base = a0 as usize;
            let size = a1 as usize;
            let alloc_type = a2 as u32;
            let flags = a3;
            let protect = a4 as u32;
            exec_native(move || arena.virtual_alloc(base, size, alloc_type, flags, protect))
                .unwrap_or(0) as u64
        }
        OP_XMEM_VIRTUAL_QUERY => {
            let addr = a1 as usize;
            let out_ptr = a2 as *mut XMemBasicInformation;
            if out_ptr.is_null() {
                return 0;
            }
            let arena = Arc::clone(&rt.xmem);
            match exec_native(move || arena.virtual_query(addr)) {
                Some(info) => {
                    // SAFETY: Caller passed writable 64-byte `XMemBasicInformation*`.
                    unsafe { std::ptr::write_unaligned(out_ptr, info) };
                    1
                }
                None => 0,
            }
        }
        OP_XMEM_TRANSFER_MEMORY => u64::from(rt.xmem.transfer_memory(a0 as u32, a1 as usize)),
        OP_XMEM_GET_WORKING_SET_STATS => {
            let out_ptr = a1 as *mut XMemWorkingSetPrefix;
            if out_ptr.is_null() {
                return 0;
            }
            let stats = rt.xmem.working_set_statistics();
            // SAFETY: Writes common 16-byte `(title_limit, title_used)` prefix.
            unsafe { std::ptr::write_unaligned(out_ptr, stats) };
            1
        }
        OP_XMEM_IS_TOOLS_MEM_AVAILABLE => u64::from(rt.xmem.is_tools_mem_available()),
        OP_XMEM_TRY_WIN32_VALLOC => {
            let arena = Arc::clone(&rt.xmem);
            let addr = a0 as usize;
            let size = a1 as usize;
            let alloc_type = a2 as u32;
            let protect = a3 as u32;
            let out_ptr = a4 as *mut u64;
            match exec_native(move || {
                arena.try_win32_virtual_alloc(addr, size, alloc_type, protect)
            }) {
                ArenaOpResult::NotOwned => 0,
                ArenaOpResult::Ok(base) => {
                    if !out_ptr.is_null() {
                        // SAFETY: Caller passes writable `u64` result slot.
                        unsafe { *out_ptr = base as u64 };
                    }
                    1
                }
                ArenaOpResult::Error(err) => {
                    if !out_ptr.is_null() {
                        // SAFETY: Caller passes writable `u64` error slot.
                        unsafe { *out_ptr = u64::from(err) };
                    }
                    2
                }
            }
        }
        OP_XMEM_TRY_WIN32_VFREE => {
            let arena = Arc::clone(&rt.xmem);
            let addr = a0 as usize;
            let size = a1 as usize;
            let free_type = a2 as u32;
            let out_err = a3 as *mut u32;
            match exec_native(move || arena.try_win32_virtual_free(addr, size, free_type)) {
                ArenaOpResult::NotOwned => 0,
                ArenaOpResult::Ok(()) => 1,
                ArenaOpResult::Error(err) => {
                    if !out_err.is_null() {
                        // SAFETY: Caller passes writable `u32` error slot.
                        unsafe { *out_err = err };
                    }
                    2
                }
            }
        }
        OP_XMEM_TRY_WIN32_VQUERY => {
            let arena = Arc::clone(&rt.xmem);
            let addr = a0 as usize;
            let out_mbi = a1 as *mut u8;
            let buf_len = a2 as usize;
            let out_next = a3 as *mut u64;
            let (hit, next_res) = exec_native(move || {
                (
                    arena.virtual_query(addr),
                    arena.next_reservation_after(addr),
                )
            });
            if !out_next.is_null() {
                // SAFETY: Caller passes optional writable next-reservation output.
                unsafe { *out_next = next_res.unwrap_or(0) as u64 };
            }
            match hit {
                None => 0,
                Some(info) => {
                    if out_mbi.is_null() || buf_len < 48 {
                        2
                    } else {
                        // SAFETY: First 48 bytes of `XMemBasicInformation` are exact Win64
                        // `MEMORY_BASIC_INFORMATION`; caller verified `buf_len >= 48`.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                (&info as *const XMemBasicInformation).cast::<u8>(),
                                out_mbi,
                                48,
                            );
                        }
                        48
                    }
                }
            }
        }
        OP_XGR_INITIALIZE => {
            rt.log_line("[XGR] XGameRuntimeInitialize -> S_OK");
            let xgr = Arc::clone(&rt.xgr);
            exec_native(move || xgr.initialize())
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
        OP_XGR_GET_DEVICE_TYPE => u64::from(rt.xgr.gdk_device_type()),
        OP_XGR_GET_TITLE_ID => u64::from(rt.xgr.xbox_title_id()),
        OP_XGR_GET_CONSOLE_ID => {
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, rt.xgr.console_id()) }
        }
        OP_XGR_GET_SANDBOX_ID => {
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, rt.xgr.sandbox_id()) }
        }
        OP_XGR_GET_APP_DEVICE_ID => {
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, rt.xgr.app_specific_device_id()) }
        }
        OP_XGR_PLS_GET_PATH => {
            let path = rt.xgr.pls_guest_path();
            if a2 != 0 {
                // SAFETY: Caller passes optional writable `usize` out-length pointer.
                unsafe { *(a2 as *mut usize) = path.len() + 1 };
            }
            if a0 == 0 || a1 == 0 {
                return HRESULT_S_OK;
            }
            // SAFETY: Caller passes writable UTF-8 buffer `(a0, a1)`.
            unsafe { write_guest_cstr(a0, a1, path) }
        }
        OP_XGR_PLS_GET_SPACE_INFO => {
            let out_ptr = a0 as *mut PersistentLocalStorageSpaceInfo;
            if out_ptr.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            let xgr = Arc::clone(&rt.xgr);
            let info = exec_native(move || xgr.pls_space_info());
            // SAFETY: Caller passes writable 32-byte `PersistentLocalStorageSpaceInfo*`.
            unsafe { std::ptr::write_unaligned(out_ptr, info) };
            HRESULT_S_OK
        }
        OP_XGR_GAMESAVE_WRITE_BLOB => {
            if a0 == 0 {
                return HRESULT_E_INVALIDARG;
            }
            // SAFETY: Caller passes readable `GameSaveWriteReq` and valid UTF-8/byte slices.
            let req = unsafe { *(a0 as *const GameSaveWriteReq) };
            let container = unsafe { read_guest_str(req.container_ptr, req.container_len) };
            let blob = unsafe { read_guest_str(req.blob_ptr, req.blob_len) };
            let data = if req.data_ptr == 0 || req.data_len == 0 {
                Vec::new()
            } else {
                let len = (req.data_len as usize).min(64 * 1024 * 1024);
                // SAFETY: Caller passes readable blob payload slice.
                unsafe { std::slice::from_raw_parts(req.data_ptr as *const u8, len) }.to_vec()
            };
            let xgr = Arc::clone(&rt.xgr);
            match exec_native(move || xgr.write_save_blob(&container, &blob, &data)) {
                Ok(()) => HRESULT_S_OK,
                Err(_) => HRESULT_E_INVALIDARG,
            }
        }
        OP_XGR_GAMESAVE_READ_BLOB => {
            if a0 == 0 {
                return HRESULT_E_INVALIDARG;
            }
            let req_ptr = a0 as *mut GameSaveReadReq;
            // SAFETY: Caller passes readable/writable `GameSaveReadReq`.
            let req = unsafe { *req_ptr };
            let container = unsafe { read_guest_str(req.container_ptr, req.container_len) };
            let blob = unsafe { read_guest_str(req.blob_ptr, req.blob_len) };
            let xgr = Arc::clone(&rt.xgr);
            match exec_native(move || xgr.read_save_blob(&container, &blob)) {
                Ok(bytes) => {
                    // SAFETY: Write exact blob byte size into `req.out_size`.
                    unsafe { (*req_ptr).out_size = bytes.len() as u64 };
                    if req.out_ptr == 0 || (req.out_cap as usize) < bytes.len() {
                        return 0x8083_0007; // E_GS_PROVIDED_BUFFER_TOO_SMALL
                    }
                    // SAFETY: Caller buffer capacity >= `bytes.len()`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            req.out_ptr as *mut u8,
                            bytes.len(),
                        );
                    }
                    HRESULT_S_OK
                }
                Err(_) => 0x8083_0008, // E_GS_BLOB_NOT_FOUND
            }
        }
        OP_XGR_GAMESAVE_ENUM_BLOBS => {
            if a0 == 0 {
                return HRESULT_E_INVALIDARG;
            }
            let req_ptr = a0 as *mut GameSaveEnumReq;
            // SAFETY: Caller passes readable/writable `GameSaveEnumReq`.
            let req = unsafe { *req_ptr };
            let container = unsafe { read_guest_str(req.container_ptr, req.container_len) };
            let prefix = if req.prefix_ptr != 0 && req.prefix_len != 0 {
                // SAFETY: Valid UTF-8 prefix slice.
                Some(unsafe { read_guest_str(req.prefix_ptr, req.prefix_len) })
            } else {
                None
            };
            let xgr = Arc::clone(&rt.xgr);
            match exec_native(move || xgr.enumerate_save_blobs(&container, prefix.as_deref())) {
                Ok(entries) => {
                    let mut packed = Vec::new();
                    for e in &entries {
                        packed.extend_from_slice(&e.size.to_le_bytes());
                        packed.extend_from_slice(&(e.name.len() as u32).to_le_bytes());
                        packed.extend_from_slice(e.name.as_bytes());
                    }
                    // SAFETY: Update output count and packed byte length.
                    unsafe {
                        (*req_ptr).out_count = entries.len() as u64;
                        (*req_ptr).out_bytes = packed.len() as u64;
                    }
                    if req.out_ptr != 0
                        && (req.out_cap as usize) >= packed.len()
                        && !packed.is_empty()
                    {
                        // SAFETY: Caller buffer holds `packed`.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                packed.as_ptr(),
                                req.out_ptr as *mut u8,
                                packed.len(),
                            );
                        }
                    }
                    HRESULT_S_OK
                }
                Err(_) => HRESULT_E_INVALIDARG,
            }
        }
        OP_D3D12X_CREATE_DEVICE | OP_D3D12X_XBOX_CREATE_DEVICE => {
            let out_device_ptr = a0 as *mut *mut c_void;
            if out_device_ptr.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            let entry_name = if opcode == OP_D3D12X_XBOX_CREATE_DEVICE {
                "D3D12XboxCreateDevice"
            } else {
                "D3D12CreateDevice"
            };
            if rt.gpu_backend_mode.eq_ignore_ascii_case("vulkan") {
                match rt.vulkan_backend() {
                    Ok(vk) => match create_vulkan_com_device(vk, d3d12_vtable_probe_callback) {
                        Ok(dev) => {
                            // SAFETY: Caller supplied non-null writable COM output pointer `ppDevice`.
                            unsafe {
                                *out_device_ptr = dev.cast::<c_void>();
                            }
                            rt.log_line(&format!(
                                "[D3D12X] {entry_name} (vulkan M2.4) -> S_OK (device=0x{:x})",
                                dev as usize
                            ));
                            return HRESULT_S_OK;
                        }
                        Err(err) => {
                            rt.log_line(&format!(
                                "[D3D12X] Vulkan COM device creation failed: {err}"
                            ));
                            return HRESULT_E_NOTIMPL;
                        }
                    },
                    Err(err) => {
                        rt.log_line(&format!("[D3D12X] Vulkan backend init failed: {err}"));
                        return HRESULT_E_NOTIMPL;
                    }
                }
            }
            let Some(vtbl) = rt.null_vtable() else {
                return HRESULT_E_NOTIMPL;
            };
            let dev = create_null_device(vtbl);
            // SAFETY: Caller supplied non-null writable COM output pointer `ppDevice`.
            unsafe {
                *out_device_ptr = dev.cast::<c_void>();
            }
            rt.log_line(&format!(
                "[D3D12X] {entry_name} (null M1) -> S_OK (device=0x{:x})",
                dev as usize
            ));
            HRESULT_S_OK
        }
        OP_D3D12X_CONFIGURE_VIRTUAL_MEMORY => {
            let params_ptr = a0 as *mut D3D12ConfigureVirtualMemoryXParams;
            let pre = if !params_ptr.is_null() {
                // SAFETY: Caller passes pointer to readable 32-byte `D3D12ConfigureVirtualMemoryXParams`.
                Some(unsafe { *params_ptr })
            } else {
                None
            };
            // SAFETY: Caller passes pointer to 32-byte `D3D12ConfigureVirtualMemoryXParams`.
            let hr = unsafe { configure_virtual_memory_x(params_ptr) };
            if rt.trace_hle
                && let Some(p_in) = pre
            {
                // SAFETY: `params_ptr` was non-null when `pre` was read.
                let p_out = unsafe { *params_ptr };
                rt.log_line(&format!(
                    "[HLE] D3D12ConfigureVirtualMemoryX(flags_or_mode=0x{:08x}, field_04=0x{:08x}, field_08=0x{:08x}) -> hr=0x{:08x} low=0x{:016x} high=0x{:016x}",
                    p_in.flags_or_mode,
                    p_in.field_04,
                    p_in.field_08,
                    hr as u32,
                    p_out.low_address_out,
                    p_out.high_address_out
                ));
            }
            hr as u32 as u64
        }
        OP_D3D12X_SERIALIZE_VERSIONED_ROOT_SIG => {
            // SAFETY: Caller passes versioned root-signature desc and output `ID3DBlob**` pointers.
            unsafe {
                serialize_versioned_root_signature(
                    a0 as *const D3D12VersionedRootSignatureDesc,
                    a1 as *mut *mut c_void,
                    a2 as *mut *mut c_void,
                ) as u32 as u64
            }
        }
        OP_XG_CREATE_TEXTURE_COMPUTER => {
            // SAFETY: Caller passes 56-byte `D3D12ResourceDesc*` and writable COM output pointer.
            unsafe {
                xg_create_texture_computer(
                    a0 as *const D3D12ResourceDesc,
                    a1 as *mut *mut c_void,
                    d3d12_vtable_probe_callback,
                ) as u32 as u64
            }
        }
        OP_XG_SUGGEST_SWIZZLE_MODE => {
            // SAFETY: Caller passes `D3D12ResourceDesc*`, flags, and writable `u32*`.
            unsafe {
                xg_suggest_swizzle_mode(a0 as *const D3D12ResourceDesc, a1 as u32, a2 as *mut u32)
                    as u32 as u64
            }
        }
        OP_XG_CREATE_TYPED_COMPUTER => {
            // SAFETY: The PE shim forwards the matching descriptor and writable output pointer.
            unsafe {
                xg_create_typed_computer(
                    a0 as u32,
                    a1 as *const c_void,
                    a2 as *mut *mut c_void,
                    d3d12_vtable_probe_callback,
                ) as u32 as u64
            }
        }
        OP_XG_COMPUTE_OPTIMAL_TILE_MODE => {
            if a0 == 0 {
                return u64::from(u32::MAX);
            }
            // SAFETY: The synchronous PE shim provides a seven-word argument array.
            let [dim, format, width, height, depth, samples, bind] =
                unsafe { (a0 as *const [u32; 7]).read_unaligned() };
            u64::from(xg_compute_optimal_tile_mode(
                dim, format, width, height, depth, samples, bind,
            ))
        }
        OP_DSTORAGE_GET_FACTORY => {
            // SAFETY: Caller passes writable `IDStorageFactoryX**` output pointer.
            unsafe {
                dstorage_get_factory(
                    a0 as *mut *mut c_void,
                    d3d12_vtable_probe_callback,
                    rt.merged_dir.clone(),
                    Some(rt.saves_dir.clone()),
                ) as u32 as u64
            }
        }
        OP_GAMEINPUT_CREATE => {
            // SAFETY: Caller passes writable `IGameInput**` output pointer.
            unsafe {
                gameinput_create(
                    Arc::clone(&rt.gameinput),
                    a0 as *mut *mut c_void,
                    d3d12_vtable_probe_callback,
                ) as u32 as u64
            }
        }
        // ---- Audio co-processor shims (AcpHal / xapu / chad) ----
        OP_ACPHAL_CREATE => {
            // SAFETY: Caller passes writable `IAcpHal**` output pointer.
            unsafe {
                acphal_create(a0 as *mut *mut c_void, d3d12_vtable_probe_callback) as u32 as u64
            }
        }
        OP_ACPHAL_ALLOC_SHAPE_CONTEXTS => {
            let _ = rt.audio_coproc.allocate_shape_contexts();
            HRESULT_S_OK
        }
        OP_ACPHAL_RELEASE_SHAPE_CONTEXTS => {
            let _ = rt.audio_coproc.release_shape_contexts();
            HRESULT_S_OK
        }
        OP_APU_CREATE_HEAP => {
            let _ = rt.audio_coproc.apu_create_heap(a0 as u32, a1 as u32);
            HRESULT_S_OK
        }
        OP_APU_ALLOC => {
            let va_out = a0 as *mut *mut c_void;
            let apu_out = a1 as *mut u32;
            if va_out.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            match rt.audio_coproc.apu_alloc(a2 as u32, a3 as u32) {
                Some((va, apu)) => {
                    // SAFETY: `va_out` is non-null and `apu_out` is optional writable output.
                    unsafe {
                        *va_out = va as *mut c_void;
                        if !apu_out.is_null() {
                            *apu_out = apu;
                        }
                    }
                    HRESULT_S_OK
                }
                None => HRESULT_E_INVALIDARG,
            }
        }
        OP_APU_FREE => {
            let _ = rt.audio_coproc.apu_free(a0 as usize);
            HRESULT_S_OK
        }
        OP_APU_MAP_APU_ADDRESS => rt.audio_coproc.apu_map_apu_address(a0 as u32) as u64,
        OP_APU_MAP_VIRTUAL_ADDRESS => rt
            .audio_coproc
            .apu_map_virtual_address(a0 as usize)
            .map(u64::from)
            .unwrap_or(0),
        OP_APU_IS_VIRTUAL_ADDRESS_VALID => u64::from(
            rt.audio_coproc
                .apu_is_virtual_address_valid(a0 as usize, a1 as u32),
        ),
        OP_XAPU_CONNECT => {
            let out_conn = a2 as *mut u64;
            if out_conn.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            match rt.audio_coproc.xapu_connect() {
                Some(id) => {
                    // SAFETY: Caller passed writable `Connection**` pointer.
                    unsafe { *out_conn = id };
                    HRESULT_S_OK
                }
                None => HRESULT_E_NOTIMPL,
            }
        }
        OP_XAPU_DISCONNECT => {
            if rt.audio_coproc.xapu_disconnect(a0) {
                HRESULT_S_OK
            } else {
                HRESULT_E_INVALIDARG
            }
        }
        OP_XAPU_ENQUEUE_COMMAND => {
            let out_ticket = a2 as *mut u32;
            // If the caller supplied a non-null audio packet header (`magic = 0x58415055`,
            // `frame_count: u32`, followed by interleaved stereo `f32` samples), push the PCM
            // frames into the wait-free `x2l-audio` PipeWire/null ring buffer without blocking.
            if a1 != 0 {
                let hdr = a1 as *const u32;
                // SAFETY: Caller passed a readable command buffer; inspect the 8-byte header first.
                let (magic, sample_count) = unsafe { (*hdr, *hdr.add(1)) };
                if magic == 0x5841_5055 && sample_count > 0 && sample_count <= 16_384 {
                    // SAFETY: Caller provided `sample_count` contiguous `f32` samples after header.
                    let samples = unsafe {
                        std::slice::from_raw_parts(hdr.add(2).cast::<f32>(), sample_count as usize)
                    };
                    let out = rt.audio_output.get_or_init(|| {
                        let prefer_pw = std::env::var("X2L_AUDIO_BACKEND")
                            .map(|v| !v.eq_ignore_ascii_case("null"))
                            .unwrap_or(true);
                        exec_native(move || AudioOutput::open(DEFAULT_SAMPLE_RATE_HZ, prefer_pw))
                    });
                    let _ = out.push_f32(samples);
                }
            }
            match rt.audio_coproc.xapu_enqueue_command(a0) {
                Some(ticket) => {
                    if !out_ticket.is_null() {
                        // SAFETY: Caller passed optional writable ticket pointer.
                        unsafe { *out_ticket = ticket };
                    }
                    HRESULT_S_OK
                }
                None => HRESULT_E_INVALIDARG,
            }
        }
        OP_XAPU_DEQUEUE_RESULT => match rt.audio_coproc.xapu_dequeue_result(a0) {
            Some(ticket) => {
                let out_res = a1 as *mut u32;
                if !out_res.is_null() {
                    // SAFETY: Caller passed writable result buffer.
                    unsafe { *out_res = ticket };
                }
                HRESULT_S_OK
            }
            None => 1, // S_FALSE (no pending result)
        },
        OP_XAPU_GET_RESULT_COUNT => {
            let count = rt.audio_coproc.xapu_result_count(a0);
            let out_count = a1 as *mut u32;
            if !out_count.is_null() {
                // SAFETY: Caller passed writable `u32*` count output.
                unsafe { *out_count = count };
            }
            u64::from(count)
        }
        OP_CHAD_CONNECT => {
            let out_dsp = a5 as *mut u64;
            if out_dsp.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            match rt.audio_coproc.chad_connect(a1 as u32) {
                Some(id) => {
                    // SAFETY: Caller passed writable `Dsp**` output pointer.
                    unsafe { *out_dsp = id };
                    HRESULT_S_OK
                }
                None => HRESULT_E_NOTIMPL,
            }
        }
        OP_CHAD_DISCONNECT => {
            if rt.audio_coproc.chad_disconnect(a0) {
                HRESULT_S_OK
            } else {
                HRESULT_E_INVALIDARG
            }
        }
        OP_CHAD_ACTIVATE => {
            let out_stream = a3 as *mut u64;
            if out_stream.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            match rt.audio_coproc.chad_activate(a0) {
                Some(id) => {
                    // SAFETY: Caller passed writable `Stream**` output pointer.
                    unsafe { *out_stream = id };
                    HRESULT_S_OK
                }
                None => HRESULT_E_INVALIDARG,
            }
        }
        OP_CHAD_DEACTIVATE => {
            if rt.audio_coproc.chad_deactivate(a0) {
                HRESULT_S_OK
            } else {
                HRESULT_E_INVALIDARG
            }
        }
        OP_CHAD_SUBMIT_COMMAND => {
            let out_ticket = a2 as *mut u32;
            match rt.audio_coproc.chad_submit_command(a0) {
                Some(ticket) => {
                    if !out_ticket.is_null() {
                        // SAFETY: Caller passed writable ticket pointer.
                        unsafe { *out_ticket = ticket };
                    }
                    HRESULT_S_OK
                }
                None => HRESULT_E_INVALIDARG,
            }
        }
        OP_XAPU_CONNECT_DECODER => {
            // SAFETY: The PE shim forwards the descriptors, handle output, and optional live event.
            unsafe {
                gdk_com::xapu_connect(
                    &rt.audio_coproc,
                    a0 as *const c_void,
                    a1 as *mut c_void,
                    a2 as *mut u64,
                    a3 as usize,
                ) as u32 as u64
            }
        }
        OP_XAPU_DISCONNECT_DECODER => {
            let Some(event) = exec_native(|| rt.audio_coproc.xapu_disconnect_with_event(a0)) else {
                return HRESULT_E_INVALIDARG;
            };
            if a1 != 0 {
                // SAFETY: The PE shim supplies a writable local event handle.
                unsafe { (a1 as *mut usize).write_unaligned(event) };
            }
            HRESULT_S_OK
        }
        OP_XAPU_SUBMIT => {
            let signal = if a3 != 0 {
                // SAFETY: The PE shim sends its Win64 event-signalling function.
                Some(unsafe { std::mem::transmute::<u64, extern "win64" fn(usize)>(a3) })
            } else {
                None
            };
            // SAFETY: The command and optional output counter are live for this synchronous call.
            unsafe {
                gdk_com::xapu_enqueue(
                    &rt.audio_coproc,
                    a0,
                    a1 as *const c_void,
                    a2 as *mut u32,
                    signal,
                ) as u32 as u64
            }
        }
        OP_XAPU_POP => {
            // SAFETY: The PE caller supplies the result buffer matching the connection variant.
            unsafe { gdk_com::xapu_dequeue(&rt.audio_coproc, a0, a1 as *mut c_void) as u32 as u64 }
        }
        OP_XAPU_RESULT_COUNTS => {
            let (pending, total) = rt.audio_coproc.xapu_result_counts(a0);
            let out_count = a1 as *mut u32;
            if !out_count.is_null() {
                // SAFETY: Caller passed writable count output.
                unsafe { out_count.write_unaligned(total) };
            }
            u64::from(pending)
        }
        OP_CHAD_CONNECT_STREAM_LIMIT => {
            let out_dsp = a5 as *mut u64;
            if out_dsp.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            let max_streams = if a0 == 0 && a3 == 0 { a1 } else { a3 };
            match rt.audio_coproc.chad_connect(max_streams as u32) {
                Some(id) => {
                    // SAFETY: Caller passed writable `Dsp**` output pointer.
                    unsafe { *out_dsp = id };
                    HRESULT_S_OK
                }
                None => HRESULT_E_NOTIMPL,
            }
        }
        OP_CHAD_ACTIVATE_STATUS => {
            let out_stream = a3 as *mut u64;
            if out_stream.is_null() {
                return HRESULT_E_INVALIDARG;
            }
            let out_status = a2 as *mut *const x2l_audio::coproc::ChadStatus;
            match rt.audio_coproc.chad_activate_with_status(a0) {
                Some((id, status)) => {
                    // SAFETY: The caller provides live stream and optional status outputs.
                    unsafe {
                        *out_stream = id;
                        if !out_status.is_null() {
                            *out_status = Arc::as_ptr(&status);
                        }
                    };
                    HRESULT_S_OK
                }
                None => HRESULT_E_INVALIDARG,
            }
        }
        OP_CHAD_SUBMIT_STATUS => {
            let out_ticket = a2 as *mut u32;
            let status = if a1 == 0 { 0 } else { HRESULT_E_NOTIMPL as i32 };
            match rt.audio_coproc.chad_execute(a0, || status) {
                Some((ticket, status)) => {
                    if !out_ticket.is_null() {
                        // SAFETY: The PE caller supplies the optional ticket pointer.
                        unsafe { *out_ticket = ticket };
                    }
                    status as u32 as u64
                }
                None => HRESULT_E_INVALIDARG,
            }
        }
        OP_CHAD_DECOMPRESS => {
            // SAFETY: The extension forwards a live descriptor and an optional ticket output.
            unsafe {
                gdk_com::chad_decompress(&rt.audio_coproc, a0, a1 as *const _, a2 as *mut u32)
                    as u32 as u64
            }
        }
        // ---- PSM AppNotify ----
        OP_PSM_REGISTER_APP_STATE | OP_PSM_REGISTER_CONSTRAINED => {
            let out_reg = a2 as *mut u64;
            if out_reg.is_null() {
                return 87; // ERROR_INVALID_PARAMETER
            }
            let cookie = rt.appnotify.register(a0 as usize, a1 as usize);
            // SAFETY: Caller passed writable `Registration**` output pointer.
            unsafe { *out_reg = cookie };
            0 // ERROR_SUCCESS
        }
        OP_PSM_UNREGISTER_APP_STATE | OP_PSM_UNREGISTER_CONSTRAINED => {
            let _ = rt.appnotify.unregister(a0);
            0
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
    // Pre-initialize runtime state (and W^X vtable thunks + SUD-free native worker) during ELF
    // init before Wine enables Syscall User Dispatch (SUD) on guest PE threads.
    let rt = runtime();
    let _ = rt.null_vtable();
    start_native_gpu_worker();

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
    use x2l_gpu_vk::NullD3D12DeviceX;

    #[test]
    fn bridge_dispatch_handshake_kernelx_xgr_and_null_d3d12x() {
        // SAFETY: Test exercises valid opcodes and stack-owned output pointers in-process.
        unsafe {
            assert_eq!(
                x2l_bridge_dispatch(OP_HANDSHAKE, 0, 0, 0, 0, 0, 0, 0),
                X2L_BRIDGE_MAGIC
            );
            assert_eq!(
                x2l_bridge_dispatch(OP_KERNELX_GET_CONSOLE_TYPE, 0, 0, 0, 0, 0, 0, 0),
                u64::from(x2l_hle_kernelx::CONSOLE_TYPE_SERIES_X)
            );
            let ptr = x2l_bridge_dispatch(OP_KERNELX_XMEM_ALLOC, 256, 0, 0, 0, 0, 0, 0);
            assert_ne!(ptr, 0);
            assert_eq!(
                x2l_bridge_dispatch(OP_KERNELX_XMEM_FREE, ptr, 0, 0, 0, 0, 0, 0),
                1
            );
            assert_eq!(
                x2l_bridge_dispatch(OP_XGR_INITIALIZE, 0, 0, 0, 0, 0, 0, 0),
                HRESULT_S_OK
            );
            assert_eq!(
                x2l_bridge_dispatch(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0, 0),
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
                    0,
                    0,
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
