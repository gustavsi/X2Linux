// Thin Win64 PE shim for generic Microsoft GDK `xgameruntime.dll`.
// Win64 ABI signatures and struct layouts informed by Marvisak/VIctory (commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-hle/src/modules/) and
// xcl-dev/xcl (commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/xgameruntime/).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-28: clean-room offline GDK runtime, Wine-thread-native
// `XTaskQueue`/`XAsync`, XDG-persisted `XGameSave`/`XPersistentLocalStorage`, local `XUser`,
// `XPackage`, and strictly offline `XStore` (zero DRM bypass or license spoofing).

//! Thin Win64 PE shim for offline `xgameruntime.dll` forwarding persistence and telemetry to
//! `libx2l_core.so` while running all `XTaskQueue`/`XAsync` guest callbacks on Wine threads.

mod package;
mod query_api;
mod taskqueue;

pub use package::*;
pub use query_api::Guid;
pub use taskqueue::*;

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, c_char, c_void};
use std::mem::{offset_of, size_of};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use x2l_shim_common::{
    HRESULT_E_INVALIDARG, HRESULT_E_NOTIMPL, HRESULT_E_POINTER, HRESULT_S_OK,
    OP_XGR_GAMESAVE_ENUM_BLOBS, OP_XGR_GAMESAVE_READ_BLOB, OP_XGR_GAMESAVE_ROOT,
    OP_XGR_GAMESAVE_WRITE_BLOB, OP_XGR_GET_APP_DEVICE_ID, OP_XGR_GET_CONSOLE_ID,
    OP_XGR_GET_DEVICE_TYPE, OP_XGR_GET_SANDBOX_ID, OP_XGR_GET_TITLE_ID, OP_XGR_INITIALIZE,
    OP_XGR_PLS_GET_PATH, OP_XGR_PLS_GET_SPACE_INFO, OP_XGR_STORE_OFFLINE, OP_XGR_UNINITIALIZE,
    OP_XGR_USER_GET_GAMERTAG, call_bridge, handle_dll_main, record_stub,
};

const HRESULT_INSUFFICIENT_BUFFER: i32 = -2147024774; // 0x8007007A
const HRESULT_E_USER_OFFLINE_TOKEN: i32 = -1994108655; // 0x89245111
const HRESULT_E_GS_BUFFER_TOO_SMALL: i32 = -2138898425; // 0x80830007
const HRESULT_E_GS_BLOB_NOT_FOUND: i32 = -2138898424; // 0x80830008

// ============================================================================
// Verified Win64 GDK `#[repr(C)]` ABI Layouts
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XSystemVersion {
    pub major: u16,
    pub minor: u16,
    pub build: u16,
    pub revision: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XSystemAnalyticsInfo {
    pub os_version: XSystemVersion,
    pub hosting_os_version: XSystemVersion,
    pub family: [u8; 64],
    pub form: [u8; 64],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XSystemRuntimeInfo {
    pub runtime_version: XSystemVersion,
    pub available_version: XSystemVersion,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XPersistentLocalStorageSpaceInfo {
    pub available_free_bytes: u64,
    pub total_free_bytes: u64,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XUserLocalId {
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XPackageDetails {
    pub package_identifier: *const c_char,
    pub version: XSystemVersion,
    pub kind: u32,
    pub display_name: *const c_char,
    pub description: *const c_char,
    pub publisher: *const c_char,
    pub store_id: *const c_char,
    pub installing: bool,
    pub index: u32,
    pub count: u32,
    pub age_restricted: bool,
    pub title_id: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XGameSaveBlobInfo {
    pub name: *const c_char,
    pub size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XGameSaveBlob {
    pub info: XGameSaveBlobInfo,
    pub data: *mut u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct XNetworkingConnectivityHint {
    pub connectivity_level: u32,
    pub connectivity_cost: u32,
    pub iana_interface_type: u32,
    pub network_initialized: bool,
    pub approaching_data_limit: bool,
    pub over_data_limit: bool,
    pub roaming: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct XStoreGameLicense {
    pub sku_store_id: [u8; 18],
    pub is_active: bool,
    pub is_trial_owned_by_this_user: bool,
    pub is_disc_license: bool,
    pub is_trial: bool,
    pub trial_time_remaining_in_seconds: u32,
    pub trial_unique_id: [u8; 64],
    pub expiration_date: i64,
}

const _: () = assert!(size_of::<XSystemVersion>() == 8);
const _: () = assert!(size_of::<XSystemAnalyticsInfo>() == 144);
const _: () = assert!(offset_of!(XSystemAnalyticsInfo, family) == 16);
const _: () = assert!(offset_of!(XSystemAnalyticsInfo, form) == 80);
const _: () = assert!(size_of::<XPersistentLocalStorageSpaceInfo>() == 32);
const _: () = assert!(size_of::<XUserLocalId>() == 8);
const _: () = assert!(size_of::<XPackageDetails>() == 80);
const _: () = assert!(size_of::<XGameSaveBlobInfo>() == 16);
const _: () = assert!(size_of::<XGameSaveBlob>() == 24);
const _: () = assert!(size_of::<XNetworkingConnectivityHint>() == 16);
const _: () = assert!(size_of::<XStoreGameLicense>() == 104);
const _: () = assert!(offset_of!(XStoreGameLicense, expiration_date) == 96);

#[repr(C)]
struct GameSaveWriteReq {
    container_ptr: u64,
    container_len: u64,
    blob_ptr: u64,
    blob_len: u64,
    data_ptr: u64,
    data_len: u64,
}

#[repr(C)]
struct GameSaveReadReq {
    container_ptr: u64,
    container_len: u64,
    blob_ptr: u64,
    blob_len: u64,
    out_ptr: u64,
    out_cap: u64,
    out_size: u64,
}

#[repr(C)]
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

// ============================================================================
// Internal Process-Local Handle Tables & Helpers
// ============================================================================

const LOCAL_USER_HANDLE: usize = 0x0001_0001;
const PACKAGE_MOUNT_HANDLE: usize = 0x0002_0001;
const DISPLAY_DEFERRAL_HANDLE: usize = 0x0003_0001;
const GAMESAVE_PROVIDER_HANDLE: usize = 0x0004_0001;

static TIME_SENSITIVE: AtomicBool = AtomicBool::new(false);
static ERROR_OPTIONS: AtomicU32 = AtomicU32::new(0);
static NEXT_EVENT_TOKEN: AtomicU64 = AtomicU64::new(1);
static NEXT_GS_HANDLE: AtomicU64 = AtomicU64::new(0x0005_0001);

struct GameSaveUpdateState {
    container_name: String,
    pending_writes: Vec<(String, Vec<u8>)>,
}

static GS_CONTAINERS: OnceLock<Mutex<BTreeMap<usize, String>>> = OnceLock::new();
static GS_UPDATES: OnceLock<Mutex<BTreeMap<usize, GameSaveUpdateState>>> = OnceLock::new();
static PREFERRED_PORT_REGISTRATIONS: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());

fn gs_containers() -> &'static Mutex<BTreeMap<usize, String>> {
    GS_CONTAINERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn gs_updates() -> &'static Mutex<BTreeMap<usize, GameSaveUpdateState>> {
    GS_UPDATES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn read_cstr_opt(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: Caller supplied a valid NUL-terminated C string.
    unsafe { Some(CStr::from_ptr(ptr).to_string_lossy().into_owned()) }
}

fn copy_cstr_to_guest(src: &str, cap: usize, buf: *mut u8, used: *mut usize) -> i32 {
    let needed = src.len() + 1;
    if !used.is_null() {
        // SAFETY: Caller supplied optional writable `used` size pointer.
        unsafe { *used = needed };
    }
    if buf.is_null() || cap < needed {
        return HRESULT_INSUFFICIENT_BUFFER;
    }
    // SAFETY: `buf` is non-null and `cap >= src.len() + 1`.
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr(), buf, src.len());
        *buf.add(src.len()) = 0;
    }
    HRESULT_S_OK
}

fn query_bridge_string(op: u32, fallback: &str) -> String {
    let mut buf = [0u8; 512];
    // SAFETY: `buf` is a writable stack buffer of length 512.
    let hr = unsafe {
        call_bridge(
            op,
            buf.as_mut_ptr() as usize as u64,
            buf.len() as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    };
    if hr != HRESULT_S_OK {
        return fallback.to_owned();
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(0);
    std::str::from_utf8(&buf[..len])
        .unwrap_or(fallback)
        .to_owned()
}

// ============================================================================
// DLL Entry & Bootstrap Exports
// ============================================================================

#[unsafe(no_mangle)]
pub extern "system" fn DllMain(_inst: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    handle_dll_main(reason)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameRuntimeInitialize() -> i32 {
    // SAFETY: `OP_XGR_INITIALIZE` takes no pointers.
    unsafe { call_bridge(OP_XGR_INITIALIZE, 0, 0, 0, 0, 0, 0).unwrap_or(0) as i32 }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameRuntimeInitializeWithOptions(_options: *const c_void) -> i32 {
    XGameRuntimeInitialize()
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameRuntimeUninitialize() {
    // SAFETY: `OP_XGR_UNINITIALIZE` takes no pointers.
    let _ = unsafe { call_bridge(OP_XGR_UNINITIALIZE, 0, 0, 0, 0, 0, 0) };
}

#[unsafe(no_mangle)]
pub extern "win64" fn InitializeApiImpl(_a0: u64, _a1: u64) -> i32 {
    XGameRuntimeInitialize()
}

#[unsafe(no_mangle)]
pub extern "win64" fn InitializeApiImplEx(_a0: u64, _a1: u64, _a2: u64) -> i32 {
    XGameRuntimeInitialize()
}

#[unsafe(no_mangle)]
pub extern "win64" fn InitializeApiImplEx2(
    _a0: u64,
    _a1: u64,
    _a2: u64,
    _options: *mut c_void,
) -> i32 {
    XGameRuntimeInitialize()
}

#[unsafe(no_mangle)]
pub extern "win64" fn UninitializeApiImpl() {
    XGameRuntimeUninitialize();
}

#[unsafe(no_mangle)]
pub extern "win64" fn QueryApiImpl(
    api_id: *const Guid,
    iid: *const Guid,
    out_iface: *mut *mut c_void,
) -> i32 {
    query_api::dispatch_query_api_impl(api_id, iid, out_iface)
}

// ============================================================================
// Runtime, System, Error, Thread, Display, & Persistent Local Storage
// ============================================================================

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetDeviceType() -> u32 {
    // SAFETY: `OP_XGR_GET_DEVICE_TYPE` takes no pointers.
    unsafe { call_bridge(OP_XGR_GET_DEVICE_TYPE, 0, 0, 0, 0, 0, 0).unwrap_or(4) as u32 }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetConsoleId(capacity: usize, buf: *mut u8, used: *mut usize) -> i32 {
    let val = query_bridge_string(
        OP_XGR_GET_CONSOLE_ID,
        "00000000-0000-0000-0000-000000000001",
    );
    copy_cstr_to_guest(&val, capacity, buf, used)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetXboxLiveSandboxId(
    capacity: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    let val = query_bridge_string(OP_XGR_GET_SANDBOX_ID, "RETAIL");
    copy_cstr_to_guest(&val, capacity, buf, used)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetAnalyticsInfo() -> XSystemAnalyticsInfo {
    let dev_type = XSystemGetDeviceType();
    let mut info = XSystemAnalyticsInfo {
        os_version: XSystemVersion {
            major: 10,
            minor: 0,
            build: 22000,
            revision: 0,
        },
        hosting_os_version: XSystemVersion {
            major: 10,
            minor: 0,
            build: 22000,
            revision: 0,
        },
        family: [0u8; 64],
        form: [0u8; 64],
    };
    let family = b"Windows.Xbox";
    let form: &[u8] = if dev_type == 3 {
        b"Xbox Series S"
    } else {
        b"Xbox Series X"
    };
    info.family[..family.len()].copy_from_slice(family);
    info.form[..form.len()].copy_from_slice(form);
    info
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetRuntimeInfo() -> XSystemRuntimeInfo {
    let v = XSystemVersion {
        major: 10,
        minor: 0,
        build: 26100,
        revision: 0,
    };
    XSystemRuntimeInfo {
        runtime_version: v,
        available_version: v,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemGetAppSpecificDeviceId(
    capacity: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    let val = query_bridge_string(OP_XGR_GET_APP_DEVICE_ID, "X2LINUX-LOCAL-DEVICE-0001");
    copy_cstr_to_guest(&val, capacity, buf, used)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemIsHandleValid(handle: *mut c_void) -> bool {
    let raw = handle as usize;
    raw != 0
        && (raw == LOCAL_USER_HANDLE
            || raw == PACKAGE_MOUNT_HANDLE
            || raw == DISPLAY_DEFERRAL_HANDLE
            || raw == GAMESAVE_PROVIDER_HANDLE
            || package_monitor_is_valid(handle)
            || is_valid_queue_handle(handle))
}

#[unsafe(no_mangle)]
pub extern "win64" fn XSystemAllowFullDownloadBandwidth(_enable: bool) {
    // There are no managed package downloads whose bandwidth can be adjusted.
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameGetXboxTitleId(out_title_id: *mut u32) -> i32 {
    if out_title_id.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `OP_XGR_GET_TITLE_ID` takes no pointers.
    let id =
        unsafe { call_bridge(OP_XGR_GET_TITLE_ID, 0, 0, 0, 0, 0, 0).unwrap_or(0x1000_0001) as u32 };
    // SAFETY: `out_title_id` was checked non-null.
    unsafe { *out_title_id = id };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameGetXboxTitleIdAsync(async_block: *mut XAsyncBlock) -> i32 {
    let mut id = 0u32;
    let hr = XGameGetXboxTitleId(&mut id);
    complete_builtin_async(async_block, hr, Some(id.to_le_bytes().to_vec()))
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameGetXboxTitleIdResult(
    async_block: *mut XAsyncBlock,
    out_title_id: *mut u32,
) -> i32 {
    if out_title_id.is_null() {
        return HRESULT_E_POINTER;
    }
    match take_builtin_async_result(async_block) {
        Ok((_, Some(bytes))) if bytes.len() >= 4 => {
            let id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            // SAFETY: `out_title_id` was checked non-null.
            unsafe { *out_title_id = id };
            HRESULT_S_OK
        }
        Ok(_) => HRESULT_E_INVALIDARG,
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameGetClientId(out_guid: *mut [u8; 16]) -> i32 {
    if out_guid.is_null() {
        return HRESULT_E_POINTER;
    }
    // Deterministic offline local client GUID.
    // SAFETY: `out_guid` was checked non-null.
    unsafe {
        *out_guid = [
            0x58, 0x32, 0x4C, 0x00, 0x01, 0x00, 0x40, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01,
        ];
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XErrorSetCallback(_callback: *const c_void, _context: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "win64" fn XErrorSetOptions(options: u32, _reserved: usize) {
    ERROR_OPTIONS.store(options, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "win64" fn XErrorReport(hr: i32, msg: *const c_char) {
    if x2l_shim_common::trace_hle_enabled() {
        let text = read_cstr_opt(msg).unwrap_or_default();
        x2l_shim_common::trace_hle_line(&format!(
            "XErrorReport(hr=0x{:08x}, msg=\"{text}\")",
            hr as u32
        ));
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XThreadIsTimeSensitive() -> bool {
    TIME_SENSITIVE.load(Ordering::Relaxed)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XThreadSetTimeSensitive(is_time_sensitive: bool) -> i32 {
    TIME_SENSITIVE.store(is_time_sensitive, Ordering::Relaxed);
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XThreadAssertNotTimeSensitive() {}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameRuntimeIsFeatureAvailable(_feature: u32) -> bool {
    false
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDisplayTryEnableHdrMode(
    _options: *const c_void,
    _out_info: *mut c_void,
) -> u32 {
    0 // XDisplayHdrModeResult::Unknown / SDR
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDisplayTryAcquireTimeoutDeferral(out_handle: *mut *mut c_void) -> i32 {
    if out_handle.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_handle` was verified non-null.
    unsafe { *out_handle = DISPLAY_DEFERRAL_HANDLE as *mut c_void };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDisplayAcquireTimeoutDeferral(out_handle: *mut *mut c_void) -> i32 {
    XDisplayTryAcquireTimeoutDeferral(out_handle)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XDisplayCloseTimeoutDeferralHandle(_handle: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "win64" fn XPersistentLocalStorageGetPathSize(out_size: *mut usize) -> i32 {
    if out_size.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `OP_XGR_PLS_GET_PATH` writes required NUL-terminated byte count to `a2`.
    unsafe {
        call_bridge(OP_XGR_PLS_GET_PATH, 0, 0, out_size as usize as u64, 0, 0, 0)
            .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPersistentLocalStorageGetPath(
    capacity: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    let mut req_size = 0usize;
    let hr = XPersistentLocalStorageGetPathSize(&mut req_size);
    if hr != HRESULT_S_OK {
        return hr;
    }
    if !used.is_null() {
        // SAFETY: Caller passed optional writable `used`.
        unsafe { *used = req_size };
    }
    if buf.is_null() || capacity < req_size {
        return HRESULT_INSUFFICIENT_BUFFER;
    }
    // SAFETY: `buf` has at least `req_size` bytes.
    unsafe {
        call_bridge(
            OP_XGR_PLS_GET_PATH,
            buf as usize as u64,
            capacity as u64,
            used as usize as u64,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPersistentLocalStorageGetSpaceInfo(
    out_info: *mut XPersistentLocalStorageSpaceInfo,
) -> i32 {
    if out_info.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_info` is a writable 32-byte `XPersistentLocalStorageSpaceInfo*`.
    unsafe {
        call_bridge(
            OP_XGR_PLS_GET_SPACE_INFO,
            out_info as usize as u64,
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
pub extern "win64" fn XPersistentLocalStoragePromptUserForSpaceAsync(
    requested_bytes: u64,
    async_block: *mut XAsyncBlock,
) -> i32 {
    let mut info = XPersistentLocalStorageSpaceInfo::default();
    let hr = XPersistentLocalStorageGetSpaceInfo(&mut info);
    let result_hr = if hr == HRESULT_S_OK && info.available_free_bytes >= requested_bytes {
        HRESULT_S_OK
    } else {
        HRESULT_E_INVALIDARG
    };
    complete_builtin_async(async_block, result_hr, None)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPersistentLocalStoragePromptUserForSpaceResult(
    async_block: *mut XAsyncBlock,
) -> i32 {
    take_builtin_async_result(async_block)
        .map(|_| HRESULT_S_OK)
        .unwrap_or_else(|e| e)
}

// ============================================================================
// XUser (Offline Local Profile)
// ============================================================================

#[unsafe(no_mangle)]
pub extern "win64" fn XUserAddAsync(_options: u32, async_block: *mut XAsyncBlock) -> i32 {
    complete_builtin_async(
        async_block,
        HRESULT_S_OK,
        Some((LOCAL_USER_HANDLE as u64).to_le_bytes().to_vec()),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserAddResult(
    async_block: *mut XAsyncBlock,
    out_user: *mut *mut c_void,
) -> i32 {
    if out_user.is_null() {
        return HRESULT_E_POINTER;
    }
    match take_builtin_async_result(async_block) {
        Ok((_, Some(bytes))) if bytes.len() >= 8 => {
            let h = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            // SAFETY: `out_user` was verified non-null.
            unsafe { *out_user = h as *mut c_void };
            HRESULT_S_OK
        }
        Ok(_) => {
            // SAFETY: `out_user` was verified non-null.
            unsafe { *out_user = LOCAL_USER_HANDLE as *mut c_void };
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserCloseHandle(_user: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserDuplicateHandle(user: *mut c_void, out_user: *mut *mut c_void) -> i32 {
    if user.is_null() || out_user.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    // SAFETY: `out_user` was verified non-null.
    unsafe { *out_user = user };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetId(_user: *mut c_void, out_xuid: *mut u64) -> i32 {
    if out_xuid.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_xuid` was verified non-null.
    unsafe { *out_xuid = 0x0009_0000_0000_0001 };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetLocalId(_user: *mut c_void, out_local_id: *mut XUserLocalId) -> i32 {
    if out_local_id.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_local_id` was verified non-null.
    unsafe { *out_local_id = XUserLocalId { value: 1 } };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetGamertag(
    _user: *mut c_void,
    _component: u32,
    buf_size: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    let tag = query_bridge_string(OP_XGR_USER_GET_GAMERTAG, "LinuxPlayer");
    copy_cstr_to_guest(&tag, buf_size, buf, used)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetState(_user: *mut c_void, out_state: *mut u32) -> i32 {
    if out_state.is_null() {
        return HRESULT_E_POINTER;
    }
    // `XUserState::SignedIn = 0`
    // SAFETY: `out_state` was verified non-null.
    unsafe { *out_state = 0 };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetAgeGroup(_user: *mut c_void, out_age: *mut u32) -> i32 {
    if out_age.is_null() {
        return HRESULT_E_POINTER;
    }
    // `XUserAgeGroup::Adult = 3`
    // SAFETY: `out_age` was verified non-null.
    unsafe { *out_age = 3 };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserCheckPrivilege(
    _user: *mut c_void,
    _options: u32,
    _privilege: u32,
    out_has_privilege: *mut bool,
    out_reason: *mut u32,
) -> i32 {
    if out_has_privilege.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_has_privilege` is non-null; `out_reason` is optional.
    unsafe {
        *out_has_privilege = true;
        if !out_reason.is_null() {
            *out_reason = 0;
        }
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserResolvePrivilegeWithUiAsync(
    user: *mut c_void,
    options: u32,
    privilege: u32,
    block: *mut XAsyncBlock,
) -> i32 {
    if user as usize != LOCAL_USER_HANDLE || block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let mut has_privilege = false;
    let status = XUserCheckPrivilege(
        user,
        options,
        privilege,
        &mut has_privilege,
        ptr::null_mut(),
    );
    if status != HRESULT_S_OK {
        return status;
    }
    if !has_privilege {
        return record_stub(
            "xgameruntime.dll",
            "XUserResolvePrivilegeWithUiAsync",
            HRESULT_E_NOTIMPL as u64,
        ) as i32;
    }
    complete_builtin_async(block, HRESULT_S_OK, None)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserResolvePrivilegeWithUiResult(block: *mut XAsyncBlock) -> i32 {
    take_builtin_async_result(block).map_or_else(|hr| hr, |(status, _)| status)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "win64" fn XUserGetTokenAndSignatureAsync(
    _user: *mut c_void,
    _options: u32,
    _method: *const c_char,
    _url: *const c_char,
    _header_count: usize,
    _headers: *const c_void,
    _body_size: usize,
    _body: *const c_void,
    async_block: *mut XAsyncBlock,
) -> i32 {
    // Strictly offline: complete cleanly with `E_GAMERUNTIME_NOT_SIGNED_IN` / offline token error.
    complete_builtin_async(async_block, HRESULT_E_USER_OFFLINE_TOKEN, None)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetTokenAndSignatureResultSize(
    async_block: *mut XAsyncBlock,
    out_size: *mut usize,
) -> i32 {
    XAsyncGetResultSize(async_block, out_size)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetTokenAndSignatureResult(
    async_block: *mut XAsyncBlock,
    _buffer_size: usize,
    _buffer: *mut c_void,
    _out_data: *mut *mut c_void,
    _out_used: *mut usize,
) -> i32 {
    take_builtin_async_result(async_block)
        .map(|(st, _)| st)
        .unwrap_or_else(|e| e)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "win64" fn XUserGetTokenAndSignatureUtf16Async(
    _user: *mut c_void,
    _options: u32,
    _method: *const u16,
    _url: *const u16,
    _header_count: usize,
    _headers: *const c_void,
    _body_size: usize,
    _body: *const c_void,
    async_block: *mut XAsyncBlock,
) -> i32 {
    complete_builtin_async(async_block, HRESULT_E_USER_OFFLINE_TOKEN, None)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetTokenAndSignatureUtf16ResultSize(
    async_block: *mut XAsyncBlock,
    out_size: *mut usize,
) -> i32 {
    XUserGetTokenAndSignatureResultSize(async_block, out_size)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetTokenAndSignatureUtf16Result(
    async_block: *mut XAsyncBlock,
    buffer_size: usize,
    buffer: *mut c_void,
    out_data: *mut *mut c_void,
    out_used: *mut usize,
) -> i32 {
    XUserGetTokenAndSignatureResult(async_block, buffer_size, buffer, out_data, out_used)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserRegisterForChangeEvent(
    _queue: *mut c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out_token: *mut u64,
) -> i32 {
    if out_token.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_token` was verified non-null.
    unsafe { *out_token = NEXT_EVENT_TOKEN.fetch_add(1, Ordering::Relaxed) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserUnregisterForChangeEvent(_token: u64, _wait: bool) -> bool {
    true
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserRegisterForDefaultAudioEndpointUtf16Changed(
    _queue: *mut c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out_token: *mut u64,
) -> i32 {
    if out_token.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_token` was verified non-null.
    unsafe { *out_token = NEXT_EVENT_TOKEN.fetch_add(1, Ordering::Relaxed) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserUnregisterForDefaultAudioEndpointUtf16Changed(
    _token: u64,
    _wait: bool,
) -> bool {
    true
}

#[unsafe(no_mangle)]
pub extern "win64" fn XUserGetMaxUsers(out_max: *mut u32) -> i32 {
    if out_max.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_max` was verified non-null.
    unsafe { *out_max = 1 };
    HRESULT_S_OK
}

// ============================================================================
// XGameSave (Atomic XDG Persistence via `libx2l_core.so`)
// ============================================================================

pub type XGameSaveBlobInfoCallback =
    unsafe extern "win64" fn(info: *const XGameSaveBlobInfo, context: *mut c_void) -> bool;

fn gamesave_root_status() -> i32 {
    let mut scratch = [0u8; 4096];
    // SAFETY: The bridge receives writable storage for the native save path.
    unsafe {
        call_bridge(
            OP_XGR_GAMESAVE_ROOT,
            scratch.as_mut_ptr() as u64,
            scratch.len() as u64,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveInitializeProvider(
    user: *mut c_void,
    scid: *const c_char,
    _sync_on_demand: bool,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the provider handle.
    unsafe { out.write(ptr::null_mut()) };
    if user as usize != LOCAL_USER_HANDLE || scid.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let status = gamesave_root_status();
    if status == HRESULT_S_OK {
        // SAFETY: The caller supplies writable storage for the provider handle.
        unsafe { out.write(GAMESAVE_PROVIDER_HANDLE as *mut c_void) };
    }
    status
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveInitializeProviderAsync(
    _user: *mut c_void,
    _scid: *const c_char,
    _sync_on_demand: bool,
    async_block: *mut XAsyncBlock,
) -> i32 {
    let _ = gamesave_root_status();
    complete_builtin_async(
        async_block,
        HRESULT_S_OK,
        Some((GAMESAVE_PROVIDER_HANDLE as u64).to_le_bytes().to_vec()),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveInitializeProviderResult(
    async_block: *mut XAsyncBlock,
    out_provider: *mut *mut c_void,
) -> i32 {
    if out_provider.is_null() {
        return HRESULT_E_POINTER;
    }
    match take_builtin_async_result(async_block) {
        Ok(_) => {
            // SAFETY: `out_provider` was verified non-null.
            unsafe { *out_provider = GAMESAVE_PROVIDER_HANDLE as *mut c_void };
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveCloseProvider(_provider: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveCreateContainer(
    _provider: *mut c_void,
    name: *const c_char,
    out_container: *mut *mut c_void,
) -> i32 {
    let Some(cname) = read_cstr_opt(name) else {
        return HRESULT_E_INVALIDARG;
    };
    if out_container.is_null() {
        return HRESULT_E_POINTER;
    }
    let h = NEXT_GS_HANDLE.fetch_add(1, Ordering::Relaxed) as usize;
    if let Ok(mut map) = gs_containers().lock() {
        map.insert(h, cname);
    }
    // SAFETY: `out_container` was verified non-null.
    unsafe { *out_container = h as *mut c_void };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveCloseContainer(container: *mut c_void) {
    if let Ok(mut map) = gs_containers().lock() {
        map.remove(&(container as usize));
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveEnumerateBlobInfo(
    container: *mut c_void,
    context: *mut c_void,
    callback: Option<XGameSaveBlobInfoCallback>,
) -> i32 {
    XGameSaveEnumerateBlobInfoByName(container, ptr::null(), context, callback)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveEnumerateBlobInfoByName(
    container: *mut c_void,
    prefix: *const c_char,
    context: *mut c_void,
    callback: Option<XGameSaveBlobInfoCallback>,
) -> i32 {
    let Some(cb) = callback else {
        return HRESULT_E_INVALIDARG;
    };
    let Some(cname) = gs_containers()
        .lock()
        .ok()
        .and_then(|m| m.get(&(container as usize)).cloned())
    else {
        return HRESULT_E_INVALIDARG;
    };
    let pfx = read_cstr_opt(prefix).unwrap_or_default();
    let mut packed = vec![0u8; 64 * 1024];
    let mut req = GameSaveEnumReq {
        container_ptr: cname.as_ptr() as usize as u64,
        container_len: cname.len() as u64,
        prefix_ptr: pfx.as_ptr() as usize as u64,
        prefix_len: pfx.len() as u64,
        out_ptr: packed.as_mut_ptr() as usize as u64,
        out_cap: packed.len() as u64,
        out_count: 0,
        out_bytes: 0,
    };
    // SAFETY: `req` and `packed` remain alive for the synchronous bridge call.
    let hr = unsafe {
        call_bridge(
            OP_XGR_GAMESAVE_ENUM_BLOBS,
            (&mut req as *mut GameSaveEnumReq) as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_NOTIMPL as u64) as i32
    };
    if hr != HRESULT_S_OK {
        return hr;
    }
    let used = (req.out_bytes as usize).min(packed.len());
    let mut off = 0usize;
    while off + 8 <= used {
        let sz = u32::from_le_bytes(packed[off..off + 4].try_into().unwrap());
        let nlen = u32::from_le_bytes(packed[off + 4..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if off + nlen > used {
            break;
        }
        let mut c_name = Vec::with_capacity(nlen + 1);
        c_name.extend_from_slice(&packed[off..off + nlen]);
        c_name.push(0);
        off += nlen;
        let info = XGameSaveBlobInfo {
            name: c_name.as_ptr().cast::<c_char>(),
            size: sz,
        };
        // SAFETY: `XGameSaveBlobInfoCallback` takes `(info, context)` per GDK ABI.
        if !unsafe { cb(&info, context) } {
            break;
        }
    }
    HRESULT_S_OK
}

fn read_single_save_blob(container_name: &str, blob_name: &str) -> Result<Vec<u8>, i32> {
    let mut probe = GameSaveReadReq {
        container_ptr: container_name.as_ptr() as usize as u64,
        container_len: container_name.len() as u64,
        blob_ptr: blob_name.as_ptr() as usize as u64,
        blob_len: blob_name.len() as u64,
        out_ptr: 0,
        out_cap: 0,
        out_size: 0,
    };
    // SAFETY: Probe call reads exact byte length into `probe.out_size`.
    let hr = unsafe {
        call_bridge(
            OP_XGR_GAMESAVE_READ_BLOB,
            (&mut probe as *mut GameSaveReadReq) as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_GS_BLOB_NOT_FOUND as u64) as i32
    };
    if hr == HRESULT_E_GS_BLOB_NOT_FOUND {
        return Err(HRESULT_E_GS_BLOB_NOT_FOUND);
    }
    let size = probe.out_size as usize;
    let mut data = vec![0u8; size];
    if size == 0 {
        return Ok(data);
    }
    probe.out_ptr = data.as_mut_ptr() as usize as u64;
    probe.out_cap = data.len() as u64;
    // SAFETY: `data` has exact capacity `size`.
    let hr2 = unsafe {
        call_bridge(
            OP_XGR_GAMESAVE_READ_BLOB,
            (&mut probe as *mut GameSaveReadReq) as usize as u64,
            0,
            0,
            0,
            0,
            0,
        )
        .unwrap_or(HRESULT_E_GS_BLOB_NOT_FOUND as u64) as i32
    };
    if hr2 == HRESULT_S_OK {
        Ok(data)
    } else {
        Err(hr2)
    }
}

fn pack_blobs_into_guest_buf(
    blobs: &[(String, Vec<u8>)],
    capacity: usize,
    out: *mut XGameSaveBlob,
) -> i32 {
    let header_bytes = blobs.len() * size_of::<XGameSaveBlob>();
    let mut total = header_bytes;
    for (name, data) in blobs {
        total += name.len() + 1 + data.len();
    }
    if out.is_null() || capacity < total {
        return HRESULT_E_GS_BUFFER_TOO_SMALL;
    }
    let base = out.cast::<u8>();
    let mut cursor = header_bytes;
    for (idx, (name, data)) in blobs.iter().enumerate() {
        // SAFETY: `capacity >= total` verified above; all pointers stay inside caller's buffer.
        unsafe {
            let name_ptr = base.add(cursor);
            ptr::copy_nonoverlapping(name.as_ptr(), name_ptr, name.len());
            *name_ptr.add(name.len()) = 0;
            cursor += name.len() + 1;

            let data_ptr = base.add(cursor);
            if !data.is_empty() {
                ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
                cursor += data.len();
            }

            *out.add(idx) = XGameSaveBlob {
                info: XGameSaveBlobInfo {
                    name: name_ptr.cast::<c_char>(),
                    size: data.len() as u32,
                },
                data: data_ptr,
            };
        }
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveReadBlobData(
    container: *mut c_void,
    blob_names: *const *const c_char,
    blob_count: *mut u32,
    blobs_size: usize,
    blobs_out: *mut XGameSaveBlob,
) -> i32 {
    if blob_names.is_null() || blob_count.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let Some(cname) = gs_containers()
        .lock()
        .ok()
        .and_then(|m| m.get(&(container as usize)).cloned())
    else {
        return HRESULT_E_INVALIDARG;
    };
    // SAFETY: Caller passes readable `blob_count` and `*blob_count` C-string pointers.
    let count = unsafe { *blob_count } as usize;
    let mut loaded = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: `i < count`.
        let name_ptr = unsafe { *blob_names.add(i) };
        let Some(bname) = read_cstr_opt(name_ptr) else {
            return HRESULT_E_INVALIDARG;
        };
        match read_single_save_blob(&cname, &bname) {
            Ok(data) => loaded.push((bname, data)),
            Err(hr) => return hr,
        }
    }
    let hr = pack_blobs_into_guest_buf(&loaded, blobs_size, blobs_out);
    if hr == HRESULT_S_OK {
        // SAFETY: `blob_count` was verified non-null.
        unsafe { *blob_count = loaded.len() as u32 };
    }
    hr
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveReadBlobDataAsync(
    container: *mut c_void,
    blob_names: *const *const c_char,
    blob_count: u32,
    async_block: *mut XAsyncBlock,
) -> i32 {
    if blob_names.is_null() || async_block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let Some(cname) = gs_containers()
        .lock()
        .ok()
        .and_then(|m| m.get(&(container as usize)).cloned())
    else {
        return complete_builtin_async(async_block, HRESULT_E_INVALIDARG, None);
    };
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&blob_count.to_le_bytes());
    for i in 0..(blob_count as usize) {
        // SAFETY: `i < blob_count`.
        let name_ptr = unsafe { *blob_names.add(i) };
        let Some(bname) = read_cstr_opt(name_ptr) else {
            return complete_builtin_async(async_block, HRESULT_E_INVALIDARG, None);
        };
        match read_single_save_blob(&cname, &bname) {
            Ok(data) => {
                encoded.extend_from_slice(&(bname.len() as u32).to_le_bytes());
                encoded.extend_from_slice(bname.as_bytes());
                encoded.extend_from_slice(&(data.len() as u32).to_le_bytes());
                encoded.extend_from_slice(&data);
            }
            Err(hr) => return complete_builtin_async(async_block, hr, None),
        }
    }
    complete_builtin_async(async_block, HRESULT_S_OK, Some(encoded))
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveReadBlobDataResult(
    async_block: *mut XAsyncBlock,
    blobs_size: usize,
    blobs_out: *mut XGameSaveBlob,
    count_of_blobs: *mut u32,
) -> i32 {
    let bytes = match take_builtin_async_result(async_block) {
        Ok((_, Some(b))) if b.len() >= 4 => b,
        Ok(_) => return HRESULT_E_INVALIDARG,
        Err(hr) => return hr,
    };
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut off = 4usize;
    let mut loaded = Vec::with_capacity(count);
    for _ in 0..count {
        if off + 4 > bytes.len() {
            return HRESULT_E_INVALIDARG;
        }
        let nlen = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if off + nlen + 4 > bytes.len() {
            return HRESULT_E_INVALIDARG;
        }
        let name = String::from_utf8_lossy(&bytes[off..off + nlen]).into_owned();
        off += nlen;
        let dlen = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if off + dlen > bytes.len() {
            return HRESULT_E_INVALIDARG;
        }
        loaded.push((name, bytes[off..off + dlen].to_vec()));
        off += dlen;
    }
    let hr = pack_blobs_into_guest_buf(&loaded, blobs_size, blobs_out);
    if hr == HRESULT_S_OK && !count_of_blobs.is_null() {
        // SAFETY: Caller passed optional writable `count_of_blobs`.
        unsafe { *count_of_blobs = loaded.len() as u32 };
    }
    hr
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveCreateUpdate(
    container: *mut c_void,
    _display_name: *const c_char,
    out_update: *mut *mut c_void,
) -> i32 {
    if out_update.is_null() {
        return HRESULT_E_POINTER;
    }
    let Some(cname) = gs_containers()
        .lock()
        .ok()
        .and_then(|m| m.get(&(container as usize)).cloned())
    else {
        return HRESULT_E_INVALIDARG;
    };
    let h = NEXT_GS_HANDLE.fetch_add(1, Ordering::Relaxed) as usize;
    if let Ok(mut map) = gs_updates().lock() {
        map.insert(
            h,
            GameSaveUpdateState {
                container_name: cname,
                pending_writes: Vec::new(),
            },
        );
    }
    // SAFETY: `out_update` was verified non-null.
    unsafe { *out_update = h as *mut c_void };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveSubmitBlobWrite(
    update: *mut c_void,
    blob_name: *const c_char,
    data: *const u8,
    byte_count: usize,
) -> i32 {
    let Some(bname) = read_cstr_opt(blob_name) else {
        return HRESULT_E_INVALIDARG;
    };
    let payload = if data.is_null() || byte_count == 0 {
        Vec::new()
    } else {
        // SAFETY: Caller passes readable `[data .. data + byte_count]`.
        unsafe { std::slice::from_raw_parts(data, byte_count) }.to_vec()
    };
    let Ok(mut map) = gs_updates().lock() else {
        return HRESULT_E_INVALIDARG;
    };
    let Some(st) = map.get_mut(&(update as usize)) else {
        return HRESULT_E_INVALIDARG;
    };
    st.pending_writes.push((bname, payload));
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveSubmitUpdateAsync(
    update: *mut c_void,
    async_block: *mut XAsyncBlock,
) -> i32 {
    let staged = gs_updates().lock().ok().and_then(|mut m| {
        m.get_mut(&(update as usize)).map(|u| {
            (
                u.container_name.clone(),
                std::mem::take(&mut u.pending_writes),
            )
        })
    });
    let Some((cname, writes)) = staged else {
        return complete_builtin_async(async_block, HRESULT_E_INVALIDARG, None);
    };
    for (bname, data) in writes {
        let hr = write_save_blob(&cname, &bname, &data);
        if hr != HRESULT_S_OK {
            return complete_builtin_async(async_block, hr, None);
        }
    }
    complete_builtin_async(async_block, HRESULT_S_OK, None)
}

fn write_save_blob(container: &str, name: &str, data: &[u8]) -> i32 {
    let req = GameSaveWriteReq {
        container_ptr: container.as_ptr() as usize as u64,
        container_len: container.len() as u64,
        blob_ptr: name.as_ptr() as usize as u64,
        blob_len: name.len() as u64,
        data_ptr: data.as_ptr() as usize as u64,
        data_len: data.len() as u64,
    };
    // SAFETY: `req`, `container`, `name`, and `data` remain alive across the synchronous bridge call.
    unsafe {
        call_bridge(
            OP_XGR_GAMESAVE_WRITE_BLOB,
            (&req as *const GameSaveWriteReq) as usize as u64,
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
pub extern "win64" fn XGameSaveSubmitUpdate(update: *mut c_void) -> i32 {
    let Ok(mut updates) = gs_updates().lock() else {
        return HRESULT_E_INVALIDARG;
    };
    let Some(state) = updates.get_mut(&(update as usize)) else {
        return HRESULT_E_INVALIDARG;
    };
    let [(name, data)] = state.pending_writes.as_slice() else {
        if state.pending_writes.is_empty() {
            return HRESULT_S_OK;
        }
        // The current native bridge can commit one blob atomically.
        return record_stub(
            "xgameruntime.dll",
            "XGameSaveSubmitUpdate",
            HRESULT_E_NOTIMPL as u64,
        ) as i32;
    };
    let status = write_save_blob(&state.container_name, name, data);
    if status == HRESULT_S_OK {
        state.pending_writes.clear();
    }
    status
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveSubmitUpdateResult(async_block: *mut XAsyncBlock) -> i32 {
    take_builtin_async_result(async_block)
        .map(|_| HRESULT_S_OK)
        .unwrap_or_else(|e| e)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveCloseUpdate(update: *mut c_void) {
    if let Ok(mut map) = gs_updates().lock() {
        map.remove(&(update as usize));
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveGetRemainingQuotaAsync(
    _provider: *mut c_void,
    async_block: *mut XAsyncBlock,
) -> i32 {
    complete_builtin_async(
        async_block,
        HRESULT_S_OK,
        Some(OFFLINE_GAMESAVE_QUOTA.to_le_bytes().to_vec()),
    )
}

const OFFLINE_GAMESAVE_QUOTA: i64 = 256 * 1024 * 1024;

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveGetRemainingQuota(
    provider: *mut c_void,
    out_quota: *mut i64,
) -> i32 {
    if out_quota.is_null() {
        return HRESULT_E_POINTER;
    }
    if provider as usize != GAMESAVE_PROVIDER_HANDLE {
        return HRESULT_E_INVALIDARG;
    }
    // SAFETY: The caller supplies writable storage for the quota.
    unsafe { out_quota.write(OFFLINE_GAMESAVE_QUOTA) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameSaveGetRemainingQuotaResult(
    async_block: *mut XAsyncBlock,
    out_quota: *mut i64,
) -> i32 {
    if out_quota.is_null() {
        return HRESULT_E_POINTER;
    }
    match take_builtin_async_result(async_block) {
        Ok((_, Some(bytes))) if bytes.len() >= 8 => {
            let q = i64::from_le_bytes(bytes[..8].try_into().unwrap());
            // SAFETY: `out_quota` was verified non-null.
            unsafe { *out_quota = q };
            HRESULT_S_OK
        }
        Ok(_) => HRESULT_E_INVALIDARG,
        Err(hr) => hr,
    }
}

// ============================================================================
// XStore (Strictly Offline — Zero License Spoofing / Zero DRM Bypass)
// ============================================================================

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreCreateContext(_user: *mut c_void, out_ctx: *mut *mut c_void) -> i32 {
    if !out_ctx.is_null() {
        // SAFETY: Clear output handle on offline refusal.
        unsafe { *out_ctx = ptr::null_mut() };
    }
    // SAFETY: Offline-only; never fakes commercial licenses.
    unsafe {
        call_bridge(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0).unwrap_or(HRESULT_E_NOTIMPL as u64)
            as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreCloseContextHandle(_ctx: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreQueryGameLicenseAsync(
    _ctx: *mut c_void,
    _async_block: *mut XAsyncBlock,
) -> i32 {
    // SAFETY: Offline-only; never fakes commercial licenses.
    unsafe {
        call_bridge(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0).unwrap_or(HRESULT_E_NOTIMPL as u64)
            as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreQueryGameLicenseResult(
    _async_block: *mut XAsyncBlock,
    _out_license: *mut XStoreGameLicense,
) -> i32 {
    HRESULT_E_NOTIMPL
}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreQueryAddOnLicensesAsync(
    _ctx: *mut c_void,
    _async_block: *mut XAsyncBlock,
) -> i32 {
    // SAFETY: Offline-only; never fakes commercial licenses.
    unsafe {
        call_bridge(OP_XGR_STORE_OFFLINE, 0, 0, 0, 0, 0, 0).unwrap_or(HRESULT_E_NOTIMPL as u64)
            as i32
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XStoreQueryAddOnLicensesResult(
    _async_block: *mut XAsyncBlock,
    _count: u32,
    _out_licenses: *mut c_void,
) -> i32 {
    HRESULT_E_NOTIMPL
}

// ============================================================================
// XNetworking, XGameUi, & XLaunch (Offline Telemetry)
// ============================================================================

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingGetConnectivityHint(
    out_hint: *mut XNetworkingConnectivityHint,
) -> i32 {
    if out_hint.is_null() {
        return HRESULT_E_POINTER;
    }
    // Report offline/local-only network initialized cleanly (`None = 1`, `Unrestricted = 1`).
    // SAFETY: `out_hint` was verified non-null.
    unsafe {
        *out_hint = XNetworkingConnectivityHint {
            connectivity_level: 1,  // None (offline)
            connectivity_cost: 1,   // Unrestricted
            iana_interface_type: 6, // Ethernet
            network_initialized: true,
            approaching_data_limit: false,
            over_data_limit: false,
            roaming: false,
        };
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingRegisterConnectivityHintChanged(
    _queue: *mut c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out_token: *mut u64,
) -> i32 {
    if out_token.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out_token` was verified non-null.
    unsafe { *out_token = NEXT_EVENT_TOKEN.fetch_add(1, Ordering::Relaxed) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingUnregisterConnectivityHintChanged(
    _token: u64,
    _wait: bool,
) -> bool {
    true
}

const DEFAULT_MULTIPLAYER_PORT: u16 = 3074;
static PREFERRED_PORT_ASYNC_IDENTITY: u8 = 2;

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingQueryPreferredLocalUdpMultiplayerPort(out: *mut u16) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for a port number.
    unsafe { out.write(DEFAULT_MULTIPLAYER_PORT) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingQueryPreferredLocalUdpMultiplayerPortAsync(
    block: *mut XAsyncBlock,
) -> i32 {
    complete_builtin_async_with_identity(
        block,
        (&raw const PREFERRED_PORT_ASYNC_IDENTITY).cast(),
        HRESULT_S_OK,
        Some(DEFAULT_MULTIPLAYER_PORT.to_le_bytes().to_vec()),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingQueryPreferredLocalUdpMultiplayerPortAsyncResult(
    block: *mut XAsyncBlock,
    out: *mut u16,
) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    match read_builtin_async_payload(block, &raw const PREFERRED_PORT_ASYNC_IDENTITY) {
        Ok(bytes) => {
            // SAFETY: The caller supplies writable storage for a port number.
            unsafe { out.write(u16::from_le_bytes(bytes)) };
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingRegisterPreferredLocalUdpMultiplayerPortChanged(
    queue: *mut c_void,
    _context: *mut c_void,
    callback: *const c_void,
    out_token: *mut u64,
) -> i32 {
    if out_token.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the registration token.
    unsafe { out_token.write(0) };
    if callback.is_null() || !is_valid_queue_handle(queue) {
        return HRESULT_E_INVALIDARG;
    }
    let Ok(mut registrations) = PREFERRED_PORT_REGISTRATIONS.lock() else {
        return HRESULT_E_INVALIDARG;
    };
    let token = NEXT_EVENT_TOKEN.fetch_add(1, Ordering::Relaxed);
    // The offline preference is immutable, so there are no change callbacks.
    registrations.insert(token);
    // SAFETY: The caller supplies writable storage for the registration token.
    unsafe { out_token.write(token) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingUnregisterPreferredLocalUdpMultiplayerPortChanged(
    token: u64,
    _wait: bool,
) -> bool {
    PREFERRED_PORT_REGISTRATIONS
        .lock()
        .is_ok_and(|mut registrations| registrations.remove(&token))
}

#[unsafe(no_mangle)]
pub extern "win64" fn XNetworkingVerifyServerCertificate(
    _request: *mut c_void,
    _security_info: *const c_void,
) -> i32 {
    // Offline runtime never falsely approves untrusted TLS certificates.
    record_stub(
        "xgameruntime.dll",
        "XNetworkingVerifyServerCertificate",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "win64" fn XGameUiShowMessageDialogAsync(
    async_block: *mut XAsyncBlock,
    _title: *const c_char,
    _content: *const c_char,
    _first_button: *const c_char,
    _second_button: *const c_char,
    _third_button: *const c_char,
    default_button: u32,
    _cancel_button: u32,
) -> i32 {
    complete_builtin_async(
        async_block,
        HRESULT_S_OK,
        Some(default_button.to_le_bytes().to_vec()),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XGameUiShowMessageDialogResult(
    async_block: *mut XAsyncBlock,
    out_button: *mut u32,
) -> i32 {
    if out_button.is_null() {
        return HRESULT_E_POINTER;
    }
    match take_builtin_async_result(async_block) {
        Ok((_, Some(bytes))) if bytes.len() >= 4 => {
            let btn = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            // SAFETY: `out_button` was verified non-null.
            unsafe { *out_button = btn };
            HRESULT_S_OK
        }
        Ok(_) => {
            // SAFETY: `out_button` was verified non-null.
            unsafe { *out_button = 0 };
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XLaunchNewGame(
    _executable: *const c_char,
    _arguments: *const c_char,
    _user: *mut c_void,
) {
    let _ = record_stub("xgameruntime.dll", "XLaunchNewGame", 0);
}

#[unsafe(no_mangle)]
pub extern "win64" fn XLaunchUri(_user: *mut c_void, _uri: *const c_char) -> i32 {
    record_stub("xgameruntime.dll", "XLaunchUri", HRESULT_E_NOTIMPL as u64) as i32
}
