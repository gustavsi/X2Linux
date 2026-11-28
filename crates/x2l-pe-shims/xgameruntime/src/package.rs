// Win64 adapters for the offline package model.
// Slot positions are documented in query_api.rs. Public XPackage documentation
// supplies the named export signatures and structure layouts.

#[path = "../../../x2l-hle-xgr/src/package.rs"]
mod offline;

use crate::{
    HRESULT_E_INVALIDARG, HRESULT_E_NOTIMPL, HRESULT_E_POINTER, HRESULT_S_OK, PACKAGE_MOUNT_HANDLE,
    XAsyncBlock, XPackageDetails, XSystemVersion, complete_builtin_async_with_identity,
    copy_cstr_to_guest, is_valid_queue_handle, query_bridge_string, read_builtin_async_payload,
    record_stub,
};
pub use offline::PackageInstallationProgress as XPackageInstallationProgress;
use offline::{
    NO_PACKAGE, PACKAGE_IDENTIFIER, PACKAGE_MOUNT_ROOT, PackageMonitors, is_local_package,
};
use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::{LazyLock, Mutex};
use x2l_shim_common::OP_XGR_PACKAGE_MOUNT_ROOT;

const HRESULT_ACCESS_DENIED: i32 = 0x8007_0005_u32 as i32;
const HRESULT_FILE_NOT_FOUND: i32 = 0x8007_0002_u32 as i32;
static MONITORS: LazyLock<Mutex<PackageMonitors>> = LazyLock::new(Mutex::default);
static INSTALL_ASYNC_IDENTITY: u8 = 0;
static MOUNT_ASYNC_IDENTITY: u8 = 1;

pub type XPackageEnumerationCallback =
    unsafe extern "win64" fn(context: *mut c_void, details: *const XPackageDetails) -> bool;
pub type XPackageInstallationProgressCallback =
    unsafe extern "win64" fn(context: *mut c_void, monitor: *mut c_void);
pub type XPackageInstalledCallback =
    unsafe extern "win64" fn(context: *mut c_void, details: *const XPackageDetails);

pub(crate) fn package_monitor_is_valid(handle: *mut c_void) -> bool {
    MONITORS
        .lock()
        .is_ok_and(|monitors| monitors.progress(handle as usize).is_some())
}

fn validate_package(package: *const c_char) -> Result<(), i32> {
    if package.is_null() {
        return Err(HRESULT_E_INVALIDARG);
    }
    // SAFETY: The caller supplies a NUL-terminated package identifier.
    let identifier = unsafe { CStr::from_ptr(package) };
    if !is_local_package(identifier) {
        return Err(NO_PACKAGE);
    }
    Ok(())
}

fn validate_selectors(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
) -> Result<(), i32> {
    if count != 0 && selectors.is_null() {
        return Err(HRESULT_E_INVALIDARG);
    }
    validate_package(package)
}

fn open_monitor(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
) -> Result<usize, i32> {
    validate_selectors(package, count, selectors)?;
    MONITORS
        .lock()
        .map_err(|_| HRESULT_E_INVALIDARG)?
        .create(PACKAGE_IDENTIFIER)
}

fn complete_handle(
    block: *mut XAsyncBlock,
    identity: *const u8,
    result: Result<usize, i32>,
) -> i32 {
    let (status, payload) = match result {
        Ok(handle) => (HRESULT_S_OK, Some(handle.to_le_bytes().to_vec())),
        Err(hr) => (hr, None),
    };
    complete_builtin_async_with_identity(block, identity.cast(), status, payload)
}

fn get_handle_result(block: *mut XAsyncBlock, identity: *const u8, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for one handle.
    unsafe { out.write(ptr::null_mut()) };
    match read_builtin_async_payload(block, identity) {
        Ok(bytes) => {
            // SAFETY: The caller supplies writable storage for one handle.
            unsafe { out.write(usize::from_le_bytes(bytes) as *mut c_void) };
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetCurrentProcessPackageIdentifier(
    capacity: usize,
    buffer: *mut u8,
) -> i32 {
    copy_cstr_to_guest(
        PACKAGE_IDENTIFIER
            .to_str()
            .expect("ASCII package identifier"),
        capacity,
        buffer,
        ptr::null_mut(),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageIsPackagedProcess() -> bool {
    true
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageCreateInstallationMonitor(
    package_identifier: *const c_char,
    selector_count: u32,
    selectors: *const c_void,
    minimum_update_interval_ms: u32,
    queue: *mut c_void,
    out_monitor: *mut *mut c_void,
) -> i32 {
    if out_monitor.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the output handle.
    unsafe { out_monitor.write(ptr::null_mut()) };
    if (!queue.is_null() || minimum_update_interval_ms != 0) && !is_valid_queue_handle(queue) {
        return HRESULT_E_INVALIDARG;
    }
    match open_monitor(package_identifier, selector_count, selectors) {
        Ok(handle) => {
            // SAFETY: The caller supplies writable storage for the output handle.
            unsafe { out_monitor.write(handle as *mut c_void) };
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageCloseInstallationMonitorHandle(monitor: *mut c_void) {
    if let Ok(mut monitors) = MONITORS.lock() {
        monitors.close(monitor as usize);
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetInstallationProgress(
    monitor: *mut c_void,
    out: *mut XPackageInstallationProgress,
) {
    if out.is_null() {
        return;
    }
    let progress = MONITORS
        .lock()
        .ok()
        .and_then(|monitors| monitors.progress(monitor as usize));
    // SAFETY: The caller supplies writable storage for one installation snapshot.
    unsafe { out.write(progress.unwrap_or_default()) };
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageUpdateInstallationMonitor(_monitor: *mut c_void) -> bool {
    // A fully unpacked installation has no progress changes to report.
    false
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageRegisterInstallationProgressChanged(
    monitor: *mut c_void,
    _context: *mut c_void,
    callback: Option<XPackageInstallationProgressCallback>,
    out_token: *mut u64,
) -> i32 {
    if out_token.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the registration token.
    unsafe { out_token.write(0) };
    if callback.is_none() {
        return HRESULT_E_INVALIDARG;
    }
    let Some(token) = MONITORS
        .lock()
        .ok()
        .and_then(|mut monitors| monitors.register_progress(monitor as usize))
    else {
        return HRESULT_E_INVALIDARG;
    };
    // SAFETY: The caller supplies writable storage for the registration token.
    unsafe { out_token.write(token) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageUnregisterInstallationProgressChanged(
    monitor: *mut c_void,
    token: u64,
    _wait: bool,
) -> bool {
    MONITORS
        .lock()
        .is_ok_and(|mut monitors| monitors.unregister_progress(monitor as usize, token))
}

#[link(name = "kernel32")]
// SAFETY: This declaration matches the Win32 locale API and its system ABI.
unsafe extern "system" {
    fn GetUserDefaultLocaleName(locale_name: *mut u16, locale_name_count: i32) -> i32;
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetUserLocale(capacity: usize, buffer: *mut u8) -> i32 {
    let mut locale = [0u16; 85];
    // SAFETY: The buffer is writable for the supplied number of UTF-16 code units.
    let len = unsafe { GetUserDefaultLocaleName(locale.as_mut_ptr(), locale.len() as i32) };
    let name = if len > 1 {
        String::from_utf16_lossy(&locale[..len as usize - 1])
    } else {
        "en-US".to_owned()
    };
    copy_cstr_to_guest(&name, capacity, buffer, ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageFindChunkAvailability(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
    out: *mut u32,
) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    if let Err(hr) = validate_selectors(package, count, selectors) {
        return hr;
    }
    // SAFETY: The caller supplies writable storage for XPackageChunkAvailability (Ready = 0).
    unsafe { out.write(0) };
    HRESULT_S_OK
}

pub type XPackageChunkAvailabilityCallback =
    unsafe extern "win64" fn(*mut c_void, *const c_void, u32) -> bool;

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageEnumerateChunkAvailability(
    package: *const c_char,
    selector_type: u32,
    _context: *mut c_void,
    callback: Option<XPackageChunkAvailabilityCallback>,
) -> i32 {
    if selector_type > 3 || callback.is_none() {
        return HRESULT_E_INVALIDARG;
    }
    // An unpacked installation exposes no manifest-defined selectors.
    validate_package(package).map_or_else(|hr| hr, |()| HRESULT_S_OK)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageChangeChunkInstallOrder(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
) -> i32 {
    validate_selectors(package, count, selectors).map_or_else(|hr| hr, |()| HRESULT_S_OK)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageInstallChunks(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
    interval_ms: u32,
    _suppress_confirmation: bool,
    queue: *mut c_void,
    out: *mut *mut c_void,
) -> i32 {
    XPackageCreateInstallationMonitor(package, count, selectors, interval_ms, queue, out)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageInstallChunksAsync(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
    _interval_ms: u32,
    _suppress_confirmation: bool,
    block: *mut XAsyncBlock,
) -> i32 {
    if block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let monitor = open_monitor(package, count, selectors);
    let status = complete_handle(block, &raw const INSTALL_ASYNC_IDENTITY, monitor);
    if status != HRESULT_S_OK
        && let Ok(handle) = monitor
    {
        XPackageCloseInstallationMonitorHandle(handle as *mut c_void);
    }
    status
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageInstallChunksResult(
    block: *mut XAsyncBlock,
    out: *mut *mut c_void,
) -> i32 {
    get_handle_result(block, &raw const INSTALL_ASYNC_IDENTITY, out)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageEstimateDownloadSize(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
    out_bytes: *mut u64,
    out_confirm: *mut bool,
) -> i32 {
    if out_bytes.is_null() {
        return HRESULT_E_POINTER;
    }
    if let Err(hr) = validate_selectors(package, count, selectors) {
        return hr;
    }
    // SAFETY: The caller supplies writable storage for the size and optional confirmation flag.
    unsafe {
        out_bytes.write(0);
        if !out_confirm.is_null() {
            out_confirm.write(false);
        }
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageUninstallChunks(
    package: *const c_char,
    count: u32,
    selectors: *const c_void,
) -> i32 {
    validate_selectors(package, count, selectors).map_or_else(|hr| hr, |()| HRESULT_ACCESS_DENIED)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageRegisterPackageInstalled(
    queue: *mut c_void,
    _context: *mut c_void,
    callback: Option<XPackageInstalledCallback>,
    out_token: *mut u64,
) -> i32 {
    if out_token.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the registration token.
    unsafe { out_token.write(0) };
    if callback.is_none() || !is_valid_queue_handle(queue) {
        return HRESULT_E_INVALIDARG;
    }
    let Ok(mut monitors) = MONITORS.lock() else {
        return HRESULT_E_INVALIDARG;
    };
    // SAFETY: The caller supplies writable storage for the registration token.
    unsafe { out_token.write(monitors.register_installed()) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageUnregisterPackageInstalled(token: u64, _wait: bool) -> bool {
    MONITORS
        .lock()
        .is_ok_and(|mut monitors| monitors.unregister_installed(token))
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageMount(package: *const c_char, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the mount handle.
    unsafe { out.write(ptr::null_mut()) };
    if let Err(hr) = validate_package(package) {
        return hr;
    }
    // SAFETY: The caller supplies writable storage for the mount handle.
    unsafe { out.write(PACKAGE_MOUNT_HANDLE as *mut c_void) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageMountWithUiAsync(
    package: *const c_char,
    block: *mut XAsyncBlock,
) -> i32 {
    complete_handle(
        block,
        &raw const MOUNT_ASYNC_IDENTITY,
        validate_package(package).map(|()| PACKAGE_MOUNT_HANDLE),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageMountWithUiResult(
    block: *mut XAsyncBlock,
    out: *mut *mut c_void,
) -> i32 {
    get_handle_result(block, &raw const MOUNT_ASYNC_IDENTITY, out)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetMountPathSize(_mount: *mut c_void, out: *mut usize) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    let root = query_bridge_string(OP_XGR_PACKAGE_MOUNT_ROOT, PACKAGE_MOUNT_ROOT);
    // SAFETY: The caller supplies writable storage for the path size.
    unsafe { out.write(root.len() + 1) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetMountPath(
    _mount: *mut c_void,
    capacity: usize,
    buf: *mut u8,
) -> i32 {
    let root = query_bridge_string(OP_XGR_PACKAGE_MOUNT_ROOT, PACKAGE_MOUNT_ROOT);
    copy_cstr_to_guest(&root, capacity, buf, ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageCloseMountHandle(_mount: *mut c_void) {}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetIdentifier(
    path_or_name: *const c_char,
    capacity: usize,
    buffer: *mut u8,
) -> i32 {
    if !path_or_name.is_null() {
        // SAFETY: The caller supplies a NUL-terminated path or package name.
        let name = unsafe { CStr::from_ptr(path_or_name) };
        if !is_local_package(name) && !name.to_bytes().eq_ignore_ascii_case(b"G:") {
            return NO_PACKAGE;
        }
    }
    XPackageGetCurrentProcessPackageIdentifier(capacity, buffer)
}

pub type XPackageFeatureEnumerationCallback =
    unsafe extern "win64" fn(*mut c_void, *const c_void) -> bool;

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageEnumerateFeatures(
    package: *const c_char,
    _context: *mut c_void,
    callback: Option<XPackageFeatureEnumerationCallback>,
) -> i32 {
    if callback.is_none() {
        return HRESULT_E_INVALIDARG;
    }
    // An unpacked installation exposes no manifest-defined features.
    validate_package(package).map_or_else(|hr| hr, |()| HRESULT_S_OK)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageUninstallPackage(_package: *const c_char) -> bool {
    false
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageUninstallUWPInstance(_package_name: *const c_char) -> i32 {
    record_stub(
        "xgameruntime.dll",
        "XPackageUninstallUWPInstance",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}

#[repr(C)]
pub struct XPackageWriteStats {
    pub interval: u64,
    pub budget: u64,
    pub elapsed: u64,
    pub bytes_written: u64,
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetWriteStats(_out: *mut XPackageWriteStats) -> i32 {
    record_stub(
        "xgameruntime.dll",
        "XPackageGetWriteStats",
        HRESULT_E_NOTIMPL as u64,
    ) as i32
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageGetPackageKind(package: *const c_char, out: *mut u32) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    if let Err(hr) = validate_package(package) {
        return hr;
    }
    // SAFETY: The caller supplies writable storage for the package kind (Game = 0).
    unsafe { out.write(0) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPackageEnumeratePackages(
    kind: u32,
    scope: u32,
    context: *mut c_void,
    callback: Option<XPackageEnumerationCallback>,
) -> i32 {
    let Some(cb) = callback else {
        return HRESULT_E_INVALIDARG;
    };
    if kind > 2 || scope > 2 {
        return HRESULT_E_INVALIDARG;
    }
    if kind != 0 {
        return HRESULT_S_OK;
    }
    let details = XPackageDetails {
        package_identifier: PACKAGE_IDENTIFIER.as_ptr(),
        version: XSystemVersion {
            major: 1,
            minor: 0,
            build: 0,
            revision: 0,
        },
        kind,
        display_name: c"Generic Console Package".as_ptr(),
        description: c"".as_ptr(),
        publisher: c"".as_ptr(),
        store_id: c"".as_ptr(),
        installing: false,
        index: 0,
        count: 1,
        age_restricted: false,
        title_id: c"".as_ptr(),
    };
    // SAFETY: The guest callback receives details and strings valid for its entire call.
    unsafe { cb(context, &details) };
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XPersistentLocalStorageMountForPackage(
    package: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: The caller supplies writable storage for the mount handle.
    unsafe { out.write(ptr::null_mut()) };
    match validate_package(package) {
        Ok(()) => HRESULT_ACCESS_DENIED,
        Err(NO_PACKAGE) => HRESULT_FILE_NOT_FOUND,
        Err(hr) => hr,
    }
}
