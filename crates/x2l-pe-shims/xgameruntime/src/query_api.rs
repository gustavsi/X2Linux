// Win64 `xgameruntime.dll!QueryApiImpl` COM-style provider dispatch table.
// GUID pairs and provider vtable slot layouts informed by Marvisak/VIctory
// (https://github.com/Marvisak/VIctory, commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-hle/src/modules/{xgameruntime,xasync,xsystem,xuser,xgame,storage,xnetworking,xpackage,display,xgamesave,xgameui}.rs)
// and xcl-dev/xcl (https://github.com/xcl-dev/xcl, commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102,
// crates/xcl/src/hle/xgameruntime/{api,gdk_table}.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Clean-room implementation for X2Linux (2026-11-28) backed directly by X2Linux's Wine-native
// `xgameruntime.dll` exports and strictly offline XStore policy.

use crate::taskqueue::{
    XAsyncBegin, XAsyncBlock, XAsyncCancel, XAsyncComplete, XAsyncGetResult, XAsyncGetResultSize,
    XAsyncGetStatus, XAsyncProvider, XAsyncRun, XAsyncSchedule, XAsyncWork, XTaskQueueCallback,
    XTaskQueueCloseHandle, XTaskQueueCreate, XTaskQueueCreateComposite, XTaskQueueDispatch,
    XTaskQueueDuplicateHandle, XTaskQueueGetCurrentProcessTaskQueue, XTaskQueueGetPort,
    XTaskQueueMonitorCallback, XTaskQueueRegisterMonitor, XTaskQueueRegisterWaiter,
    XTaskQueueSetCurrentProcessTaskQueue, XTaskQueueSubmitCallback,
    XTaskQueueSubmitDelayedCallback, XTaskQueueTerminate, XTaskQueueTerminatedCallback,
    XTaskQueueUnregisterMonitor, XTaskQueueUnregisterWaiter,
};
use crate::{
    HRESULT_E_INVALIDARG, HRESULT_E_NOTIMPL, HRESULT_E_POINTER, HRESULT_S_OK,
    XDisplayAcquireTimeoutDeferral, XDisplayCloseTimeoutDeferralHandle, XDisplayTryEnableHdrMode,
    XErrorReport, XErrorSetCallback, XErrorSetOptions, XGameGetXboxTitleId,
    XGameRuntimeIsFeatureAvailable, XGameSaveBlobInfoCallback, XGameSaveCloseContainer,
    XGameSaveCloseProvider, XGameSaveCloseUpdate, XGameSaveCreateContainer, XGameSaveCreateUpdate,
    XGameSaveEnumerateBlobInfoByName, XGameSaveGetRemainingQuotaAsync,
    XGameSaveGetRemainingQuotaResult, XGameSaveInitializeProviderAsync,
    XGameSaveInitializeProviderResult, XGameSaveReadBlobDataAsync, XGameSaveReadBlobDataResult,
    XGameSaveSubmitBlobWrite, XGameSaveSubmitUpdateAsync, XGameSaveSubmitUpdateResult,
    XGameUiShowMessageDialogAsync, XGameUiShowMessageDialogResult, XLaunchNewGame, XLaunchUri,
    XNetworkingConnectivityHint, XNetworkingGetConnectivityHint,
    XNetworkingRegisterConnectivityHintChanged, XNetworkingUnregisterConnectivityHintChanged,
    XNetworkingVerifyServerCertificate, XPackageCloseMountHandle, XPackageEnumeratePackages,
    XPackageEnumerationCallback, XPackageFindChunkAvailability,
    XPackageGetCurrentProcessPackageIdentifier, XPackageGetMountPath, XPackageGetMountPathSize,
    XPackageIsPackagedProcess, XPackageMountWithUiAsync, XPackageMountWithUiResult,
    XPersistentLocalStorageGetPath, XPersistentLocalStorageGetPathSize,
    XPersistentLocalStorageGetSpaceInfo, XPersistentLocalStoragePromptUserForSpaceAsync,
    XPersistentLocalStoragePromptUserForSpaceResult, XPersistentLocalStorageSpaceInfo,
    XSystemAnalyticsInfo, XSystemGetAnalyticsInfo, XSystemGetAppSpecificDeviceId,
    XSystemGetConsoleId, XSystemGetXboxLiveSandboxId, XThreadAssertNotTimeSensitive,
    XThreadIsTimeSensitive, XThreadSetTimeSensitive, XUserAddAsync, XUserAddResult,
    XUserCheckPrivilege, XUserCloseHandle, XUserDuplicateHandle, XUserGetAgeGroup,
    XUserGetGamertag, XUserGetId, XUserGetLocalId, XUserGetMaxUsers, XUserGetState,
    XUserGetTokenAndSignatureAsync, XUserGetTokenAndSignatureResult,
    XUserGetTokenAndSignatureResultSize, XUserLocalId, XUserRegisterForChangeEvent,
    XUserUnregisterForChangeEvent,
};
use std::ffi::{c_char, c_void};
use std::ptr;
use x2l_shim_common::record_stub;

const HRESULT_E_NOINTERFACE: i32 = -2147467262; // 0x80004002
const HRESULT_REGDB_E_CLASSNOTREG: i32 = -2147221164; // 0x80040154
const HRESULT_E_USER_NOT_FOUND: i32 = -1994108669; // 0x89235103 (E_GAMEUSER_NO_DEFAULT_USER)

/// Win64 COM `GUID` (16 bytes, little-endian fields matching `GUID`/`UUID`).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<Guid>() == 16);

impl Guid {
    pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Self {
        Self {
            data1,
            data2,
            data3,
            data4,
        }
    }

    pub fn format_canonical(&self) -> String {
        format!(
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            self.data1,
            self.data2,
            self.data3,
            self.data4[0],
            self.data4[1],
            self.data4[2],
            self.data4[3],
            self.data4[4],
            self.data4[5],
            self.data4[6],
            self.data4[7],
        )
    }
}

/// Static COM-style provider object returned by `QueryApiImpl`.
#[repr(C)]
struct ApiObject {
    vtbl: *const *const c_void,
}

// SAFETY: `ApiObject` points exclusively to immutable `'static` Win64 function-pointer vtables.
unsafe impl Sync for ApiObject {}

#[repr(transparent)]
struct Vtable<const N: usize>([*const c_void; N]);

// SAFETY: Every slot holds an immutable `'static` Win64 function pointer.
unsafe impl<const N: usize> Sync for Vtable<N> {}

type This = *mut c_void;

// ============================================================================
// Common IUnknown & Missing-Slot Telemetry Helpers
// ============================================================================

extern "win64" fn unk_query_interface(
    this: This,
    _riid: *const Guid,
    out_obj: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_obj.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: Caller supplied writable `out_obj`; static provider objects stay alive forever.
    unsafe { *out_obj = this };
    HRESULT_S_OK
}

extern "win64" fn unk_add_ref(_this: This) -> u32 {
    1
}

extern "win64" fn unk_release(_this: This) -> u32 {
    1
}

extern "win64" fn missing_slot<const TAG: u32, const SLOT: u32>(_this: This) -> i32 {
    let prov = match TAG {
        1 => "XAsyncProvider",
        2 => "XSystemProvider",
        3 => "XUserProvider",
        4 => "XPackageProvider",
        5 => "XGameSaveProvider",
        6 => "XNetworkingProvider",
        7 => "XGameUiProvider",
        8 => "XDisplayProvider",
        _ => "QueryApiProvider",
    };
    let sym = format!("{prov}[{SLOT}]");
    record_stub("xgameruntime.dll", &sym, HRESULT_E_NOTIMPL as u64) as i32
}

// ============================================================================
// Provider 1: XAsync / XTaskQueue / XThread (CLSID = IID = 073b7dcb-...)
// ============================================================================

const CLSID_XASYNC: Guid = Guid::new(
    0x073b_7dcb,
    0x1fcf,
    0x4030,
    [0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28],
);

extern "win64" fn xa_get_status(_: This, ab: *mut XAsyncBlock, wait: bool) -> i32 {
    XAsyncGetStatus(ab, wait)
}
extern "win64" fn xa_get_result_size(_: This, ab: *mut XAsyncBlock, out: *mut usize) -> i32 {
    XAsyncGetResultSize(ab, out)
}
extern "win64" fn xa_cancel(_: This, ab: *mut XAsyncBlock) {
    XAsyncCancel(ab)
}
extern "win64" fn xa_run(_: This, ab: *mut XAsyncBlock, work: Option<XAsyncWork>) -> i32 {
    XAsyncRun(ab, work)
}
extern "win64" fn xa_begin(
    _: This,
    ab: *mut XAsyncBlock,
    ctx: *mut c_void,
    id: *const c_void,
    name: *const c_char,
    prov: Option<XAsyncProvider>,
) -> i32 {
    XAsyncBegin(ab, ctx, id, name, prov)
}
extern "win64" fn xa_schedule(_: This, ab: *mut XAsyncBlock, delay_ms: u32) -> i32 {
    XAsyncSchedule(ab, delay_ms)
}
extern "win64" fn xa_complete(_: This, ab: *mut XAsyncBlock, status: i32, sz: usize) {
    XAsyncComplete(ab, status, sz)
}
extern "win64" fn xa_get_result(
    _: This,
    ab: *mut XAsyncBlock,
    id: *const c_void,
    cap: usize,
    buf: *mut c_void,
    used: *mut usize,
) -> i32 {
    XAsyncGetResult(ab, id, cap, buf, used)
}
extern "win64" fn xtq_create(_: This, wm: u32, cm: u32, out: *mut *mut c_void) -> i32 {
    XTaskQueueCreate(wm, cm, out)
}
extern "win64" fn xtq_create_composite(
    _: This,
    wp: *mut c_void,
    cp: *mut c_void,
    out: *mut *mut c_void,
) -> i32 {
    XTaskQueueCreateComposite(wp, cp, out)
}
extern "win64" fn xtq_get_port(_: This, q: *mut c_void, p: u32, out: *mut *mut c_void) -> i32 {
    XTaskQueueGetPort(q, p, out)
}
extern "win64" fn xtq_dup(_: This, q: *mut c_void, out: *mut *mut c_void) -> i32 {
    XTaskQueueDuplicateHandle(q, out)
}
extern "win64" fn xtq_dispatch(_: This, q: *mut c_void, p: u32, timeout_ms: u32) -> bool {
    XTaskQueueDispatch(q, p, timeout_ms)
}
extern "win64" fn xtq_close(_: This, q: *mut c_void) {
    XTaskQueueCloseHandle(q)
}
extern "win64" fn xtq_submit(
    _: This,
    q: *mut c_void,
    p: u32,
    ctx: *mut c_void,
    cb: Option<XTaskQueueCallback>,
) -> i32 {
    XTaskQueueSubmitCallback(q, p, ctx, cb)
}
extern "win64" fn xtq_submit_delayed(
    _: This,
    q: *mut c_void,
    p: u32,
    delay_ms: u32,
    ctx: *mut c_void,
    cb: Option<XTaskQueueCallback>,
) -> i32 {
    XTaskQueueSubmitDelayedCallback(q, p, delay_ms, ctx, cb)
}
extern "win64" fn xtq_reg_waiter(
    _: This,
    q: *mut c_void,
    p: u32,
    wh: *mut c_void,
    ctx: *mut c_void,
    cb: Option<XTaskQueueCallback>,
    tok: *mut u64,
) -> i32 {
    XTaskQueueRegisterWaiter(q, p, wh, ctx, cb, tok)
}
extern "win64" fn xtq_unreg_waiter(_: This, q: *mut c_void, tok: u64) {
    XTaskQueueUnregisterWaiter(q, tok)
}
extern "win64" fn xtq_terminate(
    _: This,
    q: *mut c_void,
    wait: bool,
    ctx: *mut c_void,
    cb: Option<XTaskQueueTerminatedCallback>,
) -> i32 {
    XTaskQueueTerminate(q, wait, ctx, cb)
}
extern "win64" fn xtq_reg_monitor(
    _: This,
    q: *mut c_void,
    ctx: *mut c_void,
    cb: Option<XTaskQueueMonitorCallback>,
    tok: *mut u64,
) -> i32 {
    XTaskQueueRegisterMonitor(q, ctx, cb, tok)
}
extern "win64" fn xtq_unreg_monitor(_: This, q: *mut c_void, tok: u64) {
    XTaskQueueUnregisterMonitor(q, tok)
}
extern "win64" fn xtq_get_proc(_: This, out: *mut *mut c_void) -> bool {
    XTaskQueueGetCurrentProcessTaskQueue(out)
}
extern "win64" fn xtq_set_proc(_: This, q: *mut c_void) {
    XTaskQueueSetCurrentProcessTaskQueue(q)
}
extern "win64" fn xthr_is_ts(_: This) -> bool {
    XThreadIsTimeSensitive()
}
extern "win64" fn xthr_set_ts(_: This, ts: bool) -> i32 {
    XThreadSetTimeSensitive(ts)
}
extern "win64" fn xthr_assert_not_ts(_: This) {
    XThreadAssertNotTimeSensitive()
}

static XASYNC_VTABLE: Vtable<31> = Vtable([
    unk_query_interface as *const c_void, // 0
    unk_add_ref as *const c_void,         // 1
    unk_release as *const c_void,         // 2
    xa_get_status as *const c_void,       // 3
    xa_get_result_size as *const c_void,  // 4
    xa_cancel as *const c_void,           // 5
    xa_run as *const c_void,              // 6
    xa_begin as *const c_void,            // 7
    missing_slot::<1, 8> as *const c_void, // 8 (XAsyncBeginAlloc)
    xa_schedule as *const c_void,         // 9
    xa_complete as *const c_void,         // 10
    xa_get_result as *const c_void,       // 11
    xtq_create as *const c_void,          // 12
    xtq_create_composite as *const c_void, // 13
    xtq_get_port as *const c_void,        // 14
    xtq_dup as *const c_void,             // 15
    xtq_dispatch as *const c_void,        // 16
    xtq_close as *const c_void,           // 17
    xtq_submit as *const c_void,          // 18
    xtq_submit_delayed as *const c_void,  // 19
    xtq_reg_waiter as *const c_void,      // 20
    xtq_unreg_waiter as *const c_void,    // 21
    xtq_terminate as *const c_void,       // 22
    xtq_reg_monitor as *const c_void,     // 23
    xtq_unreg_monitor as *const c_void,   // 24
    xtq_get_proc as *const c_void,        // 25
    xtq_set_proc as *const c_void,        // 26
    xthr_is_ts as *const c_void,          // 27
    xthr_set_ts as *const c_void,         // 28
    xthr_assert_not_ts as *const c_void,  // 29
    missing_slot::<1, 30> as *const c_void,
]);

static XASYNC_OBJ: ApiObject = ApiObject {
    vtbl: XASYNC_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 2: XSystem & XSystemAnalytics
// ============================================================================

const CLSID_XSYSTEM: Guid = Guid::new(
    0xe349_bd1a,
    0xfc20,
    0x4e40,
    [0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f],
);
const IID_XSYSTEM_V1: Guid = Guid::new(
    0x6fd7_1f09,
    0x7513,
    0x49f0,
    [0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52],
);
const IID_XSYSTEM_V2: Guid = Guid::new(
    0x1861_cf2e,
    0xe18b,
    0x4834,
    [0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf],
);
const IID_XSYSTEM_ALIAS: Guid = Guid::new(
    0x67ce_4bfc,
    0xb1d1,
    0x4ac7,
    [0xbc, 0x3a, 0xcb, 0x92, 0x19, 0xa9, 0x7a, 0x85],
);
const CLSID_XSYSTEM_ANALYTICS: Guid = Guid::new(
    0xb884_675d,
    0xb738,
    0x4a9c,
    [0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b],
);

extern "win64" fn xsys_console_id(_: This, cap: usize, buf: *mut u8, used: *mut usize) -> i32 {
    XSystemGetConsoleId(cap, buf, used)
}
extern "win64" fn xsys_sandbox_id(_: This, cap: usize, buf: *mut u8, used: *mut usize) -> i32 {
    XSystemGetXboxLiveSandboxId(cap, buf, used)
}
extern "win64" fn xsys_app_dev_id(_: This, cap: usize, buf: *mut u8, used: *mut usize) -> i32 {
    XSystemGetAppSpecificDeviceId(cap, buf, used)
}
extern "win64" fn xsys_analytics_info(
    _: This,
    out_info: *mut XSystemAnalyticsInfo,
) -> *mut XSystemAnalyticsInfo {
    if !out_info.is_null() {
        let info = XSystemGetAnalyticsInfo();
        // SAFETY: Win64 hidden struct-return buffer `out_info` is writable for `XSystemAnalyticsInfo`.
        unsafe { out_info.write(info) };
    }
    out_info
}

static XSYSTEM_VTABLE: Vtable<6> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xsys_console_id as *const c_void,
    xsys_sandbox_id as *const c_void,
    xsys_app_dev_id as *const c_void,
]);
static XSYSTEM_OBJ: ApiObject = ApiObject {
    vtbl: XSYSTEM_VTABLE.0.as_ptr(),
};

static XSYSTEM_ANALYTICS_VTABLE: Vtable<4> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xsys_analytics_info as *const c_void,
]);
static XSYSTEM_ANALYTICS_OBJ: ApiObject = ApiObject {
    vtbl: XSYSTEM_ANALYTICS_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 3: XGame, XGameRuntimeFeature, XGameInvite, XGameEvent, XError
// ============================================================================

const CLSID_XGAME: Guid = Guid::new(
    0x973a_344e,
    0x24bf,
    0x4d0f,
    [0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29],
);
const IID_XGAME: Guid = Guid::new(
    0x2549_f142,
    0x6419,
    0x4a06,
    [0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34],
);
const IID_XGAME_ALIAS: Guid = Guid::new(
    0x5084_9859,
    0x0ad8,
    0x4f81,
    [0x80, 0xe4, 0x5b, 0xc7, 0x86, 0x26, 0xf8, 0x52],
);
const CLSID_XGAME_FEATURE: Guid = Guid::new(
    0x8836_fe87,
    0xedb9,
    0x4fe3,
    [0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40],
);
const CLSID_XGAME_INVITE: Guid = Guid::new(
    0x0651_aae2,
    0x4012,
    0x4077,
    [0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c],
);
const IID_XGAME_INVITE_V2: Guid = Guid::new(
    0x014d_1cc3,
    0xbcfe,
    0x41ff,
    [0xb2, 0xf0, 0xe1, 0xef, 0x07, 0x15, 0x58, 0x28],
);
const CLSID_XGAME_EVENT: Guid = Guid::new(
    0xbbfb_dcc7,
    0xbfe7,
    0x409b,
    [0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d],
);
const CLSID_XERROR: Guid = Guid::new(
    0x8ca4_67f7,
    0x22e8,
    0x4096,
    [0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8],
);

extern "win64" fn xgame_title_id(_: This, out_id: *mut u32) -> i32 {
    XGameGetXboxTitleId(out_id)
}
extern "win64" fn xgame_launch_new(
    _: This,
    exe: *const c_char,
    args: *const c_char,
    user: *mut c_void,
) {
    XLaunchNewGame(exe, args, user)
}
extern "win64" fn xgame_restart_on_crash(_: This, _args: *const c_char, _reserved: u32) -> i32 {
    HRESULT_S_OK
}
extern "win64" fn xgame_is_feature(_: This, feature: u32) -> bool {
    XGameRuntimeIsFeatureAvailable(feature)
}
extern "win64" fn xgame_invite_reg(
    _: This,
    q: *mut c_void,
    ctx: *mut c_void,
    cb: *const c_void,
    tok: *mut u64,
) -> i32 {
    XUserRegisterForChangeEvent(q, ctx, cb, tok)
}
extern "win64" fn xgame_invite_unreg(_: This, tok: u64, wait: bool) -> bool {
    XUserUnregisterForChangeEvent(tok, wait)
}
extern "win64" fn xgame_event_write(_: This, _name: *const c_char) -> i32 {
    HRESULT_S_OK
}
extern "win64" fn xerr_report(_: This, hr: i32, msg: *const c_char) -> bool {
    XErrorReport(hr, msg);
    true
}
extern "win64" fn xerr_set_cb(_: This, cb: *const c_void, ctx: *mut c_void) {
    XErrorSetCallback(cb, ctx)
}
extern "win64" fn xerr_set_opts(_: This, dbg: u32, non_dbg: usize) {
    XErrorSetOptions(dbg, non_dbg)
}

static XGAME_VTABLE: Vtable<6> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xgame_title_id as *const c_void,
    xgame_launch_new as *const c_void,
    xgame_restart_on_crash as *const c_void,
]);
static XGAME_OBJ: ApiObject = ApiObject {
    vtbl: XGAME_VTABLE.0.as_ptr(),
};

static XGAME_FEATURE_VTABLE: Vtable<4> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xgame_is_feature as *const c_void,
]);
static XGAME_FEATURE_OBJ: ApiObject = ApiObject {
    vtbl: XGAME_FEATURE_VTABLE.0.as_ptr(),
};

static XGAME_INVITE_VTABLE: Vtable<5> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xgame_invite_reg as *const c_void,
    xgame_invite_unreg as *const c_void,
]);
static XGAME_INVITE_OBJ: ApiObject = ApiObject {
    vtbl: XGAME_INVITE_VTABLE.0.as_ptr(),
};

static XGAME_EVENT_VTABLE: Vtable<4> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xgame_event_write as *const c_void,
]);
static XGAME_EVENT_OBJ: ApiObject = ApiObject {
    vtbl: XGAME_EVENT_VTABLE.0.as_ptr(),
};

static XERROR_VTABLE: Vtable<6> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xerr_report as *const c_void,
    xerr_set_cb as *const c_void,
    xerr_set_opts as *const c_void,
]);
static XERROR_OBJ: ApiObject = ApiObject {
    vtbl: XERROR_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 4: XPersistentLocalStorage
// ============================================================================

const CLSID_PLS: Guid = Guid::new(
    0xf4fa_f4d4,
    0x2d04,
    0x4fce,
    [0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84],
);
const IID_PLS_V1: Guid = Guid::new(
    0x41a4_e10c,
    0x5a7e,
    0x41d9,
    [0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6],
);
const IID_PLS_V2: Guid = Guid::new(
    0xd294_11df,
    0x0794,
    0x4553,
    [0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d],
);

extern "win64" fn pls_path_size(_: This, out_sz: *mut usize) -> i32 {
    XPersistentLocalStorageGetPathSize(out_sz)
}
extern "win64" fn pls_path(_: This, cap: usize, buf: *mut u8, used: *mut usize) -> i32 {
    XPersistentLocalStorageGetPath(cap, buf, used)
}
extern "win64" fn pls_space_info(_: This, out: *mut XPersistentLocalStorageSpaceInfo) -> i32 {
    XPersistentLocalStorageGetSpaceInfo(out)
}
extern "win64" fn pls_prompt_async(_: This, bytes: u64, ab: *mut XAsyncBlock) -> i32 {
    XPersistentLocalStoragePromptUserForSpaceAsync(bytes, ab)
}
extern "win64" fn pls_prompt_result(_: This, ab: *mut XAsyncBlock) -> i32 {
    XPersistentLocalStoragePromptUserForSpaceResult(ab)
}

static PLS_VTABLE: Vtable<8> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    pls_path_size as *const c_void,
    pls_path as *const c_void,
    pls_space_info as *const c_void,
    pls_prompt_async as *const c_void,
    pls_prompt_result as *const c_void,
]);
static PLS_OBJ: ApiObject = ApiObject {
    vtbl: PLS_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 5: XUser & XUser Devices
// ============================================================================

const CLSID_XUSER: Guid = Guid::new(
    0x01ac_d177,
    0x91f9,
    0x4763,
    [0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0],
);
const IID_XUSER_MAIN: Guid = Guid::new(
    0xeb9b_f948,
    0x18dc,
    0x4d82,
    [0xbb, 0xcc, 0x40, 0xe0, 0xa8, 0x09, 0xc4, 0xc0],
);
const IID_XUSER_GAMERTAG: Guid = Guid::new(
    0xcef4_fac0,
    0x7676,
    0x4a94,
    [0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74],
);
const CLSID_XUSER_DEVICES: Guid = Guid::new(
    0x7d82_4997,
    0x10dc,
    0x45ab,
    [0x86, 0xb7, 0x27, 0x37, 0x76, 0x7c, 0x0b, 0xf1],
);
const IID_XUSER_DEVICES: Guid = Guid::new(
    0x0cc6_a956,
    0xe7e1,
    0x4fdf,
    [0x93, 0x41, 0x9d, 0x5d, 0xa6, 0x49, 0xeb, 0xc8],
);

extern "win64" fn xu_dup(_: This, u: *mut c_void, out: *mut *mut c_void) -> i32 {
    XUserDuplicateHandle(u, out)
}
extern "win64" fn xu_close(_: This, u: *mut c_void) {
    XUserCloseHandle(u)
}
extern "win64" fn xu_compare(_: This, a: *mut c_void, b: *mut c_void) -> i32 {
    if a == b { 0 } else { -1 }
}
extern "win64" fn xu_max_users(_: This, out: *mut u32) -> i32 {
    XUserGetMaxUsers(out)
}
extern "win64" fn xu_add_async(_: This, opts: u32, ab: *mut XAsyncBlock) -> i32 {
    XUserAddAsync(opts, ab)
}
extern "win64" fn xu_add_result(_: This, ab: *mut XAsyncBlock, out: *mut *mut c_void) -> i32 {
    XUserAddResult(ab, out)
}
extern "win64" fn xu_get_local_id(_: This, u: *mut c_void, out: *mut XUserLocalId) -> i32 {
    XUserGetLocalId(u, out)
}
extern "win64" fn xu_find_by_local_id(_: This, local_id: u64, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    if local_id == 1 {
        // SAFETY: `out` was verified non-null.
        unsafe { *out = crate::LOCAL_USER_HANDLE as *mut c_void };
        HRESULT_S_OK
    } else {
        // SAFETY: `out` was verified non-null.
        unsafe { *out = ptr::null_mut() };
        HRESULT_E_USER_NOT_FOUND
    }
}
extern "win64" fn xu_get_id(_: This, u: *mut c_void, out: *mut u64) -> i32 {
    XUserGetId(u, out)
}
extern "win64" fn xu_find_by_id(_: This, xuid: u64, out: *mut *mut c_void) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    if xuid == 0x0009_0000_0000_0001 {
        // SAFETY: `out` was verified non-null.
        unsafe { *out = crate::LOCAL_USER_HANDLE as *mut c_void };
        HRESULT_S_OK
    } else {
        // SAFETY: `out` was verified non-null.
        unsafe { *out = ptr::null_mut() };
        HRESULT_E_USER_NOT_FOUND
    }
}
extern "win64" fn xu_is_guest(_: This, _u: *mut c_void, out: *mut bool) -> i32 {
    if out.is_null() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: `out` was verified non-null.
    unsafe { *out = false };
    HRESULT_S_OK
}
extern "win64" fn xu_get_state(_: This, u: *mut c_void, out: *mut u32) -> i32 {
    XUserGetState(u, out)
}
extern "win64" fn xu_get_gamertag(
    _: This,
    u: *mut c_void,
    comp: u32,
    cap: usize,
    buf: *mut u8,
    used: *mut usize,
) -> i32 {
    XUserGetGamertag(u, comp, cap, buf, used)
}
extern "win64" fn xu_get_age(_: This, u: *mut c_void, out: *mut u32) -> i32 {
    XUserGetAgeGroup(u, out)
}
extern "win64" fn xu_check_priv(
    _: This,
    u: *mut c_void,
    opts: u32,
    priv_id: u32,
    has: *mut bool,
    reason: *mut u32,
) -> i32 {
    XUserCheckPrivilege(u, opts, priv_id, has, reason)
}
#[allow(clippy::too_many_arguments)]
extern "win64" fn xu_token_async(
    _: This,
    u: *mut c_void,
    opts: u32,
    m: *const c_char,
    url: *const c_char,
    hc: usize,
    hs: *const c_void,
    bs: usize,
    body: *const c_void,
    ab: *mut XAsyncBlock,
) -> i32 {
    XUserGetTokenAndSignatureAsync(u, opts, m, url, hc, hs, bs, body, ab)
}
extern "win64" fn xu_token_size(_: This, ab: *mut XAsyncBlock, out: *mut usize) -> i32 {
    XUserGetTokenAndSignatureResultSize(ab, out)
}
extern "win64" fn xu_token_result(
    _: This,
    ab: *mut XAsyncBlock,
    bs: usize,
    buf: *mut c_void,
    data: *mut *mut c_void,
    used: *mut usize,
) -> i32 {
    XUserGetTokenAndSignatureResult(ab, bs, buf, data, used)
}
extern "win64" fn xu_reg_change(
    _: This,
    q: *mut c_void,
    ctx: *mut c_void,
    cb: *const c_void,
    tok: *mut u64,
) -> i32 {
    XUserRegisterForChangeEvent(q, ctx, cb, tok)
}
extern "win64" fn xu_unreg_change(_: This, tok: u64, wait: bool) -> bool {
    XUserUnregisterForChangeEvent(tok, wait)
}
extern "win64" fn xu_find_for_device(_: This, _dev: *const c_void, out: *mut *mut c_void) -> i32 {
    if !out.is_null() {
        // SAFETY: Clear output handle when no device association exists.
        unsafe { *out = ptr::null_mut() };
    }
    HRESULT_E_USER_NOT_FOUND
}

static XUSER_VTABLE: Vtable<35> = Vtable([
    unk_query_interface as *const c_void,   // 0
    unk_add_ref as *const c_void,           // 1
    unk_release as *const c_void,           // 2
    xu_dup as *const c_void,                // 3
    xu_close as *const c_void,              // 4
    xu_compare as *const c_void,            // 5
    xu_max_users as *const c_void,          // 6
    xu_add_async as *const c_void,          // 7
    xu_add_result as *const c_void,         // 8
    xu_get_local_id as *const c_void,       // 9
    xu_find_by_local_id as *const c_void,   // 10
    xu_get_id as *const c_void,             // 11
    xu_find_by_id as *const c_void,         // 12
    xu_is_guest as *const c_void,           // 13
    xu_get_state as *const c_void,          // 14
    xu_get_gamertag as *const c_void,       // 15
    missing_slot::<3, 16> as *const c_void, // 16
    missing_slot::<3, 17> as *const c_void, // 17
    missing_slot::<3, 18> as *const c_void, // 18
    xu_get_age as *const c_void,            // 19
    xu_check_priv as *const c_void,         // 20
    missing_slot::<3, 21> as *const c_void, // 21
    missing_slot::<3, 22> as *const c_void, // 22
    xu_token_async as *const c_void,        // 23
    xu_token_size as *const c_void,         // 24
    xu_token_result as *const c_void,       // 25
    missing_slot::<3, 26> as *const c_void, // 26
    missing_slot::<3, 27> as *const c_void, // 27
    missing_slot::<3, 28> as *const c_void, // 28
    missing_slot::<3, 29> as *const c_void, // 29
    missing_slot::<3, 30> as *const c_void, // 30
    missing_slot::<3, 31> as *const c_void, // 31
    missing_slot::<3, 32> as *const c_void, // 32
    xu_reg_change as *const c_void,         // 33
    xu_unreg_change as *const c_void,       // 34
]);
static XUSER_OBJ: ApiObject = ApiObject {
    vtbl: XUSER_VTABLE.0.as_ptr(),
};

static XUSER_GAMERTAG_VTABLE: Vtable<4> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xu_get_gamertag as *const c_void,
]);
static XUSER_GAMERTAG_OBJ: ApiObject = ApiObject {
    vtbl: XUSER_GAMERTAG_VTABLE.0.as_ptr(),
};

static XUSER_DEVICES_VTABLE: Vtable<6> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xu_find_for_device as *const c_void,
    xu_reg_change as *const c_void,
    xu_unreg_change as *const c_void,
]);
static XUSER_DEVICES_OBJ: ApiObject = ApiObject {
    vtbl: XUSER_DEVICES_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 6: XPackage
// ============================================================================

const CLSID_XPACKAGE: Guid = Guid::new(
    0xaf40_6016,
    0xe850,
    0x4aa8,
    [0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e],
);
const IID_XPACKAGE: Guid = Guid::new(
    0x0b9d_556e,
    0xab70,
    0x43f4,
    [0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40],
);

extern "win64" fn xpkg_cur_id(_: This, sz: usize, buf: *mut u8) -> i32 {
    XPackageGetCurrentProcessPackageIdentifier(sz, buf)
}
extern "win64" fn xpkg_is_packaged(_: This) -> bool {
    XPackageIsPackagedProcess()
}
extern "win64" fn xpkg_find_avail(
    _: This,
    pkg: *const c_char,
    cnt: u32,
    sels: *const c_void,
    out: *mut u32,
) -> i32 {
    XPackageFindChunkAvailability(pkg, cnt, sels, out)
}
extern "win64" fn xpkg_mount_path_size(_: This, m: *mut c_void, out: *mut usize) -> i32 {
    XPackageGetMountPathSize(m, out)
}
extern "win64" fn xpkg_mount_path(_: This, m: *mut c_void, cap: usize, buf: *mut u8) -> i32 {
    XPackageGetMountPath(m, cap, buf)
}
extern "win64" fn xpkg_close_mount(_: This, m: *mut c_void) {
    XPackageCloseMountHandle(m)
}
extern "win64" fn xpkg_mount_async(_: This, pkg: *const c_char, ab: *mut XAsyncBlock) -> i32 {
    XPackageMountWithUiAsync(pkg, ab)
}
extern "win64" fn xpkg_mount_result(_: This, ab: *mut XAsyncBlock, out: *mut *mut c_void) -> i32 {
    XPackageMountWithUiResult(ab, out)
}
extern "win64" fn xpkg_enum(
    _: This,
    kind: u32,
    scope: u32,
    ctx: *mut c_void,
    cb: Option<XPackageEnumerationCallback>,
) -> i32 {
    XPackageEnumeratePackages(kind, scope, ctx, cb)
}

static XPACKAGE_VTABLE: Vtable<40> = Vtable([
    unk_query_interface as *const c_void,   // 0
    unk_add_ref as *const c_void,           // 1
    unk_release as *const c_void,           // 2
    xpkg_cur_id as *const c_void,           // 3
    xpkg_is_packaged as *const c_void,      // 4
    missing_slot::<4, 5> as *const c_void,  // 5
    missing_slot::<4, 6> as *const c_void,  // 6
    missing_slot::<4, 7> as *const c_void,  // 7
    missing_slot::<4, 8> as *const c_void,  // 8
    missing_slot::<4, 9> as *const c_void,  // 9
    missing_slot::<4, 10> as *const c_void, // 10
    missing_slot::<4, 11> as *const c_void, // 11
    xpkg_find_avail as *const c_void,       // 12
    missing_slot::<4, 13> as *const c_void, // 13
    missing_slot::<4, 14> as *const c_void, // 14
    missing_slot::<4, 15> as *const c_void, // 15
    missing_slot::<4, 16> as *const c_void, // 16
    missing_slot::<4, 17> as *const c_void, // 17
    missing_slot::<4, 18> as *const c_void, // 18
    missing_slot::<4, 19> as *const c_void, // 19
    missing_slot::<4, 20> as *const c_void, // 20
    missing_slot::<4, 21> as *const c_void, // 21
    missing_slot::<4, 22> as *const c_void, // 22
    missing_slot::<4, 23> as *const c_void, // 23
    xpkg_mount_path_size as *const c_void,  // 24
    xpkg_mount_path as *const c_void,       // 25
    xpkg_close_mount as *const c_void,      // 26
    missing_slot::<4, 27> as *const c_void, // 27
    missing_slot::<4, 28> as *const c_void, // 28
    missing_slot::<4, 29> as *const c_void, // 29
    missing_slot::<4, 30> as *const c_void, // 30
    missing_slot::<4, 31> as *const c_void, // 31
    missing_slot::<4, 32> as *const c_void, // 32
    missing_slot::<4, 33> as *const c_void, // 33
    missing_slot::<4, 34> as *const c_void, // 34
    missing_slot::<4, 35> as *const c_void, // 35
    missing_slot::<4, 36> as *const c_void, // 36
    xpkg_mount_async as *const c_void,      // 37
    xpkg_mount_result as *const c_void,     // 38
    xpkg_enum as *const c_void,             // 39
]);
static XPACKAGE_OBJ: ApiObject = ApiObject {
    vtbl: XPACKAGE_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 7: XGameSave
// ============================================================================

const CLSID_XGAMESAVE: Guid = Guid::new(
    0x704c_3f58,
    0xe629,
    0x4cc2,
    [0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2],
);
const IID_XGAMESAVE_V2: Guid = Guid::new(
    0x1bff_f3af,
    0xf14a,
    0x40a3,
    [0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9],
);

extern "win64" fn xgs_init_async(
    _: This,
    u: *mut c_void,
    scid: *const c_char,
    sync: bool,
    ab: *mut XAsyncBlock,
) -> i32 {
    XGameSaveInitializeProviderAsync(u, scid, sync, ab)
}
extern "win64" fn xgs_init_result(_: This, ab: *mut XAsyncBlock, out: *mut *mut c_void) -> i32 {
    XGameSaveInitializeProviderResult(ab, out)
}
extern "win64" fn xgs_close_prov(_: This, p: *mut c_void) {
    XGameSaveCloseProvider(p)
}
extern "win64" fn xgs_quota_async(_: This, p: *mut c_void, ab: *mut XAsyncBlock) -> i32 {
    XGameSaveGetRemainingQuotaAsync(p, ab)
}
extern "win64" fn xgs_quota_result(_: This, ab: *mut XAsyncBlock, out: *mut i64) -> i32 {
    XGameSaveGetRemainingQuotaResult(ab, out)
}
extern "win64" fn xgs_create_container(
    _: This,
    p: *mut c_void,
    name: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    XGameSaveCreateContainer(p, name, out)
}
extern "win64" fn xgs_close_container(_: This, c: *mut c_void) {
    XGameSaveCloseContainer(c)
}
extern "win64" fn xgs_enum_blobs(
    _: This,
    c: *mut c_void,
    prefix: *const c_char,
    ctx: *mut c_void,
    cb: Option<XGameSaveBlobInfoCallback>,
) -> i32 {
    XGameSaveEnumerateBlobInfoByName(c, prefix, ctx, cb)
}
extern "win64" fn xgs_read_async(
    _: This,
    c: *mut c_void,
    names: *const *const c_char,
    cnt: u32,
    ab: *mut XAsyncBlock,
) -> i32 {
    XGameSaveReadBlobDataAsync(c, names, cnt, ab)
}
extern "win64" fn xgs_read_result(
    _: This,
    ab: *mut XAsyncBlock,
    bs: usize,
    blobs: *mut crate::XGameSaveBlob,
    cnt: *mut u32,
) -> i32 {
    XGameSaveReadBlobDataResult(ab, bs, blobs, cnt)
}
extern "win64" fn xgs_create_update(
    _: This,
    c: *mut c_void,
    name: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    XGameSaveCreateUpdate(c, name, out)
}
extern "win64" fn xgs_close_update(_: This, u: *mut c_void) {
    XGameSaveCloseUpdate(u)
}
extern "win64" fn xgs_write_blob(
    _: This,
    u: *mut c_void,
    name: *const c_char,
    data: *const u8,
    len: usize,
) -> i32 {
    XGameSaveSubmitBlobWrite(u, name, data, len)
}
extern "win64" fn xgs_submit_async(_: This, u: *mut c_void, ab: *mut XAsyncBlock) -> i32 {
    XGameSaveSubmitUpdateAsync(u, ab)
}
extern "win64" fn xgs_submit_result(_: This, ab: *mut XAsyncBlock) -> i32 {
    XGameSaveSubmitUpdateResult(ab)
}

static XGAMESAVE_VTABLE: Vtable<30> = Vtable([
    unk_query_interface as *const c_void,   // 0
    unk_add_ref as *const c_void,           // 1
    unk_release as *const c_void,           // 2
    missing_slot::<5, 3> as *const c_void,  // 3
    xgs_init_async as *const c_void,        // 4
    xgs_init_result as *const c_void,       // 5
    xgs_close_prov as *const c_void,        // 6
    missing_slot::<5, 7> as *const c_void,  // 7
    xgs_quota_async as *const c_void,       // 8
    xgs_quota_result as *const c_void,      // 9
    missing_slot::<5, 10> as *const c_void, // 10
    missing_slot::<5, 11> as *const c_void, // 11 (DeleteContainerAsync)
    missing_slot::<5, 12> as *const c_void, // 12 (DeleteContainerResult)
    missing_slot::<5, 13> as *const c_void, // 13 (GetContainerInfo)
    missing_slot::<5, 14> as *const c_void, // 14 (EnumerateContainerInfo)
    missing_slot::<5, 15> as *const c_void, // 15
    xgs_create_container as *const c_void,  // 16
    xgs_close_container as *const c_void,   // 17
    missing_slot::<5, 18> as *const c_void, // 18
    xgs_enum_blobs as *const c_void,        // 19
    missing_slot::<5, 20> as *const c_void, // 20
    xgs_read_async as *const c_void,        // 21
    xgs_read_result as *const c_void,       // 22
    xgs_create_update as *const c_void,     // 23
    xgs_close_update as *const c_void,      // 24
    xgs_write_blob as *const c_void,        // 25
    missing_slot::<5, 26> as *const c_void, // 26 (SubmitBlobDelete)
    missing_slot::<5, 27> as *const c_void, // 27
    xgs_submit_async as *const c_void,      // 28
    xgs_submit_result as *const c_void,     // 29
]);
static XGAMESAVE_OBJ: ApiObject = ApiObject {
    vtbl: XGAMESAVE_VTABLE.0.as_ptr(),
};

// ============================================================================
// Provider 8: XNetworking, XDisplay, & XGameUi (Offline)
// ============================================================================

const CLSID_XNETWORKING: Guid = Guid::new(
    0x37e5_6907,
    0x2f10,
    0x41e8,
    [0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a],
);
const IID_XNETWORKING_V2: Guid = Guid::new(
    0xbf23_46b2,
    0x39af,
    0x4658,
    [0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3],
);
const CLSID_XDISPLAY_HDR: Guid = Guid::new(
    0x03f0_fe74,
    0xfdd9,
    0x4e5c,
    [0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5],
);
const IID_XDISPLAY_HDR: Guid = Guid::new(
    0x35f0_7670,
    0x706e,
    0x4bfb,
    [0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3],
);
const CLSID_XDISPLAY_SHELL: Guid = Guid::new(
    0x1b33_9674,
    0x328d,
    0x4283,
    [0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39],
);
const CLSID_XGAMEUI: Guid = Guid::new(
    0xdfcd_4649,
    0x4ff8,
    0x4043,
    [0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0],
);
const IID_XGAMEUI: Guid = Guid::new(
    0xeaf6_69df,
    0x5542,
    0x4590,
    [0x99, 0xa3, 0x8d, 0xc0, 0x61, 0xf8, 0x37, 0xcc],
);
const CLSID_XSTORE: Guid = Guid::new(
    0x0dd1_12ac,
    0x7c24,
    0x448c,
    [0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c],
);

extern "win64" fn xnet_verify_cert(_: This, req: *mut c_void, info: *const c_void) -> i32 {
    XNetworkingVerifyServerCertificate(req, info)
}
extern "win64" fn xnet_hint(_: This, out: *mut XNetworkingConnectivityHint) -> i32 {
    XNetworkingGetConnectivityHint(out)
}
extern "win64" fn xnet_reg(
    _: This,
    q: *mut c_void,
    ctx: *mut c_void,
    cb: *const c_void,
    tok: *mut u64,
) -> i32 {
    XNetworkingRegisterConnectivityHintChanged(q, ctx, cb, tok)
}
extern "win64" fn xnet_unreg(_: This, tok: u64, wait: bool) -> bool {
    XNetworkingUnregisterConnectivityHintChanged(tok, wait)
}

static XNETWORKING_VTABLE: Vtable<18> = Vtable([
    unk_query_interface as *const c_void,   // 0
    unk_add_ref as *const c_void,           // 1
    unk_release as *const c_void,           // 2
    missing_slot::<6, 3> as *const c_void,  // 3
    missing_slot::<6, 4> as *const c_void,  // 4
    missing_slot::<6, 5> as *const c_void,  // 5
    xnet_reg as *const c_void,              // 6
    xnet_unreg as *const c_void,            // 7
    missing_slot::<6, 8> as *const c_void,  // 8
    missing_slot::<6, 9> as *const c_void,  // 9
    missing_slot::<6, 10> as *const c_void, // 10
    missing_slot::<6, 11> as *const c_void, // 11
    missing_slot::<6, 12> as *const c_void, // 12
    missing_slot::<6, 13> as *const c_void, // 13
    xnet_verify_cert as *const c_void,      // 14
    xnet_hint as *const c_void,             // 15
    xnet_reg as *const c_void,              // 16
    xnet_unreg as *const c_void,            // 17
]);
static XNETWORKING_OBJ: ApiObject = ApiObject {
    vtbl: XNETWORKING_VTABLE.0.as_ptr(),
};

extern "win64" fn xdisp_try_hdr(_: This, opts: *const c_void, out: *mut c_void) -> u32 {
    XDisplayTryEnableHdrMode(opts, out)
}
extern "win64" fn xdisp_launch_uri(_: This, user: *mut c_void, uri: *const c_char) -> i32 {
    XLaunchUri(user, uri)
}
extern "win64" fn xdisp_acq_defer(_: This, out: *mut *mut c_void) -> i32 {
    XDisplayAcquireTimeoutDeferral(out)
}
extern "win64" fn xdisp_close_defer(_: This, h: *mut c_void) {
    XDisplayCloseTimeoutDeferralHandle(h)
}

static XDISPLAY_HDR_VTABLE: Vtable<5> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    missing_slot::<8, 3> as *const c_void,
    xdisp_try_hdr as *const c_void,
]);
static XDISPLAY_HDR_OBJ: ApiObject = ApiObject {
    vtbl: XDISPLAY_HDR_VTABLE.0.as_ptr(),
};

static XDISPLAY_SHELL_VTABLE: Vtable<6> = Vtable([
    unk_query_interface as *const c_void,
    unk_add_ref as *const c_void,
    unk_release as *const c_void,
    xdisp_launch_uri as *const c_void,
    xdisp_acq_defer as *const c_void,
    xdisp_close_defer as *const c_void,
]);
static XDISPLAY_SHELL_OBJ: ApiObject = ApiObject {
    vtbl: XDISPLAY_SHELL_VTABLE.0.as_ptr(),
};

#[allow(clippy::too_many_arguments)]
extern "win64" fn xui_msg_async(
    _: This,
    ab: *mut XAsyncBlock,
    t: *const c_char,
    c: *const c_char,
    b1: *const c_char,
    b2: *const c_char,
    b3: *const c_char,
    def: u32,
    can: u32,
) -> i32 {
    XGameUiShowMessageDialogAsync(ab, t, c, b1, b2, b3, def, can)
}
extern "win64" fn xui_msg_result(_: This, ab: *mut XAsyncBlock, out: *mut u32) -> i32 {
    XGameUiShowMessageDialogResult(ab, out)
}

static XGAMEUI_VTABLE: Vtable<20> = Vtable([
    unk_query_interface as *const c_void,   // 0
    unk_add_ref as *const c_void,           // 1
    unk_release as *const c_void,           // 2
    xui_msg_async as *const c_void,         // 3
    xui_msg_result as *const c_void,        // 4
    missing_slot::<7, 5> as *const c_void,  // 5
    missing_slot::<7, 6> as *const c_void,  // 6
    missing_slot::<7, 7> as *const c_void,  // 7
    missing_slot::<7, 8> as *const c_void,  // 8
    missing_slot::<7, 9> as *const c_void,  // 9
    missing_slot::<7, 10> as *const c_void, // 10
    missing_slot::<7, 11> as *const c_void, // 11
    missing_slot::<7, 12> as *const c_void, // 12
    missing_slot::<7, 13> as *const c_void, // 13
    missing_slot::<7, 14> as *const c_void, // 14
    missing_slot::<7, 15> as *const c_void, // 15
    missing_slot::<7, 16> as *const c_void, // 16
    missing_slot::<7, 17> as *const c_void, // 17
    missing_slot::<7, 18> as *const c_void, // 18
    missing_slot::<7, 19> as *const c_void, // 19
]);
static XGAMEUI_OBJ: ApiObject = ApiObject {
    vtbl: XGAMEUI_VTABLE.0.as_ptr(),
};

// ============================================================================
// Public Dispatcher
// ============================================================================

/// Dispatch `xgameruntime.dll!QueryApiImpl(clsid, iid, out_iface)` against verified offline
/// provider vtables, recording any unrecognized `(clsid, iid)` pair with `caller_rva`.
pub fn dispatch_query_api_impl(
    clsid_ptr: *const Guid,
    iid_ptr: *const Guid,
    out_iface: *mut *mut c_void,
) -> i32 {
    if !out_iface.is_null() {
        // SAFETY: Clear output pointer before dispatch so callers never see stale memory on error.
        unsafe { *out_iface = ptr::null_mut() };
    }
    if clsid_ptr.is_null() || iid_ptr.is_null() || out_iface.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    // SAFETY: Non-null `clsid_ptr` and `iid_ptr` point to 16-byte Win64 `GUID` structs.
    let (clsid, iid) = unsafe { (*clsid_ptr, *iid_ptr) };

    // Strictly offline XStore refusal: never spoof commercial licenses or DRM checks.
    if clsid == CLSID_XSTORE {
        let key = format!(
            "QueryApiImpl[XStore:{}/{}]",
            clsid.format_canonical(),
            iid.format_canonical()
        );
        return record_stub("xgameruntime.dll", &key, HRESULT_REGDB_E_CLASSNOTREG as u64) as i32;
    }

    let matched: Option<Result<&'static ApiObject, i32>> = match clsid {
        c if c == CLSID_XASYNC => {
            if iid == CLSID_XASYNC {
                Some(Ok(&XASYNC_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XSYSTEM => {
            if iid == IID_XSYSTEM_V1 || iid == IID_XSYSTEM_V2 || iid == IID_XSYSTEM_ALIAS || iid == CLSID_XSYSTEM {
                Some(Ok(&XSYSTEM_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XSYSTEM_ANALYTICS => {
            if iid == CLSID_XSYSTEM_ANALYTICS {
                Some(Ok(&XSYSTEM_ANALYTICS_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XGAME => {
            if iid == IID_XGAME || iid == IID_XGAME_ALIAS || iid == CLSID_XGAME {
                Some(Ok(&XGAME_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XGAME_FEATURE => {
            if iid == CLSID_XGAME_FEATURE {
                Some(Ok(&XGAME_FEATURE_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XGAME_INVITE => {
            if iid == CLSID_XGAME_INVITE || iid == IID_XGAME_INVITE_V2 {
                Some(Ok(&XGAME_INVITE_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XGAME_EVENT => {
            if iid == CLSID_XGAME_EVENT {
                Some(Ok(&XGAME_EVENT_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XERROR => {
            if iid == CLSID_XERROR {
                Some(Ok(&XERROR_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_PLS => {
            if iid == IID_PLS_V1 || iid == IID_PLS_V2 || iid == CLSID_PLS {
                Some(Ok(&PLS_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XUSER => {
            if iid == CLSID_XUSER || iid == IID_XUSER_MAIN {
                Some(Ok(&XUSER_OBJ))
            } else if iid == IID_XUSER_GAMERTAG {
                Some(Ok(&XUSER_GAMERTAG_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XUSER_DEVICES => {
            if iid == IID_XUSER_DEVICES || iid == CLSID_XUSER_DEVICES {
                Some(Ok(&XUSER_DEVICES_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XPACKAGE => {
            if iid == IID_XPACKAGE || iid == CLSID_XPACKAGE {
                Some(Ok(&XPACKAGE_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XGAMESAVE => {
            if iid == CLSID_XGAMESAVE || iid == IID_XGAMESAVE_V2 {
                Some(Ok(&XGAMESAVE_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XNETWORKING => {
            if iid == CLSID_XNETWORKING || iid == IID_XNETWORKING_V2 {
                Some(Ok(&XNETWORKING_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XDISPLAY_HDR => {
            if iid == IID_XDISPLAY_HDR || iid == CLSID_XDISPLAY_HDR {
                Some(Ok(&XDISPLAY_HDR_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XDISPLAY_SHELL => {
            if iid == CLSID_XDISPLAY_SHELL {
                Some(Ok(&XDISPLAY_SHELL_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        c if c == CLSID_XGAMEUI => {
            if iid == IID_XGAMEUI || iid == CLSID_XGAMEUI {
                Some(Ok(&XGAMEUI_OBJ))
            } else {
                Some(Err(HRESULT_E_NOINTERFACE))
            }
        }
        _ => None,
    };

    match matched {
        Some(Ok(obj)) => {
            // SAFETY: `out_iface` was checked non-null and `obj` is `'static`.
            unsafe { *out_iface = (obj as *const ApiObject as *mut ApiObject).cast() };
            HRESULT_S_OK
        }
        Some(Err(hr)) => {
            let key = format!(
                "QueryApiImpl[{}/{}]",
                clsid.format_canonical(),
                iid.format_canonical()
            );
            record_stub("xgameruntime.dll", &key, hr as u64) as i32
        }
        None => {
            let key = format!(
                "QueryApiImpl[{}/{}]",
                clsid.format_canonical(),
                iid.format_canonical()
            );
            record_stub("xgameruntime.dll", &key, HRESULT_REGDB_E_CLASSNOTREG as u64) as i32
        }
    }
}
