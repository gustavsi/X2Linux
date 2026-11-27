//! Clean-room synthetic console PE32+ end-to-end test title (`x86_64-pc-windows-gnu`).
//!
//! Verifies all M1 compatibility invariants under `x2linux run`:
//! 1. `kernelx!GetConsoleType` & `TitleMemoryStatus`
//! 2. `kernelx!XMemAlloc` + memory read/write + `XMemFree`
//! 3. `xgameruntime!XGameRuntimeInitialize`, offline `XStoreCreateContext`, `XGameRuntimeUninitialize`
//! 4. `d3d12_xs!D3D12CreateDevice` (Null M1 COM object) + spare vtable probe slot (slot 45) + `Release`
//! 5. Unknown ordinal `kernelx.dll!#999` auto-stub call without crashing (telemetry in `report.json`)
//! 6. Multi-threaded static TLS (`thread_local!`) + panic/unwind handling
//! 7. Copy-on-write OverlayFS reads, overwrite, create, and whiteout delete inside `G:\`

use std::cell::Cell;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::thread;

thread_local! {
    static SYNTHETIC_TLS_MARKER: Cell<u64> = const { Cell::new(0x5832_4C5F_544C_5331) };
}

#[cfg(target_os = "windows")]
mod console_imports {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    pub struct TitleMemoryStatus {
        pub dw_length: u32,
        pub dw_reserved: u32,
        pub ull_total_mem: u64,
        pub ull_avail_mem: u64,
        pub ull_legacy_used: u64,
        pub ull_legacy_peak: u64,
        pub ull_legacy_avail: u64,
        pub ull_title_used: u64,
        pub ull_title_avail: u64,
        pub ull_legacy_page_faults: u64,
        pub ull_title_page_faults: u64,
    }

    #[link(name = "kernelx", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void;
        pub fn XMemFree(ptr: *mut c_void, attrs: u64);
        pub fn GetConsoleType() -> u32;
        pub fn TitleMemoryStatus(out: *mut TitleMemoryStatus) -> i32;
        #[link_ordinal(999)]
        pub fn x2l_synthetic_ordinal_probe() -> u64;
    }

    #[link(name = "xgameruntime", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XGameRuntimeInitialize() -> i32;
        pub fn XGameRuntimeUninitialize();
        pub fn XStoreCreateContext(user: *mut c_void, out_ctx: *mut *mut c_void) -> i32;
    }

    #[link(name = "d3d12_xs", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn D3D12CreateDevice(
            adapter: *mut c_void,
            min_feature_level: u32,
            riid: *const c_void,
            out_device: *mut *mut c_void,
        ) -> i32;
    }
}

fn check(name: &str, ok: bool, failures: &mut u32) {
    if ok {
        println!("[PASS] {name}");
    } else {
        eprintln!("[FAIL] {name}");
        *failures += 1;
    }
}

fn main() -> ExitCode {
    let mut failures = 0u32;

    #[cfg(target_os = "windows")]
    run_console_api_checks(&mut failures);

    // 6. Multi-threaded static TLS + panic/unwind handling
    let tls_handles: Vec<_> = (0u64..4)
        .map(|tid| {
            thread::spawn(move || {
                SYNTHETIC_TLS_MARKER.with(|slot| {
                    let initial = slot.get();
                    slot.set(initial.wrapping_add(tid + 1));
                    (initial, slot.get())
                })
            })
        })
        .collect();
    let mut tls_ok = true;
    for (idx, handle) in tls_handles.into_iter().enumerate() {
        match handle.join() {
            Ok((initial, updated)) => {
                if initial != 0x5832_4C5F_544C_5331
                    || updated != 0x5832_4C5F_544C_5331u64.wrapping_add(idx as u64 + 1)
                {
                    tls_ok = false;
                }
            }
            Err(_) => tls_ok = false,
        }
    }
    check("multi-threaded static TLS isolation", tls_ok, &mut failures);

    let unwind_ok = std::panic::catch_unwind(|| {
        panic!("synthetic-unwind-probe");
    })
    .is_err();
    check("SEH / Rust panic unwind handling", unwind_ok, &mut failures);

    // 7. OverlayFS copy-on-write & whiteout checks (active when fixture files exist)
    run_overlay_fixture_checks(&mut failures);

    if failures == 0 {
        println!("[x2l-testtitle] ALL CHECKS PASSED");
        ExitCode::SUCCESS
    } else {
        eprintln!("[x2l-testtitle] {failures} CHECK(S) FAILED");
        ExitCode::from(1)
    }
}

#[cfg(target_os = "windows")]
fn run_console_api_checks(failures: &mut u32) {
    use console_imports::*;
    use std::ffi::c_void;

    // 1. GetConsoleType & TitleMemoryStatus
    // SAFETY: Exported by `kernelx.dll` PE shim.
    let console_type = unsafe { GetConsoleType() };
    check(
        &format!("kernelx!GetConsoleType returned 0x{console_type:04x}"),
        console_type == 0x0007 || console_type == 0x0006,
        failures,
    );

    let mut mem_status = TitleMemoryStatus::default();
    // SAFETY: `mem_status` is a valid writable `TITLEMEMORYSTATUS` struct.
    let status_ok = unsafe { TitleMemoryStatus(&mut mem_status) } != 0
        && mem_status.ull_total_mem >= 8_000_000_000;
    check(
        &format!(
            "kernelx!TitleMemoryStatus (total={} B)",
            mem_status.ull_total_mem
        ),
        status_ok,
        failures,
    );

    // 2. XMemAlloc + write/read + XMemFree
    // SAFETY: Valid 4096-byte allocation from `kernelx!XMemAlloc`.
    let xmem_ok = unsafe {
        let ptr = XMemAlloc(4096, 0x1000) as *mut u8;
        if ptr.is_null() {
            false
        } else {
            let slice = std::slice::from_raw_parts_mut(ptr, 4096);
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i & 0xFF) as u8;
            }
            let verified = slice
                .iter()
                .enumerate()
                .all(|(i, &b)| b == (i & 0xFF) as u8);
            XMemFree(ptr as *mut c_void, 0);
            verified
        }
    };
    check(
        "kernelx!XMemAlloc + write/read + XMemFree",
        xmem_ok,
        failures,
    );

    // 3. XGameRuntimeInitialize / offline XStore / XGameRuntimeUninitialize
    // SAFETY: Offline XGameRuntime shim calls.
    let xgr_init_hr = unsafe { XGameRuntimeInitialize() };
    check(
        &format!("xgameruntime!XGameRuntimeInitialize (hr=0x{xgr_init_hr:08x})"),
        xgr_init_hr == 0,
        failures,
    );

    let mut store_ctx: *mut c_void = std::ptr::null_mut();
    // SAFETY: Verifies XStore stays offline (E_NOTIMPL = 0x80004001) and never spoofs licenses.
    let store_hr = unsafe { XStoreCreateContext(std::ptr::null_mut(), &mut store_ctx) };
    check(
        &format!("xgameruntime!XStoreCreateContext offline refusal (hr=0x{store_hr:08x})"),
        store_hr == -2147467263,
        failures,
    );

    // SAFETY: Matches successful `XGameRuntimeInitialize`.
    unsafe { XGameRuntimeUninitialize() };
    check("xgameruntime!XGameRuntimeUninitialize", true, failures);

    // 4. D3D12CreateDevice (Null M1 COM object) + spare vtable probe slot 45 + Release
    let mut dev: *mut c_void = std::ptr::null_mut();
    // SAFETY: `dev` is a writable COM output pointer.
    let d3d_hr =
        unsafe { D3D12CreateDevice(std::ptr::null_mut(), 0xb000, std::ptr::null(), &mut dev) };
    let mut d3d_ok = d3d_hr == 0 && !dev.is_null();
    if d3d_ok {
        // SAFETY: `dev` points to a COM `NullD3D12DeviceX` whose first field is `*const usize`
        // vtable with 128 Win64 slots (`slot 2 = Release`, `slot 45 = spare probe thunk`).
        unsafe {
            let vtbl = *(dev as *const *const usize);
            let probe_45: extern "win64" fn(*mut c_void) -> u64 =
                std::mem::transmute(*vtbl.add(45));
            let release: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vtbl.add(2));
            let probe_ret = probe_45(dev);
            let rem_refs = release(dev);
            d3d_ok = probe_ret == 0 && rem_refs == 0;
        }
    }
    check(
        "d3d12_xs!D3D12CreateDevice + spare vtable probe[45] + Release",
        d3d_ok,
        failures,
    );

    // 5. Unknown ordinal kernelx.dll!#999 auto-stub
    // SAFETY: Exported via `kernelx.def` NONAME ordinal `#999` auto-stub thunk.
    let ord_ret = unsafe { x2l_synthetic_ordinal_probe() };
    check(
        &format!("kernelx.dll!#999 unknown-ordinal auto-stub (ret=0x{ord_ret:x})"),
        ord_ret == 0,
        failures,
    );
}

fn run_overlay_fixture_checks(failures: &mut u32) {
    let keep = Path::new("fixture_keep.txt");
    if !keep.is_file() {
        println!("[SKIP] OverlayFS G:\\ fixture files not present in working directory");
        return;
    }

    let read_ok = fs::read_to_string(keep)
        .map(|s| s.trim() == "immutable-keep-payload")
        .unwrap_or(false);
    check("overlay read fixture_keep.txt", read_ok, failures);

    let modify = Path::new("fixture_modify.txt");
    let cow_ok = fs::write(modify, "overwritten-inside-overlay\n").is_ok()
        && fs::read_to_string(modify)
            .map(|s| s == "overwritten-inside-overlay\n")
            .unwrap_or(false);
    check("overlay CoW overwrite fixture_modify.txt", cow_ok, failures);

    let created = Path::new("fixture_created.txt");
    let create_ok = fs::write(created, "created-inside-overlay\n").is_ok()
        && fs::read_to_string(created)
            .map(|s| s == "created-inside-overlay\n")
            .unwrap_or(false);
    check("overlay create fixture_created.txt", create_ok, failures);

    let delete_target = Path::new("fixture_delete.txt");
    let delete_ok = fs::remove_file(delete_target).is_ok() && !delete_target.exists();
    check(
        "overlay whiteout delete fixture_delete.txt",
        delete_ok,
        failures,
    );
}
