//! Clean-room synthetic console PE32+ executable for X2Linux integration testing
//! (`x86_64-pc-windows-gnu`).
//!
//! Declares real PE32+ import entries for console libraries (`kernelx.dll`, `xgameruntime.dll`,
//! `d3d12_xs.dll`), an import by ordinal on `kernelx.dll`, and a static TLS slot so `x2linux
//! inspect` can verify header, TLS, and import policy detection against a genuine cross-compiled
//! executable.

use std::cell::Cell;
use std::hint::black_box;

thread_local! {
    static SYNTHETIC_TLS_MARKER: Cell<u64> = const { Cell::new(0x5832_4C5F_544C_5331) };
}

#[cfg(target_os = "windows")]
mod console_imports {
    use std::ffi::c_void;

    #[link(name = "kernelx", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void;
        pub fn XMemFree(ptr: *mut c_void, attrs: u64);
        pub fn GetConsoleType() -> u32;
        #[link_ordinal(999)]
        pub fn x2l_synthetic_ordinal_probe() -> u32;
    }

    #[link(name = "xgameruntime", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XGameRuntimeInitialize() -> i32;
        pub fn XGameRuntimeUninitialize();
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

fn main() {
    let tls_val = SYNTHETIC_TLS_MARKER.with(|slot| {
        let current = slot.get();
        slot.set(current.wrapping_add(1));
        slot.get()
    });
    println!("x2l-testtitle: clean-room synthetic console fixture (tls=0x{tls_val:016x})");

    // Reference every declared console import so the linker emits live IAT/INT descriptors
    // without executing them unless invoked inside a live compatibility runtime.
    #[cfg(target_os = "windows")]
    if black_box(false) {
        // SAFETY: guarded by `black_box(false)` for static PE import emission in M1.1;
        // runtime invocation tests activate under the M1.2 Wine/ntsync bridge.
        unsafe {
            let mem = console_imports::XMemAlloc(4096, 0);
            console_imports::XMemFree(mem, 0);
            let _ = black_box(console_imports::GetConsoleType());
            let _ = black_box(console_imports::x2l_synthetic_ordinal_probe());
            let _ = black_box(console_imports::XGameRuntimeInitialize());
            console_imports::XGameRuntimeUninitialize();
            let mut dev = std::ptr::null_mut();
            let _ = black_box(console_imports::D3D12CreateDevice(
                std::ptr::null_mut(),
                0xb000,
                std::ptr::null(),
                &mut dev,
            ));
        }
    }

    let _ = black_box(tls_val);
}
