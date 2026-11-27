//! Ultra-thin PE-side shim DLL bridge (`x86_64-pc-windows-gnu`).
//!
//! Wine/Proton (`ntsync`) owns PE32+ loading, TLS, SEH, and `ucrtbase`. This crate only
//! exposes minimal `extern "win64"` console shims and forwards into native `libx2l_core.so`
//! via `__wine_unix_call`.

/// ABI version expected from native `libx2l_core.so`.
pub const EXPECTED_CORE_ABI_VERSION: u32 = 1;

/// Minimal `win64` ABI handshake stub for M0 cross-compilation verification.
#[unsafe(no_mangle)]
pub extern "win64" fn x2l_shim_abi_version() -> u32 {
    EXPECTED_CORE_ABI_VERSION
}
