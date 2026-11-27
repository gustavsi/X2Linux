//! Native Linux core runtime shared library (`libx2l_core.so`).
//!
//! Invoked by thin PE-side shim DLLs (`crates/x2l-pe-shims`) across the Wine `__wine_unix_call`
//! boundary. Owns all heavy HLE, OverlayFS, Vulkan 1.3/1.4, direct SPIR-V, PipeWire, and SDL3 work.

use x2l_common::report::REPORT_SCHEMA_VERSION;

/// ABI version shared between `x2l-pe-shims` and `libx2l_core.so`.
pub const X2L_UNIX_CALL_ABI_VERSION: u32 = 1;

/// C-ABI entry point returning the core runtime Unix-call ABI version.
#[unsafe(no_mangle)]
pub extern "C" fn x2l_core_abi_version() -> u32 {
    X2L_UNIX_CALL_ABI_VERSION
}

/// C-ABI entry point returning the `report.json` telemetry schema version.
#[unsafe(no_mangle)]
pub extern "C" fn x2l_core_report_schema_version() -> u32 {
    REPORT_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_abi_and_report_versions() {
        assert_eq!(x2l_core_abi_version(), X2L_UNIX_CALL_ABI_VERSION);
        assert_eq!(x2l_core_report_schema_version(), REPORT_SCHEMA_VERSION);
    }
}
