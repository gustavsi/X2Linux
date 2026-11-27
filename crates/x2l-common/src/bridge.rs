//! Shared ABI contract between thin Win64 PE shims (`crates/x2l-pe-shims`) and native Linux
//! `libx2l_core.so` loaded in-process via `LD_PRELOAD`.

/// Magic return value from [`OP_HANDSHAKE`] confirming ABI version 1 compatibility.
pub const X2L_BRIDGE_MAGIC: u64 = 0x5832_4C5F_0001_0001;

/// Environment variable populated by `libx2l_core.so`'s ELF `.init_array` constructor with
/// the hex virtual address (`0x...`) of the process-local `extern "win64"` dispatch entrypoint.
pub const ENV_BRIDGE_FN: &str = "X2L_BRIDGE_FN";

/// Environment variable set by `x2linux run` (`1`) to activate the in-process bridge.
pub const ENV_BRIDGE_ENABLED: &str = "X2L_BRIDGE_ENABLED";

/// Standard Win32/GDK `S_OK`.
pub const HRESULT_S_OK: u64 = 0x0000_0000;
/// Standard Win32/GDK `E_NOTIMPL`.
pub const HRESULT_E_NOTIMPL: u64 = 0x8000_4001;
/// Standard Win32/GDK `E_INVALIDARG`.
pub const HRESULT_E_INVALIDARG: u64 = 0x8007_0057;

// Core bridge & telemetry opcodes
pub const OP_HANDSHAKE: u32 = 0x0001;
pub const OP_FLUSH_REPORT: u32 = 0x0002;
pub const OP_RECORD_STUB: u32 = 0x0010;
pub const OP_RECORD_UNKNOWN_ORDINAL: u32 = 0x0011;
pub const OP_RECORD_UNMAPPED_PATH: u32 = 0x0012;
pub const OP_RECORD_VTABLE_PROBE: u32 = 0x0013;

// kernelx HLE opcodes
pub const OP_KERNELX_GET_CONSOLE_TYPE: u32 = 0x0100;
pub const OP_KERNELX_GET_DEVICE_TYPE: u32 = 0x0101;
pub const OP_KERNELX_XMEM_ALLOC: u32 = 0x0102;
pub const OP_KERNELX_XMEM_FREE: u32 = 0x0103;
pub const OP_KERNELX_TITLE_MEMORY_STATUS: u32 = 0x0104;
pub const OP_KERNELX_VIRTUAL_ALLOC: u32 = 0x0105;

// XGameRuntime HLE opcodes
pub const OP_XGR_INITIALIZE: u32 = 0x0200;
pub const OP_XGR_UNINITIALIZE: u32 = 0x0201;
pub const OP_XGR_USER_GET_GAMERTAG: u32 = 0x0202;
pub const OP_XGR_PACKAGE_MOUNT_ROOT: u32 = 0x0203;
pub const OP_XGR_GAMESAVE_ROOT: u32 = 0x0204;
pub const OP_XGR_STORE_OFFLINE: u32 = 0x0205;

// D3D12.X Null M1 / Vulkan M2 opcodes
pub const OP_D3D12X_CREATE_DEVICE: u32 = 0x0300;
pub const OP_D3D12X_CAPTURE_SHADER: u32 = 0x0301;
