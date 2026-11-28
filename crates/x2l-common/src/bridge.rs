//! Shared ABI contract between thin Win64 PE shims (`crates/x2l-pe-shims`) and native Linux
//! `libx2l_core.so` loaded in-process via `LD_PRELOAD`.

mod exception;
pub use exception::{CrashSnapshot, CrashSnapshotV2};

/// Magic return value from [`OP_HANDSHAKE`] confirming ABI version 1 compatibility.
pub const X2L_BRIDGE_MAGIC: u64 = 0x5832_4C5F_0001_0001;

/// Environment variable populated by `libx2l_core.so`'s ELF `.init_array` constructor with
/// the hex virtual address (`0x...`) of the process-local `extern "win64"` dispatch entrypoint.
pub const ENV_BRIDGE_FN: &str = "X2L_BRIDGE_FN";

/// Environment variable set by `x2linux run` (`1`) to activate the in-process bridge.
pub const ENV_BRIDGE_ENABLED: &str = "X2L_BRIDGE_ENABLED";
/// Environment variable selecting the GPU backend (`"null"` or `"vulkan"`).
pub const ENV_GPU_BACKEND: &str = "X2L_GPU_BACKEND";
/// Environment variable selecting GPU presentation mode (`"headless"`, `"window"`, `"auto"`, `"wayland"`, `"x11"`).
pub const ENV_GPU_PRESENT_MODE: &str = "X2L_GPU_PRESENT_MODE";
/// Environment variable pointing to the per-title XDG SPIR-V cache directory.
pub const ENV_SPIRV_CACHE_DIR: &str = "X2L_SPIRV_CACHE_DIR";
/// Optional environment variable specifying a P6 PPM frame dump path for `PresentX`.
pub const ENV_FRAME_DUMP: &str = "X2L_FRAME_DUMP";
/// Optional environment variable enabling Khronos Vulkan Validation Layers (`1`/`true`).
pub const ENV_GPU_VALIDATION: &str = "X2L_GPU_VALIDATION";
/// Optional environment variable preferring Tier-1 `VK_EXT_external_memory_host` zero-copy (`1`/`0`).
pub const ENV_PREFER_REBAR_ZERO_COPY: &str = "X2L_PREFER_REBAR_ZERO_COPY";
/// Optional environment variable enabling line-buffered HLE call tracing (`1`/`true`).
pub const ENV_TRACE_HLE: &str = "X2L_TRACE_HLE";

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
pub const OP_TRACE_HLE_LINE: u32 = 0x0014;
/// Snapshot form passes a pointer in a0 and zeroes in a1 through a4. Legacy shims
/// pass zero in a5 for the original prefix. Extended shims pass the snapshot size.
pub const OP_RECORD_EXCEPTION: u32 = 0x0015;
pub const OP_RECORD_DEBUG_STRING: u32 = 0x0016;
pub const OP_RECORD_MISSING_PROC: u32 = 0x0017;

// kernelx & xmem HLE opcodes
pub const OP_KERNELX_GET_CONSOLE_TYPE: u32 = 0x0100;
pub const OP_KERNELX_GET_DEVICE_TYPE: u32 = 0x0101;
pub const OP_KERNELX_XMEM_ALLOC: u32 = 0x0102;
pub const OP_KERNELX_XMEM_FREE: u32 = 0x0103;
pub const OP_KERNELX_TITLE_MEMORY_STATUS: u32 = 0x0104;
pub const OP_KERNELX_VIRTUAL_ALLOC: u32 = 0x0105;

pub const OP_XMEM_ALLOC_DEFAULT: u32 = 0x0110;
pub const OP_XMEM_FREE_DEFAULT: u32 = 0x0111;
pub const OP_XMEM_ALLOC_PHYSICAL_PAGES: u32 = 0x0112;
pub const OP_XMEM_FREE_PHYSICAL_PAGES: u32 = 0x0113;
pub const OP_XMEM_MAP_PHYSICAL_PAGES: u32 = 0x0114;
pub const OP_XMEM_MAP_PHYSICAL_PAGES_SCATTER: u32 = 0x0115;
pub const OP_XMEM_VIRTUAL_ALLOC: u32 = 0x0116;
pub const OP_XMEM_VIRTUAL_QUERY: u32 = 0x0117;
pub const OP_XMEM_TRANSFER_MEMORY: u32 = 0x0118;
pub const OP_XMEM_GET_WORKING_SET_STATS: u32 = 0x0119;
pub const OP_XMEM_IS_TOOLS_MEM_AVAILABLE: u32 = 0x011A;
pub const OP_XMEM_TRY_WIN32_VALLOC: u32 = 0x011B;
pub const OP_XMEM_TRY_WIN32_VFREE: u32 = 0x011C;
pub const OP_XMEM_TRY_WIN32_VQUERY: u32 = 0x011D;

// XGameRuntime HLE opcodes
pub const OP_XGR_INITIALIZE: u32 = 0x0200;
pub const OP_XGR_UNINITIALIZE: u32 = 0x0201;
pub const OP_XGR_USER_GET_GAMERTAG: u32 = 0x0202;
pub const OP_XGR_PACKAGE_MOUNT_ROOT: u32 = 0x0203;
pub const OP_XGR_GAMESAVE_ROOT: u32 = 0x0204;
pub const OP_XGR_STORE_OFFLINE: u32 = 0x0205;
pub const OP_XGR_GET_DEVICE_TYPE: u32 = 0x0206;
pub const OP_XGR_GET_TITLE_ID: u32 = 0x0207;
pub const OP_XGR_GET_CONSOLE_ID: u32 = 0x0208;
pub const OP_XGR_GET_SANDBOX_ID: u32 = 0x0209;
pub const OP_XGR_GET_APP_DEVICE_ID: u32 = 0x020A;
pub const OP_XGR_PLS_GET_PATH: u32 = 0x020B;
pub const OP_XGR_PLS_GET_SPACE_INFO: u32 = 0x020C;
pub const OP_XGR_GAMESAVE_WRITE_BLOB: u32 = 0x020D;
pub const OP_XGR_GAMESAVE_READ_BLOB: u32 = 0x020E;
pub const OP_XGR_GAMESAVE_ENUM_BLOBS: u32 = 0x020F;

// D3D12.X Null M1 / Vulkan M2 / Xbox M3.1 opcodes
pub const OP_D3D12X_CREATE_DEVICE: u32 = 0x0300;
pub const OP_D3D12X_CAPTURE_SHADER: u32 = 0x0301;
pub const OP_D3D12X_XBOX_CREATE_DEVICE: u32 = 0x0302;
pub const OP_D3D12X_CONFIGURE_VIRTUAL_MEMORY: u32 = 0x0303;
pub const OP_D3D12X_SERIALIZE_VERSIONED_ROOT_SIG: u32 = 0x0304;

// xg_xs.dll Texture Address Computer opcodes
pub const OP_XG_CREATE_TEXTURE_COMPUTER: u32 = 0x0400;
pub const OP_XG_SUGGEST_SWIZZLE_MODE: u32 = 0x0401;

// XCRDAPI.dll DirectStorage opcodes
pub const OP_DSTORAGE_GET_FACTORY: u32 = 0x0500;

// GameInput.dll opcodes
pub const OP_GAMEINPUT_CREATE: u32 = 0x0600;

// Audio co-processor shims (AcpHal / xapu / chad)
pub const OP_ACPHAL_CREATE: u32 = 0x0700;
pub const OP_ACPHAL_ALLOC_SHAPE_CONTEXTS: u32 = 0x0701;
pub const OP_ACPHAL_RELEASE_SHAPE_CONTEXTS: u32 = 0x0702;
pub const OP_APU_CREATE_HEAP: u32 = 0x0703;
pub const OP_APU_ALLOC: u32 = 0x0704;
pub const OP_APU_FREE: u32 = 0x0705;
pub const OP_APU_MAP_APU_ADDRESS: u32 = 0x0706;
pub const OP_APU_MAP_VIRTUAL_ADDRESS: u32 = 0x0707;
pub const OP_APU_IS_VIRTUAL_ADDRESS_VALID: u32 = 0x0708;

pub const OP_XAPU_CONNECT: u32 = 0x0710;
pub const OP_XAPU_DISCONNECT: u32 = 0x0711;
pub const OP_XAPU_ENQUEUE_COMMAND: u32 = 0x0712;
pub const OP_XAPU_DEQUEUE_RESULT: u32 = 0x0713;
pub const OP_XAPU_GET_RESULT_COUNT: u32 = 0x0714;

pub const OP_CHAD_CONNECT: u32 = 0x0720;
pub const OP_CHAD_DISCONNECT: u32 = 0x0721;
pub const OP_CHAD_ACTIVATE: u32 = 0x0722;
pub const OP_CHAD_DEACTIVATE: u32 = 0x0723;
pub const OP_CHAD_SUBMIT_COMMAND: u32 = 0x0724;

// PSM AppNotify opcodes
pub const OP_PSM_REGISTER_APP_STATE: u32 = 0x0800;
pub const OP_PSM_UNREGISTER_APP_STATE: u32 = 0x0801;
pub const OP_PSM_REGISTER_CONSTRAINED: u32 = 0x0802;
pub const OP_PSM_UNREGISTER_CONSTRAINED: u32 = 0x0803;
