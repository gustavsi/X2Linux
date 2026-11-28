//! Shared X2Linux primitives: XDG paths, `x2linux.toml` configuration, bridge opcodes, and `report.json` telemetry.

pub mod bridge;
pub mod config;
pub mod report;
pub mod xdg;

pub use config::{
    AudioBackend, AudioConfig, ConsoleConfig, DeviceType, GpuBackend, GpuConfig, LoggingConfig,
    OverlayFsMode, RuntimeConfig, StubsConfig, X2LinuxConfig,
};
pub use report::{
    GpuTelemetry, HostRuntimeMetadata, MissingImportEntry, TelemetryReport, UnknownOrdinalEntry,
    UnmappedPathEntry, UnmappedVtableSlotEntry,
};
pub use xdg::{TitleOverlayDirs, XdgError, XdgPaths};
