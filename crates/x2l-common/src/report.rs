//! Sanitized `report.json` telemetry schema.
//!
//! X2Linux forbids silent stubs: every unimplemented import, unknown ordinal, unmapped path,
//! or unmapped `D3D12.X` vtable slot records its caller RVA and hit count in `report.json`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Current schema version for `report.json`.
pub const REPORT_SCHEMA_VERSION: u32 = 1;

/// Host environment metadata captured for each runtime session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HostRuntimeMetadata {
    pub kernel_release: String,
    pub ntsync_available: bool,
    pub ntsync_enabled: bool,
    pub overlayfs_mode: String,
    pub wine_binary: String,
    pub device_type: String,
}

/// GPU backend telemetry captured for each runtime session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuTelemetry {
    pub backend: String,
    pub present_surface: String,
    pub device_name: String,
    pub driver_info: String,
    pub external_memory_host: bool,
    pub descriptor_buffer: bool,
    pub userfaultfd_mode: String,
    pub shaders_compiled: u64,
    pub frames_presented: u64,
}

/// Win64 register snapshot captured inside the fatal VEH handler (when valid in `ContextFlags`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExceptionRegisters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rip: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rsp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rax: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rcx: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rdx: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r8: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r9: Option<String>,
}

/// Fatal exception snapshot captured by the non-invasive VEH crash guard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExceptionRecord {
    pub code: String,
    pub address: String,
    pub rva: String,
    pub module: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registers: Option<ExceptionRegisters>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backtrace_rvas: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_debug_strings: Vec<String>,
}

/// Top-level sanitized runtime telemetry report (`report.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryReport {
    pub schema_version: u32,
    pub title_id: String,
    #[serde(default)]
    pub host: HostRuntimeMetadata,
    #[serde(default)]
    pub gpu: GpuTelemetry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_exception: Option<ExceptionRecord>,
    pub missing_imports: BTreeMap<String, MissingImportEntry>,
    pub unknown_ordinals: BTreeMap<String, UnknownOrdinalEntry>,
    pub unmapped_paths: BTreeMap<String, UnmappedPathEntry>,
    pub unmapped_vtable_slots: BTreeMap<String, UnmappedVtableSlotEntry>,
}

/// Recorded calls to an unimplemented named import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingImportEntry {
    pub module: String,
    pub symbol: String,
    pub count: u64,
    pub caller_rvas: Vec<String>,
}

/// Recorded calls to an unknown DLL export ordinal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnknownOrdinalEntry {
    pub module: String,
    pub ordinal: u16,
    pub count: u64,
    pub caller_rvas: Vec<String>,
}

/// Recorded accesses to an unmapped console path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnmappedPathEntry {
    pub path: String,
    pub count: u64,
    pub caller_rvas: Vec<String>,
}

/// Recorded calls to an unmapped `D3D12.X` COM vtable slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnmappedVtableSlotEntry {
    pub interface: String,
    pub slot_index: u32,
    pub count: u64,
    pub caller_rvas: Vec<String>,
}

impl TelemetryReport {
    /// Create a fresh telemetry report for a title identifier.
    pub fn new(title_id: impl Into<String>) -> Self {
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            title_id: title_id.into(),
            host: HostRuntimeMetadata::default(),
            gpu: GpuTelemetry::default(),
            last_exception: None,
            missing_imports: BTreeMap::new(),
            unknown_ordinals: BTreeMap::new(),
            unmapped_paths: BTreeMap::new(),
            unmapped_vtable_slots: BTreeMap::new(),
        }
    }

    /// Record a missing named import hit with its caller RVA.
    pub fn record_missing_import(&mut self, module: &str, symbol: &str, caller_rva: u64) {
        let key = format!("{module}!{symbol}");
        let rva = format_rva(caller_rva);
        let entry = self
            .missing_imports
            .entry(key)
            .or_insert_with(|| MissingImportEntry {
                module: module.to_owned(),
                symbol: symbol.to_owned(),
                count: 0,
                caller_rvas: Vec::new(),
            });
        entry.count += 1;
        push_unique(&mut entry.caller_rvas, rva);
    }

    /// Record an unknown ordinal hit with its caller RVA.
    pub fn record_unknown_ordinal(&mut self, module: &str, ordinal: u16, caller_rva: u64) {
        let key = format!("{module}#{ordinal}");
        let rva = format_rva(caller_rva);
        let entry = self
            .unknown_ordinals
            .entry(key)
            .or_insert_with(|| UnknownOrdinalEntry {
                module: module.to_owned(),
                ordinal,
                count: 0,
                caller_rvas: Vec::new(),
            });
        entry.count += 1;
        push_unique(&mut entry.caller_rvas, rva);
    }

    /// Record an unmapped console path access with its caller RVA.
    pub fn record_unmapped_path(&mut self, path: &str, caller_rva: u64) {
        let rva = format_rva(caller_rva);
        let entry = self
            .unmapped_paths
            .entry(path.to_owned())
            .or_insert_with(|| UnmappedPathEntry {
                path: path.to_owned(),
                count: 0,
                caller_rvas: Vec::new(),
            });
        entry.count += 1;
        push_unique(&mut entry.caller_rvas, rva);
    }

    /// Record an unmapped `D3D12.X` COM vtable slot hit with its caller RVA.
    pub fn record_unmapped_vtable_slot(
        &mut self,
        interface: &str,
        slot_index: u32,
        caller_rva: u64,
    ) {
        let key = format!("{interface}[{slot_index}]");
        let rva = format_rva(caller_rva);
        let entry =
            self.unmapped_vtable_slots
                .entry(key)
                .or_insert_with(|| UnmappedVtableSlotEntry {
                    interface: interface.to_owned(),
                    slot_index,
                    count: 0,
                    caller_rvas: Vec::new(),
                });
        entry.count += 1;
        push_unique(&mut entry.caller_rvas, rva);
    }

    /// Record a fatal guest exception captured by the VEH crash guard.
    pub fn record_exception(&mut self, code: u32, address: u64, rva: u64, module: &str) {
        self.record_exception_detailed(code, address, rva, module, None, &[], &[]);
    }

    /// Record a fatal guest exception with optional register context, guest backtrace RVAs,
    /// and recent `OutputDebugString` tail.
    #[allow(clippy::too_many_arguments)]
    pub fn record_exception_detailed(
        &mut self,
        code: u32,
        address: u64,
        rva: u64,
        module: &str,
        registers: Option<ExceptionRegisters>,
        backtrace_rvas: &[u64],
        recent_debug_strings: &[String],
    ) {
        self.last_exception = Some(ExceptionRecord {
            code: format!("0x{code:08x}"),
            address: format!("0x{address:016x}"),
            rva: format_rva(rva),
            module: module.to_owned(),
            registers,
            backtrace_rvas: backtrace_rvas.iter().map(|&v| format_rva(v)).collect(),
            recent_debug_strings: recent_debug_strings.to_vec(),
        });
    }

    /// Serialize this telemetry report as pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a telemetry report from JSON.
    pub fn from_json_str(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }
}

fn format_rva(rva: u64) -> String {
    format!("0x{rva:08x}")
}

fn push_unique(list: &mut Vec<String>, value: String) {
    if !list.iter().any(|existing| existing == &value) {
        list.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_roundtrip_records_counts_and_unique_rvas() {
        let mut report = TelemetryReport::new("synthetic-title");
        report.record_missing_import("kernelx.dll", "XMemAlloc", 0x1040);
        report.record_missing_import("kernelx.dll", "XMemAlloc", 0x1040);
        report.record_missing_import("kernelx.dll", "XMemAlloc", 0x2080);
        report.record_unknown_ordinal("kernelx.dll", 42, 0x3000);
        report.record_unmapped_path("X:\\content\\placeholder.dat", 0x4000);
        report.record_unmapped_vtable_slot("ID3D12DeviceX", 73, 0x5000);

        let json = report.to_json_pretty().expect("serialize report.json");
        let decoded = TelemetryReport::from_json_str(&json).expect("deserialize report.json");

        assert_eq!(decoded, report);
        let import = &decoded.missing_imports["kernelx.dll!XMemAlloc"];
        assert_eq!(import.count, 3);
        assert_eq!(import.caller_rvas, vec!["0x00001040", "0x00002080"]);
        assert_eq!(decoded.unknown_ordinals["kernelx.dll#42"].count, 1);
        assert_eq!(decoded.unmapped_vtable_slots["ID3D12DeviceX[73]"].count, 1);
        assert!(!json.contains("last_exception"));
        assert!(decoded.last_exception.is_none());
    }

    #[test]
    fn telemetry_records_last_exception_when_present() {
        let mut report = TelemetryReport::new("synthetic-title");
        report.record_exception(0xC000_0005, 0x0000_0001_4000_1234, 0x1234, "test.exe");
        let json = report.to_json_pretty().expect("serialize report.json");
        assert!(json.contains("\"last_exception\""));
        let decoded = TelemetryReport::from_json_str(&json).expect("deserialize report.json");
        let exc = decoded.last_exception.expect("last_exception populated");
        assert_eq!(exc.code, "0xc0000005");
        assert_eq!(exc.address, "0x0000000140001234");
        assert_eq!(exc.rva, "0x00001234");
        assert_eq!(exc.module, "test.exe");
    }
}
