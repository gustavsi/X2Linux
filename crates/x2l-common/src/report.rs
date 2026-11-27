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

/// Top-level sanitized runtime telemetry report (`report.json`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryReport {
    pub schema_version: u32,
    pub title_id: String,
    #[serde(default)]
    pub host: HostRuntimeMetadata,
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
    }
}
