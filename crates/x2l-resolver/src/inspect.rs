//! High-level read-only PE32+ inspection report (`x2linux inspect`).

use crate::pe::{ImportDll, ImportSymbol, PeError, PeImage, TlsInfo};
use crate::policy::{ImportPolicy, ImportResolution};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;
use x2l_common::report::{REPORT_SCHEMA_VERSION, TelemetryReport};

/// Errors returned while inspecting a PE32+ file on disk.
#[derive(Debug, Error)]
pub enum InspectError {
    #[error("failed to open '{path}' for read-only inspection: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("PE32+ parse failure in '{path}': {source}")]
    Pe {
        path: String,
        #[source]
        source: PeError,
    },
}

/// Serializable static TLS summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TlsSummary {
    pub directory_rva: String,
    pub template_bytes: u64,
    pub zero_fill_bytes: u32,
    pub total_bytes: u64,
    pub has_callbacks: bool,
}

impl From<TlsInfo> for TlsSummary {
    fn from(info: TlsInfo) -> Self {
        Self {
            directory_rva: format!("0x{:08x}", info.directory_rva),
            template_bytes: info.template_bytes(),
            zero_fill_bytes: info.size_of_zero_fill,
            total_bytes: info.total_bytes(),
            has_callbacks: info.address_of_callbacks != 0,
        }
    }
}

/// One classified import entry in an [`InspectReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InspectedImport {
    pub dll: String,
    pub symbol: String,
    pub delay_load: bool,
    pub iat_rva: String,
    pub resolution: ImportResolution,
}

/// Aggregate import classification counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportCounts {
    pub total: usize,
    pub implemented: usize,
    pub system_fallback: usize,
    pub game_local: usize,
    pub stubs: usize,
    pub unknown_ordinals: usize,
}

/// Complete serializable output of `x2linux inspect`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InspectReport {
    pub schema_version: u32,
    pub file: String,
    pub machine: String,
    pub is_dll: bool,
    pub subsystem: u16,
    pub image_base: String,
    pub size_of_image: String,
    pub entry_rva: String,
    pub stack_reserve: String,
    pub has_relocations: bool,
    pub exception_entries: u32,
    pub tls: Option<TlsSummary>,
    pub console_dlls: Vec<String>,
    pub counts: ImportCounts,
    pub imports: Vec<InspectedImport>,
}

/// Inspect a PE32+ executable on disk strictly read-only.
pub fn inspect_pe_file(path: &Path, policy: &ImportPolicy) -> Result<InspectReport, InspectError> {
    let mut file = File::open(path).map_err(|source| InspectError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| InspectError::Io {
            path: path.display().to_string(),
            source,
        })?;
    inspect_pe_bytes(&bytes, &path.display().to_string(), path.parent(), policy)
}

/// Inspect a PE32+ executable from in-memory bytes.
pub fn inspect_pe_bytes(
    bytes: &[u8],
    file_label: &str,
    title_dir: Option<&Path>,
    policy: &ImportPolicy,
) -> Result<InspectReport, InspectError> {
    let pe = PeImage::parse(bytes).map_err(|source| InspectError::Pe {
        path: file_label.to_owned(),
        source,
    })?;

    let mut all_dlls: Vec<ImportDll> = pe.imports().map_err(|source| InspectError::Pe {
        path: file_label.to_owned(),
        source,
    })?;
    let delay = pe.delay_imports().map_err(|source| InspectError::Pe {
        path: file_label.to_owned(),
        source,
    })?;
    all_dlls.extend(delay);

    let tls = pe
        .tls()
        .map_err(|source| InspectError::Pe {
            path: file_label.to_owned(),
            source,
        })?
        .map(TlsSummary::from);

    let mut console_set = BTreeSet::new();
    let mut counts = ImportCounts::default();
    let mut imports = Vec::new();

    for dll_group in &all_dlls {
        if policy.is_console_dll(&dll_group.dll) {
            console_set.insert(dll_group.dll.to_ascii_lowercase());
        }
        for entry in &dll_group.entries {
            let resolution = policy.classify(&dll_group.dll, &entry.symbol, title_dir);
            counts.total += 1;
            match &resolution {
                ImportResolution::Implemented { .. } => counts.implemented += 1,
                ImportResolution::SystemFallback { .. } => counts.system_fallback += 1,
                ImportResolution::GameLocal { .. } => counts.game_local += 1,
                ImportResolution::Stub { .. } => counts.stubs += 1,
                ImportResolution::UnknownOrdinal { .. } => counts.unknown_ordinals += 1,
            }
            imports.push(InspectedImport {
                dll: dll_group.dll.clone(),
                symbol: entry.symbol.to_string(),
                delay_load: dll_group.delay_load,
                iat_rva: format!("0x{:08x}", entry.iat_rva),
                resolution,
            });
        }
    }

    Ok(InspectReport {
        schema_version: REPORT_SCHEMA_VERSION,
        file: file_label.to_owned(),
        machine: format!("0x{:04x}", pe.machine),
        is_dll: pe.is_dll(),
        subsystem: pe.subsystem,
        image_base: format!("0x{:016x}", pe.image_base),
        size_of_image: format!("0x{:08x}", pe.size_of_image),
        entry_rva: format!("0x{:08x}", pe.entry_rva),
        stack_reserve: format!("0x{:x}", pe.stack_reserve),
        has_relocations: pe.has_relocations(),
        exception_entries: pe.exception_entry_count(),
        tls,
        console_dlls: console_set.into_iter().collect(),
        counts,
        imports,
    })
}

impl InspectReport {
    /// Serialize the inspection report to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Project unresolved stubs and unknown ordinals into a `report.json` [`TelemetryReport`].
    pub fn to_telemetry_report(&self, title_id: impl Into<String>) -> TelemetryReport {
        let mut telemetry = TelemetryReport::new(title_id);
        for item in &self.imports {
            let iat_rva =
                u64::from_str_radix(item.iat_rva.trim_start_matches("0x"), 16).unwrap_or(0);
            match &item.resolution {
                ImportResolution::Stub { resolved_name, .. } => {
                    telemetry.record_missing_import(&item.dll, resolved_name, iat_rva);
                }
                ImportResolution::UnknownOrdinal { ordinal, .. } => {
                    telemetry.record_unknown_ordinal(&item.dll, *ordinal, iat_rva);
                }
                _ => {}
            }
        }
        telemetry
    }
}

impl fmt::Display for InspectReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "File:           {}", self.file)?;
        writeln!(
            f,
            "Format:         PE32+ {} (machine {}, subsystem {})",
            if self.is_dll { "DLL" } else { "EXE" },
            self.machine,
            self.subsystem
        )?;
        writeln!(
            f,
            "ImageBase:      {} (size {}, entry {})",
            self.image_base, self.size_of_image, self.entry_rva
        )?;
        writeln!(f, "StackReserve:   {}", self.stack_reserve)?;
        writeln!(
            f,
            "Relocations:    {}",
            if self.has_relocations { "yes" } else { "no" }
        )?;
        match &self.tls {
            Some(tls) => writeln!(
                f,
                "Static TLS:     present at {} (template {} B, zero-fill {} B, total {} B, callbacks: {})",
                tls.directory_rva,
                tls.template_bytes,
                tls.zero_fill_bytes,
                tls.total_bytes,
                if tls.has_callbacks { "yes" } else { "no" }
            )?,
            None => writeln!(f, "Static TLS:     none")?,
        }
        writeln!(
            f,
            "Exceptions:     {} .pdata entries",
            self.exception_entries
        )?;
        if self.console_dlls.is_empty() {
            writeln!(f, "Console DLLs:   none detected")?;
        } else {
            writeln!(f, "Console DLLs:   {}", self.console_dlls.join(", "))?;
        }

        writeln!(
            f,
            "\nImports: {} total ({} implemented, {} system, {} local, {} stubs, {} unknown ordinals)",
            self.counts.total,
            self.counts.implemented,
            self.counts.system_fallback,
            self.counts.game_local,
            self.counts.stubs,
            self.counts.unknown_ordinals
        )?;

        let mut by_dll: BTreeMap<String, Vec<&InspectedImport>> = BTreeMap::new();
        for imp in &self.imports {
            let label = if imp.delay_load {
                format!("{} (delay-load)", imp.dll)
            } else {
                imp.dll.clone()
            };
            by_dll.entry(label).or_default().push(imp);
        }

        for (dll, list) in by_dll {
            let mut implemented = 0usize;
            let mut system = 0usize;
            let mut local = 0usize;
            let mut stubs = 0usize;
            let mut unknown_ords = 0usize;
            for imp in &list {
                match imp.resolution {
                    ImportResolution::Implemented { .. } => implemented += 1,
                    ImportResolution::SystemFallback { .. } => system += 1,
                    ImportResolution::GameLocal { .. } => local += 1,
                    ImportResolution::Stub { .. } => stubs += 1,
                    ImportResolution::UnknownOrdinal { .. } => unknown_ords += 1,
                }
            }
            let mut parts = Vec::new();
            if implemented > 0 {
                parts.push(format!("{implemented} implemented"));
            }
            if system > 0 {
                parts.push(format!("{system} system"));
            }
            if local > 0 {
                parts.push(format!("{local} local"));
            }
            if stubs > 0 {
                parts.push(format!("{stubs} stubs"));
            }
            if unknown_ords > 0 {
                parts.push(format!("{unknown_ords} unknown-ordinals"));
            }
            writeln!(f, "  {dll}: {}", parts.join(", "))?;
            for imp in list {
                match &imp.resolution {
                    ImportResolution::Stub {
                        resolved_name,
                        default_return,
                        ..
                    } => {
                        let label = if imp.symbol != *resolved_name {
                            format!("{} -> {}", imp.symbol, resolved_name)
                        } else {
                            resolved_name.clone()
                        };
                        writeln!(f, "      [STUB ret={default_return}] {label}")?;
                    }
                    ImportResolution::UnknownOrdinal {
                        ordinal,
                        default_return,
                        ..
                    } => {
                        writeln!(f, "      [UNKNOWN ORDINAL ret={default_return}] #{ordinal}")?;
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }
}

/// Helper to inspect symbol formatting in tests.
pub fn format_symbol(symbol: &ImportSymbol) -> String {
    symbol.to_string()
}
