//! Data-driven import policy engine: API-set aliases, ordinal DBs, and default return rules.

use crate::pe::ImportSymbol;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Errors raised while loading alias or ordinal policy TOML files.
#[derive(Debug, Error)]
pub enum PolicyError {
    #[error("failed to read policy file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML in '{path}': {source}")]
    Toml {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid ordinal key '{key}' in '{path}': must be a u16 integer")]
    InvalidOrdinal { path: String, key: String },
}

/// Target route for a DLL after evaluating `data/aliases.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DllRoute {
    Hle(String),
    System(String),
}

/// Single resolved ordinal entry from `data/ordinals/*.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrdinalEntry {
    pub name: String,
    #[serde(default)]
    pub default_return: Option<String>,
}

/// Classification outcome for one imported symbol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImportResolution {
    Implemented {
        module: String,
        resolved_name: String,
    },
    SystemFallback {
        target: String,
        resolved_name: String,
    },
    GameLocal {
        path: String,
    },
    Stub {
        module: String,
        resolved_name: String,
        default_return: String,
    },
    UnknownOrdinal {
        module: String,
        ordinal: u16,
        default_return: String,
    },
}

/// Loaded import resolution policy combining `data/aliases.toml` and `data/ordinals/*.toml`.
#[derive(Debug, Clone, Default)]
pub struct ImportPolicy {
    exact_aliases: BTreeMap<String, DllRoute>,
    prefix_aliases: Vec<(String, DllRoute)>,
    module_defaults: BTreeMap<String, String>,
    implemented_symbols: BTreeMap<String, BTreeSet<String>>,
    ordinals: BTreeMap<String, BTreeMap<u16, OrdinalEntry>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct AliasesFile {
    exact: BTreeMap<String, String>,
    prefix: BTreeMap<String, String>,
    module_defaults: BTreeMap<String, String>,
    implemented: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct OrdinalsFile {
    dll: Option<String>,
    ordinals: BTreeMap<String, OrdinalEntry>,
}

/// Known Wine/Windows system DLLs that resolve through the NT floor when no HLE rule overrides them.
const SYSTEM_DLLS: &[&str] = &[
    "kernel32.dll",
    "kernelbase.dll",
    "ntdll.dll",
    "ucrtbase.dll",
    "msvcrt.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "user32.dll",
    "gdi32.dll",
    "advapi32.dll",
    "shell32.dll",
    "ole32.dll",
    "oleaut32.dll",
    "bcrypt.dll",
    "bcryptprimitives.dll",
    "ws2_32.dll",
    "winmm.dll",
    "rpcrt4.dll",
    "sechost.dll",
    "shlwapi.dll",
    "crypt32.dll",
    "userenv.dll",
    "winhttp.dll",
    "mfplat.dll",
    "mfreadwrite.dll",
    "xaudio2_9.dll",
];

impl ImportPolicy {
    /// Built-in default policy matching the repository's `data/aliases.toml`.
    pub fn builtin_defaults() -> Self {
        let mut policy = Self::default();
        let _ = policy.load_aliases_str(
            include_str!("../../../data/aliases.toml"),
            "<builtin:aliases.toml>",
        );
        let _ = policy.load_ordinals_str(
            "kernelx.dll",
            include_str!("../../../data/ordinals/kernelx.toml"),
            "<builtin:kernelx.toml>",
        );
        let _ = policy.load_ordinals_str(
            "gameinput.dll",
            include_str!("../../../data/ordinals/gameinput.toml"),
            "<builtin:gameinput.toml>",
        );
        let _ = policy.load_ordinals_str(
            "xaudio2_9.dll",
            include_str!("../../../data/ordinals/xaudio2_9.toml"),
            "<builtin:xaudio2_9.toml>",
        );
        policy
    }

    /// Load `aliases.toml` and every `*.toml` inside `ordinals_dir`.
    pub fn load_from_paths(
        aliases_path: Option<&Path>,
        ordinals_dir: Option<&Path>,
    ) -> Result<Self, PolicyError> {
        let mut policy = Self::default();
        match aliases_path {
            Some(path) => {
                let raw = fs::read_to_string(path).map_err(|source| PolicyError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
                policy.load_aliases_str(&raw, &path.display().to_string())?;
            }
            None => {
                policy = Self::builtin_defaults();
            }
        }

        if let Some(dir) = ordinals_dir {
            let entries = fs::read_dir(dir).map_err(|source| PolicyError::Io {
                path: dir.display().to_string(),
                source,
            })?;
            let mut files = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                    files.push(path);
                }
            }
            files.sort();
            for path in files {
                let raw = fs::read_to_string(&path).map_err(|source| PolicyError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
                let fallback_dll = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|stem| normalize_dll_name(&format!("{stem}.dll")))
                    .unwrap_or_default();
                policy.load_ordinals_str(&fallback_dll, &raw, &path.display().to_string())?;
            }
        }

        Ok(policy)
    }

    /// Parse alias rules from TOML text.
    pub fn load_aliases_str(&mut self, input: &str, label: &str) -> Result<(), PolicyError> {
        let parsed: AliasesFile = toml::from_str(input).map_err(|source| PolicyError::Toml {
            path: label.to_owned(),
            source,
        })?;

        for (dll, target) in parsed.exact {
            if let Some(route) = parse_route(&target) {
                self.exact_aliases.insert(normalize_dll_name(&dll), route);
            }
        }

        for (prefix, target) in parsed.prefix {
            if let Some(route) = parse_route(&target) {
                let key = prefix.to_ascii_lowercase();
                if let Some(existing) = self.prefix_aliases.iter_mut().find(|(p, _)| p == &key) {
                    existing.1 = route;
                } else {
                    self.prefix_aliases.push((key, route));
                }
            }
        }
        // Match longest prefix first for deterministic API-set resolution.
        self.prefix_aliases
            .sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));

        for (dll, ret) in parsed.module_defaults {
            self.module_defaults.insert(normalize_dll_name(&dll), ret);
        }

        for (module, symbols) in parsed.implemented {
            let key = module.to_ascii_lowercase();
            let set = self.implemented_symbols.entry(key).or_default();
            for sym in symbols {
                set.insert(sym);
            }
        }

        Ok(())
    }

    /// Parse ordinal mappings from TOML text.
    pub fn load_ordinals_str(
        &mut self,
        default_dll: &str,
        input: &str,
        label: &str,
    ) -> Result<(), PolicyError> {
        let parsed: OrdinalsFile = toml::from_str(input).map_err(|source| PolicyError::Toml {
            path: label.to_owned(),
            source,
        })?;
        let dll = normalize_dll_name(parsed.dll.as_deref().unwrap_or(default_dll));
        let table = self.ordinals.entry(dll).or_default();

        for (ord_str, entry) in parsed.ordinals {
            let ordinal: u16 = ord_str.parse().map_err(|_| PolicyError::InvalidOrdinal {
                path: label.to_owned(),
                key: ord_str,
            })?;
            table.insert(ordinal, entry);
        }
        Ok(())
    }

    /// Resolve a DLL name through exact and prefix alias tables.
    pub fn route_dll(&self, dll: &str) -> Option<&DllRoute> {
        let key = normalize_dll_name(dll);
        if let Some(route) = self.exact_aliases.get(&key) {
            return Some(route);
        }
        for (prefix, route) in &self.prefix_aliases {
            if key.starts_with(prefix) {
                return Some(route);
            }
        }
        None
    }

    /// Look up an ordinal mapping for a DLL (normalized lowercase).
    pub fn lookup_ordinal(&self, dll: &str, ordinal: u16) -> Option<&OrdinalEntry> {
        let key = normalize_dll_name(dll);
        self.ordinals.get(&key).and_then(|m| m.get(&ordinal))
    }

    /// Determine whether a DLL routes to an X2Linux console HLE module.
    pub fn is_console_dll(&self, dll: &str) -> bool {
        matches!(self.route_dll(dll), Some(DllRoute::Hle(_)))
    }

    /// Default stub return value configured for a DLL (defaults to `"0x0"`).
    pub fn default_return_for(&self, dll: &str) -> String {
        let key = normalize_dll_name(dll);
        self.module_defaults
            .get(&key)
            .cloned()
            .unwrap_or_else(|| "0x0".to_owned())
    }

    /// Classify a single imported symbol using alias rules, ordinal DB, optional title directory
    /// scan (strictly read-only), and system DLL fallback rules.
    pub fn classify(
        &self,
        dll: &str,
        symbol: &ImportSymbol,
        title_dir: Option<&Path>,
    ) -> ImportResolution {
        let dll_key = normalize_dll_name(dll);
        let module_default = self.default_return_for(&dll_key);

        if let Some(route) = self.route_dll(&dll_key) {
            return match route {
                DllRoute::Hle(module) => match symbol {
                    ImportSymbol::Name { name, .. } => {
                        if self.is_implemented(module, name) {
                            ImportResolution::Implemented {
                                module: module.clone(),
                                resolved_name: name.clone(),
                            }
                        } else {
                            ImportResolution::Stub {
                                module: module.clone(),
                                resolved_name: name.clone(),
                                default_return: module_default,
                            }
                        }
                    }
                    ImportSymbol::Ordinal { ordinal } => {
                        if let Some(entry) = self.lookup_ordinal(&dll_key, *ordinal) {
                            if self.is_implemented(module, &entry.name) {
                                ImportResolution::Implemented {
                                    module: module.clone(),
                                    resolved_name: entry.name.clone(),
                                }
                            } else {
                                ImportResolution::Stub {
                                    module: module.clone(),
                                    resolved_name: entry.name.clone(),
                                    default_return: entry
                                        .default_return
                                        .clone()
                                        .unwrap_or(module_default),
                                }
                            }
                        } else {
                            ImportResolution::UnknownOrdinal {
                                module: module.clone(),
                                ordinal: *ordinal,
                                default_return: module_default,
                            }
                        }
                    }
                },
                DllRoute::System(target) => ImportResolution::SystemFallback {
                    target: target.clone(),
                    resolved_name: self.display_symbol(&dll_key, symbol),
                },
            };
        }

        if let Some(dir) = title_dir
            && let Some(local_path) = find_local_dll_readonly(dir, &dll_key)
        {
            return ImportResolution::GameLocal { path: local_path };
        }

        if SYSTEM_DLLS.contains(&dll_key.as_str()) {
            return ImportResolution::SystemFallback {
                target: dll_key.clone(),
                resolved_name: self.display_symbol(&dll_key, symbol),
            };
        }

        match symbol {
            ImportSymbol::Name { name, .. } => ImportResolution::Stub {
                module: dll_key,
                resolved_name: name.clone(),
                default_return: module_default,
            },
            ImportSymbol::Ordinal { ordinal } => {
                if let Some(entry) = self.lookup_ordinal(&dll_key, *ordinal) {
                    ImportResolution::Stub {
                        module: dll_key,
                        resolved_name: entry.name.clone(),
                        default_return: entry.default_return.clone().unwrap_or(module_default),
                    }
                } else {
                    ImportResolution::UnknownOrdinal {
                        module: dll_key,
                        ordinal: *ordinal,
                        default_return: module_default,
                    }
                }
            }
        }
    }

    fn is_implemented(&self, module: &str, symbol: &str) -> bool {
        self.implemented_symbols
            .get(&module.to_ascii_lowercase())
            .is_some_and(|set| set.contains(symbol))
    }

    fn display_symbol(&self, dll_key: &str, symbol: &ImportSymbol) -> String {
        match symbol {
            ImportSymbol::Name { name, .. } => name.clone(),
            ImportSymbol::Ordinal { ordinal } => self
                .lookup_ordinal(dll_key, *ordinal)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| format!("#{ordinal}")),
        }
    }
}

fn parse_route(spec: &str) -> Option<DllRoute> {
    if let Some(module) = spec.strip_prefix("hle:") {
        Some(DllRoute::Hle(module.to_owned()))
    } else {
        spec.strip_prefix("system:")
            .map(|target| DllRoute::System(target.to_owned()))
    }
}

pub(crate) fn normalize_dll_name(dll: &str) -> String {
    dll.trim().to_ascii_lowercase()
}

fn find_local_dll_readonly(dir: &Path, dll_lower: &str) -> Option<String> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if name.to_string_lossy().to_ascii_lowercase() == dll_lower && entry.path().is_file() {
            return Some(entry.path().display().to_string());
        }
    }
    None
}
