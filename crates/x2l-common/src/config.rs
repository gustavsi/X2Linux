//! Strongly-typed `x2linux.toml` configuration schema.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Errors raised while loading or saving `x2linux.toml`.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML configuration: {0}")]
    Parse(#[from] toml::de::Error),
}

/// Top-level per-title configuration (`~/.config/x2linux/<title>/x2linux.toml`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct X2LinuxConfig {
    pub console: ConsoleConfig,
    pub runtime: RuntimeConfig,
    pub gpu: GpuConfig,
    pub audio: AudioConfig,
    pub logging: LoggingConfig,
    pub stubs: StubsConfig,
}

impl X2LinuxConfig {
    /// Parse configuration from a TOML string.
    pub fn from_toml_str(input: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(input)?)
    }

    /// Load configuration from a file on disk.
    pub fn load_file(path: &Path) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_toml_str(&content)
    }
}

/// Target console hardware profile.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    #[default]
    SeriesX,
    SeriesS,
}

/// `[console]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsoleConfig {
    pub device_type: DeviceType,
    pub gamertag: String,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::SeriesX,
            gamertag: "LinuxPlayer".to_owned(),
        }
    }
}

/// OverlayFS mount selection mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayFsMode {
    #[default]
    Auto,
    Kernel,
    Fuse,
}

/// `[runtime]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    pub use_ntsync: bool,
    pub overlayfs_mode: OverlayFsMode,
    pub wine_binary: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            use_ntsync: true,
            overlayfs_mode: OverlayFsMode::Auto,
            wine_binary: "wine".to_owned(),
        }
    }
}

/// GPU backend selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    #[default]
    Null,
    Vulkan,
}

/// `[gpu]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    pub backend: GpuBackend,
    pub validation: bool,
    pub prefer_rebar_zero_copy: bool,
    pub capture_shaders: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Null,
            validation: false,
            prefer_rebar_zero_copy: true,
            capture_shaders: true,
        }
    }
}

/// Native Linux audio backend selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioBackend {
    #[default]
    Pipewire,
    Sdl3,
}

/// `[audio]` section.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub backend: AudioBackend,
}

/// `[logging]` section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub level: String,
    pub write_report_json: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_owned(),
            write_report_json: true,
        }
    }
}

/// `[stubs]` section (default return overrides keyed by `module.dll!Symbol`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StubsConfig {
    pub overrides: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_example_toml() {
        let example_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("x2linux.example.toml");
        let cfg = X2LinuxConfig::load_file(&example_path).expect("parse x2linux.example.toml");

        assert_eq!(cfg.console.device_type, DeviceType::SeriesX);
        assert_eq!(cfg.console.gamertag, "LinuxPlayer");
        assert!(cfg.runtime.use_ntsync);
        assert_eq!(cfg.runtime.overlayfs_mode, OverlayFsMode::Auto);
        assert_eq!(cfg.gpu.backend, GpuBackend::Null);
        assert!(cfg.gpu.prefer_rebar_zero_copy);
        assert_eq!(cfg.audio.backend, AudioBackend::Pipewire);
        assert!(cfg.logging.write_report_json);
    }

    #[test]
    fn parses_stub_overrides() {
        let input = r#"
            [console]
            device_type = "series_s"

            [stubs.overrides]
            "xgameruntime.dll!XStoreQueryGameLicenseAsync" = "0x80004001"
        "#;
        let cfg = X2LinuxConfig::from_toml_str(input).expect("parse overrides");
        assert_eq!(cfg.console.device_type, DeviceType::SeriesS);
        assert_eq!(
            cfg.stubs
                .overrides
                .get("xgameruntime.dll!XStoreQueryGameLicenseAsync")
                .map(String::as_str),
            Some("0x80004001")
        );
    }
}
