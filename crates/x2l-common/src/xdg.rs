//! Linux XDG Base Directory resolution for X2Linux.
//!
//! Default directories:
//! - Config: `~/.config/x2linux` (`$XDG_CONFIG_HOME/x2linux`)
//! - Data:   `~/.local/share/x2linux` (`$XDG_DATA_HOME/x2linux`)
//! - Cache:  `~/.cache/x2linux` (`$XDG_CACHE_HOME/x2linux`)

use std::path::PathBuf;
use thiserror::Error;

const APP_DIR_NAME: &str = "x2linux";

/// Errors raised while resolving XDG directories or validating title identifiers.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum XdgError {
    #[error("unable to determine user home directory ($HOME is unset or empty)")]
    MissingHome,
    #[error("invalid title identifier '{0}': must be non-empty and contain only [A-Za-z0-9._-]")]
    InvalidTitleId(String),
}

/// Resolved Linux XDG directories for X2Linux.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdgPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

/// Per-title writable directories backing the read-only OverlayFS mount and console storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TitleOverlayDirs {
    pub root: PathBuf,
    pub upperdir: PathBuf,
    pub workdir: PathBuf,
    pub merged: PathBuf,
    pub saves: PathBuf,
    pub pls: PathBuf,
    pub temp: PathBuf,
}

impl XdgPaths {
    /// Resolve XDG paths from the current process environment.
    pub fn resolve() -> Result<Self, XdgError> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Resolve XDG paths using a custom environment lookup function (deterministic for tests).
    pub fn from_lookup<F>(lookup: F) -> Result<Self, XdgError>
    where
        F: Fn(&str) -> Option<String>,
    {
        let home = lookup("HOME").filter(|v| !v.is_empty()).map(PathBuf::from);

        let config_base = non_empty_abs(&lookup("XDG_CONFIG_HOME"))
            .or_else(|| home.as_ref().map(|h| h.join(".config")))
            .ok_or(XdgError::MissingHome)?;

        let data_base = non_empty_abs(&lookup("XDG_DATA_HOME"))
            .or_else(|| home.as_ref().map(|h| h.join(".local/share")))
            .ok_or(XdgError::MissingHome)?;

        let cache_base = non_empty_abs(&lookup("XDG_CACHE_HOME"))
            .or_else(|| home.as_ref().map(|h| h.join(".cache")))
            .ok_or(XdgError::MissingHome)?;

        Ok(Self {
            config_dir: config_base.join(APP_DIR_NAME),
            data_dir: data_base.join(APP_DIR_NAME),
            cache_dir: cache_base.join(APP_DIR_NAME),
        })
    }

    /// Return the per-title config path: `~/.config/x2linux/<title>/x2linux.toml`.
    pub fn title_config_file(&self, title_id: &str) -> Result<PathBuf, XdgError> {
        validate_title_id(title_id)?;
        Ok(self.config_dir.join(title_id).join("x2linux.toml"))
    }

    /// Return the per-title writable OverlayFS and storage directories under `~/.local/share/x2linux/<title>/`.
    pub fn title_overlay_dirs(&self, title_id: &str) -> Result<TitleOverlayDirs, XdgError> {
        validate_title_id(title_id)?;
        let root = self.data_dir.join(title_id);
        Ok(TitleOverlayDirs {
            upperdir: root.join("overlay"),
            workdir: root.join("work"),
            merged: root.join("merged"),
            saves: root.join("saves"),
            pls: root.join("pls"),
            temp: root.join("temp"),
            root,
        })
    }

    /// Return the per-title cache directory: `~/.cache/x2linux/<title>/`.
    pub fn title_cache_dir(&self, title_id: &str) -> Result<PathBuf, XdgError> {
        validate_title_id(title_id)?;
        Ok(self.cache_dir.join(title_id))
    }

    /// Return the per-title raw shader capture directory: `~/.cache/x2linux/<title>/shaders_raw/`.
    pub fn title_shaders_raw_dir(&self, title_id: &str) -> Result<PathBuf, XdgError> {
        Ok(self.title_cache_dir(title_id)?.join("shaders_raw"))
    }

    /// Return the per-title compiled SPIR-V cache directory: `~/.cache/x2linux/<title>/spirv/`.
    pub fn title_spirv_cache_dir(&self, title_id: &str) -> Result<PathBuf, XdgError> {
        Ok(self.title_cache_dir(title_id)?.join("spirv"))
    }
}

fn non_empty_abs(val: &Option<String>) -> Option<PathBuf> {
    let raw = val.as_deref()?.trim();
    if raw.is_empty() {
        return None;
    }
    let path = PathBuf::from(raw);
    if path.is_absolute() { Some(path) } else { None }
}

/// Validate a title identifier used as a single path component under XDG roots.
pub fn validate_title_id(title_id: &str) -> Result<(), XdgError> {
    let valid = !title_id.is_empty()
        && title_id != "."
        && title_id != ".."
        && title_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if valid {
        Ok(())
    } else {
        Err(XdgError::InvalidTitleId(title_id.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn resolves_default_home_xdg_paths() {
        let env = HashMap::from([("HOME", "/home/tester")]);
        let paths = XdgPaths::from_lookup(|k| env.get(k).map(|v| (*v).to_owned())).unwrap();

        assert_eq!(
            paths.config_dir,
            PathBuf::from("/home/tester/.config/x2linux")
        );
        assert_eq!(
            paths.data_dir,
            PathBuf::from("/home/tester/.local/share/x2linux")
        );
        assert_eq!(
            paths.cache_dir,
            PathBuf::from("/home/tester/.cache/x2linux")
        );

        assert_eq!(
            paths.title_config_file("synthetic-title").unwrap(),
            PathBuf::from("/home/tester/.config/x2linux/synthetic-title/x2linux.toml")
        );
        assert_eq!(
            paths.title_spirv_cache_dir("synthetic-title").unwrap(),
            PathBuf::from("/home/tester/.cache/x2linux/synthetic-title/spirv")
        );
    }

    #[test]
    fn respects_explicit_xdg_overrides() {
        let env = HashMap::from([
            ("HOME", "/home/tester"),
            ("XDG_CONFIG_HOME", "/srv/cfg"),
            ("XDG_DATA_HOME", "/srv/data"),
            ("XDG_CACHE_HOME", "/srv/cache"),
        ]);
        let paths = XdgPaths::from_lookup(|k| env.get(k).map(|v| (*v).to_owned())).unwrap();

        assert_eq!(paths.config_dir, PathBuf::from("/srv/cfg/x2linux"));
        assert_eq!(paths.data_dir, PathBuf::from("/srv/data/x2linux"));
        assert_eq!(paths.cache_dir, PathBuf::from("/srv/cache/x2linux"));
    }

    #[test]
    fn rejects_path_traversal_title_ids() {
        let env = HashMap::from([("HOME", "/home/tester")]);
        let paths = XdgPaths::from_lookup(|k| env.get(k).map(|v| (*v).to_owned())).unwrap();

        assert!(paths.title_config_file("").is_err());
        assert!(paths.title_config_file("..").is_err());
        assert!(paths.title_config_file("../escape").is_err());
        assert!(paths.title_overlay_dirs("nested/path").is_err());
    }
}
