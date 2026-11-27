//! Offline `XGameRuntime` HLE: `XGameRuntimeInitialize`, local `XUser`, `XPackage` (`G:\`),
//! `XGameSave` (`saves/`), and strictly offline `XStore`.
//!
//! X2Linux never implements DRM decryption or `XStore` license spoofing.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use x2l_common::ConsoleConfig;
use x2l_common::bridge::{HRESULT_E_NOTIMPL, HRESULT_S_OK};

/// Offline local user profile derived from `x2linux.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalUserProfile {
    pub gamertag: String,
}

impl From<&ConsoleConfig> for LocalUserProfile {
    fn from(cfg: &ConsoleConfig) -> Self {
        Self {
            gamertag: cfg.gamertag.clone(),
        }
    }
}

/// Stateful offline `XGameRuntime` HLE context.
#[derive(Debug)]
pub struct XgrState {
    initialized: AtomicBool,
    user: LocalUserProfile,
    saves_dir: PathBuf,
    mount_root: String,
}

impl XgrState {
    pub fn new(gamertag: impl Into<String>, saves_dir: impl Into<PathBuf>) -> Self {
        Self {
            initialized: AtomicBool::new(false),
            user: LocalUserProfile {
                gamertag: gamertag.into(),
            },
            saves_dir: saves_dir.into(),
            mount_root: "G:\\".to_owned(),
        }
    }

    pub fn initialize(&self) -> u64 {
        self.initialized.store(true, Ordering::SeqCst);
        let _ = fs::create_dir_all(&self.saves_dir);
        HRESULT_S_OK
    }

    pub fn uninitialize(&self) {
        self.initialized.store(false, Ordering::SeqCst);
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    pub fn gamertag(&self) -> &str {
        &self.user.gamertag
    }

    pub fn package_mount_root(&self) -> &str {
        &self.mount_root
    }

    pub fn saves_dir(&self) -> &Path {
        &self.saves_dir
    }

    /// `XStore` is intentionally offline and never spoofs commercial licenses.
    pub fn store_offline_result(&self) -> u64 {
        HRESULT_E_NOTIMPL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_xgr_lifecycle_and_store_refusal() {
        let dir = std::env::temp_dir().join(format!("x2l-xgr-test-{}", std::process::id()));
        let state = XgrState::new("LinuxPlayer", &dir);
        assert!(!state.is_initialized());
        assert_eq!(state.initialize(), HRESULT_S_OK);
        assert!(state.is_initialized());
        assert_eq!(state.gamertag(), "LinuxPlayer");
        assert_eq!(state.package_mount_root(), "G:\\");
        assert_eq!(state.store_offline_result(), HRESULT_E_NOTIMPL);
        state.uninitialize();
        assert!(!state.is_initialized());
        let _ = fs::remove_dir_all(&dir);
    }
}
