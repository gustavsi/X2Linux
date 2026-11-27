//! Offline `XGameRuntime` HLE: `XTaskQueue`, `XAsync`, local `XUser`, `XGameSave`, and `XPersistentLocalStorage`.
//!
//! X2Linux never implements DRM decryption or `XStore` license spoofing.

use x2l_common::ConsoleConfig;

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
