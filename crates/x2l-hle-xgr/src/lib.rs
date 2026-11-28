// Offline `XGameRuntime` HLE and GDK Win64 ABI semantics informed by Marvisak/VIctory
// (https://github.com/Marvisak/VIctory, commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-hle/src/modules/{xgameruntime,xasync,xsystem,xuser,xpackage,xstore,xgame}.rs)
// and xcl-dev/xcl (https://github.com/xcl-dev/xcl, commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102,
// crates/xcl/src/hle/xgameruntime/).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-28: clean-room Linux XDG persistence for `XGameSave`,
// `XPersistentLocalStorage`, offline `XUser`, `XPackage`, and strictly offline `XStore`.

//! Offline `XGameRuntime` HLE: `XGameRuntimeInitialize`, system/device identity,
//! `XPersistentLocalStorage`, local `XUser`, `XPackage` (`G:\`), atomic XDG `XGameSave`
//! containers/blobs (`saves/`), and strictly offline `XStore`.
//!
//! X2Linux never implements DRM decryption or `XStore` license spoofing.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use x2l_common::bridge::{HRESULT_E_NOTIMPL, HRESULT_S_OK};
use x2l_common::{ConsoleConfig, DeviceType};

/// Offline local user profile derived from `x2linux.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalUserProfile {
    pub gamertag: String,
    pub xuid: u64,
    pub local_id: u64,
}

impl From<&ConsoleConfig> for LocalUserProfile {
    fn from(cfg: &ConsoleConfig) -> Self {
        Self {
            gamertag: cfg.gamertag.clone(),
            xuid: 0x0009_0000_0000_0001,
            local_id: 1,
        }
    }
}

/// 32-byte `XPersistentLocalStorageSpaceInfo` layout matching GDK Win64 ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PersistentLocalStorageSpaceInfo {
    pub available_free_bytes: u64, // @0
    pub total_free_bytes: u64,     // @8
    pub used_bytes: u64,           // @16
    pub total_bytes: u64,          // @24
}

const _: () = assert!(std::mem::size_of::<PersistentLocalStorageSpaceInfo>() == 32);

/// One enumerated `XGameSave` blob entry `(name, size_bytes)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveBlobEntry {
    pub name: String,
    pub size: u32,
}

/// Stateful offline `XGameRuntime` HLE context owned by `libx2l_core.so`.
#[derive(Debug)]
pub struct XgrState {
    initialized: AtomicBool,
    device: DeviceType,
    title_id: u32,
    user: LocalUserProfile,
    saves_dir: PathBuf,
    mount_root: String,
    pls_guest_path: String,
    console_id: String,
    sandbox_id: String,
    app_device_id: String,
}

impl XgrState {
    pub fn new(gamertag: impl Into<String>, saves_dir: impl Into<PathBuf>) -> Self {
        Self::with_device_and_title(gamertag, saves_dir, DeviceType::SeriesX, "default-title")
    }

    pub fn with_device_and_title(
        gamertag: impl Into<String>,
        saves_dir: impl Into<PathBuf>,
        device: DeviceType,
        title_slug: &str,
    ) -> Self {
        // Deterministic non-zero synthetic 32-bit title ID derived from title slug.
        let mut h: u32 = 0x811C_9DC5;
        for &b in title_slug.as_bytes() {
            h ^= u32::from(b);
            h = h.wrapping_mul(0x0100_0193);
        }
        let title_id = if h == 0 { 0x1000_0001 } else { h };

        Self {
            initialized: AtomicBool::new(false),
            device,
            title_id,
            user: LocalUserProfile {
                gamertag: gamertag.into(),
                xuid: 0x0009_0000_0000_0001,
                local_id: 1,
            },
            saves_dir: saves_dir.into(),
            mount_root: "G:\\".to_owned(),
            pls_guest_path: "S:\\pls".to_owned(),
            console_id: "00000000-0000-0000-0000-000000000001".to_owned(),
            sandbox_id: "RETAIL".to_owned(),
            app_device_id: "X2LINUX-LOCAL-DEVICE-0001".to_owned(),
        }
    }

    pub fn initialize(&self) -> u64 {
        self.initialized.store(true, Ordering::SeqCst);
        let _ = fs::create_dir_all(&self.saves_dir);
        let _ = fs::create_dir_all(self.saves_dir.join("pls"));
        let _ = fs::create_dir_all(self.saves_dir.join("containers"));
        HRESULT_S_OK
    }

    pub fn uninitialize(&self) {
        self.initialized.store(false, Ordering::SeqCst);
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Standard GDK `XSystemDeviceType` enum value (`3 = Lockhart / Series S`, `4 = Anaconda / Series X`).
    pub fn gdk_device_type(&self) -> u32 {
        match self.device {
            DeviceType::SeriesS => 3,
            DeviceType::SeriesX => 4,
        }
    }

    pub fn xbox_title_id(&self) -> u32 {
        self.title_id
    }

    pub fn console_id(&self) -> &str {
        &self.console_id
    }

    pub fn sandbox_id(&self) -> &str {
        &self.sandbox_id
    }

    pub fn app_specific_device_id(&self) -> &str {
        &self.app_device_id
    }

    pub fn gamertag(&self) -> &str {
        &self.user.gamertag
    }

    pub fn user_xuid(&self) -> u64 {
        self.user.xuid
    }

    pub fn user_local_id(&self) -> u64 {
        self.user.local_id
    }

    pub fn package_mount_root(&self) -> &str {
        &self.mount_root
    }

    pub fn pls_guest_path(&self) -> &str {
        &self.pls_guest_path
    }

    pub fn saves_dir(&self) -> &Path {
        &self.saves_dir
    }

    /// Compute `XPersistentLocalStorageSpaceInfo` (4 GiB virtual PLS quota minus actual `saves/pls` usage).
    pub fn pls_space_info(&self) -> PersistentLocalStorageSpaceInfo {
        const TOTAL_QUOTA: u64 = 4 * 1024 * 1024 * 1024;
        let pls_dir = self.saves_dir.join("pls");
        let _ = fs::create_dir_all(&pls_dir);
        let used = dir_size_bytes(&pls_dir).min(TOTAL_QUOTA);
        let free = TOTAL_QUOTA.saturating_sub(used);
        PersistentLocalStorageSpaceInfo {
            available_free_bytes: free,
            total_free_bytes: free,
            used_bytes: used,
            total_bytes: TOTAL_QUOTA,
        }
    }

    /// Atomically persist an `XGameSave` blob under `$XDG_DATA_HOME/x2linux/<title>/saves/containers/<container>/<blob>`.
    pub fn write_save_blob(
        &self,
        container: &str,
        blob_name: &str,
        data: &[u8],
    ) -> std::io::Result<()> {
        let dir = self.container_dir(container)?;
        fs::create_dir_all(&dir)?;
        let safe_blob = sanitize_name(blob_name)?;
        let final_path = dir.join(&safe_blob);
        let tmp_path = dir.join(format!(".{safe_blob}.tmp"));
        fs::write(&tmp_path, data)?;
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Read an `XGameSave` blob from XDG saves storage.
    pub fn read_save_blob(&self, container: &str, blob_name: &str) -> std::io::Result<Vec<u8>> {
        let dir = self.container_dir(container)?;
        let safe_blob = sanitize_name(blob_name)?;
        fs::read(dir.join(safe_blob))
    }

    /// Enumerate blobs inside a save container (optionally filtering by `prefix`).
    pub fn enumerate_save_blobs(
        &self,
        container: &str,
        prefix: Option<&str>,
    ) -> std::io::Result<Vec<SaveBlobEntry>> {
        let dir = self.container_dir(container)?;
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        for item in fs::read_dir(&dir)? {
            let item = item?;
            let fname = item.file_name().to_string_lossy().into_owned();
            if fname.starts_with('.') {
                continue;
            }
            if let Some(pfx) = prefix
                && !pfx.is_empty()
                && !fname.starts_with(pfx)
            {
                continue;
            }
            if let Ok(meta) = item.metadata()
                && meta.is_file()
            {
                entries.push(SaveBlobEntry {
                    name: fname,
                    size: meta.len().min(u32::MAX as u64) as u32,
                });
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    fn container_dir(&self, container: &str) -> std::io::Result<PathBuf> {
        let safe = sanitize_name(container)?;
        Ok(self.saves_dir.join("containers").join(safe))
    }

    /// `XStore` is intentionally offline and never spoofs commercial licenses.
    pub fn store_offline_result(&self) -> u64 {
        HRESULT_E_NOTIMPL
    }
}

fn sanitize_name(raw: &str) -> std::io::Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.contains("..")
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains('\0')
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid save container or blob name",
        ));
    }
    Ok(trimmed.to_owned())
}

fn dir_size_bytes(path: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in rd.flatten() {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                total = total.saturating_add(meta.len());
            } else if meta.is_dir() {
                total = total.saturating_add(dir_size_bytes(&entry.path()));
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_xgr_lifecycle_pls_gamesave_and_store_refusal() {
        let dir = std::env::temp_dir().join(format!("x2l-xgr-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let state = XgrState::with_device_and_title(
            "LinuxPlayer",
            &dir,
            DeviceType::SeriesX,
            "synthetic-xgr-title",
        );
        assert!(!state.is_initialized());
        assert_eq!(state.initialize(), HRESULT_S_OK);
        assert!(state.is_initialized());
        assert_eq!(state.gdk_device_type(), 4);
        assert_ne!(state.xbox_title_id(), 0);
        assert_eq!(state.gamertag(), "LinuxPlayer");
        assert_eq!(state.package_mount_root(), "G:\\");
        assert_eq!(state.pls_guest_path(), "S:\\pls");
        assert_eq!(state.store_offline_result(), HRESULT_E_NOTIMPL);

        let space = state.pls_space_info();
        assert_eq!(space.total_bytes, 4 * 1024 * 1024 * 1024);
        assert_eq!(space.available_free_bytes, space.total_bytes);

        state
            .write_save_blob("slot0", "profile.bin", b"synthetic-save-payload")
            .expect("write save blob");
        assert_eq!(
            state
                .read_save_blob("slot0", "profile.bin")
                .expect("read save blob"),
            b"synthetic-save-payload"
        );
        let list = state
            .enumerate_save_blobs("slot0", Some("prof"))
            .expect("enum save blobs");
        assert_eq!(
            list,
            vec![SaveBlobEntry {
                name: "profile.bin".into(),
                size: 22
            }]
        );
        assert!(state.write_save_blob("../escape", "bad", b"x").is_err());

        state.uninitialize();
        assert!(!state.is_initialized());
        let _ = fs::remove_dir_all(&dir);
    }
}
