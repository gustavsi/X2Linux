//! Offline package identity and monitor state for an unpacked installation.
//! This module also builds into the PE shim to keep opaque guest handles local.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;

pub const PACKAGE_IDENTIFIER: &CStr = c"X2Linux.GenericPackage_1.0.0.0_x64__0000000000000";
pub const PACKAGE_MOUNT_ROOT: &str = "G:\\";
pub const NO_PACKAGE: i32 = 0x8924_5208_u32 as i32;

pub fn is_local_package(identifier: &CStr) -> bool {
    identifier == PACKAGE_IDENTIFIER
        || identifier
            .to_bytes()
            .eq_ignore_ascii_case(PACKAGE_MOUNT_ROOT.as_bytes())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackageInstallationProgress {
    pub total_bytes: u64,
    pub installed_bytes: u64,
    pub launch_bytes: u64,
    pub launchable: bool,
    pub completed: bool,
}

const _: () = assert!(std::mem::size_of::<PackageInstallationProgress>() == 32);
const _: () = assert!(std::mem::offset_of!(PackageInstallationProgress, launchable) == 24);
const _: () = assert!(std::mem::offset_of!(PackageInstallationProgress, completed) == 25);

/// Installation is complete before X2Linux launches the process. Package manifests
/// and byte counts are unavailable for unpacked folders.
const INSTALLED: PackageInstallationProgress = PackageInstallationProgress {
    total_bytes: 0,
    installed_bytes: 0,
    launch_bytes: 0,
    launchable: true,
    completed: true,
};

/// Process-local IDs are never reused. Registrations track lifetime only because
/// an unpacked installation does not change during the session.
#[derive(Debug)]
pub struct PackageMonitors {
    next_id: usize,
    monitors: BTreeMap<usize, BTreeSet<u64>>,
    installed_registrations: BTreeSet<u64>,
}

impl Default for PackageMonitors {
    fn default() -> Self {
        Self {
            next_id: 0x0006_0000_0000,
            monitors: BTreeMap::new(),
            installed_registrations: BTreeSet::new(),
        }
    }
}

impl PackageMonitors {
    fn next_id(&mut self) -> usize {
        self.next_id += 1;
        self.next_id
    }

    pub fn create(&mut self, identifier: &CStr) -> Result<usize, i32> {
        if !is_local_package(identifier) {
            return Err(NO_PACKAGE);
        }
        let id = self.next_id();
        self.monitors.insert(id, BTreeSet::new());
        Ok(id)
    }

    pub fn close(&mut self, id: usize) -> bool {
        self.monitors.remove(&id).is_some()
    }

    pub fn progress(&self, id: usize) -> Option<PackageInstallationProgress> {
        self.monitors.contains_key(&id).then_some(INSTALLED)
    }

    pub fn register_progress(&mut self, id: usize) -> Option<u64> {
        if !self.monitors.contains_key(&id) {
            return None;
        }
        let token = self.next_id() as u64;
        self.monitors.get_mut(&id)?.insert(token);
        Some(token)
    }

    pub fn unregister_progress(&mut self, id: usize, token: u64) -> bool {
        self.monitors
            .get_mut(&id)
            .is_some_and(|tokens| tokens.remove(&token))
    }

    pub fn register_installed(&mut self) -> u64 {
        let token = self.next_id() as u64;
        self.installed_registrations.insert(token);
        token
    }

    pub fn unregister_installed(&mut self, token: u64) -> bool {
        self.installed_registrations.remove(&token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitors_accept_the_local_package_and_legacy_root_only() {
        let mut monitors = PackageMonitors::default();
        assert_eq!(monitors.create(c"other-package"), Err(NO_PACKAGE));
        assert_eq!(monitors.create(c""), Err(NO_PACKAGE));
        assert_eq!(monitors.create(c"G:\\subdirectory"), Err(NO_PACKAGE));
        let first = monitors.create(PACKAGE_IDENTIFIER).unwrap();
        let second = monitors.create(c"g:\\").unwrap();
        assert_ne!(first, second);
        let progress = monitors.progress(first).unwrap();
        assert!(progress.launchable && progress.completed);
        assert_eq!(progress.total_bytes, progress.installed_bytes);
        assert_eq!(progress.launch_bytes, progress.total_bytes);
        assert_eq!(monitors.progress(second), Some(progress));
        assert_eq!(monitors.progress(0), None);
    }

    #[test]
    fn closing_a_monitor_invalidates_it_without_affecting_other_monitors() {
        let mut monitors = PackageMonitors::default();
        let first = monitors.create(PACKAGE_IDENTIFIER).unwrap();
        let second = monitors.create(PACKAGE_IDENTIFIER).unwrap();
        let first_token = monitors.register_progress(first).unwrap();
        let second_token = monitors.register_progress(second).unwrap();
        assert_ne!(first_token, second_token);
        assert!(!monitors.unregister_progress(second, first_token));
        assert!(monitors.close(first));
        assert!(!monitors.close(first));
        assert_eq!(monitors.progress(first), None);
        assert_eq!(monitors.register_progress(first), None);
        assert!(!monitors.unregister_progress(first, first_token));
        assert!(monitors.unregister_progress(second, second_token));
        assert!(!monitors.unregister_progress(second, second_token));
        assert_eq!(monitors.progress(second), Some(INSTALLED));
        assert_ne!(monitors.create(PACKAGE_IDENTIFIER).unwrap(), first);
    }

    #[test]
    fn installed_and_progress_registrations_have_distinct_lifetimes() {
        let mut monitors = PackageMonitors::default();
        let monitor = monitors.create(PACKAGE_IDENTIFIER).unwrap();
        let progress = monitors.register_progress(monitor).unwrap();
        let installed = monitors.register_installed();
        assert_ne!(progress, installed);
        assert!(!monitors.unregister_installed(progress));
        assert!(!monitors.unregister_progress(monitor, installed));
        assert!(monitors.close(monitor));
        assert!(monitors.unregister_installed(installed));
        assert!(!monitors.unregister_installed(installed));
    }
}
