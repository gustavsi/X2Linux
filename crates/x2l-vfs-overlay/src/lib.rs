//! Kernel-enforced read-only copy-on-write VFS isolation for X2Linux.
//!
//! Uses an unprivileged Linux user + mount namespace (`CLONE_NEWUSER | CLONE_NEWNS`) with
//! Kernel OverlayFS, falling back to `fuse-overlayfs` when unprivileged kernel overlay mounts
//! are unavailable. The user's title directory is mounted exclusively as `lowerdir` and is
//! never modified.

use nix::mount::{MntFlags, MsFlags, mount, umount2};
use nix::sched::{CloneFlags, unshare};
use nix::unistd::{Gid, Uid};
use std::fs;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;
use tracing::{info, warn};

/// Active network isolation mode for a guest runtime session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkIsolationMode {
    /// Kernel-enforced isolated `CLONE_NEWNET` network namespace (loopback-only; zero external routes).
    IsolatedNetns,
    /// Host restricted unprivileged network namespaces; outbound HTTP/HTTPS routed to a local dead-sink proxy.
    ProxySinkFallback,
    /// Explicit `--allow-network` / `offline = false` opt-in; host network inherited.
    Unrestricted,
}

impl NetworkIsolationMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IsolatedNetns => "isolated_netns",
            Self::ProxySinkFallback => "proxy_sink_fallback",
            Self::Unrestricted => "unrestricted",
        }
    }
}

/// Strategy for establishing the read-only OverlayFS mount.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OverlayMode {
    /// Try unprivileged Kernel OverlayFS first, then fall back to `fuse-overlayfs`.
    #[default]
    Auto,
    /// Require Kernel OverlayFS inside an unprivileged user/mount namespace.
    Kernel,
    /// Require `fuse-overlayfs`.
    Fuse,
}

/// Active backend backing an [`OverlayMount`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayBackend {
    KernelOverlayFs,
    FuseOverlayFs,
}

/// Errors returned while validating or mounting an OverlayFS sandbox.
#[derive(Debug, Error)]
pub enum OverlayError {
    #[error("read-only lowerdir '{path}' does not exist or is not a directory")]
    InvalidLowerDir { path: PathBuf },
    #[error("failed to canonicalize path '{path}': {source}")]
    Canonicalize {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to prepare overlay directory '{path}': {source}")]
    PrepareDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("unsafe overlay layout: {0}")]
    UnsafeLayout(String),
    #[error("path '{path}' contains characters invalid in OverlayFS mount options")]
    InvalidMountOptionPath { path: PathBuf },
    #[error("failed to enter unprivileged user/mount namespace: {0}")]
    NamespaceSetup(String),
    #[error("kernel OverlayFS mount failed: {0}")]
    KernelMount(String),
    #[error("fuse-overlayfs mount failed: {0}")]
    FuseMount(String),
}

/// Validated, canonicalized OverlayFS directory layout guaranteeing `lowerdir` isolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlaySpec {
    lowerdir: PathBuf,
    upperdir: PathBuf,
    workdir: PathBuf,
    merged: PathBuf,
    mode: OverlayMode,
}

impl OverlaySpec {
    /// Validate and canonicalize all paths, enforcing read-only `lowerdir` safety invariants.
    ///
    /// Guarantees:
    /// - `lowerdir` exists and is a real directory.
    /// - `upperdir`, `workdir`, and `merged` are created if absent and canonicalized.
    /// - None of `upperdir`, `workdir`, or `merged` overlaps with `lowerdir` (neither inside
    ///   `lowerdir` nor containing `lowerdir`), preventing accidental writes to the title tree.
    /// - `upperdir`, `workdir`, and `merged` are mutually non-overlapping.
    pub fn new(
        lowerdir: impl AsRef<Path>,
        upperdir: impl AsRef<Path>,
        workdir: impl AsRef<Path>,
        merged: impl AsRef<Path>,
        mode: OverlayMode,
    ) -> Result<Self, OverlayError> {
        let lower_input = lowerdir.as_ref();
        if !lower_input.is_dir() {
            return Err(OverlayError::InvalidLowerDir {
                path: lower_input.to_path_buf(),
            });
        }
        let lower = canonicalize(lower_input)?;
        if !lower.is_dir() {
            return Err(OverlayError::InvalidLowerDir { path: lower });
        }

        let upper = ensure_canonical_dir(upperdir.as_ref())?;
        let work = ensure_canonical_dir(workdir.as_ref())?;
        let merged_dir = ensure_canonical_dir(merged.as_ref())?;

        for (label, candidate) in [
            ("upperdir", &upper),
            ("workdir", &work),
            ("merged", &merged_dir),
        ] {
            if paths_overlap(&lower, candidate) {
                return Err(OverlayError::UnsafeLayout(format!(
                    "{label} ('{}') overlaps read-only lowerdir ('{}')",
                    candidate.display(),
                    lower.display()
                )));
            }
        }

        if paths_overlap(&upper, &work) {
            return Err(OverlayError::UnsafeLayout(format!(
                "upperdir ('{}') and workdir ('{}') must not overlap",
                upper.display(),
                work.display()
            )));
        }
        if paths_overlap(&upper, &merged_dir) || paths_overlap(&work, &merged_dir) {
            return Err(OverlayError::UnsafeLayout(format!(
                "merged ('{}') must not overlap upperdir or workdir",
                merged_dir.display()
            )));
        }

        for p in [&lower, &upper, &work, &merged_dir] {
            validate_overlay_option_path(p)?;
        }

        Ok(Self {
            lowerdir: lower,
            upperdir: upper,
            workdir: work,
            merged: merged_dir,
            mode,
        })
    }

    pub fn lowerdir(&self) -> &Path {
        &self.lowerdir
    }

    pub fn upperdir(&self) -> &Path {
        &self.upperdir
    }

    pub fn workdir(&self) -> &Path {
        &self.workdir
    }

    pub fn merged(&self) -> &Path {
        &self.merged
    }

    pub fn mode(&self) -> OverlayMode {
        self.mode
    }

    /// Format the `lowerdir=...,upperdir=...,workdir=...` OverlayFS option string.
    pub fn mount_options(&self) -> String {
        format!(
            "lowerdir={},upperdir={},workdir={}",
            self.lowerdir.display(),
            self.upperdir.display(),
            self.workdir.display()
        )
    }
}

/// Active read-only copy-on-write OverlayFS mount.
#[derive(Debug)]
pub struct OverlayMount {
    spec: OverlaySpec,
    backend: OverlayBackend,
    mounted: bool,
}

impl OverlayMount {
    /// Mount the validated [`OverlaySpec`] using Kernel OverlayFS (unprivileged userns) or
    /// `fuse-overlayfs` according to [`OverlaySpec::mode`].
    pub fn mount(spec: OverlaySpec) -> Result<Self, OverlayError> {
        match spec.mode {
            OverlayMode::Kernel => {
                mount_kernel_overlay(&spec)?;
                Ok(Self {
                    spec,
                    backend: OverlayBackend::KernelOverlayFs,
                    mounted: true,
                })
            }
            OverlayMode::Fuse => {
                mount_fuse_overlay(&spec)?;
                Ok(Self {
                    spec,
                    backend: OverlayBackend::FuseOverlayFs,
                    mounted: true,
                })
            }
            OverlayMode::Auto => match mount_kernel_overlay(&spec) {
                Ok(()) => Ok(Self {
                    spec,
                    backend: OverlayBackend::KernelOverlayFs,
                    mounted: true,
                }),
                Err(kernel_err) => {
                    warn!(
                        error = %kernel_err,
                        "unprivileged kernel OverlayFS unavailable; falling back to fuse-overlayfs"
                    );
                    mount_fuse_overlay(&spec)?;
                    Ok(Self {
                        spec,
                        backend: OverlayBackend::FuseOverlayFs,
                        mounted: true,
                    })
                }
            },
        }
    }

    pub fn spec(&self) -> &OverlaySpec {
        &self.spec
    }

    pub fn backend(&self) -> OverlayBackend {
        self.backend
    }

    /// Explicitly unmount the overlay filesystem.
    pub fn unmount(mut self) -> Result<(), OverlayError> {
        self.unmount_inner()
    }

    fn unmount_inner(&mut self) -> Result<(), OverlayError> {
        if !self.mounted {
            return Ok(());
        }
        self.mounted = false;
        match self.backend {
            OverlayBackend::KernelOverlayFs => umount2(&self.spec.merged, MntFlags::MNT_DETACH)
                .map_err(|e| OverlayError::KernelMount(format!("umount2 failed: {e}"))),
            OverlayBackend::FuseOverlayFs => {
                let status = Command::new("fusermount3")
                    .arg("-u")
                    .arg(&self.spec.merged)
                    .status()
                    .or_else(|_| {
                        Command::new("fusermount")
                            .arg("-u")
                            .arg(&self.spec.merged)
                            .status()
                    })
                    .map_err(|e| OverlayError::FuseMount(format!("fusermount failed: {e}")))?;
                if status.success() {
                    Ok(())
                } else {
                    Err(OverlayError::FuseMount(format!(
                        "fusermount exited with status {status}"
                    )))
                }
            }
        }
    }
}

impl Drop for OverlayMount {
    fn drop(&mut self) {
        if let Err(err) = self.unmount_inner() {
            warn!(error = %err, merged = %self.spec.merged.display(), "failed to unmount overlay on drop");
        }
    }
}

fn preflight_kernel_userns() -> Result<(), OverlayError> {
    // Entering CLONE_NEWUSER irreversibly drops setuid capability for child helpers such as
    // `fusermount3`. Reject known-blocked environments BEFORE calling `unshare` so `Auto` mode
    // can cleanly fall back to `fuse-overlayfs` in the original host namespace.
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("Threads:")
                && let Ok(threads) = rest.trim().parse::<usize>()
                && threads > 1
            {
                return Err(OverlayError::NamespaceSetup(format!(
                    "process has {threads} threads; CLONE_NEWUSER requires a single-threaded process"
                )));
            }
        }
    }

    if let Ok(val) = fs::read_to_string("/proc/sys/user/max_user_namespaces")
        && val.trim() == "0"
    {
        return Err(OverlayError::NamespaceSetup(
            "kernel user.max_user_namespaces is 0".into(),
        ));
    }

    if let Ok(val) = fs::read_to_string("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
        && val.trim() == "1"
    {
        return Err(OverlayError::NamespaceSetup(
            "kernel.apparmor_restrict_unprivileged_userns=1 restricts unprivileged userns mounts"
                .into(),
        ));
    }

    Ok(())
}

/// Bring the `lo` loopback interface UP inside the current network namespace using `SIOCSIFFLAGS`.
///
/// Allows guest-local `127.0.0.1` loopback IPC while keeping all external network routes absent.
pub fn bring_loopback_up() -> io::Result<()> {
    // SAFETY: Standard Linux `AF_INET` UDP socket for interface flag ioctl (`SOCK_CLOEXEC`).
    let raw = unsafe {
        nix::libc::socket(
            nix::libc::AF_INET,
            nix::libc::SOCK_DGRAM | nix::libc::SOCK_CLOEXEC,
            0,
        )
    };
    if raw < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `raw` is a newly created valid file descriptor owned exclusively by this function.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: `libc::ifreq` is a plain-old-data C struct valid when zero-initialized.
    let mut req: nix::libc::ifreq = unsafe { std::mem::zeroed() };
    req.ifr_name[0] = b'l' as nix::libc::c_char;
    req.ifr_name[1] = b'o' as nix::libc::c_char;

    // SAFETY: `fd` is a valid AF_INET socket and `req` is an initialized `ifreq` for `"lo"`.
    if unsafe { nix::libc::ioctl(fd.as_raw_fd(), nix::libc::SIOCGIFFLAGS as _, &mut req) } < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `SIOCGIFFLAGS` populates `req.ifr_ifru.ifru_flags`.
    let flags = unsafe { req.ifr_ifru.ifru_flags };
    req.ifr_ifru.ifru_flags = flags | nix::libc::IFF_UP as nix::libc::c_short;
    // SAFETY: `SIOCSIFFLAGS` applies `IFF_UP` to `"lo"` inside our private network namespace.
    if unsafe { nix::libc::ioctl(fd.as_raw_fd(), nix::libc::SIOCSIFFLAGS as _, &mut req) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Enter a private, loopback-only Linux network namespace (`CLONE_NEWNET`) so no external
/// network interfaces or routes exist while Unix-domain sockets (Wayland, X11, PipeWire) remain
/// accessible through the filesystem.
///
/// Pass `already_in_userns = true` when the process has already entered `CLONE_NEWUSER` (such as
/// after a successful Kernel OverlayFS mount).
pub fn enter_isolated_netns(already_in_userns: bool) -> Result<(), OverlayError> {
    if already_in_userns {
        unshare(CloneFlags::CLONE_NEWNET).map_err(|e| {
            OverlayError::NamespaceSetup(format!("unshare(CLONE_NEWNET) failed: {e}"))
        })?;
    } else {
        preflight_kernel_userns()?;
        let uid = Uid::current();
        let gid = Gid::current();
        unshare(CloneFlags::CLONE_NEWUSER | CloneFlags::CLONE_NEWNET).map_err(|e| {
            OverlayError::NamespaceSetup(format!(
                "unshare(CLONE_NEWUSER | CLONE_NEWNET) failed: {e}"
            ))
        })?;
        fs::write("/proc/self/setgroups", "deny").map_err(|e| {
            OverlayError::NamespaceSetup(format!("write /proc/self/setgroups: {e}"))
        })?;
        fs::write("/proc/self/uid_map", format!("0 {uid} 1\n"))
            .map_err(|e| OverlayError::NamespaceSetup(format!("write /proc/self/uid_map: {e}")))?;
        fs::write("/proc/self/gid_map", format!("0 {gid} 1\n"))
            .map_err(|e| OverlayError::NamespaceSetup(format!("write /proc/self/gid_map: {e}")))?;
    }

    bring_loopback_up().map_err(|e| {
        OverlayError::NamespaceSetup(format!("failed to bring lo UP in netns: {e}"))
    })?;
    info!("entered isolated loopback-only Linux network namespace (CLONE_NEWNET)");
    Ok(())
}

fn mount_kernel_overlay(spec: &OverlaySpec) -> Result<(), OverlayError> {
    preflight_kernel_userns()?;

    let uid = Uid::current();
    let gid = Gid::current();

    unshare(CloneFlags::CLONE_NEWUSER | CloneFlags::CLONE_NEWNS)
        .map_err(|e| OverlayError::NamespaceSetup(format!("unshare failed: {e}")))?;

    fs::write("/proc/self/setgroups", "deny")
        .map_err(|e| OverlayError::NamespaceSetup(format!("write /proc/self/setgroups: {e}")))?;
    fs::write("/proc/self/uid_map", format!("0 {uid} 1\n"))
        .map_err(|e| OverlayError::NamespaceSetup(format!("write /proc/self/uid_map: {e}")))?;
    fs::write("/proc/self/gid_map", format!("0 {gid} 1\n"))
        .map_err(|e| OverlayError::NamespaceSetup(format!("write /proc/self/gid_map: {e}")))?;

    // Isolate mount propagation recursively so mounts stay private to this namespace.
    mount::<str, str, str, str>(None, "/", None, MsFlags::MS_REC | MsFlags::MS_PRIVATE, None)
        .map_err(|e| OverlayError::NamespaceSetup(format!("make root mount MS_PRIVATE: {e}")))?;

    let opts = spec.mount_options();
    mount(
        Some("overlay"),
        &spec.merged,
        Some("overlay"),
        MsFlags::MS_NODEV | MsFlags::MS_NOSUID,
        Some(opts.as_str()),
    )
    .map_err(|e| OverlayError::KernelMount(e.to_string()))?;

    info!(
        lowerdir = %spec.lowerdir.display(),
        merged = %spec.merged.display(),
        "mounted read-only lowerdir via kernel OverlayFS"
    );
    Ok(())
}

fn mount_fuse_overlay(spec: &OverlaySpec) -> Result<(), OverlayError> {
    let opts = spec.mount_options();
    let output = Command::new("fuse-overlayfs")
        .arg("-o")
        .arg(&opts)
        .arg(&spec.merged)
        .output()
        .map_err(|e| OverlayError::FuseMount(format!("failed to spawn fuse-overlayfs: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(OverlayError::FuseMount(format!(
            "fuse-overlayfs exited with {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    info!(
        lowerdir = %spec.lowerdir.display(),
        merged = %spec.merged.display(),
        "mounted read-only lowerdir via fuse-overlayfs"
    );
    Ok(())
}

fn canonicalize(path: &Path) -> Result<PathBuf, OverlayError> {
    fs::canonicalize(path).map_err(|source| OverlayError::Canonicalize {
        path: path.to_path_buf(),
        source,
    })
}

fn ensure_canonical_dir(path: &Path) -> Result<PathBuf, OverlayError> {
    fs::create_dir_all(path).map_err(|source| OverlayError::PrepareDir {
        path: path.to_path_buf(),
        source,
    })?;
    canonicalize(path)
}

fn paths_overlap(a: &Path, b: &Path) -> bool {
    a == b || a.starts_with(b) || b.starts_with(a)
}

fn validate_overlay_option_path(path: &Path) -> Result<(), OverlayError> {
    let s = path.to_string_lossy();
    if s.contains(',') || s.contains(':') || s.contains('\0') {
        return Err(OverlayError::InvalidMountOptionPath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new(label: &str) -> Self {
            let id = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("x2l-overlay-{label}-{}-{id}", std::process::id()));
            fs::create_dir_all(&dir).expect("create temp test root");
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn validates_clean_layout_and_canonicalizes_symlinks() {
        let root = TempRoot::new("clean");
        let real_lower = root.path().join("title_ro");
        fs::create_dir_all(&real_lower).unwrap();
        let sentinel = real_lower.join("manifest.txt");
        fs::write(&sentinel, b"immutable-payload").unwrap();

        let lower_link = root.path().join("title_symlink");
        symlink(&real_lower, &lower_link).unwrap();

        let upper = root.path().join("state/overlay");
        let work = root.path().join("state/work");
        let merged = root.path().join("state/merged");

        let spec = OverlaySpec::new(&lower_link, &upper, &work, &merged, OverlayMode::Auto)
            .expect("valid overlay spec");

        assert_eq!(spec.lowerdir(), fs::canonicalize(&real_lower).unwrap());
        assert_eq!(spec.upperdir(), fs::canonicalize(&upper).unwrap());
        assert_eq!(spec.workdir(), fs::canonicalize(&work).unwrap());
        assert_eq!(spec.merged(), fs::canonicalize(&merged).unwrap());
        assert!(spec.mount_options().starts_with("lowerdir="));
        assert_eq!(fs::read(&sentinel).unwrap(), b"immutable-payload");
    }

    #[test]
    fn rejects_upperdir_inside_lowerdir_or_via_symlink() {
        let root = TempRoot::new("overlap-lower");
        let lower = root.path().join("title_ro");
        fs::create_dir_all(&lower).unwrap();
        let nested_upper = lower.join("evil_upper");
        let work = root.path().join("work");
        let merged = root.path().join("merged");

        let err = OverlaySpec::new(&lower, &nested_upper, &work, &merged, OverlayMode::Auto)
            .expect_err("upperdir inside lowerdir must fail");
        assert!(matches!(err, OverlayError::UnsafeLayout(_)));

        // Even via a symlink outside lowerdir pointing into lowerdir.
        let alias_upper = root.path().join("alias_upper");
        symlink(&lower, &alias_upper).unwrap();
        let err = OverlaySpec::new(&lower, &alias_upper, &work, &merged, OverlayMode::Auto)
            .expect_err("symlinked upperdir pointing to lowerdir must fail");
        assert!(matches!(err, OverlayError::UnsafeLayout(_)));
    }

    #[test]
    fn rejects_lowerdir_inside_upperdir_and_overlapping_workdirs() {
        let root = TempRoot::new("overlap-parent");
        let parent = root.path().join("parent");
        let lower = parent.join("title_ro");
        fs::create_dir_all(&lower).unwrap();
        let work = root.path().join("work");
        let merged = root.path().join("merged");

        let err = OverlaySpec::new(&lower, &parent, &work, &merged, OverlayMode::Auto)
            .expect_err("lowerdir inside upperdir must fail");
        assert!(matches!(err, OverlayError::UnsafeLayout(_)));

        let upper = root.path().join("upper");
        let nested_work = upper.join("work");
        let err = OverlaySpec::new(&lower, &upper, &nested_work, &merged, OverlayMode::Auto)
            .expect_err("workdir inside upperdir must fail");
        assert!(matches!(err, OverlayError::UnsafeLayout(_)));
    }

    #[test]
    fn rejects_missing_lowerdir() {
        let root = TempRoot::new("missing");
        let missing = root.path().join("does_not_exist");
        let err = OverlaySpec::new(
            &missing,
            root.path().join("upper"),
            root.path().join("work"),
            root.path().join("merged"),
            OverlayMode::Auto,
        )
        .expect_err("missing lowerdir must fail");
        assert!(matches!(err, OverlayError::InvalidLowerDir { .. }));
    }
}
