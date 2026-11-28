//! `x2linux run <dir|exe>` runtime launcher: OverlayFS mount, isolated Wine prefix,
//! `ntsync` detection, `LD_PRELOAD` `libx2l_core.so` bridge, and `report.json` flush.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{info, warn};
use x2l_common::bridge::{
    ENV_BRIDGE_ENABLED, ENV_FRAME_DUMP, ENV_GPU_BACKEND, ENV_GPU_VALIDATION,
    ENV_PREFER_REBAR_ZERO_COPY, ENV_SPIRV_CACHE_DIR, X2L_BRIDGE_MAGIC,
};
use x2l_common::{
    DeviceType, GpuBackend, HostRuntimeMetadata, OverlayFsMode, TelemetryReport, X2LinuxConfig,
    XdgPaths,
};
use x2l_vfs_overlay::{OverlayBackend, OverlayMode, OverlayMount, OverlaySpec};

/// Options passed from `x2linux run`.
pub struct RunOptions {
    pub target: PathBuf,
    pub title_id: Option<String>,
    pub config: Option<PathBuf>,
    pub overlay_mode: Option<OverlayFsMode>,
    pub gpu: Option<GpuBackend>,
    pub shims_dir: Option<PathBuf>,
    pub core_lib: Option<PathBuf>,
}

/// Execute an unpacked title directory or `.exe` inside a read-only OverlayFS sandbox under Wine.
pub fn execute_run(opts: RunOptions) -> Result<i32, String> {
    let (lowerdir, exe_rel) = resolve_target(&opts.target)?;
    let title_id = derive_title_id(opts.title_id.as_deref(), &lowerdir);

    let xdg = XdgPaths::resolve().map_err(|e| e.to_string())?;
    let mut cfg = load_title_config(&xdg, &title_id, opts.config.as_deref())?;
    if let Some(mode) = opts.overlay_mode {
        cfg.runtime.overlayfs_mode = mode;
    }
    if let Some(backend) = opts.gpu {
        cfg.gpu.backend = backend;
    }

    let overlay_dirs = xdg
        .title_overlay_dirs(&title_id)
        .map_err(|e| e.to_string())?;
    let shaders_raw_dir = xdg
        .title_shaders_raw_dir(&title_id)
        .map_err(|e| e.to_string())?;
    let spirv_cache_dir = xdg
        .title_spirv_cache_dir(&title_id)
        .map_err(|e| e.to_string())?;
    let logs_dir = overlay_dirs.root.join("logs");
    let prefix_dir = overlay_dirs.root.join("prefix");

    for dir in [
        &overlay_dirs.saves,
        &overlay_dirs.pls,
        &overlay_dirs.temp,
        &logs_dir,
        &prefix_dir,
        &shaders_raw_dir,
        &spirv_cache_dir,
    ] {
        fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create directory '{}': {e}", dir.display()))?;
    }

    let overlay_mode = match cfg.runtime.overlayfs_mode {
        OverlayFsMode::Auto => OverlayMode::Auto,
        OverlayFsMode::Kernel => OverlayMode::Kernel,
        OverlayFsMode::Fuse => OverlayMode::Fuse,
    };

    let spec = OverlaySpec::new(
        &lowerdir,
        &overlay_dirs.upperdir,
        &overlay_dirs.workdir,
        &overlay_dirs.merged,
        overlay_mode,
    )
    .map_err(|e| format!("overlay validation failed: {e}"))?;

    let mount = OverlayMount::mount(spec).map_err(|e| format!("overlay mount failed: {e}"))?;
    let active_backend = mount.backend();

    let run_result = run_inside_mounted_overlay(
        &title_id,
        &exe_rel,
        &cfg,
        &overlay_dirs.merged,
        &overlay_dirs.saves,
        &overlay_dirs.pls,
        &overlay_dirs.temp,
        &prefix_dir,
        &logs_dir,
        &shaders_raw_dir,
        &spirv_cache_dir,
        active_backend,
        opts.shims_dir.as_deref(),
        opts.core_lib.as_deref(),
    );

    if let Err(unmount_err) = mount.unmount() {
        warn!(error = %unmount_err, "overlay unmount reported a warning");
    }

    run_result
}

#[allow(clippy::too_many_arguments)]
fn run_inside_mounted_overlay(
    title_id: &str,
    exe_rel: &Path,
    cfg: &X2LinuxConfig,
    merged: &Path,
    saves: &Path,
    pls: &Path,
    temp: &Path,
    prefix_dir: &Path,
    logs_dir: &Path,
    shaders_raw_dir: &Path,
    spirv_cache_dir: &Path,
    backend: OverlayBackend,
    shims_dir_override: Option<&Path>,
    core_lib_override: Option<&Path>,
) -> Result<i32, String> {
    let wine_bin = resolve_wine_binary(&cfg.runtime.wine_binary)?;
    let shims_dir = resolve_shims_dir(shims_dir_override)?;
    let core_lib = resolve_core_lib(core_lib_override)?;

    let ntsync_available = Path::new("/dev/ntsync").exists();
    let ntsync_enabled = cfg.runtime.use_ntsync && ntsync_available;
    let backend_name = match backend {
        OverlayBackend::KernelOverlayFs => "kernel_userns",
        OverlayBackend::FuseOverlayFs => "fuse_overlayfs",
    };
    let device_str = match cfg.console.device_type {
        DeviceType::SeriesX => "series_x",
        DeviceType::SeriesS => "series_s",
    };
    let gpu_backend_str = match cfg.gpu.backend {
        GpuBackend::Null => "null",
        GpuBackend::Vulkan => "vulkan",
    };

    prepare_wine_prefix(prefix_dir, merged, saves, pls, temp)?;

    let log_path = logs_dir.join("x2linux.log");
    let report_path = logs_dir.join("report.json");
    let _ = fs::remove_file(&report_path);

    append_log(
        &log_path,
        &format!(
            "[LAUNCHER] title={title_id} exe={} wine={} overlay={backend_name} ntsync_dev={ntsync_available} ntsync_enabled={ntsync_enabled} bridge_magic=0x{X2L_BRIDGE_MAGIC:016x}",
            exe_rel.display(),
            wine_bin,
        ),
    );
    info!(
        title_id,
        wine = %wine_bin,
        overlay = backend_name,
        ntsync_available,
        ntsync_enabled,
        shims = %shims_dir.display(),
        core_lib = %core_lib.display(),
        "launching guest executable under Wine"
    );

    let win_exe = format!("G:\\{}", exe_rel.to_string_lossy().replace('/', "\\"));
    let wine_dll_path = shims_dir.to_string_lossy().into_owned();
    let wine_path_win = format!("Z:{}", shims_dir.to_string_lossy().replace('/', "\\"));
    let ld_preload = match std::env::var("LD_PRELOAD") {
        Ok(existing) if !existing.is_empty() => format!("{}:{existing}", core_lib.display()),
        _ => core_lib.display().to_string(),
    };

    let mut cmd = Command::new(&wine_bin);
    cmd.arg(&win_exe)
        .current_dir(merged)
        .env("WINEPREFIX", prefix_dir)
        .env("WINEARCH", "win64")
        .env("WINEDEBUG", "-all")
        .env("WINEDLLPATH", &wine_dll_path)
        .env("WINEPATH", &wine_path_win)
        .env(
            "WINEDLLOVERRIDES",
            "kernelx,xgameruntime,d3d12_xs=n;mscoree,mshtml=",
        )
        .env("LD_PRELOAD", &ld_preload)
        .env(ENV_BRIDGE_ENABLED, "1")
        .env("X2L_TITLE_ID", title_id)
        .env("X2L_DEVICE_TYPE", device_str)
        .env("X2L_GAMERTAG", &cfg.console.gamertag)
        .env("X2L_SAVES_DIR", saves)
        .env("X2L_SHADERS_RAW_DIR", shaders_raw_dir)
        .env(ENV_SPIRV_CACHE_DIR, spirv_cache_dir)
        .env(ENV_GPU_BACKEND, gpu_backend_str)
        .env(
            ENV_GPU_VALIDATION,
            if cfg.gpu.validation { "1" } else { "0" },
        )
        .env(
            ENV_PREFER_REBAR_ZERO_COPY,
            if cfg.gpu.prefer_rebar_zero_copy {
                "1"
            } else {
                "0"
            },
        )
        .env("X2L_REPORT_PATH", &report_path)
        .env("X2L_LOG_PATH", &log_path)
        .env("X2L_OVERLAY_BACKEND", backend_name)
        .env(
            "X2L_NTSYNC_AVAILABLE",
            if ntsync_available { "true" } else { "false" },
        )
        .env(
            "X2L_NTSYNC_ENABLED",
            if ntsync_enabled { "true" } else { "false" },
        )
        .env("X2L_WINE_BINARY", &wine_bin);

    if let Ok(dump) = std::env::var(ENV_FRAME_DUMP)
        && !dump.is_empty()
    {
        cmd.env(ENV_FRAME_DUMP, dump);
    }

    if ntsync_enabled {
        cmd.env("NTSYNC", "1")
            .env("WINEESYNC", "0")
            .env("WINEFSYNC", "0");
    } else {
        cmd.env("NTSYNC", "0");
    }

    let status = cmd
        .status()
        .map_err(|e| format!("failed to launch '{wine_bin}': {e}"))?;
    let exit_code = status.code().unwrap_or(1);

    // Ensure wineserver closes open handles inside `merged/` before unmounting OverlayFS.
    let _ = Command::new("wineserver")
        .arg("-w")
        .env("WINEPREFIX", prefix_dir)
        .env_remove("LD_PRELOAD")
        .env_remove(ENV_BRIDGE_ENABLED)
        .status();

    if cfg.logging.write_report_json && !report_path.is_file() {
        let mut fallback = TelemetryReport::new(title_id);
        fallback.host = HostRuntimeMetadata {
            kernel_release: read_kernel_release(),
            ntsync_available,
            ntsync_enabled,
            overlayfs_mode: backend_name.to_owned(),
            wine_binary: wine_bin.clone(),
            device_type: device_str.to_owned(),
        };
        if let Ok(json) = fallback.to_json_pretty() {
            let _ = fs::write(&report_path, json);
        }
    }

    append_log(
        &log_path,
        &format!(
            "[LAUNCHER] guest exited with code {exit_code}; report={}",
            report_path.display()
        ),
    );
    println!(
        "[x2linux] exited code={exit_code} overlay={backend_name} ntsync={ntsync_enabled} report={}",
        report_path.display()
    );

    Ok(exit_code)
}

fn prepare_wine_prefix(
    prefix_dir: &Path,
    merged: &Path,
    saves: &Path,
    pls: &Path,
    temp: &Path,
) -> Result<(), String> {
    let drive_c = prefix_dir.join("drive_c");
    let dosdevices = prefix_dir.join("dosdevices");
    fs::create_dir_all(&drive_c)
        .map_err(|e| format!("failed to create '{}': {e}", drive_c.display()))?;
    fs::create_dir_all(&dosdevices)
        .map_err(|e| format!("failed to create '{}': {e}", dosdevices.display()))?;

    let links: &[(&str, &Path)] = &[
        ("c:", Path::new("../drive_c")),
        ("g:", merged),
        ("s:", saves),
        ("p:", pls),
        ("t:", temp),
        ("z:", Path::new("/")),
    ];

    for (drive, target) in links {
        let link_path = dosdevices.join(drive);
        if fs::symlink_metadata(&link_path).is_ok() {
            let _ = fs::remove_file(&link_path);
        }
        symlink(target, &link_path).map_err(|e| {
            format!(
                "failed to symlink '{}' -> '{}': {e}",
                link_path.display(),
                target.display()
            )
        })?;
    }
    Ok(())
}

fn resolve_target(target: &Path) -> Result<(PathBuf, PathBuf), String> {
    let canonical = fs::canonicalize(target)
        .map_err(|e| format!("target '{}' does not exist: {e}", target.display()))?;
    if canonical.is_file() {
        let parent = canonical
            .parent()
            .ok_or_else(|| format!("target '{}' has no parent directory", canonical.display()))?
            .to_path_buf();
        let file_name = canonical
            .file_name()
            .ok_or_else(|| format!("target '{}' has no filename", canonical.display()))?;
        return Ok((parent, PathBuf::from(file_name)));
    }
    if !canonical.is_dir() {
        return Err(format!(
            "target '{}' is neither a file nor a directory",
            canonical.display()
        ));
    }
    let mut exes = Vec::new();
    let entries = fs::read_dir(&canonical)
        .map_err(|e| format!("failed to read directory '{}': {e}", canonical.display()))?;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file()
            && p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("exe"))
            && let Some(name) = p.file_name()
        {
            exes.push(PathBuf::from(name));
        }
    }
    exes.sort();
    let chosen = exes.into_iter().next().ok_or_else(|| {
        format!(
            "no .exe found in '{}'; pass the executable path directly",
            canonical.display()
        )
    })?;
    Ok((canonical, chosen))
}

fn derive_title_id(explicit: Option<&str>, lowerdir: &Path) -> String {
    let raw = explicit
        .map(str::to_owned)
        .or_else(|| {
            lowerdir
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "title".into());
    let sanitized: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() || sanitized == "." || sanitized == ".." {
        "title".into()
    } else {
        sanitized
    }
}

fn load_title_config(
    xdg: &XdgPaths,
    title_id: &str,
    explicit: Option<&Path>,
) -> Result<X2LinuxConfig, String> {
    if let Some(path) = explicit {
        return X2LinuxConfig::load_file(path).map_err(|e| e.to_string());
    }
    let default_cfg = xdg.title_config_file(title_id).map_err(|e| e.to_string())?;
    if default_cfg.is_file() {
        X2LinuxConfig::load_file(&default_cfg).map_err(|e| e.to_string())
    } else {
        Ok(X2LinuxConfig::default())
    }
}

fn resolve_wine_binary(configured: &str) -> Result<String, String> {
    for candidate in [configured, "wine64", "wine"] {
        if Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Ok(candidate.to_owned());
        }
    }
    Err(format!(
        "no working Wine binary found (tried '{configured}', 'wine64', 'wine')"
    ))
}

fn resolve_shims_dir(override_dir: Option<&Path>) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(p) = override_dir {
        candidates.push(p.to_path_buf());
    }
    if let Ok(env_p) = std::env::var("X2L_PE_SHIMS_DIR") {
        candidates.push(PathBuf::from(env_p));
    }
    candidates.push(PathBuf::from("target/pe-shims/debug"));
    candidates.push(PathBuf::from("target/pe-shims/release"));
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("../pe-shims/debug"));
        candidates.push(dir.join("../pe-shims/release"));
    }

    for dir in candidates {
        if dir.join("kernelx.dll").is_file()
            && dir.join("xgameruntime.dll").is_file()
            && dir.join("d3d12_xs.dll").is_file()
        {
            return fs::canonicalize(&dir).map_err(|e| e.to_string());
        }
    }
    Err(
        "PE shims (kernelx.dll, xgameruntime.dll, d3d12_xs.dll) not found; run `./scripts/build_pe_shims.sh` first"
            .into(),
    )
}

fn resolve_core_lib(override_lib: Option<&Path>) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Some(p) = override_lib {
        candidates.push(p.to_path_buf());
    }
    if let Ok(env_p) = std::env::var("X2L_CORE_LIB") {
        candidates.push(PathBuf::from(env_p));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("libx2l_core.so"));
    }
    candidates.push(PathBuf::from("target/debug/libx2l_core.so"));
    candidates.push(PathBuf::from("target/release/libx2l_core.so"));

    for lib in candidates {
        if lib.is_file() {
            return fs::canonicalize(&lib).map_err(|e| e.to_string());
        }
    }
    Err("libx2l_core.so not found; run `cargo build -p x2l-core` first".into())
}

fn append_log(path: &Path, line: &str) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

fn read_kernel_release() -> String {
    fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "linux".into())
}
