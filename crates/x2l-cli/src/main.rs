//! `x2linux` command-line interface for inspecting, launching, and analyzing unpacked GDK software.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracing::info;
use tracing_subscriber::EnvFilter;
use x2l_common::{GpuBackend, OverlayFsMode, X2LinuxConfig, XdgPaths};
use x2l_resolver::{ImportPolicy, inspect_pe_file};
use x2l_vfs_overlay::OverlayMode;

/// Native Linux x86-64 compatibility layer for unpacked Xbox Series X|S (GDK / Scarlett) software.
#[derive(Debug, Parser)]
#[command(name = "x2linux", version, about, long_about = None)]
struct Cli {
    /// Override log level filter (e.g. error, warn, info, debug, trace).
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Inspect a PE32+ executable's imports, console DLLs, and unresolved symbols (read-only).
    Inspect {
        /// Path to an unpacked PE32+ executable.
        executable: PathBuf,
        /// Emit machine-readable JSON summary.
        #[arg(long)]
        json: bool,
        /// Optional path to an API-set / console DLL aliases TOML file.
        #[arg(long)]
        aliases: Option<PathBuf>,
        /// Optional directory containing per-DLL ordinal mapping TOML files.
        #[arg(long)]
        ordinals_dir: Option<PathBuf>,
    },
    /// Launch an unpacked title directory inside a read-only Kernel OverlayFS sandbox.
    Run {
        /// Path to the unpacked title directory (mounted strictly read-only as lowerdir).
        title_dir: PathBuf,
        /// Optional path to a custom x2linux.toml configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Override OverlayFS backend mode.
        #[arg(long, value_enum)]
        overlay_mode: Option<CliOverlayMode>,
        /// Override GPU backend.
        #[arg(long, value_enum)]
        gpu: Option<CliGpuBackend>,
    },
    /// Disassemble a raw RDNA2 shader binary or console shader container.
    Disasm {
        /// Path to the shader binary to disassemble.
        shader: PathBuf,
    },
    /// Batch-inspect or recompile console shaders directly to binary SPIR-V 1.6.
    Shaders {
        /// Directory containing console shader binaries.
        shader_dir: PathBuf,
        /// Recompile decoded RDNA2 shaders directly to binary SPIR-V 1.6.
        #[arg(long)]
        recomp: bool,
        /// Validate generated SPIR-V modules with `spirv-val`.
        #[arg(long)]
        validate_spirv: bool,
        /// Optional output directory for compiled `.spv` modules.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliOverlayMode {
    Auto,
    Kernel,
    Fuse,
}

impl From<CliOverlayMode> for OverlayFsMode {
    fn from(value: CliOverlayMode) -> Self {
        match value {
            CliOverlayMode::Auto => Self::Auto,
            CliOverlayMode::Kernel => Self::Kernel,
            CliOverlayMode::Fuse => Self::Fuse,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliGpuBackend {
    Null,
    Vulkan,
}

impl From<CliGpuBackend> for GpuBackend {
    fn from(value: CliGpuBackend) -> Self {
        match value {
            CliGpuBackend::Null => Self::Null,
            CliGpuBackend::Vulkan => Self::Vulkan,
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    match run_cli(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("x2linux: {msg}");
            ExitCode::FAILURE
        }
    }
}

fn init_tracing(level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .try_init();
}

fn load_inspect_policy(
    aliases: Option<&Path>,
    ordinals_dir: Option<&Path>,
) -> Result<ImportPolicy, String> {
    let default_aliases = Path::new("data/aliases.toml");
    let default_ordinals = Path::new("data/ordinals");

    let resolved_aliases = aliases.or_else(|| default_aliases.is_file().then_some(default_aliases));
    let resolved_ordinals =
        ordinals_dir.or_else(|| default_ordinals.is_dir().then_some(default_ordinals));

    ImportPolicy::load_from_paths(resolved_aliases, resolved_ordinals).map_err(|e| e.to_string())
}

fn run_cli(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::Inspect {
            executable,
            json,
            aliases,
            ordinals_dir,
        } => {
            let policy = load_inspect_policy(aliases.as_deref(), ordinals_dir.as_deref())?;
            let report = inspect_pe_file(&executable, &policy).map_err(|e| e.to_string())?;
            if json {
                let text = report.to_json_pretty().map_err(|e| e.to_string())?;
                println!("{text}");
            } else {
                print!("{report}");
            }
            Ok(())
        }
        Commands::Run {
            title_dir,
            config,
            overlay_mode,
            gpu,
        } => {
            let xdg = XdgPaths::resolve().map_err(|e| e.to_string())?;
            let mut cfg = match config {
                Some(path) => X2LinuxConfig::load_file(&path).map_err(|e| e.to_string())?,
                None => X2LinuxConfig::default(),
            };
            if let Some(mode) = overlay_mode {
                cfg.runtime.overlayfs_mode = mode.into();
            }
            if let Some(backend) = gpu {
                cfg.gpu.backend = backend.into();
            }
            let resolved_overlay_mode = match cfg.runtime.overlayfs_mode {
                OverlayFsMode::Auto => OverlayMode::Auto,
                OverlayFsMode::Kernel => OverlayMode::Kernel,
                OverlayFsMode::Fuse => OverlayMode::Fuse,
            };
            info!(
                title_dir = %title_dir.display(),
                overlay = ?resolved_overlay_mode,
                gpu = ?cfg.gpu.backend,
                data_dir = %xdg.data_dir.display(),
                "run requested"
            );
            println!(
                "[x2linux] run '{}' (overlay={:?}, gpu={:?}) — M1 launcher wiring in progress.",
                title_dir.display(),
                resolved_overlay_mode,
                cfg.gpu.backend
            );
            Ok(())
        }
        Commands::Disasm { shader } => {
            info!(shader = %shader.display(), "disasm requested");
            println!(
                "[x2linux] disasm '{}' — M2 RDNA2 decoder wiring pending.",
                shader.display()
            );
            Ok(())
        }
        Commands::Shaders {
            shader_dir,
            recomp,
            validate_spirv,
            out,
        } => {
            info!(
                shader_dir = %shader_dir.display(),
                recomp,
                validate_spirv,
                out = ?out,
                "shaders batch requested"
            );
            println!(
                "[x2linux] shaders '{}' (recomp={recomp}, validate_spirv={validate_spirv}) — M2 direct rspirv pipeline pending.",
                shader_dir.display()
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_subcommands() {
        assert!(
            Cli::try_parse_from([
                "x2linux",
                "inspect",
                "/tmp/Title.exe",
                "--json",
                "--aliases",
                "data/aliases.toml",
                "--ordinals-dir",
                "data/ordinals",
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "x2linux",
                "run",
                "/tmp/title",
                "--overlay-mode",
                "auto",
                "--gpu",
                "vulkan"
            ])
            .is_ok()
        );
        assert!(Cli::try_parse_from(["x2linux", "disasm", "/tmp/shader.bin"]).is_ok());
        assert!(
            Cli::try_parse_from([
                "x2linux",
                "shaders",
                "/tmp/shaders",
                "--recomp",
                "--validate-spirv"
            ])
            .is_ok()
        );
    }
}
