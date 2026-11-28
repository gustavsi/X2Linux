//! `x2linux` command-line interface for inspecting, launching, and analyzing unpacked GDK software.

mod run;

use clap::{Parser, Subcommand, ValueEnum};
use run::{RunOptions, execute_run};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tracing::info;
use tracing_subscriber::EnvFilter;
use x2l_common::{GpuBackend, OverlayFsMode, PresentMode};
use x2l_rdna2_spirv::{
    CompileOptions, ContainerKind, GraphicsCompileOptions, ShaderStage, WaveSize, classify,
    compile_compute_spv_with_options, compile_graphics_spv, disassemble_reachable,
    format_disassembly, parse_container, spirv_words_to_bytes, validate_spirv_vulkan13,
};
use x2l_resolver::{ImportPolicy, inspect_pe_file};

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
    /// Launch an unpacked title directory or executable inside a read-only OverlayFS sandbox.
    Run {
        /// Path to the unpacked title directory or `.exe` (parent folder mounted strictly read-only).
        target: PathBuf,
        /// Optional title identifier for XDG config/data/cache isolation.
        #[arg(long)]
        title_id: Option<String>,
        /// Optional path to a custom x2linux.toml configuration file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Override OverlayFS backend mode.
        #[arg(long, value_enum)]
        overlay_mode: Option<CliOverlayMode>,
        /// Override GPU backend.
        #[arg(long, value_enum)]
        gpu: Option<CliGpuBackend>,
        /// Present frames to a native Wayland/X11 Vulkan swapchain window (falls back to headless if unavailable).
        #[arg(long, conflicts_with = "headless")]
        window: bool,
        /// Force headless offscreen rendering (default).
        #[arg(long, conflicts_with = "window")]
        headless: bool,
        /// Optional frame count forwarded to the guest executable (`--frames <N>`).
        #[arg(long)]
        frames: Option<u32>,
        /// Override directory containing cross-compiled PE shim DLLs.
        #[arg(long)]
        shims_dir: Option<PathBuf>,
        /// Override path to native `libx2l_core.so`.
        #[arg(long)]
        core_lib: Option<PathBuf>,
    },
    /// Disassemble a raw RDNA2 shader binary or console shader container.
    Disasm {
        /// Path to the shader binary to disassemble.
        shader: PathBuf,
        /// Optional output path to emit a compiled binary SPIR-V 1.6 `.spv` module.
        #[arg(long)]
        spirv_out: Option<PathBuf>,
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
        Ok(code) => ExitCode::from(code as u8),
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

fn collect_code_stages(
    parsed: &x2l_rdna2_spirv::ParsedContainer,
) -> Vec<(String, bool, Vec<u32>, usize)> {
    if !parsed.variants.is_empty() {
        parsed
            .variants
            .iter()
            .map(|v| {
                let words = v.code_words();
                (v.hw_stage.to_string(), v.wave32, words, v.segments.len())
            })
            .collect()
    } else if let Some(words) = parsed.primary_code_words() {
        vec![("RAW".to_owned(), true, words, 1)]
    } else {
        Vec::new()
    }
}

fn execute_disasm(shader: &Path, spirv_out: Option<&Path>) -> Result<i32, String> {
    info!(shader = %shader.display(), "disassembling shader");
    let bytes =
        fs::read(shader).map_err(|e| format!("failed to read '{}': {e}", shader.display()))?;
    let parsed = parse_container(&bytes)
        .map_err(|e| format!("failed to parse '{}': {e}", shader.display()))?;

    let stages = collect_code_stages(&parsed);
    println!(
        "Shader: {}\nFormat: {} | SHA-256: {} | Variants: {}",
        shader.display(),
        parsed.kind,
        parsed.content_sha256,
        stages.len()
    );

    let mut compiled_primary = false;
    for (idx, (label, wave32, code_words, segs)) in stages.iter().enumerate() {
        println!(
            "\n--- Stage #{idx}: {label} (wave{}, {} bytes code, {segs} segment(s)) ---",
            if *wave32 { 32 } else { 64 },
            code_words.len() * 4,
        );
        if code_words.is_empty() {
            println!("  (no native RDNA2 code bytes)");
            continue;
        }
        let insts = disassemble_reachable(code_words)
            .map_err(|e| format!("decode failed for stage #{idx}: {e}"))?;
        print!("{}", format_disassembly(&insts));

        if let Some(out_path) = spirv_out
            && !compiled_primary
        {
            let spv = compile_compute_spv_with_options(
                &insts,
                CompileOptions {
                    wave_size: WaveSize::from_wave32(*wave32),
                    ..CompileOptions::default()
                },
            )
            .map_err(|e| format!("SPIR-V 1.6 lowering failed: {e}"))?;
            let spv_bytes = spirv_words_to_bytes(&spv);
            if let Some(parent) = out_path.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create '{}': {e}", parent.display()))?;
            }
            fs::write(out_path, &spv_bytes)
                .map_err(|e| format!("failed to write '{}': {e}", out_path.display()))?;
            println!(
                "\n[spirv] emitted {} words ({} bytes) -> {}",
                spv.len(),
                spv_bytes.len(),
                out_path.display()
            );
            compiled_primary = true;
        }
    }

    if spirv_out.is_some() && !compiled_primary {
        return Err("no native RDNA2 code stage available to compile to SPIR-V".into());
    }
    Ok(0)
}

fn collect_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("failed to read dir '{}': {e}", dir.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry error in '{}': {e}", dir.display()))?;
        paths.push(entry.path());
    }
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_files_recursive(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn execute_shaders_batch(
    shader_dir: &Path,
    recomp: bool,
    validate_spirv: bool,
    out_dir: Option<&Path>,
) -> Result<i32, String> {
    info!(
        shader_dir = %shader_dir.display(),
        recomp,
        validate_spirv,
        out = ?out_dir,
        "running batch shader analysis"
    );
    let mut files = Vec::new();
    collect_files_recursive(shader_dir, &mut files)?;

    if let Some(dir) = out_dir {
        fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create output dir '{}': {e}", dir.display()))?;
    }

    let mut by_format: BTreeMap<ContainerKind, usize> = BTreeMap::new();
    let mut by_stage: BTreeMap<String, usize> = BTreeMap::new();
    let mut mnemonics: BTreeMap<String, usize> = BTreeMap::new();
    let mut stages_decoded = 0usize;
    let mut compiled_ok = 0usize;
    let mut compiled_err = 0usize;
    let mut validated_ok = 0usize;
    let mut validated_err = 0usize;

    let do_compile = recomp || validate_spirv || out_dir.is_some();

    for path in &files {
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let fmt = classify(&bytes);
        *by_format.entry(fmt).or_insert(0) += 1;

        let Ok(parsed) = parse_container(&bytes) else {
            continue;
        };

        for (stage_idx, (stage_label, wave32, code_words, _)) in
            collect_code_stages(&parsed).into_iter().enumerate()
        {
            if code_words.is_empty() {
                continue;
            }
            let Ok(insts) = disassemble_reachable(&code_words) else {
                continue;
            };
            stages_decoded += 1;
            *by_stage.entry(stage_label.clone()).or_insert(0) += 1;
            for inst in &insts {
                *mnemonics.entry(inst.mnemonic.clone()).or_insert(0) += 1;
            }

            if do_compile {
                let compiled = match stage_label.as_str() {
                    "VS" => compile_graphics_spv(
                        ShaderStage::Vertex,
                        &insts,
                        &parsed.signature,
                        &GraphicsCompileOptions::default(),
                    ),
                    "PS" => compile_graphics_spv(
                        ShaderStage::Pixel,
                        &insts,
                        &parsed.signature,
                        &GraphicsCompileOptions::default(),
                    ),
                    _ => compile_compute_spv_with_options(
                        &insts,
                        CompileOptions {
                            wave_size: WaveSize::from_wave32(wave32),
                            ..CompileOptions::default()
                        },
                    ),
                };
                match compiled {
                    Ok(words) => {
                        compiled_ok += 1;
                        if validate_spirv {
                            match validate_spirv_vulkan13(&words) {
                                Ok(()) => validated_ok += 1,
                                Err(e) => {
                                    validated_err += 1;
                                    eprintln!(
                                        "[spirv-val FAIL] {} stage #{stage_idx}: {e}",
                                        path.display()
                                    );
                                }
                            }
                        }
                        if let Some(dir) = out_dir {
                            let stem = path
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("shader");
                            let out_file = dir.join(format!("{stem}_s{stage_idx}.spv"));
                            fs::write(&out_file, spirv_words_to_bytes(&words)).map_err(|e| {
                                format!("failed to write '{}': {e}", out_file.display())
                            })?;
                        }
                    }
                    Err(_) => {
                        compiled_err += 1;
                    }
                }
            }
        }
    }

    println!(
        "Scanned {} file(s) in {}",
        files.len(),
        shader_dir.display()
    );
    println!("Formats:");
    for (fmt, count) in &by_format {
        println!("  {fmt}: {count}");
    }
    println!("Decoded RDNA2 stages: {stages_decoded}");
    if !by_stage.is_empty() {
        println!("Stages:");
        for (stage, count) in &by_stage {
            println!("  {stage:<8} {count}");
        }
    }

    let mut top: Vec<(&String, &usize)> = mnemonics.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    if !top.is_empty() {
        println!("Top mnemonics:");
        for (mnem, count) in top.into_iter().take(15) {
            println!("  {mnem:<22} {count}");
        }
    }

    if do_compile {
        println!(
            "SPIR-V 1.6 recompile: {compiled_ok} succeeded, {compiled_err} unsupported/failed"
        );
    }
    if validate_spirv {
        println!("spirv-val (vulkan1.3): {validated_ok} passed, {validated_err} failed");
        if validated_err > 0 {
            return Err(format!(
                "{validated_err} SPIR-V module(s) failed validation"
            ));
        }
    }

    Ok(0)
}

fn run_cli(cli: Cli) -> Result<i32, String> {
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
            Ok(0)
        }
        Commands::Run {
            target,
            title_id,
            config,
            overlay_mode,
            gpu,
            window,
            headless,
            frames,
            shims_dir,
            core_lib,
        } => {
            let present_mode = if window {
                Some(PresentMode::Window)
            } else if headless {
                Some(PresentMode::Headless)
            } else {
                None
            };
            let mut guest_args = Vec::new();
            if let Some(n) = frames {
                guest_args.push("--frames".to_owned());
                guest_args.push(n.to_string());
            }
            execute_run(RunOptions {
                target,
                title_id,
                config,
                overlay_mode: overlay_mode.map(Into::into),
                gpu: gpu.map(Into::into),
                present_mode,
                guest_args,
                shims_dir,
                core_lib,
            })
        }
        Commands::Disasm { shader, spirv_out } => execute_disasm(&shader, spirv_out.as_deref()),
        Commands::Shaders {
            shader_dir,
            recomp,
            validate_spirv,
            out,
        } => execute_shaders_batch(&shader_dir, recomp, validate_spirv, out.as_deref()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use x2l_rdna2_spirv::container::build_synthetic_console_dxbc;

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
                "--title-id",
                "synthetic-test",
                "--overlay-mode",
                "auto",
                "--gpu",
                "null"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "x2linux",
                "disasm",
                "/tmp/shader.bin",
                "--spirv-out",
                "/tmp/out.spv"
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "x2linux",
                "shaders",
                "/tmp/shaders",
                "--recomp",
                "--validate-spirv",
                "--out",
                "/tmp/spv"
            ])
            .is_ok()
        );
    }

    #[test]
    fn disasm_and_shaders_batch_work_on_synthetic_console_container() {
        let tmp = std::env::temp_dir().join(format!("x2l_cli_shaders_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).expect("create tmp dir");

        // v_mul_legacy_f32 v0, v0, v1 ; s_endpgm
        let code = [
            (0x07u32 << 25) | (1u32 << 9) | 256u32,
            (0x17Fu32 << 23) | (0x01u32 << 16),
        ];
        let dxbc = build_synthetic_console_dxbc(5, 0x60, true, &code);
        let shader_path = tmp.join("sample.dxbc");
        let spv_path = tmp.join("sample.spv");
        let batch_out = tmp.join("batch_spv");
        fs::write(&shader_path, &dxbc).expect("write fixture");

        assert_eq!(
            execute_disasm(&shader_path, Some(&spv_path)).expect("disasm"),
            0
        );
        let spv_bytes = fs::read(&spv_path).expect("read spv");
        assert!(spv_bytes.len() >= 20);
        assert_eq!(&spv_bytes[0..4], &0x0723_0203u32.to_le_bytes());

        let validate = x2l_rdna2_spirv::is_spirv_val_available();
        assert_eq!(
            execute_shaders_batch(&tmp, true, validate, Some(&batch_out)).expect("shaders batch"),
            0
        );
        assert!(batch_out.join("sample_s0.spv").is_file());

        let _ = fs::remove_dir_all(&tmp);
    }
}
