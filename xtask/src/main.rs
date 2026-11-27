//! Repository automation tasks for X2Linux (`cargo xtask <command>`).

use std::env;
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Maximum permitted size for a staged file (5 MiB).
pub const MAX_STAGED_FILE_BYTES: u64 = 5 * 1024 * 1024;

/// File extensions that must never be committed to this repository.
pub const BLOCKED_EXTENSIONS: &[&str] = &["exe", "dll", "xvc", "msixvc", "pak", "bin"];

/// Pre-commit hook script enforcing X2Linux's clean-room asset hygiene policy.
pub const PRE_COMMIT_HOOK: &str = r#"#!/usr/bin/env bash
set -euo pipefail

# X2Linux pre-commit asset-policy guard (installed by `cargo xtask install-hooks`).
# Blocks large binaries (> 5 MiB), console/Windows binaries, raw shader dumps, and report.json.

MAX_BYTES=$((5 * 1024 * 1024))
FAILED=0

while IFS= read -r -d '' path; do
  # Skip deletions.
  if [[ ! -e "$path" ]]; then
    continue
  fi

  lower="${path,,}"
  base="$(basename "$lower")"

  if [[ "$base" == "report.json" ]]; then
    echo "[x2linux pre-commit] BLOCKED: diagnostic telemetry file '$path' must not be committed." >&2
    FAILED=1
    continue
  fi

  if [[ "$lower" == shaders_raw/* || "$lower" == */shaders_raw/* ]]; then
    echo "[x2linux pre-commit] BLOCKED: raw shader capture path '$path' must not be committed." >&2
    FAILED=1
    continue
  fi

  case "$base" in
    *.exe|*.dll|*.xvc|*.msixvc|*.pak|*.bin)
      echo "[x2linux pre-commit] BLOCKED: forbidden binary/archive extension in '$path'." >&2
      FAILED=1
      continue
      ;;
  esac

  size=$(wc -c < "$path")
  if (( size > MAX_BYTES )); then
    echo "[x2linux pre-commit] BLOCKED: '$path' is ${size} bytes (limit: ${MAX_BYTES} bytes)." >&2
    FAILED=1
  fi
done < <(git diff --cached --name-only --diff-filter=ACM -z)

if (( FAILED != 0 )); then
  echo "[x2linux pre-commit] Commit aborted by X2Linux asset-policy guard." >&2
  exit 1
fi
"#;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("install-hooks") => match install_hooks_at(&repo_root()) {
            Ok(hook_path) => {
                println!(
                    "[xtask] Installed pre-commit asset-policy hook at {}",
                    hook_path.display()
                );
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("[xtask] Failed to install hooks: {err}");
                ExitCode::FAILURE
            }
        },
        Some("--help" | "-h" | "help") | None => {
            print_help();
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("[xtask] Unknown subcommand: {other}");
            print_help();
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!("X2Linux workspace automation (cargo xtask)");
    println!();
    println!("USAGE:");
    println!("    cargo xtask <SUBCOMMAND>");
    println!();
    println!("SUBCOMMANDS:");
    println!("    install-hooks    Install Git pre-commit hook enforcing clean-room asset hygiene");
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives directly under workspace root")
        .to_path_buf()
}

/// Install the pre-commit asset-policy hook inside `<repo>/.git/hooks/pre-commit`.
pub fn install_hooks_at(repo: &Path) -> io::Result<PathBuf> {
    let git_dir = repo.join(".git");
    if !git_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no .git directory found at {}", git_dir.display()),
        ));
    }
    let hooks_dir = git_dir.join("hooks");
    fs::create_dir_all(&hooks_dir)?;
    let hook_path = hooks_dir.join("pre-commit");
    fs::write(&hook_path, PRE_COMMIT_HOOK)?;
    let mut perms = fs::metadata(&hook_path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&hook_path, perms)?;
    Ok(hook_path)
}

/// Evaluate whether a staged path and file size violate X2Linux's asset hygiene policy.
pub fn check_asset_policy(rel_path: &Path, size_bytes: u64) -> Result<(), String> {
    let normalized = rel_path.to_string_lossy().to_ascii_lowercase();
    let file_name = rel_path
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    if file_name == "report.json" {
        return Err(format!(
            "telemetry file '{}' is blocked",
            rel_path.display()
        ));
    }
    if normalized.starts_with("shaders_raw/") || normalized.contains("/shaders_raw/") {
        return Err(format!(
            "raw shader capture path '{}' is blocked",
            rel_path.display()
        ));
    }
    if let Some(ext) = rel_path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        && BLOCKED_EXTENSIONS.contains(&ext.as_str())
    {
        return Err(format!(
            "forbidden extension '.{ext}' in '{}'",
            rel_path.display()
        ));
    }
    if size_bytes > MAX_STAGED_FILE_BYTES {
        return Err(format!(
            "file '{}' exceeds {} byte limit ({size_bytes} bytes)",
            rel_path.display(),
            MAX_STAGED_FILE_BYTES
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("x2l-xtask-{label}-{}-{id}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn policy_blocks_forbidden_paths_and_sizes() {
        assert!(check_asset_policy(Path::new("src/main.rs"), 1024).is_ok());
        assert!(check_asset_policy(Path::new("report.json"), 128).is_err());
        assert!(check_asset_policy(Path::new("tmp/report.json"), 128).is_err());
        assert!(check_asset_policy(Path::new("shaders_raw/0001.spv"), 64).is_err());
        assert!(check_asset_policy(Path::new("cache/shaders_raw/blob.dat"), 64).is_err());
        assert!(check_asset_policy(Path::new("Title.EXE"), 64).is_err());
        assert!(check_asset_policy(Path::new("kernelx.dll"), 64).is_err());
        assert!(check_asset_policy(Path::new("pkg.xvc"), 64).is_err());
        assert!(check_asset_policy(Path::new("pkg.msixvc"), 64).is_err());
        assert!(check_asset_policy(Path::new("data.pak"), 64).is_err());
        assert!(check_asset_policy(Path::new("dump.bin"), 64).is_err());
        assert!(check_asset_policy(Path::new("big.dat"), MAX_STAGED_FILE_BYTES + 1).is_err());
        assert!(check_asset_policy(Path::new("exact.dat"), MAX_STAGED_FILE_BYTES).is_ok());
    }

    #[test]
    fn installs_executable_pre_commit_hook() {
        let dir = temp_dir("install-hook");
        fs::create_dir_all(dir.join(".git")).expect("create .git dir");

        let hook_path = install_hooks_at(&dir).expect("install hook");
        assert!(hook_path.is_file());
        let mode = fs::metadata(&hook_path)
            .expect("hook metadata")
            .permissions()
            .mode();
        assert_ne!(mode & 0o111, 0, "hook must be executable");

        let contents = fs::read_to_string(&hook_path).expect("read hook");
        assert!(contents.contains("report.json"));
        assert!(contents.contains("shaders_raw/"));

        let _ = fs::remove_dir_all(&dir);
    }
}
