//! End-to-end Read-Only Immutability integration test for `x2l-vfs-overlay`.
//!
//! Runs with `harness = false` so the process stays single-threaded, enabling unprivileged
//! `CLONE_NEWUSER | CLONE_NEWNS` Kernel OverlayFS mounting (with automatic `fuse-overlayfs`
//! fallback when kernel userns mounts are blocked by host policy).

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use x2l_vfs_overlay::{OverlayBackend, OverlayMode, OverlayMount, OverlaySpec};

struct TempFixture(PathBuf);

impl TempFixture {
    fn new(label: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("x2l-immutability-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp fixture root");
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn main() -> ExitCode {
    match run_immutability_test() {
        Ok(backend) => {
            println!(
                "test readonly_immutability::overlay_preserves_lowerdir_sha256_tree ({backend:?}) ... ok"
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("test readonly_immutability FAILED: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run_immutability_test() -> Result<OverlayBackend, String> {
    let fixture = TempFixture::new("e2e");
    let lower = fixture.path().join("title_ro");
    let upper = fixture.path().join("state/overlay");
    let work = fixture.path().join("state/work");
    let merged = fixture.path().join("state/merged");

    // 1. Populate nested read-only lowerdir fixture.
    fs::create_dir_all(lower.join("config")).map_err(|e| format!("mkdir config: {e}"))?;
    fs::create_dir_all(lower.join("assets")).map_err(|e| format!("mkdir assets: {e}"))?;
    fs::write(lower.join("config/settings.txt"), b"resolution=3840x2160\n")
        .map_err(|e| format!("write settings.txt: {e}"))?;
    fs::write(
        lower.join("assets/keep.dat"),
        b"original-immutable-keep-bytes",
    )
    .map_err(|e| format!("write keep.dat: {e}"))?;
    fs::write(lower.join("assets/modify.dat"), b"original-before-cow")
        .map_err(|e| format!("write modify.dat: {e}"))?;
    fs::write(lower.join("assets/delete_me.dat"), b"must-survive-whiteout")
        .map_err(|e| format!("write delete_me.dat: {e}"))?;

    let hash_before = snapshot_sha256_tree(&lower)?;

    // 2. Mount overlay in Auto mode (KernelOverlayFs first, FuseOverlayFs fallback).
    let spec = OverlaySpec::new(&lower, &upper, &work, &merged, OverlayMode::Auto)
        .map_err(|e| format!("OverlaySpec::new failed: {e}"))?;

    let mount = OverlayMount::mount(spec).map_err(|e| {
        format!(
            "OverlayMount::mount failed (neither unprivileged userns kernel overlayfs nor fuse-overlayfs succeeded in this environment): {e}"
        )
    })?;
    let backend = mount.backend();
    println!("[x2l-vfs-overlay] Active backend: {backend:?}");

    // 3. Exercise read, copy-on-write overwrite, create, and whiteout delete inside merged/.
    let read_keep =
        fs::read(merged.join("assets/keep.dat")).map_err(|e| format!("read keep.dat: {e}"))?;
    if read_keep != b"original-immutable-keep-bytes" {
        return Err("merged/assets/keep.dat did not match lowerdir content".into());
    }

    fs::write(merged.join("assets/modify.dat"), b"overwritten-in-upperdir")
        .map_err(|e| format!("overwrite modify.dat: {e}"))?;
    let read_modified = fs::read(merged.join("assets/modify.dat"))
        .map_err(|e| format!("reread modify.dat: {e}"))?;
    if read_modified != b"overwritten-in-upperdir" {
        return Err("merged/assets/modify.dat did not reflect CoW overwrite".into());
    }

    fs::create_dir_all(merged.join("new_dir")).map_err(|e| format!("mkdir new_dir: {e}"))?;
    fs::write(merged.join("new_dir/created.txt"), b"fresh-upperdir-file")
        .map_err(|e| format!("create new_dir/created.txt: {e}"))?;

    fs::remove_file(merged.join("assets/delete_me.dat"))
        .map_err(|e| format!("remove delete_me.dat: {e}"))?;
    if merged.join("assets/delete_me.dat").exists() {
        return Err("merged/assets/delete_me.dat should be hidden by whiteout".into());
    }

    // 4. Unmount cleanly.
    mount
        .unmount()
        .map_err(|e| format!("OverlayMount::unmount failed: {e}"))?;

    // 5. Verify all writes and whiteout landed inside upperdir.
    let upper_mod = fs::read(upper.join("assets/modify.dat"))
        .map_err(|e| format!("expected CoW file in upperdir: {e}"))?;
    if upper_mod != b"overwritten-in-upperdir" {
        return Err("upperdir/assets/modify.dat content mismatch".into());
    }

    let upper_new = fs::read(upper.join("new_dir/created.txt"))
        .map_err(|e| format!("expected created file in upperdir: {e}"))?;
    if upper_new != b"fresh-upperdir-file" {
        return Err("upperdir/new_dir/created.txt content mismatch".into());
    }

    if !has_overlay_whiteout(&upper.join("assets"), "delete_me.dat") {
        return Err("expected OverlayFS whiteout marker in upperdir/assets/".into());
    }

    // 6. Verify lowerdir SHA-256 tree is 100% byte-for-byte identical before and after.
    let hash_after = snapshot_sha256_tree(&lower)?;
    if hash_before != hash_after {
        return Err(format!(
            "CRITICAL: lowerdir mutated! before={hash_before:?}, after={hash_after:?}"
        ));
    }

    Ok(backend)
}

fn has_overlay_whiteout(upper_subdir: &Path, deleted_name: &str) -> bool {
    // Kernel OverlayFS & fuse-overlayfs standard whiteout: char device 0:0 at deleted path.
    let direct = upper_subdir.join(deleted_name);
    if let Ok(meta) = fs::symlink_metadata(&direct)
        && meta.file_type().is_char_device()
        && meta.rdev() == 0
    {
        return true;
    }
    // Alternative AUFS/OCI whiteout representation: `.wh.<name>`.
    let wh_file = upper_subdir.join(format!(".wh.{deleted_name}"));
    fs::symlink_metadata(&wh_file).is_ok()
}

fn snapshot_sha256_tree(root: &Path) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    walk_and_hash(root, root, &mut out)?;
    Ok(out)
}

fn walk_and_hash(
    root: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            walk_and_hash(root, &path, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| format!("strip_prefix: {e}"))?
                .to_string_lossy()
                .into_owned();
            let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            out.insert(rel, sha256_hex(&bytes));
        }
    }
    Ok(())
}

/// Minimal pure-Rust FIPS 180-4 SHA-256 for self-contained read-only integrity verification.
fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut msg = input.to_vec();
    let bit_len = (input.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let (chunks, _) = msg.as_chunks::<64>();
    for chunk in chunks {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let o = i * 4;
            *word = u32::from_be_bytes([chunk[o], chunk[o + 1], chunk[o + 2], chunk[o + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    h.iter().map(|word| format!("{word:08x}")).collect()
}
