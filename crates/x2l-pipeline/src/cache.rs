// Adapted from Marvisak/VIctory (https://github.com/Marvisak/VIctory),
// commit 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-pipeline/src/cache.rs
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Modified for X2Linux on 2026-11-27: replaced blake3/anyhow with pure-Rust FIPS 180-4 SHA-256
// and `thiserror`, added `fsync` before atomic rename, and wired directly to `x2l_common::XdgPaths`.

use crate::Converter;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use x2l_common::{XdgError, XdgPaths};

/// Errors raised by the persistent XDG content-addressed cache.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(
        "cache path '{0}' is inside the X2Linux source repository; refusing to write derived data there"
    )]
    InsideRepository(PathBuf),
    #[error("XDG path error: {0}")]
    Xdg(#[from] XdgError),
    #[error("converter '{id}' v{version} failed: {message}")]
    Convert {
        id: &'static str,
        version: u32,
        message: String,
    },
    #[error("cache I/O failure: {0}")]
    Io(#[from] io::Error),
}

/// Content-addressed SHA-256 cache key derived from `(converter.id(), converter.version(), input)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey([u8; 32]);

impl CacheKey {
    /// Compute the deterministic cache key for a converter and input payload.
    pub fn new(converter: &dyn Converter, input: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(converter.id().as_bytes());
        hasher.update(&[0]);
        hasher.update(&converter.version().to_le_bytes());
        hasher.update(input);
        Self(hasher.finalize())
    }

    /// Lowercase 64-character hexadecimal representation.
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for b in &self.0 {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }
}

/// Content-addressed on-disk cache using atomic write-then-rename publication.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

/// Alias kept for ergonomics across workspace crates.
pub type PipelineCache = Cache;

impl Cache {
    /// Open or create a cache rooted at `root`. Refuses locations inside the X2Linux repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, PipelineError> {
        let root = root.into();
        ensure_outside_repo(&root)?;
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Open a per-title cache under `$XDG_CACHE_HOME/x2linux/<title_id>/`.
    pub fn open_xdg_title(xdg: &XdgPaths, title_id: &str) -> Result<Self, PipelineError> {
        let root = xdg.title_cache_dir(title_id)?;
        Self::open(root)
    }

    /// Open a per-title SPIR-V shader cache under `$XDG_CACHE_HOME/x2linux/<title_id>/spirv/`.
    pub fn open_xdg_spirv(xdg: &XdgPaths, title_id: &str) -> Result<Self, PipelineError> {
        let root = xdg.title_spirv_cache_dir(title_id)?;
        Self::open(root)
    }

    /// Return the root directory for cached artifacts.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the two-level sharded file path `<root>/<xx>/<rest>` for `key`.
    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Return the cached artifact path for `input`, running `converter` and publishing
    /// atomically (staging file → `sync_all` → `rename`) on cache miss.
    pub fn get_or_convert(
        &self,
        converter: &dyn Converter,
        input: &[u8],
    ) -> Result<PathBuf, PipelineError> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if path.is_file() {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(path);
        }

        let output = converter.convert(input)?;
        let dir = path.parent().expect("sharded cache path has a parent");
        fs::create_dir_all(dir)?;

        let (tmp, mut file) = tempfile_in(dir)?;
        let write_res = file.write_all(&output).and_then(|()| file.sync_all());
        drop(file);
        if let Err(err) = write_res {
            let _ = fs::remove_file(&tmp);
            return Err(err.into());
        }

        if let Err(err) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            if !path.is_file() {
                return Err(err.into());
            }
        }

        Ok(path)
    }
}

fn ensure_outside_repo(candidate: &Path) -> Result<(), PipelineError> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    let resolved = resolve_nearest_existing(candidate);
    if resolved.starts_with(&repo_root) {
        return Err(PipelineError::InsideRepository(candidate.to_path_buf()));
    }
    Ok(())
}

fn resolve_nearest_existing(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let mut cur = abs.as_path();
    let mut tail = Vec::new();
    loop {
        if let Ok(canon) = cur.canonicalize() {
            let mut out = canon;
            for comp in tail.into_iter().rev() {
                out.push(comp);
            }
            return out;
        }
        match (cur.parent(), cur.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name.to_os_string());
                cur = parent;
            }
            _ => return abs,
        }
    }
}

fn tempfile_in(dir: &Path) -> io::Result<(PathBuf, File)> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    create_tempfile_in(dir, &COUNTER)
}

fn create_tempfile_in(dir: &Path, counter: &AtomicU64) -> io::Result<(PathBuf, File)> {
    for _ in 0..128 {
        let n = counter.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(".tmp-{}-{n}", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not create a unique cache staging file after 128 attempts",
    ))
}

/// Self-contained FIPS 180-4 SHA-256 streaming hasher (zero external crate dependencies).
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            h: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(input.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        let (chunks, rest) = input.as_chunks::<64>();
        for block in chunks {
            self.compress(block);
        }
        if !rest.is_empty() {
            self.buf[..rest.len()].copy_from_slice(rest);
            self.buf_len = rest.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);

        let mut out = [0u8; 32];
        for (i, word) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, chunk: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a_2f98,
            0x7137_4491,
            0xb5c0_fbcf,
            0xe9b5_dba5,
            0x3956_c25b,
            0x59f1_11f1,
            0x923f_82a4,
            0xab1c_5ed5,
            0xd807_aa98,
            0x1283_5b01,
            0x2431_85be,
            0x550c_7dc3,
            0x72be_5d74,
            0x80de_b1fe,
            0x9bdc_06a7,
            0xc19b_f174,
            0xe49b_69c1,
            0xefbe_4786,
            0x0fc1_9dc6,
            0x240c_a1cc,
            0x2de9_2c6f,
            0x4a74_84aa,
            0x5cb0_a9dc,
            0x76f9_88da,
            0x983e_5152,
            0xa831_c66d,
            0xb003_27c8,
            0xbf59_7fc7,
            0xc6e0_0bf3,
            0xd5a7_9147,
            0x06ca_6351,
            0x1429_2967,
            0x27b7_0a85,
            0x2e1b_2138,
            0x4d2c_6dfc,
            0x5338_0d13,
            0x650a_7354,
            0x766a_0abb,
            0x81c2_c92e,
            0x9272_2c85,
            0xa2bf_e8a1,
            0xa81a_664b,
            0xc24b_8b70,
            0xc76c_51a3,
            0xd192_e819,
            0xd699_0624,
            0xf40e_3585,
            0x106a_a070,
            0x19a4_c116,
            0x1e37_6c08,
            0x2748_774c,
            0x34b0_bcb5,
            0x391c_0cb3,
            0x4ed8_aa4a,
            0x5b9c_ca4f,
            0x682e_6ff3,
            0x748f_82ee,
            0x78a5_636f,
            0x84c8_7814,
            0x8cc7_0208,
            0x90be_fffa,
            0xa450_6ceb,
            0xbef9_a3f7,
            0xc671_78f2,
        ];

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

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = self.h;
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

        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(hh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let id = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("x2l-pipeline-{label}-{}-{id}", std::process::id()));
            fs::create_dir_all(&path).expect("create temp dir");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct Upper {
        id: &'static str,
        version: u32,
        runs: AtomicUsize,
    }

    impl Converter for Upper {
        fn id(&self) -> &'static str {
            self.id
        }

        fn version(&self) -> u32 {
            self.version
        }

        fn convert(&self, input: &[u8]) -> Result<Vec<u8>, PipelineError> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            Ok(input.to_ascii_uppercase())
        }
    }

    #[test]
    fn converts_once_then_hits_and_persists_atomically() {
        let tmp = TempDir::new("hit");
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            id: "test.upper",
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(a, b);
        assert_eq!(fs::read(&a).unwrap(), b"ABC");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
        // No leftover staging files remain beside the published entry.
        assert_eq!(fs::read_dir(a.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn version_and_converter_id_invalidate_cache_key() {
        let v1 = Upper {
            id: "shader.spirv",
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let v2 = Upper {
            id: "shader.spirv",
            version: 2,
            runs: AtomicUsize::new(0),
        };
        let other = Upper {
            id: "texture.detile",
            version: 1,
            runs: AtomicUsize::new(0),
        };
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&v2, b"x"));
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&other, b"x"));
    }

    #[test]
    fn opens_under_xdg_cache_home_for_title_and_spirv() {
        let tmp = TempDir::new("xdg");
        let xdg_cache = tmp.path().join("xdg-cache");
        let env = HashMap::from([
            ("HOME", tmp.path().to_string_lossy().into_owned()),
            ("XDG_CACHE_HOME", xdg_cache.to_string_lossy().into_owned()),
        ]);
        let xdg = XdgPaths::from_lookup(|k| env.get(k).cloned()).unwrap();

        let title_cache = Cache::open_xdg_title(&xdg, "synthetic-title").unwrap();
        assert_eq!(
            title_cache.root(),
            xdg_cache.join("x2linux").join("synthetic-title")
        );

        let spirv_cache = Cache::open_xdg_spirv(&xdg, "synthetic-title").unwrap();
        assert_eq!(
            spirv_cache.root(),
            xdg_cache
                .join("x2linux")
                .join("synthetic-title")
                .join("spirv")
        );
    }

    #[test]
    fn failed_publication_cleans_up_staging_file() {
        let tmp = TempDir::new("fail-pub");
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            id: "test.upper",
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        fs::create_dir_all(&path).unwrap();

        assert!(cache.get_or_convert(&conv, b"abc").is_err());
        assert!(path.is_dir());
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn staging_collision_retries_are_bounded() {
        let tmp = TempDir::new("collisions");
        let counter = AtomicU64::new(0);
        for n in 0..128 {
            fs::write(
                tmp.path().join(format!(".tmp-{}-{n}", std::process::id())),
                b"stale",
            )
            .unwrap();
        }
        let err = create_tempfile_in(tmp.path(), &counter).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(counter.load(Ordering::Relaxed), 128);
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(matches!(
            Cache::open(inside),
            Err(PipelineError::InsideRepository(_))
        ));
    }
}
