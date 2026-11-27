// Adapted from Marvisak/VIctory (https://github.com/Marvisak/VIctory),
// commit 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-pipeline/src/lib.rs
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Modified for X2Linux on 2026-11-27: wired to Linux XDG cache paths (`x2l-common`),
// pure-Rust SHA-256 content addressing, and typed `PipelineError`.

//! Content-addressed Linux XDG disk cache for deterministic conversions (SPIR-V shaders,
//! detiled textures, and compiled pipeline state) under `$XDG_CACHE_HOME/x2linux/<title>/`.
//!
//! Nothing cached here is ever written inside the X2Linux repository or the user's read-only
//! software directory.

mod cache;

pub use cache::{Cache, CacheKey, PipelineCache, PipelineError};

/// A deterministic transformation of input bytes into cached output bytes.
pub trait Converter: Send + Sync {
    /// Stable identifier, e.g. `"shader.rdna2_spirv"` or `"texture.detile"`.
    fn id(&self) -> &'static str;
    /// Bump whenever the output format or compiler logic changes to invalidate stale entries.
    fn version(&self) -> u32;
    /// Perform the deterministic conversion.
    fn convert(&self, input: &[u8]) -> Result<Vec<u8>, PipelineError>;
}
