//! Content-addressed XDG disk cache for compiled SPIR-V shaders and Vulkan pipeline state.

use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors raised by the persistent XDG shader and pipeline cache.
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("cache I/O failure: {0}")]
    Io(#[from] std::io::Error),
}

/// Handle to a per-title cache directory under `~/.cache/x2linux/<title>/spirv/`.
#[derive(Debug, Clone)]
pub struct PipelineCache {
    root: PathBuf,
}

impl PipelineCache {
    /// Construct a cache handle rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Return the root directory for cached artifacts.
    pub fn root(&self) -> &Path {
        &self.root
    }
}
