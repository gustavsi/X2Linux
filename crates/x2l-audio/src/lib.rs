//! Native Linux audio backend (`XAudio2` → PipeWire with SDL3 fallback).

use thiserror::Error;
use x2l_common::AudioBackend;

/// Errors raised while initializing or streaming native Linux audio.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("audio backend error: {0}")]
    Backend(String),
}

/// Active audio output selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSessionConfig {
    pub backend: AudioBackend,
    pub sample_rate_hz: u32,
}

impl Default for AudioSessionConfig {
    fn default() -> Self {
        Self {
            backend: AudioBackend::Pipewire,
            sample_rate_hz: 48_000,
        }
    }
}
