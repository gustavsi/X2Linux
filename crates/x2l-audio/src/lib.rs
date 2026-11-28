//! Native Linux audio backend (`XAudio2` / `xapu` → PipeWire with deterministic headless fallback)
//! and console audio co-processor state (`AcpHal`, `xapu`, `chad`).

pub mod coproc;
pub mod pipewire;

pub use coproc::AudioCoprocState;
pub use pipewire::{
    ActiveAudioBackend, AudioOutput, AudioRingBuffer, DEFAULT_CHANNELS,
    DEFAULT_RING_CAPACITY_SAMPLES, DEFAULT_SAMPLE_RATE_HZ, NullAudioSink, PipeWireAudioSink,
    mix_add_stereo, probe_pipewire_socket, upmix_and_resample_to_stereo,
};

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
