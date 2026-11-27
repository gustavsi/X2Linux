//! Platform-level console data formats for X2Linux:
//! - [`bcpack`]: lossless BCPack decompression for BC1–BC7 GPU texture block streams
//! - [`texture`]: RDNA2 (GFX10.3) macro/micro-tiled texture layout & linear detiling
//! - [`xma`]: pure-Rust XMA2 hardware audio stream packet walker and PCM decoder
//! - [`opus`]: RFC 6716 Opus packet framing & PipeWire-ready PCM conversion helpers
//! - [`zlib`]: bounded single-stream and parallel multi-chunk zlib decompression

pub mod bcpack;
mod bcpack_tables;
pub mod opus;
pub mod texture;
pub mod xma;
pub mod zlib;

use thiserror::Error;

pub use zlib::{decompress_zlib, decompress_zlib_bounded, decompress_zlib_parallel};

/// Top-level format error unifying `x2l-formats` decoders.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("{0}")]
    BcPack(#[from] bcpack::Error),
    #[error("texture layout error: {0:?}")]
    Texture(texture::Error),
    #[error("{0}")]
    Xma(#[from] xma::Error),
    #[error("{0}")]
    Opus(#[from] opus::OpusError),
    #[error("{0}")]
    Zlib(#[from] zlib::ZlibError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_bcpack_tables_load_and_reject_truncated_streams() {
        let tables = bcpack::Tables::builtin().expect("parse embedded BCPack tables");
        let mut dst = vec![0u8; 64];
        assert!(bcpack::decode(1, &[], &mut dst, &tables).is_err());
        assert!(bcpack::decode(7, &[0u8; 12], &mut dst, &tables).is_err());
        assert!(bcpack::decode(2, &[0u8; 64], &mut dst, &tables).is_err());
    }

    #[test]
    fn xma_rejects_malformed_and_truncated_streams_without_panicking() {
        assert!(xma::packet(&[], 0).is_none());
        assert!(xma::packet(&[0xFF; 3], 0).is_none());
        let mut dec = xma::Decoder::new(48_000, 2);
        let mut pcm = Vec::new();
        assert!(dec.decode(&[], 0, &mut pcm).is_err());
        assert!(dec.decode(&[0xFF; 8], 64, &mut pcm).is_err());
        assert_eq!(pcm.len(), xma::FRAME_SAMPLES * 2 * 2);
    }
}
