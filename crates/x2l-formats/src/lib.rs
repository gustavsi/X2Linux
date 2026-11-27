//! Console data formats: GPU texture detiling, parallel zlib/BCPack decompression, and Opus/XMA2 audio decoding.

use thiserror::Error;

/// Errors raised while decoding console textures, compressed streams, or audio containers.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("unsupported format: {0}")]
    Unsupported(&'static str),
    #[error("I/O or decompression failure: {0}")]
    Io(#[from] std::io::Error),
}

/// Decompress a zlib-framed payload using `flate2`.
pub fn decompress_zlib(input: &[u8]) -> Result<Vec<u8>, FormatError> {
    use std::io::Read;
    let mut decoder = flate2::read::ZlibDecoder::new(input);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    Ok(out)
}
