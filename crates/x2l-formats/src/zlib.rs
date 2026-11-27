//! Single-stream and parallel multi-chunk zlib decompression with output size bounds.

use flate2::read::ZlibDecoder;
use std::io::Read;
use thiserror::Error;

/// Default maximum decompressed output size per chunk (256 MiB) guarding against decompression bombs.
pub const DEFAULT_MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;

/// Errors raised while decompressing zlib payloads.
#[derive(Debug, Error)]
pub enum ZlibError {
    #[error("zlib stream exceeds maximum permitted output size ({limit} bytes)")]
    OutputLimitExceeded { limit: usize },
    #[error("zlib decompression failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Decompress a single zlib-framed payload with an explicit maximum output limit.
pub fn decompress_zlib_bounded(
    input: &[u8],
    max_output_bytes: usize,
) -> Result<Vec<u8>, ZlibError> {
    if input.is_empty() {
        return Err(ZlibError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "empty zlib payload",
        )));
    }
    let mut decoder = ZlibDecoder::new(input);
    let mut out = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let next_len = out
            .len()
            .checked_add(n)
            .ok_or(ZlibError::OutputLimitExceeded {
                limit: max_output_bytes,
            })?;
        if next_len > max_output_bytes {
            return Err(ZlibError::OutputLimitExceeded {
                limit: max_output_bytes,
            });
        }
        out.extend_from_slice(&buf[..n]);
    }
    Ok(out)
}

/// Decompress a single zlib-framed payload using [`DEFAULT_MAX_DECOMPRESSED_BYTES`].
pub fn decompress_zlib(input: &[u8]) -> Result<Vec<u8>, ZlibError> {
    decompress_zlib_bounded(input, DEFAULT_MAX_DECOMPRESSED_BYTES)
}

/// Decompress multiple independent zlib-framed chunks in parallel using scoped OS threads,
/// preserving input chunk order.
pub fn decompress_zlib_parallel(
    chunks: &[&[u8]],
    max_chunk_bytes: usize,
) -> Result<Vec<Vec<u8>>, ZlibError> {
    if chunks.len() <= 1 {
        return chunks
            .iter()
            .map(|c| decompress_zlib_bounded(c, max_chunk_bytes))
            .collect();
    }

    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(chunks.len());
    let batch_size = chunks.len().div_ceil(worker_count);

    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .chunks(batch_size)
            .map(|slice| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|chunk| decompress_zlib_bounded(chunk, max_chunk_bytes))
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .collect();

        let mut results = Vec::with_capacity(chunks.len());
        for handle in handles {
            let batch = handle.join().map_err(|_| {
                ZlibError::Io(std::io::Error::other("parallel zlib worker panicked"))
            })??;
            results.extend(batch);
        }
        Ok(results)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use std::io::Write;

    fn compress(bytes: &[u8]) -> Vec<u8> {
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(bytes).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn parallel_zlib_roundtrip_preserves_order() {
        let originals: Vec<Vec<u8>> = (0..16)
            .map(|i| vec![(i * 17) as u8; 1024 + i * 37])
            .collect();
        let compressed: Vec<Vec<u8>> = originals.iter().map(|b| compress(b)).collect();
        let refs: Vec<&[u8]> = compressed.iter().map(Vec::as_slice).collect();

        let decoded = decompress_zlib_parallel(&refs, 1024 * 1024).expect("parallel decompress");
        assert_eq!(decoded, originals);
    }

    #[test]
    fn rejects_malformed_truncated_and_oversized_zlib_streams() {
        assert!(decompress_zlib(&[]).is_err());
        assert!(decompress_zlib(b"not-a-zlib-stream").is_err());

        let payload = vec![0xAB; 4096];
        let z = compress(&payload);
        assert!(decompress_zlib(&z[..z.len() / 2]).is_err());
        assert!(matches!(
            decompress_zlib_bounded(&z, 1024),
            Err(ZlibError::OutputLimitExceeded { limit: 1024 })
        ));
    }
}
