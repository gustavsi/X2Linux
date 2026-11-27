//! Opus packet framing (RFC 6716) and PCM buffer conversion helpers for `x2l-audio` (PipeWire).

use thiserror::Error;

/// Standard console Opus playback sample rate (48 kHz).
pub const OPUS_SAMPLE_RATE_HZ: u32 = 48_000;

/// Errors raised while inspecting Opus packets or decoding audio streams.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum OpusError {
    #[error("Opus packet is empty or truncated")]
    Truncated,
    #[error("invalid Opus frame count code or framing")]
    InvalidFraming,
    #[error("unsupported channel count {0} (expected 1 or 2)")]
    UnsupportedChannels(u8),
}

/// Opus coding mode from the TOC byte (RFC 6716 §3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

/// Audio bandwidth signaled by the Opus TOC byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusBandwidth {
    Narrowband,
    Mediumband,
    Wideband,
    SuperWideband,
    Fullband,
}

/// Parsed RFC 6716 Opus packet Table-of-Contents (TOC) metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusPacketHeader {
    pub config: u8,
    pub mode: OpusMode,
    pub bandwidth: OpusBandwidth,
    pub stereo: bool,
    pub frame_duration_us: u32,
    pub frame_count: u8,
}

impl OpusPacketHeader {
    /// Number of decoded PCM samples per channel at 48 kHz for this packet.
    pub const fn samples_per_channel_48k(&self) -> usize {
        let per_frame =
            (self.frame_duration_us as usize * OPUS_SAMPLE_RATE_HZ as usize) / 1_000_000;
        per_frame * (self.frame_count as usize)
    }

    /// Number of channels signaled in the TOC header (`1` = mono, `2` = stereo).
    pub const fn channels(&self) -> u8 {
        if self.stereo { 2 } else { 1 }
    }
}

/// Parse an RFC 6716 Opus packet header without panicking on truncated or malformed packets.
pub fn parse_opus_packet_header(packet: &[u8]) -> Result<OpusPacketHeader, OpusError> {
    let &toc = packet.first().ok_or(OpusError::Truncated)?;
    let config = toc >> 3;
    let stereo = (toc & 0x04) != 0;
    let code = toc & 0x03;

    let (mode, bandwidth, frame_duration_us) = match config {
        0..=3 => (
            OpusMode::SilkOnly,
            OpusBandwidth::Narrowband,
            silk_duration_us(config),
        ),
        4..=7 => (
            OpusMode::SilkOnly,
            OpusBandwidth::Mediumband,
            silk_duration_us(config - 4),
        ),
        8..=11 => (
            OpusMode::SilkOnly,
            OpusBandwidth::Wideband,
            silk_duration_us(config - 8),
        ),
        12..=13 => (
            OpusMode::Hybrid,
            OpusBandwidth::SuperWideband,
            if config == 12 { 10_000 } else { 20_000 },
        ),
        14..=15 => (
            OpusMode::Hybrid,
            OpusBandwidth::Fullband,
            if config == 14 { 10_000 } else { 20_000 },
        ),
        16..=19 => (
            OpusMode::CeltOnly,
            OpusBandwidth::Narrowband,
            celt_duration_us(config - 16),
        ),
        20..=23 => (
            OpusMode::CeltOnly,
            OpusBandwidth::Wideband,
            celt_duration_us(config - 20),
        ),
        24..=27 => (
            OpusMode::CeltOnly,
            OpusBandwidth::SuperWideband,
            celt_duration_us(config - 24),
        ),
        28..=31 => (
            OpusMode::CeltOnly,
            OpusBandwidth::Fullband,
            celt_duration_us(config - 28),
        ),
        _ => unreachable!("5-bit config is always 0..=31"),
    };

    let frame_count = match code {
        0 => 1,
        1 | 2 => {
            if packet.len() < 2 {
                return Err(OpusError::Truncated);
            }
            2
        }
        3 => {
            let &count_byte = packet.get(1).ok_or(OpusError::Truncated)?;
            let count = count_byte & 0x3F;
            if count == 0 || count > 48 {
                return Err(OpusError::InvalidFraming);
            }
            count
        }
        _ => unreachable!(),
    };

    // RFC 6716 §3.2.5: total packet duration may not exceed 120 ms.
    let total_us = frame_duration_us
        .checked_mul(u32::from(frame_count))
        .ok_or(OpusError::InvalidFraming)?;
    if total_us > 120_000 {
        return Err(OpusError::InvalidFraming);
    }

    Ok(OpusPacketHeader {
        config,
        mode,
        bandwidth,
        stereo,
        frame_duration_us,
        frame_count,
    })
}

const fn silk_duration_us(idx: u8) -> u32 {
    match idx {
        0 => 10_000,
        1 => 20_000,
        2 => 40_000,
        _ => 60_000,
    }
}

const fn celt_duration_us(idx: u8) -> u32 {
    match idx {
        0 => 2_500,
        1 => 5_000,
        2 => 10_000,
        _ => 20_000,
    }
}

/// Interleaved floating-point PCM buffer ready for PipeWire (`x2l-audio`).
#[derive(Debug, Clone, PartialEq)]
pub struct PcmBuffer {
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub samples: Vec<f32>,
}

impl PcmBuffer {
    /// Convert interleaved 16-bit signed PCM into normalized `[-1.0, 1.0]` f32 PCM.
    pub fn from_i16(sample_rate_hz: u32, channels: u8, pcm: &[i16]) -> Result<Self, OpusError> {
        if channels != 1 && channels != 2 {
            return Err(OpusError::UnsupportedChannels(channels));
        }
        let samples = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
        Ok(Self {
            sample_rate_hz,
            channels,
            samples,
        })
    }

    /// Convert normalized f32 PCM back to clamped interleaved `i16`.
    pub fn to_i16(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_celt_fullband_stereo_20ms_packet() {
        // config = 31 (CELT Fullband 20ms), stereo = 1, code = 0 (1 frame) -> (31 << 3) | 4 = 0xFC
        let hdr = parse_opus_packet_header(&[0xFC, 0x00, 0x01]).expect("valid Opus header");
        assert_eq!(hdr.mode, OpusMode::CeltOnly);
        assert_eq!(hdr.bandwidth, OpusBandwidth::Fullband);
        assert!(hdr.stereo);
        assert_eq!(hdr.channels(), 2);
        assert_eq!(hdr.frame_duration_us, 20_000);
        assert_eq!(hdr.frame_count, 1);
        assert_eq!(hdr.samples_per_channel_48k(), 960);
    }

    #[test]
    fn rejects_empty_truncated_and_overlong_opus_packets() {
        assert_eq!(parse_opus_packet_header(&[]), Err(OpusError::Truncated));
        // code = 3 (arbitrary frame count) with missing count byte
        assert_eq!(parse_opus_packet_header(&[0xFF]), Err(OpusError::Truncated));
        // code = 3 with count = 0
        assert_eq!(
            parse_opus_packet_header(&[0xFF, 0x00]),
            Err(OpusError::InvalidFraming)
        );
        // config = 31 (20ms) * 7 frames = 140ms > 120ms RFC 6716 limit
        assert_eq!(
            parse_opus_packet_header(&[0xFF, 0x07]),
            Err(OpusError::InvalidFraming)
        );
    }

    #[test]
    fn pcm_i16_f32_conversion_roundtrip() {
        let buf = PcmBuffer::from_i16(OPUS_SAMPLE_RATE_HZ, 2, &[0, 16384, -16384, 32767]).unwrap();
        assert_eq!(buf.channels, 2);
        assert_eq!(buf.samples.len(), 4);
        let back = buf.to_i16();
        assert_eq!(back[0], 0);
        assert!((back[1] - 16384).abs() <= 1);
        assert!((back[2] + 16384).abs() <= 1);
        assert!(PcmBuffer::from_i16(48_000, 0, &[]).is_err());
    }
}
