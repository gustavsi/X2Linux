// Container format layout informed by xcl-dev/xcl (https://github.com/xcl-dev/xcl,
// commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/gpu/xbox.rs) and
// Marvisak/VIctory (https://github.com/Marvisak/VIctory,
// commit 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-shader/src/dxbc.rs & native.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27 with strict bounds-checked parsing and zero panics.

//! Console shader container classifier and unpacker (`DXBC/XDAT/XBCH/FSSM`, `DXIL`, `ELF`, raw RDNA2).

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;
use x2l_formats::decompress_zlib_bounded;

/// Maximum permitted decompressed size for a single FSSM payload (64 MiB).
pub const MAX_FSSM_BYTES: usize = 64 * 1024 * 1024;
/// Maximum permitted aggregate decompressed output across all variants in one container (128 MiB).
pub const MAX_TOTAL_FSSM_BYTES: usize = 128 * 1024 * 1024;
/// Maximum DXBC part count accepted before rejecting as malformed.
const MAX_DXBC_PARTS: usize = 256;
/// Maximum XDAT entry count accepted before rejecting as malformed.
const MAX_XDAT_ENTRIES: usize = 256;

const MAGIC_DXBC: &[u8; 4] = b"DXBC";
const MAGIC_DXIL: &[u8; 4] = b"DXIL";
const MAGIC_NOBC: &[u8; 4] = b"NOBC";
const MAGIC_XDAT: &[u8; 4] = b"XDAT";
const MAGIC_XBCH: &[u8; 4] = b"XBCH";
const MAGIC_FSSM: &[u8; 4] = b"FSSM";
const MAGIC_ELF: &[u8; 4] = b"\x7fELF";

const XDAT_KIND_XBCH: u32 = 0x0D;
const XBCH_VERSION_3: u32 = 3;
const FSSM_VERSION_101: u32 = 0x101;
const ABSENT_U64: u64 = 0xDEAD_BEEF_DEAD_BEEF;

const S_ENDPGM: u32 = 0xBF81_0000;
const S_CODE_END: u32 = 0xBF9F_0000;

/// FSSM hardware stage record pointer slots inside the FSSM payload (relative to FSSM + 0x10).
const FSSM_STAGE_SLOTS: &[(usize, HwStage)] = &[
    (0x08, HwStage::Ls),
    (0x10, HwStage::Hs),
    (0x18, HwStage::Es),
    (0x20, HwStage::Gs),
    (0x28, HwStage::Vs),
    (0x30, HwStage::Ps),
    (0x48, HwStage::Ngg),
    (0x60, HwStage::Cs),
];

/// Errors raised while parsing or validating console shader containers.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ContainerError {
    #[error("shader blob is empty or truncated while reading {0}")]
    Truncated(&'static str),
    #[error("invalid container magic or version: {0}")]
    InvalidHeader(String),
    #[error("checked offset/size overflow while reading {0}")]
    Overflow(&'static str),
    #[error("out-of-bounds slice while reading {0}")]
    OutOfBounds(&'static str),
    #[error("unaligned or empty RDNA2 code slice ({0} bytes)")]
    InvalidCodeSlice(usize),
    #[error("XBCH zlib decompression failed: {0}")]
    Decompress(String),
    #[error("decompressed FSSM size mismatch: expected {expected} bytes, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },
    #[error("container exceeds safety limit: {0}")]
    LimitExceeded(String),
}

/// Top-level classification of an input shader blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainerKind {
    ConsoleDxbc,
    StandaloneXbch,
    StandaloneFssm,
    Dxil,
    Elf,
    RawRdna2,
    Unknown,
}

impl fmt::Display for ContainerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConsoleDxbc => f.write_str("console_dxbc"),
            Self::StandaloneXbch => f.write_str("standalone_xbch"),
            Self::StandaloneFssm => f.write_str("standalone_fssm"),
            Self::Dxil => f.write_str("dxil"),
            Self::Elf => f.write_str("elf"),
            Self::RawRdna2 => f.write_str("raw_rdna2"),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

/// High-level Direct3D / console API shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShaderStage {
    Pixel,
    Vertex,
    Geometry,
    Hull,
    Domain,
    Compute,
    Library,
    Mesh,
    Amplification,
    Unknown(u16),
}

impl ShaderStage {
    pub fn from_dxil_kind(kind: u16) -> Self {
        match kind {
            0 => Self::Pixel,
            1 => Self::Vertex,
            2 => Self::Geometry,
            3 => Self::Hull,
            4 => Self::Domain,
            5 => Self::Compute,
            6 => Self::Library,
            13 => Self::Mesh,
            14 => Self::Amplification,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pixel => f.write_str("PS"),
            Self::Vertex => f.write_str("VS"),
            Self::Geometry => f.write_str("GS"),
            Self::Hull => f.write_str("HS"),
            Self::Domain => f.write_str("DS"),
            Self::Compute => f.write_str("CS"),
            Self::Library => f.write_str("LIB"),
            Self::Mesh => f.write_str("MS"),
            Self::Amplification => f.write_str("AS"),
            Self::Unknown(v) => write!(f, "STAGE({v})"),
        }
    }
}

/// RDNA2 / PAL hardware shader stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HwStage {
    Ls,
    Hs,
    Es,
    Gs,
    Vs,
    Ps,
    Cs,
    Ngg,
    Unknown(u8),
}

impl HwStage {
    pub fn from_hw_id(id: u8) -> Self {
        match id {
            0 => Self::Ls,
            1 => Self::Hs,
            2 => Self::Es,
            3 => Self::Gs,
            4 => Self::Vs,
            5 => Self::Ps,
            6 => Self::Cs,
            7 => Self::Ngg,
            other => Self::Unknown(other),
        }
    }
}

impl fmt::Display for HwStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ls => f.write_str("LS"),
            Self::Hs => f.write_str("HS"),
            Self::Es => f.write_str("ES"),
            Self::Gs => f.write_str("GS"),
            Self::Vs => f.write_str("VS"),
            Self::Ps => f.write_str("PS"),
            Self::Cs => f.write_str("CS"),
            Self::Ngg => f.write_str("NGG"),
            Self::Unknown(v) => write!(f, "HW({v})"),
        }
    }
}

/// Sub-region classification inside an extracted shader code slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentKind {
    Program,
    EmbeddedDataPool,
    Padding,
}

/// Byte span inside an extracted code slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeSegment {
    pub offset: usize,
    pub len: usize,
    pub kind: SegmentKind,
}

/// One unpacked hardware-stage variant extracted from an FSSM payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FssmVariant {
    pub xbch_offset: usize,
    pub xbch_version: u32,
    pub compressed_size: usize,
    pub fssm_version: u32,
    pub hw_stage: HwStage,
    pub wave32: bool,
    #[serde(skip_serializing)]
    pub code: Vec<u8>,
    pub code_len: usize,
    pub segments: Vec<CodeSegment>,
}

impl FssmVariant {
    /// Return the extracted code slice as little-endian instruction DWORDs.
    pub fn code_words(&self) -> Vec<u32> {
        let (chunks, _) = self.code.as_chunks::<4>();
        chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
    }
}

/// One DXBC input or output signature parameter (`ISG1`/`OSG1` or `ISGN`/`OSGN`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignatureElement {
    pub stream: u32,
    pub semantic_name: String,
    pub semantic_index: u32,
    pub system_value_type: u32,
    pub component_type: u32,
    pub register: u32,
    pub mask: u8,
    pub rw_mask: u8,
    pub min_precision: u32,
}

/// Parsed DXBC input and output stage signatures (`ISG1`/`ISGN` and `OSG1`/`OSGN`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageSignature {
    pub inputs: Vec<SignatureElement>,
    pub outputs: Vec<SignatureElement>,
}

/// Parsed console shader container metadata and extracted native RDNA2 code variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParsedContainer {
    pub kind: ContainerKind,
    pub api_stage: Option<ShaderStage>,
    pub shader_model: Option<(u8, u8)>,
    pub has_nobc_placeholder: bool,
    pub content_sha256: String,
    pub variants: Vec<FssmVariant>,
    #[serde(default)]
    pub signature: StageSignature,
    #[serde(skip_serializing)]
    pub raw_code: Option<Vec<u8>>,
}

impl ParsedContainer {
    /// Return the primary RDNA2 instruction DWORD slice (first FSSM variant or raw code).
    pub fn primary_code_words(&self) -> Option<Vec<u32>> {
        if let Some(first) = self.variants.first() {
            return Some(first.code_words());
        }
        self.raw_code.as_ref().map(|bytes| {
            let (chunks, _) = bytes.as_chunks::<4>();
            chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
        })
    }
}

fn rd_u32(b: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let s = b.get(off..end)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn rd_u64(b: &[u8], off: usize) -> Option<u64> {
    let end = off.checked_add(8)?;
    let s = b.get(off..end)?;
    Some(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// Quickly classify an input shader blob by header signature.
pub fn classify(bytes: &[u8]) -> ContainerKind {
    if bytes.len() < 4 {
        return ContainerKind::Unknown;
    }
    let magic = &bytes[0..4];
    if magic == MAGIC_DXBC {
        if has_console_xdat_in_dxbc(bytes) {
            return ContainerKind::ConsoleDxbc;
        }
        return ContainerKind::Dxil;
    }
    if magic == MAGIC_XBCH {
        return ContainerKind::StandaloneXbch;
    }
    if magic == MAGIC_FSSM {
        return ContainerKind::StandaloneFssm;
    }
    if magic == MAGIC_ELF {
        return ContainerKind::Elf;
    }
    if bytes.len() >= 12 && &bytes[8..12] == MAGIC_DXIL {
        return ContainerKind::Dxil;
    }
    if bytes.len().is_multiple_of(4) && looks_like_rdna2(bytes) {
        return ContainerKind::RawRdna2;
    }
    ContainerKind::Unknown
}

fn looks_like_rdna2(bytes: &[u8]) -> bool {
    let (chunks, rem) = bytes.as_chunks::<4>();
    if !rem.is_empty() || chunks.is_empty() {
        return false;
    }
    chunks.iter().any(|c| {
        let w = u32::from_le_bytes(*c);
        w == S_ENDPGM || w == S_CODE_END || (w >> 23) == 0x17F
    })
}

fn has_console_xdat_in_dxbc(bytes: &[u8]) -> bool {
    let Ok(parts) = dxbc_parts(bytes) else {
        return false;
    };
    parts.into_iter().any(|(tag, body)| {
        &tag == MAGIC_DXIL && body.get(0x18..0x1C) == Some(MAGIC_XDAT.as_slice())
    })
}

/// Parse a console shader container, standalone `XBCH`/`FSSM`, or raw RDNA2 code blob.
pub fn parse_container(bytes: &[u8]) -> Result<ParsedContainer, ContainerError> {
    if bytes.len() < 4 {
        return Err(ContainerError::Truncated("container header"));
    }
    let content_sha256 = sha256_hex(bytes);
    let kind = classify(bytes);

    match kind {
        ContainerKind::ConsoleDxbc => parse_console_dxbc(bytes, content_sha256),
        ContainerKind::StandaloneXbch => {
            let mut total = 0usize;
            let variants = parse_xbch(bytes, 0, &mut total)?;
            Ok(ParsedContainer {
                kind,
                api_stage: None,
                shader_model: None,
                has_nobc_placeholder: false,
                content_sha256,
                variants,
                signature: StageSignature::default(),
                raw_code: None,
            })
        }
        ContainerKind::StandaloneFssm => {
            let variants = parse_fssm(bytes, 0, 0, bytes.len())?;
            Ok(ParsedContainer {
                kind,
                api_stage: None,
                shader_model: None,
                has_nobc_placeholder: false,
                content_sha256,
                variants,
                signature: StageSignature::default(),
                raw_code: None,
            })
        }
        ContainerKind::Dxil => parse_dxil_metadata(bytes, content_sha256),
        ContainerKind::Elf => Ok(ParsedContainer {
            kind,
            api_stage: None,
            shader_model: None,
            has_nobc_placeholder: false,
            content_sha256,
            variants: Vec::new(),
            signature: StageSignature::default(),
            raw_code: None,
        }),
        ContainerKind::RawRdna2 => {
            if !bytes.len().is_multiple_of(4) || bytes.is_empty() {
                return Err(ContainerError::InvalidCodeSlice(bytes.len()));
            }
            Ok(ParsedContainer {
                kind,
                api_stage: None,
                shader_model: None,
                has_nobc_placeholder: false,
                content_sha256,
                variants: Vec::new(),
                signature: StageSignature::default(),
                raw_code: Some(bytes.to_vec()),
            })
        }
        ContainerKind::Unknown => Err(ContainerError::InvalidHeader(format!(
            "unrecognized magic {:02x} {:02x} {:02x} {:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3]
        ))),
    }
}

/// Parse a DXBC `ISG1`/`OSG1` (`extended = true`, 32-byte rows) or `ISGN`/`OSGN`
/// (`extended = false`, 24-byte rows) chunk body with strict bounds checking.
pub fn parse_dxbc_signature(
    body: &[u8],
    extended: bool,
) -> Result<Vec<SignatureElement>, ContainerError> {
    if body.len() < 8 {
        return Err(ContainerError::Truncated("DXBC signature header"));
    }
    let count = rd_u32(body, 0).ok_or(ContainerError::Truncated("signature count"))? as usize;
    if count > MAX_DXBC_PARTS {
        return Err(ContainerError::LimitExceeded(format!(
            "signature element count {count} exceeds limit {MAX_DXBC_PARTS}"
        )));
    }
    let table_off =
        rd_u32(body, 4).ok_or(ContainerError::Truncated("signature table offset"))? as usize;
    let row_size = if extended { 32usize } else { 24usize };
    let table_bytes = count
        .checked_mul(row_size)
        .ok_or(ContainerError::Overflow("signature table size"))?;
    let table_end = table_off
        .checked_add(table_bytes)
        .ok_or(ContainerError::Overflow("signature table end"))?;
    let table = body
        .get(table_off..table_end)
        .ok_or(ContainerError::OutOfBounds("signature element table"))?;

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let row = &table[i * row_size..(i + 1) * row_size];
        let (stream, name_off, sem_idx, sv_type, comp_type, register, mask, rw_mask, min_prec) =
            if extended {
                (
                    rd_u32(row, 0).unwrap(),
                    rd_u32(row, 4).unwrap() as usize,
                    rd_u32(row, 8).unwrap(),
                    rd_u32(row, 12).unwrap(),
                    rd_u32(row, 16).unwrap(),
                    rd_u32(row, 20).unwrap(),
                    row[24],
                    row[25],
                    rd_u32(row, 28).unwrap(),
                )
            } else {
                (
                    0,
                    rd_u32(row, 0).unwrap() as usize,
                    rd_u32(row, 4).unwrap(),
                    rd_u32(row, 8).unwrap(),
                    rd_u32(row, 12).unwrap(),
                    rd_u32(row, 16).unwrap(),
                    row[20],
                    row[21],
                    0,
                )
            };
        let str_tail = body.get(name_off..).ok_or(ContainerError::OutOfBounds(
            "signature semantic name offset",
        ))?;
        let nul_pos = str_tail
            .iter()
            .position(|&b| b == 0)
            .ok_or(ContainerError::Truncated(
                "signature semantic NUL terminator",
            ))?;
        let semantic_name = String::from_utf8_lossy(&str_tail[..nul_pos]).into_owned();
        out.push(SignatureElement {
            stream,
            semantic_name,
            semantic_index: sem_idx,
            system_value_type: sv_type,
            component_type: comp_type,
            register,
            mask,
            rw_mask,
            min_precision: min_prec,
        });
    }
    Ok(out)
}

fn extract_dxbc_signatures(parts: &[DxbcPart<'_>]) -> Result<StageSignature, ContainerError> {
    let mut sig = StageSignature::default();
    if let Some((_, body)) = parts.iter().find(|(t, _)| t == b"ISG1") {
        sig.inputs = parse_dxbc_signature(body, true)?;
    } else if let Some((_, body)) = parts.iter().find(|(t, _)| t == b"ISGN") {
        sig.inputs = parse_dxbc_signature(body, false)?;
    }
    if let Some((_, body)) = parts.iter().find(|(t, _)| t == b"OSG1") {
        sig.outputs = parse_dxbc_signature(body, true)?;
    } else if let Some((_, body)) = parts.iter().find(|(t, _)| t == b"OSGN") {
        sig.outputs = parse_dxbc_signature(body, false)?;
    }
    Ok(sig)
}

/// Build a synthetic `ISG1`/`OSG1` (or `ISGN`/`OSGN`) chunk body for unit tests.
pub fn build_signature_chunk_body(elements: &[SignatureElement], extended: bool) -> Vec<u8> {
    let row_size = if extended { 32usize } else { 24usize };
    let table_off = 8usize;
    let strings_base = table_off + elements.len() * row_size;
    let mut strings = Vec::new();
    let mut name_offsets = Vec::with_capacity(elements.len());
    for e in elements {
        name_offsets.push((strings_base + strings.len()) as u32);
        strings.extend_from_slice(e.semantic_name.as_bytes());
        strings.push(0);
    }

    let mut out = Vec::with_capacity(strings_base + strings.len());
    out.extend_from_slice(&(elements.len() as u32).to_le_bytes());
    out.extend_from_slice(&(table_off as u32).to_le_bytes());
    for (e, name_off) in elements.iter().zip(name_offsets) {
        if extended {
            out.extend_from_slice(&e.stream.to_le_bytes());
        }
        out.extend_from_slice(&name_off.to_le_bytes());
        out.extend_from_slice(&e.semantic_index.to_le_bytes());
        out.extend_from_slice(&e.system_value_type.to_le_bytes());
        out.extend_from_slice(&e.component_type.to_le_bytes());
        out.extend_from_slice(&e.register.to_le_bytes());
        out.push(e.mask);
        out.push(e.rw_mask);
        out.extend_from_slice(&[0u8, 0u8]);
        if extended {
            out.extend_from_slice(&e.min_precision.to_le_bytes());
        }
    }
    out.extend_from_slice(&strings);
    out
}

/// A parsed DXBC chunk `(fourcc, body)`.
pub type DxbcPart<'a> = ([u8; 4], &'a [u8]);

/// Extract `(fourcc, body)` slices from a DXBC container with strict bounds validation.
pub fn dxbc_parts(bytes: &[u8]) -> Result<Vec<DxbcPart<'_>>, ContainerError> {
    if bytes.get(0..4) != Some(MAGIC_DXBC.as_slice()) {
        return Err(ContainerError::InvalidHeader("expected DXBC magic".into()));
    }
    if bytes.len() < 0x20 {
        return Err(ContainerError::Truncated("DXBC header"));
    }
    let declared_size = rd_u32(bytes, 0x18).ok_or(ContainerError::Truncated("DXBC size"))? as usize;
    if declared_size < 0x20 || declared_size > bytes.len() {
        return Err(ContainerError::OutOfBounds("DXBC declared total size"));
    }
    let part_count =
        rd_u32(bytes, 0x1C).ok_or(ContainerError::Truncated("DXBC part count"))? as usize;
    if part_count > MAX_DXBC_PARTS {
        return Err(ContainerError::LimitExceeded(format!(
            "DXBC part count {part_count} exceeds limit {MAX_DXBC_PARTS}"
        )));
    }
    let table_bytes = part_count
        .checked_mul(4)
        .ok_or(ContainerError::Overflow("DXBC offset table"))?;
    let table_end = 0x20usize
        .checked_add(table_bytes)
        .ok_or(ContainerError::Overflow("DXBC offset table end"))?;
    if table_end > declared_size {
        return Err(ContainerError::Truncated("DXBC part offset table"));
    }

    let container = &bytes[..declared_size];
    let mut parts = Vec::with_capacity(part_count);
    for i in 0..part_count {
        let off_pos = 0x20 + i * 4;
        let part_off = rd_u32(container, off_pos)
            .ok_or(ContainerError::Truncated("DXBC part offset"))? as usize;
        let hdr_end = part_off
            .checked_add(8)
            .ok_or(ContainerError::Overflow("DXBC part header"))?;
        let hdr = container
            .get(part_off..hdr_end)
            .ok_or(ContainerError::OutOfBounds("DXBC part header"))?;
        let tag = [hdr[0], hdr[1], hdr[2], hdr[3]];
        let body_len = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let body_end = hdr_end
            .checked_add(body_len)
            .ok_or(ContainerError::Overflow("DXBC part body"))?;
        let body = container
            .get(hdr_end..body_end)
            .ok_or(ContainerError::OutOfBounds("DXBC part body"))?;
        parts.push((tag, body));
    }
    Ok(parts)
}

fn parse_dxil_program_header(body: &[u8]) -> Result<(ShaderStage, (u8, u8), bool), ContainerError> {
    if body.len() < 24 {
        return Err(ContainerError::Truncated("DXIL program header"));
    }
    if body.get(8..12) != Some(MAGIC_DXIL.as_slice()) {
        return Err(ContainerError::InvalidHeader(
            "missing DXIL tag at +0x08".into(),
        ));
    }
    let packed = rd_u32(body, 0).ok_or(ContainerError::Truncated("DXIL program version"))?;
    let kind_raw = (packed >> 16) as u16;
    let sm_major = ((packed >> 4) & 0xF) as u8;
    let sm_minor = (packed & 0xF) as u8;
    let bc_off =
        rd_u32(body, 0x10).ok_or(ContainerError::Truncated("DXIL bitcode offset"))? as usize;
    let bc_len = rd_u32(body, 0x14).ok_or(ContainerError::Truncated("DXIL bitcode size"))? as usize;

    let bc_start = 8usize
        .checked_add(bc_off)
        .ok_or(ContainerError::Overflow("DXIL bitcode start"))?;
    let has_nobc = bc_len >= 4
        && bc_start
            .checked_add(4)
            .and_then(|end| body.get(bc_start..end))
            == Some(MAGIC_NOBC.as_slice());

    Ok((
        ShaderStage::from_dxil_kind(kind_raw),
        (sm_major, sm_minor),
        has_nobc,
    ))
}

fn parse_dxil_metadata(
    bytes: &[u8],
    content_sha256: String,
) -> Result<ParsedContainer, ContainerError> {
    let (dxil_body, signature) = if bytes.get(0..4) == Some(MAGIC_DXBC.as_slice()) {
        let parts = dxbc_parts(bytes)?;
        let sig = extract_dxbc_signatures(&parts)?;
        let body = parts
            .into_iter()
            .find(|(tag, _)| tag == MAGIC_DXIL)
            .map(|(_, body)| body);
        (body, sig)
    } else {
        (Some(bytes), StageSignature::default())
    };
    let (api_stage, shader_model, has_nobc) = match dxil_body {
        Some(body) if body.len() >= 24 && body.get(8..12) == Some(MAGIC_DXIL.as_slice()) => {
            let (st, sm, nobc) = parse_dxil_program_header(body)?;
            (Some(st), Some(sm), nobc)
        }
        _ => (None, None, false),
    };
    Ok(ParsedContainer {
        kind: ContainerKind::Dxil,
        api_stage,
        shader_model,
        has_nobc_placeholder: has_nobc,
        content_sha256,
        variants: Vec::new(),
        signature,
        raw_code: None,
    })
}

fn parse_console_dxbc(
    bytes: &[u8],
    content_sha256: String,
) -> Result<ParsedContainer, ContainerError> {
    let parts = dxbc_parts(bytes)?;
    let signature = extract_dxbc_signatures(&parts)?;
    let dxil_body = parts
        .iter()
        .find(|(tag, _)| tag == MAGIC_DXIL)
        .map(|(_, body)| *body)
        .ok_or_else(|| ContainerError::InvalidHeader("DXBC missing DXIL part".into()))?;

    let (api_stage, shader_model, has_nobc) = parse_dxil_program_header(dxil_body)?;

    // XDAT starts at DXIL body + 0x18.
    let xdat = dxil_body
        .get(0x18..)
        .ok_or(ContainerError::Truncated("XDAT header"))?;
    if xdat.get(0..4) != Some(MAGIC_XDAT.as_slice()) {
        return Err(ContainerError::InvalidHeader(
            "expected XDAT magic at DXIL+0x18".into(),
        ));
    }
    let entry_count =
        rd_u32(xdat, 0x04).ok_or(ContainerError::Truncated("XDAT entry count"))? as usize;
    if entry_count > MAX_XDAT_ENTRIES {
        return Err(ContainerError::LimitExceeded(format!(
            "XDAT entry count {entry_count} exceeds limit {MAX_XDAT_ENTRIES}"
        )));
    }
    let table_off =
        rd_u32(xdat, 0x08).ok_or(ContainerError::Truncated("XDAT table offset"))? as usize;
    let table_bytes = entry_count
        .checked_mul(12)
        .ok_or(ContainerError::Overflow("XDAT table size"))?;
    let table_end = table_off
        .checked_add(table_bytes)
        .ok_or(ContainerError::Overflow("XDAT table end"))?;
    let table = dxil_body
        .get(table_off..table_end)
        .ok_or(ContainerError::OutOfBounds("XDAT entry table"))?;

    let mut variants = Vec::new();
    let mut total_decompressed = 0usize;

    for i in 0..entry_count {
        let e = &table[i * 12..(i + 1) * 12];
        let kind = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
        let offset = u32::from_le_bytes([e[4], e[5], e[6], e[7]]) as usize;
        let size = u32::from_le_bytes([e[8], e[9], e[10], e[11]]) as usize;

        let end = offset
            .checked_add(size)
            .ok_or(ContainerError::Overflow("XDAT entry slice"))?;
        let slice = dxil_body
            .get(offset..end)
            .ok_or(ContainerError::OutOfBounds("XDAT entry slice"))?;

        if kind == XDAT_KIND_XBCH {
            let mut extracted = parse_xbch(slice, offset, &mut total_decompressed)?;
            variants.append(&mut extracted);
        }
    }

    Ok(ParsedContainer {
        kind: ContainerKind::ConsoleDxbc,
        api_stage: Some(api_stage),
        shader_model: Some(shader_model),
        has_nobc_placeholder: has_nobc,
        content_sha256,
        variants,
        signature,
        raw_code: None,
    })
}

fn parse_xbch(
    slice: &[u8],
    xbch_offset: usize,
    total_decompressed: &mut usize,
) -> Result<Vec<FssmVariant>, ContainerError> {
    if slice.len() < 0x18 {
        return Err(ContainerError::Truncated("XBCH header"));
    }
    if slice.get(0..4) != Some(MAGIC_XBCH.as_slice()) {
        return Err(ContainerError::InvalidHeader("expected XBCH magic".into()));
    }
    let version = rd_u32(slice, 0x04).ok_or(ContainerError::Truncated("XBCH version"))?;
    if version != XBCH_VERSION_3 {
        return Err(ContainerError::InvalidHeader(format!(
            "unsupported XBCH version {version} (expected {XBCH_VERSION_3})"
        )));
    }
    let packed_u64 = rd_u64(slice, 0x08).ok_or(ContainerError::Truncated("XBCH packed size"))?;
    let unpacked_u64 =
        rd_u64(slice, 0x10).ok_or(ContainerError::Truncated("XBCH unpacked size"))?;

    let packed_len =
        usize::try_from(packed_u64).map_err(|_| ContainerError::Overflow("XBCH packed size"))?;
    let unpacked_len = usize::try_from(unpacked_u64)
        .map_err(|_| ContainerError::Overflow("XBCH unpacked size"))?;

    if unpacked_len == 0 || unpacked_len > MAX_FSSM_BYTES {
        return Err(ContainerError::LimitExceeded(format!(
            "XBCH unpacked size {unpacked_len} outside 1..={MAX_FSSM_BYTES}"
        )));
    }
    let next_total = total_decompressed
        .checked_add(unpacked_len)
        .ok_or(ContainerError::Overflow("aggregate FSSM size"))?;
    if next_total > MAX_TOTAL_FSSM_BYTES {
        return Err(ContainerError::LimitExceeded(format!(
            "aggregate unpacked FSSM size {next_total} exceeds {MAX_TOTAL_FSSM_BYTES}"
        )));
    }

    let z_end = 0x18usize
        .checked_add(packed_len)
        .ok_or(ContainerError::Overflow("XBCH zlib stream end"))?;
    let compressed = slice
        .get(0x18..z_end)
        .ok_or(ContainerError::OutOfBounds("XBCH zlib stream"))?;

    let fssm = decompress_zlib_bounded(compressed, unpacked_len)
        .map_err(|e| ContainerError::Decompress(e.to_string()))?;
    if fssm.len() != unpacked_len {
        return Err(ContainerError::SizeMismatch {
            expected: unpacked_len,
            actual: fssm.len(),
        });
    }
    *total_decompressed = next_total;

    parse_fssm(&fssm, xbch_offset, version, packed_len)
}

fn parse_fssm(
    fssm: &[u8],
    xbch_offset: usize,
    xbch_version: u32,
    compressed_size: usize,
) -> Result<Vec<FssmVariant>, ContainerError> {
    if fssm.len() < 0x10 {
        return Err(ContainerError::Truncated("FSSM header"));
    }
    if fssm.get(0..4) != Some(MAGIC_FSSM.as_slice()) {
        return Err(ContainerError::InvalidHeader("expected FSSM magic".into()));
    }
    let version = rd_u32(fssm, 0x04).ok_or(ContainerError::Truncated("FSSM version"))?;
    if version != FSSM_VERSION_101 {
        return Err(ContainerError::InvalidHeader(format!(
            "unsupported FSSM version 0x{version:x} (expected 0x{FSSM_VERSION_101:x})"
        )));
    }
    let payload_size_u64 =
        rd_u64(fssm, 0x08).ok_or(ContainerError::Truncated("FSSM payload size"))?;
    let payload_size = usize::try_from(payload_size_u64)
        .map_err(|_| ContainerError::Overflow("FSSM payload size"))?;
    let payload = &fssm[0x10..];
    if payload_size != payload.len() {
        return Err(ContainerError::SizeMismatch {
            expected: payload_size,
            actual: payload.len(),
        });
    }

    let mut variants = Vec::new();

    // Primary path: FSSM stage pointer table (payload +0x00 == 6, pointers at +0x08..+0x60).
    if payload.len() >= 0x68 && rd_u64(payload, 0x00) == Some(6) {
        for &(slot_off, hw_stage) in FSSM_STAGE_SLOTS {
            let Some(rec_off_u64) = rd_u64(payload, slot_off) else {
                continue;
            };
            if rec_off_u64 == 0 || rec_off_u64 == ABSENT_U64 {
                continue;
            }
            let rec_off = usize::try_from(rec_off_u64)
                .map_err(|_| ContainerError::Overflow("FSSM stage record offset"))?;
            let rec_end = rec_off
                .checked_add(0x54)
                .ok_or(ContainerError::Overflow("FSSM stage record end"))?;
            let rec = payload
                .get(rec_off..rec_end)
                .ok_or(ContainerError::OutOfBounds("FSSM stage record"))?;

            let code_off = usize::try_from(
                rd_u64(rec, 0x00).ok_or(ContainerError::Truncated("stage code offset"))?,
            )
            .map_err(|_| ContainerError::Overflow("stage code offset"))?;
            let code_len = usize::try_from(
                rd_u64(rec, 0x08).ok_or(ContainerError::Truncated("stage code len"))?,
            )
            .map_err(|_| ContainerError::Overflow("stage code len"))?;
            let flags = rd_u32(rec, 0x14).ok_or(ContainerError::Truncated("stage flags"))?;
            let wave32 = (flags & (1 << 5)) != 0;

            if code_len == 0 || !code_len.is_multiple_of(4) {
                return Err(ContainerError::InvalidCodeSlice(code_len));
            }
            let code_end = code_off
                .checked_add(code_len)
                .ok_or(ContainerError::Overflow("stage code end"))?;
            let code_slice = payload
                .get(code_off..code_end)
                .ok_or(ContainerError::OutOfBounds("stage code slice"))?;

            let segments = segment_code_region(code_slice);
            variants.push(FssmVariant {
                xbch_offset,
                xbch_version,
                compressed_size,
                fssm_version: version,
                hw_stage,
                wave32,
                code: code_slice.to_vec(),
                code_len,
                segments,
            });
        }
    }

    // Fallback path: single fixed record at FSSM +0xC0/+0xC8 (payload +0xB0/+0xB8).
    if variants.is_empty() && fssm.len() >= 0xE0 {
        let code_off_u64 = rd_u64(fssm, 0xC0).unwrap_or(ABSENT_U64);
        let code_len_u64 = rd_u64(fssm, 0xC8).unwrap_or(0);
        if code_off_u64 != ABSENT_U64 && code_len_u64 > 0 {
            let code_off = usize::try_from(code_off_u64)
                .map_err(|_| ContainerError::Overflow("fallback code offset"))?;
            let code_len = usize::try_from(code_len_u64)
                .map_err(|_| ContainerError::Overflow("fallback code len"))?;
            if !code_len.is_multiple_of(4) {
                return Err(ContainerError::InvalidCodeSlice(code_len));
            }
            let code_end = code_off
                .checked_add(code_len)
                .ok_or(ContainerError::Overflow("fallback code end"))?;
            let code_slice = payload
                .get(code_off..code_end)
                .ok_or(ContainerError::OutOfBounds("fallback code slice"))?;
            let flags = rd_u32(fssm, 0xD4).unwrap_or(0);
            let wave32 = flags != 0xDEAD_BEEF && (flags & (1 << 5)) != 0;
            let hw_stage = parse_optional_hw_descriptor(fssm).unwrap_or(HwStage::Unknown(0xFF));
            let segments = segment_code_region(code_slice);
            variants.push(FssmVariant {
                xbch_offset,
                xbch_version,
                compressed_size,
                fssm_version: version,
                hw_stage,
                wave32,
                code: code_slice.to_vec(),
                code_len,
                segments,
            });
        }
    }

    if variants.is_empty() {
        return Err(ContainerError::InvalidHeader(
            "FSSM contained no valid hardware-stage code records".into(),
        ));
    }

    Ok(variants)
}

fn parse_optional_hw_descriptor(fssm: &[u8]) -> Option<HwStage> {
    let desc = rd_u64(fssm, 0x100)?;
    if (desc >> 16) == 0x0000_00C0_0002 {
        Some(HwStage::from_hw_id((desc & 0xFF) as u8))
    } else {
        None
    }
}

/// Segment an extracted RDNA2 code slice into program code, trailing `s_code_end`/zero padding,
/// and counted embedded constant pools (`[u32 N, N DWORDs, S_ENDPGM]`).
pub fn segment_code_region(code: &[u8]) -> Vec<CodeSegment> {
    let (chunks, _) = code.as_chunks::<4>();
    let words: Vec<u32> = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();
    if words.is_empty() {
        return Vec::new();
    }

    // Find the first s_endpgm to locate potential post-exit pools/padding.
    let mut segments = Vec::new();
    let Some(first_end) = words.iter().position(|&w| w == S_ENDPGM) else {
        segments.push(CodeSegment {
            offset: 0,
            len: words.len() * 4,
            kind: SegmentKind::Program,
        });
        return segments;
    };

    let prog_end_word = first_end + 1;
    segments.push(CodeSegment {
        offset: 0,
        len: prog_end_word * 4,
        kind: SegmentKind::Program,
    });

    let mut pos = prog_end_word;
    while pos < words.len() {
        // Skip padding runs of s_code_end or zero DWORDs.
        if words[pos] == S_CODE_END || words[pos] == 0 {
            let start = pos;
            while pos < words.len() && (words[pos] == S_CODE_END || words[pos] == 0) {
                pos += 1;
            }
            segments.push(CodeSegment {
                offset: start * 4,
                len: (pos - start) * 4,
                kind: SegmentKind::Padding,
            });
            continue;
        }

        // Check for counted embedded constant pool: [count N, N DWORDs, S_ENDPGM].
        let count = words[pos] as usize;
        if count > 0
            && let Some(end_marker_idx) = pos.checked_add(1).and_then(|p| p.checked_add(count))
            && end_marker_idx < words.len()
            && words[end_marker_idx] == S_ENDPGM
        {
            let span_words = count + 2;
            segments.push(CodeSegment {
                offset: pos * 4,
                len: span_words * 4,
                kind: SegmentKind::EmbeddedDataPool,
            });
            pos += span_words;
            continue;
        }

        // Otherwise, treat the remaining chunk up to the next s_endpgm (if any) as another program.
        let start = pos;
        if let Some(rel_end) = words[pos..].iter().position(|&w| w == S_ENDPGM) {
            pos += rel_end + 1;
        } else {
            pos = words.len();
        }
        segments.push(CodeSegment {
            offset: start * 4,
            len: (pos - start) * 4,
            kind: SegmentKind::Program,
        });
    }

    segments
}

/// Pure-Rust FIPS 180-4 SHA-256 hex digest for content-addressed shader identification.
fn sha256_hex(input: &[u8]) -> String {
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
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
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
    h.iter().map(|w| format!("{w:08x}")).collect()
}

/// Build a synthetic `DXBC -> DXIL(NOBC) -> XDAT -> XBCH v3 (zlib) -> FSSM v0x101` container
/// for clean-room unit and CLI integration testing.
pub fn build_synthetic_console_dxbc(
    api_stage_id: u16,
    hw_stage_slot_offset: usize,
    wave32: bool,
    code_words: &[u32],
) -> Vec<u8> {
    // 1. Build FSSM payload:
    //    +0x00: u64 = 6
    //    +0x08..+0x68: stage pointer slots (absent = ABSENT_U64)
    //    +0x70..+0xC4: stage record (0x54 bytes)
    //    +0xD0..: code bytes
    let code_bytes: Vec<u8> = code_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let rec_off = 0x70usize;
    let code_off = 0xD0usize;
    let mut payload = vec![0u8; code_off + code_bytes.len()];
    payload[0..8].copy_from_slice(&6u64.to_le_bytes());
    for &(slot, _) in FSSM_STAGE_SLOTS {
        payload[slot..slot + 8].copy_from_slice(&ABSENT_U64.to_le_bytes());
    }
    payload[hw_stage_slot_offset..hw_stage_slot_offset + 8]
        .copy_from_slice(&(rec_off as u64).to_le_bytes());
    payload[rec_off..rec_off + 8].copy_from_slice(&(code_off as u64).to_le_bytes());
    payload[rec_off + 8..rec_off + 16].copy_from_slice(&(code_bytes.len() as u64).to_le_bytes());
    let flags: u32 = if wave32 { 1 << 5 } else { 0 };
    payload[rec_off + 0x14..rec_off + 0x18].copy_from_slice(&flags.to_le_bytes());
    payload[code_off..code_off + code_bytes.len()].copy_from_slice(&code_bytes);

    // 2. Wrap in FSSM v0x101 header
    let mut fssm = Vec::with_capacity(0x10 + payload.len());
    fssm.extend_from_slice(MAGIC_FSSM);
    fssm.extend_from_slice(&FSSM_VERSION_101.to_le_bytes());
    fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    fssm.extend_from_slice(&payload);

    // 3. Compress FSSM with minimal stored-block zlib (RFC 1950, zero extra deps)
    let z = zlib_store_encode(&fssm);

    // 4. Wrap in XBCH v3
    let mut xbch = Vec::with_capacity(0x18 + z.len());
    xbch.extend_from_slice(MAGIC_XBCH);
    xbch.extend_from_slice(&XBCH_VERSION_3.to_le_bytes());
    xbch.extend_from_slice(&(z.len() as u64).to_le_bytes());
    xbch.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
    xbch.extend_from_slice(&z);

    // 5. Build DXIL part body: 24 B header + 12 B XDAT header + 12 B XDAT table + 4 B NOBC + XBCH
    let table_off = 0x24u32;
    let nobc_off = 0x30u32;
    let xbch_off = 0x34u32;
    let mut dxil_body = Vec::new();
    let prog_ver = (u32::from(api_stage_id) << 16) | (6 << 4); // SM 6.0
    dxil_body.extend_from_slice(&prog_ver.to_le_bytes());
    dxil_body.extend_from_slice(&0u32.to_le_bytes()); // size in dwords (patched below)
    dxil_body.extend_from_slice(MAGIC_DXIL);
    dxil_body.extend_from_slice(&0x0100u32.to_le_bytes());
    dxil_body.extend_from_slice(&(nobc_off - 8).to_le_bytes()); // bitcode offset relative to +8
    dxil_body.extend_from_slice(&4u32.to_le_bytes()); // bitcode size (4 B "NOBC")
    // XDAT at +0x18
    dxil_body.extend_from_slice(MAGIC_XDAT);
    dxil_body.extend_from_slice(&1u32.to_le_bytes()); // 1 entry
    dxil_body.extend_from_slice(&table_off.to_le_bytes());
    // XDAT entry 0 at +0x24
    dxil_body.extend_from_slice(&XDAT_KIND_XBCH.to_le_bytes());
    dxil_body.extend_from_slice(&xbch_off.to_le_bytes());
    dxil_body.extend_from_slice(&(xbch.len() as u32).to_le_bytes());
    // NOBC at +0x30
    dxil_body.extend_from_slice(MAGIC_NOBC);
    // XBCH at +0x34
    dxil_body.extend_from_slice(&xbch);
    while dxil_body.len() % 4 != 0 {
        dxil_body.push(0);
    }
    let dwords = (dxil_body.len() / 4) as u32;
    dxil_body[4..8].copy_from_slice(&dwords.to_le_bytes());

    // 6. Wrap in DXBC container
    let part_off = 0x24u32;
    let total_size = (0x24 + 8 + dxil_body.len()) as u32;
    let mut dxbc = Vec::with_capacity(total_size as usize);
    dxbc.extend_from_slice(MAGIC_DXBC);
    dxbc.extend_from_slice(&[0u8; 16]); // digest
    dxbc.extend_from_slice(&1u16.to_le_bytes()); // major
    dxbc.extend_from_slice(&0u16.to_le_bytes()); // minor
    dxbc.extend_from_slice(&total_size.to_le_bytes());
    dxbc.extend_from_slice(&1u32.to_le_bytes()); // 1 part
    dxbc.extend_from_slice(&part_off.to_le_bytes());
    dxbc.extend_from_slice(MAGIC_DXIL);
    dxbc.extend_from_slice(&(dxil_body.len() as u32).to_le_bytes());
    dxbc.extend_from_slice(&dxil_body);
    dxbc
}

/// Minimal RFC 1950 + RFC 1951 stored-block zlib encoder for clean-room synthetic fixtures.
fn zlib_store_encode(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 32);
    out.push(0x78); // CMF: deflate, 32K window
    out.push(0x01); // FLG: FCHECK passes ((0x78*256 + 0x01) % 31 == 0)

    let mut pos = 0usize;
    loop {
        let rem = data.len() - pos;
        let block_len = rem.min(65_535);
        let is_final = pos + block_len == data.len();
        out.push(if is_final { 0x01 } else { 0x00 }); // BFINAL | BTYPE=00 (stored)
        let len_u16 = block_len as u16;
        let nlen_u16 = !len_u16;
        out.extend_from_slice(&len_u16.to_le_bytes());
        out.extend_from_slice(&nlen_u16.to_le_bytes());
        out.extend_from_slice(&data[pos..pos + block_len]);
        pos += block_len;
        if is_final {
            break;
        }
    }

    // Adler-32 trailer (big-endian)
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in data {
        s1 = (s1 + u32::from(b)) % 65_521;
        s2 = (s2 + s1) % 65_521;
    }
    let adler = (s2 << 16) | s1;
    out.extend_from_slice(&adler.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpacks_synthetic_console_dxbc_with_xbch_fssm_and_constant_pool() {
        // Program: s_nop 0; s_endpgm; embedded constant pool of 2 DWORDs + closing s_endpgm; s_code_end padding
        let words = [
            0xBF80_0000, // s_nop 0
            S_ENDPGM,    // s_endpgm
            2,           // pool count = 2
            0x3F80_0000, // 1.0f
            0x4000_0000, // 2.0f
            S_ENDPGM,    // pool closing marker
            S_CODE_END,  // padding
        ];
        let blob = build_synthetic_console_dxbc(5, 0x60, true, &words);
        assert_eq!(classify(&blob), ContainerKind::ConsoleDxbc);

        let parsed = parse_container(&blob).expect("parse synthetic console DXBC");
        assert_eq!(parsed.api_stage, Some(ShaderStage::Compute));
        assert_eq!(parsed.shader_model, Some((6, 0)));
        assert!(parsed.has_nobc_placeholder);
        assert_eq!(parsed.content_sha256.len(), 64);
        assert_eq!(parsed.variants.len(), 1);

        let v = &parsed.variants[0];
        assert_eq!(v.xbch_version, 3);
        assert_eq!(v.fssm_version, 0x101);
        assert_eq!(v.hw_stage, HwStage::Cs);
        assert!(v.wave32);
        assert_eq!(v.code_words(), words);
        assert_eq!(v.segments.len(), 3);
        assert_eq!(v.segments[0].kind, SegmentKind::Program);
        assert_eq!(v.segments[1].kind, SegmentKind::EmbeddedDataPool);
        assert_eq!(v.segments[2].kind, SegmentKind::Padding);
    }

    #[test]
    fn rejects_malformed_truncated_and_corrupt_containers_without_panicking() {
        assert!(parse_container(&[]).is_err());
        assert!(parse_container(b"BAD!").is_err());
        assert!(parse_container(b"DXBC\x00\x00").is_err());
        assert!(parse_container(b"XBCH\x03\x00\x00\x00").is_err());
        assert!(parse_container(b"FSSM\x01\x01\x00\x00").is_err());

        // Truncate valid container at every prefix boundary.
        let valid = build_synthetic_console_dxbc(5, 0x60, false, &[S_ENDPGM]);
        for cut in [4, 16, 32, 48, valid.len() / 2, valid.len() - 1] {
            assert!(
                parse_container(&valid[..cut]).is_err(),
                "truncated at {cut} must fail cleanly"
            );
        }
    }

    #[test]
    fn classifies_elf_and_raw_rdna2() {
        assert_eq!(classify(b"\x7fELF\x02\x01\x01\x00"), ContainerKind::Elf);
        let raw = S_ENDPGM.to_le_bytes();
        assert_eq!(classify(&raw), ContainerKind::RawRdna2);
        let parsed = parse_container(&raw).expect("parse raw RDNA2");
        assert_eq!(parsed.primary_code_words(), Some(vec![S_ENDPGM]));
    }
}
