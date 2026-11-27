// Adapted from xcl-dev/xcl (https://github.com/xcl-dev/xcl),
// commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/pe.rs
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Modified for X2Linux on 2026-11-27: read-only file-layout PE32+ parser hardened with
// checked arithmetic across all header, section, directory, import, delay-import, and TLS walks.

//! Bounds-checked, read-only PE32+ (AMD64) binary parser for `x2l-resolver`.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// `IMAGE_FILE_MACHINE_AMD64`.
pub const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
/// `IMAGE_FILE_RELOCS_STRIPPED`.
pub const IMAGE_FILE_RELOCS_STRIPPED: u16 = 0x0001;
/// `IMAGE_FILE_DLL`.
pub const IMAGE_FILE_DLL: u16 = 0x2000;
/// `IMAGE_NT_OPTIONAL_HDR64_MAGIC`.
pub const PE32_PLUS_MAGIC: u16 = 0x020B;

pub const DIR_EXPORT: usize = 0;
pub const DIR_IMPORT: usize = 1;
pub const DIR_EXCEPTION: usize = 3;
pub const DIR_BASERELOC: usize = 5;
pub const DIR_TLS: usize = 9;
pub const DIR_DELAY_IMPORT: usize = 13;

const MAX_SECTIONS: usize = 1024;
const MAX_IMPORT_DLLS: u32 = 4096;
const MAX_IMPORT_THUNKS: u32 = 65_536;
const MAX_CSTR_LEN: usize = 4096;

/// Structured errors returned when inspecting invalid, non-PE, or malformed binaries.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PeError {
    #[error("file is truncated or malformed while reading {0}")]
    Truncated(&'static str),
    #[error(
        "invalid PE signature ({0}) — not an unpacked PE32+ binary (encrypted XVC/MSIXVC packages are unsupported)"
    )]
    BadMagic(&'static str),
    #[error("unsupported PE machine 0x{0:04x} (expected AMD64 0x8664)")]
    NotAmd64(u16),
    #[error("arithmetic overflow while computing offset in {0}")]
    Overflow(&'static str),
    #[error("RVA 0x{0:08x} is outside mapped file sections")]
    BadRva(u32),
    #[error("unterminated string at RVA 0x{0:08x}")]
    UnterminatedString(u32),
    #[error("section count {0} exceeds parser safety limit ({MAX_SECTIONS})")]
    TooManySections(usize),
}

/// PE data directory entry (`IMAGE_DATA_DIRECTORY`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataDir {
    pub rva: u32,
    pub size: u32,
}

impl DataDir {
    pub fn is_present(self) -> bool {
        self.rva != 0 && self.size != 0
    }
}

/// PE section header summary (`IMAGE_SECTION_HEADER`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    pub name: String,
    pub virtual_address: u32,
    pub virtual_size: u32,
    pub raw_offset: u32,
    pub raw_size: u32,
    pub characteristics: u32,
}

impl Section {
    pub fn mapped_size(&self) -> u32 {
        if self.virtual_size == 0 {
            self.raw_size
        } else {
            self.virtual_size
        }
    }
}

/// Imported symbol identifier (either by name or by 16-bit ordinal).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImportSymbol {
    Name { hint: u16, name: String },
    Ordinal { ordinal: u16 },
}

impl fmt::Display for ImportSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name { name, .. } => f.write_str(name),
            Self::Ordinal { ordinal } => write!(f, "#{ordinal}"),
        }
    }
}

/// Single entry inside a DLL's import table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportEntry {
    pub iat_rva: u32,
    pub symbol: ImportSymbol,
}

/// One imported DLL (regular or delay-load) and its symbols.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportDll {
    pub dll: String,
    pub delay_load: bool,
    pub entries: Vec<ImportEntry>,
}

/// Parsed `IMAGE_TLS_DIRECTORY64` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TlsInfo {
    pub directory_rva: u32,
    pub start_raw_va: u64,
    pub end_raw_va: u64,
    pub address_of_index: u64,
    pub address_of_callbacks: u64,
    pub size_of_zero_fill: u32,
    pub characteristics: u32,
}

impl TlsInfo {
    /// Initialized static TLS template length in bytes.
    pub fn template_bytes(&self) -> u64 {
        self.end_raw_va.saturating_sub(self.start_raw_va)
    }

    /// Total static TLS allocation size (initialized template + zero-fill tail).
    pub fn total_bytes(&self) -> u64 {
        self.template_bytes()
            .saturating_add(u64::from(self.size_of_zero_fill))
    }
}

/// Read-only parsed PE32+ (AMD64) binary view.
#[derive(Debug)]
pub struct PeImage<'a> {
    data: &'a [u8],
    pub machine: u16,
    pub characteristics: u16,
    pub timestamp: u32,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub image_base: u64,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub entry_rva: u32,
    pub stack_reserve: u64,
    pub stack_commit: u64,
    pub sections: Vec<Section>,
    pub dirs: [DataDir; 16],
}

fn rd_u16(data: &[u8], off: usize) -> Option<u16> {
    let end = off.checked_add(2)?;
    let b = data.get(off..end)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn rd_u32(data: &[u8], off: usize) -> Option<u32> {
    let end = off.checked_add(4)?;
    let b = data.get(off..end)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn rd_u64(data: &[u8], off: usize) -> Option<u64> {
    let end = off.checked_add(8)?;
    let b = data.get(off..end)?;
    Some(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn add_usize(base: usize, delta: usize, ctx: &'static str) -> Result<usize, PeError> {
    base.checked_add(delta).ok_or(PeError::Overflow(ctx))
}

impl<'a> PeImage<'a> {
    /// Parse and validate a PE32+ AMD64 binary from raw file bytes.
    pub fn parse(data: &'a [u8]) -> Result<Self, PeError> {
        let dos_magic = rd_u16(data, 0).ok_or(PeError::Truncated("DOS header"))?;
        if dos_magic != 0x5A4D {
            return Err(PeError::BadMagic("MZ"));
        }

        let nt_off = rd_u32(data, 0x3C).ok_or(PeError::Truncated("e_lfanew"))? as usize;
        let pe_sig = rd_u32(data, nt_off).ok_or(PeError::Truncated("NT signature"))?;
        if pe_sig != 0x0000_4550 {
            return Err(PeError::BadMagic("PE\\0\\0"));
        }

        let coff = add_usize(nt_off, 4, "COFF header")?;
        let machine = rd_u16(data, coff).ok_or(PeError::Truncated("COFF header"))?;
        let nsec = rd_u16(data, add_usize(coff, 2, "COFF header")?)
            .ok_or(PeError::Truncated("COFF header"))? as usize;
        if nsec > MAX_SECTIONS {
            return Err(PeError::TooManySections(nsec));
        }
        let timestamp = rd_u32(data, add_usize(coff, 4, "COFF header")?)
            .ok_or(PeError::Truncated("COFF header"))?;
        let opt_size = rd_u16(data, add_usize(coff, 16, "COFF header")?)
            .ok_or(PeError::Truncated("COFF header"))? as usize;
        let characteristics = rd_u16(data, add_usize(coff, 18, "COFF header")?)
            .ok_or(PeError::Truncated("COFF header"))?;

        let opt = add_usize(coff, 20, "optional header")?;
        if opt_size < 112 {
            return Err(PeError::Truncated("PE32+ optional header size"));
        }
        let opt_magic = rd_u16(data, opt).ok_or(PeError::Truncated("optional header"))?;
        if machine != IMAGE_FILE_MACHINE_AMD64 {
            return Err(PeError::NotAmd64(machine));
        }
        if opt_magic != PE32_PLUS_MAGIC {
            return Err(PeError::BadMagic("PE32+ optional magic 0x20B"));
        }

        let entry_rva = rd_u32(data, add_usize(opt, 16, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let image_base = rd_u64(data, add_usize(opt, 24, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let size_of_image = rd_u32(data, add_usize(opt, 56, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let size_of_headers = rd_u32(data, add_usize(opt, 60, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let subsystem = rd_u16(data, add_usize(opt, 68, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let dll_characteristics = rd_u16(data, add_usize(opt, 70, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let stack_reserve = rd_u64(data, add_usize(opt, 72, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let stack_commit = rd_u64(data, add_usize(opt, 80, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))?;
        let num_rva_and_sizes = rd_u32(data, add_usize(opt, 108, "optional header")?)
            .ok_or(PeError::Truncated("optional header"))? as usize;

        let max_dirs = opt_size.saturating_sub(112) / 8;
        let ndirs = num_rva_and_sizes.min(16).min(max_dirs);
        let mut dirs = [DataDir::default(); 16];
        let dir_base = add_usize(opt, 112, "data directories")?;
        for (i, d) in dirs.iter_mut().enumerate().take(ndirs) {
            let entry_off = add_usize(
                dir_base,
                i.checked_mul(8)
                    .ok_or(PeError::Overflow("data directory"))?,
                "data directory",
            )?;
            d.rva = rd_u32(data, entry_off).ok_or(PeError::Truncated("data directories"))?;
            d.size = rd_u32(data, add_usize(entry_off, 4, "data directory")?)
                .ok_or(PeError::Truncated("data directories"))?;
        }

        let sec_base = add_usize(opt, opt_size, "section table")?;
        let mut sections = Vec::with_capacity(nsec);
        for i in 0..nsec {
            let off = add_usize(
                sec_base,
                i.checked_mul(40)
                    .ok_or(PeError::Overflow("section table"))?,
                "section table",
            )?;
            let name_end = add_usize(off, 8, "section name")?;
            let raw_name = data
                .get(off..name_end)
                .ok_or(PeError::Truncated("section table"))?;
            let nul = raw_name.iter().position(|&b| b == 0).unwrap_or(8);
            sections.push(Section {
                name: String::from_utf8_lossy(&raw_name[..nul]).into_owned(),
                virtual_size: rd_u32(data, add_usize(off, 8, "section table")?)
                    .ok_or(PeError::Truncated("section table"))?,
                virtual_address: rd_u32(data, add_usize(off, 12, "section table")?)
                    .ok_or(PeError::Truncated("section table"))?,
                raw_size: rd_u32(data, add_usize(off, 16, "section table")?)
                    .ok_or(PeError::Truncated("section table"))?,
                raw_offset: rd_u32(data, add_usize(off, 20, "section table")?)
                    .ok_or(PeError::Truncated("section table"))?,
                characteristics: rd_u32(data, add_usize(off, 36, "section table")?)
                    .ok_or(PeError::Truncated("section table"))?,
            });
        }

        Ok(Self {
            data,
            machine,
            characteristics,
            timestamp,
            subsystem,
            dll_characteristics,
            image_base,
            size_of_image,
            size_of_headers,
            entry_rva,
            stack_reserve,
            stack_commit,
            sections,
            dirs,
        })
    }

    pub fn is_dll(&self) -> bool {
        self.characteristics & IMAGE_FILE_DLL != 0
    }

    pub fn has_relocations(&self) -> bool {
        (self.characteristics & IMAGE_FILE_RELOCS_STRIPPED == 0)
            && self.dirs[DIR_BASERELOC].is_present()
    }

    pub fn exception_entry_count(&self) -> u32 {
        self.dirs[DIR_EXCEPTION].size / 12
    }

    fn rva_tail(&self, rva: u32) -> Option<&'a [u8]> {
        if rva < self.size_of_headers {
            let off = rva as usize;
            let len = (self.size_of_headers - rva) as usize;
            let end = off.checked_add(len)?.min(self.data.len());
            return self.data.get(off..end);
        }
        for s in &self.sections {
            let span = s.mapped_size().max(s.raw_size);
            let sec_end = s.virtual_address.checked_add(span)?;
            if rva >= s.virtual_address && rva < sec_end {
                let delta = rva - s.virtual_address;
                if delta >= s.raw_size {
                    return None;
                }
                let off = (s.raw_offset as usize).checked_add(delta as usize)?;
                let avail = (s.raw_size - delta) as usize;
                let end = off.checked_add(avail)?.min(self.data.len());
                return self.data.get(off..end);
            }
        }
        None
    }

    fn bytes_at(&self, rva: u32, len: usize) -> Option<&'a [u8]> {
        self.rva_tail(rva)?.get(..len)
    }

    fn u16_at(&self, rva: u32) -> Option<u16> {
        let b = self.bytes_at(rva, 2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32_at(&self, rva: u32) -> Option<u32> {
        let b = self.bytes_at(rva, 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64_at(&self, rva: u32) -> Option<u64> {
        let b = self.bytes_at(rva, 8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn cstr_at(&self, rva: u32) -> Result<String, PeError> {
        let tail = self.rva_tail(rva).ok_or(PeError::BadRva(rva))?;
        let end = tail
            .iter()
            .take(MAX_CSTR_LEN)
            .position(|&b| b == 0)
            .ok_or(PeError::UnterminatedString(rva))?;
        Ok(String::from_utf8_lossy(&tail[..end]).into_owned())
    }

    fn va_to_rva(&self, va: u64) -> Option<u32> {
        let delta = va.checked_sub(self.image_base)?;
        u32::try_from(delta).ok()
    }

    fn read_thunks(&self, int_rva: u32, iat_rva: u32) -> Result<Vec<ImportEntry>, PeError> {
        let mut entries = Vec::new();
        let lookup_rva = if int_rva != 0 { int_rva } else { iat_rva };
        if lookup_rva == 0 {
            return Ok(entries);
        }
        for idx in 0..MAX_IMPORT_THUNKS {
            let byte_off = idx
                .checked_mul(8)
                .ok_or(PeError::Overflow("import thunk"))?;
            let slot_rva = lookup_rva
                .checked_add(byte_off)
                .ok_or(PeError::BadRva(lookup_rva))?;
            let raw = self.u64_at(slot_rva).ok_or(PeError::BadRva(slot_rva))?;
            if raw == 0 {
                break;
            }
            let symbol = if raw & (1u64 << 63) != 0 {
                ImportSymbol::Ordinal {
                    ordinal: (raw & 0xFFFF) as u16,
                }
            } else {
                let hn_rva = u32::try_from(raw & 0x7FFF_FFFF).map_err(|_| PeError::BadRva(0))?;
                let hint = self.u16_at(hn_rva).ok_or(PeError::BadRva(hn_rva))?;
                let name_rva = hn_rva.checked_add(2).ok_or(PeError::BadRva(hn_rva))?;
                let name = self.cstr_at(name_rva)?;
                ImportSymbol::Name { hint, name }
            };
            let entry_iat = iat_rva
                .checked_add(byte_off)
                .ok_or(PeError::BadRva(iat_rva))?;
            entries.push(ImportEntry {
                iat_rva: entry_iat,
                symbol,
            });
        }
        Ok(entries)
    }

    /// Parse regular imports (`IMAGE_DIRECTORY_ENTRY_IMPORT`).
    pub fn imports(&self) -> Result<Vec<ImportDll>, PeError> {
        let dir = self.dirs[DIR_IMPORT];
        if !dir.is_present() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for idx in 0..MAX_IMPORT_DLLS {
            let desc_off = idx
                .checked_mul(20)
                .ok_or(PeError::Overflow("import descriptor"))?;
            let desc_rva = dir
                .rva
                .checked_add(desc_off)
                .ok_or(PeError::BadRva(dir.rva))?;
            let read_field = |off: u32| -> Result<u32, PeError> {
                let rva = desc_rva.checked_add(off).ok_or(PeError::BadRva(desc_rva))?;
                self.u32_at(rva).ok_or(PeError::BadRva(rva))
            };
            let int_rva = read_field(0)?;
            let name_rva = read_field(12)?;
            let iat_rva = read_field(16)?;
            if int_rva == 0 && name_rva == 0 && iat_rva == 0 {
                break;
            }
            let dll = self.cstr_at(name_rva)?;
            let entries = self.read_thunks(int_rva, iat_rva)?;
            out.push(ImportDll {
                dll,
                delay_load: false,
                entries,
            });
        }
        Ok(out)
    }

    /// Parse delay-load imports (`IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`).
    pub fn delay_imports(&self) -> Result<Vec<ImportDll>, PeError> {
        let dir = self.dirs[DIR_DELAY_IMPORT];
        if !dir.is_present() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for idx in 0..MAX_IMPORT_DLLS {
            let desc_off = idx
                .checked_mul(32)
                .ok_or(PeError::Overflow("delay-import descriptor"))?;
            let desc_rva = dir
                .rva
                .checked_add(desc_off)
                .ok_or(PeError::BadRva(dir.rva))?;
            let read_field = |off: u32| -> Result<u32, PeError> {
                let rva = desc_rva.checked_add(off).ok_or(PeError::BadRva(desc_rva))?;
                self.u32_at(rva).ok_or(PeError::BadRva(rva))
            };
            let attrs = read_field(0)?;
            let mut name_rva = read_field(4)?;
            let mut iat_rva = read_field(12)?;
            let mut int_rva = read_field(16)?;
            if name_rva == 0 && iat_rva == 0 {
                break;
            }
            if attrs & 1 == 0 {
                name_rva = self.va_to_rva(u64::from(name_rva)).unwrap_or(name_rva);
                iat_rva = self.va_to_rva(u64::from(iat_rva)).unwrap_or(iat_rva);
                int_rva = self.va_to_rva(u64::from(int_rva)).unwrap_or(int_rva);
            }
            let dll = self.cstr_at(name_rva)?;
            let entries = self.read_thunks(int_rva, iat_rva)?;
            out.push(ImportDll {
                dll,
                delay_load: true,
                entries,
            });
        }
        Ok(out)
    }

    /// Parse static TLS directory (`IMAGE_DIRECTORY_ENTRY_TLS`), if present.
    pub fn tls(&self) -> Result<Option<TlsInfo>, PeError> {
        let dir = self.dirs[DIR_TLS];
        if !dir.is_present() {
            return Ok(None);
        }
        let field_u64 = |off: u32| -> Result<u64, PeError> {
            let rva = dir.rva.checked_add(off).ok_or(PeError::BadRva(dir.rva))?;
            self.u64_at(rva).ok_or(PeError::BadRva(rva))
        };
        let field_u32 = |off: u32| -> Result<u32, PeError> {
            let rva = dir.rva.checked_add(off).ok_or(PeError::BadRva(dir.rva))?;
            self.u32_at(rva).ok_or(PeError::BadRva(rva))
        };
        Ok(Some(TlsInfo {
            directory_rva: dir.rva,
            start_raw_va: field_u64(0)?,
            end_raw_va: field_u64(8)?,
            address_of_index: field_u64(16)?,
            address_of_callbacks: field_u64(24)?,
            size_of_zero_fill: field_u32(32)?,
            characteristics: field_u32(36)?,
        }))
    }
}
