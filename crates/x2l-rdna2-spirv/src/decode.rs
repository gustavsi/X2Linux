// RDNA2 (GFX10.3) + console GFX10.1 opcode rules informed by xcl-dev/xcl
// (https://github.com/xcl-dev/xcl, commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102,
// crates/xcl/src/gpu/rdna2_xbox.rs & rdna2.rs) and Marvisak/VIctory
// (https://github.com/Marvisak/VIctory, commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-shader/src/gfx10/mod.rs & fields.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27 with strict bounds-checked decoding and reachable CFG walk.

//! RDNA2 (GFX10.3) + Xbox `gfx10.1` instruction decoder and disassembler.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use thiserror::Error;

/// Errors raised while decoding RDNA2 instruction streams.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DecodeError {
    #[error("instruction stream is not DWORD-aligned ({0} bytes)")]
    UnalignedBytes(usize),
    #[error("instruction at PC 0x{pc:04x} truncated: needed {needed} DWORDs, only {avail} left")]
    Truncated {
        pc: u32,
        needed: usize,
        avail: usize,
    },
    #[error("branch at PC 0x{pc:04x} targets out-of-range or unaligned PC {target}")]
    InvalidBranchTarget { pc: u32, target: i64 },
}

/// RDNA2 instruction encoding family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    Sop1,
    Sop2,
    Sopc,
    Sopk,
    Sopp,
    Vop1,
    Vop2,
    Vopc,
    Vop3,
    Vintrp,
    Smem,
    Vmem,
    Mimg,
    Ds,
    Export,
    Unknown,
}

/// Typed source operand for scalar or vector ALU instructions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SrcOperand {
    Sgpr(u8),
    Vgpr(u8),
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    M0,
    Scc,
    Vccz,
    Execz,
    Null,
    InlineInt(i32),
    InlineFloat(f32),
    Literal(u32),
    Special(u16),
}

impl fmt::Display for SrcOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sgpr(n) => write!(f, "s{n}"),
            Self::Vgpr(n) => write!(f, "v{n}"),
            Self::VccLo => f.write_str("vcc_lo"),
            Self::VccHi => f.write_str("vcc_hi"),
            Self::ExecLo => f.write_str("exec_lo"),
            Self::ExecHi => f.write_str("exec_hi"),
            Self::M0 => f.write_str("m0"),
            Self::Scc => f.write_str("scc"),
            Self::Vccz => f.write_str("vccz"),
            Self::Execz => f.write_str("execz"),
            Self::Null => f.write_str("null"),
            Self::InlineInt(v) => write!(f, "{v}"),
            Self::InlineFloat(v) => write!(f, "{v}"),
            Self::Literal(bits) => write!(f, "0x{bits:08x}"),
            Self::Special(code) => write!(f, "src({code})"),
        }
    }
}

/// Typed destination operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DstOperand {
    None,
    Sgpr(u8),
    Vgpr(u8),
    VccLo,
    VccHi,
    ExecLo,
    ExecHi,
    M0,
    Null,
}

impl fmt::Display for DstOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Sgpr(n) => write!(f, "s{n}"),
            Self::Vgpr(n) => write!(f, "v{n}"),
            Self::VccLo => f.write_str("vcc_lo"),
            Self::VccHi => f.write_str("vcc_hi"),
            Self::ExecLo => f.write_str("exec_lo"),
            Self::ExecHi => f.write_str("exec_hi"),
            Self::M0 => f.write_str("m0"),
            Self::Null => f.write_str("null"),
        }
    }
}

/// VOP3 input/output modifiers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValuModifiers {
    pub abs: [bool; 3],
    pub neg: [bool; 3],
    pub clamp: bool,
    pub omod: u8,
}

/// Compact VALU extension word (SDWA / DPP16 / DPP8).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValuExt {
    #[default]
    None,
    Sdwa {
        word: u32,
    },
    Dpp16 {
        word: u32,
    },
    Dpp8 {
        word: u32,
        fetch_inactive: bool,
    },
}

/// Typed extra payload for cross-lane, DS, and memory instructions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InstExtra {
    #[default]
    None,
    Permlane {
        fi: bool,
        bc: bool,
    },
    DsSwizzle {
        offset: u16,
        vsrc: u8,
    },
    DsLds {
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
        offset0: u8,
        offset1: u8,
        gds: bool,
        is_write: bool,
        paired: bool,
    },
    Exp {
        target: u8,
        en: u8,
        compr: bool,
        done: bool,
        vm: bool,
        vsrc: [u8; 4],
    },
    Vintrp {
        /// 0 = p1, 1 = p2, 2 = mov
        op: u8,
        vdst: u8,
        attr: u8,
        chan: u8,
        vsrc: u8,
    },
    Mimg {
        /// Full 8-bit opcode (`op_lo | (op_hi << 7)`):
        /// 0x00=load, 0x01=load_mip, 0x08=store, 0x09=store_mip, 0x20=sample, 0x24=sample_l
        op: u16,
        dim: u8,
        dmask: u8,
        unrm: bool,
        r128: bool,
        tfe: bool,
        lwe: bool,
        a16: bool,
        d16: bool,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
        /// Number of explicit address VGPRs stored in `addrs` (1 when `nsa==0`, else `1 + 4*nsa`).
        addr_count: u8,
        /// Whether `nsa > 0` (non-sequential address VGPR list).
        is_nsa: bool,
        addrs: [u8; 13],
    },
    Smem {
        /// First SGPR index (already shifted `SBASE << 1`).
        sbase: u8,
        /// First destination SGPR index.
        sdata: u8,
        /// Raw 7-bit scalar selector (`0x7D` = NULL / 0).
        soffset: u8,
        /// Sign-extended 21-bit byte displacement.
        ioffset: i32,
        /// Transfer width in DWORDs (1, 2, or 4).
        dwords: u8,
        /// True for `s_buffer_load_*` (V# descriptor), false for `s_load_*` (64-bit address).
        is_buffer: bool,
    },
    Mubuf {
        vaddr: u8,
        vdata: u8,
        /// First SGPR of V# descriptor (already shifted `SRSRC << 2`).
        srsrc: u8,
        /// Raw 8-bit scalar offset selector.
        soffset: u8,
        /// Unsigned 12-bit byte offset.
        offset: u16,
        offen: bool,
        idxen: bool,
        lds: bool,
        tfe: bool,
        /// Transfer width in DWORDs (1, 2, 3, or 4).
        dwords: u8,
        is_store: bool,
    },
    Flat {
        /// Segment: 0=FLAT, 1=SCRATCH, 2=GLOBAL, 3=reserved.
        seg: u8,
        addr: u8,
        data: u8,
        vdst: u8,
        /// 0x7D = NULL (use 64-bit VGPR pair at `addr`), otherwise 64-bit SGPR base + 32-bit `v[addr]`.
        saddr: u8,
        /// Signed 12-bit byte offset for GLOBAL, unsigned 11-bit for FLAT.
        offset: i16,
        lds: bool,
        /// Transfer width in DWORDs (1, 2, 3, or 4).
        dwords: u8,
        is_store: bool,
    },
}

impl InstExtra {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}

/// One decoded RDNA2 / Xbox `gfx10.1` instruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodedInst {
    /// Byte offset from the start of the code slice.
    pub pc: u32,
    /// Instruction width in 32-bit DWORDs (1, 2, or 3).
    pub word_len: u8,
    pub encoding: Encoding,
    pub opcode: u16,
    pub mnemonic: String,
    pub dst: DstOperand,
    pub srcs: Vec<SrcOperand>,
    pub literal: Option<u32>,
    pub simm16: Option<i16>,
    /// Resolved byte target for direct branches (`PC + 4 + 4 * sign_extend(SIMM16)`).
    pub branch_target: Option<u32>,
    pub modifiers: ValuModifiers,
    pub ext: ValuExt,
    #[serde(default, skip_serializing_if = "InstExtra::is_none")]
    pub extra: InstExtra,
    pub is_terminator: bool,
    pub raw_words: Vec<u32>,
}

impl fmt::Display for DecodedInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}: {}", self.pc, self.mnemonic)?;
        let mut first = true;
        if self.dst != DstOperand::None {
            write!(f, " {}", self.dst)?;
            first = false;
        }
        for (idx, src) in self.srcs.iter().enumerate() {
            if first {
                write!(f, " ")?;
                first = false;
            } else {
                write!(f, ", ")?;
            }
            let neg = idx < 3 && self.modifiers.neg[idx];
            let abs = idx < 3 && self.modifiers.abs[idx];
            if neg {
                write!(f, "-")?;
            }
            if abs {
                write!(f, "|{src}|")?;
            } else {
                write!(f, "{src}")?;
            }
        }
        if let Some(target) = self.branch_target {
            write!(f, " 0x{target:04x}")?;
        } else if let Some(imm) = self.simm16
            && self.encoding == Encoding::Sopp
            && !matches!(self.opcode, 0x01 | 0x1F)
        {
            write!(f, " 0x{:04x}", imm as u16)?;
        }
        if self.modifiers.clamp {
            write!(f, " clamp")?;
        }
        match self.modifiers.omod {
            1 => write!(f, " mul:2")?,
            2 => write!(f, " mul:4")?,
            3 => write!(f, " div:2")?,
            _ => {}
        }
        match self.ext {
            ValuExt::None => {}
            ValuExt::Sdwa { word } => write!(f, " sdwa(0x{word:08x})")?,
            ValuExt::Dpp16 { word } => write!(f, " dpp16(0x{word:08x})")?,
            ValuExt::Dpp8 {
                word,
                fetch_inactive,
            } => write!(f, " dpp8(0x{word:08x}, fi={fetch_inactive})")?,
        }
        Ok(())
    }
}

/// Disassemble a little-endian byte slice of RDNA2 machine code.
pub fn disassemble_bytes(bytes: &[u8]) -> Result<Vec<DecodedInst>, DecodeError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(DecodeError::UnalignedBytes(bytes.len()));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let words: Vec<u32> = chunks.iter().map(|c| u32::from_le_bytes(*c)).collect();
    disassemble(&words)
}

/// Linearly disassemble a slice of RDNA2 DWORDs from offset 0.
pub fn disassemble(words: &[u32]) -> Result<Vec<DecodedInst>, DecodeError> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < words.len() {
        let inst = decode_one(words, idx)?;
        let step = inst.word_len as usize;
        out.push(inst);
        idx += step;
    }
    Ok(out)
}

/// Decode all instructions reachable from PC 0 following scalar branches across multiple
/// `s_endpgm` exits, skipping unreachable trailing constant pools or padding.
pub fn disassemble_reachable(words: &[u32]) -> Result<Vec<DecodedInst>, DecodeError> {
    if words.is_empty() {
        return Ok(Vec::new());
    }
    let mut decoded: BTreeMap<usize, DecodedInst> = BTreeMap::new();
    let mut queued: BTreeSet<usize> = BTreeSet::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();
    worklist.push_back(0);
    queued.insert(0);

    while let Some(mut idx) = worklist.pop_front() {
        while idx < words.len() {
            if decoded.contains_key(&idx) {
                break;
            }
            let inst = decode_one(words, idx)?;
            let step = inst.word_len as usize;
            let next_idx = idx + step;
            let target_idx = inst.branch_target.map(|b| (b / 4) as usize);
            let is_uncond_branch = inst.encoding == Encoding::Sopp && inst.opcode == 0x02;
            let is_term = inst.is_terminator;

            if let Some(t) = target_idx {
                if t >= words.len() {
                    return Err(DecodeError::InvalidBranchTarget {
                        pc: inst.pc,
                        target: (t * 4) as i64,
                    });
                }
                if queued.insert(t) {
                    worklist.push_back(t);
                }
            }

            decoded.insert(idx, inst);
            if is_term || is_uncond_branch {
                break;
            }
            idx = next_idx;
        }
    }

    Ok(decoded.into_values().collect())
}

/// Format decoded instructions into multi-line human-readable assembly text.
pub fn format_disassembly(insts: &[DecodedInst]) -> String {
    let mut out = String::new();
    for inst in insts {
        out.push_str(&inst.to_string());
        out.push('\n');
    }
    out
}

/// Decode a single RDNA2 / Xbox `gfx10.1` instruction starting at `words[idx]`.
pub fn decode_one(words: &[u32], idx: usize) -> Result<DecodedInst, DecodeError> {
    let pc = (idx * 4) as u32;
    let w0 = *words.get(idx).ok_or(DecodeError::Truncated {
        pc,
        needed: 1,
        avail: 0,
    })?;

    // 1. VOP2: bit 31 == 0, except VOP1 (bits 31:25 == 0x3F) and VOPC (bits 31:25 == 0x3E)
    if (w0 & 0x8000_0000) == 0 {
        let top7 = w0 >> 25;
        return match top7 {
            0x3F => decode_vop1(words, idx, pc, w0),
            0x3E => decode_vopc(words, idx, pc, w0),
            _ => decode_vop2(words, idx, pc, w0),
        };
    }

    // 2. Scalar 32-bit encodings (bits 31:30 == 0b10)
    if (w0 >> 30) == 0b10 {
        let top9 = w0 >> 23;
        return match top9 {
            0x17F => decode_sopp(pc, w0),
            0x17E => decode_sopc(words, idx, pc, w0),
            0x17D => decode_sop1(words, idx, pc, w0),
            _ if (w0 >> 28) == 0xB => decode_sopk(words, idx, pc, w0),
            _ => decode_sop2(words, idx, pc, w0),
        };
    }

    // 3. 32/64/NSA encodings with bits 31:30 == 0b11
    let top6 = w0 >> 26;
    match top6 {
        0x32 => decode_vintrp(pc, w0),
        0x35 => decode_vop3(words, idx, pc, w0),
        0x3D => decode_smem(words, idx, pc, w0),
        0x36 => decode_ds(words, idx, pc, w0),
        0x38 => decode_mubuf(words, idx, pc, w0),
        0x37 => decode_flat(words, idx, pc, w0),
        0x3C => decode_mimg(words, idx, pc, w0),
        0x3E => decode_exp(words, idx, pc, w0),
        0x39..=0x3B => decode_two_word_generic(words, idx, pc, w0, Encoding::Vmem, "vmem"),
        _ => Ok(DecodedInst {
            pc,
            word_len: 1,
            encoding: Encoding::Unknown,
            opcode: (w0 >> 24) as u16,
            mnemonic: format!(".long 0x{w0:08x}"),
            dst: DstOperand::None,
            srcs: Vec::new(),
            literal: None,
            simm16: None,
            branch_target: None,
            modifiers: ValuModifiers::default(),
            ext: ValuExt::None,
            extra: InstExtra::None,
            is_terminator: false,
            raw_words: vec![w0],
        }),
    }
}

fn fetch_word(words: &[u32], idx: usize, offset: usize, pc: u32) -> Result<u32, DecodeError> {
    let target = idx.checked_add(offset).ok_or(DecodeError::Truncated {
        pc,
        needed: offset + 1,
        avail: words.len().saturating_sub(idx),
    })?;
    words.get(target).copied().ok_or(DecodeError::Truncated {
        pc,
        needed: offset + 1,
        avail: words.len().saturating_sub(idx),
    })
}

fn decode_scalar_dst(code: u8) -> DstOperand {
    match code {
        0..=105 => DstOperand::Sgpr(code),
        106 => DstOperand::VccLo,
        107 => DstOperand::VccHi,
        124 => DstOperand::M0,
        125 => DstOperand::Null,
        126 => DstOperand::ExecLo,
        127 => DstOperand::ExecHi,
        _ => DstOperand::Sgpr(code),
    }
}

fn decode_src9(raw: u16, literal: Option<u32>) -> SrcOperand {
    match raw {
        0..=105 => SrcOperand::Sgpr(raw as u8),
        106 => SrcOperand::VccLo,
        107 => SrcOperand::VccHi,
        124 => SrcOperand::M0,
        125 => SrcOperand::Null,
        126 => SrcOperand::ExecLo,
        127 => SrcOperand::ExecHi,
        128..=192 => SrcOperand::InlineInt((raw as i32) - 128),
        193..=208 => SrcOperand::InlineInt(-((raw as i32) - 192)),
        240 => SrcOperand::InlineFloat(0.5),
        241 => SrcOperand::InlineFloat(-0.5),
        242 => SrcOperand::InlineFloat(1.0),
        243 => SrcOperand::InlineFloat(-1.0),
        244 => SrcOperand::InlineFloat(2.0),
        245 => SrcOperand::InlineFloat(-2.0),
        246 => SrcOperand::InlineFloat(4.0),
        247 => SrcOperand::InlineFloat(-4.0),
        248 => SrcOperand::InlineFloat(std::f32::consts::FRAC_1_PI * 0.5),
        251 => SrcOperand::Vccz,
        252 => SrcOperand::Execz,
        253 => SrcOperand::Scc,
        255 => SrcOperand::Literal(literal.unwrap_or(0)),
        256..=511 => SrcOperand::Vgpr((raw - 256) as u8),
        other => SrcOperand::Special(other),
    }
}

fn is_valu_ext_selector(src0: u16) -> bool {
    matches!(src0, 0xE9 | 0xEA | 0xF9 | 0xFA)
}

fn decode_valu_ext(src0: u16, ext_word: u32) -> (SrcOperand, ValuExt) {
    let low = (ext_word & 0xFF) as u16;
    match src0 {
        0xF9 => {
            // SDWA: bit 24 (s0) selects scalar vs VGPR
            let is_scalar = ((ext_word >> 24) & 1) != 0;
            let op = if is_scalar {
                decode_src9(low, None)
            } else {
                SrcOperand::Vgpr(low as u8)
            };
            (op, ValuExt::Sdwa { word: ext_word })
        }
        0xFA => (
            SrcOperand::Vgpr(low as u8),
            ValuExt::Dpp16 { word: ext_word },
        ),
        0xE9 => (
            SrcOperand::Vgpr(low as u8),
            ValuExt::Dpp8 {
                word: ext_word,
                fetch_inactive: false,
            },
        ),
        0xEA => (
            SrcOperand::Vgpr(low as u8),
            ValuExt::Dpp8 {
                word: ext_word,
                fetch_inactive: true,
            },
        ),
        _ => (decode_src9(src0, None), ValuExt::None),
    }
}

fn decode_sopp(pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 16) & 0x7F) as u16;
    let simm16 = (w0 & 0xFFFF) as i16;
    let mnemonic = match op {
        0x00 => "s_nop",
        0x01 => "s_endpgm",
        0x02 => "s_branch",
        0x04 => "s_cbranch_scc0",
        0x05 => "s_cbranch_scc1",
        0x06 => "s_cbranch_vccz",
        0x07 => "s_cbranch_vccnz",
        0x08 => "s_cbranch_execz",
        0x09 => "s_cbranch_execnz",
        0x0A => "s_barrier",
        0x0C => "s_waitcnt",
        0x10 => "s_sendmsg",
        0x1B => "s_endpgm_saved",
        0x1F => "s_code_end",
        0x20 => "s_inst_prefetch",
        0x21 => "s_clause",
        0x23 => "s_waitcnt_depctr",
        _ => "",
    };
    let name = if mnemonic.is_empty() {
        format!("sopp_0x{op:02x}")
    } else {
        mnemonic.to_owned()
    };

    let is_branch = matches!(op, 0x02 | 0x04..=0x09);
    let branch_target = if is_branch {
        let delta = i64::from(simm16) * 4;
        let target = i64::from(pc) + 4 + delta;
        if target < 0 || target > i64::from(u32::MAX) {
            return Err(DecodeError::InvalidBranchTarget { pc, target });
        }
        Some(target as u32)
    } else {
        None
    };

    Ok(DecodedInst {
        pc,
        word_len: 1,
        encoding: Encoding::Sopp,
        opcode: op,
        mnemonic: name,
        dst: DstOperand::None,
        srcs: Vec::new(),
        literal: None,
        simm16: Some(simm16),
        branch_target,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::None,
        is_terminator: matches!(op, 0x01 | 0x1B | 0x1F),
        raw_words: vec![w0],
    })
}

fn decode_sopc(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 16) & 0x7F) as u16;
    let s0 = (w0 & 0xFF) as u16;
    let s1 = ((w0 >> 8) & 0xFF) as u16;
    let needs_lit = s0 == 0xFF || s1 == 0xFF;
    let literal = if needs_lit {
        Some(fetch_word(words, idx, 1, pc)?)
    } else {
        None
    };
    let mnemonic = match op {
        0x00 => "s_cmp_eq_i32",
        0x01 => "s_cmp_lg_i32",
        0x02 => "s_cmp_gt_i32",
        0x03 => "s_cmp_ge_i32",
        0x04 => "s_cmp_lt_i32",
        0x05 => "s_cmp_le_i32",
        0x06 => "s_cmp_eq_u32",
        0x07 => "s_cmp_lg_u32",
        0x08 => "s_cmp_gt_u32",
        0x09 => "s_cmp_ge_u32",
        0x0A => "s_cmp_lt_u32",
        0x0B => "s_cmp_le_u32",
        0x0C => "s_bitcmp0_b32",
        0x0D => "s_bitcmp1_b32",
        0x10 => "s_cmp_eq_u64",
        0x11 => "s_cmp_lg_u64",
        _ => "",
    };
    let name = if mnemonic.is_empty() {
        format!("sopc_0x{op:02x}")
    } else {
        mnemonic.to_owned()
    };
    let mut raw_words = vec![w0];
    if let Some(lit) = literal {
        raw_words.push(lit);
    }
    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Sopc,
        opcode: op,
        mnemonic: name,
        dst: DstOperand::None,
        srcs: vec![decode_src9(s0, literal), decode_src9(s1, literal)],
        literal,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words,
    })
}

fn decode_sop1(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 8) & 0xFF) as u16;
    let s0 = (w0 & 0xFF) as u16;
    let sdst = ((w0 >> 16) & 0x7F) as u8;
    let literal = if s0 == 0xFF {
        Some(fetch_word(words, idx, 1, pc)?)
    } else {
        None
    };
    let mnemonic = match op {
        0x03 => "s_mov_b32",
        0x04 => "s_mov_b64",
        0x05 => "s_cmov_b32",
        0x06 => "s_cmov_b64",
        0x07 => "s_not_b32",
        0x08 => "s_not_b64",
        0x1F => "s_getpc_b64",
        0x20 => "s_setpc_b64",
        0x21 => "s_swappc_b64",
        0x24 => "s_and_saveexec_b64",
        0x25 => "s_or_saveexec_b64",
        0x26 => "s_xor_saveexec_b64",
        0x27 => "s_andn2_saveexec_b64",
        0x3C => "s_and_saveexec_b32",
        0x3D => "s_or_saveexec_b32",
        0x3E => "s_xor_saveexec_b32",
        0x3F => "s_andn2_saveexec_b32",
        _ => "",
    };
    let name = if mnemonic.is_empty() {
        format!("sop1_0x{op:02x}")
    } else {
        mnemonic.to_owned()
    };
    let mut raw_words = vec![w0];
    if let Some(lit) = literal {
        raw_words.push(lit);
    }
    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Sop1,
        opcode: op,
        mnemonic: name,
        dst: decode_scalar_dst(sdst),
        srcs: vec![decode_src9(s0, literal)],
        literal,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::None,
        is_terminator: op == 0x20,
        raw_words,
    })
}

fn decode_sopk(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 23) & 0x1F) as u16;
    let sdst = ((w0 >> 16) & 0x7F) as u8;
    let simm16 = (w0 & 0xFFFF) as i16;
    let literal = if op == 0x15 {
        Some(fetch_word(words, idx, 1, pc)?)
    } else {
        None
    };
    let mnemonic = match op {
        0x00 => "s_movk_i32",
        0x02 => "s_cmovk_i32",
        0x03 => "s_cmpk_eq_i32",
        0x04 => "s_cmpk_lg_i32",
        0x05 => "s_cmpk_gt_i32",
        0x06 => "s_cmpk_ge_i32",
        0x07 => "s_cmpk_lt_i32",
        0x08 => "s_cmpk_le_i32",
        0x09 => "s_cmpk_eq_u32",
        0x0A => "s_cmpk_lg_u32",
        0x0F => "s_addk_i32",
        0x10 => "s_mulk_i32",
        0x15 => "s_setreg_imm32_b32",
        0x17 => "s_waitcnt_vscnt",
        0x18 => "s_waitcnt_vmcnt",
        0x19 => "s_waitcnt_expcnt",
        0x1A => "s_waitcnt_lgkmcnt",
        _ => "",
    };
    let name = if mnemonic.is_empty() {
        format!("sopk_0x{op:02x}")
    } else {
        mnemonic.to_owned()
    };
    let is_waitcnt = matches!(op, 0x17..=0x1A);
    let mut raw_words = vec![w0];
    let mut srcs = if is_waitcnt {
        vec![
            decode_src9(u16::from(sdst), None),
            SrcOperand::InlineInt(i32::from(simm16)),
        ]
    } else {
        vec![SrcOperand::InlineInt(i32::from(simm16))]
    };
    if let Some(lit) = literal {
        raw_words.push(lit);
        srcs.push(SrcOperand::Literal(lit));
    }
    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Sopk,
        opcode: op,
        mnemonic: name,
        dst: if is_waitcnt {
            DstOperand::None
        } else {
            decode_scalar_dst(sdst)
        },
        srcs,
        literal,
        simm16: Some(simm16),
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words,
    })
}

fn decode_sop2(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 23) & 0x7F) as u16;
    let sdst = ((w0 >> 16) & 0x7F) as u8;
    let s1 = ((w0 >> 8) & 0xFF) as u16;
    let s0 = (w0 & 0xFF) as u16;
    let needs_lit = s0 == 0xFF || s1 == 0xFF;
    let literal = if needs_lit {
        Some(fetch_word(words, idx, 1, pc)?)
    } else {
        None
    };
    let mnemonic = match op {
        0x00 => "s_add_u32",
        0x01 => "s_sub_u32",
        0x02 => "s_add_i32",
        0x03 => "s_sub_i32",
        0x04 => "s_addc_u32",
        0x05 => "s_subb_u32",
        0x06 => "s_min_i32",
        0x07 => "s_min_u32",
        0x08 => "s_max_i32",
        0x09 => "s_max_u32",
        0x0A => "s_cselect_b32",
        0x0B => "s_cselect_b64",
        0x0E => "s_and_b32",
        0x0F => "s_and_b64",
        0x10 => "s_or_b32",
        0x11 => "s_or_b64",
        0x12 => "s_xor_b32",
        0x13 => "s_xor_b64",
        0x14 => "s_andn2_b32",
        0x15 => "s_andn2_b64",
        0x1C => "s_lshl_b32",
        0x1E => "s_lshr_b32",
        0x20 => "s_ashr_i32",
        0x2C => "s_mul_i32",
        _ => "",
    };
    let name = if mnemonic.is_empty() {
        format!("sop2_0x{op:02x}")
    } else {
        mnemonic.to_owned()
    };
    let mut raw_words = vec![w0];
    if let Some(lit) = literal {
        raw_words.push(lit);
    }
    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Sop2,
        opcode: op,
        mnemonic: name,
        dst: decode_scalar_dst(sdst),
        srcs: vec![decode_src9(s0, literal), decode_src9(s1, literal)],
        literal,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words,
    })
}

fn decode_vop1(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 9) & 0xFF) as u16;
    let vdst = ((w0 >> 17) & 0xFF) as u8;
    let src0 = (w0 & 0x1FF) as u16;

    let mut raw_words = vec![w0];
    let (src_op, literal, ext) = if is_valu_ext_selector(src0) {
        let ext_word = fetch_word(words, idx, 1, pc)?;
        raw_words.push(ext_word);
        let (op0, ext) = decode_valu_ext(src0, ext_word);
        (op0, None, ext)
    } else if src0 == 0xFF {
        let lit = fetch_word(words, idx, 1, pc)?;
        raw_words.push(lit);
        (SrcOperand::Literal(lit), Some(lit), ValuExt::None)
    } else {
        (decode_src9(src0, None), None, ValuExt::None)
    };

    let mnemonic = vop1_mnemonic(op);
    let dst = if op == 0x00 {
        DstOperand::None
    } else if op == 0x02 {
        decode_scalar_dst(vdst & 0x7F)
    } else {
        DstOperand::Vgpr(vdst)
    };
    let srcs = if op == 0x00 { Vec::new() } else { vec![src_op] };

    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Vop1,
        opcode: op,
        mnemonic,
        dst,
        srcs,
        literal,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words,
    })
}

fn decode_vopc(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 17) & 0xFF) as u16;
    let vsrc1 = ((w0 >> 9) & 0xFF) as u8;
    let src0 = (w0 & 0x1FF) as u16;

    let mut raw_words = vec![w0];
    let (src0_op, literal, ext) = if is_valu_ext_selector(src0) {
        let ext_word = fetch_word(words, idx, 1, pc)?;
        raw_words.push(ext_word);
        let (op0, ext) = decode_valu_ext(src0, ext_word);
        (op0, None, ext)
    } else if src0 == 0xFF {
        let lit = fetch_word(words, idx, 1, pc)?;
        raw_words.push(lit);
        (SrcOperand::Literal(lit), Some(lit), ValuExt::None)
    } else {
        (decode_src9(src0, None), None, ValuExt::None)
    };

    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Vopc,
        opcode: op,
        mnemonic: vopc_mnemonic(op),
        dst: DstOperand::VccLo,
        srcs: vec![src0_op, SrcOperand::Vgpr(vsrc1)],
        literal,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words,
    })
}

fn decode_vop2(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let op = ((w0 >> 25) & 0x3F) as u16;
    let vdst = ((w0 >> 17) & 0xFF) as u8;
    let vsrc1 = ((w0 >> 9) & 0xFF) as u8;
    let src0 = (w0 & 0x1FF) as u16;

    let mandatory_lit = matches!(op, 0x20 | 0x21 | 0x2C | 0x2D | 0x37 | 0x38);
    let mut raw_words = vec![w0];
    let (src0_op, literal, ext) = if !mandatory_lit && is_valu_ext_selector(src0) {
        let ext_word = fetch_word(words, idx, 1, pc)?;
        raw_words.push(ext_word);
        let (op0, ext) = decode_valu_ext(src0, ext_word);
        (op0, None, ext)
    } else if mandatory_lit || src0 == 0xFF {
        let lit = fetch_word(words, idx, 1, pc)?;
        raw_words.push(lit);
        (decode_src9(src0, Some(lit)), Some(lit), ValuExt::None)
    } else {
        (decode_src9(src0, None), None, ValuExt::None)
    };

    // Mandatory-literal operand order:
    //   madmk / fmamk: D, S0, K, S1
    //   madak / fmaak: D, S0, S1, K
    let srcs = match op {
        0x20 | 0x2C | 0x37 => vec![
            src0_op,
            SrcOperand::Literal(literal.unwrap_or(0)),
            SrcOperand::Vgpr(vsrc1),
        ],
        0x21 | 0x2D | 0x38 => vec![
            src0_op,
            SrcOperand::Vgpr(vsrc1),
            SrcOperand::Literal(literal.unwrap_or(0)),
        ],
        _ => vec![src0_op, SrcOperand::Vgpr(vsrc1)],
    };

    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Vop2,
        opcode: op,
        mnemonic: vop2_mnemonic(op),
        dst: DstOperand::Vgpr(vdst),
        srcs,
        literal,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words,
    })
}

fn decode_vop3(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let op = ((w0 >> 16) & 0x3FF) as u16;
    let dst_raw = (w0 & 0xFF) as u8;
    let abs_bits = (w0 >> 8) & 0x7;
    let clamp = ((w0 >> 15) & 1) != 0;

    let s0 = (w1 & 0x1FF) as u16;
    let s1 = ((w1 >> 9) & 0x1FF) as u16;
    let s2 = ((w1 >> 18) & 0x1FF) as u16;
    let omod = ((w1 >> 27) & 0x3) as u8;
    let neg_bits = (w1 >> 29) & 0x7;

    // Hardware rule: if ANY raw 9-bit source field equals 0xFF (even unused ones), a shared
    // 32-bit literal DWORD follows at +8 bytes.
    let needs_lit = s0 == 0xFF || s1 == 0xFF || s2 == 0xFF;
    let mut raw_words = vec![w0, w1];
    let literal = if needs_lit {
        let lit = fetch_word(words, idx, 2, pc)?;
        raw_words.push(lit);
        Some(lit)
    } else {
        None
    };

    let modifiers = ValuModifiers {
        abs: [
            (abs_bits & 1) != 0,
            (abs_bits & 2) != 0,
            (abs_bits & 4) != 0,
        ],
        neg: [
            (neg_bits & 1) != 0,
            (neg_bits & 2) != 0,
            (neg_bits & 4) != 0,
        ],
        clamp,
        omod,
    };

    let mut extra = InstExtra::None;
    let (mnemonic, dst, num_srcs) = match op {
        // VOP3-encoded VOPC (0x000..=0x0FF): destination is scalar mask register (sdst).
        0x000..=0x0FF => (vopc_mnemonic(op), decode_scalar_dst(dst_raw), 2usize),
        // VOP3-encoded VOP2 (0x100..=0x13F), including gfx10.1 V_MAC_F32 (0x11F) and V_CNDMASK_B32 (0x101).
        0x101 => ("v_cndmask_b32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x100..=0x13F => (vop2_mnemonic(op - 0x100), DstOperand::Vgpr(dst_raw), 2),
        // VOP3-native 3-operand instructions + console gfx10.1 V_MAD_F32 (0x141).
        0x140 => ("v_fma_legacy_f32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x141 => ("v_mad_f32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x143 => ("v_mad_u32_u24".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x148 => ("v_bfe_u32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x149 => ("v_bfe_i32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x14A => ("v_bfi_b32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x14B => ("v_fma_f32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x151 => ("v_med3_f32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x169 => ("v_mul_lo_u32".to_owned(), DstOperand::Vgpr(dst_raw), 2),
        0x180..=0x1FF => {
            let v1_op = op - 0x180;
            let srcs = if v1_op == 0 { 0 } else { 1 };
            let dst = if v1_op == 0 {
                DstOperand::None
            } else if v1_op == 0x02 {
                decode_scalar_dst(dst_raw & 0x7F)
            } else {
                DstOperand::Vgpr(dst_raw)
            };
            (vop1_mnemonic(v1_op), dst, srcs)
        }
        0x30F => ("v_add_co_u32".to_owned(), DstOperand::Vgpr(dst_raw), 2),
        0x310 => ("v_sub_co_u32".to_owned(), DstOperand::Vgpr(dst_raw), 2),
        0x346 => ("v_lshl_add_u32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        // Cross-lane VOP3A operations (GFX10.3 verified)
        0x360 => (
            "v_readlane_b32".to_owned(),
            decode_scalar_dst(dst_raw & 0x7F),
            2,
        ),
        0x361 => ("v_writelane_b32".to_owned(), DstOperand::Vgpr(dst_raw), 2),
        0x36D => ("v_add3_u32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x36F => ("v_lshl_or_b32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x371 => ("v_and_or_b32".to_owned(), DstOperand::Vgpr(dst_raw), 3),
        0x377 | 0x378 => {
            extra = InstExtra::Permlane {
                fi: ((w0 >> 11) & 1) != 0,
                bc: ((w0 >> 12) & 1) != 0,
            };
            let name = if op == 0x377 {
                "v_permlane16_b32"
            } else {
                "v_permlanex16_b32"
            };
            (name.to_owned(), DstOperand::Vgpr(dst_raw), 3)
        }
        0x365 => (
            "v_mbcnt_lo_u32_b32".to_owned(),
            DstOperand::Vgpr(dst_raw),
            2,
        ),
        0x366 => (
            "v_mbcnt_hi_u32_b32".to_owned(),
            DstOperand::Vgpr(dst_raw),
            2,
        ),
        _ => (format!("vop3_0x{op:03x}"), DstOperand::Vgpr(dst_raw), 3),
    };

    let mut srcs = Vec::with_capacity(num_srcs);
    if num_srcs >= 1 {
        srcs.push(decode_src9(s0, literal));
    }
    if num_srcs >= 2 {
        srcs.push(decode_src9(s1, literal));
    }
    if num_srcs >= 3 {
        srcs.push(decode_src9(s2, literal));
    }

    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Vop3,
        opcode: op,
        mnemonic,
        dst,
        srcs,
        literal,
        simm16: None,
        branch_target: None,
        modifiers,
        ext: ValuExt::None,
        extra,
        is_terminator: false,
        raw_words,
    })
}

fn decode_vintrp(pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let vdst = ((w0 >> 18) & 0xFF) as u8;
    let op = ((w0 >> 16) & 0x3) as u8;
    let attr = ((w0 >> 10) & 0x3F) as u8;
    let chan = ((w0 >> 8) & 0x3) as u8;
    let vsrc = (w0 & 0xFF) as u8;
    let mnemonic = match op {
        0 => "v_interp_p1_f32",
        1 => "v_interp_p2_f32",
        2 => "v_interp_mov_f32",
        _ => "v_interp_unknown",
    }
    .to_owned();
    Ok(DecodedInst {
        pc,
        word_len: 1,
        encoding: Encoding::Vintrp,
        opcode: u16::from(op),
        mnemonic,
        dst: DstOperand::Vgpr(vdst),
        srcs: vec![SrcOperand::Vgpr(vsrc)],
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::Vintrp {
            op,
            vdst,
            attr,
            chan,
            vsrc,
        },
        is_terminator: false,
        raw_words: vec![w0],
    })
}

fn decode_exp(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let en = (w0 & 0xF) as u8;
    let target = ((w0 >> 4) & 0x3F) as u8;
    let compr = ((w0 >> 10) & 1) != 0;
    let done = ((w0 >> 11) & 1) != 0;
    let vm = ((w0 >> 12) & 1) != 0;
    let vsrc = [
        (w1 & 0xFF) as u8,
        ((w1 >> 8) & 0xFF) as u8,
        ((w1 >> 16) & 0xFF) as u8,
        ((w1 >> 24) & 0xFF) as u8,
    ];
    let target_name = match target {
        0..=7 => format!("mrt{target}"),
        8 => "mrtz".to_owned(),
        9 => "null".to_owned(),
        12..=16 => format!("pos{}", target - 12),
        20 => "prim".to_owned(),
        32..=63 => format!("param{}", target - 32),
        other => format!("target{other}"),
    };
    Ok(DecodedInst {
        pc,
        word_len: 2,
        encoding: Encoding::Export,
        opcode: u16::from(target),
        mnemonic: format!("exp_{target_name}"),
        dst: DstOperand::None,
        srcs: vsrc.into_iter().map(SrcOperand::Vgpr).collect(),
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::Exp {
            target,
            en,
            compr,
            done,
            vm,
            vsrc,
        },
        is_terminator: false,
        raw_words: vec![w0, w1],
    })
}

fn decode_mimg(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let op_lo = (w0 >> 18) & 0x7F;
    let op_hi = w0 & 1;
    let op = (op_lo | (op_hi << 7)) as u16;
    let nsa = ((w0 >> 1) & 0x3) as usize;
    let dim = ((w0 >> 3) & 0x7) as u8;
    let dmask = ((w0 >> 8) & 0xF) as u8;
    let unrm = ((w0 >> 12) & 1) != 0;
    let r128 = ((w0 >> 15) & 1) != 0;
    let tfe = ((w0 >> 16) & 1) != 0;
    let lwe = ((w0 >> 17) & 1) != 0;

    let vaddr = (w1 & 0xFF) as u8;
    let vdata = ((w1 >> 8) & 0xFF) as u8;
    let srsrc = (((w1 >> 16) & 0x1F) << 2) as u8;
    let ssamp = (((w1 >> 21) & 0x1F) << 2) as u8;
    let a16 = ((w1 >> 30) & 1) != 0;
    let d16 = ((w1 >> 31) & 1) != 0;

    let mut raw_words = vec![w0, w1];
    let mut addrs = [0u8; 13];
    addrs[0] = vaddr;
    let mut addr_count = 1u8;
    for n in 0..nsa {
        let nw = fetch_word(words, idx, 2 + n, pc)?;
        raw_words.push(nw);
        for b in 0..4 {
            if (addr_count as usize) < addrs.len() {
                addrs[addr_count as usize] = ((nw >> (b * 8)) & 0xFF) as u8;
                addr_count += 1;
            }
        }
    }

    let mnemonic = match op {
        0x00 => "image_load".to_owned(),
        0x01 => "image_load_mip".to_owned(),
        0x08 => "image_store".to_owned(),
        0x09 => "image_store_mip".to_owned(),
        0x20 => "image_sample".to_owned(),
        0x24 => "image_sample_l".to_owned(),
        _ => format!("mimg_0x{op:02x}"),
    };
    let is_store = matches!(op, 0x08 | 0x09);
    let dst = if is_store {
        DstOperand::None
    } else {
        DstOperand::Vgpr(vdata)
    };
    Ok(DecodedInst {
        pc,
        word_len: raw_words.len() as u8,
        encoding: Encoding::Mimg,
        opcode: op,
        mnemonic,
        dst,
        srcs: vec![
            SrcOperand::Vgpr(vaddr),
            SrcOperand::Sgpr(srsrc),
            SrcOperand::Sgpr(ssamp),
        ],
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::Mimg {
            op,
            dim,
            dmask,
            unrm,
            r128,
            tfe,
            lwe,
            a16,
            d16,
            vdata,
            srsrc,
            ssamp,
            addr_count,
            is_nsa: nsa > 0,
            addrs,
        },
        is_terminator: false,
        raw_words,
    })
}

fn decode_ds(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    // GFX10.3 verified via llvm-mc -mcpu=gfx1030: op = W0[25:18], gds = W0[17]
    let op = ((w0 >> 18) & 0xFF) as u16;
    let gds = ((w0 >> 17) & 1) != 0;
    let offset0 = (w0 & 0xFF) as u8;
    let offset1 = ((w0 >> 8) & 0xFF) as u8;
    let offset = (w0 & 0xFFFF) as u16;
    let addr = (w1 & 0xFF) as u8;
    let data0 = ((w1 >> 8) & 0xFF) as u8;
    let data1 = ((w1 >> 16) & 0xFF) as u8;
    let vdst = ((w1 >> 24) & 0xFF) as u8;

    let (mnemonic, dst, srcs, extra) = match op {
        0x0D => (
            "ds_write_b32".to_owned(),
            DstOperand::None,
            vec![SrcOperand::Vgpr(addr), SrcOperand::Vgpr(data0)],
            InstExtra::DsLds {
                addr,
                data0,
                data1,
                vdst,
                offset0,
                offset1,
                gds,
                is_write: true,
                paired: false,
            },
        ),
        0x0E => (
            "ds_write2_b32".to_owned(),
            DstOperand::None,
            vec![
                SrcOperand::Vgpr(addr),
                SrcOperand::Vgpr(data0),
                SrcOperand::Vgpr(data1),
            ],
            InstExtra::DsLds {
                addr,
                data0,
                data1,
                vdst,
                offset0,
                offset1,
                gds,
                is_write: true,
                paired: true,
            },
        ),
        0x35 => (
            "ds_swizzle_b32".to_owned(),
            DstOperand::Vgpr(vdst),
            vec![SrcOperand::Vgpr(addr)],
            InstExtra::DsSwizzle { offset, vsrc: addr },
        ),
        0x36 => (
            "ds_read_b32".to_owned(),
            DstOperand::Vgpr(vdst),
            vec![SrcOperand::Vgpr(addr)],
            InstExtra::DsLds {
                addr,
                data0,
                data1,
                vdst,
                offset0,
                offset1,
                gds,
                is_write: false,
                paired: false,
            },
        ),
        0x37 => (
            "ds_read2_b32".to_owned(),
            DstOperand::Vgpr(vdst),
            vec![SrcOperand::Vgpr(addr)],
            InstExtra::DsLds {
                addr,
                data0,
                data1,
                vdst,
                offset0,
                offset1,
                gds,
                is_write: false,
                paired: true,
            },
        ),
        _ => (
            format!("ds_0x{op:02x}"),
            DstOperand::None,
            Vec::new(),
            InstExtra::None,
        ),
    };
    Ok(DecodedInst {
        pc,
        word_len: 2,
        encoding: Encoding::Ds,
        opcode: op,
        mnemonic,
        dst,
        srcs,
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra,
        is_terminator: false,
        raw_words: vec![w0, w1],
    })
}

fn decode_smem(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let op = ((w0 >> 18) & 0xFF) as u16;
    let sdata = ((w0 >> 6) & 0x7F) as u8;
    let sbase = ((w0 & 0x3F) << 1) as u8;
    let soffset = ((w1 >> 25) & 0x7F) as u8;
    let ioffset_raw = w1 & 0x1F_FFFF;
    let ioffset = ((ioffset_raw << 11) as i32) >> 11;

    let (mnemonic, extra) = match op {
        0x00..=0x02 => {
            let dwords = match op {
                0x00 => 1,
                0x01 => 2,
                _ => 4,
            };
            let name = match op {
                0x00 => "s_load_dword",
                0x01 => "s_load_dwordx2",
                _ => "s_load_dwordx4",
            };
            (
                name.to_owned(),
                InstExtra::Smem {
                    sbase,
                    sdata,
                    soffset,
                    ioffset,
                    dwords,
                    is_buffer: false,
                },
            )
        }
        0x08..=0x0A => {
            let dwords = match op {
                0x08 => 1,
                0x09 => 2,
                _ => 4,
            };
            let name = match op {
                0x08 => "s_buffer_load_dword",
                0x09 => "s_buffer_load_dwordx2",
                _ => "s_buffer_load_dwordx4",
            };
            (
                name.to_owned(),
                InstExtra::Smem {
                    sbase,
                    sdata,
                    soffset,
                    ioffset,
                    dwords,
                    is_buffer: true,
                },
            )
        }
        _ => (format!("smem_0x{op:02x}"), InstExtra::None),
    };

    Ok(DecodedInst {
        pc,
        word_len: 2,
        encoding: Encoding::Smem,
        opcode: op,
        mnemonic,
        dst: decode_scalar_dst(sdata),
        srcs: vec![SrcOperand::Sgpr(sbase)],
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra,
        is_terminator: false,
        raw_words: vec![w0, w1],
    })
}

fn decode_mubuf(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let op_ext = (w0 >> 25) & 1;
    let op = ((w0 >> 18) & 0x7F) as u16;
    let lds = ((w0 >> 16) & 1) != 0;
    let idxen = ((w0 >> 13) & 1) != 0;
    let offen = ((w0 >> 12) & 1) != 0;
    let offset = (w0 & 0xFFF) as u16;
    let soffset = ((w1 >> 24) & 0xFF) as u8;
    let tfe = ((w1 >> 23) & 1) != 0;
    let srsrc = (((w1 >> 16) & 0x1F) << 2) as u8;
    let vdata = ((w1 >> 8) & 0xFF) as u8;
    let vaddr = (w1 & 0xFF) as u8;

    let (mnemonic, extra, dst) = if op_ext == 0 {
        // GFX10.3 verified: x1=0C/1C, x2=0D/1D, x4=0E/1E, x3=0F/1F
        let parsed = match op {
            0x0C => Some(("buffer_load_dword", 1u8, false)),
            0x0D => Some(("buffer_load_dwordx2", 2, false)),
            0x0E => Some(("buffer_load_dwordx4", 4, false)),
            0x0F => Some(("buffer_load_dwordx3", 3, false)),
            0x1C => Some(("buffer_store_dword", 1, true)),
            0x1D => Some(("buffer_store_dwordx2", 2, true)),
            0x1E => Some(("buffer_store_dwordx4", 4, true)),
            0x1F => Some(("buffer_store_dwordx3", 3, true)),
            _ => None,
        };
        if let Some((name, dwords, is_store)) = parsed {
            let dst = if is_store {
                DstOperand::None
            } else {
                DstOperand::Vgpr(vdata)
            };
            (
                name.to_owned(),
                InstExtra::Mubuf {
                    vaddr,
                    vdata,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    lds,
                    tfe,
                    dwords,
                    is_store,
                },
                dst,
            )
        } else {
            (
                format!("mubuf_0x{op:02x}"),
                InstExtra::None,
                DstOperand::None,
            )
        }
    } else {
        (
            format!("mubuf_ext_0x{op:02x}"),
            InstExtra::None,
            DstOperand::None,
        )
    };

    Ok(DecodedInst {
        pc,
        word_len: 2,
        encoding: Encoding::Vmem,
        opcode: op,
        mnemonic,
        dst,
        srcs: vec![SrcOperand::Vgpr(vaddr), SrcOperand::Sgpr(srsrc)],
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra,
        is_terminator: false,
        raw_words: vec![w0, w1],
    })
}

fn decode_flat(words: &[u32], idx: usize, pc: u32, w0: u32) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let op = ((w0 >> 18) & 0x7F) as u16;
    let seg = ((w0 >> 14) & 0x3) as u8;
    let lds = ((w0 >> 13) & 1) != 0;
    let raw_off = (w0 & 0xFFF) as u16;
    let offset = if seg == 2 {
        ((raw_off << 4) as i16) >> 4
    } else {
        (raw_off & 0x7FF) as i16
    };
    let vdst = ((w1 >> 24) & 0xFF) as u8;
    let saddr = ((w1 >> 16) & 0x7F) as u8;
    let data = ((w1 >> 8) & 0xFF) as u8;
    let addr = (w1 & 0xFF) as u8;

    let prefix = match seg {
        0 => "flat",
        1 => "scratch",
        2 => "global",
        _ => "flat_res",
    };

    let parsed = match op {
        0x0C => Some(("load_dword", 1u8, false)),
        0x0D => Some(("load_dwordx2", 2, false)),
        0x0E => Some(("load_dwordx4", 4, false)),
        0x0F => Some(("load_dwordx3", 3, false)),
        0x1C => Some(("store_dword", 1, true)),
        0x1D => Some(("store_dwordx2", 2, true)),
        0x1E => Some(("store_dwordx4", 4, true)),
        0x1F => Some(("store_dwordx3", 3, true)),
        _ => None,
    };

    let (mnemonic, extra, dst) = if let Some((stem, dwords, is_store)) = parsed {
        let dst = if is_store {
            DstOperand::None
        } else {
            DstOperand::Vgpr(vdst)
        };
        (
            format!("{prefix}_{stem}"),
            InstExtra::Flat {
                seg,
                addr,
                data,
                vdst,
                saddr,
                offset,
                lds,
                dwords,
                is_store,
            },
            dst,
        )
    } else {
        (
            format!("{prefix}_0x{op:02x}"),
            InstExtra::None,
            DstOperand::None,
        )
    };

    Ok(DecodedInst {
        pc,
        word_len: 2,
        encoding: Encoding::Vmem,
        opcode: op,
        mnemonic,
        dst,
        srcs: vec![SrcOperand::Vgpr(addr)],
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra,
        is_terminator: false,
        raw_words: vec![w0, w1],
    })
}

fn decode_two_word_generic(
    words: &[u32],
    idx: usize,
    pc: u32,
    w0: u32,
    encoding: Encoding,
    prefix: &str,
) -> Result<DecodedInst, DecodeError> {
    let w1 = fetch_word(words, idx, 1, pc)?;
    let op = ((w0 >> 18) & 0xFF) as u16;
    Ok(DecodedInst {
        pc,
        word_len: 2,
        encoding,
        opcode: op,
        mnemonic: format!("{prefix}_0x{op:02x}"),
        dst: DstOperand::None,
        srcs: Vec::new(),
        literal: None,
        simm16: None,
        branch_target: None,
        modifiers: ValuModifiers::default(),
        ext: ValuExt::None,
        extra: InstExtra::None,
        is_terminator: false,
        raw_words: vec![w0, w1],
    })
}

fn vop1_mnemonic(op: u16) -> String {
    match op {
        0x00 => "v_nop".into(),
        0x01 => "v_mov_b32".into(),
        0x02 => "v_readfirstlane_b32".into(),
        0x05 => "v_cvt_f32_i32".into(),
        0x06 => "v_cvt_f32_u32".into(),
        0x07 => "v_cvt_u32_f32".into(),
        0x08 => "v_cvt_i32_f32".into(),
        0x0A => "v_cvt_f16_f32".into(),
        0x0B => "v_cvt_f32_f16".into(),
        0x2A => "v_rcp_f32".into(),
        0x2E => "v_rsq_f32".into(),
        0x33 => "v_sqrt_f32".into(),
        0x37 => "v_not_b32".into(),
        _ => format!("vop1_0x{op:02x}"),
    }
}

fn vop2_mnemonic(op: u16) -> String {
    match op {
        0x01 => "v_cndmask_b32".into(),
        0x03 => "v_add_f32".into(),
        0x04 => "v_sub_f32".into(),
        0x05 => "v_subrev_f32".into(),
        0x06 => "v_fmac_legacy_f32".into(),
        0x07 => "v_mul_legacy_f32".into(),
        0x08 => "v_mul_f32".into(),
        0x0B => "v_mul_u32_u24".into(),
        0x0F => "v_min_f32".into(),
        0x10 => "v_max_f32".into(),
        0x16 => "v_lshrrev_b32".into(),
        0x18 => "v_ashrrev_i32".into(),
        0x1A => "v_lshlrev_b32".into(),
        0x1B => "v_and_b32".into(),
        0x1C => "v_or_b32".into(),
        0x1D => "v_xor_b32".into(),
        // Console gfx10.1 non-fused MAC/MAD family:
        0x1F => "v_mac_f32".into(),
        0x20 => "v_madmk_f32".into(),
        0x21 => "v_madak_f32".into(),
        0x25 => "v_add_nc_u32".into(),
        0x26 => "v_sub_nc_u32".into(),
        0x27 => "v_subrev_nc_u32".into(),
        0x2B => "v_fmac_f32".into(),
        0x2C => "v_fmamk_f32".into(),
        0x2D => "v_fmaak_f32".into(),
        0x2F => "v_cvt_pkrtz_f16_f32".into(),
        0x37 => "v_fmamk_f16".into(),
        0x38 => "v_fmaak_f16".into(),
        _ => format!("vop2_0x{op:02x}"),
    }
}

fn vopc_mnemonic(op: u16) -> String {
    const PREDS: [&str; 16] = [
        "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq",
        "nlt", "tru",
    ];
    match op {
        0x10..=0x1F => format!("v_cmp_{}_f32", PREDS[(op - 0x10) as usize]),
        0x80..=0x8F => format!("v_cmp_{}_i32", PREDS[(op - 0x80) as usize]),
        0xC0..=0xCF => format!("v_cmp_{}_u32", PREDS[(op - 0xC0) as usize]),
        _ => format!("vopc_0x{op:02x}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_sopp(op: u16, simm16: i16) -> u32 {
        (0x17Fu32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16 as u16)
    }

    fn enc_sop2(op: u16, sdst: u8, s0: u8, s1: u8) -> u32 {
        (0b10u32 << 30)
            | (u32::from(op & 0x7F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(s1) << 8)
            | u32::from(s0)
    }

    fn enc_vop2(op: u16, vdst: u8, vsrc1: u8, src0: u16) -> u32 {
        (u32::from(op & 0x3F) << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(vsrc1) << 9)
            | u32::from(src0 & 0x1FF)
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_vop3(
        op: u16,
        dst: u8,
        s0: u16,
        s1: u16,
        s2: u16,
        abs: u8,
        neg: u8,
        clamp: bool,
    ) -> [u32; 2] {
        let w0 = (0x35u32 << 26)
            | (u32::from(op & 0x3FF) << 16)
            | (u32::from(clamp) << 15)
            | (u32::from(abs & 0x7) << 8)
            | u32::from(dst);
        let w1 = (u32::from(neg & 0x7) << 29)
            | (u32::from(s2 & 0x1FF) << 18)
            | (u32::from(s1 & 0x1FF) << 9)
            | u32::from(s0 & 0x1FF);
        [w0, w1]
    }

    #[test]
    fn decodes_sopp_sop2_sopk_sopc_and_branches() {
        let words = [
            enc_sop2(0x00, 2, 0xFF, 1), // s_add_u32 s2, 0x12345678, s1
            0x1234_5678,
            (0xBu32 << 28) | (3u32 << 16) | 42, // s_movk_i32 s3, 42
            (0x17Eu32 << 23) | (0x06u32 << 16) | (3u32 << 8) | 2, // s_cmp_eq_u32 s2, s3
            enc_sopp(0x05, 1), // s_cbranch_scc1 +1 -> skips next 1 word (s_waitcnt)
            enc_sopp(0x0C, 0), // s_waitcnt 0
            enc_sopp(0x01, 0), // s_endpgm
        ];
        let insts = disassemble(&words).expect("disassemble scalar stream");
        assert_eq!(insts[0].mnemonic, "s_add_u32");
        assert_eq!(insts[0].word_len, 2);
        assert_eq!(insts[0].literal, Some(0x1234_5678));
        assert_eq!(insts[1].mnemonic, "s_movk_i32");
        assert_eq!(insts[2].mnemonic, "s_cmp_eq_u32");
        assert_eq!(insts[3].mnemonic, "s_cbranch_scc1");
        assert_eq!(insts[3].branch_target, Some(0x18));
        assert_eq!(insts[4].mnemonic, "s_waitcnt");
        assert_eq!(insts[5].mnemonic, "s_endpgm");
        assert!(insts[5].is_terminator);
    }

    #[test]
    fn decodes_vop2_vop3_legacy_mul_gfx101_leftovers_and_unused_src_literal() {
        let mad_w = enc_vop3(0x141, 4, 256, 257, 258, 0b001, 0b010, true); // v_mad_f32 v4, |v0|, -v1, v2 clamp
        let mac3_w = enc_vop3(0x11F, 5, 256, 257, 0, 0, 0, false); // v_mac_f32 v5, v0, v1
        let vnop_lit = enc_vop3(0x180, 0, 0, 0xFF, 0, 0, 0, false); // VOP3 v_nop with unused SRC1==255 -> 12 bytes!

        let words = [
            enc_vop2(0x07, 1, 2, 256), // v_mul_legacy_f32 v1, v0, v2
            enc_vop2(0x1F, 3, 2, 256), // v_mac_f32 v3, v0, v2
            enc_vop2(0x20, 3, 2, 256), // v_madmk_f32 v3, v0, K, v2
            0x3F80_0000,
            enc_vop2(0x21, 3, 2, 256), // v_madak_f32 v3, v0, v2, K
            0x4000_0000,
            mad_w[0],
            mad_w[1],
            mac3_w[0],
            mac3_w[1],
            vnop_lit[0],
            vnop_lit[1],
            0xDEAD_BEEF,
            enc_sopp(0x01, 0),
        ];

        let insts = disassemble(&words).expect("disassemble vector + gfx10.1 stream");
        assert_eq!(insts[0].mnemonic, "v_mul_legacy_f32");
        assert_eq!(insts[1].mnemonic, "v_mac_f32");
        assert_eq!(insts[2].mnemonic, "v_madmk_f32");
        assert_eq!(insts[2].srcs[1], SrcOperand::Literal(0x3F80_0000));
        assert_eq!(insts[3].mnemonic, "v_madak_f32");
        assert_eq!(insts[3].srcs[2], SrcOperand::Literal(0x4000_0000));
        assert_eq!(insts[4].mnemonic, "v_mad_f32");
        assert!(insts[4].modifiers.abs[0]);
        assert!(insts[4].modifiers.neg[1]);
        assert!(insts[4].modifiers.clamp);
        assert_eq!(insts[5].mnemonic, "v_mac_f32");
        assert_eq!(insts[5].encoding, Encoding::Vop3);
        assert_eq!(insts[6].mnemonic, "v_nop");
        assert_eq!(
            insts[6].word_len, 3,
            "VOP3 unused SRC==255 must consume 12 bytes"
        );
        assert_eq!(insts[7].mnemonic, "s_endpgm");
    }

    #[test]
    fn reachable_walk_handles_multiple_endpgm_and_skips_trailing_pool() {
        let words = [
            enc_sopp(0x05, 1), // pc=0: s_cbranch_scc1 -> pc=8
            enc_sopp(0x01, 0), // pc=4: s_endpgm (early exit)
            enc_sopp(0x00, 0), // pc=8: s_nop (reachable past first s_endpgm!)
            enc_sopp(0x01, 0), // pc=12: s_endpgm
            2,                 // pc=16: unreachable constant pool
            0xFFFF_FFFF,       // would truncate if decoded as a multi-word instruction
            0x3F80_0000,
            enc_sopp(0x01, 0),
        ];
        let insts = disassemble_reachable(&words).expect("reachable decode");
        assert_eq!(insts.len(), 4);
        assert_eq!(insts[2].pc, 8);
        assert_eq!(insts[3].pc, 12);
    }

    #[test]
    fn rejects_truncated_literal_vop3_and_unaligned_bytes() {
        assert!(matches!(
            disassemble_bytes(&[0, 1, 2]),
            Err(DecodeError::UnalignedBytes(3))
        ));
        // VOP2 with SRC0=255 missing literal DWORD
        let trunc_lit = [enc_vop2(0x03, 0, 1, 0xFF)];
        assert!(matches!(
            disassemble(&trunc_lit),
            Err(DecodeError::Truncated { .. })
        ));
        // VOP3 missing second DWORD
        let trunc_vop3 = [(0x35u32 << 26) | (0x14Bu32 << 16)];
        assert!(matches!(
            disassemble(&trunc_vop3),
            Err(DecodeError::Truncated { .. })
        ));
    }

    #[test]
    fn roundtrips_mesh_sendmsg_exp_prim_and_mbcnt() {
        use crate::encode as asm;
        let send = asm::s_sendmsg_gs_alloc_req();
        assert_eq!(send, 0xBF90_0009);
        let prim = asm::exp_prim(9, true);
        assert_eq!(prim, [0xF800_0941, 0x0000_0009]);
        let mbcnt = asm::v_mbcnt_lo_u32_b32(0, asm::inline_i(-1), asm::inline_i(0));
        assert_eq!(mbcnt, [0xD765_0000, 0x0001_00C1]);

        let words = [send, prim[0], prim[1], mbcnt[0], mbcnt[1], asm::s_endpgm()];
        let insts = disassemble(&words).expect("disassemble mesh instructions");
        assert_eq!(insts[0].mnemonic, "s_sendmsg");
        assert_eq!(insts[0].simm16, Some(9));
        assert_eq!(insts[1].mnemonic, "exp_prim");
        assert_eq!(
            insts[1].extra,
            InstExtra::Exp {
                target: 20,
                en: 1,
                compr: false,
                done: true,
                vm: false,
                vsrc: [9, 0, 0, 0],
            }
        );
        assert_eq!(insts[2].mnemonic, "v_mbcnt_lo_u32_b32");
        assert_eq!(insts[2].dst, DstOperand::Vgpr(0));
        assert_eq!(
            insts[2].srcs,
            vec![SrcOperand::InlineInt(-1), SrcOperand::InlineInt(0)]
        );
    }

    #[test]
    fn roundtrips_gfx103_prefetch_clause_waitcnt_f16_and_bit_ops() {
        use crate::encode as asm;
        assert_eq!(asm::s_inst_prefetch(3), 0xBFA0_0003);
        assert_eq!(asm::s_clause(1), 0xBFA1_0001);
        assert_eq!(asm::s_waitcnt_depctr(0), 0xBFA3_0000);
        assert_eq!(asm::s_waitcnt_vscnt(asm::NULL, 3), 0xBBFD_0003);
        assert_eq!(asm::s_waitcnt_vmcnt(asm::NULL, 2), 0xBC7D_0002);
        assert_eq!(asm::s_waitcnt_expcnt(asm::NULL, 1), 0xBCFD_0001);
        assert_eq!(asm::s_waitcnt_lgkmcnt(asm::NULL, 0), 0xBD7D_0000);
        assert_eq!(asm::v_cvt_pkrtz_f16_f32(5, asm::v(6), 7), 0x5E0A_0F06);
        assert_eq!(asm::v_cvt_f16_f32(5, asm::v(6)), 0x7E0A_1506);
        assert_eq!(asm::v_cvt_f32_f16(5, asm::v(6)), 0x7E0A_1706);
        assert_eq!(asm::v_mul_u32_u24(5, asm::v(6), 7), 0x160A_0F06);
        assert_eq!(
            asm::v_mad_u32_u24(5, asm::v(6), asm::v(7), asm::v(8)),
            [0xD543_0005, 0x0422_0F06]
        );
        assert_eq!(
            asm::v_bfe_i32(5, asm::v(6), asm::v(7), asm::v(8)),
            [0xD549_0005, 0x0422_0F06]
        );
        assert_eq!(
            asm::v_lshl_add_u32(5, asm::v(6), asm::v(7), asm::v(8)),
            [0xD746_0005, 0x0422_0F06]
        );
        assert_eq!(
            asm::v_add3_u32(5, asm::v(6), asm::v(7), asm::v(8)),
            [0xD76D_0005, 0x0422_0F06]
        );
        assert_eq!(
            asm::v_lshl_or_b32(5, asm::v(6), asm::v(7), asm::v(8)),
            [0xD76F_0005, 0x0422_0F06]
        );
        assert_eq!(
            asm::v_and_or_b32(5, asm::v(6), asm::v(7), asm::v(8)),
            [0xD771_0005, 0x0422_0F06]
        );

        let mut a = asm::Asm::new();
        a.push(asm::s_inst_prefetch(3))
            .push(asm::s_clause(1))
            .push(asm::s_waitcnt_depctr(0))
            .push(asm::s_waitcnt_vscnt(asm::NULL, 3))
            .push(asm::s_waitcnt_vmcnt(asm::NULL, 2))
            .push(asm::s_waitcnt_expcnt(asm::NULL, 1))
            .push(asm::s_waitcnt_lgkmcnt(asm::NULL, 0))
            .push(asm::v_cvt_pkrtz_f16_f32(5, asm::v(6), 7))
            .push(asm::v_cvt_f16_f32(5, asm::v(6)))
            .push(asm::v_cvt_f32_f16(5, asm::v(6)))
            .push(asm::v_mul_u32_u24(5, asm::v(6), 7))
            .push(asm::v_lshrrev_b32(0, asm::v(1), 2))
            .push(asm::v_ashrrev_i32(0, asm::v(1), 2))
            .push(asm::v_lshlrev_b32(0, asm::v(1), 2))
            .push2(asm::v_mad_u32_u24(5, asm::v(6), asm::v(7), asm::v(8)))
            .push2(asm::v_bfe_i32(5, asm::v(6), asm::v(7), asm::v(8)))
            .push2(asm::v_lshl_add_u32(5, asm::v(6), asm::v(7), asm::v(8)))
            .push2(asm::v_add3_u32(5, asm::v(6), asm::v(7), asm::v(8)))
            .push2(asm::v_lshl_or_b32(5, asm::v(6), asm::v(7), asm::v(8)))
            .push2(asm::v_and_or_b32(5, asm::v(6), asm::v(7), asm::v(8)))
            .push(asm::s_endpgm());
        let insts = disassemble(a.words()).expect("disassemble gfx10.3 stream");
        let names: Vec<&str> = insts.iter().map(|i| i.mnemonic.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "s_inst_prefetch",
                "s_clause",
                "s_waitcnt_depctr",
                "s_waitcnt_vscnt",
                "s_waitcnt_vmcnt",
                "s_waitcnt_expcnt",
                "s_waitcnt_lgkmcnt",
                "v_cvt_pkrtz_f16_f32",
                "v_cvt_f16_f32",
                "v_cvt_f32_f16",
                "v_mul_u32_u24",
                "v_lshrrev_b32",
                "v_ashrrev_i32",
                "v_lshlrev_b32",
                "v_mad_u32_u24",
                "v_bfe_i32",
                "v_lshl_add_u32",
                "v_add3_u32",
                "v_lshl_or_b32",
                "v_and_or_b32",
                "s_endpgm",
            ]
        );
        assert_eq!(insts[3].dst, DstOperand::None);
        assert_eq!(insts[3].srcs[0], SrcOperand::Null);
    }
}
