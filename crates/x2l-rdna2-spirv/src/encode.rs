// Clean-room RDNA2 (GFX10.3) + console GFX10.1 instruction encoder for test assembly.
// Encodings verified against AMD RDNA2 ISA documentation and LLVM 22.1.8 AMDGPU (`gfx1030`).
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Minimal RDNA2 (GFX10.3) instruction assembler for unit and Vulkan GPU instruction tests.

/// Standard scalar operand selector constants (0..105 = SGPR, 256..511 = VGPR).
pub const VCC_LO: u16 = 106;
pub const VCC_HI: u16 = 107;
pub const M0: u16 = 124;
pub const NULL: u16 = 125;
pub const EXEC_LO: u16 = 126;
pub const EXEC_HI: u16 = 127;
pub const SRC_VCCZ: u16 = 251;
pub const SRC_EXECZ: u16 = 252;
pub const SRC_SCC: u16 = 253;
pub const SRC_LIT: u16 = 255;
pub const SRC_DPP8: u16 = 0xE9;
pub const SRC_DPP8_FI: u16 = 0xEA;
pub const SRC_SDWA: u16 = 0xF9;
pub const SRC_DPP16: u16 = 0xFA;

/// Scalar register selector (`s0..s105`).
pub const fn s(n: u8) -> u16 {
    n as u16
}

/// Vector register 9-bit source selector (`v0..v255` -> `256..511`).
pub const fn v(n: u8) -> u16 {
    256 + n as u16
}

/// Inline integer constant selector (`-16..=64`).
pub const fn inline_i(val: i32) -> u16 {
    if val >= 0 && val <= 64 {
        (128 + val) as u16
    } else if val >= -16 && val < 0 {
        (192 - val) as u16
    } else {
        panic!("inline_i value out of RDNA2 inline range -16..=64")
    }
}

/// Inline float constant selectors.
pub const F_0_5: u16 = 240;
pub const F_NEG_0_5: u16 = 241;
pub const F_1_0: u16 = 242;
pub const F_NEG_1_0: u16 = 243;
pub const F_2_0: u16 = 244;
pub const F_NEG_2_0: u16 = 245;
pub const F_4_0: u16 = 246;
pub const F_NEG_4_0: u16 = 247;

/// Construct a clean-room AMD RDNA2 (GFX10.3) 128-bit V# buffer resource descriptor.
///
/// Layout per AMD RDNA2 ISA Table 37:
/// - `rsrc0`: `base_addr[31:0]`
/// - `rsrc1`: `base_addr[47:32] | (stride << 16)`
/// - `rsrc2`: `num_records`
/// - `rsrc3`: `dst_sel (xyzw = 4,5,6,7) | (oob_select << 28) | (add_tid << 23) | (1 << 24)`
pub const fn make_vsharp(
    base_addr: u64,
    stride: u16,
    num_records: u32,
    oob_select: u8,
    add_tid: bool,
) -> [u32; 4] {
    let rsrc0 = base_addr as u32;
    let rsrc1 = ((base_addr >> 32) as u32 & 0xFFFF) | ((stride as u32 & 0x3FFF) << 16);
    let rsrc2 = num_records;
    // Standard identity channel swizzle X=R(4), Y=G(5), Z=B(6), W=A(7), resource_level=1
    let dst_sel = 4u32 | (5u32 << 3) | (6u32 << 6) | (7u32 << 9);
    let rsrc3 = dst_sel
        | (if add_tid { 1u32 << 23 } else { 0 })
        | (1u32 << 24)
        | (((oob_select as u32) & 0x3) << 28);
    [rsrc0, rsrc1, rsrc2, rsrc3]
}

/// Ergonomic RDNA2 instruction stream builder for tests.
#[derive(Debug, Default, Clone)]
pub struct Asm {
    words: Vec<u32>,
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn into_words(self) -> Vec<u32> {
        self.words
    }

    /// Current byte PC offset (next instruction address).
    pub fn pc(&self) -> u32 {
        (self.words.len() * 4) as u32
    }

    pub fn push(&mut self, w: u32) -> &mut Self {
        self.words.push(w);
        self
    }

    pub fn push2(&mut self, ws: [u32; 2]) -> &mut Self {
        self.words.extend_from_slice(&ws);
        self
    }

    pub fn push_lit(&mut self, w: u32, lit: u32) -> &mut Self {
        self.words.push(w);
        self.words.push(lit);
        self
    }

    pub fn push2_lit(&mut self, ws: [u32; 2], lit: u32) -> &mut Self {
        self.words.extend_from_slice(&ws);
        self.words.push(lit);
        self
    }

    /// Reserve a SOPP branch placeholder and return its word index for `patch_branch`.
    pub fn branch_placeholder(&mut self, op: u16) -> usize {
        let idx = self.words.len();
        self.words.push(sopp(op, 0));
        idx
    }

    /// Patch a previously reserved SOPP branch to target the current PC.
    pub fn patch_branch_here(&mut self, branch_word_idx: usize) {
        let target_words = self.words.len() as isize;
        let delta = target_words - (branch_word_idx as isize + 1);
        let simm16 = i16::try_from(delta).expect("branch delta fits i16");
        self.words[branch_word_idx] =
            (self.words[branch_word_idx] & 0xFFFF_0000) | u32::from(simm16 as u16);
    }

    /// Emit a backward/forward SOPP branch targeting a known byte PC.
    pub fn branch_to_pc(&mut self, op: u16, target_pc: u32) -> &mut Self {
        let cur_words = self.words.len() as isize;
        let target_words = (target_pc / 4) as isize;
        let delta = target_words - (cur_words + 1);
        let simm16 = i16::try_from(delta).expect("branch delta fits i16");
        self.push(sopp(op, simm16))
    }
}

// ---------------------------------------------------------------------------
// Scalar encoders: SOPP, SOP1, SOP2, SOPK, SOPC
// ---------------------------------------------------------------------------

pub const fn sopp(op: u16, simm16: i16) -> u32 {
    (0x17Fu32 << 23) | (((op as u32) & 0x7F) << 16) | ((simm16 as u16) as u32)
}

pub const fn s_nop(n: u16) -> u32 {
    sopp(0x00, n as i16)
}
pub const fn s_endpgm() -> u32 {
    sopp(0x01, 0)
}
pub const fn s_branch(simm16: i16) -> u32 {
    sopp(0x02, simm16)
}
pub const fn s_cbranch_scc0(simm16: i16) -> u32 {
    sopp(0x04, simm16)
}
pub const fn s_cbranch_scc1(simm16: i16) -> u32 {
    sopp(0x05, simm16)
}
pub const fn s_cbranch_vccz(simm16: i16) -> u32 {
    sopp(0x06, simm16)
}
pub const fn s_cbranch_vccnz(simm16: i16) -> u32 {
    sopp(0x07, simm16)
}
pub const fn s_cbranch_execz(simm16: i16) -> u32 {
    sopp(0x08, simm16)
}
pub const fn s_cbranch_execnz(simm16: i16) -> u32 {
    sopp(0x09, simm16)
}
pub const fn s_barrier() -> u32 {
    sopp(0x0A, 0)
}
pub const fn s_waitcnt(simm16: i16) -> u32 {
    sopp(0x0C, simm16)
}
/// AMD RDNA2 `MSG_GS_ALLOC_REQ` message ID (`SIMM16 = 0x0009`, bytes `09 00 90 bf`).
pub const MSG_GS_ALLOC_REQ: i16 = 9;
pub const fn s_sendmsg(simm16: i16) -> u32 {
    sopp(0x10, simm16)
}
pub const fn s_sendmsg_gs_alloc_req() -> u32 {
    s_sendmsg(MSG_GS_ALLOC_REQ)
}
pub const fn s_inst_prefetch(simm16: i16) -> u32 {
    sopp(0x20, simm16)
}
pub const fn s_clause(simm16: i16) -> u32 {
    sopp(0x21, simm16)
}
pub const fn s_waitcnt_depctr(simm16: i16) -> u32 {
    sopp(0x23, simm16)
}

pub const fn sop1(op: u16, sdst: u16, s0: u16) -> u32 {
    (0x17Du32 << 23)
        | (((sdst as u32) & 0x7F) << 16)
        | (((op as u32) & 0xFF) << 8)
        | ((s0 as u32) & 0xFF)
}

pub const fn s_mov_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x03, sdst, s0)
}
pub const fn s_mov_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x04, sdst, s0)
}
pub const fn s_cmov_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x05, sdst, s0)
}
pub const fn s_ff1_i32_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x14, sdst, s0)
}
pub const fn s_not_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x07, sdst, s0)
}
pub const fn s_not_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x08, sdst, s0)
}
pub const fn s_swappc_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x21, sdst, s0)
}
pub const fn s_and_saveexec_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x24, sdst, s0)
}
pub const fn s_or_saveexec_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x25, sdst, s0)
}
pub const fn s_xor_saveexec_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x26, sdst, s0)
}
pub const fn s_andn2_saveexec_b64(sdst: u16, s0: u16) -> u32 {
    sop1(0x27, sdst, s0)
}
pub const fn s_and_saveexec_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x3C, sdst, s0)
}
pub const fn s_or_saveexec_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x3D, sdst, s0)
}
pub const fn s_xor_saveexec_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x3E, sdst, s0)
}
pub const fn s_andn2_saveexec_b32(sdst: u16, s0: u16) -> u32 {
    sop1(0x3F, sdst, s0)
}

pub const fn sop2(op: u16, sdst: u16, s0: u16, s1: u16) -> u32 {
    (0b10u32 << 30)
        | (((op as u32) & 0x7F) << 23)
        | (((sdst as u32) & 0x7F) << 16)
        | (((s1 as u32) & 0xFF) << 8)
        | ((s0 as u32) & 0xFF)
}

pub const fn s_add_u32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x00, sdst, s0, s1)
}
pub const fn s_sub_u32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x01, sdst, s0, s1)
}
pub const fn s_add_i32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x02, sdst, s0, s1)
}
pub const fn s_sub_i32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x03, sdst, s0, s1)
}
pub const fn s_cselect_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x0A, sdst, s0, s1)
}
pub const fn s_cselect_b64(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x0B, sdst, s0, s1)
}
pub const fn s_and_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x0E, sdst, s0, s1)
}
pub const fn s_and_b64(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x0F, sdst, s0, s1)
}
pub const fn s_or_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x10, sdst, s0, s1)
}
pub const fn s_or_b64(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x11, sdst, s0, s1)
}
pub const fn s_xor_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x12, sdst, s0, s1)
}
pub const fn s_xor_b64(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x13, sdst, s0, s1)
}
pub const fn s_andn2_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x14, sdst, s0, s1)
}
pub const fn s_andn2_b64(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x15, sdst, s0, s1)
}
pub const fn s_lshl_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x1C, sdst, s0, s1)
}
pub const fn s_lshr_b32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x1E, sdst, s0, s1)
}
pub const fn s_mul_i32(sdst: u16, s0: u16, s1: u16) -> u32 {
    sop2(0x2C, sdst, s0, s1)
}

pub const fn sopk(op: u16, sdst: u16, simm16: i16) -> u32 {
    (0xBu32 << 28)
        | (((op as u32) & 0x1F) << 23)
        | (((sdst as u32) & 0x7F) << 16)
        | ((simm16 as u16) as u32)
}

pub const fn s_movk_i32(sdst: u16, simm16: i16) -> u32 {
    sopk(0x00, sdst, simm16)
}
pub const fn s_addk_i32(sdst: u16, simm16: i16) -> u32 {
    sopk(0x0F, sdst, simm16)
}
pub const fn s_mulk_i32(sdst: u16, simm16: i16) -> u32 {
    sopk(0x10, sdst, simm16)
}
pub const fn s_waitcnt_vscnt(ssrc: u16, simm16: i16) -> u32 {
    sopk(0x17, ssrc, simm16)
}
pub const fn s_waitcnt_vmcnt(ssrc: u16, simm16: i16) -> u32 {
    sopk(0x18, ssrc, simm16)
}
pub const fn s_waitcnt_expcnt(ssrc: u16, simm16: i16) -> u32 {
    sopk(0x19, ssrc, simm16)
}
pub const fn s_waitcnt_lgkmcnt(ssrc: u16, simm16: i16) -> u32 {
    sopk(0x1A, ssrc, simm16)
}

pub const fn sopc(op: u16, s0: u16, s1: u16) -> u32 {
    (0x17Eu32 << 23)
        | (((op as u32) & 0x7F) << 16)
        | (((s1 as u32) & 0xFF) << 8)
        | ((s0 as u32) & 0xFF)
}

pub const fn s_cmp_eq_u32(s0: u16, s1: u16) -> u32 {
    sopc(0x06, s0, s1)
}
pub const fn s_cmp_lg_u32(s0: u16, s1: u16) -> u32 {
    sopc(0x07, s0, s1)
}
pub const fn s_cmp_gt_u32(s0: u16, s1: u16) -> u32 {
    sopc(0x08, s0, s1)
}
pub const fn s_cmp_lt_u32(s0: u16, s1: u16) -> u32 {
    sopc(0x0A, s0, s1)
}
pub const fn s_cmp_lt_i32(s0: u16, s1: u16) -> u32 {
    sopc(0x04, s0, s1)
}

// ---------------------------------------------------------------------------
// Vector ALU encoders: VOP1, VOP2, VOPC, VOP3, DPP16, DPP8
// ---------------------------------------------------------------------------

pub const fn vop1(op: u16, dst: u8, src0: u16) -> u32 {
    (0x3Fu32 << 25) | ((dst as u32) << 17) | (((op as u32) & 0xFF) << 9) | ((src0 as u32) & 0x1FF)
}

pub const fn v_nop() -> u32 {
    vop1(0x00, 0, 0)
}
pub const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    vop1(0x01, vdst, src0)
}
/// `v_readfirstlane_b32 sdst, vsrc0` (VOP1 `0x02`, scalar destination).
pub const fn v_readfirstlane_b32(sdst: u8, vsrc0: u8) -> u32 {
    vop1(0x02, sdst & 0x7F, v(vsrc0))
}
pub const fn v_cvt_f32_i32(vdst: u8, src0: u16) -> u32 {
    vop1(0x05, vdst, src0)
}
pub const fn v_cvt_f32_u32(vdst: u8, src0: u16) -> u32 {
    vop1(0x06, vdst, src0)
}
pub const fn v_cvt_u32_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x07, vdst, src0)
}
pub const fn v_cvt_i32_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x08, vdst, src0)
}
pub const fn v_cvt_f16_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x0A, vdst, src0)
}
pub const fn v_cvt_f32_f16(vdst: u8, src0: u16) -> u32 {
    vop1(0x0B, vdst, src0)
}
pub const fn v_fract_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x20, vdst, src0)
}
pub const fn v_trunc_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x21, vdst, src0)
}
pub const fn v_ceil_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x22, vdst, src0)
}
pub const fn v_rndne_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x23, vdst, src0)
}
pub const fn v_floor_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x24, vdst, src0)
}
pub const fn v_exp_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x25, vdst, src0)
}
pub const fn v_log_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x27, vdst, src0)
}
pub const fn v_rcp_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x2A, vdst, src0)
}
pub const fn v_rsq_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x2E, vdst, src0)
}
pub const fn v_sqrt_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x33, vdst, src0)
}
pub const fn v_sin_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x35, vdst, src0)
}
pub const fn v_cos_f32(vdst: u8, src0: u16) -> u32 {
    vop1(0x36, vdst, src0)
}
pub const fn v_not_b32(vdst: u8, src0: u16) -> u32 {
    vop1(0x37, vdst, src0)
}

pub const fn vop2(op: u16, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (((op as u32) & 0x3F) << 25)
        | ((vdst as u32) << 17)
        | ((vsrc1 as u32) << 9)
        | ((src0 as u32) & 0x1FF)
}

pub const fn v_cndmask_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x01, vdst, src0, vsrc1)
}
pub const fn v_add_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x03, vdst, src0, vsrc1)
}
pub const fn v_sub_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x04, vdst, src0, vsrc1)
}
pub const fn v_mul_legacy_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x07, vdst, src0, vsrc1)
}
pub const fn v_mul_f16(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x35, vdst, src0, vsrc1)
}
pub const fn v_mul_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x08, vdst, src0, vsrc1)
}
pub const fn v_mul_u32_u24(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x0B, vdst, src0, vsrc1)
}
pub const fn v_min_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x0F, vdst, src0, vsrc1)
}
pub const fn v_max_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x10, vdst, src0, vsrc1)
}
pub const fn v_lshrrev_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x16, vdst, src0, vsrc1)
}
pub const fn v_ashrrev_i32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x18, vdst, src0, vsrc1)
}
pub const fn v_lshlrev_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x1A, vdst, src0, vsrc1)
}
pub const fn v_and_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x1B, vdst, src0, vsrc1)
}
pub const fn v_or_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x1C, vdst, src0, vsrc1)
}
pub const fn v_xor_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x1D, vdst, src0, vsrc1)
}
pub const fn v_mac_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x1F, vdst, src0, vsrc1)
}
pub const fn v_add_nc_u32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x25, vdst, src0, vsrc1)
}
pub const fn v_sub_nc_u32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x26, vdst, src0, vsrc1)
}
pub const fn v_fmac_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x2B, vdst, src0, vsrc1)
}
pub const fn v_add_co_ci_u32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x28, vdst, src0, vsrc1)
}
pub const fn v_mul_hi_u32_u24(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x0C, vdst, src0, vsrc1)
}
pub const fn v_bfrev_b32(vdst: u8, src0: u16) -> u32 {
    vop1(0x38, vdst, src0)
}
pub const fn v_cvt_pkrtz_f16_f32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    vop2(0x2F, vdst, src0, vsrc1)
}

pub const fn vopc(op: u16, src0: u16, vsrc1: u8) -> u32 {
    (0x3Eu32 << 25) | (((op as u32) & 0xFF) << 17) | ((vsrc1 as u32) << 9) | ((src0 as u32) & 0x1FF)
}

pub const fn v_cmp_lt_f32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0x01, src0, vsrc1)
}
pub const fn v_cmp_eq_f32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0x02, src0, vsrc1)
}
pub const fn v_cmp_gt_f32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0x04, src0, vsrc1)
}
pub const fn v_cmpx_gt_f32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0x14, src0, vsrc1)
}
pub const fn v_cmpx_eq_i32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0x92, src0, vsrc1)
}
pub const fn v_cmpx_eq_u32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0xD2, src0, vsrc1)
}
pub const fn v_cmp_lt_u32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0xC1, src0, vsrc1)
}
pub const fn v_cmp_eq_u32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0xC2, src0, vsrc1)
}
pub const fn v_cmp_ge_u32(src0: u16, vsrc1: u8) -> u32 {
    vopc(0xC6, src0, vsrc1)
}

#[allow(clippy::too_many_arguments)]
pub const fn vop3(
    op: u16,
    dst: u8,
    s0: u16,
    s1: u16,
    s2: u16,
    abs_bits: u8,
    neg_bits: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    let w0 = (0x35u32 << 26)
        | (((op as u32) & 0x3FF) << 16)
        | ((clamp as u32) << 15)
        | (((abs_bits as u32) & 0x7) << 8)
        | (dst as u32);
    let w1 = (((neg_bits as u32) & 0x7) << 29)
        | (((omod as u32) & 0x3) << 27)
        | (((s2 as u32) & 0x1FF) << 18)
        | (((s1 as u32) & 0x1FF) << 9)
        | ((s0 as u32) & 0x1FF);
    [w0, w1]
}

pub const fn v_add_co_u32(vdst: u8, sdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    vop3b_add(0x30F, vdst, sdst, src0, src1, 0)
}
pub const fn v_add_co_ci_u32_e64(vdst: u8, sdst: u8, src0: u16, src1: u16, mask: u16) -> [u32; 2] {
    vop3b_add(0x128, vdst, sdst, src0, src1, mask)
}
const fn vop3b_add(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, mask: u16) -> [u32; 2] {
    let [w0, w1] = vop3(op, vdst, src0, src1, mask, 0, 0, 0, false);
    [w0 | (((sdst & 0x7f) as u32) << 8), w1]
}
pub const fn v_add_lshl_u32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x347, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_min3_f32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x151, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_max3_f32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x154, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_med3_f32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x157, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_mad_f32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x141, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_mad_u32_u24(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x143, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_bfe_u32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x148, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_bfe_i32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x149, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_bfi_b32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x14A, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_fma_f32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x14B, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_mul_lo_u32(vdst: u8, s0: u16, s1: u16) -> [u32; 2] {
    vop3(0x169, vdst, s0, s1, 0, 0, 0, 0, false)
}
pub const fn v_lshl_add_u32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x346, vdst, s0, s1, s2, 0, 0, 0, false)
}
/// `v_mbcnt_lo_u32_b32 vdst, s0, s1` (VOP3 `0x365`, LLVM gfx1030 verified `d7650000 000100c1`).
pub const fn v_mbcnt_lo_u32_b32(vdst: u8, s0: u16, s1: u16) -> [u32; 2] {
    vop3(0x365, vdst, s0, s1, 0, 0, 0, 0, false)
}
pub const fn v_mbcnt_hi_u32_b32(vdst: u8, s0: u16, s1: u16) -> [u32; 2] {
    vop3(0x366, vdst, s0, s1, 0, 0, 0, 0, false)
}
pub const fn v_add3_u32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x36D, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_lshl_or_b32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x36F, vdst, s0, s1, s2, 0, 0, 0, false)
}
pub const fn v_and_or_b32(vdst: u8, s0: u16, s1: u16, s2: u16) -> [u32; 2] {
    vop3(0x371, vdst, s0, s1, s2, 0, 0, 0, false)
}

// ---------------------------------------------------------------------------
// Cross-lane encoders: readlane, writelane, permlane16, ds_swizzle, DPP16/8
// ---------------------------------------------------------------------------

/// `v_readlane_b32 sdst, vsrc0, ssrc1` (VOP3A `0x360`).
pub const fn v_readlane_b32(sdst: u8, vsrc0: u8, lane_src1: u16) -> [u32; 2] {
    vop3(0x360, sdst & 0x7F, v(vsrc0), lane_src1, 0, 0, 0, 0, false)
}

/// `v_writelane_b32 vdst, ssrc0, ssrc1` (VOP3A `0x361`).
pub const fn v_writelane_b32(vdst: u8, data_src0: u16, lane_src1: u16) -> [u32; 2] {
    vop3(0x361, vdst, data_src0, lane_src1, 0, 0, 0, 0, false)
}

/// `v_permlane16_b32` (`0x377`) or `v_permlanex16_b32` (`0x378`).
#[allow(clippy::too_many_arguments)]
pub const fn v_permlane16(
    cross: bool,
    vdst: u8,
    vsrc0: u8,
    src1: u16,
    src2: u16,
    fi: bool,
    bc: bool,
) -> [u32; 2] {
    let op = if cross { 0x378 } else { 0x377 };
    let w0 = (0x35u32 << 26)
        | ((op as u32) << 16)
        | ((bc as u32) << 12)
        | ((fi as u32) << 11)
        | (vdst as u32);
    let w1 = (((src2 as u32) & 0x1FF) << 18)
        | (((src1 as u32) & 0x1FF) << 9)
        | (v(vsrc0) as u32 & 0x1FF);
    [w0, w1]
}

/// `ds_swizzle_b32 vdst, vsrc offset:swizzle_offset` (DS `0x35`, GFX10.3 `op[25:18]`).
pub const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    let w0 = (0x36u32 << 26) | (0x35u32 << 18) | (offset as u32);
    let w1 = ((vdst as u32) << 24) | (vsrc as u32);
    [w0, w1]
}

pub const fn ds_add_u32(addr: u8, data: u8, offset: u16) -> [u32; 2] {
    [
        (0x36u32 << 26) | offset as u32,
        ((data as u32) << 8) | addr as u32,
    ]
}
pub const fn ds_add_rtn_u32(vdst: u8, addr: u8, data: u8, offset: u16) -> [u32; 2] {
    let [w0, w1] = ds_add_u32(addr, data, offset);
    [w0 | (0x20 << 18), w1 | ((vdst as u32) << 24)]
}

/// `ds_write_b32 vaddr, vdata0 offset:offset16` (DS `0x0D`, GFX10.3 `op[25:18]`).
pub const fn ds_write_b32(addr: u8, data0: u8, offset: u16) -> [u32; 2] {
    let w0 = (0x36u32 << 26) | (0x0Du32 << 18) | (offset as u32);
    let w1 = ((data0 as u32) << 8) | (addr as u32);
    [w0, w1]
}

/// `ds_write2_b32 vaddr, vdata0, vdata1 offset0:off0 offset1:off1` (DS `0x0E`, scaled by 4 bytes).
pub const fn ds_write2_b32(addr: u8, data0: u8, data1: u8, offset0: u8, offset1: u8) -> [u32; 2] {
    let w0 = (0x36u32 << 26) | (0x0Eu32 << 18) | ((offset1 as u32) << 8) | (offset0 as u32);
    let w1 = ((data1 as u32) << 16) | ((data0 as u32) << 8) | (addr as u32);
    [w0, w1]
}

/// `ds_read_b32 vdst, vaddr offset:offset16` (DS `0x36`, GFX10.3 `op[25:18]`).
pub const fn ds_read_b32(vdst: u8, addr: u8, offset: u16) -> [u32; 2] {
    let w0 = (0x36u32 << 26) | (0x36u32 << 18) | (offset as u32);
    let w1 = ((vdst as u32) << 24) | (addr as u32);
    [w0, w1]
}

/// `ds_read2_b32 vdst, vaddr offset0:off0 offset1:off1` (DS `0x37`, scaled by 4 bytes).
pub const fn ds_read2_b32(vdst: u8, addr: u8, offset0: u8, offset1: u8) -> [u32; 2] {
    let w0 = (0x36u32 << 26) | (0x37u32 << 18) | ((offset1 as u32) << 8) | (offset0 as u32);
    let w1 = ((vdst as u32) << 24) | (addr as u32);
    [w0, w1]
}

// ---------------------------------------------------------------------------
// Graphics encoders: EXP, VINTRP, MIMG (LLVM 22.1.8 gfx1030 verified)
// ---------------------------------------------------------------------------

/// EXP target constants.
pub const EXP_MRT0: u8 = 0;
pub const EXP_MRTZ: u8 = 8;
pub const EXP_NULL: u8 = 9;
pub const EXP_POS0: u8 = 12;
pub const EXP_PRIM: u8 = 20;
pub const EXP_PARAM0: u8 = 32;

/// `exp target, vsrc0, vsrc1, vsrc2, vsrc3 [compr] [done] [vm]` (`top6 == 0x3E`, 2 DWORDs).
pub const fn exp(target: u8, en: u8, compr: bool, done: bool, vm: bool, vsrc: [u8; 4]) -> [u32; 2] {
    let w0 = (0x3Eu32 << 26)
        | ((vm as u32) << 12)
        | ((done as u32) << 11)
        | ((compr as u32) << 10)
        | (((target as u32) & 0x3F) << 4)
        | ((en as u32) & 0xF);
    let w1 = (vsrc[0] as u32)
        | ((vsrc[1] as u32) << 8)
        | ((vsrc[2] as u32) << 16)
        | ((vsrc[3] as u32) << 24);
    [w0, w1]
}

pub const fn exp_pos0(vsrc: [u8; 4], done: bool) -> [u32; 2] {
    exp(EXP_POS0, 0xF, false, done, false, vsrc)
}

pub const fn exp_param(index: u8, vsrc: [u8; 4], done: bool) -> [u32; 2] {
    exp(EXP_PARAM0 + index, 0xF, false, done, false, vsrc)
}

pub const fn exp_mrt(index: u8, vsrc: [u8; 4], compr: bool, done: bool, vm: bool) -> [u32; 2] {
    exp(EXP_MRT0 + index, 0xF, compr, done, vm, vsrc)
}

pub const fn exp_mrtz(vdepth: u8, done: bool, vm: bool) -> [u32; 2] {
    exp(EXP_MRTZ, 0x1, false, done, vm, [vdepth, 0, 0, 0])
}

pub const fn exp_null(done: bool, vm: bool) -> [u32; 2] {
    exp(EXP_NULL, 0x0, false, done, vm, [0, 0, 0, 0])
}

/// `exp prim vsrc0, off, off, off [done]` (`target = 20`, `en = 0x1`).
pub const fn exp_prim(vsrc0: u8, done: bool) -> [u32; 2] {
    exp(EXP_PRIM, 0x1, false, done, false, [vsrc0, 0, 0, 0])
}

/// VINTRP 1-DWORD encoding (`top6 == 0x32`).
pub const fn vintrp(op: u8, vdst: u8, attr: u8, chan: u8, vsrc: u8) -> u32 {
    (0x32u32 << 26)
        | ((vdst as u32) << 18)
        | (((op as u32) & 0x3) << 16)
        | (((attr as u32) & 0x3F) << 10)
        | (((chan as u32) & 0x3) << 8)
        | (vsrc as u32)
}

pub const fn v_interp_p1_f32(vdst: u8, vsrc: u8, attr: u8, chan: u8) -> u32 {
    vintrp(0, vdst, attr, chan, vsrc)
}
pub const fn v_interp_p2_f32(vdst: u8, vsrc: u8, attr: u8, chan: u8) -> u32 {
    vintrp(1, vdst, attr, chan, vsrc)
}
pub const fn v_interp_mov_f32(vdst: u8, param: u8, attr: u8, chan: u8) -> u32 {
    vintrp(2, vdst, attr, chan, param)
}

/// MIMG 2-DWORD non-NSA encoding (`top6 == 0x3C`, GFX10.3 verified).
/// `srsrc` and `ssamp` are actual SGPR base indices (must be 4-aligned).
#[allow(clippy::too_many_arguments)]
pub const fn mimg(
    op: u16,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    ssamp: u8,
    dmask: u8,
    dim: u8,
    unrm: bool,
) -> [u32; 2] {
    let op_lo = (op as u32) & 0x7F;
    let op_hi = ((op as u32) >> 7) & 1;
    let w0 = (0x3Cu32 << 26)
        | (op_lo << 18)
        | ((unrm as u32) << 12)
        | (((dmask as u32) & 0xF) << 8)
        | (((dim as u32) & 0x7) << 3)
        | op_hi;
    let w1 = ((((ssamp as u32) >> 2) & 0x1F) << 21)
        | ((((srsrc as u32) >> 2) & 0x1F) << 16)
        | ((vdata as u32) << 8)
        | (vaddr as u32);
    [w0, w1]
}

pub const fn image_load_2d(vdata: u8, vaddr: u8, srsrc: u8, dmask: u8) -> [u32; 2] {
    mimg(0x00, vdata, vaddr, srsrc, 0, dmask, 1, true)
}
pub const fn image_store_2d(vdata: u8, vaddr: u8, srsrc: u8, dmask: u8) -> [u32; 2] {
    mimg(0x08, vdata, vaddr, srsrc, 0, dmask, 1, true)
}
pub const fn image_sample_2d(vdata: u8, vaddr: u8, srsrc: u8, ssamp: u8, dmask: u8) -> [u32; 2] {
    mimg(0x20, vdata, vaddr, srsrc, ssamp, dmask, 1, false)
}
pub const fn image_sample_l_2d(vdata: u8, vaddr: u8, srsrc: u8, ssamp: u8, dmask: u8) -> [u32; 2] {
    mimg(0x24, vdata, vaddr, srsrc, ssamp, dmask, 1, false)
}
/// The address registers start with bias, followed by the two coordinates.
pub const fn image_sample_b_2d(vdata: u8, vaddr: u8, srsrc: u8, ssamp: u8, dmask: u8) -> [u32; 2] {
    mimg(0x25, vdata, vaddr, srsrc, ssamp, dmask, 1, false)
}
pub const fn image_sample_lz_2d(vdata: u8, vaddr: u8, srsrc: u8, ssamp: u8, dmask: u8) -> [u32; 2] {
    mimg(0x27, vdata, vaddr, srsrc, ssamp, dmask, 1, false)
}

/// Build a DPP16 extension DWORD (`SRC0 = 0xFA`).
#[allow(clippy::too_many_arguments)]
pub const fn dpp16_word(
    vsrc0: u8,
    ctrl: u16,
    fi: bool,
    bc: bool,
    row_mask: u8,
    bank_mask: u8,
) -> u32 {
    (vsrc0 as u32)
        | (((ctrl as u32) & 0x1FF) << 8)
        | ((fi as u32) << 18)
        | ((bc as u32) << 19)
        | (((bank_mask as u32) & 0xF) << 24)
        | (((row_mask as u32) & 0xF) << 28)
}

/// Build a DPP8 extension DWORD (`SRC0 = 0xE9` or `0xEA`).
pub const fn dpp8_word(vsrc0: u8, lanes: [u8; 8]) -> u32 {
    let mut w = vsrc0 as u32;
    let mut i = 0;
    while i < 8 {
        w |= ((lanes[i] as u32) & 0x7) << (8 + 3 * i);
        i += 1;
    }
    w
}

// ---------------------------------------------------------------------------
// Memory encoders: SMEM, MUBUF, GLOBAL (LLVM 22.1.8 gfx1030 verified)
// ---------------------------------------------------------------------------

/// SMEM 64-bit encoding (`top6 == 0x3D`).
/// `sbase` is the actual SGPR index (must be even; encoded as `sbase >> 1`).
/// `soffset` is raw 7-bit scalar selector (`0x7D` = NULL / zero).
pub const fn smem(op: u8, sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    let w0 = (0x3Du32 << 26)
        | ((op as u32) << 18)
        | (((sdata as u32) & 0x7F) << 6)
        | (((sbase as u32) >> 1) & 0x3F);
    let w1 = (((soffset as u32) & 0x7F) << 25) | ((ioffset as u32) & 0x1F_FFFF);
    [w0, w1]
}

pub const fn s_load_dword(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x00, sdata, sbase, soffset, ioffset)
}
pub const fn s_load_dwordx2(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x01, sdata, sbase, soffset, ioffset)
}
pub const fn s_load_dwordx4(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x02, sdata, sbase, soffset, ioffset)
}
pub const fn s_load_dwordx8(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x03, sdata, sbase, soffset, ioffset)
}
pub const fn s_buffer_load_dword(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x08, sdata, sbase, soffset, ioffset)
}
pub const fn s_buffer_load_dwordx2(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x09, sdata, sbase, soffset, ioffset)
}
pub const fn s_buffer_load_dwordx4(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x0A, sdata, sbase, soffset, ioffset)
}
pub const fn s_buffer_load_dwordx8(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
    smem(0x0B, sdata, sbase, soffset, ioffset)
}

/// MUBUF 64-bit encoding (`top6 == 0x38`).
/// `srsrc` is the actual SGPR base index (must be 4-aligned; encoded as `srsrc >> 2`).
#[allow(clippy::too_many_arguments)]
pub const fn mubuf(
    op: u8,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    let w0 = (0x38u32 << 26)
        | (((op as u32) & 0x7F) << 18)
        | ((idxen as u32) << 13)
        | ((offen as u32) << 12)
        | ((offset as u32) & 0xFFF);
    let w1 = (((soffset as u32) & 0xFF) << 24)
        | ((((srsrc as u32) >> 2) & 0x1F) << 16)
        | ((vdata as u32) << 8)
        | (vaddr as u32);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
pub const fn buffer_atomic_add(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
    return_old: bool,
) -> [u32; 2] {
    let [w0, w1] = mubuf(0x32, vdata, vaddr, srsrc, soffset, offset, offen, idxen);
    [w0 | ((return_old as u32) << 14), w1]
}

pub const fn buffer_load_ushort(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x0A, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_store_short(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x1A, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}

pub const fn buffer_load_dword(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x0C, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_load_dwordx2(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x0D, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_load_dwordx4(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x0E, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_load_dwordx3(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x0F, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_store_dword(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x1C, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_store_dwordx2(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x1D, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_store_dwordx4(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x1E, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}
pub const fn buffer_store_dwordx3(
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u16,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> [u32; 2] {
    mubuf(0x1F, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
}

/// FLAT/GLOBAL 64-bit encoding (`top6 == 0x37`, `seg == 2` for GLOBAL).
/// `saddr = 0x7D` selects 64-bit VGPR pair `v[addr:addr+1]`; otherwise `s[saddr:saddr+1] + v[addr]`.
pub const fn flat_global(
    op: u8,
    seg: u8,
    vdst: u8,
    addr: u8,
    data: u8,
    saddr: u8,
    offset: i16,
) -> [u32; 2] {
    let w0 = (0x37u32 << 26)
        | (((op as u32) & 0x7F) << 18)
        | (((seg as u32) & 0x3) << 14)
        | ((offset as u32) & 0xFFF);
    let w1 = ((vdst as u32) << 24)
        | (((saddr as u32) & 0x7F) << 16)
        | ((data as u32) << 8)
        | (addr as u32);
    [w0, w1]
}

pub const fn global_atomic_add(
    addr: u8,
    data: u8,
    saddr: u8,
    offset: i16,
    return_old: Option<u8>,
) -> [u32; 2] {
    let (vdst, glc) = match return_old {
        Some(vdst) => (vdst, 1 << 16),
        None => (0, 0),
    };
    let [w0, w1] = flat_global(0x32, 2, vdst, addr, data, saddr, offset);
    [w0 | glc, w1]
}

pub const fn global_load_dword(vdst: u8, addr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x0C, 2, vdst, addr, 0, saddr, offset)
}
pub const fn global_load_dwordx2(vdst: u8, addr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x0D, 2, vdst, addr, 0, saddr, offset)
}
pub const fn global_load_dwordx4(vdst: u8, addr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x0E, 2, vdst, addr, 0, saddr, offset)
}
pub const fn global_load_dwordx3(vdst: u8, addr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x0F, 2, vdst, addr, 0, saddr, offset)
}
pub const fn global_store_dword(addr: u8, data: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x1C, 2, 0, addr, data, saddr, offset)
}
pub const fn global_store_dwordx2(addr: u8, data: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x1D, 2, 0, addr, data, saddr, offset)
}
pub const fn global_store_dwordx4(addr: u8, data: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x1E, 2, 0, addr, data, saddr, offset)
}
pub const fn global_store_dwordx3(addr: u8, data: u8, saddr: u8, offset: i16) -> [u32; 2] {
    flat_global(0x1F, 2, 0, addr, data, saddr, offset)
}
