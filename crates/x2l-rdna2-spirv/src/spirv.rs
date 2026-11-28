// Clean-room direct binary SPIR-V 1.6 compute lowering for X2Linux.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Direct `RDNA2 -> Binary SPIR-V 1.6` compute shader emitter (`rspirv`).
//!
//! Emits Vulkan 1.3-compliant binary SPIR-V 1.6 directly via `rspirv` without HLSL text,
//! DXIL, DXC, or `vkd3d-proton`. Supports both portable workgroup-shared lowering (runs
//! identically on RADV, ANV, NVK, and Lavapipe at any SIMD subgroup width) and single-subgroup
//! lowering (`VK_EXT_subgroup_size_control` / `OpGroupNonUniform*`), plus 3-binding guest
//! memory lowering (`SMEM`, `MUBUF` with clean-room AMD RDNA2 V# decoding, and `GLOBAL`/`FLAT`).

use crate::container::parse_container;
use crate::decode::{
    DecodedInst, DstOperand, Encoding, InstExtra, SrcOperand, ValuExt, ValuModifiers,
    disassemble_reachable,
};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, GLOp, LoopControl, MemoryModel, MemorySemantics, Scope, SelectionControl,
    StorageClass,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use x2l_pipeline::{Converter, PipelineError};

#[derive(Clone, Copy)]
pub(crate) enum MemoryWrite {
    Store,
    AtomicAdd(Option<u8>),
}

impl MemoryWrite {
    pub(crate) fn for_inst(inst: &DecodedInst) -> Self {
        if inst.is_atomic_add() {
            Self::AtomicAdd(match inst.dst {
                DstOperand::Vgpr(reg) => Some(reg),
                _ => None,
            })
        } else {
            Self::Store
        }
    }

    pub(crate) fn result(self) -> Option<u8> {
        match self {
            Self::Store => None,
            Self::AtomicAdd(dst) => dst,
        }
    }

    pub(crate) fn emit(
        self,
        b: &mut Builder,
        uint: u32,
        ptr: u32,
        val: u32,
        scope: Scope,
        result_var: u32,
    ) -> Result<(), SpirvError> {
        match self {
            Self::Store => b.store(ptr, val, None, [])?,
            Self::AtomicAdd(dst) => {
                let scope = b.constant_bit32(uint, scope as u32);
                let relaxed = b.constant_bit32(uint, MemorySemantics::NONE.bits());
                let old = b.atomic_i_add(uint, None, ptr, scope, relaxed, val)?;
                if dst.is_some() {
                    b.store(result_var, old, None, [])?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Alu<'a> {
    pub(crate) b: &'a mut Builder,
    pub(crate) uint: u32,
    pub(crate) sint: u32,
    pub(crate) f32_ty: u32,
    pub(crate) bool_ty: u32,
    pub(crate) glsl: u32,
    pub(crate) c_true: u32,
    pub(crate) c_false: u32,
    pub(crate) u32_consts: &'a mut BTreeMap<u32, u32>,
    pub(crate) f32_consts: &'a mut BTreeMap<u32, u32>,
}

impl Alu<'_> {
    pub(crate) fn eval_f32_cmp_pred(
        &mut self,
        code: u16,
        a: u32,
        b: u32,
    ) -> Result<u32, SpirvError> {
        match code {
            0x0 => Ok(self.c_false),
            0x1 => Ok(self.b.f_ord_less_than(self.bool_ty, None, a, b)?),
            0x2 => Ok(self.b.f_ord_equal(self.bool_ty, None, a, b)?),
            0x3 => Ok(self.b.f_ord_less_than_equal(self.bool_ty, None, a, b)?),
            0x4 => Ok(self.b.f_ord_greater_than(self.bool_ty, None, a, b)?),
            0x5 => Ok(self.b.f_ord_not_equal(self.bool_ty, None, a, b)?),
            0x6 => Ok(self.b.f_ord_greater_than_equal(self.bool_ty, None, a, b)?),
            0x7 => {
                let na = self.b.is_nan(self.bool_ty, None, a)?;
                let nb = self.b.is_nan(self.bool_ty, None, b)?;
                let any_nan = self.b.logical_or(self.bool_ty, None, na, nb)?;
                Ok(self.b.logical_not(self.bool_ty, None, any_nan)?)
            }
            0x8 => {
                let na = self.b.is_nan(self.bool_ty, None, a)?;
                let nb = self.b.is_nan(self.bool_ty, None, b)?;
                Ok(self.b.logical_or(self.bool_ty, None, na, nb)?)
            }
            0x9 => {
                let ge = self.b.f_ord_greater_than_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, ge)?)
            }
            0xA => {
                let lg = self.b.f_ord_not_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, lg)?)
            }
            0xB => {
                let gt = self.b.f_ord_greater_than(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, gt)?)
            }
            0xC => {
                let le = self.b.f_ord_less_than_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, le)?)
            }
            0xD => {
                let eq = self.b.f_ord_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, eq)?)
            }
            0xE => {
                let lt = self.b.f_ord_less_than(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, lt)?)
            }
            0xF => Ok(self.c_true),
            _ => unreachable!(),
        }
    }

    pub(crate) fn eval_int_cmp_pred(
        &mut self,
        code: u16,
        a: u32,
        b: u32,
        signed: bool,
    ) -> Result<u32, SpirvError> {
        let (lhs, rhs) = if signed {
            (
                self.b.bitcast(self.sint, None, a)?,
                self.b.bitcast(self.sint, None, b)?,
            )
        } else {
            (a, b)
        };
        match code {
            0x0 => Ok(self.c_false),
            0x1 => Ok(if signed {
                self.b.s_less_than(self.bool_ty, None, lhs, rhs)?
            } else {
                self.b.u_less_than(self.bool_ty, None, lhs, rhs)?
            }),
            0x2 => Ok(self.b.i_equal(self.bool_ty, None, a, b)?),
            0x3 => Ok(if signed {
                self.b.s_less_than_equal(self.bool_ty, None, lhs, rhs)?
            } else {
                self.b.u_less_than_equal(self.bool_ty, None, lhs, rhs)?
            }),
            0x4 => Ok(if signed {
                self.b.s_greater_than(self.bool_ty, None, lhs, rhs)?
            } else {
                self.b.u_greater_than(self.bool_ty, None, lhs, rhs)?
            }),
            0x5 => Ok(self.b.i_not_equal(self.bool_ty, None, a, b)?),
            0x6 => Ok(if signed {
                self.b.s_greater_than_equal(self.bool_ty, None, lhs, rhs)?
            } else {
                self.b.u_greater_than_equal(self.bool_ty, None, lhs, rhs)?
            }),
            0x7 => Ok(self.c_true),
            _ => unreachable!(),
        }
    }

    fn const_u32(&mut self, value: u32) -> u32 {
        *self
            .u32_consts
            .entry(value)
            .or_insert_with(|| self.b.constant_bit32(self.uint, value))
    }

    fn const_f32(&mut self, value: f32) -> u32 {
        let bits = value.to_bits();
        *self
            .f32_consts
            .entry(bits)
            .or_insert_with(|| self.b.constant_bit32(self.f32_ty, bits))
    }

    pub(crate) fn apply_f32_sign_mods(
        &mut self,
        mut bits: u32,
        abs: bool,
        neg: bool,
    ) -> Result<u32, SpirvError> {
        if abs {
            let mask = self.const_u32(0x7fff_ffff);
            bits = self.b.bitwise_and(self.uint, None, bits, mask)?;
        }
        if neg {
            let sign = self.const_u32(0x8000_0000);
            bits = self.b.bitwise_xor(self.uint, None, bits, sign)?;
        }
        Ok(bits)
    }

    fn is_f32_nan_bits(&mut self, bits: u32) -> Result<u32, SpirvError> {
        let magnitude_mask = self.const_u32(0x7fff_ffff);
        let infinity = self.const_u32(f32::INFINITY.to_bits());
        let magnitude = self.b.bitwise_and(self.uint, None, bits, magnitude_mask)?;
        Ok(self
            .b
            .u_greater_than(self.bool_ty, None, magnitude, infinity)?)
    }

    fn ext_unary(&mut self, op: GLOp, f: u32) -> Result<u32, SpirvError> {
        Ok(self
            .b
            .ext_inst(self.f32_ty, None, self.glsl, op as u32, [Operand::IdRef(f)])?)
    }

    pub(crate) fn emit_unary(&mut self, mnemonic: &str, f: u32) -> Result<u32, SpirvError> {
        let op = match mnemonic {
            "v_fract_f32" => GLOp::Fract,
            "v_trunc_f32" => GLOp::Trunc,
            "v_ceil_f32" => GLOp::Ceil,
            "v_rndne_f32" => GLOp::RoundEven,
            "v_floor_f32" => GLOp::Floor,
            "v_exp_f32" => GLOp::Exp2,
            "v_log_f32" => GLOp::Log2,
            "v_sin_f32" => GLOp::Sin,
            "v_cos_f32" => GLOp::Cos,
            "v_rsq_f32" => GLOp::InverseSqrt,
            "v_sqrt_f32" => GLOp::Sqrt,
            _ => unreachable!("unary float operation {mnemonic}"),
        };
        match op {
            GLOp::Sin | GLOp::Cos => self.emit_trig(op, f),
            GLOp::Exp2 => {
                let result = self.ext_unary(op, f)?;
                self.flush_denormal(result)
            }
            GLOp::Log2 => {
                let f = self.flush_denormal(f)?;
                let zero = self.const_f32(0.0);
                let is_zero = self.b.f_ord_equal(self.bool_ty, None, f, zero)?;
                let negative = self.b.f_ord_less_than(self.bool_ty, None, f, zero)?;
                let invalid = self.b.logical_or(self.bool_ty, None, negative, is_zero)?;
                let one = self.const_f32(1.0);
                let safe = self.b.select(self.f32_ty, None, invalid, one, f)?;
                let result = self.ext_unary(op, safe)?;
                let neg_inf = self.const_f32(f32::NEG_INFINITY);
                let nan = self.const_f32(f32::from_bits(0xffc0_0000));
                let result = self.b.select(self.f32_ty, None, is_zero, neg_inf, result)?;
                Ok(self.b.select(self.f32_ty, None, negative, nan, result)?)
            }
            _ => self.ext_unary(op, f),
        }
    }

    fn flush_denormal(&mut self, f: u32) -> Result<u32, SpirvError> {
        let bits = self.b.bitcast(self.uint, None, f)?;
        let exponent_mask = self.const_u32(0x7f80_0000);
        let sign_mask = self.const_u32(0x8000_0000);
        let zero = self.const_u32(0);
        let exponent = self.b.bitwise_and(self.uint, None, bits, exponent_mask)?;
        let sign = self.b.bitwise_and(self.uint, None, bits, sign_mask)?;
        let flush = self.b.i_equal(self.bool_ty, None, exponent, zero)?;
        let result = self.b.select(self.uint, None, flush, sign, bits)?;
        Ok(self.b.bitcast(self.f32_ty, None, result)?)
    }

    fn emit_trig(&mut self, op: GLOp, f: u32) -> Result<u32, SpirvError> {
        // Reduce in turns before converting to radians, preserving the fractional input.
        let whole = self.ext_unary(GLOp::RoundEven, f)?;
        let turns = self.b.f_sub(self.f32_ty, None, f, whole)?;
        let tau = self.const_f32(std::f32::consts::TAU);
        let radians = self.b.f_mul(self.f32_ty, None, turns, tau)?;
        let mut result = self.ext_unary(op, radians)?;
        // The exact quadrants avoid residual error from the binary32 approximation of pi.
        let zero = self.const_f32(0.0);
        let quarter = self.const_f32(0.25);
        let neg_quarter = self.const_f32(-0.25);
        let half = self.const_f32(0.5);
        let one = self.const_f32(1.0);
        let neg_one = self.const_f32(-1.0);
        let abs_turns = self.ext_unary(GLOp::FAbs, turns)?;
        let is_half = self.b.f_ord_equal(self.bool_ty, None, abs_turns, half)?;
        if op == GLOp::Sin {
            let is_pos_quarter = self.b.f_ord_equal(self.bool_ty, None, turns, quarter)?;
            let is_neg_quarter = self.b.f_ord_equal(self.bool_ty, None, turns, neg_quarter)?;
            let is_zero = self.b.f_ord_equal(self.bool_ty, None, f, zero)?;
            result = self.b.select(self.f32_ty, None, is_half, zero, result)?;
            result = self
                .b
                .select(self.f32_ty, None, is_pos_quarter, one, result)?;
            result = self
                .b
                .select(self.f32_ty, None, is_neg_quarter, neg_one, result)?;
            Ok(self.b.select(self.f32_ty, None, is_zero, f, result)?)
        } else {
            let is_quarter = self.b.f_ord_equal(self.bool_ty, None, abs_turns, quarter)?;
            result = self.b.select(self.f32_ty, None, is_quarter, zero, result)?;
            Ok(self.b.select(self.f32_ty, None, is_half, neg_one, result)?)
        }
    }

    pub(crate) fn emit_min_max_med3(
        &mut self,
        mnemonic: &str,
        a: u32,
        b: u32,
        c: u32,
    ) -> Result<u32, SpirvError> {
        if mnemonic == "v_max3_f32" {
            let max_ab = self.emit_min_max_f32(a, b, true)?;
            return self.emit_min_max_f32(max_ab, c, true);
        }
        let min_ab = self.emit_min_max_f32(a, b, false)?;
        if mnemonic == "v_min3_f32" {
            return self.emit_min_max_f32(min_ab, c, false);
        }

        let max_ab = self.emit_min_max_f32(a, b, true)?;
        let min_hi = self.emit_min_max_f32(max_ab, c, false)?;
        let median = self.emit_min_max_f32(min_ab, min_hi, true)?;
        // The median instruction takes the minimum path if any source is NaN.
        let minimum = self.emit_min_max_f32(min_ab, c, false)?;
        let a_bits = self.b.bitcast(self.uint, None, a)?;
        let b_bits = self.b.bitcast(self.uint, None, b)?;
        let c_bits = self.b.bitcast(self.uint, None, c)?;
        let a_nan = self.is_f32_nan_bits(a_bits)?;
        let b_nan = self.is_f32_nan_bits(b_bits)?;
        let c_nan = self.is_f32_nan_bits(c_bits)?;
        let ab_nan = self.b.logical_or(self.bool_ty, None, a_nan, b_nan)?;
        let any_nan = self.b.logical_or(self.bool_ty, None, ab_nan, c_nan)?;
        Ok(self.b.select(self.f32_ty, None, any_nan, minimum, median)?)
    }

    /// Bit-exact IEEE/RDNA2 `v_min_f32` / `v_max_f32` (handles NaN -> numeric operand and `-0` vs `+0`).
    pub(crate) fn emit_min_max_f32(
        &mut self,
        a: u32,
        b: u32,
        is_max: bool,
    ) -> Result<u32, SpirvError> {
        let a_bits = self.b.bitcast(self.uint, None, a)?;
        let b_bits = self.b.bitcast(self.uint, None, b)?;
        let a_nan = self.is_f32_nan_bits(a_bits)?;
        let b_nan = self.is_f32_nan_bits(b_bits)?;

        let sign_mask = self.const_u32(0x8000_0000);

        // Ordered unsigned key: neg => !bits, nonneg => bits ^ 0x8000_0000
        let a_neg = self
            .b
            .u_greater_than_equal(self.bool_ty, None, a_bits, sign_mask)?;
        let a_not = self.b.not(self.uint, None, a_bits)?;
        let a_flip = self.b.bitwise_xor(self.uint, None, a_bits, sign_mask)?;
        let a_key = self.b.select(self.uint, None, a_neg, a_not, a_flip)?;

        let b_neg = self
            .b
            .u_greater_than_equal(self.bool_ty, None, b_bits, sign_mask)?;
        let b_not = self.b.not(self.uint, None, b_bits)?;
        let b_flip = self.b.bitwise_xor(self.uint, None, b_bits, sign_mask)?;
        let b_key = self.b.select(self.uint, None, b_neg, b_not, b_flip)?;

        let pick_a = if is_max {
            self.b.u_greater_than(self.bool_ty, None, a_key, b_key)?
        } else {
            self.b.u_less_than(self.bool_ty, None, a_key, b_key)?
        };
        let ord_bits = self.b.select(self.uint, None, pick_a, a_bits, b_bits)?;
        // If a is NaN -> b; else if b is NaN -> a; else ordered result
        let not_a_nan_sel = self.b.select(self.uint, None, b_nan, a_bits, ord_bits)?;
        let res_bits = self
            .b
            .select(self.uint, None, a_nan, b_bits, not_a_nan_sel)?;
        Ok(self.b.bitcast(self.f32_ty, None, res_bits)?)
    }

    /// Safe saturating `v_cvt_i32_f32` (NaN -> 0, >= 2^31 -> INT_MAX, < -2^31 -> INT_MIN).
    pub(crate) fn emit_cvt_i32_f32(&mut self, f: u32) -> Result<u32, SpirvError> {
        let zero_f = self.const_f32(0.0);
        let zero_u = self.const_u32(0);
        let bits = self.b.bitcast(self.uint, None, f)?;
        let is_nan = self.is_f32_nan_bits(bits)?;
        let f_min = self.const_f32(-2147483648.0_f32);
        let f_max = self.const_f32(2147483648.0_f32); // 2^31 exact in f32
        let too_low = self.b.f_ord_less_than(self.bool_ty, None, f, f_min)?;
        let too_high = self
            .b
            .f_ord_greater_than_equal(self.bool_ty, None, f, f_max)?;
        let bad = self.b.logical_or(self.bool_ty, None, is_nan, too_low)?;
        let bad = self.b.logical_or(self.bool_ty, None, bad, too_high)?;
        let safe_f = self.b.select(self.f32_ty, None, bad, zero_f, f)?;
        let s = self.b.convert_f_to_s(self.sint, None, safe_f)?;
        let raw = self.b.bitcast(self.uint, None, s)?;
        let c_imin = self.const_u32(i32::MIN as u32);
        let c_imax = self.const_u32(i32::MAX as u32);
        let r1 = self.b.select(self.uint, None, too_high, c_imax, raw)?;
        let r2 = self.b.select(self.uint, None, too_low, c_imin, r1)?;
        Ok(self.b.select(self.uint, None, is_nan, zero_u, r2)?)
    }

    /// Safe saturating `v_cvt_u32_f32` (NaN / < 0 -> 0, >= 2^32 -> UINT_MAX).
    pub(crate) fn emit_cvt_u32_f32(&mut self, f: u32) -> Result<u32, SpirvError> {
        let zero_f = self.const_f32(0.0);
        let zero_u = self.const_u32(0);
        let bits = self.b.bitcast(self.uint, None, f)?;
        let is_nan = self.is_f32_nan_bits(bits)?;
        let f_max = self.const_f32(4294967296.0_f32); // 2^32 exact in f32
        let is_neg = self.b.f_ord_less_than(self.bool_ty, None, f, zero_f)?;
        let too_high = self
            .b
            .f_ord_greater_than_equal(self.bool_ty, None, f, f_max)?;
        let bad = self.b.logical_or(self.bool_ty, None, is_nan, is_neg)?;
        let bad = self.b.logical_or(self.bool_ty, None, bad, too_high)?;
        let safe_f = self.b.select(self.f32_ty, None, bad, zero_f, f)?;
        let raw = self.b.convert_f_to_u(self.uint, None, safe_f)?;
        let c_umax = self.const_u32(u32::MAX);
        let r1 = self.b.select(self.uint, None, too_high, c_umax, raw)?;
        let zero_cond = self.b.logical_or(self.bool_ty, None, is_nan, is_neg)?;
        Ok(self.b.select(self.uint, None, zero_cond, zero_u, r1)?)
    }
}

/// Byte stride of one `Wave` record in the Vulkan `StorageBuffer` ABI (`66_072` bytes).
pub const WAVE_STATE_STRIDE: u32 = 66_072;

/// Number of 32-bit DWORDs allocated for compute LDS (`Workgroup u32[4096]` = 16 KiB,
/// fitting cleanly alongside the 256-byte cross-lane scratchpad inside the 32 KiB Vulkan limit).
pub const LDS_DWORDS: u32 = 4096;

const HALT_PC: u32 = u32::MAX;
const NUM_SGPRS: u32 = 128;
const NUM_VGPRS: u32 = 256;
const MAX_LANES: u32 = 64;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Guest wave width targeted by the emitted compute shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaveSize {
    #[default]
    Wave32,
    Wave64,
}

impl WaveSize {
    pub fn width(self) -> u32 {
        match self {
            Self::Wave32 => 32,
            Self::Wave64 => 64,
        }
    }

    pub fn from_wave32(wave32: bool) -> Self {
        if wave32 { Self::Wave32 } else { Self::Wave64 }
    }
}

/// Strategy used to lower cross-lane shuffles and wave-wide ballots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossLaneMode {
    /// Portable single-workgroup lowering using `LocalInvocationIndex` + `Workgroup u32[64]`
    /// scratchpad + `OpControlBarrier`. Requires only `Capability::Shader` and executes
    /// bit-identically across all Vulkan 1.3 devices (including Lavapipe at any SIMD width).
    #[default]
    WorkgroupShared,
    /// Native single-subgroup lowering using `SubgroupLocalInvocationId` +
    /// `OpGroupNonUniformBallot` + `OpGroupNonUniformShuffle`. Requires a Vulkan compute pipeline
    /// pinned to `requiredSubgroupSize == wave_size.width()`.
    Subgroup,
}

/// Configuration for direct SPIR-V 1.6 compute lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileOptions {
    pub wave_size: WaveSize,
    pub cross_lane_mode: CrossLaneMode,
    /// Emit `RoundingModeRTE` and `SignedZeroInfNanPreserve` for binary32.
    pub strict_float_controls: bool,
    /// Emit `DenormPreserve` for binary32 (disable on devices such as Lavapipe where
    /// `shaderDenormPreserveFloat32 == false`).
    pub denorm_preserve: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            wave_size: WaveSize::Wave32,
            cross_lane_mode: CrossLaneMode::WorkgroupShared,
            strict_float_controls: true,
            denorm_preserve: true,
        }
    }
}

/// Errors raised while lowering decoded RDNA2 instructions to binary SPIR-V 1.6 or validating.
#[derive(Debug, Error)]
pub enum SpirvError {
    #[error("cannot compile empty instruction stream")]
    EmptyProgram,
    #[error("branch at PC 0x{pc:04x} targets unknown instruction PC 0x{target:04x}")]
    InvalidBranchTarget { pc: u32, target: u32 },
    #[error("unsupported instruction at PC 0x{pc:04x}: {mnemonic}")]
    UnsupportedInstruction { pc: u32, mnemonic: String },
    #[error("unsupported operand at PC 0x{pc:04x}: {detail}")]
    UnsupportedOperand { pc: u32, detail: String },
    #[error("unsupported VALU extension at PC 0x{pc:04x}: {detail}")]
    UnsupportedValuExt { pc: u32, detail: String },
    #[error("SPIR-V builder error: {0}")]
    Builder(String),
    #[error("SPIR-V byte stream length {0} is not a multiple of 4")]
    UnalignedSpirvBytes(usize),
    #[error("spirv-val executable not found on PATH: {0}")]
    ValidatorMissing(String),
    #[error("spirv-val validation failed:\n{0}")]
    ValidationFailed(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<rspirv::dr::Error> for SpirvError {
    fn from(value: rspirv::dr::Error) -> Self {
        Self::Builder(value.to_string())
    }
}

/// Compile decoded RDNA2 instructions into a Vulkan 1.3-valid binary SPIR-V 1.6 compute module
/// using default options (`Wave32`, `WorkgroupShared`, strict float controls).
pub fn compile_compute_spv(insts: &[DecodedInst]) -> Result<Vec<u32>, SpirvError> {
    compile_compute_spv_with_options(insts, CompileOptions::default())
}

/// Compile decoded RDNA2 instructions into binary SPIR-V 1.6 with explicit options.
pub fn compile_compute_spv_with_options(
    insts: &[DecodedInst],
    opts: CompileOptions,
) -> Result<Vec<u32>, SpirvError> {
    if insts.is_empty() {
        return Err(SpirvError::EmptyProgram);
    }

    let uses_memory = insts.iter().any(|i| {
        matches!(
            i.extra,
            InstExtra::Smem { .. } | InstExtra::Mubuf { .. } | InstExtra::Flat { .. }
        )
    });
    let uses_lds = insts
        .iter()
        .any(|i| matches!(i.extra, InstExtra::DsLds { .. }));
    let blocks = build_basic_blocks(insts)?;
    let mut ctx = EmitCtx::new(opts, uses_memory, uses_lds);
    ctx.emit_module(&blocks)?;
    Ok(ctx.b.module().assemble())
}

/// Serialize assembled SPIR-V words to little-endian bytes.
pub fn spirv_words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Parse little-endian SPIR-V bytes back into `u32` words.
pub fn spirv_bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, SpirvError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(SpirvError::UnalignedSpirvBytes(bytes.len()));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    Ok(chunks.iter().map(|c| u32::from_le_bytes(*c)).collect())
}

/// Returns `true` if `spirv-val` is available on `$PATH`.
pub fn is_spirv_val_available() -> bool {
    Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Validate assembled SPIR-V words with `spirv-val --target-env vulkan1.3`.
pub fn validate_spirv_vulkan13(words: &[u32]) -> Result<(), SpirvError> {
    let bytes = spirv_words_to_bytes(words);
    let unique = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_path =
        std::env::temp_dir().join(format!("x2l_spv_val_{}_{}.spv", std::process::id(), unique));
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    fs::write(&tmp_path, &bytes)?;
    let _guard = Cleanup(tmp_path.clone());

    let output = Command::new("spirv-val")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg(&tmp_path)
        .output()
        .map_err(|e| SpirvError::ValidatorMissing(e.to_string()))?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let detail = if stderr.is_empty() { stdout } else { stderr };
        Err(SpirvError::ValidationFailed(detail))
    }
}

/// `x2l-pipeline::Converter` implementation compiling console shader containers or raw RDNA2
/// compute streams into cached binary SPIR-V 1.6 under `$XDG_CACHE_HOME/x2linux/<title>/spirv/`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rdna2SpirvConverter;

impl Converter for Rdna2SpirvConverter {
    fn id(&self) -> &'static str {
        "shader.rdna2_spirv"
    }

    fn version(&self) -> u32 {
        2
    }

    fn convert(&self, input: &[u8]) -> Result<Vec<u8>, PipelineError> {
        let conv_err = |msg: String| PipelineError::Convert {
            id: self.id(),
            version: self.version(),
            message: msg,
        };
        let parsed =
            parse_container(input).map_err(|e| conv_err(format!("container parse failed: {e}")))?;
        let (code_words, wave32) =
            if let Some(v) = parsed.variants.iter().find(|v| !v.code.is_empty()) {
                (v.code_words(), v.wave32)
            } else if let Some(words) = parsed.primary_code_words() {
                (words, true)
            } else {
                return Err(conv_err("container has no native RDNA2 code stages".into()));
            };
        let insts = disassemble_reachable(&code_words)
            .map_err(|e| conv_err(format!("RDNA2 decode failed: {e}")))?;
        let words = compile_compute_spv_with_options(
            &insts,
            CompileOptions {
                wave_size: WaveSize::from_wave32(wave32),
                ..CompileOptions::default()
            },
        )
        .map_err(|e| conv_err(format!("SPIR-V lowering failed: {e}")))?;
        Ok(spirv_words_to_bytes(&words))
    }
}

// ---------------------------------------------------------------------------
// Basic-block partitioning
// ---------------------------------------------------------------------------

struct BasicBlock<'a> {
    id: u32,
    insts: Vec<&'a DecodedInst>,
    fallthrough_block: Option<u32>,
}

fn build_basic_blocks<'a>(
    insts: &'a [DecodedInst],
) -> Result<(Vec<BasicBlock<'a>>, BTreeMap<u32, u32>), SpirvError> {
    let mut pc_to_idx = BTreeMap::new();
    for (idx, inst) in insts.iter().enumerate() {
        pc_to_idx.insert(inst.pc, idx);
    }

    let mut leaders = BTreeSet::new();
    leaders.insert(0usize);

    for (idx, inst) in insts.iter().enumerate() {
        if let Some(target) = inst.branch_target {
            let &target_idx = pc_to_idx
                .get(&target)
                .ok_or(SpirvError::InvalidBranchTarget {
                    pc: inst.pc,
                    target,
                })?;
            leaders.insert(target_idx);
            if idx + 1 < insts.len() {
                leaders.insert(idx + 1);
            }
        } else if inst.is_terminator && idx + 1 < insts.len() {
            leaders.insert(idx + 1);
        }
    }

    let leader_list: Vec<usize> = leaders.into_iter().collect();
    let mut pc_to_block = BTreeMap::new();
    for (block_id, &start_idx) in leader_list.iter().enumerate() {
        pc_to_block.insert(insts[start_idx].pc, block_id as u32);
    }

    let mut blocks = Vec::with_capacity(leader_list.len());
    for (block_id, &start_idx) in leader_list.iter().enumerate() {
        let end_idx = leader_list
            .get(block_id + 1)
            .copied()
            .unwrap_or(insts.len());
        let slice: Vec<&'a DecodedInst> = insts[start_idx..end_idx].iter().collect();
        let last = slice.last().expect("non-empty basic block");
        let next_pc = last.pc.wrapping_add(u32::from(last.word_len) * 4);
        let fallthrough_block = pc_to_block.get(&next_pc).copied();
        blocks.push(BasicBlock {
            id: block_id as u32,
            insts: slice,
            fallthrough_block,
        });
    }

    Ok((blocks, pc_to_block))
}

// ---------------------------------------------------------------------------
// SPIR-V 1.6 emitter
// ---------------------------------------------------------------------------

struct EmitCtx {
    opts: CompileOptions,
    b: Builder,
    glsl: u32,
    // Scalar/vector/struct types
    void: u32,
    bool_ty: u32,
    uint: u32,
    sint: u32,
    f32_ty: u32,
    vec2: u32,
    uvec2: u32,
    uvec3: u32,
    uvec4: u32,
    u32_pair_ty: u32,
    // Pointer types
    ptr_sb_uint: u32,
    ptr_sb_uvec2: u32,
    ptr_wg_uint: u32,
    ptr_fn_uint: u32,
    ptr_fn_uvec2: u32,
    ptr_fn_sgprs: u32,
    ptr_fn_vgprs: u32,
    // Global variables
    state_in: u32,
    state_out: u32,
    memory_buf: Option<u32>,
    wg_scratch: Option<u32>,
    lds_buf: Option<u32>,
    lane_builtin: u32,
    workgroup_id: u32,
    // Constants
    c_u0: u32,
    c_u1: u32,
    c_u2: u32,
    c_u3: u32,
    c_u4: u32,
    c_u31: u32,
    c_u32: u32,
    c_u128: u32,
    c_u256: u32,
    c_halt: u32,
    c_f0: u32,
    c_f1: u32,
    c_f2: u32,
    c_f4: u32,
    c_f05: u32,
    c_abs_mask: u32,
    c_sign_mask: u32,
    c_exp_mask: u32,
    c_subgroup_scope: u32,
    c_wg_scope: u32,
    c_wg_sem: u32,
    c_true: u32,
    c_false: u32,
    u32_consts: BTreeMap<u32, u32>,
    f32_consts: BTreeMap<u32, u32>,
    // Function-local variables (initialized in emit_module)
    var_pc: u32,
    var_sgprs: u32,
    var_vgprs: u32,
    var_exec: u32,
    var_vcc: u32,
    var_scc: u32,
    var_idx: u32,
    var_tmp0: u32,
    var_tmp1: u32,
    lane_id: u32,
    mem_words_len: u32,
}

impl EmitCtx {
    fn new(opts: CompileOptions, uses_memory: bool, uses_lds: bool) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 6);
        b.capability(Capability::Shader);
        if opts.cross_lane_mode == CrossLaneMode::Subgroup {
            b.capability(Capability::GroupNonUniform);
            b.capability(Capability::GroupNonUniformBallot);
            b.capability(Capability::GroupNonUniformShuffle);
        }
        if opts.strict_float_controls {
            b.capability(Capability::RoundingModeRTE);
            b.capability(Capability::SignedZeroInfNanPreserve);
        }
        if opts.denorm_preserve {
            b.capability(Capability::DenormPreserve);
        }

        let glsl = b.ext_inst_import("GLSL.std.450");
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = b.type_void();
        let bool_ty = b.type_bool();
        let uint = b.type_int(32, 0);
        let sint = b.type_int(32, 1);
        let f32_ty = b.type_float(32);
        let vec2 = b.type_vector(f32_ty, 2);
        let uvec2 = b.type_vector(uint, 2);
        let uvec3 = b.type_vector(uint, 3);
        let uvec4 = b.type_vector(uint, 4);
        let u32_pair_ty = b.type_struct([uint, uint]);

        let c_u0 = b.constant_bit32(uint, 0);
        let c_u1 = b.constant_bit32(uint, 1);
        let c_u2 = b.constant_bit32(uint, 2);
        let c_u3 = b.constant_bit32(uint, 3);
        let c_u4 = b.constant_bit32(uint, 4);
        let c_u31 = b.constant_bit32(uint, 31);
        let c_u32 = b.constant_bit32(uint, 32);
        let c_u64 = b.constant_bit32(uint, MAX_LANES);
        let c_u128 = b.constant_bit32(uint, NUM_SGPRS);
        let c_u256 = b.constant_bit32(uint, NUM_VGPRS);
        let c_halt = b.constant_bit32(uint, HALT_PC);
        let c_f0 = b.constant_bit32(f32_ty, 0.0_f32.to_bits());
        let c_f1 = b.constant_bit32(f32_ty, 1.0_f32.to_bits());
        let c_f2 = b.constant_bit32(f32_ty, 2.0_f32.to_bits());
        let c_f4 = b.constant_bit32(f32_ty, 4.0_f32.to_bits());
        let c_f05 = b.constant_bit32(f32_ty, 0.5_f32.to_bits());
        let c_abs_mask = b.constant_bit32(uint, 0x7FFF_FFFF);
        let c_sign_mask = b.constant_bit32(uint, 0x8000_0000);
        let c_exp_mask = b.constant_bit32(uint, 0x7F80_0000);
        let c_subgroup_scope = b.constant_bit32(uint, Scope::Subgroup as u32);
        let c_wg_scope = b.constant_bit32(uint, Scope::Workgroup as u32);
        let c_wg_sem = b.constant_bit32(
            uint,
            (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits(),
        );
        let c_true = b.constant_true(bool_ty);
        let c_false = b.constant_false(bool_ty);

        // StorageBuffer Wave ABI block (set 0, binding 0 & 1)
        let sb_sgprs = b.type_array(uint, c_u128);
        let sb_vgpr_lane = b.type_array(uint, c_u256);
        let sb_vgpr_lanes = b.type_array(sb_vgpr_lane, c_u64);
        let wave_ty = b.type_struct([sb_sgprs, sb_vgpr_lanes, uvec2, uvec2, uint]);
        let waves_ty = b.type_runtime_array(wave_ty);
        let block_ty = b.type_struct([waves_ty]);

        for (ty, stride) in [
            (sb_sgprs, 4u32),
            (sb_vgpr_lane, 4),
            (sb_vgpr_lanes, NUM_VGPRS * 4),
            (waves_ty, WAVE_STATE_STRIDE),
        ] {
            b.decorate(ty, Decoration::ArrayStride, [Operand::LiteralBit32(stride)]);
        }
        for (member, offset) in [0u32, 512, 66_048, 66_056, 66_064].into_iter().enumerate() {
            b.member_decorate(
                wave_ty,
                member as u32,
                Decoration::Offset,
                [Operand::LiteralBit32(offset)],
            );
        }
        b.decorate(block_ty, Decoration::Block, []);
        b.member_decorate(block_ty, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);

        let sb = StorageClass::StorageBuffer;
        let ptr_sb_block = b.type_pointer(None, sb, block_ty);
        let ptr_sb_uint = b.type_pointer(None, sb, uint);
        let ptr_sb_uvec2 = b.type_pointer(None, sb, uvec2);

        let state_in = b.variable(ptr_sb_block, None, sb, None);
        let state_out = b.variable(ptr_sb_block, None, sb, None);
        for (var, binding) in [(state_in, 0u32), (state_out, 1)] {
            b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(binding)]);
        }
        b.decorate(state_in, Decoration::NonWritable, []);

        // Optional binding 2: flat/test MemoryBuffer (`Block { u32[] }`, stride 4)
        let memory_buf = if uses_memory {
            let mem_arr = b.type_runtime_array(uint);
            b.decorate(mem_arr, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
            let mem_block = b.type_struct([mem_arr]);
            b.decorate(mem_block, Decoration::Block, []);
            b.member_decorate(mem_block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
            let ptr_sb_mem = b.type_pointer(None, sb, mem_block);
            let var = b.variable(ptr_sb_mem, None, sb, None);
            b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(2)]);
            Some(var)
        } else {
            None
        };

        // Optional Workgroup scratchpad (`Workgroup u32[64]`, distinct undecorated array type)
        let ptr_wg_uint = b.type_pointer(None, StorageClass::Workgroup, uint);
        let wg_scratch = if opts.cross_lane_mode == CrossLaneMode::WorkgroupShared {
            let wg_arr_id = b.id();
            let wg_arr_ty = b.type_array_id(Some(wg_arr_id), uint, c_u64);
            let ptr_wg_arr = b.type_pointer(None, StorageClass::Workgroup, wg_arr_ty);
            Some(b.variable(ptr_wg_arr, None, StorageClass::Workgroup, None))
        } else {
            None
        };

        // Optional compute LDS (`Workgroup u32[LDS_DWORDS]`, 16 KiB, distinct undecorated array type)
        let lds_buf = if uses_lds {
            let c_lds = b.constant_bit32(uint, LDS_DWORDS);
            let lds_arr_id = b.id();
            let lds_arr_ty = b.type_array_id(Some(lds_arr_id), uint, c_lds);
            let ptr_lds_arr = b.type_pointer(None, StorageClass::Workgroup, lds_arr_ty);
            Some(b.variable(ptr_lds_arr, None, StorageClass::Workgroup, None))
        } else {
            None
        };

        // Input built-ins
        let input_sc = StorageClass::Input;
        let ptr_in_uint = b.type_pointer(None, input_sc, uint);
        let ptr_in_uvec3 = b.type_pointer(None, input_sc, uvec3);
        let lane_builtin = b.variable(ptr_in_uint, None, input_sc, None);
        let workgroup_id = b.variable(ptr_in_uvec3, None, input_sc, None);
        let lane_bi = match opts.cross_lane_mode {
            CrossLaneMode::WorkgroupShared => BuiltIn::LocalInvocationIndex,
            CrossLaneMode::Subgroup => BuiltIn::SubgroupLocalInvocationId,
        };
        b.decorate(
            lane_builtin,
            Decoration::BuiltIn,
            [Operand::BuiltIn(lane_bi)],
        );
        b.decorate(
            workgroup_id,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::WorkgroupId)],
        );

        // Distinct undecorated Function-local array types (must NOT carry ArrayStride).
        let fn_sgprs_id = b.id();
        let fn_sgprs_ty = b.type_array_id(Some(fn_sgprs_id), uint, c_u128);
        let fn_vgprs_id = b.id();
        let fn_vgprs_ty = b.type_array_id(Some(fn_vgprs_id), uint, c_u256);

        let fn_sc = StorageClass::Function;
        let ptr_fn_uint = b.type_pointer(None, fn_sc, uint);
        let ptr_fn_uvec2 = b.type_pointer(None, fn_sc, uvec2);
        let ptr_fn_sgprs = b.type_pointer(None, fn_sc, fn_sgprs_ty);
        let ptr_fn_vgprs = b.type_pointer(None, fn_sc, fn_vgprs_ty);

        let mut u32_consts = BTreeMap::new();
        for &(val, id) in &[
            (0, c_u0),
            (1, c_u1),
            (2, c_u2),
            (3, c_u3),
            (4, c_u4),
            (31, c_u31),
            (32, c_u32),
            (MAX_LANES, c_u64),
            (NUM_SGPRS, c_u128),
            (NUM_VGPRS, c_u256),
            (HALT_PC, c_halt),
            (0x7FFF_FFFF, c_abs_mask),
            (0x8000_0000, c_sign_mask),
            (0x7F80_0000, c_exp_mask),
        ] {
            u32_consts.insert(val, id);
        }
        Self {
            opts,
            b,
            glsl,
            void,
            bool_ty,
            uint,
            sint,
            f32_ty,
            vec2,
            uvec2,
            uvec3,
            uvec4,
            u32_pair_ty,
            ptr_sb_uint,
            ptr_sb_uvec2,
            ptr_wg_uint,
            ptr_fn_uint,
            ptr_fn_uvec2,
            ptr_fn_sgprs,
            ptr_fn_vgprs,
            state_in,
            state_out,
            memory_buf,
            wg_scratch,
            lds_buf,
            lane_builtin,
            workgroup_id,
            c_u0,
            c_u1,
            c_u2,
            c_u3,
            c_u4,
            c_u31,
            c_u32,
            c_u128,
            c_u256,
            c_halt,
            c_f0,
            c_f1,
            c_f2,
            c_f4,
            c_f05,
            c_abs_mask,
            c_sign_mask,
            c_exp_mask,
            c_subgroup_scope,
            c_wg_scope,
            c_wg_sem,
            c_true,
            c_false,
            u32_consts,
            f32_consts: [(0.0_f32.to_bits(), c_f0), (1.0_f32.to_bits(), c_f1)].into(),
            var_pc: 0,
            var_sgprs: 0,
            var_vgprs: 0,
            var_exec: 0,
            var_vcc: 0,
            var_scc: 0,
            var_idx: 0,
            var_tmp0: 0,
            var_tmp1: 0,
            lane_id: 0,
            mem_words_len: 0,
        }
    }

    fn const_u32(&mut self, val: u32) -> u32 {
        if let Some(&id) = self.u32_consts.get(&val) {
            id
        } else {
            let id = self.b.constant_bit32(self.uint, val);
            self.u32_consts.insert(val, id);
            id
        }
    }

    fn emit_module(
        &mut self,
        (blocks, pc_to_block): &(Vec<BasicBlock<'_>>, BTreeMap<u32, u32>),
    ) -> Result<(), SpirvError> {
        let fn_ty = self.b.type_function(self.void, []);
        let entry = self
            .b
            .begin_function(self.void, None, FunctionControl::NONE, fn_ty)?;

        self.b.begin_block(None)?;

        // Declare ALL Function variables at the very start of the entry block.
        let fn_sc = StorageClass::Function;
        self.var_pc = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        self.var_sgprs = self.b.variable(self.ptr_fn_sgprs, None, fn_sc, None);
        self.var_vgprs = self.b.variable(self.ptr_fn_vgprs, None, fn_sc, None);
        self.var_exec = self.b.variable(self.ptr_fn_uvec2, None, fn_sc, None);
        self.var_vcc = self.b.variable(self.ptr_fn_uvec2, None, fn_sc, None);
        self.var_scc = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        self.var_idx = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        self.var_tmp0 = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        self.var_tmp1 = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);

        let wg_val = self.b.load(self.uvec3, None, self.workgroup_id, None, [])?;
        let wave_id = self.b.composite_extract(self.uint, None, wg_val, [0])?;
        self.lane_id = self.b.load(self.uint, None, self.lane_builtin, None, [])?;

        if let Some(mem_buf) = self.memory_buf {
            self.mem_words_len = self.b.array_length(self.uint, None, mem_buf, 0)?;
        }

        // Load initial SGPR[0..128] and VGPR[lane][0..256]
        self.emit_copy_sgprs_from_sb(wave_id)?;
        self.emit_copy_vgprs_from_sb(wave_id, self.lane_id)?;

        // Load initial EXEC, VCC, SCC
        let p_in_exec = self.b.access_chain(
            self.ptr_sb_uvec2,
            None,
            self.state_in,
            [self.c_u0, wave_id, self.c_u2],
        )?;
        let exec_init = self.b.load(self.uvec2, None, p_in_exec, None, [])?;
        self.b.store(self.var_exec, exec_init, None, [])?;

        let p_in_vcc = self.b.access_chain(
            self.ptr_sb_uvec2,
            None,
            self.state_in,
            [self.c_u0, wave_id, self.c_u3],
        )?;
        let vcc_init = self.b.load(self.uvec2, None, p_in_vcc, None, [])?;
        self.b.store(self.var_vcc, vcc_init, None, [])?;

        let p_in_scc = self.b.access_chain(
            self.ptr_sb_uint,
            None,
            self.state_in,
            [self.c_u0, wave_id, self.c_u4],
        )?;
        let scc_init = self.b.load(self.uint, None, p_in_scc, None, [])?;
        self.b.store(self.var_scc, scc_init, None, [])?;

        self.b.store(self.var_pc, self.c_u0, None, [])?;

        // Main structured CFG dispatcher (workgroup-uniform)
        let loop_header = self.b.id();
        let switch_header = self.b.id();
        let switch_merge = self.b.id();
        let loop_continue = self.b.id();
        let loop_exit = self.b.id();
        let default_label = self.b.id();
        let case_labels: Vec<(u32, u32)> = blocks.iter().map(|bb| (bb.id, self.b.id())).collect();

        self.b.branch(loop_header)?;

        self.b.begin_block(Some(loop_header))?;
        let cur_pc = self.b.load(self.uint, None, self.var_pc, None, [])?;
        let done = self.b.i_equal(self.bool_ty, None, cur_pc, self.c_halt)?;
        self.b
            .loop_merge(loop_exit, loop_continue, LoopControl::NONE, [])?;
        self.b
            .branch_conditional(done, loop_exit, switch_header, [])?;

        self.b.begin_block(Some(switch_header))?;
        self.b
            .selection_merge(switch_merge, SelectionControl::NONE)?;
        self.b.switch(
            cur_pc,
            default_label,
            case_labels
                .iter()
                .map(|&(id, lbl)| (Operand::LiteralBit32(id), lbl)),
        )?;

        for (bb, &(_, lbl)) in blocks.iter().zip(case_labels.iter()) {
            self.b.begin_block(Some(lbl))?;
            let mut set_pc = false;
            for inst in &bb.insts {
                if self.emit_inst(inst, bb.fallthrough_block, pc_to_block)? {
                    set_pc = true;
                }
            }
            if !set_pc {
                let next_id = bb.fallthrough_block.unwrap_or(HALT_PC);
                let next_c = self.const_u32(next_id);
                self.b.store(self.var_pc, next_c, None, [])?;
            }
            self.b.branch(switch_merge)?;
        }

        self.b.begin_block(Some(default_label))?;
        self.b.store(self.var_pc, self.c_halt, None, [])?;
        self.b.branch(switch_merge)?;

        self.b.begin_block(Some(switch_merge))?;
        self.b.branch(loop_continue)?;

        self.b.begin_block(Some(loop_continue))?;
        self.b.branch(loop_header)?;

        // loop_exit: write back per-lane VGPRs, then lane 0 writes scalar state
        self.b.begin_block(Some(loop_exit))?;
        self.emit_copy_vgprs_to_sb(wave_id, self.lane_id)?;

        let save_scalar = self.b.id();
        let return_block = self.b.id();
        let is_lane_zero = self
            .b
            .i_equal(self.bool_ty, None, self.lane_id, self.c_u0)?;
        self.b
            .selection_merge(return_block, SelectionControl::NONE)?;
        self.b
            .branch_conditional(is_lane_zero, save_scalar, return_block, [])?;

        self.b.begin_block(Some(save_scalar))?;
        self.emit_copy_sgprs_to_sb(wave_id)?;

        let exec_final = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
        let p_out_exec = self.b.access_chain(
            self.ptr_sb_uvec2,
            None,
            self.state_out,
            [self.c_u0, wave_id, self.c_u2],
        )?;
        self.b.store(p_out_exec, exec_final, None, [])?;

        let vcc_final = self.b.load(self.uvec2, None, self.var_vcc, None, [])?;
        let p_out_vcc = self.b.access_chain(
            self.ptr_sb_uvec2,
            None,
            self.state_out,
            [self.c_u0, wave_id, self.c_u3],
        )?;
        self.b.store(p_out_vcc, vcc_final, None, [])?;

        let scc_final = self.b.load(self.uint, None, self.var_scc, None, [])?;
        let p_out_scc = self.b.access_chain(
            self.ptr_sb_uint,
            None,
            self.state_out,
            [self.c_u0, wave_id, self.c_u4],
        )?;
        self.b.store(p_out_scc, scc_final, None, [])?;
        self.b.branch(return_block)?;

        self.b.begin_block(Some(return_block))?;
        self.b.ret()?;
        self.b.end_function()?;

        let mut interface = vec![
            self.state_in,
            self.state_out,
            self.lane_builtin,
            self.workgroup_id,
        ];
        if let Some(mem) = self.memory_buf {
            interface.push(mem);
        }
        if let Some(wg) = self.wg_scratch {
            interface.push(wg);
        }
        if let Some(lds) = self.lds_buf {
            interface.push(lds);
        }
        self.b
            .entry_point(ExecutionModel::GLCompute, entry, "main", interface);
        self.b.execution_mode(
            entry,
            ExecutionMode::LocalSize,
            [self.opts.wave_size.width(), 1, 1],
        );
        if self.opts.strict_float_controls {
            self.b
                .execution_mode(entry, ExecutionMode::RoundingModeRTE, [32]);
            self.b
                .execution_mode(entry, ExecutionMode::SignedZeroInfNanPreserve, [32]);
        }
        if self.opts.denorm_preserve {
            self.b
                .execution_mode(entry, ExecutionMode::DenormPreserve, [32]);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Entry/exit structured register copy loops
    // -----------------------------------------------------------------------

    fn emit_copy_sgprs_from_sb(&mut self, wave_id: u32) -> Result<(), SpirvError> {
        let hdr = self.b.id();
        let body = self.b.id();
        let cont = self.b.id();
        let exit = self.b.id();

        self.b.store(self.var_idx, self.c_u0, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(hdr))?;
        let i = self.b.load(self.uint, None, self.var_idx, None, [])?;
        let cmp = self.b.u_less_than(self.bool_ty, None, i, self.c_u128)?;
        self.b.loop_merge(exit, cont, LoopControl::NONE, [])?;
        self.b.branch_conditional(cmp, body, exit, [])?;

        self.b.begin_block(Some(body))?;
        let src_p = self.b.access_chain(
            self.ptr_sb_uint,
            None,
            self.state_in,
            [self.c_u0, wave_id, self.c_u0, i],
        )?;
        let val = self.b.load(self.uint, None, src_p, None, [])?;
        let dst_p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_sgprs, [i])?;
        self.b.store(dst_p, val, None, [])?;
        self.b.branch(cont)?;

        self.b.begin_block(Some(cont))?;
        let next = self.b.i_add(self.uint, None, i, self.c_u1)?;
        self.b.store(self.var_idx, next, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(exit))?;
        Ok(())
    }

    fn emit_copy_vgprs_from_sb(&mut self, wave_id: u32, lane: u32) -> Result<(), SpirvError> {
        let hdr = self.b.id();
        let body = self.b.id();
        let cont = self.b.id();
        let exit = self.b.id();

        self.b.store(self.var_idx, self.c_u0, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(hdr))?;
        let i = self.b.load(self.uint, None, self.var_idx, None, [])?;
        let cmp = self.b.u_less_than(self.bool_ty, None, i, self.c_u256)?;
        self.b.loop_merge(exit, cont, LoopControl::NONE, [])?;
        self.b.branch_conditional(cmp, body, exit, [])?;

        self.b.begin_block(Some(body))?;
        let src_p = self.b.access_chain(
            self.ptr_sb_uint,
            None,
            self.state_in,
            [self.c_u0, wave_id, self.c_u1, lane, i],
        )?;
        let val = self.b.load(self.uint, None, src_p, None, [])?;
        let dst_p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_vgprs, [i])?;
        self.b.store(dst_p, val, None, [])?;
        self.b.branch(cont)?;

        self.b.begin_block(Some(cont))?;
        let next = self.b.i_add(self.uint, None, i, self.c_u1)?;
        self.b.store(self.var_idx, next, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(exit))?;
        Ok(())
    }

    fn emit_copy_sgprs_to_sb(&mut self, wave_id: u32) -> Result<(), SpirvError> {
        let hdr = self.b.id();
        let body = self.b.id();
        let cont = self.b.id();
        let exit = self.b.id();

        self.b.store(self.var_idx, self.c_u0, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(hdr))?;
        let i = self.b.load(self.uint, None, self.var_idx, None, [])?;
        let cmp = self.b.u_less_than(self.bool_ty, None, i, self.c_u128)?;
        self.b.loop_merge(exit, cont, LoopControl::NONE, [])?;
        self.b.branch_conditional(cmp, body, exit, [])?;

        self.b.begin_block(Some(body))?;
        let src_p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_sgprs, [i])?;
        let val = self.b.load(self.uint, None, src_p, None, [])?;
        let dst_p = self.b.access_chain(
            self.ptr_sb_uint,
            None,
            self.state_out,
            [self.c_u0, wave_id, self.c_u0, i],
        )?;
        self.b.store(dst_p, val, None, [])?;
        self.b.branch(cont)?;

        self.b.begin_block(Some(cont))?;
        let next = self.b.i_add(self.uint, None, i, self.c_u1)?;
        self.b.store(self.var_idx, next, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(exit))?;
        Ok(())
    }

    fn emit_copy_vgprs_to_sb(&mut self, wave_id: u32, lane: u32) -> Result<(), SpirvError> {
        let hdr = self.b.id();
        let body = self.b.id();
        let cont = self.b.id();
        let exit = self.b.id();

        self.b.store(self.var_idx, self.c_u0, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(hdr))?;
        let i = self.b.load(self.uint, None, self.var_idx, None, [])?;
        let cmp = self.b.u_less_than(self.bool_ty, None, i, self.c_u256)?;
        self.b.loop_merge(exit, cont, LoopControl::NONE, [])?;
        self.b.branch_conditional(cmp, body, exit, [])?;

        self.b.begin_block(Some(body))?;
        let src_p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_vgprs, [i])?;
        let val = self.b.load(self.uint, None, src_p, None, [])?;
        let dst_p = self.b.access_chain(
            self.ptr_sb_uint,
            None,
            self.state_out,
            [self.c_u0, wave_id, self.c_u1, lane, i],
        )?;
        self.b.store(dst_p, val, None, [])?;
        self.b.branch(cont)?;

        self.b.begin_block(Some(cont))?;
        let next = self.b.i_add(self.uint, None, i, self.c_u1)?;
        self.b.store(self.var_idx, next, None, [])?;
        self.b.branch(hdr)?;

        self.b.begin_block(Some(exit))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Cross-lane primitives (WorkgroupShared vs Subgroup)
    // -----------------------------------------------------------------------

    fn gather_from_lane(&mut self, src_bits: u32, safe_lane: u32) -> Result<u32, SpirvError> {
        match self.opts.cross_lane_mode {
            CrossLaneMode::Subgroup => Ok(self.b.group_non_uniform_shuffle(
                self.uint,
                None,
                self.c_subgroup_scope,
                src_bits,
                safe_lane,
            )?),
            CrossLaneMode::WorkgroupShared => {
                let scratch = self.wg_scratch.expect("wg_scratch allocated");
                let own_p = self
                    .b
                    .access_chain(self.ptr_wg_uint, None, scratch, [self.lane_id])?;
                self.b.store(own_p, src_bits, None, [])?;
                self.b
                    .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_sem)?;
                let target_p = self
                    .b
                    .access_chain(self.ptr_wg_uint, None, scratch, [safe_lane])?;
                let picked = self.b.load(self.uint, None, target_p, None, [])?;
                self.b
                    .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_sem)?;
                Ok(picked)
            }
        }
    }

    fn ballot_wave(&mut self, vote: u32) -> Result<(u32, Option<u32>), SpirvError> {
        match self.opts.cross_lane_mode {
            CrossLaneMode::Subgroup => {
                let ballot = self.b.group_non_uniform_ballot(
                    self.uvec4,
                    None,
                    self.c_subgroup_scope,
                    vote,
                )?;
                let lo = self.b.composite_extract(self.uint, None, ballot, [0])?;
                let hi = match self.opts.wave_size {
                    WaveSize::Wave32 => None,
                    WaveSize::Wave64 => {
                        Some(self.b.composite_extract(self.uint, None, ballot, [1])?)
                    }
                };
                Ok((lo, hi))
            }
            CrossLaneMode::WorkgroupShared => {
                let scratch = self.wg_scratch.expect("wg_scratch allocated");
                let shift = self
                    .b
                    .bitwise_and(self.uint, None, self.lane_id, self.c_u31)?;
                let bit = self
                    .b
                    .shift_left_logical(self.uint, None, self.c_u1, shift)?;
                let slot_val = self.b.select(self.uint, None, vote, bit, self.c_u0)?;
                let own_p = self
                    .b
                    .access_chain(self.ptr_wg_uint, None, scratch, [self.lane_id])?;
                self.b.store(own_p, slot_val, None, [])?;
                self.b
                    .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_sem)?;

                // Structured 32-step OR reduction into var_tmp0 (lo) and var_tmp1 (hi for Wave64)
                self.b.store(self.var_tmp0, self.c_u0, None, [])?;
                self.b.store(self.var_tmp1, self.c_u0, None, [])?;
                self.b.store(self.var_idx, self.c_u0, None, [])?;

                let hdr = self.b.id();
                let body = self.b.id();
                let cont = self.b.id();
                let exit = self.b.id();
                self.b.branch(hdr)?;

                self.b.begin_block(Some(hdr))?;
                let k = self.b.load(self.uint, None, self.var_idx, None, [])?;
                let cmp = self.b.u_less_than(self.bool_ty, None, k, self.c_u32)?;
                self.b.loop_merge(exit, cont, LoopControl::NONE, [])?;
                self.b.branch_conditional(cmp, body, exit, [])?;

                self.b.begin_block(Some(body))?;
                let p_lo = self.b.access_chain(self.ptr_wg_uint, None, scratch, [k])?;
                let v_lo = self.b.load(self.uint, None, p_lo, None, [])?;
                let acc_lo = self.b.load(self.uint, None, self.var_tmp0, None, [])?;
                let new_lo = self.b.bitwise_or(self.uint, None, acc_lo, v_lo)?;
                self.b.store(self.var_tmp0, new_lo, None, [])?;

                if self.opts.wave_size == WaveSize::Wave64 {
                    let k_hi = self.b.i_add(self.uint, None, k, self.c_u32)?;
                    let p_hi = self
                        .b
                        .access_chain(self.ptr_wg_uint, None, scratch, [k_hi])?;
                    let v_hi = self.b.load(self.uint, None, p_hi, None, [])?;
                    let acc_hi = self.b.load(self.uint, None, self.var_tmp1, None, [])?;
                    let new_hi = self.b.bitwise_or(self.uint, None, acc_hi, v_hi)?;
                    self.b.store(self.var_tmp1, new_hi, None, [])?;
                }
                self.b.branch(cont)?;

                self.b.begin_block(Some(cont))?;
                let next = self.b.i_add(self.uint, None, k, self.c_u1)?;
                self.b.store(self.var_idx, next, None, [])?;
                self.b.branch(hdr)?;

                self.b.begin_block(Some(exit))?;
                self.b
                    .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_sem)?;

                let lo = self.b.load(self.uint, None, self.var_tmp0, None, [])?;
                let hi = if self.opts.wave_size == WaveSize::Wave64 {
                    Some(self.b.load(self.uint, None, self.var_tmp1, None, [])?)
                } else {
                    None
                };
                Ok((lo, hi))
            }
        }
    }

    // -----------------------------------------------------------------------
    // 64-bit unsigned address helpers (two u32 words, zero Capability::Int64)
    // -----------------------------------------------------------------------

    fn u64_add(
        &mut self,
        a_lo: u32,
        a_hi: u32,
        b_lo: u32,
        b_hi: u32,
    ) -> Result<(u32, u32), SpirvError> {
        let sum_carry = self.b.i_add_carry(self.u32_pair_ty, None, a_lo, b_lo)?;
        let lo = self.b.composite_extract(self.uint, None, sum_carry, [0])?;
        let carry = self.b.composite_extract(self.uint, None, sum_carry, [1])?;
        let hi1 = self.b.i_add(self.uint, None, a_hi, b_hi)?;
        let hi = self.b.i_add(self.uint, None, hi1, carry)?;
        Ok((lo, hi))
    }

    fn u64_mul_u32(&mut self, a: u32, b: u32) -> Result<(u32, u32), SpirvError> {
        let prod = self.b.u_mul_extended(self.u32_pair_ty, None, a, b)?;
        let lo = self.b.composite_extract(self.uint, None, prod, [0])?;
        let hi = self.b.composite_extract(self.uint, None, prod, [1])?;
        Ok((lo, hi))
    }

    fn u64_le(&mut self, a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> Result<u32, SpirvError> {
        let hi_lt = self.b.u_less_than(self.bool_ty, None, a_hi, b_hi)?;
        let hi_eq = self.b.i_equal(self.bool_ty, None, a_hi, b_hi)?;
        let lo_le = self.b.u_less_than_equal(self.bool_ty, None, a_lo, b_lo)?;
        let eq_and_le = self.b.logical_and(self.bool_ty, None, hi_eq, lo_le)?;
        Ok(self.b.logical_or(self.bool_ty, None, hi_lt, eq_and_le)?)
    }

    // -----------------------------------------------------------------------
    // Register & mask helpers
    // -----------------------------------------------------------------------

    fn load_sgpr(&mut self, idx: u8) -> Result<u32, SpirvError> {
        let idx_c = self.const_u32(u32::from(idx));
        let p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_sgprs, [idx_c])?;
        Ok(self.b.load(self.uint, None, p, None, [])?)
    }

    fn store_sgpr(&mut self, idx: u8, val: u32) -> Result<(), SpirvError> {
        let idx_c = self.const_u32(u32::from(idx));
        let p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_sgprs, [idx_c])?;
        self.b.store(p, val, None, [])?;
        Ok(())
    }

    fn load_vgpr(&mut self, idx: u8) -> Result<u32, SpirvError> {
        let idx_c = self.const_u32(u32::from(idx));
        let p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_vgprs, [idx_c])?;
        Ok(self.b.load(self.uint, None, p, None, [])?)
    }

    fn store_vgpr_raw(&mut self, idx: u8, val: u32) -> Result<(), SpirvError> {
        let idx_c = self.const_u32(u32::from(idx));
        let p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_vgprs, [idx_c])?;
        self.b.store(p, val, None, [])?;
        Ok(())
    }

    fn store_vgpr_conditional(
        &mut self,
        idx: u8,
        cond: u32,
        new_val: u32,
    ) -> Result<(), SpirvError> {
        let old_val = self.load_vgpr(idx)?;
        let merged = self.b.select(self.uint, None, cond, new_val, old_val)?;
        self.store_vgpr_raw(idx, merged)
    }

    fn store_vgpr_exec_masked(&mut self, idx: u8, new_val: u32) -> Result<(), SpirvError> {
        let active = self.eval_lane_active()?;
        self.store_vgpr_conditional(idx, active, new_val)
    }

    fn load_uvec2_word(&mut self, var: u32, lane: u32) -> Result<u32, SpirvError> {
        let vec = self.b.load(self.uvec2, None, var, None, [])?;
        Ok(self.b.composite_extract(self.uint, None, vec, [lane])?)
    }

    fn store_uvec2_word(&mut self, var: u32, lane: u32, word: u32) -> Result<(), SpirvError> {
        let vec = self.b.load(self.uvec2, None, var, None, [])?;
        let lo = if lane == 0 {
            word
        } else {
            self.b.composite_extract(self.uint, None, vec, [0])?
        };
        let hi = if lane == 1 {
            word
        } else {
            self.b.composite_extract(self.uint, None, vec, [1])?
        };
        let updated = self.b.composite_construct(self.uvec2, None, [lo, hi])?;
        self.b.store(var, updated, None, [])?;
        Ok(())
    }

    fn store_scc_bool(&mut self, cond: u32) -> Result<(), SpirvError> {
        let word = self.b.select(self.uint, None, cond, self.c_u1, self.c_u0)?;
        self.b.store(self.var_scc, word, None, [])?;
        Ok(())
    }

    fn load_scc_bool(&mut self) -> Result<u32, SpirvError> {
        let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
        Ok(self.b.i_not_equal(self.bool_ty, None, scc, self.c_u0)?)
    }

    fn eval_mask_bit_for_lane(&mut self, mask_uvec2: u32, lane: u32) -> Result<u32, SpirvError> {
        let lo = self.b.composite_extract(self.uint, None, mask_uvec2, [0])?;
        let word = match self.opts.wave_size {
            WaveSize::Wave32 => lo,
            WaveSize::Wave64 => {
                let hi = self.b.composite_extract(self.uint, None, mask_uvec2, [1])?;
                let is_hi = self
                    .b
                    .u_greater_than_equal(self.bool_ty, None, lane, self.c_u32)?;
                self.b.select(self.uint, None, is_hi, hi, lo)?
            }
        };
        let shift = self.b.bitwise_and(self.uint, None, lane, self.c_u31)?;
        let bit = self
            .b
            .shift_left_logical(self.uint, None, self.c_u1, shift)?;
        let masked = self.b.bitwise_and(self.uint, None, word, bit)?;
        Ok(self.b.i_not_equal(self.bool_ty, None, masked, self.c_u0)?)
    }

    fn eval_mask_bit(&mut self, mask_uvec2: u32) -> Result<u32, SpirvError> {
        self.eval_mask_bit_for_lane(mask_uvec2, self.lane_id)
    }

    fn eval_lane_active(&mut self) -> Result<u32, SpirvError> {
        let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
        self.eval_mask_bit(exec)
    }

    fn is_mask_zero(&mut self, mask_var: u32) -> Result<u32, SpirvError> {
        let vec = self.b.load(self.uvec2, None, mask_var, None, [])?;
        let lo = self.b.composite_extract(self.uint, None, vec, [0])?;
        let combined = match self.opts.wave_size {
            WaveSize::Wave32 => lo,
            WaveSize::Wave64 => {
                let hi = self.b.composite_extract(self.uint, None, vec, [1])?;
                self.b.bitwise_or(self.uint, None, lo, hi)?
            }
        };
        Ok(self.b.i_equal(self.bool_ty, None, combined, self.c_u0)?)
    }

    fn load_src_u32(&mut self, pc: u32, src: SrcOperand) -> Result<u32, SpirvError> {
        match src {
            SrcOperand::Sgpr(n) => self.load_sgpr(n),
            SrcOperand::Vgpr(n) => self.load_vgpr(n),
            SrcOperand::VccLo => self.load_uvec2_word(self.var_vcc, 0),
            SrcOperand::VccHi => self.load_uvec2_word(self.var_vcc, 1),
            SrcOperand::ExecLo => self.load_uvec2_word(self.var_exec, 0),
            SrcOperand::ExecHi => self.load_uvec2_word(self.var_exec, 1),
            SrcOperand::M0 => self.load_sgpr(124),
            SrcOperand::Scc => self
                .b
                .load(self.uint, None, self.var_scc, None, [])
                .map_err(Into::into),
            SrcOperand::Vccz => {
                let z = self.is_mask_zero(self.var_vcc)?;
                Ok(self.b.select(self.uint, None, z, self.c_u1, self.c_u0)?)
            }
            SrcOperand::Execz => {
                let z = self.is_mask_zero(self.var_exec)?;
                Ok(self.b.select(self.uint, None, z, self.c_u1, self.c_u0)?)
            }
            SrcOperand::Null => Ok(self.c_u0),
            SrcOperand::InlineInt(v) => Ok(self.const_u32(v as u32)),
            SrcOperand::InlineFloat(f) => Ok(self.const_u32(f.to_bits())),
            SrcOperand::Literal(bits) => Ok(self.const_u32(bits)),
            SrcOperand::Special(code) => Err(SpirvError::UnsupportedOperand {
                pc,
                detail: format!("special source operand {code}"),
            }),
        }
    }

    /// Load a true 64-bit scalar operand (`b64` SOP1/SOP2), independent of wave width.
    /// Zero-extends literals and sign-extends negative inline integers per AMD RDNA2 ISA.
    fn load_scalar_b64(&mut self, pc: u32, src: SrcOperand) -> Result<(u32, u32), SpirvError> {
        match src {
            SrcOperand::VccLo => {
                let v = self.b.load(self.uvec2, None, self.var_vcc, None, [])?;
                Ok((
                    self.b.composite_extract(self.uint, None, v, [0])?,
                    self.b.composite_extract(self.uint, None, v, [1])?,
                ))
            }
            SrcOperand::ExecLo => {
                let v = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
                Ok((
                    self.b.composite_extract(self.uint, None, v, [0])?,
                    self.b.composite_extract(self.uint, None, v, [1])?,
                ))
            }
            SrcOperand::Sgpr(n) => {
                let next = n.checked_add(1).filter(|&x| x < 128).ok_or_else(|| {
                    SpirvError::UnsupportedOperand {
                        pc,
                        detail: format!("b64 scalar pair s{n} out of range"),
                    }
                })?;
                Ok((self.load_sgpr(n)?, self.load_sgpr(next)?))
            }
            SrcOperand::InlineInt(v) => {
                let lo = self.const_u32(v as u32);
                let hi = self.const_u32(if v < 0 { u32::MAX } else { 0 });
                Ok((lo, hi))
            }
            SrcOperand::Literal(bits) => Ok((self.const_u32(bits), self.c_u0)),
            SrcOperand::Null => Ok((self.c_u0, self.c_u0)),
            other => Ok((self.load_src_u32(pc, other)?, self.c_u0)),
        }
    }

    fn load_wave_mask(&mut self, pc: u32, src: SrcOperand) -> Result<u32, SpirvError> {
        match src {
            SrcOperand::VccLo => Ok(self.b.load(self.uvec2, None, self.var_vcc, None, [])?),
            SrcOperand::ExecLo => Ok(self.b.load(self.uvec2, None, self.var_exec, None, [])?),
            SrcOperand::Sgpr(n) => {
                let lo = self.load_sgpr(n)?;
                let hi = match self.opts.wave_size {
                    WaveSize::Wave32 => self.c_u0,
                    WaveSize::Wave64 => {
                        let next = n.checked_add(1).filter(|&x| x < 128).ok_or_else(|| {
                            SpirvError::UnsupportedOperand {
                                pc,
                                detail: format!("wave64 mask pair s{n} out of range"),
                            }
                        })?;
                        self.load_sgpr(next)?
                    }
                };
                Ok(self.b.composite_construct(self.uvec2, None, [lo, hi])?)
            }
            other => {
                let lo = self.load_src_u32(pc, other)?;
                Ok(self
                    .b
                    .composite_construct(self.uvec2, None, [lo, self.c_u0])?)
            }
        }
    }

    fn store_scalar_dst(&mut self, pc: u32, dst: DstOperand, val: u32) -> Result<(), SpirvError> {
        match dst {
            DstOperand::Sgpr(n) => self.store_sgpr(n, val),
            DstOperand::VccLo => self.store_uvec2_word(self.var_vcc, 0, val),
            DstOperand::VccHi => self.store_uvec2_word(self.var_vcc, 1, val),
            DstOperand::ExecLo => self.store_uvec2_word(self.var_exec, 0, val),
            DstOperand::ExecHi => self.store_uvec2_word(self.var_exec, 1, val),
            DstOperand::M0 => self.store_sgpr(124, val),
            DstOperand::Null | DstOperand::None => Ok(()),
            DstOperand::Vgpr(v) => Err(SpirvError::UnsupportedOperand {
                pc,
                detail: format!("scalar write to VGPR v{v}"),
            }),
        }
    }

    fn store_mask_pair(
        &mut self,
        pc: u32,
        dst: DstOperand,
        lo: u32,
        hi: Option<u32>,
    ) -> Result<(), SpirvError> {
        match dst {
            DstOperand::VccLo => match hi {
                Some(h) => {
                    let pair = self.b.composite_construct(self.uvec2, None, [lo, h])?;
                    self.b.store(self.var_vcc, pair, None, [])?;
                    Ok(())
                }
                None => self.store_uvec2_word(self.var_vcc, 0, lo),
            },
            DstOperand::ExecLo => match hi {
                Some(h) => {
                    let pair = self.b.composite_construct(self.uvec2, None, [lo, h])?;
                    self.b.store(self.var_exec, pair, None, [])?;
                    Ok(())
                }
                None => self.store_uvec2_word(self.var_exec, 0, lo),
            },
            DstOperand::Sgpr(n) => {
                self.store_sgpr(n, lo)?;
                if let Some(h) = hi {
                    let next = n.checked_add(1).filter(|&x| x < 128).ok_or_else(|| {
                        SpirvError::UnsupportedOperand {
                            pc,
                            detail: format!("64-bit scalar destination s{n} out of range"),
                        }
                    })?;
                    self.store_sgpr(next, h)?;
                }
                Ok(())
            }
            DstOperand::Null | DstOperand::None => Ok(()),
            other => Err(SpirvError::UnsupportedOperand {
                pc,
                detail: format!("invalid mask destination {other}"),
            }),
        }
    }

    fn load_dst_as_src(&mut self, pc: u32, dst: DstOperand) -> Result<u32, SpirvError> {
        match dst {
            DstOperand::Sgpr(n) => self.load_sgpr(n),
            DstOperand::VccLo => self.load_uvec2_word(self.var_vcc, 0),
            DstOperand::VccHi => self.load_uvec2_word(self.var_vcc, 1),
            DstOperand::ExecLo => self.load_uvec2_word(self.var_exec, 0),
            DstOperand::ExecHi => self.load_uvec2_word(self.var_exec, 1),
            DstOperand::M0 => self.load_sgpr(124),
            DstOperand::Null | DstOperand::None => Ok(self.c_u0),
            DstOperand::Vgpr(v) => Err(SpirvError::UnsupportedOperand {
                pc,
                detail: format!("cannot read VGPR v{v} as scalar"),
            }),
        }
    }

    fn load_raw_scalar_selector(&mut self, pc: u32, code: u8) -> Result<u32, SpirvError> {
        let src = match code {
            0..=105 => SrcOperand::Sgpr(code),
            106 => SrcOperand::VccLo,
            107 => SrcOperand::VccHi,
            124 => SrcOperand::M0,
            125 => SrcOperand::Null,
            126 => SrcOperand::ExecLo,
            127 => SrcOperand::ExecHi,
            128..=192 => SrcOperand::InlineInt(i32::from(code) - 128),
            193..=208 => SrcOperand::InlineInt(-(i32::from(code) - 192)),
            240 => SrcOperand::InlineFloat(0.5),
            241 => SrcOperand::InlineFloat(-0.5),
            242 => SrcOperand::InlineFloat(1.0),
            243 => SrcOperand::InlineFloat(-1.0),
            244 => SrcOperand::InlineFloat(2.0),
            245 => SrcOperand::InlineFloat(-2.0),
            246 => SrcOperand::InlineFloat(4.0),
            247 => SrcOperand::InlineFloat(-4.0),
            other => SrcOperand::Special(u16::from(other)),
        };
        self.load_src_u32(pc, src)
    }

    // -----------------------------------------------------------------------
    // Float input/output modifiers & IEEE helpers
    // -----------------------------------------------------------------------

    fn apply_float_in_mods(&mut self, bits: u32, abs: bool, neg: bool) -> Result<u32, SpirvError> {
        let bits = self.alu().apply_f32_sign_mods(bits, abs, neg)?;
        Ok(self.b.bitcast(self.f32_ty, None, bits)?)
    }

    fn load_src_f32(
        &mut self,
        pc: u32,
        src: SrcOperand,
        mods: ValuModifiers,
        slot: usize,
    ) -> Result<u32, SpirvError> {
        let bits = self.load_src_u32(pc, src)?;
        let abs = mods.abs.get(slot).copied().unwrap_or(false);
        let neg = mods.neg.get(slot).copied().unwrap_or(false);
        self.apply_float_in_mods(bits, abs, neg)
    }

    fn is_f32_nan_bits(&mut self, bits: u32) -> Result<u32, SpirvError> {
        let abs_b = self.b.bitwise_and(self.uint, None, bits, self.c_abs_mask)?;
        Ok(self
            .b
            .u_greater_than(self.bool_ty, None, abs_b, self.c_exp_mask)?)
    }

    fn apply_float_out_mods(
        &mut self,
        mut val: u32,
        mods: ValuModifiers,
    ) -> Result<u32, SpirvError> {
        val = match mods.omod {
            1 => {
                let m = self.b.f_mul(self.f32_ty, None, val, self.c_f2)?;
                self.b.decorate(m, Decoration::NoContraction, []);
                m
            }
            2 => {
                let m = self.b.f_mul(self.f32_ty, None, val, self.c_f4)?;
                self.b.decorate(m, Decoration::NoContraction, []);
                m
            }
            3 => {
                let m = self.b.f_mul(self.f32_ty, None, val, self.c_f05)?;
                self.b.decorate(m, Decoration::NoContraction, []);
                m
            }
            _ => val,
        };
        if mods.clamp {
            // DX10_CLAMP=1 semantics: NaN -> +0.0, <=0.0 -> +0.0, >=1.0 -> 1.0
            let bits = self.b.bitcast(self.uint, None, val)?;
            let is_nan = self.is_f32_nan_bits(bits)?;
            let le_zero = self
                .b
                .f_ord_less_than_equal(self.bool_ty, None, val, self.c_f0)?;
            let ge_one = self
                .b
                .f_ord_greater_than_equal(self.bool_ty, None, val, self.c_f1)?;
            let clamped_hi = self.b.select(self.f32_ty, None, ge_one, self.c_f1, val)?;
            let zero_or_hi = self
                .b
                .select(self.f32_ty, None, le_zero, self.c_f0, clamped_hi)?;
            val = self
                .b
                .select(self.f32_ty, None, is_nan, self.c_f0, zero_or_hi)?;
        }
        Ok(val)
    }

    fn emit_legacy_mul_f32(&mut self, a: u32, rhs: u32) -> Result<u32, SpirvError> {
        let a_zero = self.b.f_ord_equal(self.bool_ty, None, a, self.c_f0)?;
        let b_zero = self.b.f_ord_equal(self.bool_ty, None, rhs, self.c_f0)?;
        let any_zero = self.b.logical_or(self.bool_ty, None, a_zero, b_zero)?;
        let prod = self.b.f_mul(self.f32_ty, None, a, rhs)?;
        self.b.decorate(prod, Decoration::NoContraction, []);
        Ok(self
            .b
            .select(self.f32_ty, None, any_zero, self.c_f0, prod)?)
    }

    /// Map AMD RDNA2 16-bit inline float selectors (240..=248) to IEEE-754 binary16 bit patterns.
    fn inline_float_to_f16_bits(v: f32) -> u16 {
        match v.to_bits() {
            0x3F00_0000 => 0x3800, // 0.5
            0xBF00_0000 => 0xB800, // -0.5
            0x3F80_0000 => 0x3C00, // 1.0
            0xBF80_0000 => 0xBC00, // -1.0
            0x4000_0000 => 0x4000, // 2.0
            0xC000_0000 => 0xC000, // -2.0
            0x4080_0000 => 0x4400, // 4.0
            0xC080_0000 => 0xC400, // -4.0
            _ => 0x3118,           // 1 / (2 * pi)
        }
    }

    /// Bit-exact pure-integer IEEE-754 binary32 -> binary16 conversion with round-toward-zero (RTZ).
    /// Returns a `u32` whose low 16 bits hold the binary16 result and high 16 bits are zero.
    fn emit_f32_to_f16_rtz(&mut self, bits: u32) -> Result<u32, SpirvError> {
        let c_10 = self.const_u32(10);
        let c_13 = self.const_u32(13);
        let c_16 = self.const_u32(16);
        let c_23 = self.const_u32(23);
        let c_103 = self.const_u32(103);
        let c_112 = self.const_u32(112);
        let c_113 = self.const_u32(113);
        let c_126 = self.const_u32(126);
        let c_143 = self.const_u32(143);
        let c_255 = self.const_u32(255);
        let c_sign16 = self.const_u32(0x8000);
        let c_frac_mask = self.const_u32(0x007F_FFFF);
        let c_hidden = self.const_u32(0x0080_0000);
        let c_inf16 = self.const_u32(0x7C00);
        let c_qnan16 = self.const_u32(0x7E00);
        let c_max_fin = self.const_u32(0x7BFF);

        let sign_sh = self.b.shift_right_logical(self.uint, None, bits, c_16)?;
        let sign = self.b.bitwise_and(self.uint, None, sign_sh, c_sign16)?;
        let exp_sh = self.b.shift_right_logical(self.uint, None, bits, c_23)?;
        let exp = self.b.bitwise_and(self.uint, None, exp_sh, c_255)?;
        let frac = self.b.bitwise_and(self.uint, None, bits, c_frac_mask)?;
        let frac10 = self.b.shift_right_logical(self.uint, None, frac, c_13)?;

        // Special (Inf / NaN): exp == 255
        let is_special = self.b.i_equal(self.bool_ty, None, exp, c_255)?;
        let frac_zero = self.b.i_equal(self.bool_ty, None, frac, self.c_u0)?;
        let nan_val = self.b.bitwise_or(self.uint, None, c_qnan16, frac10)?;
        let special_mag = self
            .b
            .select(self.uint, None, frac_zero, c_inf16, nan_val)?;

        // Finite overflow under RTZ: exp >= 143 -> 0x7BFF (65504.0)
        let is_overflow = self
            .b
            .u_greater_than_equal(self.bool_ty, None, exp, c_143)?;

        // Normal: 113 <= exp <= 142
        let is_normal = self
            .b
            .u_greater_than_equal(self.bool_ty, None, exp, c_113)?;
        let exp_unbiased = self.b.i_sub(self.uint, None, exp, c_112)?;
        let exp_shifted = self
            .b
            .shift_left_logical(self.uint, None, exp_unbiased, c_10)?;
        let norm_mag = self.b.bitwise_or(self.uint, None, exp_shifted, frac10)?;

        // Subnormal: 103 <= exp <= 112 (safe shift strictly in 14..=23)
        let ge_103 = self
            .b
            .u_greater_than_equal(self.bool_ty, None, exp, c_103)?;
        let lt_113 = self.b.u_less_than(self.bool_ty, None, exp, c_113)?;
        let is_sub = self.b.logical_and(self.bool_ty, None, ge_103, lt_113)?;
        let raw_sh = self.b.i_sub(self.uint, None, c_126, exp)?;
        let safe_sh = self.b.select(self.uint, None, is_sub, raw_sh, self.c_u0)?;
        let sig = self.b.bitwise_or(self.uint, None, c_hidden, frac)?;
        let sub_mag = self.b.shift_right_logical(self.uint, None, sig, safe_sh)?;

        let sub_or_zero = self.b.select(self.uint, None, is_sub, sub_mag, self.c_u0)?;
        let norm_or_sub = self
            .b
            .select(self.uint, None, is_normal, norm_mag, sub_or_zero)?;
        let fin_mag = self
            .b
            .select(self.uint, None, is_overflow, c_max_fin, norm_or_sub)?;
        let mag = self
            .b
            .select(self.uint, None, is_special, special_mag, fin_mag)?;
        Ok(self.b.bitwise_or(self.uint, None, sign, mag)?)
    }

    // -----------------------------------------------------------------------
    // Instruction dispatch
    // -----------------------------------------------------------------------

    /// Emit one decoded instruction. Returns `true` if the instruction wrote `var_pc`.
    fn emit_inst(
        &mut self,
        inst: &DecodedInst,
        fallthrough: Option<u32>,
        pc_to_block: &BTreeMap<u32, u32>,
    ) -> Result<bool, SpirvError> {
        if matches!(inst.ext, ValuExt::Sdwa { .. }) {
            return Err(SpirvError::UnsupportedValuExt {
                pc: inst.pc,
                detail: "SDWA not yet implemented".into(),
            });
        }

        match inst.encoding {
            Encoding::Sopp => self.emit_sopp(inst, fallthrough, pc_to_block),
            Encoding::Sop1 => {
                self.emit_sop1(inst)?;
                Ok(false)
            }
            Encoding::Sopk => {
                self.emit_sopk(inst)?;
                Ok(false)
            }
            Encoding::Sop2 => {
                self.emit_sop2(inst)?;
                Ok(false)
            }
            Encoding::Sopc => {
                self.emit_sopc(inst)?;
                Ok(false)
            }
            Encoding::Vop1 => {
                self.emit_vop1(inst)?;
                Ok(false)
            }
            Encoding::Vop2 => {
                self.emit_vop2(inst)?;
                Ok(false)
            }
            Encoding::Vopc => {
                self.emit_vopc(inst)?;
                Ok(false)
            }
            Encoding::Vop3 => {
                self.emit_vop3(inst)?;
                Ok(false)
            }
            Encoding::Ds => {
                self.emit_ds(inst)?;
                Ok(false)
            }
            Encoding::Smem => {
                self.emit_smem(inst)?;
                Ok(false)
            }
            Encoding::Vmem => {
                self.emit_vmem(inst)?;
                Ok(false)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sopp(
        &mut self,
        inst: &DecodedInst,
        fallthrough: Option<u32>,
        pc_to_block: &BTreeMap<u32, u32>,
    ) -> Result<bool, SpirvError> {
        match inst.mnemonic.as_str() {
            "s_nop" | "s_waitcnt" | "s_inst_prefetch" | "s_clause" | "s_waitcnt_depctr" => {
                Ok(false)
            }
            "s_barrier" => {
                self.b
                    .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_sem)?;
                Ok(false)
            }
            "s_endpgm" | "s_endpgm_saved" | "s_code_end" => {
                self.b.store(self.var_pc, self.c_halt, None, [])?;
                Ok(true)
            }
            "s_branch" => {
                let target_pc = inst.branch_target.ok_or(SpirvError::InvalidBranchTarget {
                    pc: inst.pc,
                    target: 0,
                })?;
                let target_blk =
                    *pc_to_block
                        .get(&target_pc)
                        .ok_or(SpirvError::InvalidBranchTarget {
                            pc: inst.pc,
                            target: target_pc,
                        })?;
                let c = self.const_u32(target_blk);
                self.b.store(self.var_pc, c, None, [])?;
                Ok(true)
            }
            "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_vccz" | "s_cbranch_vccnz"
            | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let target_pc = inst.branch_target.ok_or(SpirvError::InvalidBranchTarget {
                    pc: inst.pc,
                    target: 0,
                })?;
                let target_blk =
                    *pc_to_block
                        .get(&target_pc)
                        .ok_or(SpirvError::InvalidBranchTarget {
                            pc: inst.pc,
                            target: target_pc,
                        })?;
                let ft_blk = fallthrough.unwrap_or(HALT_PC);
                let target_c = self.const_u32(target_blk);
                let ft_c = self.const_u32(ft_blk);

                let cond = match inst.mnemonic.as_str() {
                    "s_cbranch_scc0" => {
                        let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
                        self.b.i_equal(self.bool_ty, None, scc, self.c_u0)?
                    }
                    "s_cbranch_scc1" => self.load_scc_bool()?,
                    "s_cbranch_vccz" => self.is_mask_zero(self.var_vcc)?,
                    "s_cbranch_vccnz" => {
                        let z = self.is_mask_zero(self.var_vcc)?;
                        self.b.logical_not(self.bool_ty, None, z)?
                    }
                    "s_cbranch_execz" => self.is_mask_zero(self.var_exec)?,
                    "s_cbranch_execnz" => {
                        let z = self.is_mask_zero(self.var_exec)?;
                        self.b.logical_not(self.bool_ty, None, z)?
                    }
                    _ => unreachable!(),
                };
                let next = self.b.select(self.uint, None, cond, target_c, ft_c)?;
                self.b.store(self.var_pc, next, None, [])?;
                Ok(true)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sop1(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let s0 = *inst
            .srcs
            .first()
            .ok_or_else(|| SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "missing SOP1 src0".into(),
            })?;
        match inst.mnemonic.as_str() {
            "s_mov_b32" => {
                let val = self.load_src_u32(inst.pc, s0)?;
                self.store_scalar_dst(inst.pc, inst.dst, val)
            }
            "s_mov_b64" => {
                let (lo, hi) = self.load_scalar_b64(inst.pc, s0)?;
                self.store_mask_pair(inst.pc, inst.dst, lo, Some(hi))
            }
            "s_ff1_i32_b64" => {
                let (lo, hi) = self.load_scalar_b64(inst.pc, s0)?;
                let lo_nz = self.b.i_not_equal(self.bool_ty, None, lo, self.c_u0)?;
                let hi_nz = self.b.i_not_equal(self.bool_ty, None, hi, self.c_u0)?;
                let lo_index = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::FindILsb as u32,
                    [Operand::IdRef(lo)],
                )?;
                let hi_index = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::FindILsb as u32,
                    [Operand::IdRef(hi)],
                )?;
                let hi_index = self.b.i_add(self.uint, None, hi_index, self.c_u32)?;
                let not_found = self.const_u32(u32::MAX);
                let high_or_empty = self.b.select(self.uint, None, hi_nz, hi_index, not_found)?;
                let index = self
                    .b
                    .select(self.uint, None, lo_nz, lo_index, high_or_empty)?;
                self.store_scalar_dst(inst.pc, inst.dst, index)
            }
            "s_cmov_b32" => {
                let val = self.load_src_u32(inst.pc, s0)?;
                let old = self.load_dst_as_src(inst.pc, inst.dst)?;
                let scc = self.load_scc_bool()?;
                let res = self.b.select(self.uint, None, scc, val, old)?;
                self.store_scalar_dst(inst.pc, inst.dst, res)
            }
            "s_not_b32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                let r = self.b.not(self.uint, None, a)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(nz)
            }
            "s_not_b64" => {
                let (a_lo, a_hi) = self.load_scalar_b64(inst.pc, s0)?;
                let r_lo = self.b.not(self.uint, None, a_lo)?;
                let r_hi = self.b.not(self.uint, None, a_hi)?;
                let comb = self.b.bitwise_or(self.uint, None, r_lo, r_hi)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, comb, self.c_u0)?;
                self.store_mask_pair(inst.pc, inst.dst, r_lo, Some(r_hi))?;
                self.store_scc_bool(nz)
            }
            "s_and_saveexec_b32"
            | "s_or_saveexec_b32"
            | "s_xor_saveexec_b32"
            | "s_andn2_saveexec_b32" => {
                let src = self.load_src_u32(inst.pc, s0)?;
                let old_lo = self.load_uvec2_word(self.var_exec, 0)?;
                let new_lo = match inst.mnemonic.as_str() {
                    "s_and_saveexec_b32" => self.b.bitwise_and(self.uint, None, src, old_lo)?,
                    "s_or_saveexec_b32" => self.b.bitwise_or(self.uint, None, src, old_lo)?,
                    "s_xor_saveexec_b32" => self.b.bitwise_xor(self.uint, None, src, old_lo)?,
                    "s_andn2_saveexec_b32" => {
                        let not_old = self.b.not(self.uint, None, old_lo)?;
                        self.b.bitwise_and(self.uint, None, src, not_old)?
                    }
                    _ => unreachable!(),
                };
                self.store_scalar_dst(inst.pc, inst.dst, old_lo)?;
                self.store_uvec2_word(self.var_exec, 0, new_lo)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, new_lo, self.c_u0)?;
                self.store_scc_bool(nz)
            }
            "s_and_saveexec_b64"
            | "s_or_saveexec_b64"
            | "s_xor_saveexec_b64"
            | "s_andn2_saveexec_b64" => {
                let (s_lo, s_hi) = self.load_scalar_b64(inst.pc, s0)?;
                let old_exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
                let old_lo = self.b.composite_extract(self.uint, None, old_exec, [0])?;
                let old_hi = self.b.composite_extract(self.uint, None, old_exec, [1])?;
                let (new_lo, new_hi) = match inst.mnemonic.as_str() {
                    "s_and_saveexec_b64" => (
                        self.b.bitwise_and(self.uint, None, s_lo, old_lo)?,
                        self.b.bitwise_and(self.uint, None, s_hi, old_hi)?,
                    ),
                    "s_or_saveexec_b64" => (
                        self.b.bitwise_or(self.uint, None, s_lo, old_lo)?,
                        self.b.bitwise_or(self.uint, None, s_hi, old_hi)?,
                    ),
                    "s_xor_saveexec_b64" => (
                        self.b.bitwise_xor(self.uint, None, s_lo, old_lo)?,
                        self.b.bitwise_xor(self.uint, None, s_hi, old_hi)?,
                    ),
                    "s_andn2_saveexec_b64" => {
                        let n_lo = self.b.not(self.uint, None, old_lo)?;
                        let n_hi = self.b.not(self.uint, None, old_hi)?;
                        (
                            self.b.bitwise_and(self.uint, None, s_lo, n_lo)?,
                            self.b.bitwise_and(self.uint, None, s_hi, n_hi)?,
                        )
                    }
                    _ => unreachable!(),
                };
                self.store_mask_pair(inst.pc, inst.dst, old_lo, Some(old_hi))?;
                let updated = self
                    .b
                    .composite_construct(self.uvec2, None, [new_lo, new_hi])?;
                self.b.store(self.var_exec, updated, None, [])?;
                let comb = self.b.bitwise_or(self.uint, None, new_lo, new_hi)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, comb, self.c_u0)?;
                self.store_scc_bool(nz)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sopk(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let imm = i32::from(inst.simm16.unwrap_or(0)) as u32;
        let imm_c = self.const_u32(imm);
        match inst.mnemonic.as_str() {
            "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" => {
                Ok(())
            }
            "s_movk_i32" => self.store_scalar_dst(inst.pc, inst.dst, imm_c),
            "s_cmovk_i32" => {
                let old = self.load_dst_as_src(inst.pc, inst.dst)?;
                let scc = self.load_scc_bool()?;
                let res = self.b.select(self.uint, None, scc, imm_c, old)?;
                self.store_scalar_dst(inst.pc, inst.dst, res)
            }
            "s_addk_i32" => {
                let a = self.load_dst_as_src(inst.pc, inst.dst)?;
                let r = self.b.i_add(self.uint, None, a, imm_c)?;
                let ovf = self.signed_add_overflow(a, imm_c, r)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(ovf)
            }
            "s_mulk_i32" => {
                let a = self.load_dst_as_src(inst.pc, inst.dst)?;
                let r = self.b.i_mul(self.uint, None, a, imm_c)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)
            }
            "s_cmpk_eq_i32" | "s_cmpk_lg_i32" | "s_cmpk_gt_i32" | "s_cmpk_ge_i32"
            | "s_cmpk_lt_i32" | "s_cmpk_le_i32" | "s_cmpk_eq_u32" | "s_cmpk_lg_u32" => {
                let a = self.load_dst_as_src(inst.pc, inst.dst)?;
                let cmp_name = inst.mnemonic.replace("s_cmpk_", "s_cmp_");
                let pred = self.eval_scalar_cmp(&cmp_name, a, imm_c, inst.pc)?;
                self.store_scc_bool(pred)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sop2(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.srcs.len() < 2 {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "SOP2 requires 2 sources".into(),
            });
        }

        // 64-bit scalar bitwise / cselect instructions
        match inst.mnemonic.as_str() {
            "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" => {
                let (a_lo, a_hi) = self.load_scalar_b64(inst.pc, inst.srcs[0])?;
                let (b_lo, b_hi) = self.load_scalar_b64(inst.pc, inst.srcs[1])?;
                let (r_lo, r_hi) = match inst.mnemonic.as_str() {
                    "s_and_b64" => (
                        self.b.bitwise_and(self.uint, None, a_lo, b_lo)?,
                        self.b.bitwise_and(self.uint, None, a_hi, b_hi)?,
                    ),
                    "s_or_b64" => (
                        self.b.bitwise_or(self.uint, None, a_lo, b_lo)?,
                        self.b.bitwise_or(self.uint, None, a_hi, b_hi)?,
                    ),
                    "s_xor_b64" => (
                        self.b.bitwise_xor(self.uint, None, a_lo, b_lo)?,
                        self.b.bitwise_xor(self.uint, None, a_hi, b_hi)?,
                    ),
                    "s_andn2_b64" => {
                        let nb_lo = self.b.not(self.uint, None, b_lo)?;
                        let nb_hi = self.b.not(self.uint, None, b_hi)?;
                        (
                            self.b.bitwise_and(self.uint, None, a_lo, nb_lo)?,
                            self.b.bitwise_and(self.uint, None, a_hi, nb_hi)?,
                        )
                    }
                    _ => unreachable!(),
                };
                let comb = self.b.bitwise_or(self.uint, None, r_lo, r_hi)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, comb, self.c_u0)?;
                self.store_mask_pair(inst.pc, inst.dst, r_lo, Some(r_hi))?;
                return self.store_scc_bool(nz);
            }
            "s_cselect_b64" => {
                let (a_lo, a_hi) = self.load_scalar_b64(inst.pc, inst.srcs[0])?;
                let (b_lo, b_hi) = self.load_scalar_b64(inst.pc, inst.srcs[1])?;
                let scc = self.load_scc_bool()?;
                let r_lo = self.b.select(self.uint, None, scc, a_lo, b_lo)?;
                let r_hi = self.b.select(self.uint, None, scc, a_hi, b_hi)?;
                return self.store_mask_pair(inst.pc, inst.dst, r_lo, Some(r_hi));
            }
            _ => {}
        }

        let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
        let b = self.load_src_u32(inst.pc, inst.srcs[1])?;

        match inst.mnemonic.as_str() {
            "s_add_u32" => {
                let r = self.b.i_add(self.uint, None, a, b)?;
                let carry = self.b.u_less_than(self.bool_ty, None, r, a)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(carry)
            }
            "s_sub_u32" => {
                let r = self.b.i_sub(self.uint, None, a, b)?;
                let borrow = self.b.u_less_than(self.bool_ty, None, a, b)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(borrow)
            }
            "s_add_i32" => {
                let r = self.b.i_add(self.uint, None, a, b)?;
                let ovf = self.signed_add_overflow(a, b, r)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(ovf)
            }
            "s_sub_i32" => {
                let r = self.b.i_sub(self.uint, None, a, b)?;
                let ovf = self.signed_sub_overflow(a, b, r)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(ovf)
            }
            "s_addc_u32" => {
                let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
                let sum1 = self.b.i_add(self.uint, None, a, b)?;
                let c1 = self.b.u_less_than(self.bool_ty, None, sum1, a)?;
                let sum2 = self.b.i_add(self.uint, None, sum1, scc)?;
                let c2 = self.b.u_less_than(self.bool_ty, None, sum2, sum1)?;
                let carry = self.b.logical_or(self.bool_ty, None, c1, c2)?;
                self.store_scalar_dst(inst.pc, inst.dst, sum2)?;
                self.store_scc_bool(carry)
            }
            "s_subb_u32" => {
                let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
                let d1 = self.b.i_sub(self.uint, None, a, b)?;
                let b1 = self.b.u_less_than(self.bool_ty, None, a, b)?;
                let d2 = self.b.i_sub(self.uint, None, d1, scc)?;
                let b2 = self.b.u_less_than(self.bool_ty, None, d1, scc)?;
                let borrow = self.b.logical_or(self.bool_ty, None, b1, b2)?;
                self.store_scalar_dst(inst.pc, inst.dst, d2)?;
                self.store_scc_bool(borrow)
            }
            "s_min_i32" | "s_max_i32" | "s_min_u32" | "s_max_u32" => {
                let cond = match inst.mnemonic.as_str() {
                    "s_min_i32" => {
                        let sa = self.b.bitcast(self.sint, None, a)?;
                        let sb = self.b.bitcast(self.sint, None, b)?;
                        self.b.s_less_than(self.bool_ty, None, sa, sb)?
                    }
                    "s_max_i32" => {
                        let sa = self.b.bitcast(self.sint, None, a)?;
                        let sb = self.b.bitcast(self.sint, None, b)?;
                        self.b.s_greater_than(self.bool_ty, None, sa, sb)?
                    }
                    "s_min_u32" => self.b.u_less_than(self.bool_ty, None, a, b)?,
                    "s_max_u32" => self.b.u_greater_than(self.bool_ty, None, a, b)?,
                    _ => unreachable!(),
                };
                let r = self.b.select(self.uint, None, cond, a, b)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(cond)
            }
            "s_cselect_b32" => {
                let scc = self.load_scc_bool()?;
                let r = self.b.select(self.uint, None, scc, a, b)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" => {
                let r = match inst.mnemonic.as_str() {
                    "s_and_b32" => self.b.bitwise_and(self.uint, None, a, b)?,
                    "s_or_b32" => self.b.bitwise_or(self.uint, None, a, b)?,
                    "s_xor_b32" => self.b.bitwise_xor(self.uint, None, a, b)?,
                    "s_andn2_b32" => {
                        let nb = self.b.not(self.uint, None, b)?;
                        self.b.bitwise_and(self.uint, None, a, nb)?
                    }
                    _ => unreachable!(),
                };
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(nz)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let sh = self.b.bitwise_and(self.uint, None, b, self.c_u31)?;
                let r = match inst.mnemonic.as_str() {
                    "s_lshl_b32" => self.b.shift_left_logical(self.uint, None, a, sh)?,
                    "s_lshr_b32" => self.b.shift_right_logical(self.uint, None, a, sh)?,
                    "s_ashr_i32" => {
                        let sa = self.b.bitcast(self.sint, None, a)?;
                        let sr = self.b.shift_right_arithmetic(self.sint, None, sa, sh)?;
                        self.b.bitcast(self.uint, None, sr)?
                    }
                    _ => unreachable!(),
                };
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)?;
                self.store_scc_bool(nz)
            }
            "s_mul_i32" => {
                let r = self.b.i_mul(self.uint, None, a, b)?;
                self.store_scalar_dst(inst.pc, inst.dst, r)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn signed_add_overflow(&mut self, a: u32, b: u32, r: u32) -> Result<u32, SpirvError> {
        let axr = self.b.bitwise_xor(self.uint, None, a, r)?;
        let bxr = self.b.bitwise_xor(self.uint, None, b, r)?;
        let both = self.b.bitwise_and(self.uint, None, axr, bxr)?;
        let sign = self
            .b
            .bitwise_and(self.uint, None, both, self.c_sign_mask)?;
        Ok(self.b.i_not_equal(self.bool_ty, None, sign, self.c_u0)?)
    }

    fn signed_sub_overflow(&mut self, a: u32, b: u32, r: u32) -> Result<u32, SpirvError> {
        let axb = self.b.bitwise_xor(self.uint, None, a, b)?;
        let axr = self.b.bitwise_xor(self.uint, None, a, r)?;
        let both = self.b.bitwise_and(self.uint, None, axb, axr)?;
        let sign = self
            .b
            .bitwise_and(self.uint, None, both, self.c_sign_mask)?;
        Ok(self.b.i_not_equal(self.bool_ty, None, sign, self.c_u0)?)
    }

    fn emit_sopc(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.srcs.len() < 2 {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "SOPC requires 2 sources".into(),
            });
        }
        let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
        let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
        let pred = self.eval_scalar_cmp(&inst.mnemonic, a, b, inst.pc)?;
        self.store_scc_bool(pred)
    }

    fn eval_scalar_cmp(
        &mut self,
        mnemonic: &str,
        a: u32,
        b: u32,
        pc: u32,
    ) -> Result<u32, SpirvError> {
        match mnemonic {
            "s_cmp_eq_i32" | "s_cmp_eq_u32" => Ok(self.b.i_equal(self.bool_ty, None, a, b)?),
            "s_cmp_lg_i32" | "s_cmp_lg_u32" => Ok(self.b.i_not_equal(self.bool_ty, None, a, b)?),
            "s_cmp_gt_u32" => Ok(self.b.u_greater_than(self.bool_ty, None, a, b)?),
            "s_cmp_ge_u32" => Ok(self.b.u_greater_than_equal(self.bool_ty, None, a, b)?),
            "s_cmp_lt_u32" => Ok(self.b.u_less_than(self.bool_ty, None, a, b)?),
            "s_cmp_le_u32" => Ok(self.b.u_less_than_equal(self.bool_ty, None, a, b)?),
            "s_cmp_gt_i32" | "s_cmp_ge_i32" | "s_cmp_lt_i32" | "s_cmp_le_i32" => {
                let sa = self.b.bitcast(self.sint, None, a)?;
                let sb = self.b.bitcast(self.sint, None, b)?;
                match mnemonic {
                    "s_cmp_gt_i32" => Ok(self.b.s_greater_than(self.bool_ty, None, sa, sb)?),
                    "s_cmp_ge_i32" => {
                        Ok(self.b.s_greater_than_equal(self.bool_ty, None, sa, sb)?)
                    }
                    "s_cmp_lt_i32" => Ok(self.b.s_less_than(self.bool_ty, None, sa, sb)?),
                    "s_cmp_le_i32" => Ok(self.b.s_less_than_equal(self.bool_ty, None, sa, sb)?),
                    _ => unreachable!(),
                }
            }
            "s_bitcmp0_b32" | "s_bitcmp1_b32" => {
                let sh = self.b.bitwise_and(self.uint, None, b, self.c_u31)?;
                let bit = self.b.shift_left_logical(self.uint, None, self.c_u1, sh)?;
                let masked = self.b.bitwise_and(self.uint, None, a, bit)?;
                if mnemonic == "s_bitcmp0_b32" {
                    Ok(self.b.i_equal(self.bool_ty, None, masked, self.c_u0)?)
                } else {
                    Ok(self.b.i_not_equal(self.bool_ty, None, masked, self.c_u0)?)
                }
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc,
                mnemonic: mnemonic.to_owned(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // DPP16 & DPP8 preprocessing helper for VOP1 / VOP2
    // -----------------------------------------------------------------------

    /// Resolve optional DPP16/DPP8 extension on `src0`.
    /// Returns `(src0_bits, write_gate_bool, dpp_mods)` where `write_gate_bool` must be ANDed
    /// with `lane_active` when committing the VGPR destination.
    fn resolve_valu_src0_with_dpp(
        &mut self,
        inst: &DecodedInst,
    ) -> Result<(u32, u32, ValuModifiers), SpirvError> {
        let s0 = *inst
            .srcs
            .first()
            .ok_or_else(|| SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "VALU missing src0".into(),
            })?;
        match inst.ext {
            ValuExt::None => Ok((self.load_src_u32(inst.pc, s0)?, self.c_true, inst.modifiers)),
            ValuExt::Dpp16 { word } => {
                let vsrc = (word & 0xFF) as u8;
                let ctrl = ((word >> 8) & 0x1FF) as u16;
                let fi = ((word >> 18) & 1) != 0;
                let bc = ((word >> 19) & 1) != 0;
                let neg0 = ((word >> 20) & 1) != 0;
                let abs0 = ((word >> 21) & 1) != 0;
                let neg1 = ((word >> 22) & 1) != 0;
                let abs1 = ((word >> 23) & 1) != 0;
                let bank_mask = (word >> 24) & 0xF;
                let row_mask = (word >> 28) & 0xF;

                let c_15 = self.const_u32(15);
                let c_inv15 = self.const_u32(!15u32);
                let c_inv3 = self.const_u32(!3u32);
                let r = self.b.bitwise_and(self.uint, None, self.lane_id, c_inv15)?;
                let k = self.b.bitwise_and(self.uint, None, self.lane_id, c_15)?;

                let (safe_j, in_range) = match ctrl {
                    0x000..=0x0FF => {
                        let qbase = self.b.bitwise_and(self.uint, None, self.lane_id, c_inv3)?;
                        let qlane = self
                            .b
                            .bitwise_and(self.uint, None, self.lane_id, self.c_u3)?;
                        let sh = self
                            .b
                            .shift_left_logical(self.uint, None, qlane, self.c_u1)?;
                        let c_ctrl = self.const_u32(u32::from(ctrl));
                        let sel = self
                            .b
                            .bit_field_u_extract(self.uint, None, c_ctrl, sh, self.c_u2)?;
                        let j = self.b.i_add(self.uint, None, qbase, sel)?;
                        (j, self.c_true)
                    }
                    0x101..=0x10F => {
                        let n = self.const_u32(u32::from(ctrl & 0xF));
                        let c_16 = self.const_u32(16);
                        let kn = self.b.i_add(self.uint, None, k, n)?;
                        let ok = self.b.u_less_than(self.bool_ty, None, kn, c_16)?;
                        let raw_j = self.b.i_add(self.uint, None, r, kn)?;
                        let sj = self.b.select(self.uint, None, ok, raw_j, self.lane_id)?;
                        (sj, ok)
                    }
                    0x111..=0x11F => {
                        let n = self.const_u32(u32::from(ctrl & 0xF));
                        let ok = self.b.u_greater_than_equal(self.bool_ty, None, k, n)?;
                        let diff = self.b.i_sub(self.uint, None, k, n)?;
                        let raw_j = self.b.i_add(self.uint, None, r, diff)?;
                        let sj = self.b.select(self.uint, None, ok, raw_j, self.lane_id)?;
                        (sj, ok)
                    }
                    0x121..=0x12F => {
                        let n = self.const_u32(u32::from(ctrl & 0xF));
                        let diff = self.b.i_sub(self.uint, None, k, n)?;
                        let wrap = self.b.bitwise_and(self.uint, None, diff, c_15)?;
                        let j = self.b.i_add(self.uint, None, r, wrap)?;
                        (j, self.c_true)
                    }
                    0x140 => (
                        self.b.bitwise_xor(self.uint, None, self.lane_id, c_15)?,
                        self.c_true,
                    ),
                    0x141 => {
                        let c_7 = self.const_u32(7);
                        (
                            self.b.bitwise_xor(self.uint, None, self.lane_id, c_7)?,
                            self.c_true,
                        )
                    }
                    0x150..=0x15F => {
                        let n = self.const_u32(u32::from(ctrl & 0xF));
                        (self.b.i_add(self.uint, None, r, n)?, self.c_true)
                    }
                    0x160..=0x16F => {
                        let n = self.const_u32(u32::from(ctrl & 0xF));
                        (
                            self.b.bitwise_xor(self.uint, None, self.lane_id, n)?,
                            self.c_true,
                        )
                    }
                    _ => {
                        return Err(SpirvError::UnsupportedValuExt {
                            pc: inst.pc,
                            detail: format!("unsupported DPP16 ctrl 0x{ctrl:03x}"),
                        });
                    }
                };

                let raw_src = self.load_vgpr(vsrc)?;
                let picked = self.gather_from_lane(raw_src, safe_j)?;
                let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
                let src_active = self.eval_mask_bit_for_lane(exec, safe_j)?;
                let src_ok = if fi {
                    in_range
                } else {
                    self.b
                        .logical_and(self.bool_ty, None, in_range, src_active)?
                };
                let eff_src0 = self.b.select(self.uint, None, src_ok, picked, self.c_u0)?;

                // Row & bank destination write masks
                let row_idx =
                    self.b
                        .shift_right_logical(self.uint, None, self.lane_id, self.c_u4)?;
                let row_bit = self
                    .b
                    .shift_left_logical(self.uint, None, self.c_u1, row_idx)?;
                let c_rm = self.const_u32(row_mask);
                let row_m = self.b.bitwise_and(self.uint, None, c_rm, row_bit)?;
                let row_ok = self.b.i_not_equal(self.bool_ty, None, row_m, self.c_u0)?;

                let bank_idx = self.b.bit_field_u_extract(
                    self.uint,
                    None,
                    self.lane_id,
                    self.c_u2,
                    self.c_u2,
                )?;
                let bank_bit = self
                    .b
                    .shift_left_logical(self.uint, None, self.c_u1, bank_idx)?;
                let c_bm = self.const_u32(bank_mask);
                let bank_m = self.b.bitwise_and(self.uint, None, c_bm, bank_bit)?;
                let bank_ok = self.b.i_not_equal(self.bool_ty, None, bank_m, self.c_u0)?;

                let mut gate = self.b.logical_and(self.bool_ty, None, row_ok, bank_ok)?;
                if !bc {
                    gate = self.b.logical_and(self.bool_ty, None, gate, src_ok)?;
                }

                let mut mods = inst.modifiers;
                mods.abs[0] |= abs0;
                mods.neg[0] |= neg0;
                mods.abs[1] |= abs1;
                mods.neg[1] |= neg1;

                Ok((eff_src0, gate, mods))
            }
            ValuExt::Dpp8 {
                word,
                fetch_inactive,
            } => {
                let vsrc = (word & 0xFF) as u8;
                let c_7 = self.const_u32(7);
                let c_inv7 = self.const_u32(!7u32);
                let c_8 = self.const_u32(8);
                let base = self.b.bitwise_and(self.uint, None, self.lane_id, c_inv7)?;
                let sub = self.b.bitwise_and(self.uint, None, self.lane_id, c_7)?;
                let sub3 = self.b.i_mul(self.uint, None, sub, self.c_u3)?;
                let bit_off = self.b.i_add(self.uint, None, c_8, sub3)?;
                let c_word = self.const_u32(word);
                let sel = self
                    .b
                    .bit_field_u_extract(self.uint, None, c_word, bit_off, self.c_u3)?;
                let j = self.b.i_add(self.uint, None, base, sel)?;

                let raw_src = self.load_vgpr(vsrc)?;
                let picked = self.gather_from_lane(raw_src, j)?;
                let eff_src0 = if fetch_inactive {
                    picked
                } else {
                    let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
                    let src_active = self.eval_mask_bit_for_lane(exec, j)?;
                    self.b
                        .select(self.uint, None, src_active, picked, self.c_u0)?
                };
                Ok((eff_src0, self.c_true, inst.modifiers))
            }
            ValuExt::Sdwa { .. } => unreachable!(),
        }
    }

    // -----------------------------------------------------------------------
    // Vector ALU & Cross-lane lowering
    // -----------------------------------------------------------------------

    fn alu(&mut self) -> Alu<'_> {
        Alu {
            b: &mut self.b,
            uint: self.uint,
            sint: self.sint,
            f32_ty: self.f32_ty,
            bool_ty: self.bool_ty,
            glsl: self.glsl,
            c_true: self.c_true,
            c_false: self.c_false,
            u32_consts: &mut self.u32_consts,
            f32_consts: &mut self.f32_consts,
        }
    }

    fn emit_vop1(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.mnemonic == "v_nop" {
            return Ok(());
        }
        if inst.mnemonic == "v_readfirstlane_b32" {
            let s0 = *inst
                .srcs
                .first()
                .ok_or_else(|| SpirvError::UnsupportedOperand {
                    pc: inst.pc,
                    detail: "v_readfirstlane_b32 missing src0".into(),
                })?;
            let src_bits = self.load_src_u32(inst.pc, s0)?;
            let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
            let lo = self.b.composite_extract(self.uint, None, exec, [0])?;
            let lo_nz = self.b.i_not_equal(self.bool_ty, None, lo, self.c_u0)?;
            let lo_lsb = self.b.ext_inst(
                self.uint,
                None,
                self.glsl,
                GLOp::FindILsb as u32,
                [Operand::IdRef(lo)],
            )?;
            let first_lane = match self.opts.wave_size {
                WaveSize::Wave32 => self.b.select(self.uint, None, lo_nz, lo_lsb, self.c_u0)?,
                WaveSize::Wave64 => {
                    let hi = self.b.composite_extract(self.uint, None, exec, [1])?;
                    let hi_nz = self.b.i_not_equal(self.bool_ty, None, hi, self.c_u0)?;
                    let hi_lsb_raw = self.b.ext_inst(
                        self.uint,
                        None,
                        self.glsl,
                        GLOp::FindILsb as u32,
                        [Operand::IdRef(hi)],
                    )?;
                    let hi_lsb = self.b.i_add(self.uint, None, hi_lsb_raw, self.c_u32)?;
                    let from_hi = self.b.select(self.uint, None, hi_nz, hi_lsb, self.c_u0)?;
                    self.b.select(self.uint, None, lo_nz, lo_lsb, from_hi)?
                }
            };
            let picked = self.gather_from_lane(src_bits, first_lane)?;
            return self.store_scalar_dst(inst.pc, inst.dst, picked);
        }

        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: format!("VOP1 expects VGPR dst, got {}", inst.dst),
            });
        };
        let (src0_bits, dpp_gate, mods) = self.resolve_valu_src0_with_dpp(inst)?;
        let res_bits = match inst.mnemonic.as_str() {
            "v_mov_b32" => src0_bits,
            "v_not_b32" => self.b.not(self.uint, None, src0_bits)?,
            "v_bfrev_b32" => self.b.bit_reverse(self.uint, None, src0_bits)?,
            "v_cvt_f32_i32" => {
                let sa = self.b.bitcast(self.sint, None, src0_bits)?;
                let f = self.b.convert_s_to_f(self.f32_ty, None, sa)?;
                let f = self.apply_float_out_mods(f, mods)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_cvt_f32_u32" => {
                let f = self.b.convert_u_to_f(self.f32_ty, None, src0_bits)?;
                let f = self.apply_float_out_mods(f, mods)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_cvt_u32_f32" => {
                let f = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                self.alu().emit_cvt_u32_f32(f)?
            }
            "v_cvt_i32_f32" => {
                let f = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                self.alu().emit_cvt_i32_f32(f)?
            }
            "v_cvt_f16_f32" => {
                let f = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let f = self.apply_float_out_mods(f, mods)?;
                let v2 = self
                    .b
                    .composite_construct(self.vec2, None, [f, self.c_f0])?;
                let packed = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::PackHalf2x16 as u32,
                    [Operand::IdRef(v2)],
                )?;
                let c_lo_mask = self.const_u32(0x0000_FFFF);
                let c_hi_mask = self.const_u32(0xFFFF_0000);
                let lo16 = self.b.bitwise_and(self.uint, None, packed, c_lo_mask)?;
                let old_vdst = self.load_vgpr(vdst)?;
                let old_hi = self.b.bitwise_and(self.uint, None, old_vdst, c_hi_mask)?;
                self.b.bitwise_or(self.uint, None, old_hi, lo16)?
            }
            "v_cvt_f32_f16" => {
                let mut h16 = if matches!(inst.ext, ValuExt::None)
                    && let Some(SrcOperand::InlineFloat(v)) = inst.srcs.first().copied()
                {
                    self.const_u32(u32::from(Self::inline_float_to_f16_bits(v)))
                } else {
                    let c_lo_mask = self.const_u32(0x0000_FFFF);
                    self.b.bitwise_and(self.uint, None, src0_bits, c_lo_mask)?
                };
                if mods.abs[0] {
                    let c_abs16 = self.const_u32(0x7FFF);
                    h16 = self.b.bitwise_and(self.uint, None, h16, c_abs16)?;
                }
                if mods.neg[0] {
                    let c_neg16 = self.const_u32(0x8000);
                    h16 = self.b.bitwise_xor(self.uint, None, h16, c_neg16)?;
                }
                let unpacked = self.b.ext_inst(
                    self.vec2,
                    None,
                    self.glsl,
                    GLOp::UnpackHalf2x16 as u32,
                    [Operand::IdRef(h16)],
                )?;
                let f = self.b.composite_extract(self.f32_ty, None, unpacked, [0])?;
                let f = self.apply_float_out_mods(f, mods)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_rcp_f32" => {
                let f = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let r = self.b.f_div(self.f32_ty, None, self.c_f1, f)?;
                let r = self.apply_float_out_mods(r, mods)?;
                self.b.bitcast(self.uint, None, r)?
            }
            "v_fract_f32" | "v_trunc_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_floor_f32"
            | "v_exp_f32" | "v_log_f32" | "v_sin_f32" | "v_cos_f32" | "v_rsq_f32"
            | "v_sqrt_f32" => {
                let f = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let r = self.alu().emit_unary(inst.mnemonic.as_str(), f)?;
                let r = self.apply_float_out_mods(r, mods)?;
                self.b.bitcast(self.uint, None, r)?
            }
            _ => {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: inst.mnemonic.clone(),
                });
            }
        };
        let active = self.eval_lane_active()?;
        let commit = self.b.logical_and(self.bool_ty, None, active, dpp_gate)?;
        self.store_vgpr_conditional(vdst, commit, res_bits)
    }

    fn emit_add_carry(
        &mut self,
        inst: &DecodedInst,
        src0: u32,
        commit: u32,
    ) -> Result<u32, SpirvError> {
        let src1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
        let sum_carry = self.b.i_add_carry(self.u32_pair_ty, None, src0, src1)?;
        let mut sum = self.b.composite_extract(self.uint, None, sum_carry, [0])?;
        let mut carry = self.b.composite_extract(self.uint, None, sum_carry, [1])?;
        if let Some(&mask) = inst.srcs.get(2) {
            let pair = self.load_wave_mask(inst.pc, mask)?;
            let carry_in = self.eval_mask_bit(pair)?;
            let carry_in = self
                .b
                .select(self.uint, None, carry_in, self.c_u1, self.c_u0)?;
            let sum_carry = self.b.i_add_carry(self.u32_pair_ty, None, sum, carry_in)?;
            sum = self.b.composite_extract(self.uint, None, sum_carry, [0])?;
            let carry_out = self.b.composite_extract(self.uint, None, sum_carry, [1])?;
            carry = self.b.bitwise_or(self.uint, None, carry, carry_out)?;
        }
        let carry_out = self.b.i_not_equal(self.bool_ty, None, carry, self.c_u0)?;
        let vote = self.b.logical_and(self.bool_ty, None, commit, carry_out)?;
        let (lo, hi) = self.ballot_wave(vote)?;
        self.store_mask_pair(inst.pc, inst.carry_dst(), lo, hi)?;
        Ok(sum)
    }

    fn emit_vop2(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: format!("VOP2 expects VGPR dst, got {}", inst.dst),
            });
        };
        let (src0_bits, dpp_gate, mods) = self.resolve_valu_src0_with_dpp(inst)?;
        let active = self.eval_lane_active()?;
        let commit = self.b.logical_and(self.bool_ty, None, active, dpp_gate)?;
        let res_bits = if inst.mnemonic == "v_add_co_ci_u32" {
            self.emit_add_carry(inst, src0_bits, commit)?
        } else {
            self.emit_valu_alu_core(inst, vdst, src0_bits, mods)?
        };
        self.store_vgpr_conditional(vdst, commit, res_bits)
    }

    fn emit_vop3(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.opcode <= 0x0FF {
            return self.emit_vopc(inst);
        }
        if (0x180..=0x1FF).contains(&inst.opcode) {
            return self.emit_vop1(inst);
        }
        match inst.mnemonic.as_str() {
            "v_readlane_b32" => {
                let vbits = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let lane_raw = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let mask = self.const_u32(self.opts.wave_size.width() - 1);
                let lane_clamped = self.b.bitwise_and(self.uint, None, lane_raw, mask)?;
                let picked = self.gather_from_lane(vbits, lane_clamped)?;
                return self.store_scalar_dst(inst.pc, inst.dst, picked);
            }
            "v_writelane_b32" => {
                let DstOperand::Vgpr(vdst) = inst.dst else {
                    return Err(SpirvError::UnsupportedOperand {
                        pc: inst.pc,
                        detail: "v_writelane_b32 expects VGPR dst".into(),
                    });
                };
                let data = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let lane_raw = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let mask = self.const_u32(self.opts.wave_size.width() - 1);
                let target_lane = self.b.bitwise_and(self.uint, None, lane_raw, mask)?;
                let is_target = self
                    .b
                    .i_equal(self.bool_ty, None, self.lane_id, target_lane)?;
                // Ignores EXEC per RDNA2 spec
                return self.store_vgpr_conditional(vdst, is_target, data);
            }
            "v_permlane16_b32" | "v_permlanex16_b32" => {
                let DstOperand::Vgpr(vdst) = inst.dst else {
                    return Err(SpirvError::UnsupportedOperand {
                        pc: inst.pc,
                        detail: "v_permlane16 expects VGPR dst".into(),
                    });
                };
                let (fi, bc) = match inst.extra {
                    InstExtra::Permlane { fi, bc } => (fi, bc),
                    _ => (false, false),
                };
                let src0_bits = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;

                let c_15 = self.const_u32(15);
                let c_inv15 = self.const_u32(!15u32);
                let c_8 = self.const_u32(8);
                let c_7 = self.const_u32(7);
                let c_16 = self.const_u32(16);

                let k = self.b.bitwise_and(self.uint, None, self.lane_id, c_15)?;
                let use_s1 = self.b.u_less_than(self.bool_ty, None, k, c_8)?;
                let sel_word = self.b.select(self.uint, None, use_s1, s1, s2)?;
                let k7 = self.b.bitwise_and(self.uint, None, k, c_7)?;
                let bit_off = self.b.shift_left_logical(self.uint, None, k7, self.c_u2)?;
                let nib = self
                    .b
                    .bit_field_u_extract(self.uint, None, sel_word, bit_off, self.c_u4)?;

                let mut row_base = self.b.bitwise_and(self.uint, None, self.lane_id, c_inv15)?;
                if inst.mnemonic == "v_permlanex16_b32" {
                    row_base = self.b.bitwise_xor(self.uint, None, row_base, c_16)?;
                }
                let j = self.b.i_add(self.uint, None, row_base, nib)?;
                let gathered = self.gather_from_lane(src0_bits, j)?;

                let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
                let lane_active = self.eval_mask_bit(exec)?;
                let src_active = self.eval_mask_bit_for_lane(exec, j)?;

                let (val, commit) = if fi {
                    (gathered, lane_active)
                } else if bc {
                    let v = self
                        .b
                        .select(self.uint, None, src_active, gathered, self.c_u0)?;
                    (v, lane_active)
                } else {
                    let c = self
                        .b
                        .logical_and(self.bool_ty, None, lane_active, src_active)?;
                    (gathered, c)
                };
                return self.store_vgpr_conditional(vdst, commit, val);
            }
            _ => {}
        }

        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: format!("VOP3 expects VGPR dst, got {}", inst.dst),
            });
        };
        let src0_bits = self.load_src_u32(inst.pc, inst.srcs[0])?;
        let res_bits = if matches!(inst.mnemonic.as_str(), "v_add_co_u32" | "v_add_co_ci_u32") {
            let active = self.eval_lane_active()?;
            self.emit_add_carry(inst, src0_bits, active)?
        } else {
            self.emit_valu_alu_core(inst, vdst, src0_bits, inst.modifiers)?
        };
        self.store_vgpr_exec_masked(vdst, res_bits)
    }

    fn emit_guarded_lds_read(
        &mut self,
        cond: u32,
        byte_addr: u32,
        no_ovf: u32,
    ) -> Result<u32, SpirvError> {
        let lds = self.lds_buf.expect("lds_buf allocated");
        let rem = self.b.bitwise_and(self.uint, None, byte_addr, self.c_u3)?;
        let aligned = self.b.i_equal(self.bool_ty, None, rem, self.c_u0)?;
        let word_idx = self
            .b
            .shift_right_logical(self.uint, None, byte_addr, self.c_u2)?;
        let c_lds = self.const_u32(LDS_DWORDS);
        let in_arr = self.b.u_less_than(self.bool_ty, None, word_idx, c_lds)?;
        let g1 = self.b.logical_and(self.bool_ty, None, cond, no_ovf)?;
        let g2 = self.b.logical_and(self.bool_ty, None, aligned, in_arr)?;
        let do_read = self.b.logical_and(self.bool_ty, None, g1, g2)?;

        self.b.store(self.var_tmp0, self.c_u0, None, [])?;
        let read_blk = self.b.id();
        let merge_blk = self.b.id();
        self.b.selection_merge(merge_blk, SelectionControl::NONE)?;
        self.b
            .branch_conditional(do_read, read_blk, merge_blk, [])?;

        self.b.begin_block(Some(read_blk))?;
        let p = self
            .b
            .access_chain(self.ptr_wg_uint, None, lds, [word_idx])?;
        let v = self.b.load(self.uint, None, p, None, [])?;
        self.b.store(self.var_tmp0, v, None, [])?;
        self.b.branch(merge_blk)?;

        self.b.begin_block(Some(merge_blk))?;
        Ok(self.b.load(self.uint, None, self.var_tmp0, None, [])?)
    }

    fn emit_guarded_lds_write(
        &mut self,
        cond: u32,
        byte_addr: u32,
        no_ovf: u32,
        val: u32,
        write: MemoryWrite,
    ) -> Result<(), SpirvError> {
        let lds = self.lds_buf.expect("lds_buf allocated");
        let rem = self.b.bitwise_and(self.uint, None, byte_addr, self.c_u3)?;
        let aligned = self.b.i_equal(self.bool_ty, None, rem, self.c_u0)?;
        let word_idx = self
            .b
            .shift_right_logical(self.uint, None, byte_addr, self.c_u2)?;
        let c_lds = self.const_u32(LDS_DWORDS);
        let in_arr = self.b.u_less_than(self.bool_ty, None, word_idx, c_lds)?;
        let g1 = self.b.logical_and(self.bool_ty, None, cond, no_ovf)?;
        let g2 = self.b.logical_and(self.bool_ty, None, aligned, in_arr)?;
        let do_write = self.b.logical_and(self.bool_ty, None, g1, g2)?;

        if write.result().is_some() {
            self.b.store(self.var_tmp0, self.c_u0, None, [])?;
        }
        let write_blk = self.b.id();
        let merge_blk = self.b.id();
        self.b.selection_merge(merge_blk, SelectionControl::NONE)?;
        self.b
            .branch_conditional(do_write, write_blk, merge_blk, [])?;

        self.b.begin_block(Some(write_blk))?;
        let p = self
            .b
            .access_chain(self.ptr_wg_uint, None, lds, [word_idx])?;
        write.emit(
            &mut self.b,
            self.uint,
            p,
            val,
            Scope::Workgroup,
            self.var_tmp0,
        )?;
        self.b.branch(merge_blk)?;

        self.b.begin_block(Some(merge_blk))?;
        Ok(())
    }

    fn emit_ds(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if let InstExtra::DsLds {
            addr,
            data0,
            data1,
            vdst,
            offset0,
            offset1,
            gds,
            is_write,
            paired,
        } = inst.extra
        {
            if gds {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: format!("{} (GDS unsupported)", inst.mnemonic),
                });
            }
            let base = self.load_vgpr(addr)?;
            let lane_active = self.eval_lane_active()?;
            if !paired {
                let off = u32::from(offset0) | (u32::from(offset1) << 8);
                let c_off = self.const_u32(off);
                let byte_addr = self.b.i_add(self.uint, None, base, c_off)?;
                let no_ovf = self
                    .b
                    .u_greater_than_equal(self.bool_ty, None, byte_addr, base)?;
                if is_write {
                    let val = self.load_vgpr(data0)?;
                    let write = MemoryWrite::for_inst(inst);
                    self.emit_guarded_lds_write(lane_active, byte_addr, no_ovf, val, write)?;
                    if let Some(dst) = write.result() {
                        let old = self.b.load(self.uint, None, self.var_tmp0, None, [])?;
                        self.store_vgpr_conditional(dst, lane_active, old)?;
                    }
                    return Ok(());
                } else {
                    let val = self.emit_guarded_lds_read(lane_active, byte_addr, no_ovf)?;
                    return self.store_vgpr_conditional(vdst, lane_active, val);
                }
            } else {
                let off0 = self.const_u32(u32::from(offset0) * 4);
                let off1 = self.const_u32(u32::from(offset1) * 4);
                let addr0 = self.b.i_add(self.uint, None, base, off0)?;
                let no_ovf0 = self
                    .b
                    .u_greater_than_equal(self.bool_ty, None, addr0, base)?;
                let addr1 = self.b.i_add(self.uint, None, base, off1)?;
                let no_ovf1 = self
                    .b
                    .u_greater_than_equal(self.bool_ty, None, addr1, base)?;
                if is_write {
                    let v0 = self.load_vgpr(data0)?;
                    let v1 = self.load_vgpr(data1)?;
                    self.emit_guarded_lds_write(
                        lane_active,
                        addr0,
                        no_ovf0,
                        v0,
                        MemoryWrite::Store,
                    )?;
                    if offset1 != offset0 {
                        self.emit_guarded_lds_write(
                            lane_active,
                            addr1,
                            no_ovf1,
                            v1,
                            MemoryWrite::Store,
                        )?;
                    }
                    return Ok(());
                } else {
                    let r0 = self.emit_guarded_lds_read(lane_active, addr0, no_ovf0)?;
                    let r1 = if offset1 == offset0 {
                        r0
                    } else {
                        self.emit_guarded_lds_read(lane_active, addr1, no_ovf1)?
                    };
                    self.store_vgpr_conditional(vdst, lane_active, r0)?;
                    return self.store_vgpr_conditional(vdst.wrapping_add(1), lane_active, r1);
                }
            }
        }

        let InstExtra::DsSwizzle { offset, vsrc } = inst.extra else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "ds_swizzle_b32 expects VGPR dst".into(),
            });
        };

        let c_inv31 = self.const_u32(!31u32);
        let h = self.b.bitwise_and(self.uint, None, self.lane_id, c_inv31)?;
        let t = self
            .b
            .bitwise_and(self.uint, None, self.lane_id, self.c_u31)?;
        let j = if offset < 0x8000 {
            let and_m = self.const_u32(u32::from(offset & 0x1F));
            let or_m = self.const_u32(u32::from((offset >> 5) & 0x1F));
            let xor_m = self.const_u32(u32::from((offset >> 10) & 0x1F));
            let p1 = self.b.bitwise_and(self.uint, None, t, and_m)?;
            let p2 = self.b.bitwise_or(self.uint, None, p1, or_m)?;
            let p3 = self.b.bitwise_xor(self.uint, None, p2, xor_m)?;
            self.b.bitwise_or(self.uint, None, h, p3)?
        } else if offset < 0xC000 {
            let c_inv3 = self.const_u32(!3u32);
            let qbase = self.b.bitwise_and(self.uint, None, self.lane_id, c_inv3)?;
            let qlane = self
                .b
                .bitwise_and(self.uint, None, self.lane_id, self.c_u3)?;
            let sh = self
                .b
                .shift_left_logical(self.uint, None, qlane, self.c_u1)?;
            let c_off = self.const_u32(u32::from(offset));
            let sel = self
                .b
                .bit_field_u_extract(self.uint, None, c_off, sh, self.c_u2)?;
            self.b.bitwise_or(self.uint, None, qbase, sel)?
        } else if offset < 0xE000 {
            let m = u32::from(offset & 0x1F);
            let mut d = u32::from((offset >> 5) & 0x1F);
            if ((offset >> 10) & 1) != 0 {
                d = (32 - d) & 0x1F;
            }
            let c_m = self.const_u32(m);
            let c_nm = self.const_u32((!m) & 0x1F);
            let c_d = self.const_u32(d);
            let hi_part = self.b.bitwise_and(self.uint, None, t, c_m)?;
            let td = self.b.i_add(self.uint, None, t, c_d)?;
            let lo_part = self.b.bitwise_and(self.uint, None, td, c_nm)?;
            let sub = self.b.bitwise_or(self.uint, None, hi_part, lo_part)?;
            self.b.bitwise_or(self.uint, None, h, sub)?
        } else {
            let m = u32::from(offset & 0x1F);
            let pop = m.count_ones();
            let c_m = self.const_u32(m);
            let c_27 = self.const_u32(27);
            let c_pop = self.const_u32(pop);
            let rev32 = self.b.bit_reverse(self.uint, None, t)?;
            let rev5 = self.b.shift_right_logical(self.uint, None, rev32, c_27)?;
            let rev_sh = self.b.shift_right_logical(self.uint, None, rev5, c_pop)?;
            let tm = self.b.bitwise_and(self.uint, None, t, c_m)?;
            let sub = self.b.bitwise_or(self.uint, None, rev_sh, tm)?;
            self.b.bitwise_or(self.uint, None, h, sub)?
        };

        let src_bits = self.load_vgpr(vsrc)?;
        let gathered = self.gather_from_lane(src_bits, j)?;
        let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
        let src_active = self.eval_mask_bit_for_lane(exec, j)?;
        let val = self
            .b
            .select(self.uint, None, src_active, gathered, self.c_u0)?;
        self.store_vgpr_exec_masked(vdst, val)
    }

    fn emit_valu_alu_core(
        &mut self,
        inst: &DecodedInst,
        vdst: u8,
        src0_bits: u32,
        mods: ValuModifiers,
    ) -> Result<u32, SpirvError> {
        match inst.mnemonic.as_str() {
            "v_cndmask_b32" => {
                if inst.srcs.len() < 2 {
                    return Err(SpirvError::UnsupportedOperand {
                        pc: inst.pc,
                        detail: "v_cndmask_b32 requires at least 2 sources".into(),
                    });
                }
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let mask_pair = if let Some(&mask_src) = inst.srcs.get(2) {
                    self.load_wave_mask(inst.pc, mask_src)?
                } else {
                    self.b.load(self.uvec2, None, self.var_vcc, None, [])?
                };
                let cond = self.eval_mask_bit(mask_pair)?;
                let a = self
                    .alu()
                    .apply_f32_sign_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let b = self
                    .alu()
                    .apply_f32_sign_mods(s1, mods.abs[1], mods.neg[1])?;
                Ok(self.b.select(self.uint, None, cond, b, a)?)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_min_f32" | "v_max_f32" => {
                if inst.srcs.len() < 2 {
                    return Err(SpirvError::UnsupportedOperand {
                        pc: inst.pc,
                        detail: format!("{} requires 2 sources", inst.mnemonic),
                    });
                }
                let a = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let f = match inst.mnemonic.as_str() {
                    "v_add_f32" => {
                        let r = self.b.f_add(self.f32_ty, None, a, b)?;
                        self.b.decorate(r, Decoration::NoContraction, []);
                        r
                    }
                    "v_sub_f32" => {
                        let r = self.b.f_sub(self.f32_ty, None, a, b)?;
                        self.b.decorate(r, Decoration::NoContraction, []);
                        r
                    }
                    "v_subrev_f32" => {
                        let r = self.b.f_sub(self.f32_ty, None, b, a)?;
                        self.b.decorate(r, Decoration::NoContraction, []);
                        r
                    }
                    "v_mul_f32" => {
                        let r = self.b.f_mul(self.f32_ty, None, a, b)?;
                        self.b.decorate(r, Decoration::NoContraction, []);
                        r
                    }
                    "v_mul_legacy_f32" => self.emit_legacy_mul_f32(a, b)?,
                    "v_min_f32" => self.alu().emit_min_max_f32(a, b, false)?,
                    "v_max_f32" => self.alu().emit_min_max_f32(a, b, true)?,
                    _ => unreachable!(),
                };
                let f = self.apply_float_out_mods(f, mods)?;
                Ok(self.b.bitcast(self.uint, None, f)?)
            }
            "v_mul_f16" if inst.encoding == Encoding::Vop2 => {
                let lhs = if matches!(inst.ext, ValuExt::None)
                    && let SrcOperand::InlineFloat(value) = inst.srcs[0]
                {
                    self.const_u32(u32::from(Self::inline_float_to_f16_bits(value)))
                } else {
                    src0_bits
                };
                let rhs = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let mut unpacked = [0; 2];
                for (slot, bits) in unpacked.iter_mut().zip([lhs, rhs]) {
                    let pair = self.b.ext_inst(
                        self.vec2,
                        None,
                        self.glsl,
                        GLOp::UnpackHalf2x16 as u32,
                        [Operand::IdRef(bits)],
                    )?;
                    *slot = self.b.composite_extract(self.f32_ty, None, pair, [0])?;
                }
                // A product of two binary16 significands fits exactly in binary32.
                let product = self.b.f_mul(self.f32_ty, None, unpacked[0], unpacked[1])?;
                self.b.decorate(product, Decoration::NoContraction, []);
                let pair = self
                    .b
                    .composite_construct(self.vec2, None, [product, self.c_f0])?;
                let low = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::PackHalf2x16 as u32,
                    [Operand::IdRef(pair)],
                )?;
                let old = self.load_vgpr(vdst)?;
                let high_mask = self.const_u32(0xFFFF_0000);
                let high = self.b.bitwise_and(self.uint, None, old, high_mask)?;
                Ok(self.b.bitwise_or(self.uint, None, high, low)?)
            }
            "v_fma_f32" | "v_fmamk_f32" | "v_fmaak_f32" | "v_fmac_f32" => {
                let a = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let (b, c) = if inst.mnemonic == "v_fmac_f32" {
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let old = self.load_vgpr(vdst)?;
                    let c = self.b.bitcast(self.f32_ty, None, old)?;
                    (b, c)
                } else {
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                    (b, c)
                };
                let f = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::Fma as u32,
                    [Operand::IdRef(a), Operand::IdRef(b), Operand::IdRef(c)],
                )?;
                let f = self.apply_float_out_mods(f, mods)?;
                Ok(self.b.bitcast(self.uint, None, f)?)
            }
            "v_fma_legacy_f32" | "v_fmac_legacy_f32" => {
                let a = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let (b, c) = if inst.mnemonic == "v_fmac_legacy_f32" {
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let old = self.load_vgpr(vdst)?;
                    let c = self.b.bitcast(self.f32_ty, None, old)?;
                    (b, c)
                } else {
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                    (b, c)
                };
                let prod = self.emit_legacy_mul_f32(a, b)?;
                let sum = self.b.f_add(self.f32_ty, None, prod, c)?;
                self.b.decorate(sum, Decoration::NoContraction, []);
                let sum = self.apply_float_out_mods(sum, mods)?;
                Ok(self.b.bitcast(self.uint, None, sum)?)
            }
            "v_mac_f32" | "v_mad_f32" | "v_madmk_f32" | "v_madak_f32" => {
                let a = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let (b, c) = if inst.mnemonic == "v_mac_f32" {
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let old = self.load_vgpr(vdst)?;
                    let c = self.b.bitcast(self.f32_ty, None, old)?;
                    (b, c)
                } else {
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                    (b, c)
                };
                let prod = self.b.f_mul(self.f32_ty, None, a, b)?;
                self.b.decorate(prod, Decoration::NoContraction, []);
                let sum = self.b.f_add(self.f32_ty, None, prod, c)?;
                self.b.decorate(sum, Decoration::NoContraction, []);
                let sum = self.apply_float_out_mods(sum, mods)?;
                Ok(self.b.bitcast(self.uint, None, sum)?)
            }
            "v_min3_f32" | "v_max3_f32" | "v_med3_f32" => {
                let a = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                let result = self
                    .alu()
                    .emit_min_max_med3(inst.mnemonic.as_str(), a, b, c)?;
                let result = self.apply_float_out_mods(result, mods)?;
                Ok(self.b.bitcast(self.uint, None, result)?)
            }
            "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_mul_lo_u32"
            | "v_mul_u32_u24" | "v_mul_hi_u32_u24" | "v_and_b32" | "v_or_b32" | "v_xor_b32"
            | "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                match inst.mnemonic.as_str() {
                    "v_add_nc_u32" => Ok(self.b.i_add(self.uint, None, src0_bits, b)?),
                    "v_sub_nc_u32" => Ok(self.b.i_sub(self.uint, None, src0_bits, b)?),
                    "v_subrev_nc_u32" => Ok(self.b.i_sub(self.uint, None, b, src0_bits)?),
                    "v_mul_lo_u32" => Ok(self.b.i_mul(self.uint, None, src0_bits, b)?),
                    "v_mul_u32_u24" | "v_mul_hi_u32_u24" => {
                        let c_24 = self.const_u32(0x00FF_FFFF);
                        let a24 = self.b.bitwise_and(self.uint, None, src0_bits, c_24)?;
                        let b24 = self.b.bitwise_and(self.uint, None, b, c_24)?;
                        if inst.mnemonic == "v_mul_hi_u32_u24" {
                            Ok(self.u64_mul_u32(a24, b24)?.1)
                        } else {
                            Ok(self.b.i_mul(self.uint, None, a24, b24)?)
                        }
                    }
                    "v_and_b32" => Ok(self.b.bitwise_and(self.uint, None, src0_bits, b)?),
                    "v_or_b32" => Ok(self.b.bitwise_or(self.uint, None, src0_bits, b)?),
                    "v_xor_b32" => Ok(self.b.bitwise_xor(self.uint, None, src0_bits, b)?),
                    "v_lshlrev_b32" => {
                        let sh = self.b.bitwise_and(self.uint, None, src0_bits, self.c_u31)?;
                        Ok(self.b.shift_left_logical(self.uint, None, b, sh)?)
                    }
                    "v_lshrrev_b32" => {
                        let sh = self.b.bitwise_and(self.uint, None, src0_bits, self.c_u31)?;
                        Ok(self.b.shift_right_logical(self.uint, None, b, sh)?)
                    }
                    "v_ashrrev_i32" => {
                        let sh = self.b.bitwise_and(self.uint, None, src0_bits, self.c_u31)?;
                        let sb = self.b.bitcast(self.sint, None, b)?;
                        let sr = self.b.shift_right_arithmetic(self.sint, None, sb, sh)?;
                        Ok(self.b.bitcast(self.uint, None, sr)?)
                    }
                    _ => unreachable!(),
                }
            }
            "v_cvt_pkrtz_f16_f32" => {
                let a_f = self.apply_float_in_mods(src0_bits, mods.abs[0], mods.neg[0])?;
                let b_f = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let a_bits = self.b.bitcast(self.uint, None, a_f)?;
                let b_bits = self.b.bitcast(self.uint, None, b_f)?;
                let lo = self.emit_f32_to_f16_rtz(a_bits)?;
                let hi = self.emit_f32_to_f16_rtz(b_bits)?;
                let c_16 = self.const_u32(16);
                let hi_sh = self.b.shift_left_logical(self.uint, None, hi, c_16)?;
                Ok(self.b.bitwise_or(self.uint, None, lo, hi_sh)?)
            }
            "v_mad_u32_u24" => {
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let c_24 = self.const_u32(0x00FF_FFFF);
                let a24 = self.b.bitwise_and(self.uint, None, src0_bits, c_24)?;
                let b24 = self.b.bitwise_and(self.uint, None, s1, c_24)?;
                let prod = self.b.i_mul(self.uint, None, a24, b24)?;
                Ok(self.b.i_add(self.uint, None, prod, s2)?)
            }
            "v_lshl_add_u32" => {
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let sh = self.b.bitwise_and(self.uint, None, s1, self.c_u31)?;
                let shifted = self.b.shift_left_logical(self.uint, None, src0_bits, sh)?;
                Ok(self.b.i_add(self.uint, None, shifted, s2)?)
            }
            "v_add_lshl_u32" => {
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let sum = self.b.i_add(self.uint, None, src0_bits, b)?;
                let shift = self.b.bitwise_and(self.uint, None, c, self.c_u31)?;
                Ok(self.b.shift_left_logical(self.uint, None, sum, shift)?)
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                let count = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let bit = self
                    .b
                    .bitwise_and(self.uint, None, self.lane_id, self.c_u31)?;
                let one = self.b.shift_left_logical(self.uint, None, self.c_u1, bit)?;
                let below = self.b.i_sub(self.uint, None, one, self.c_u1)?;
                let high =
                    self.b
                        .u_greater_than_equal(self.bool_ty, None, self.lane_id, self.c_u32)?;
                let mask = if inst.opcode == 0x365 {
                    let all = self.const_u32(u32::MAX);
                    self.b.select(self.uint, None, high, all, below)?
                } else {
                    self.b.select(self.uint, None, high, below, self.c_u0)?
                };
                let bits = self.b.bitwise_and(self.uint, None, src0_bits, mask)?;
                let bits = self.b.bit_count(self.uint, None, bits)?;
                Ok(self.b.i_add(self.uint, None, count, bits)?)
            }
            "v_add3_u32" => {
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let sum01 = self.b.i_add(self.uint, None, src0_bits, s1)?;
                Ok(self.b.i_add(self.uint, None, sum01, s2)?)
            }
            "v_lshl_or_b32" => {
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let sh = self.b.bitwise_and(self.uint, None, s1, self.c_u31)?;
                let shifted = self.b.shift_left_logical(self.uint, None, src0_bits, sh)?;
                Ok(self.b.bitwise_or(self.uint, None, shifted, s2)?)
            }
            "v_and_or_b32" => {
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let masked = self.b.bitwise_and(self.uint, None, src0_bits, s1)?;
                Ok(self.b.bitwise_or(self.uint, None, masked, s2)?)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let offset = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let count = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let off_m = self.b.bitwise_and(self.uint, None, offset, self.c_u31)?;
                let cnt_raw = self.b.bitwise_and(self.uint, None, count, self.c_u31)?;
                let max_cnt = self.b.i_sub(self.uint, None, self.c_u32, off_m)?;
                let cnt_safe = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::UMin as u32,
                    [Operand::IdRef(cnt_raw), Operand::IdRef(max_cnt)],
                )?;
                if inst.mnemonic == "v_bfe_u32" {
                    Ok(self
                        .b
                        .bit_field_u_extract(self.uint, None, src0_bits, off_m, cnt_safe)?)
                } else {
                    let s_base = self.b.bitcast(self.sint, None, src0_bits)?;
                    let extracted = self
                        .b
                        .bit_field_s_extract(self.sint, None, s_base, off_m, cnt_safe)?;
                    Ok(self.b.bitcast(self.uint, None, extracted)?)
                }
            }
            "v_bfi_b32" => {
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let lhs = self.b.bitwise_and(self.uint, None, src0_bits, s1)?;
                let ns0 = self.b.not(self.uint, None, src0_bits)?;
                let rhs = self.b.bitwise_and(self.uint, None, ns0, s2)?;
                Ok(self.b.bitwise_or(self.uint, None, lhs, rhs)?)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_vopc(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.srcs.len() < 2 {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "VOPC requires 2 sources".into(),
            });
        }
        let op = inst.opcode;
        let pred = match op {
            0x00..=0x1F => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], inst.modifiers, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], inst.modifiers, 1)?;
                self.alu().eval_f32_cmp_pred(op & 0xF, a, b)?
            }
            0x80..=0x87 | 0x90..=0x97 => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                self.alu().eval_int_cmp_pred(op & 7, a, b, true)?
            }
            0xC0..=0xC7 | 0xD0..=0xD7 => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                self.alu().eval_int_cmp_pred(op & 7, a, b, false)?
            }
            _ => {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: inst.mnemonic.clone(),
                });
            }
        };

        let active = self.eval_lane_active()?;
        let vote = self.b.logical_and(self.bool_ty, None, active, pred)?;
        let (lo, hi) = self.ballot_wave(vote)?;
        self.store_mask_pair(inst.pc, inst.dst, lo, hi)
    }

    // -----------------------------------------------------------------------
    // Guarded MemoryBuffer access (binding 2, OpArrayLength-checked)
    // -----------------------------------------------------------------------

    fn emit_guarded_mem_load(
        &mut self,
        cond: u32,
        addr_lo: u32,
        addr_hi: u32,
        width: u32,
    ) -> Result<u32, SpirvError> {
        let mem_buf = self.memory_buf.expect("memory_buf allocated");
        let hi_zero = self.b.i_equal(self.bool_ty, None, addr_hi, self.c_u0)?;
        let alignment_mask = self.const_u32(width - 1);
        let rem = self
            .b
            .bitwise_and(self.uint, None, addr_lo, alignment_mask)?;
        let aligned = self.b.i_equal(self.bool_ty, None, rem, self.c_u0)?;
        let word_idx = self
            .b
            .shift_right_logical(self.uint, None, addr_lo, self.c_u2)?;
        let in_arr = self
            .b
            .u_less_than(self.bool_ty, None, word_idx, self.mem_words_len)?;
        let g1 = self.b.logical_and(self.bool_ty, None, cond, hi_zero)?;
        let g2 = self.b.logical_and(self.bool_ty, None, aligned, in_arr)?;
        let do_load = self.b.logical_and(self.bool_ty, None, g1, g2)?;

        self.b.store(self.var_tmp0, self.c_u0, None, [])?;
        let load_blk = self.b.id();
        let merge_blk = self.b.id();
        self.b.selection_merge(merge_blk, SelectionControl::NONE)?;
        self.b
            .branch_conditional(do_load, load_blk, merge_blk, [])?;

        self.b.begin_block(Some(load_blk))?;
        let p = self
            .b
            .access_chain(self.ptr_sb_uint, None, mem_buf, [self.c_u0, word_idx])?;
        let v = if width == 2 {
            let scope = self.const_u32(Scope::Device as u32);
            let word = self.b.atomic_load(self.uint, None, p, scope, self.c_u0)?;
            let bit = self.halfword_bit_offset(addr_lo)?;
            let c16 = self.const_u32(16);
            self.b
                .bit_field_u_extract(self.uint, None, word, bit, c16)?
        } else {
            self.b.load(self.uint, None, p, None, [])?
        };
        self.b.store(self.var_tmp0, v, None, [])?;
        self.b.branch(merge_blk)?;

        self.b.begin_block(Some(merge_blk))?;
        Ok(self.b.load(self.uint, None, self.var_tmp0, None, [])?)
    }

    fn emit_guarded_mem_store(
        &mut self,
        cond: u32,
        addr_lo: u32,
        addr_hi: u32,
        val: u32,
        write: MemoryWrite,
        width: u32,
    ) -> Result<(), SpirvError> {
        let mem_buf = self.memory_buf.expect("memory_buf allocated");
        let hi_zero = self.b.i_equal(self.bool_ty, None, addr_hi, self.c_u0)?;
        let alignment_mask = self.const_u32(width - 1);
        let rem = self
            .b
            .bitwise_and(self.uint, None, addr_lo, alignment_mask)?;
        let aligned = self.b.i_equal(self.bool_ty, None, rem, self.c_u0)?;
        let word_idx = self
            .b
            .shift_right_logical(self.uint, None, addr_lo, self.c_u2)?;
        let in_arr = self
            .b
            .u_less_than(self.bool_ty, None, word_idx, self.mem_words_len)?;
        let g1 = self.b.logical_and(self.bool_ty, None, cond, hi_zero)?;
        let g2 = self.b.logical_and(self.bool_ty, None, aligned, in_arr)?;
        let do_store = self.b.logical_and(self.bool_ty, None, g1, g2)?;

        if write.result().is_some() {
            self.b.store(self.var_tmp0, self.c_u0, None, [])?;
        }
        let store_blk = self.b.id();
        let merge_blk = self.b.id();
        self.b.selection_merge(merge_blk, SelectionControl::NONE)?;
        self.b
            .branch_conditional(do_store, store_blk, merge_blk, [])?;

        self.b.begin_block(Some(store_blk))?;
        let p = self
            .b
            .access_chain(self.ptr_sb_uint, None, mem_buf, [self.c_u0, word_idx])?;
        if width == 2 {
            self.emit_store_halfword(p, addr_lo, val)?;
        } else {
            write.emit(&mut self.b, self.uint, p, val, Scope::Device, self.var_tmp0)?;
        }
        self.b.branch(merge_blk)?;

        self.b.begin_block(Some(merge_blk))?;
        Ok(())
    }

    fn halfword_bit_offset(&mut self, addr_lo: u32) -> Result<u32, SpirvError> {
        let half = self.b.bitwise_and(self.uint, None, addr_lo, self.c_u2)?;
        Ok(self
            .b
            .shift_left_logical(self.uint, None, half, self.c_u3)?)
    }

    fn emit_store_halfword(&mut self, ptr: u32, addr_lo: u32, val: u32) -> Result<(), SpirvError> {
        let offset = self.halfword_bit_offset(addr_lo)?;
        let count = self.const_u32(16);
        let scope = self.const_u32(Scope::Device as u32);
        let initial = self.b.atomic_load(self.uint, None, ptr, scope, self.c_u0)?;
        self.b.store(self.var_tmp1, initial, None, [])?;

        // Adjacent halfwords may be written by different lanes in the same dispatch.
        let header = self.b.id();
        let retry = self.b.id();
        let merge = self.b.id();
        self.b.branch(header)?;
        self.b.begin_block(Some(header))?;
        let expected = self.b.load(self.uint, None, self.var_tmp1, None, [])?;
        let replacement = self
            .b
            .bit_field_insert(self.uint, None, expected, val, offset, count)?;
        let actual = self.b.atomic_compare_exchange(
            self.uint,
            None,
            ptr,
            scope,
            self.c_u0,
            self.c_u0,
            replacement,
            expected,
        )?;
        let success = self.b.i_equal(self.bool_ty, None, actual, expected)?;
        self.b.loop_merge(merge, retry, LoopControl::NONE, [])?;
        self.b.branch_conditional(success, merge, retry, [])?;

        self.b.begin_block(Some(retry))?;
        self.b.store(self.var_tmp1, actual, None, [])?;
        self.b.branch(header)?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // SMEM (`s_load_dword*`, `s_buffer_load_dword*`)
    // -----------------------------------------------------------------------

    fn emit_smem(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let InstExtra::Smem {
            sbase,
            sdata,
            soffset,
            ioffset,
            dwords,
            is_buffer,
        } = inst.extra
        else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };

        let c_inv3 = self.const_u32(!3u32);
        let soff = if soffset == 0x7D {
            self.c_u0
        } else {
            self.load_raw_scalar_selector(inst.pc, soffset)?
        };
        let soff_al = self.b.bitwise_and(self.uint, None, soff, c_inv3)?;
        let ioff_lo = self.const_u32((ioffset as u32) & !3u32);
        let ioff_hi = self.const_u32(if ioffset < 0 { u32::MAX } else { 0 });

        let (disp_lo, disp_hi) = self.u64_add(soff_al, self.c_u0, ioff_lo, ioff_hi)?;

        let mut loaded = Vec::with_capacity(dwords as usize);
        if !is_buffer {
            let b_lo_raw = self.load_sgpr(sbase)?;
            let b_lo = self.b.bitwise_and(self.uint, None, b_lo_raw, c_inv3)?;
            let b_hi = self.load_sgpr(sbase.wrapping_add(1))?;
            let disp_nonneg = self.b.i_equal(self.bool_ty, None, disp_hi, self.c_u0)?;
            let (addr0_lo, addr0_hi) = self.u64_add(b_lo, b_hi, disp_lo, disp_hi)?;

            for j in 0..dwords {
                let off_j = self.const_u32(u32::from(j) * 4);
                let (a_lo, a_hi) = self.u64_add(addr0_lo, addr0_hi, off_j, self.c_u0)?;
                loaded.push(self.emit_guarded_mem_load(disp_nonneg, a_lo, a_hi, 4)?);
            }
        } else {
            // V# descriptor in s[sbase..sbase+4]
            let rsrc0 = self.load_sgpr(sbase)?;
            let rsrc1 = self.load_sgpr(sbase.wrapping_add(1))?;
            let rsrc2 = self.load_sgpr(sbase.wrapping_add(2))?;
            let c_ffff = self.const_u32(0xFFFF);
            let c_16 = self.const_u32(16);
            let c_3fff = self.const_u32(0x3FFF);
            let b_lo = self.b.bitwise_and(self.uint, None, rsrc0, c_inv3)?;
            let b_hi = self.b.bitwise_and(self.uint, None, rsrc1, c_ffff)?;
            let stride_s = self.b.shift_right_logical(self.uint, None, rsrc1, c_16)?;
            let stride = self.b.bitwise_and(self.uint, None, stride_s, c_3fff)?;
            let stride_z = self.b.i_equal(self.bool_ty, None, stride, self.c_u0)?;
            let eff_stride = self
                .b
                .select(self.uint, None, stride_z, self.c_u1, stride)?;
            let (ext_lo, ext_hi) = self.u64_mul_u32(eff_stride, rsrc2)?;

            let ioff_ok = if ioffset >= 0 {
                self.c_true
            } else {
                self.c_false
            };

            for j in 0..dwords {
                let off_j = self.const_u32(u32::from(j) * 4);
                let (d_lo, d_hi) = self.u64_add(disp_lo, disp_hi, off_j, self.c_u0)?;
                let (end_lo, end_hi) = self.u64_add(d_lo, d_hi, self.c_u4, self.c_u0)?;
                let in_ext = self.u64_le(end_lo, end_hi, ext_lo, ext_hi)?;
                let ok = self.b.logical_and(self.bool_ty, None, ioff_ok, in_ext)?;
                let (a_lo, a_hi) = self.u64_add(b_lo, b_hi, d_lo, d_hi)?;
                loaded.push(self.emit_guarded_mem_load(ok, a_lo, a_hi, 4)?);
            }
        }

        for (j, val) in loaded.into_iter().enumerate() {
            let dst_reg = sdata.wrapping_add(j as u8);
            if dst_reg < 106 {
                self.store_sgpr(dst_reg, val)?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // MUBUF & FLAT/GLOBAL (`buffer_load/store`, `global_load/store`)
    // -----------------------------------------------------------------------

    fn emit_vmem(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let write = MemoryWrite::for_inst(inst);
        match inst.extra {
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
            } => {
                if lds || tfe {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: format!("{} (lds/tfe unsupported)", inst.mnemonic),
                    });
                }
                let width = if matches!(inst.opcode, 0x0A | 0x1A) {
                    2
                } else {
                    4
                };
                self.emit_mubuf(
                    inst.pc, vaddr, vdata, srsrc, soffset, offset, offen, idxen, dwords, is_store,
                    write, width,
                )
            }
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
            } => {
                if lds || (seg != 0 && seg != 2) {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: format!("{} (seg={seg}, lds={lds})", inst.mnemonic),
                    });
                }
                self.emit_flat_global(addr, data, vdst, saddr, offset, dwords, is_store, write)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_mubuf(
        &mut self,
        pc: u32,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
        inst_offset: u16,
        offen: bool,
        idxen: bool,
        dwords: u8,
        is_store: bool,
        write: MemoryWrite,
        width: u32,
    ) -> Result<(), SpirvError> {
        // Snapshot V# descriptor s[srsrc..srsrc+4]
        let rsrc0 = self.load_sgpr(srsrc)?;
        let rsrc1 = self.load_sgpr(srsrc.wrapping_add(1))?;
        let rsrc2 = self.load_sgpr(srsrc.wrapping_add(2))?;
        let rsrc3 = self.load_sgpr(srsrc.wrapping_add(3))?;

        let c_ffff = self.const_u32(0xFFFF);
        let c_16 = self.const_u32(16);
        let c_3fff = self.const_u32(0x3FFF);
        let c_23 = self.const_u32(23);
        let c_28 = self.const_u32(28);
        let c_31 = self.c_u31;

        let base_lo = rsrc0;
        let base_hi = self.b.bitwise_and(self.uint, None, rsrc1, c_ffff)?;
        let stride_s = self.b.shift_right_logical(self.uint, None, rsrc1, c_16)?;
        let stride = self.b.bitwise_and(self.uint, None, stride_s, c_3fff)?;
        let swizzle_bit = self.b.shift_right_logical(self.uint, None, rsrc1, c_31)?;
        let is_swizzled = self
            .b
            .i_not_equal(self.bool_ty, None, swizzle_bit, self.c_u0)?;
        let num_records = rsrc2;
        let add_tid_b = self
            .b
            .bit_field_u_extract(self.uint, None, rsrc3, c_23, self.c_u1)?;
        let add_tid = self
            .b
            .i_not_equal(self.bool_ty, None, add_tid_b, self.c_u0)?;
        let oob_sel = self
            .b
            .bit_field_u_extract(self.uint, None, rsrc3, c_28, self.c_u2)?;

        // Snapshot VGPR inputs & store data before writing any VGPRs
        let vindex = if idxen {
            self.load_vgpr(vaddr)?
        } else {
            self.c_u0
        };
        let voffset = if offen {
            let reg = if idxen { vaddr.wrapping_add(1) } else { vaddr };
            self.load_vgpr(reg)?
        } else {
            self.c_u0
        };
        let mut store_vals = Vec::new();
        if is_store {
            for j in 0..dwords {
                store_vals.push(self.load_vgpr(vdata.wrapping_add(j))?);
            }
        }

        let tid_add = self
            .b
            .select(self.uint, None, add_tid, self.lane_id, self.c_u0)?;
        let (index_lo, index_hi) = self.u64_add(vindex, self.c_u0, tid_add, self.c_u0)?;
        let c_inst_off = self.const_u32(u32::from(inst_offset));
        let (off_lo, off_hi) = self.u64_add(voffset, self.c_u0, c_inst_off, self.c_u0)?;
        let soff = self.load_raw_scalar_selector(pc, soffset)?;

        let c_width = self.const_u32(width);
        let lane_active = self.eval_lane_active()?;
        let (stride_idx_lo, stride_idx_hi) = self.u64_mul_u32(stride, index_lo)?;
        let idx_overflow = self
            .b
            .i_not_equal(self.bool_ty, None, index_hi, self.c_u0)?;
        let idx_ge_num = self
            .b
            .u_greater_than_equal(self.bool_ty, None, index_lo, num_records)?;
        let idx_oob = self
            .b
            .logical_or(self.bool_ty, None, idx_overflow, idx_ge_num)?;
        let num_is_zero = self.b.i_equal(self.bool_ty, None, num_records, self.c_u0)?;
        let stride_nz = self.b.i_not_equal(self.bool_ty, None, stride, self.c_u0)?;

        let is_oob0 = self.b.i_equal(self.bool_ty, None, oob_sel, self.c_u0)?;
        let is_oob1 = self.b.i_equal(self.bool_ty, None, oob_sel, self.c_u1)?;
        let is_oob2 = self.b.i_equal(self.bool_ty, None, oob_sel, self.c_u2)?;
        let swiz_and_stride = self
            .b
            .logical_and(self.bool_ty, None, is_swizzled, stride_nz)?;

        let mut loaded = Vec::new();
        for j in 0..dwords {
            let c_j4 = self.const_u32(u32::from(j) * 4);
            let (o_lo, o_hi) = self.u64_add(off_lo, off_hi, c_j4, self.c_u0)?;
            let o_hi_nz = self.b.i_not_equal(self.bool_ty, None, o_hi, self.c_u0)?;

            // AMD Table 35 OOB checks:
            // 0: index >= num_records || o >= stride
            let o_ge_stride = self
                .b
                .u_greater_than_equal(self.bool_ty, None, o_lo, stride)?;
            let o_oob_stride = self
                .b
                .logical_or(self.bool_ty, None, o_hi_nz, o_ge_stride)?;
            let rej_0 = self
                .b
                .logical_or(self.bool_ty, None, idx_oob, o_oob_stride)?;
            // 1: index >= num_records
            let rej_1 = idx_oob;
            // 2: num_records == 0
            let rej_2 = num_is_zero;
            // 3: linear => soffset + o + width > num_records; swizzled+stride!=0 => index >= num_records || o + width > stride
            let (so_lo, so_hi) = self.u64_add(soff, self.c_u0, o_lo, o_hi)?;
            let (so4_lo, so4_hi) = self.u64_add(so_lo, so_hi, c_width, self.c_u0)?;
            let lin_ok = self.u64_le(so4_lo, so4_hi, num_records, self.c_u0)?;
            let rej_3_lin = self.b.logical_not(self.bool_ty, None, lin_ok)?;

            let (o4_lo, o4_hi) = self.u64_add(o_lo, o_hi, c_width, self.c_u0)?;
            let o4_ok = self.u64_le(o4_lo, o4_hi, stride, self.c_u0)?;
            let o4_oob = self.b.logical_not(self.bool_ty, None, o4_ok)?;
            let rej_3_swiz = self.b.logical_or(self.bool_ty, None, idx_oob, o4_oob)?;
            let rej_3 =
                self.b
                    .select(self.bool_ty, None, swiz_and_stride, rej_3_swiz, rej_3_lin)?;

            let r_23 = self.b.select(self.bool_ty, None, is_oob2, rej_2, rej_3)?;
            let r_123 = self.b.select(self.bool_ty, None, is_oob1, rej_1, r_23)?;
            let rejected = self.b.select(self.bool_ty, None, is_oob0, rej_0, r_123)?;
            let in_bounds = self.b.logical_not(self.bool_ty, None, rejected)?;
            let ok = self
                .b
                .logical_and(self.bool_ty, None, lane_active, in_bounds)?;

            // Effective linear address = base + soff + stride*index + o
            let (a1_lo, a1_hi) = self.u64_add(base_lo, base_hi, soff, self.c_u0)?;
            let (a2_lo, a2_hi) = self.u64_add(a1_lo, a1_hi, stride_idx_lo, stride_idx_hi)?;
            let (a_lo, a_hi) = self.u64_add(a2_lo, a2_hi, o_lo, o_hi)?;

            if is_store {
                self.emit_guarded_mem_store(ok, a_lo, a_hi, store_vals[j as usize], write, width)?;
            } else {
                loaded.push(self.emit_guarded_mem_load(ok, a_lo, a_hi, width)?);
            }
        }

        if let Some(dst) = write.result() {
            let old = self.b.load(self.uint, None, self.var_tmp0, None, [])?;
            self.store_vgpr_conditional(dst, lane_active, old)?;
        }
        if !is_store {
            for (j, val) in loaded.into_iter().enumerate() {
                self.store_vgpr_conditional(vdata.wrapping_add(j as u8), lane_active, val)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_flat_global(
        &mut self,
        addr: u8,
        data: u8,
        vdst: u8,
        saddr: u8,
        offset: i16,
        dwords: u8,
        is_store: bool,
        write: MemoryWrite,
    ) -> Result<(), SpirvError> {
        let (base_lo, base_hi) = if saddr == 0x7D {
            (self.load_vgpr(addr)?, self.load_vgpr(addr.wrapping_add(1))?)
        } else {
            let s_lo = self.load_sgpr(saddr)?;
            let s_hi = self.load_sgpr(saddr.wrapping_add(1))?;
            let v_off = self.load_vgpr(addr)?;
            self.u64_add(s_lo, s_hi, v_off, self.c_u0)?
        };

        let off_lo = self.const_u32(i32::from(offset) as u32);
        let off_hi = self.const_u32(if offset < 0 { u32::MAX } else { 0 });
        let (addr0_lo, addr0_hi) = self.u64_add(base_lo, base_hi, off_lo, off_hi)?;

        let mut store_vals = Vec::new();
        if is_store {
            for j in 0..dwords {
                store_vals.push(self.load_vgpr(data.wrapping_add(j))?);
            }
        }

        let lane_active = self.eval_lane_active()?;
        let mut loaded = Vec::new();
        for j in 0..dwords {
            let c_j4 = self.const_u32(u32::from(j) * 4);
            let (a_lo, a_hi) = self.u64_add(addr0_lo, addr0_hi, c_j4, self.c_u0)?;
            if is_store {
                self.emit_guarded_mem_store(
                    lane_active,
                    a_lo,
                    a_hi,
                    store_vals[j as usize],
                    write,
                    4,
                )?;
            } else {
                loaded.push(self.emit_guarded_mem_load(lane_active, a_lo, a_hi, 4)?);
            }
        }

        if let Some(dst) = write.result() {
            let old = self.b.load(self.uint, None, self.var_tmp0, None, [])?;
            self.store_vgpr_conditional(dst, lane_active, old)?;
        }
        if !is_store {
            for (j, val) in loaded.into_iter().enumerate() {
                self.store_vgpr_conditional(vdst.wrapping_add(j as u8), lane_active, val)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::build_synthetic_console_dxbc;
    use crate::decode::disassemble;
    use crate::encode as asm;
    use x2l_pipeline::Cache;

    fn assert_valid_spirv(words: &[u32]) {
        assert!(words.len() >= 5);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
        assert_eq!(words[1], 0x0001_0600, "SPIR-V 1.6 version");

        #[cfg(feature = "spirv-val")]
        {
            validate_spirv_vulkan13(words)
                .expect("spirv-val --target-env vulkan1.3 must succeed when feature enabled");
        }
        #[cfg(not(feature = "spirv-val"))]
        {
            if is_spirv_val_available() {
                validate_spirv_vulkan13(words).expect("spirv-val --target-env vulkan1.3 failed");
            }
        }
    }

    #[test]
    fn compiles_and_validates_scalar_alu_and_loop_branch_in_both_modes() {
        let mut a = asm::Asm::new();
        a.push(asm::s_movk_i32(0, 4));
        a.push(asm::s_mov_b32(1, asm::inline_i(0)));
        let loop_pc = a.pc();
        a.push(asm::s_add_u32(1, 1, 0));
        a.push(asm::s_sub_u32(0, 0, asm::inline_i(1)));
        a.push(asm::s_cmp_lg_u32(0, asm::inline_i(0)));
        a.branch_to_pc(0x05, loop_pc);
        a.push_lit(asm::s_and_b32(2, 1, asm::SRC_LIT), 0xFF);
        a.push(asm::s_or_b32(3, 2, asm::inline_i(2)));
        a.push(asm::s_xor_b32(4, 3, asm::inline_i(1)));
        a.push(asm::s_cbranch_scc0(1));
        a.push(asm::s_waitcnt(0));
        a.push(asm::s_endpgm());

        let insts = disassemble(a.words()).expect("disassemble");
        for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
            for cross_lane_mode in [CrossLaneMode::WorkgroupShared, CrossLaneMode::Subgroup] {
                let spv = compile_compute_spv_with_options(
                    &insts,
                    CompileOptions {
                        wave_size,
                        cross_lane_mode,
                        strict_float_controls: true,
                        denorm_preserve: true,
                    },
                )
                .expect("compile");
                assert_valid_spirv(&spv);
            }
        }
    }

    #[test]
    fn compiles_and_validates_vector_cross_lane_dpp_and_memory_in_both_modes() {
        let mut a = asm::Asm::new();
        a.push(asm::v_mov_b32(0, asm::F_1_0));
        a.push(asm::v_mul_legacy_f32(1, asm::inline_i(0), 0));
        a.push2(asm::v_fma_f32(2, asm::v(0), asm::v(1), asm::v(0)));
        a.push2(asm::v_mad_f32(3, asm::v(0), asm::v(1), asm::v(2)));
        a.push(asm::v_cmp_gt_f32(asm::v(0), 1));
        a.push(asm::v_cndmask_b32(4, asm::v(0), 1));
        a.push(asm::v_readfirstlane_b32(8, 0));
        a.push2(asm::v_readlane_b32(9, 0, asm::inline_i(3)));
        a.push2(asm::v_writelane_b32(5, asm::s(9), asm::inline_i(2)));
        a.push2(asm::ds_swizzle_b32(6, 0, 0x801B));
        a.push2(asm::v_permlane16(
            false,
            7,
            0,
            asm::s(2),
            asm::s(3),
            true,
            false,
        ));
        a.push_lit(
            asm::v_mov_b32(8, asm::SRC_DPP16),
            asm::dpp16_word(0, 0x1B, true, true, 0xF, 0xF),
        );
        a.push2(asm::s_load_dwordx4(12, 0, 0x7D, 0));
        a.push2(asm::s_buffer_load_dwordx2(16, 4, 0x7D, 0));
        a.push2(asm::buffer_load_dwordx4(
            10,
            0,
            4,
            asm::inline_i(0),
            0,
            true,
            false,
        ));
        a.push2(asm::buffer_store_dword(
            10,
            0,
            4,
            asm::inline_i(0),
            16,
            true,
            false,
        ));
        a.push2(asm::global_load_dword(14, 0, 0, 0));
        a.push2(asm::global_store_dword(0, 14, 0, 4));
        a.push(asm::s_endpgm());

        let insts = disassemble(a.words()).expect("disassemble");
        for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
            for cross_lane_mode in [CrossLaneMode::WorkgroupShared, CrossLaneMode::Subgroup] {
                let spv = compile_compute_spv_with_options(
                    &insts,
                    CompileOptions {
                        wave_size,
                        cross_lane_mode,
                        strict_float_controls: true,
                        denorm_preserve: false,
                    },
                )
                .expect("compile full M2.2 ISA");
                assert_valid_spirv(&spv);
            }
        }
    }

    #[test]
    fn pipeline_cache_roundtrips_synthetic_console_container_to_valid_spirv() {
        let code = [asm::v_mul_legacy_f32(0, asm::v(0), 1), asm::s_endpgm()];
        let container = build_synthetic_console_dxbc(5, 0x60, true, &code);

        let cache_dir =
            std::env::temp_dir().join(format!("x2l_spv_cache_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&cache_dir);
        let cache = Cache::open(cache_dir.clone()).expect("open cache");
        let conv = Rdna2SpirvConverter;

        let spv_path_1 = cache
            .get_or_convert(&conv, &container)
            .expect("compile & cache");
        let spv_path_2 = cache.get_or_convert(&conv, &container).expect("cache hit");
        assert_eq!(spv_path_1, spv_path_2);

        let spv_bytes = fs::read(&spv_path_1).expect("read cached spv");
        let words = spirv_bytes_to_words(&spv_bytes).expect("aligned words");
        assert_valid_spirv(&words);

        let _ = fs::remove_dir_all(&cache_dir);
    }

    #[test]
    fn rejects_empty_program_and_invalid_branch_target() {
        assert!(matches!(
            compile_compute_spv(&[]),
            Err(SpirvError::EmptyProgram)
        ));

        let bad = [asm::s_branch(10), asm::s_endpgm()];
        let insts = disassemble(&bad).expect("disassemble");
        assert!(matches!(
            compile_compute_spv(&insts),
            Err(SpirvError::InvalidBranchTarget { .. })
        ));
    }
}
