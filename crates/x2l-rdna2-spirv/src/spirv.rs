// Clean-room direct binary SPIR-V 1.6 compute lowering for X2Linux.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Phase-1 direct `RDNA2 -> Binary SPIR-V 1.6` compute shader emitter (`rspirv`).
//!
//! Emits Vulkan 1.3-compliant binary SPIR-V 1.6 directly via `rspirv` without HLSL text,
//! DXIL, DXC, or `vkd3d-proton`. Uses a single Vulkan subgroup per guest wave (Wave32 or
//! Wave64), replicated scalar state (`SGPR[128]`, `EXEC`, `VCC`, `SCC`), per-lane `VGPR[256]`,
//! and a structured `OpLoopMerge` + `OpSwitch` basic-block dispatcher.

use crate::container::parse_container;
use crate::decode::{
    DecodedInst, DstOperand, Encoding, SrcOperand, ValuExt, ValuModifiers, disassemble_reachable,
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

/// Byte stride of one `Wave` record in the Vulkan `StorageBuffer` ABI (`66_072` bytes).
///
/// Layout:
/// - offset `0`: `sgpr: [u32; 128]` (512 bytes)
/// - offset `512`: `vgpr: [[u32; 256]; 64]` lane-first (65,536 bytes)
/// - offset `66_048`: `exec: [u32; 2]` (`uvec2`, low word first)
/// - offset `66_056`: `vcc: [u32; 2]` (`uvec2`, low word first)
/// - offset `66_064`: `scc: u32` (canonical 0 or 1, +4 trailing pad bytes for 8-byte alignment)
pub const WAVE_STATE_STRIDE: u32 = 66_072;

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

/// Configuration for Phase-1 SPIR-V 1.6 compute lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileOptions {
    pub wave_size: WaveSize,
    /// Emit IEEE binary32 float-control execution modes (`RoundingModeRTE`, `DenormPreserve`,
    /// `SignedZeroInfNanPreserve`).
    pub strict_float_controls: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            wave_size: WaveSize::Wave32,
            strict_float_controls: true,
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
    #[error("unsupported VALU SDWA/DPP extension at PC 0x{pc:04x}")]
    UnsupportedValuExt { pc: u32 },
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
/// using default options (`Wave32`, strict binary32 float controls).
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

    let blocks = build_basic_blocks(insts)?;
    let mut ctx = EmitCtx::new(opts);
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
        1
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
                strict_float_controls: true,
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
    // Scalar/vector types
    void: u32,
    bool_ty: u32,
    uint: u32,
    sint: u32,
    f32_ty: u32,
    uvec2: u32,
    uvec3: u32,
    uvec4: u32,
    // Pointer types
    ptr_sb_uint: u32,
    ptr_sb_uvec2: u32,
    ptr_fn_uint: u32,
    ptr_fn_uvec2: u32,
    ptr_fn_sgprs: u32,
    ptr_fn_vgprs: u32,
    // Global variables
    state_in: u32,
    state_out: u32,
    subgroup_lane: u32,
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
    c_subgroup_scope: u32,
    c_wg_scope: u32,
    c_wg_sem: u32,
    c_true: u32,
    c_false: u32,
    u32_consts: BTreeMap<u32, u32>,
    // Function-local variables (initialized in emit_module)
    var_pc: u32,
    var_sgprs: u32,
    var_vgprs: u32,
    var_exec: u32,
    var_vcc: u32,
    var_scc: u32,
    var_idx: u32,
    lane_id: u32,
}

impl EmitCtx {
    fn new(opts: CompileOptions) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 6);
        b.capability(Capability::Shader);
        b.capability(Capability::GroupNonUniform);
        b.capability(Capability::GroupNonUniformBallot);
        if opts.strict_float_controls {
            b.capability(Capability::RoundingModeRTE);
            b.capability(Capability::DenormPreserve);
            b.capability(Capability::SignedZeroInfNanPreserve);
        }

        let glsl = b.ext_inst_import("GLSL.std.450");
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

        let void = b.type_void();
        let bool_ty = b.type_bool();
        let uint = b.type_int(32, 0);
        let sint = b.type_int(32, 1);
        let f32_ty = b.type_float(32);
        let uvec2 = b.type_vector(uint, 2);
        let uvec3 = b.type_vector(uint, 3);
        let uvec4 = b.type_vector(uint, 4);

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
        let c_subgroup_scope = b.constant_bit32(uint, Scope::Subgroup as u32);
        let c_wg_scope = b.constant_bit32(uint, Scope::Workgroup as u32);
        let c_wg_sem = b.constant_bit32(
            uint,
            (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits(),
        );
        let c_true = b.constant_true(bool_ty);
        let c_false = b.constant_false(bool_ty);

        // StorageBuffer Wave ABI block
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

        // Input built-ins
        let input_sc = StorageClass::Input;
        let ptr_in_uint = b.type_pointer(None, input_sc, uint);
        let ptr_in_uvec3 = b.type_pointer(None, input_sc, uvec3);
        let subgroup_lane = b.variable(ptr_in_uint, None, input_sc, None);
        let workgroup_id = b.variable(ptr_in_uvec3, None, input_sc, None);
        b.decorate(
            subgroup_lane,
            Decoration::BuiltIn,
            [Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId)],
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
            uvec2,
            uvec3,
            uvec4,
            ptr_sb_uint,
            ptr_sb_uvec2,
            ptr_fn_uint,
            ptr_fn_uvec2,
            ptr_fn_sgprs,
            ptr_fn_vgprs,
            state_in,
            state_out,
            subgroup_lane,
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
            c_subgroup_scope,
            c_wg_scope,
            c_wg_sem,
            c_true,
            c_false,
            u32_consts,
            var_pc: 0,
            var_sgprs: 0,
            var_vgprs: 0,
            var_exec: 0,
            var_vcc: 0,
            var_scc: 0,
            var_idx: 0,
            lane_id: 0,
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

        let wg_val = self.b.load(self.uvec3, None, self.workgroup_id, None, [])?;
        let wave_id = self.b.composite_extract(self.uint, None, wg_val, [0])?;
        self.lane_id = self.b.load(self.uint, None, self.subgroup_lane, None, [])?;

        // Load initial SGPR[0..128]
        self.emit_copy_sgprs_from_sb(wave_id)?;
        // Load initial VGPR[lane][0..256]
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

        // Start at basic block 0
        self.b.store(self.var_pc, self.c_u0, None, [])?;

        // Main structured CFG dispatcher
        let loop_header = self.b.id();
        let switch_header = self.b.id();
        let switch_merge = self.b.id();
        let loop_continue = self.b.id();
        let loop_exit = self.b.id();
        let default_label = self.b.id();
        let case_labels: Vec<(u32, u32)> = blocks.iter().map(|bb| (bb.id, self.b.id())).collect();

        self.b.branch(loop_header)?;

        // loop_header
        self.b.begin_block(Some(loop_header))?;
        let cur_pc = self.b.load(self.uint, None, self.var_pc, None, [])?;
        let done = self.b.i_equal(self.bool_ty, None, cur_pc, self.c_halt)?;
        self.b
            .loop_merge(loop_exit, loop_continue, LoopControl::NONE, [])?;
        self.b
            .branch_conditional(done, loop_exit, switch_header, [])?;

        // switch_header
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

        // Basic block cases
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

        // default case
        self.b.begin_block(Some(default_label))?;
        self.b.store(self.var_pc, self.c_halt, None, [])?;
        self.b.branch(switch_merge)?;

        // switch_merge -> loop_continue -> loop_header
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

        self.b.entry_point(
            ExecutionModel::GLCompute,
            entry,
            "main",
            [
                self.state_in,
                self.state_out,
                self.subgroup_lane,
                self.workgroup_id,
            ],
        );
        self.b.execution_mode(
            entry,
            ExecutionMode::LocalSize,
            [self.opts.wave_size.width(), 1, 1],
        );
        if self.opts.strict_float_controls {
            self.b
                .execution_mode(entry, ExecutionMode::RoundingModeRTE, [32]);
            self.b
                .execution_mode(entry, ExecutionMode::DenormPreserve, [32]);
            self.b
                .execution_mode(entry, ExecutionMode::SignedZeroInfNanPreserve, [32]);
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

    fn store_vgpr_exec_masked(&mut self, idx: u8, new_val: u32) -> Result<(), SpirvError> {
        let active = self.eval_lane_active()?;
        let old_val = self.load_vgpr(idx)?;
        let merged = self.b.select(self.uint, None, active, new_val, old_val)?;
        let idx_c = self.const_u32(u32::from(idx));
        let p = self
            .b
            .access_chain(self.ptr_fn_uint, None, self.var_vgprs, [idx_c])?;
        self.b.store(p, merged, None, [])?;
        Ok(())
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

    fn eval_mask_bit(&mut self, mask_uvec2: u32) -> Result<u32, SpirvError> {
        let lo = self.b.composite_extract(self.uint, None, mask_uvec2, [0])?;
        let word = match self.opts.wave_size {
            WaveSize::Wave32 => lo,
            WaveSize::Wave64 => {
                let hi = self.b.composite_extract(self.uint, None, mask_uvec2, [1])?;
                let is_hi =
                    self.b
                        .u_greater_than_equal(self.bool_ty, None, self.lane_id, self.c_u32)?;
                self.b.select(self.uint, None, is_hi, hi, lo)?
            }
        };
        let shift = self
            .b
            .bitwise_and(self.uint, None, self.lane_id, self.c_u31)?;
        let bit = self
            .b
            .shift_left_logical(self.uint, None, self.c_u1, shift)?;
        let masked = self.b.bitwise_and(self.uint, None, word, bit)?;
        Ok(self.b.i_not_equal(self.bool_ty, None, masked, self.c_u0)?)
    }

    fn eval_lane_active(&mut self) -> Result<u32, SpirvError> {
        let exec = self.b.load(self.uvec2, None, self.var_exec, None, [])?;
        self.eval_mask_bit(exec)
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

    fn load_src_f32(
        &mut self,
        pc: u32,
        src: SrcOperand,
        mods: ValuModifiers,
        slot: usize,
    ) -> Result<u32, SpirvError> {
        let mut bits = self.load_src_u32(pc, src)?;
        if mods.abs.get(slot).copied().unwrap_or(false) {
            bits = self.b.bitwise_and(self.uint, None, bits, self.c_abs_mask)?;
        }
        if mods.neg.get(slot).copied().unwrap_or(false) {
            bits = self
                .b
                .bitwise_xor(self.uint, None, bits, self.c_sign_mask)?;
        }
        Ok(self.b.bitcast(self.f32_ty, None, bits)?)
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
            val = self.b.ext_inst(
                self.f32_ty,
                None,
                self.glsl,
                GLOp::FClamp as u32,
                [
                    Operand::IdRef(val),
                    Operand::IdRef(self.c_f0),
                    Operand::IdRef(self.c_f1),
                ],
            )?;
        }
        Ok(val)
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

    fn load_mask_pair(&mut self, pc: u32, src: SrcOperand) -> Result<u32, SpirvError> {
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
                            detail: format!("wave64 mask destination s{n} out of range"),
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

    // -----------------------------------------------------------------------
    // Instruction lowering
    // -----------------------------------------------------------------------

    /// Emit one decoded instruction. Returns `true` if the instruction wrote `var_pc`.
    fn emit_inst(
        &mut self,
        inst: &DecodedInst,
        fallthrough: Option<u32>,
        pc_to_block: &BTreeMap<u32, u32>,
    ) -> Result<bool, SpirvError> {
        if inst.ext != ValuExt::None {
            return Err(SpirvError::UnsupportedValuExt { pc: inst.pc });
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
            "s_nop" | "s_waitcnt" => Ok(false),
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
            "s_and_saveexec_b32" | "s_or_saveexec_b32" => {
                let src = self.load_src_u32(inst.pc, s0)?;
                let old_exec_lo = self.load_uvec2_word(self.var_exec, 0)?;
                let new_exec_lo = if inst.mnemonic == "s_and_saveexec_b32" {
                    self.b.bitwise_and(self.uint, None, src, old_exec_lo)?
                } else {
                    self.b.bitwise_or(self.uint, None, src, old_exec_lo)?
                };
                self.store_scalar_dst(inst.pc, inst.dst, old_exec_lo)?;
                self.store_uvec2_word(self.var_exec, 0, new_exec_lo)?;
                let nz = self
                    .b
                    .i_not_equal(self.bool_ty, None, new_exec_lo, self.c_u0)?;
                self.store_scc_bool(nz)
            }
            "s_mov_b64" => {
                let pair = self.load_mask_pair(inst.pc, s0)?;
                let lo = self.b.composite_extract(self.uint, None, pair, [0])?;
                let hi = self.b.composite_extract(self.uint, None, pair, [1])?;
                self.store_mask_pair(inst.pc, inst.dst, lo, Some(hi))
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
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

    fn emit_sopk(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let imm = i32::from(inst.simm16.unwrap_or(0)) as u32;
        let imm_c = self.const_u32(imm);
        match inst.mnemonic.as_str() {
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

    fn emit_vop1(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.mnemonic == "v_nop" {
            return Ok(());
        }
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: format!("VOP1 expects VGPR dst, got {}", inst.dst),
            });
        };
        let s0 = *inst
            .srcs
            .first()
            .ok_or_else(|| SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: "VOP1 missing src0".into(),
            })?;
        let res_bits = match inst.mnemonic.as_str() {
            "v_mov_b32" => self.load_src_u32(inst.pc, s0)?,
            "v_not_b32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                self.b.not(self.uint, None, a)?
            }
            "v_cvt_f32_i32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                let sa = self.b.bitcast(self.sint, None, a)?;
                let f = self.b.convert_s_to_f(self.f32_ty, None, sa)?;
                let f = self.apply_float_out_mods(f, inst.modifiers)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_cvt_f32_u32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                let f = self.b.convert_u_to_f(self.f32_ty, None, a)?;
                let f = self.apply_float_out_mods(f, inst.modifiers)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_cvt_u32_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                self.b.convert_f_to_u(self.uint, None, f)?
            }
            "v_cvt_i32_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let s = self.b.convert_f_to_s(self.sint, None, f)?;
                self.b.bitcast(self.uint, None, s)?
            }
            "v_rcp_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let r = self.b.f_div(self.f32_ty, None, self.c_f1, f)?;
                let r = self.apply_float_out_mods(r, inst.modifiers)?;
                self.b.bitcast(self.uint, None, r)?
            }
            "v_rsq_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let r = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::InverseSqrt as u32,
                    [Operand::IdRef(f)],
                )?;
                let r = self.apply_float_out_mods(r, inst.modifiers)?;
                self.b.bitcast(self.uint, None, r)?
            }
            "v_sqrt_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let r = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::Sqrt as u32,
                    [Operand::IdRef(f)],
                )?;
                let r = self.apply_float_out_mods(r, inst.modifiers)?;
                self.b.bitcast(self.uint, None, r)?
            }
            _ => {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: inst.mnemonic.clone(),
                });
            }
        };
        self.store_vgpr_exec_masked(vdst, res_bits)
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

    fn emit_vop2(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: format!("VOP2 expects VGPR dst, got {}", inst.dst),
            });
        };
        let res_bits = self.emit_valu_alu_core(inst, vdst)?;
        self.store_vgpr_exec_masked(vdst, res_bits)
    }

    fn emit_vop3(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        // VOP3-encoded VOPC (opcodes 0x000..=0x0FF)
        if inst.opcode <= 0x0FF {
            return self.emit_vopc(inst);
        }
        // VOP3-encoded VOP1 (opcodes 0x180..=0x1FF)
        if (0x180..=0x1FF).contains(&inst.opcode) {
            return self.emit_vop1(inst);
        }
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedOperand {
                pc: inst.pc,
                detail: format!("VOP3 expects VGPR dst, got {}", inst.dst),
            });
        };
        let res_bits = self.emit_valu_alu_core(inst, vdst)?;
        self.store_vgpr_exec_masked(vdst, res_bits)
    }

    fn emit_valu_alu_core(&mut self, inst: &DecodedInst, vdst: u8) -> Result<u32, SpirvError> {
        let mods = inst.modifiers;
        match inst.mnemonic.as_str() {
            "v_cndmask_b32" => {
                if inst.srcs.len() < 2 {
                    return Err(SpirvError::UnsupportedOperand {
                        pc: inst.pc,
                        detail: "v_cndmask_b32 requires at least 2 sources".into(),
                    });
                }
                let s0 = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let mask_pair = if let Some(&mask_src) = inst.srcs.get(2) {
                    self.load_mask_pair(inst.pc, mask_src)?
                } else {
                    self.b.load(self.uvec2, None, self.var_vcc, None, [])?
                };
                let cond = self.eval_mask_bit(mask_pair)?;
                Ok(self.b.select(self.uint, None, cond, s1, s0)?)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_min_f32" | "v_max_f32" => {
                if inst.srcs.len() < 2 {
                    return Err(SpirvError::UnsupportedOperand {
                        pc: inst.pc,
                        detail: format!("{} requires 2 sources", inst.mnemonic),
                    });
                }
                let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
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
                    "v_min_f32" => self.b.ext_inst(
                        self.f32_ty,
                        None,
                        self.glsl,
                        GLOp::FMin as u32,
                        [Operand::IdRef(a), Operand::IdRef(b)],
                    )?,
                    "v_max_f32" => self.b.ext_inst(
                        self.f32_ty,
                        None,
                        self.glsl,
                        GLOp::FMax as u32,
                        [Operand::IdRef(a), Operand::IdRef(b)],
                    )?,
                    _ => unreachable!(),
                };
                let f = self.apply_float_out_mods(f, mods)?;
                Ok(self.b.bitcast(self.uint, None, f)?)
            }
            // Fused multiply-add family
            "v_fma_f32" | "v_fmamk_f32" | "v_fmaak_f32" | "v_fmac_f32" => {
                let (a, b, c) = if inst.mnemonic == "v_fmac_f32" {
                    if inst.srcs.len() < 2 {
                        return Err(SpirvError::UnsupportedOperand {
                            pc: inst.pc,
                            detail: "v_fmac_f32 requires 2 sources".into(),
                        });
                    }
                    let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let old = self.load_vgpr(vdst)?;
                    let c = self.b.bitcast(self.f32_ty, None, old)?;
                    (a, b, c)
                } else {
                    if inst.srcs.len() < 3 {
                        return Err(SpirvError::UnsupportedOperand {
                            pc: inst.pc,
                            detail: format!("{} requires 3 sources", inst.mnemonic),
                        });
                    }
                    let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                    (a, b, c)
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
            // Legacy FMA / FMAC (zero * anything = +0.0 before add)
            "v_fma_legacy_f32" | "v_fmac_legacy_f32" => {
                let (a, b, c) = if inst.mnemonic == "v_fmac_legacy_f32" {
                    let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let old = self.load_vgpr(vdst)?;
                    let c = self.b.bitcast(self.f32_ty, None, old)?;
                    (a, b, c)
                } else {
                    let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                    (a, b, c)
                };
                let prod = self.emit_legacy_mul_f32(a, b)?;
                let sum = self.b.f_add(self.f32_ty, None, prod, c)?;
                self.b.decorate(sum, Decoration::NoContraction, []);
                let sum = self.apply_float_out_mods(sum, mods)?;
                Ok(self.b.bitcast(self.uint, None, sum)?)
            }
            // Console gfx10.1 non-fused MAD / MAC (explicit NoContraction)
            "v_mac_f32" | "v_mad_f32" | "v_madmk_f32" | "v_madak_f32" => {
                let (a, b, c) = if inst.mnemonic == "v_mac_f32" {
                    let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let old = self.load_vgpr(vdst)?;
                    let c = self.b.bitcast(self.f32_ty, None, old)?;
                    (a, b, c)
                } else {
                    let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                    let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                    let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                    (a, b, c)
                };
                let prod = self.b.f_mul(self.f32_ty, None, a, b)?;
                self.b.decorate(prod, Decoration::NoContraction, []);
                let sum = self.b.f_add(self.f32_ty, None, prod, c)?;
                self.b.decorate(sum, Decoration::NoContraction, []);
                let sum = self.apply_float_out_mods(sum, mods)?;
                Ok(self.b.bitcast(self.uint, None, sum)?)
            }
            "v_med3_f32" => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let c = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                let min_ab = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::FMin as u32,
                    [Operand::IdRef(a), Operand::IdRef(b)],
                )?;
                let max_ab = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::FMax as u32,
                    [Operand::IdRef(a), Operand::IdRef(b)],
                )?;
                let min_hi = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::FMin as u32,
                    [Operand::IdRef(max_ab), Operand::IdRef(c)],
                )?;
                let med = self.b.ext_inst(
                    self.f32_ty,
                    None,
                    self.glsl,
                    GLOp::FMax as u32,
                    [Operand::IdRef(min_ab), Operand::IdRef(min_hi)],
                )?;
                let med = self.apply_float_out_mods(med, mods)?;
                Ok(self.b.bitcast(self.uint, None, med)?)
            }
            // Integer & bitwise vector ALU
            "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_mul_lo_u32" | "v_and_b32"
            | "v_or_b32" | "v_xor_b32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                match inst.mnemonic.as_str() {
                    "v_add_nc_u32" => Ok(self.b.i_add(self.uint, None, a, b)?),
                    "v_sub_nc_u32" => Ok(self.b.i_sub(self.uint, None, a, b)?),
                    "v_subrev_nc_u32" => Ok(self.b.i_sub(self.uint, None, b, a)?),
                    "v_mul_lo_u32" => Ok(self.b.i_mul(self.uint, None, a, b)?),
                    "v_and_b32" => Ok(self.b.bitwise_and(self.uint, None, a, b)?),
                    "v_or_b32" => Ok(self.b.bitwise_or(self.uint, None, a, b)?),
                    "v_xor_b32" => Ok(self.b.bitwise_xor(self.uint, None, a, b)?),
                    _ => unreachable!(),
                }
            }
            "v_bfe_u32" => {
                let base = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let offset = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let count = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let off_m = self.b.bitwise_and(self.uint, None, offset, self.c_u31)?;
                let cnt_m = self.b.bitwise_and(self.uint, None, count, self.c_u31)?;
                Ok(self
                    .b
                    .bit_field_u_extract(self.uint, None, base, off_m, cnt_m)?)
            }
            "v_bfi_b32" => {
                let s0 = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let s2 = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let lhs = self.b.bitwise_and(self.uint, None, s0, s1)?;
                let ns0 = self.b.not(self.uint, None, s0)?;
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
            0x10..=0x1F => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], inst.modifiers, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], inst.modifiers, 1)?;
                self.eval_f32_cmp_pred(op - 0x10, a, b)?
            }
            0x80..=0x8F => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                self.eval_int_cmp_pred(op - 0x80, a, b, true)?
            }
            0xC0..=0xCF => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                self.eval_int_cmp_pred(op - 0xC0, a, b, false)?
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
        let ballot =
            self.b
                .group_non_uniform_ballot(self.uvec4, None, self.c_subgroup_scope, vote)?;
        let lo = self.b.composite_extract(self.uint, None, ballot, [0])?;
        let hi = match self.opts.wave_size {
            WaveSize::Wave32 => None,
            WaveSize::Wave64 => Some(self.b.composite_extract(self.uint, None, ballot, [1])?),
        };
        self.store_mask_pair(inst.pc, inst.dst, lo, hi)
    }

    fn eval_f32_cmp_pred(&mut self, code: u16, a: u32, b: u32) -> Result<u32, SpirvError> {
        match code {
            0x0 => Ok(self.c_false),
            0x1 => Ok(self.b.f_ord_less_than(self.bool_ty, None, a, b)?),
            0x2 => Ok(self.b.f_ord_equal(self.bool_ty, None, a, b)?),
            0x3 => Ok(self.b.f_ord_less_than_equal(self.bool_ty, None, a, b)?),
            0x4 => Ok(self.b.f_ord_greater_than(self.bool_ty, None, a, b)?),
            0x5 => Ok(self.b.f_ord_not_equal(self.bool_ty, None, a, b)?),
            0x6 => Ok(self.b.f_ord_greater_than_equal(self.bool_ty, None, a, b)?),
            0x7 => {
                // O: neither is NaN
                let na = self.b.is_nan(self.bool_ty, None, a)?;
                let nb = self.b.is_nan(self.bool_ty, None, b)?;
                let any_nan = self.b.logical_or(self.bool_ty, None, na, nb)?;
                Ok(self.b.logical_not(self.bool_ty, None, any_nan)?)
            }
            0x8 => {
                // U: either is NaN
                let na = self.b.is_nan(self.bool_ty, None, a)?;
                let nb = self.b.is_nan(self.bool_ty, None, b)?;
                Ok(self.b.logical_or(self.bool_ty, None, na, nb)?)
            }
            0x9 => {
                // NGE = !(a >= b)
                let ge = self.b.f_ord_greater_than_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, ge)?)
            }
            0xA => {
                // NLG = !(a != b ordered)
                let lg = self.b.f_ord_not_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, lg)?)
            }
            0xB => {
                // NGT = !(a > b)
                let gt = self.b.f_ord_greater_than(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, gt)?)
            }
            0xC => {
                // NLE = !(a <= b)
                let le = self.b.f_ord_less_than_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, le)?)
            }
            0xD => {
                // NEQ = !(a == b)
                let eq = self.b.f_ord_equal(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, eq)?)
            }
            0xE => {
                // NLT = !(a < b)
                let lt = self.b.f_ord_less_than(self.bool_ty, None, a, b)?;
                Ok(self.b.logical_not(self.bool_ty, None, lt)?)
            }
            0xF => Ok(self.c_true),
            _ => unreachable!(),
        }
    }

    fn eval_int_cmp_pred(
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
            // Upper 8 codes mirror standard RDNA2 integer compare table when encountered
            0x8..=0xE => Ok(self.c_false),
            0xF => Ok(self.c_true),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::build_synthetic_console_dxbc;
    use crate::decode::disassemble;
    use x2l_pipeline::Cache;

    fn enc_sopp(op: u16, simm16: i16) -> u32 {
        (0x17Fu32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16 as u16)
    }

    fn enc_sop1(op: u16, sdst: u8, s0: u8) -> u32 {
        (0x17Du32 << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(op & 0xFF) << 8)
            | u32::from(s0)
    }

    fn enc_sop2(op: u16, sdst: u8, s0: u8, s1: u8) -> u32 {
        (0b10u32 << 30)
            | (u32::from(op & 0x7F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(s1) << 8)
            | u32::from(s0)
    }

    fn enc_sopc(op: u16, s0: u8, s1: u8) -> u32 {
        (0x17Eu32 << 23) | (u32::from(op & 0x7F) << 16) | (u32::from(s1) << 8) | u32::from(s0)
    }

    fn enc_sopk(op: u16, sdst: u8, simm16: i16) -> u32 {
        (0xBu32 << 28)
            | (u32::from(op & 0x1F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | u32::from(simm16 as u16)
    }

    fn enc_vop1(op: u16, vdst: u8, src0: u16) -> u32 {
        (0x3Fu32 << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(op & 0xFF) << 9)
            | u32::from(src0 & 0x1FF)
    }

    fn enc_vop2(op: u16, vdst: u8, vsrc1: u8, src0: u16) -> u32 {
        (u32::from(op & 0x3F) << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(vsrc1) << 9)
            | u32::from(src0 & 0x1FF)
    }

    fn enc_vopc(op: u16, vsrc1: u8, src0: u16) -> u32 {
        (0x3Eu32 << 25)
            | (u32::from(op & 0xFF) << 17)
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
    fn compiles_and_validates_scalar_alu_and_loop_branch_wave32_and_wave64() {
        // Loop counting s0 down to 0, accumulating into s1, with forward and backward branches.
        let words = [
            enc_sopk(0x00, 0, 4),   // 0x00: s_movk_i32 s0, 4
            enc_sop1(0x03, 1, 128), // 0x04: s_mov_b32 s1, 0
            // loop:
            enc_sop2(0x00, 1, 1, 0),    // 0x08: s_add_u32 s1, s1, s0
            enc_sop2(0x01, 0, 0, 129),  // 0x0c: s_sub_u32 s0, s0, 1
            enc_sopc(0x07, 0, 128),     // 0x10: s_cmp_lg_u32 s0, 0
            enc_sopp(0x05, -4),         // 0x14: s_cbranch_scc1 0x08
            enc_sop2(0x0E, 2, 1, 0xFF), // 0x18: s_and_b32 s2, s1, 0xFF
            0x0000_00FF,                // 0x1c: literal
            enc_sop2(0x10, 3, 2, 130),  // 0x20: s_or_b32 s3, s2, 2
            enc_sop2(0x12, 4, 3, 129),  // 0x24: s_xor_b32 s4, s3, 1
            enc_sopp(0x04, 1),          // 0x28: s_cbranch_scc0 +1 -> 0x30
            enc_sopp(0x0C, 0),          // 0x2c: s_waitcnt 0
            enc_sopp(0x01, 0),          // 0x30: s_endpgm
        ];
        let insts = disassemble(&words).expect("disassemble");

        for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
            let spv = compile_compute_spv_with_options(
                &insts,
                CompileOptions {
                    wave_size,
                    strict_float_controls: true,
                },
            )
            .expect("compile_compute_spv");
            assert_valid_spirv(&spv);
        }
    }

    #[test]
    fn compiles_and_validates_vector_alu_legacy_mul_fma_mad_vopc_and_cndmask() {
        let fma = enc_vop3(0x14B, 5, 256, 257, 258, 0b001, 0b010, true); // v_fma_f32 v5, |v0|, -v1, v2 clamp
        let mad = enc_vop3(0x141, 6, 256, 257, 258, 0, 0, false); // gfx10.1 v_mad_f32 v6, v0, v1, v2
        let vcmp_vop3 = enc_vop3(0x11, 10, 256, 257, 0, 0, 0, false); // v_cmp_lt_f32 s10, v0, v1
        let cnd_vop3 = enc_vop3(0x101, 7, 256, 257, 10, 0, 0, false); // v_cndmask_b32 v7, v0, v1, s10

        let words = [
            enc_vop1(0x01, 0, 242),    // v_mov_b32 v0, 1.0
            enc_vop2(0x03, 1, 0, 256), // v_add_f32 v1, v0, v0
            enc_vop2(0x08, 2, 1, 256), // v_mul_f32 v2, v0, v1
            enc_vop2(0x07, 3, 2, 128), // v_mul_legacy_f32 v3, 0, v2
            enc_vop2(0x2B, 4, 1, 256), // v_fmac_f32 v4, v0, v1
            enc_vop2(0x1F, 4, 1, 256), // gfx10.1 v_mac_f32 v4, v0, v1
            enc_vop2(0x20, 8, 1, 256), // gfx10.1 v_madmk_f32 v8, v0, 2.0, v1
            2.0_f32.to_bits(),
            enc_vop2(0x21, 9, 1, 256), // gfx10.1 v_madak_f32 v9, v0, v1, 3.0
            3.0_f32.to_bits(),
            fma[0],
            fma[1],
            mad[0],
            mad[1],
            enc_vop2(0x25, 10, 1, 256),  // v_add_nc_u32 v10, v0, v1
            enc_vop2(0x1B, 11, 10, 256), // v_and_b32 v11, v0, v10
            enc_vop2(0x1C, 12, 11, 256), // v_or_b32 v12, v0, v11
            enc_vop2(0x1D, 13, 12, 256), // v_xor_b32 v13, v0, v12
            enc_vopc(0x14, 1, 256),      // v_cmp_gt_f32 vcc_lo, v0, v1
            enc_vop2(0x01, 14, 1, 256),  // v_cndmask_b32 v14, v0, v1 (vcc)
            vcmp_vop3[0],
            vcmp_vop3[1],
            cnd_vop3[0],
            cnd_vop3[1],
            enc_sopp(0x01, 0), // s_endpgm
        ];
        let insts = disassemble(&words).expect("disassemble");

        for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
            let spv = compile_compute_spv_with_options(
                &insts,
                CompileOptions {
                    wave_size,
                    strict_float_controls: true,
                },
            )
            .expect("compile vector ALU");
            assert_valid_spirv(&spv);
        }
    }

    #[test]
    fn pipeline_cache_roundtrips_synthetic_console_container_to_valid_spirv() {
        let code = [
            enc_vop2(0x07, 0, 1, 256), // v_mul_legacy_f32 v0, v0, v1
            enc_sopp(0x01, 0),         // s_endpgm
        ];
        // api_stage_id=5 (Compute), hw_stage_slot_offset=0x60 (CS), wave32=true
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

        let bad = [enc_sopp(0x02, 10), enc_sopp(0x01, 0)];
        let insts = disassemble(&bad).expect("disassemble");
        assert!(matches!(
            compile_compute_spv(&insts),
            Err(SpirvError::InvalidBranchTarget { .. })
        ));
    }
}
