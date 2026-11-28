// Clean-room graphics-stage direct binary SPIR-V 1.6 lowering & live-register pruning for X2Linux.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Direct `RDNA2 -> Binary SPIR-V 1.6` graphics shader compiler (`Vertex` and `Fragment`)
//! with live-register / dead-register pruning, `ISG1`/`OSG1` interface mapping, `EXP`,
//! `VINTRP`, `MIMG` (`image_sample`, `image_sample_l`, `image_load`, `image_store`), and
//! SPIR-V 1.6 `OpDemoteToHelperInvocation` pixel discard.

use crate::container::{ShaderStage, StageSignature};
use crate::decode::{
    DecodedInst, DstOperand, Encoding, InstExtra, SrcOperand, ValuExt, ValuModifiers,
};
use crate::image::{ImageBindings, Mimg};
use crate::spirv::{
    Alu, HalfRounding, MemoryWrite, SpirvError, inline_float_to_f16_bits, sdwa_half_offset,
};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, InsertPoint, Instruction, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, GLOp, LoopControl, MemoryModel, Op, Scope, SelectionControl, StorageClass,
};
use std::collections::{BTreeMap, BTreeSet};

const HALT_PC: u32 = u32::MAX;

/// Configuration for graphics/pipeline-stage (`Vertex` / `Fragment` / `Compute`) SPIR-V 1.6 lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphicsCompileOptions {
    /// Negate clip-space `Position.y` in Vertex shaders (for D3D -> Vulkan clip convention).
    pub flip_y: bool,
    /// Emit `RoundingModeRTE` and `SignedZeroInfNanPreserve` for binary32.
    pub strict_float_controls: bool,
    /// Emit `DenormPreserve` for binary32 (only enable when supported by target device).
    pub denorm_preserve: bool,
    /// Enable the D3D12.X Set 0 pipeline binding layout:
    /// - Binding 0 (`Uniform`): `RootData { uvec4 words[16]; }` seeding live user-data SGPRs
    /// - Binding 1 (`StorageBuffer`, `NonWritable`): CBV/SRV memory buffer (`u32[]`)
    /// - Binding 2 (`StorageBuffer`): UAV read/write buffer (`u32[]`)
    pub d3d12_root_bindings: bool,
    /// Local workgroup size `(x, y, z)` for `ShaderStage::Compute` D3D12 pipelines (`[0,0,0]` -> `[64,1,1]`).
    pub compute_local_size: [u32; 3],
}

/// Explicit dispatch and output-bounds layout for RDNA2 Mesh Shader (`MS` -> SPIR-V 1.6 `MeshEXT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshLayout {
    pub local_size: [u32; 3],
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub wave_size: u32,
    pub lds_bytes: u32,
}

impl Default for MeshLayout {
    fn default() -> Self {
        Self {
            local_size: [32, 1, 1],
            max_vertices: 64,
            max_primitives: 64,
            wave_size: 32,
            lds_bytes: 0,
        }
    }
}

impl MeshLayout {
    /// Read the public DXBC PSV0 mesh layout, with conservative defaults when
    /// layout metadata is absent. Only triangle output is currently supported.
    pub fn from_dxbc(bytes: &[u8], wave32: bool) -> Result<Self, crate::container::ContainerError> {
        use crate::container::{ContainerError, dxbc_parts};
        let wave_size = if wave32 { 32 } else { 64 };
        let mut layout = Self {
            local_size: [wave_size, 1, 1],
            wave_size,
            ..Self::default()
        };
        let parts = dxbc_parts(bytes)?;
        let Some((_, psv)) = parts.iter().find(|(tag, _)| tag == b"PSV0") else {
            return Ok(layout);
        };
        let field = |offset: usize| -> Result<u32, ContainerError> {
            let bytes = psv
                .get(offset..offset + 4)
                .ok_or(ContainerError::Truncated("PSV0 mesh runtime info"))?;
            Ok(u32::from_le_bytes(
                bytes.try_into().expect("four-byte field"),
            ))
        };
        let runtime_size = field(0)? as usize;
        if runtime_size < 24 || runtime_size > psv.len() - 4 {
            return Err(ContainerError::InvalidHeader(
                "invalid PSV0 runtime info size".into(),
            ));
        }
        layout.lds_bytes = field(4)?;
        let counts = field(16)?;
        layout.max_vertices = counts & 0xffff;
        layout.max_primitives = counts >> 16;
        // PSVRuntimeInfo1 adds the output topology, PSVRuntimeInfo2 adds NumThreads.
        if runtime_size >= 36 && psv[31] != 2 {
            return Err(ContainerError::InvalidHeader(
                "PSV0 mesh topology must be triangles".into(),
            ));
        }
        if runtime_size >= 48 {
            layout.local_size = [field(40)?, field(44)?, field(48)?];
        }
        let local_threads = layout
            .local_size
            .into_iter()
            .try_fold(1u32, u32::checked_mul);
        if !matches!(local_threads, Some(1..=128))
            || !(1..=256).contains(&layout.max_vertices)
            || !(1..=256).contains(&layout.max_primitives)
            || layout.lds_bytes > 32 * 1024
        {
            return Err(ContainerError::InvalidHeader(
                "invalid PSV0 mesh layout bounds".into(),
            ));
        }
        Ok(layout)
    }
}

/// Result of lightweight liveness & resource analysis over a decoded graphics/compute/mesh shader.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiveRegisters {
    pub live_sgprs: BTreeSet<u8>,
    pub live_vgprs: BTreeSet<u8>,
    pub uses_memory: bool,
    pub uses_mem_store: bool,
    pub uses_lds: bool,
    pub uses_sampled_image: bool,
    pub uses_storage_image: bool,
    pub uses_demote: bool,
    pub exports_pos0: bool,
    pub exports_prim: bool,
    pub exports_mrtz: bool,
    pub uses_vertex_id: bool,
    pub param_locations: BTreeSet<u8>,
    pub mrt_locations: BTreeSet<u8>,
    pub input_locations: BTreeSet<u8>,
}

/// Analyze which SGPRs, VGPRs, stage I/O locations, descriptors, and special modes are
/// actually live in a decoded graphics shader so dead registers are pruned from SPIR-V.
pub fn analyze_live_registers(
    stage: ShaderStage,
    insts: &[DecodedInst],
    sig: &StageSignature,
) -> LiveRegisters {
    let mut live = LiveRegisters::default();

    let mark_src = |src: SrcOperand, live: &mut LiveRegisters| match src {
        SrcOperand::Sgpr(n) => {
            live.live_sgprs.insert(n);
        }
        SrcOperand::Vgpr(n) => {
            live.live_vgprs.insert(n);
        }
        SrcOperand::M0 => {
            live.live_sgprs.insert(124);
        }
        _ => {}
    };
    let mark_dst = |dst: DstOperand, live: &mut LiveRegisters| match dst {
        DstOperand::Sgpr(n) => {
            live.live_sgprs.insert(n);
        }
        DstOperand::Vgpr(n) => {
            live.live_vgprs.insert(n);
        }
        DstOperand::M0 => {
            live.live_sgprs.insert(124);
        }
        _ => {}
    };

    // Map ISG1 inputs for Vertex shaders
    if stage == ShaderStage::Vertex {
        for elem in &sig.inputs {
            if elem.system_value_type == 6 {
                // SV_VertexID -> v0
                live.uses_vertex_id = true;
                live.live_vgprs.insert(0);
            } else if elem.register != u32::MAX && elem.register < 32 {
                let loc = elem.register as u8;
                live.input_locations.insert(loc);
                for c in 0..4u8 {
                    if (elem.mask & (1 << c)) != 0 {
                        live.live_vgprs.insert(loc * 4 + c);
                    }
                }
            }
        }
    }

    for inst in insts {
        if inst.mnemonic == "s_swappc_b64" {
            continue;
        }
        if inst.mnemonic == "s_sendmsg" && inst.simm16 == Some(9) {
            live.live_sgprs.insert(124); // M0
        }
        mark_dst(inst.dst, &mut live);
        if inst.encoding != Encoding::Export {
            for &src in &inst.srcs {
                mark_src(src, &mut live);
            }
        }

        if matches!(inst.encoding, Encoding::Vopc | Encoding::Vop3)
            && inst.mnemonic.starts_with("v_cmp")
            && let DstOperand::Sgpr(reg) = inst.dst
        {
            live.live_sgprs.insert(reg + 1);
        }
        if inst.mnemonic == "v_cndmask_b32"
            && let Some(SrcOperand::Sgpr(reg)) = inst.srcs.get(2)
        {
            live.live_sgprs.insert(reg + 1);
        }

        // 64-bit scalar register pairs
        if inst.mnemonic.ends_with("_b64") {
            if let DstOperand::Sgpr(n) = inst.dst {
                live.live_sgprs.insert(n.wrapping_add(1));
            }
            for &src in &inst.srcs {
                if let SrcOperand::Sgpr(n) = src {
                    live.live_sgprs.insert(n.wrapping_add(1));
                }
            }
        }

        // Pixel discard detection: EXEC writes or `exp null`
        if stage == ShaderStage::Pixel {
            let writes_exec = matches!(inst.dst, DstOperand::ExecLo | DstOperand::ExecHi)
                || inst.mnemonic.contains("saveexec");
            if writes_exec {
                live.uses_demote = true;
            }
        }

        match inst.extra {
            InstExtra::Exp {
                target,
                en,
                compr,
                vsrc,
                ..
            } => {
                if compr {
                    if (en & 0x3) != 0 {
                        live.live_vgprs.insert(vsrc[0]);
                    }
                    if (en & 0xC) != 0 {
                        live.live_vgprs.insert(vsrc[1]);
                    }
                } else {
                    for (c, &reg) in vsrc.iter().enumerate() {
                        if (en & (1 << c)) != 0 {
                            live.live_vgprs.insert(reg);
                        }
                    }
                }
                match target {
                    0..=7 => {
                        live.mrt_locations.insert(target);
                    }
                    8 => {
                        live.exports_mrtz = true;
                    }
                    9 => {
                        if stage == ShaderStage::Pixel {
                            live.uses_demote = true;
                        }
                    }
                    12 => {
                        live.exports_pos0 = true;
                    }
                    20 => {
                        live.exports_prim = true;
                    }
                    32..=63 => {
                        live.param_locations.insert(target - 32);
                    }
                    _ => {}
                }
            }
            InstExtra::DsLds {
                addr,
                data0,
                data1,
                vdst,
                is_write,
                paired,
                ..
            } => {
                live.uses_lds = true;
                live.live_vgprs.insert(addr);
                if is_write {
                    live.live_vgprs.insert(data0);
                    if paired {
                        live.live_vgprs.insert(data1);
                    }
                } else {
                    live.live_vgprs.insert(vdst);
                    if paired {
                        live.live_vgprs.insert(vdst.wrapping_add(1));
                    }
                }
            }
            InstExtra::Vintrp { vdst, attr, .. } => {
                live.live_vgprs.insert(vdst);
                live.input_locations.insert(attr);
            }
            InstExtra::Mimg {
                op,
                dim,
                dmask,
                vdata,
                addr_count,
                is_nsa,
                addrs,
                ..
            } => {
                let is_store = matches!(op, 0x08 | 0x09);
                if is_store {
                    live.uses_storage_image = true;
                } else {
                    live.uses_sampled_image = true;
                }
                let data_regs = if matches!(op, 0x40 | 0x47) {
                    4
                } else {
                    dmask.count_ones().max(1) as u8
                };
                for j in 0..data_regs {
                    live.live_vgprs.insert(vdata.wrapping_add(j));
                }
                let coords = Mimg::address_count(op, dim);
                if is_nsa {
                    for &r in addrs
                        .iter()
                        .take((coords as usize).min(addr_count as usize))
                    {
                        live.live_vgprs.insert(r);
                    }
                } else {
                    for j in 0..coords {
                        live.live_vgprs.insert(addrs[0].wrapping_add(j));
                    }
                }
            }
            InstExtra::Smem {
                sbase,
                sdata,
                soffset,
                dwords,
                is_buffer,
                ..
            } => {
                live.uses_memory = true;
                let base_n = if is_buffer { 4u8 } else { 2u8 };
                for j in 0..base_n {
                    live.live_sgprs.insert(sbase.wrapping_add(j));
                }
                for j in 0..dwords {
                    live.live_sgprs.insert(sdata.wrapping_add(j));
                }
                if soffset < 106 {
                    live.live_sgprs.insert(soffset);
                }
            }
            InstExtra::Mubuf {
                vaddr,
                vdata,
                srsrc,
                soffset,
                offen,
                idxen,
                dwords,
                is_store,
                ..
            } => {
                if is_store {
                    live.uses_mem_store = true;
                } else {
                    live.uses_memory = true;
                }
                for j in 0..4u8 {
                    live.live_sgprs.insert(srsrc.wrapping_add(j));
                }
                if idxen {
                    live.live_vgprs.insert(vaddr);
                }
                if offen {
                    live.live_vgprs
                        .insert(if idxen { vaddr.wrapping_add(1) } else { vaddr });
                }
                for j in 0..dwords {
                    live.live_vgprs.insert(vdata.wrapping_add(j));
                }
                if soffset < 106 {
                    live.live_sgprs.insert(soffset);
                }
            }
            InstExtra::Flat {
                addr,
                data,
                vdst,
                saddr,
                dwords,
                is_store,
                ..
            } => {
                if is_store {
                    live.uses_mem_store = true;
                } else {
                    live.uses_memory = true;
                }
                if saddr == 0x7D {
                    live.live_vgprs.insert(addr);
                    live.live_vgprs.insert(addr.wrapping_add(1));
                } else {
                    live.live_sgprs.insert(saddr);
                    live.live_sgprs.insert(saddr.wrapping_add(1));
                    live.live_vgprs.insert(addr);
                }
                let base = if is_store { data } else { vdst };
                for j in 0..dwords {
                    live.live_vgprs.insert(base.wrapping_add(j));
                }
            }
            _ => {}
        }
    }

    live
}

/// Compile a decoded RDNA2 graphics or D3D12 compute shader (`ShaderStage::Vertex`,
/// `ShaderStage::Pixel`, or `ShaderStage::Compute`) directly to Vulkan 1.3-compliant binary
/// SPIR-V 1.6 with dead-register pruning.
pub fn compile_graphics_spv(
    stage: ShaderStage,
    insts: &[DecodedInst],
    sig: &StageSignature,
    opts: &GraphicsCompileOptions,
) -> Result<Vec<u32>, SpirvError> {
    if insts.is_empty() {
        return Err(SpirvError::EmptyProgram);
    }
    if !matches!(
        stage,
        ShaderStage::Vertex | ShaderStage::Pixel | ShaderStage::Compute
    ) {
        return Err(SpirvError::UnsupportedInstruction {
            pc: 0,
            mnemonic: format!(
                "compile_graphics_spv only supports Vertex/Pixel/Compute, got {stage}"
            ),
        });
    }

    let live = analyze_live_registers(stage, insts, sig);
    let blocks = build_graphics_blocks(insts)?;
    let mut ctx = GfxEmitCtx::new(stage, *opts, MeshLayout::default(), live, insts, false)?;
    ctx.emit_module(sig, &blocks)?;
    Ok(ctx.b.module().assemble())
}

/// Compile a decoded RDNA2 Mesh Shader (`ShaderStage::Mesh`) directly to Vulkan 1.3-compliant
/// binary SPIR-V 1.6 (`ExecutionModel::MeshEXT`, `SPV_EXT_mesh_shader`) with converged
/// `OpSetMeshOutputsEXT`, 9-bit GFX10.3 `exp prim` unpacking, per-lane `EXEC` masking, and
/// optional Workgroup LDS (single- or paired-word access and `s_barrier`).
pub fn compile_mesh_spv(
    insts: &[DecodedInst],
    sig: &StageSignature,
    opts: &GraphicsCompileOptions,
    layout: MeshLayout,
) -> Result<Vec<u32>, SpirvError> {
    if insts.is_empty() {
        return Err(SpirvError::EmptyProgram);
    }
    if layout.max_vertices == 0 || layout.max_primitives == 0 {
        return Err(SpirvError::UnsupportedInstruction {
            pc: 0,
            mnemonic: "mesh layout max_vertices/max_primitives must be non-zero".into(),
        });
    }
    let blocks = build_graphics_blocks(insts)?;
    // A later allocation is supported when the entire workgroup can execute
    // as one guest wave. Exports are then staged until the converged epilogue.
    let mut deferred_mesh = false;
    let mut saw_alloc = false;
    let mut saw_export = false;
    for (blk_idx, blk) in blocks.0.iter().enumerate() {
        for inst in &blk.insts {
            if inst.mnemonic == "s_sendmsg" {
                if inst.simm16 != Some(9) {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: format!(
                            "unsupported s_sendmsg imm 0x{:04x}",
                            inst.simm16.unwrap_or(0) as u16
                        ),
                    });
                }
                if saw_alloc || saw_export {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: "non-convergent or duplicate s_sendmsg(MSG_GS_ALLOC_REQ)".into(),
                    });
                }
                saw_alloc = true;
                deferred_mesh = blk_idx != 0;
            } else if inst.encoding == Encoding::Export {
                if !saw_alloc {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: "mesh EXP encountered before s_sendmsg(MSG_GS_ALLOC_REQ)".into(),
                    });
                }
                saw_export = true;
            }
        }
    }
    if !saw_alloc {
        return Err(SpirvError::UnsupportedInstruction {
            pc: 0,
            mnemonic: "mesh shader missing s_sendmsg(MSG_GS_ALLOC_REQ)".into(),
        });
    }

    if deferred_mesh {
        let threads = layout
            .local_size
            .into_iter()
            .try_fold(1u32, u32::checked_mul);
        if !matches!(threads, Some(n) if n > 0 && n <= layout.wave_size)
            || !matches!(layout.wave_size, 32 | 64)
        {
            return Err(SpirvError::UnsupportedInstruction {
                pc: 0,
                mnemonic: "conditional mesh allocation requires one guest wave per workgroup"
                    .into(),
            });
        }
    }

    let live = analyze_live_registers(ShaderStage::Mesh, insts, sig);
    let mut ctx = GfxEmitCtx::new(ShaderStage::Mesh, *opts, layout, live, insts, deferred_mesh)?;
    ctx.emit_module(sig, &blocks)?;
    Ok(ctx.b.module().assemble())
}

/// Convenience entry point for D3D12.X compute pipelines (`CreateComputePipelineState`):
/// compiles decoded RDNA2 compute instructions to pruned `GLCompute` SPIR-V 1.6 using Set 0
/// root constants (Binding 0), CBV/SRV buffer (Binding 1), and UAV read/write buffer (Binding 2).
pub fn compile_d3d12_compute_spv(
    insts: &[DecodedInst],
    local_size: [u32; 3],
    strict_float_controls: bool,
    denorm_preserve: bool,
) -> Result<Vec<u32>, SpirvError> {
    let opts = GraphicsCompileOptions {
        flip_y: false,
        strict_float_controls,
        denorm_preserve,
        d3d12_root_bindings: true,
        compute_local_size: local_size,
    };
    compile_graphics_spv(
        ShaderStage::Compute,
        insts,
        &StageSignature::default(),
        &opts,
    )
}

// ---------------------------------------------------------------------------
// Basic-block partitioning
// ---------------------------------------------------------------------------

struct GfxBlock<'a> {
    id: u32,
    insts: Vec<&'a DecodedInst>,
    fallthrough_block: Option<u32>,
}

fn build_graphics_blocks<'a>(
    insts: &'a [DecodedInst],
) -> Result<(Vec<GfxBlock<'a>>, BTreeMap<u32, u32>), SpirvError> {
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
        let last = slice.last().expect("non-empty block");
        let next_pc = last.pc.wrapping_add(u32::from(last.word_len) * 4);
        let fallthrough_block = pc_to_block.get(&next_pc).copied();
        blocks.push(GfxBlock {
            id: block_id as u32,
            insts: slice,
            fallthrough_block,
        });
    }
    Ok((blocks, pc_to_block))
}

// ---------------------------------------------------------------------------
// Graphics SPIR-V 1.6 emitter
// ---------------------------------------------------------------------------

struct GfxEmitCtx {
    stage: ShaderStage,
    opts: GraphicsCompileOptions,
    mesh_layout: MeshLayout,
    deferred_mesh: bool,
    mesh_collective: Option<u32>,
    mesh_exports: BTreeMap<u8, (u32, u32)>,
    live: LiveRegisters,
    b: Builder,
    glsl: u32,
    // Types
    void: u32,
    bool_ty: u32,
    uint: u32,
    sint: u32,
    f32_ty: u32,
    vec2: u32,
    vec4: u32,
    uvec3: u32,
    uvec4: u32,
    ptr_fn_uint: u32,
    ptr_fn_bool: u32,
    ptr_sb_uint: u32,
    ptr_uni_uint: u32,
    ptr_wg_uint: u32,
    ptr_out_vec4: u32,
    ptr_out_uvec3: u32,
    ptr_out_bool: u32,
    image_bindings: ImageBindings,
    // Globals
    interface_vars: Vec<u32>,
    in_vertex_id: Option<u32>,
    in_local_id: Option<u32>,
    in_workgroup_id: Option<u32>,
    in_local_index: Option<u32>,
    in_subgroup_lane: Option<u32>,
    in_locs: BTreeMap<u8, u32>,
    out_pos0: Option<u32>,
    out_params: BTreeMap<u8, u32>,
    out_mrts: BTreeMap<u8, u32>,
    out_mrtz: Option<u32>,
    out_mesh_pos: Option<u32>,
    out_mesh_params: BTreeMap<u8, u32>,
    out_mesh_prim_indices: Option<u32>,
    out_mesh_prim_cull: Option<u32>,
    lds_buf: Option<u32>,
    lds_dwords: u32,
    root_buf: Option<u32>,
    memory_buf: Option<u32>,
    uav_buf: Option<u32>,
    // Constants
    c_u0: u32,
    c_u1: u32,
    c_u2: u32,
    c_u3: u32,
    c_halt: u32,
    c_f0: u32,
    c_f1: u32,
    c_f2: u32,
    c_f4: u32,
    c_f05: u32,
    c_abs_mask: u32,
    c_sign_mask: u32,
    c_exp_mask: u32,
    c_true: u32,
    c_false: u32,
    c_wg_scope: u32,
    c_wg_barrier_sem: u32,
    u32_consts: BTreeMap<u32, u32>,
    f32_consts: BTreeMap<u32, u32>,
    // Pruned Function variables
    var_pc: u32,
    var_active: u32,
    var_vcc: u32,
    var_scc: u32,
    var_mem_tmp: u32,
    var_mesh_vtx_cnt: u32,
    var_mesh_prim_cnt: u32,
    mesh_local_idx: u32,
    mesh_guest_lane: u32,
    mesh_lane_bit: u32,
    subgroup_lane: u32,
    subgroup_scope: u32,
    sgpr_vars: BTreeMap<u8, u32>,
    vgpr_vars: BTreeMap<u8, u32>,
    mem_words_len: u32,
    uav_words_len: u32,
}

impl GfxEmitCtx {
    fn new(
        stage: ShaderStage,
        opts: GraphicsCompileOptions,
        mesh_layout: MeshLayout,
        live: LiveRegisters,
        insts: &[DecodedInst],
        deferred_mesh: bool,
    ) -> Result<Self, SpirvError> {
        let uses_readfirstlane = insts.iter().any(|i| i.mnemonic == "v_readfirstlane_b32");
        let uses_quad_swizzle = stage == ShaderStage::Pixel
            && insts.iter().any(|i| {
                matches!(
                    i.extra,
                    InstExtra::DsSwizzle {
                        offset: 0x8000..=0x80ff,
                        ..
                    }
                )
            });
        let uses_popcount_mask = insts.iter().any(|inst| {
            inst.mnemonic.starts_with("s_bcnt1_i32_b")
                && matches!(
                    inst.srcs.first(),
                    Some(
                        SrcOperand::ExecLo
                            | SrcOperand::ExecHi
                            | SrcOperand::VccLo
                            | SrcOperand::VccHi
                    )
                )
        });
        let uses_wave_masks = uses_readfirstlane
            || uses_quad_swizzle
            || uses_popcount_mask
            || insts.iter().any(|i| {
                matches!(
                    i.mnemonic.as_str(),
                    "s_wqm_b32" | "s_wqm_b64" | "s_andn2_wrexec_b32" | "s_andn2_wrexec_b64"
                )
            });
        let mut b = Builder::new();
        b.set_version(1, 6);
        b.capability(Capability::Shader);
        if uses_wave_masks && !deferred_mesh {
            b.capability(Capability::GroupNonUniform);
            b.capability(Capability::GroupNonUniformBallot);
        }
        if uses_quad_swizzle {
            b.capability(Capability::GroupNonUniformQuad);
        }
        if uses_readfirstlane && !deferred_mesh {
            b.capability(Capability::GroupNonUniformShuffle);
        }
        if stage == ShaderStage::Mesh {
            b.capability(Capability::MeshShadingEXT);
            b.extension("SPV_EXT_mesh_shader");
        }
        if live.uses_demote {
            b.capability(Capability::DemoteToHelperInvocation);
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
        let vec4 = b.type_vector(f32_ty, 4);
        let uvec3 = b.type_vector(uint, 3);
        let uvec4 = b.type_vector(uint, 4);

        let ptr_fn_uint = b.type_pointer(None, StorageClass::Function, uint);
        let ptr_fn_bool = b.type_pointer(None, StorageClass::Function, bool_ty);
        let ptr_sb_uint = b.type_pointer(None, StorageClass::StorageBuffer, uint);
        let ptr_uni_uint = b.type_pointer(None, StorageClass::Uniform, uint);
        let ptr_wg_uint = b.type_pointer(None, StorageClass::Workgroup, uint);
        let ptr_in_vec4 = b.type_pointer(None, StorageClass::Input, vec4);
        let ptr_in_uint = b.type_pointer(None, StorageClass::Input, uint);
        let ptr_in_uvec3 = b.type_pointer(None, StorageClass::Input, uvec3);
        let ptr_out_vec4 = b.type_pointer(None, StorageClass::Output, vec4);
        let ptr_out_uvec3 = b.type_pointer(None, StorageClass::Output, uvec3);
        let ptr_out_bool = b.type_pointer(None, StorageClass::Output, bool_ty);
        let ptr_out_f32 = b.type_pointer(None, StorageClass::Output, f32_ty);

        let c_u0 = b.constant_bit32(uint, 0);
        let c_u1 = b.constant_bit32(uint, 1);
        let c_u2 = b.constant_bit32(uint, 2);
        let c_u3 = b.constant_bit32(uint, 3);
        let c_u16 = b.constant_bit32(uint, 16);
        let c_wg_scope = c_u2;
        let c_wg_barrier_sem = b.constant_bit32(uint, 0x0108); // AcquireRelease | WorkgroupMemory
        let c_halt = b.constant_bit32(uint, HALT_PC);
        let c_f0 = b.constant_bit32(f32_ty, 0.0_f32.to_bits());
        let c_f1 = b.constant_bit32(f32_ty, 1.0_f32.to_bits());
        let c_f2 = b.constant_bit32(f32_ty, 2.0_f32.to_bits());
        let c_f4 = b.constant_bit32(f32_ty, 4.0_f32.to_bits());
        let c_f05 = b.constant_bit32(f32_ty, 0.5_f32.to_bits());
        let c_abs_mask = b.constant_bit32(uint, 0x7FFF_FFFF);
        let c_sign_mask = b.constant_bit32(uint, 0x8000_0000);
        let c_exp_mask = b.constant_bit32(uint, 0x7F80_0000);
        let c_true = b.constant_true(bool_ty);
        let c_false = b.constant_false(bool_ty);

        let mut interface_vars = Vec::new();
        let in_subgroup_lane = (uses_wave_masks && !deferred_mesh).then(|| {
            let var = b.variable(ptr_in_uint, None, StorageClass::Input, None);
            b.decorate(
                var,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId)],
            );
            if stage == ShaderStage::Pixel {
                b.decorate(var, Decoration::Flat, []);
            }
            interface_vars.push(var);
            var
        });
        let mesh_collective = deferred_mesh.then(|| {
            let words = b.constant_bit32(uint, 2 + mesh_layout.wave_size);
            let array = b.type_array(uint, words);
            let ptr = b.type_pointer(None, StorageClass::Workgroup, array);
            let var = b.variable(ptr, None, StorageClass::Workgroup, None);
            interface_vars.push(var);
            var
        });
        let mut in_locs = BTreeMap::new();
        let mut out_params = BTreeMap::new();
        let mut out_mrts = BTreeMap::new();

        // Vertex built-in VertexIndex
        let in_vertex_id = if stage == ShaderStage::Vertex && live.uses_vertex_id {
            let v = b.variable(ptr_in_uint, None, StorageClass::Input, None);
            b.decorate(
                v,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::VertexIndex)],
            );
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };

        // Compute built-ins LocalInvocationId & WorkgroupId
        let in_local_id = if stage == ShaderStage::Compute
            && (live.live_vgprs.contains(&0)
                || live.live_vgprs.contains(&1)
                || live.live_vgprs.contains(&2))
        {
            let v = b.variable(ptr_in_uvec3, None, StorageClass::Input, None);
            b.decorate(
                v,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::LocalInvocationId)],
            );
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };
        let in_workgroup_id = if stage == ShaderStage::Compute
            && (live.live_sgprs.contains(&60)
                || live.live_sgprs.contains(&61)
                || live.live_sgprs.contains(&62))
        {
            let v = b.variable(ptr_in_uvec3, None, StorageClass::Input, None);
            b.decorate(
                v,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::WorkgroupId)],
            );
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };

        // Mesh built-in LocalInvocationIndex & MeshEXT output arrays
        let in_local_index = if stage == ShaderStage::Mesh {
            let v = b.variable(ptr_in_uint, None, StorageClass::Input, None);
            b.decorate(
                v,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::LocalInvocationIndex)],
            );
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };

        let mut out_mesh_params = BTreeMap::new();
        let (out_mesh_pos, out_mesh_prim_indices, out_mesh_prim_cull) =
            if stage == ShaderStage::Mesh {
                let c_max_v = b.constant_bit32(uint, mesh_layout.max_vertices.max(1));
                let c_max_p = b.constant_bit32(uint, mesh_layout.max_primitives.max(1));

                let pos_var = if live.exports_pos0 {
                    let vtx_block = b.type_struct([vec4]);
                    b.decorate(vtx_block, Decoration::Block, []);
                    b.member_decorate(
                        vtx_block,
                        0,
                        Decoration::BuiltIn,
                        [Operand::BuiltIn(BuiltIn::Position)],
                    );
                    let pos_arr = b.type_array(vtx_block, c_max_v);
                    let ptr_arr = b.type_pointer(None, StorageClass::Output, pos_arr);
                    let v = b.variable(ptr_arr, None, StorageClass::Output, None);
                    interface_vars.push(v);
                    Some(v)
                } else {
                    None
                };

                if !live.param_locations.is_empty() {
                    let param_arr = b.type_array(vec4, c_max_v);
                    let ptr_param_arr = b.type_pointer(None, StorageClass::Output, param_arr);
                    for &loc in &live.param_locations {
                        let v = b.variable(ptr_param_arr, None, StorageClass::Output, None);
                        b.decorate(
                            v,
                            Decoration::Location,
                            [Operand::LiteralBit32(u32::from(loc))],
                        );
                        interface_vars.push(v);
                        out_mesh_params.insert(loc, v);
                    }
                }

                let (idx_var, cull_var) = if live.exports_prim {
                    let idx_arr = b.type_array(uvec3, c_max_p);
                    let ptr_idx_arr = b.type_pointer(None, StorageClass::Output, idx_arr);
                    let iv = b.variable(ptr_idx_arr, None, StorageClass::Output, None);
                    b.decorate(
                        iv,
                        Decoration::BuiltIn,
                        [Operand::BuiltIn(BuiltIn::PrimitiveTriangleIndicesEXT)],
                    );
                    b.decorate(iv, Decoration::PerPrimitiveEXT, []);
                    interface_vars.push(iv);

                    let cull_arr = b.type_array(bool_ty, c_max_p);
                    let ptr_cull_arr = b.type_pointer(None, StorageClass::Output, cull_arr);
                    let cv = b.variable(ptr_cull_arr, None, StorageClass::Output, None);
                    b.decorate(
                        cv,
                        Decoration::BuiltIn,
                        [Operand::BuiltIn(BuiltIn::CullPrimitiveEXT)],
                    );
                    b.decorate(cv, Decoration::PerPrimitiveEXT, []);
                    interface_vars.push(cv);
                    (Some(iv), Some(cv))
                } else {
                    (None, None)
                };

                (pos_var, idx_var, cull_var)
            } else {
                (None, None, None)
            };

        let lds_dwords = if live.uses_lds {
            (mesh_layout.lds_bytes.max(1024) / 4).max(1)
        } else {
            0
        };
        let lds_buf = if live.uses_lds {
            let c_lds = b.constant_bit32(uint, lds_dwords);
            let lds_arr = b.type_array(uint, c_lds);
            let ptr_lds_arr = b.type_pointer(None, StorageClass::Workgroup, lds_arr);
            let v = b.variable(ptr_lds_arr, None, StorageClass::Workgroup, None);
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };

        // Stage Input Locations (ISG1 for VS, VINTRP attrs for PS)
        for &loc in &live.input_locations {
            let v = b.variable(ptr_in_vec4, None, StorageClass::Input, None);
            b.decorate(
                v,
                Decoration::Location,
                [Operand::LiteralBit32(u32::from(loc))],
            );
            interface_vars.push(v);
            in_locs.insert(loc, v);
        }

        // Vertex Outputs: BuiltIn Position & param Locations
        let out_pos0 = if stage == ShaderStage::Vertex && live.exports_pos0 {
            let v = b.variable(ptr_out_vec4, None, StorageClass::Output, None);
            b.decorate(
                v,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::Position)],
            );
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };
        if stage == ShaderStage::Vertex {
            for &loc in &live.param_locations {
                let v = b.variable(ptr_out_vec4, None, StorageClass::Output, None);
                b.decorate(
                    v,
                    Decoration::Location,
                    [Operand::LiteralBit32(u32::from(loc))],
                );
                interface_vars.push(v);
                out_params.insert(loc, v);
            }
        }

        // Fragment Outputs: MRT Locations & optional FragDepth
        if stage == ShaderStage::Pixel {
            for &loc in &live.mrt_locations {
                let v = b.variable(ptr_out_vec4, None, StorageClass::Output, None);
                b.decorate(
                    v,
                    Decoration::Location,
                    [Operand::LiteralBit32(u32::from(loc))],
                );
                interface_vars.push(v);
                out_mrts.insert(loc, v);
            }
        }
        let out_mrtz = if stage == ShaderStage::Pixel && live.exports_mrtz {
            let v = b.variable(ptr_out_f32, None, StorageClass::Output, None);
            b.decorate(
                v,
                Decoration::BuiltIn,
                [Operand::BuiltIn(BuiltIn::FragDepth)],
            );
            interface_vars.push(v);
            Some(v)
        } else {
            None
        };

        // Optional Set 0, Binding 0 UniformBuffer for D3D12.X root constants (`uvec4 words[16]`)
        let root_sgpr_limit = if stage == ShaderStage::Compute {
            60u8
        } else {
            64u8
        };
        let needs_root_ubo =
            opts.d3d12_root_bindings && live.live_sgprs.iter().any(|&r| r < root_sgpr_limit);
        let root_buf = if needs_root_ubo {
            let root_arr = b.type_array(uvec4, c_u16);
            b.decorate(
                root_arr,
                Decoration::ArrayStride,
                [Operand::LiteralBit32(16)],
            );
            let root_block = b.type_struct([root_arr]);
            b.decorate(root_block, Decoration::Block, []);
            b.member_decorate(
                root_block,
                0,
                Decoration::Offset,
                [Operand::LiteralBit32(0)],
            );
            let ptr_uni_root = b.type_pointer(None, StorageClass::Uniform, root_block);
            let var = b.variable(ptr_uni_root, None, StorageClass::Uniform, None);
            b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(0)]);
            interface_vars.push(var);
            Some(var)
        } else {
            None
        };

        // Shared StorageBuffer type (`Block { u32[] }`) for CBV/SRV and UAV buffers.
        let sb_mem_ptr_ty = if live.uses_memory || live.uses_mem_store {
            let mem_arr = b.type_runtime_array(uint);
            b.decorate(mem_arr, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
            let mem_block = b.type_struct([mem_arr]);
            b.decorate(mem_block, Decoration::Block, []);
            b.member_decorate(mem_block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
            Some(b.type_pointer(None, StorageClass::StorageBuffer, mem_block))
        } else {
            None
        };

        let (mem_binding, img_binding, samp_binding, st_binding) = if opts.d3d12_root_bindings {
            (1u32, 3u32, 4u32, 5u32)
        } else {
            (0u32, 1u32, 2u32, 3u32)
        };

        // CBV/SRV StorageBuffer (`NonWritable`)
        let memory_buf = if live.uses_memory {
            let ptr_sb_mem = sb_mem_ptr_ty.expect("sb_mem_ptr_ty");
            let var = b.variable(ptr_sb_mem, None, StorageClass::StorageBuffer, None);
            b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(
                var,
                Decoration::Binding,
                [Operand::LiteralBit32(mem_binding)],
            );
            b.decorate(var, Decoration::NonWritable, []);
            interface_vars.push(var);
            Some(var)
        } else {
            None
        };

        // UAV Read/Write StorageBuffer (Set 0, Binding 2)
        let uav_buf = if live.uses_mem_store {
            let ptr_sb_mem = sb_mem_ptr_ty.expect("sb_mem_ptr_ty");
            let var = b.variable(ptr_sb_mem, None, StorageClass::StorageBuffer, None);
            b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(2)]);
            interface_vars.push(var);
            Some(var)
        } else {
            None
        };

        let image_bindings = ImageBindings::declare(
            &mut b,
            f32_ty,
            insts,
            stage == ShaderStage::Pixel,
            [img_binding, samp_binding, st_binding, st_binding + 1],
            &mut interface_vars,
        )?;

        let mut u32_consts = BTreeMap::new();
        for &(val, id) in &[
            (0, c_u0),
            (1, c_u1),
            (2, c_u2),
            (3, c_u3),
            (16, c_u16),
            (HALT_PC, c_halt),
            (0x7FFF_FFFF, c_abs_mask),
            (0x8000_0000, c_sign_mask),
            (0x7F80_0000, c_exp_mask),
        ] {
            u32_consts.insert(val, id);
        }

        Ok(Self {
            stage,
            opts,
            mesh_layout,
            deferred_mesh,
            mesh_collective,
            mesh_exports: BTreeMap::new(),
            live,
            b,
            glsl,
            void,
            bool_ty,
            uint,
            sint,
            f32_ty,
            vec2,
            vec4,
            uvec3,
            uvec4,
            ptr_fn_uint,
            ptr_fn_bool,
            ptr_sb_uint,
            ptr_uni_uint,
            ptr_wg_uint,
            ptr_out_vec4,
            ptr_out_uvec3,
            ptr_out_bool,
            image_bindings,
            interface_vars,
            in_vertex_id,
            in_local_id,
            in_workgroup_id,
            in_local_index,
            in_subgroup_lane,
            in_locs,
            out_pos0,
            out_params,
            out_mrts,
            out_mrtz,
            out_mesh_pos,
            out_mesh_params,
            out_mesh_prim_indices,
            out_mesh_prim_cull,
            lds_buf,
            lds_dwords,
            root_buf,
            memory_buf,
            uav_buf,
            c_u0,
            c_u1,
            c_u2,
            c_u3,
            c_halt,
            c_f0,
            c_f1,
            c_f2,
            c_f4,
            c_f05,
            c_abs_mask,
            c_sign_mask,
            c_exp_mask,
            c_true,
            c_false,
            c_wg_scope,
            c_wg_barrier_sem,
            u32_consts,
            f32_consts: [(0.0_f32.to_bits(), c_f0), (1.0_f32.to_bits(), c_f1)].into(),
            var_pc: 0,
            var_active: 0,
            var_vcc: 0,
            var_scc: 0,
            var_mem_tmp: 0,
            var_mesh_vtx_cnt: 0,
            var_mesh_prim_cnt: 0,
            mesh_local_idx: 0,
            mesh_guest_lane: 0,
            mesh_lane_bit: 0,
            subgroup_lane: 0,
            subgroup_scope: c_u3,
            sgpr_vars: BTreeMap::new(),
            vgpr_vars: BTreeMap::new(),
            mem_words_len: 0,
            uav_words_len: 0,
        })
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

    fn emit_demote_raw(&mut self) -> Result<(), SpirvError> {
        // Pinned rspirv 0.12.0 closes the block in its generated `demote_to_helper_invocation()`
        // helper; insert the OpDemoteToHelperInvocation instruction directly instead.
        self.b.insert_into_block(
            InsertPoint::End,
            Instruction::new(Op::DemoteToHelperInvocation, None, None, Vec::new()),
        )?;
        Ok(())
    }

    fn emit_module(
        &mut self,
        sig: &StageSignature,
        (blocks, pc_to_block): &(Vec<GfxBlock<'_>>, BTreeMap<u32, u32>),
    ) -> Result<(), SpirvError> {
        let fn_ty = self.b.type_function(self.void, []);
        let entry = self
            .b
            .begin_function(self.void, None, FunctionControl::NONE, fn_ty)?;
        self.b.begin_block(None)?;

        // 1. Allocate ONLY pruned live Function variables at the top of the entry block
        let fn_sc = StorageClass::Function;
        self.var_pc = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        self.var_active = self.b.variable(self.ptr_fn_bool, None, fn_sc, None);
        self.var_vcc = self.b.variable(self.ptr_fn_bool, None, fn_sc, None);
        self.var_scc = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        self.var_mem_tmp = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        if self.stage == ShaderStage::Mesh {
            self.var_mesh_vtx_cnt = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
            self.var_mesh_prim_cnt = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
        }

        if self.deferred_mesh {
            let ptr_fn_vec4 = self.b.type_pointer(None, fn_sc, self.vec4);
            for target in self.mesh_export_targets() {
                let ptr = if target == 20 {
                    self.ptr_fn_uint
                } else {
                    ptr_fn_vec4
                };
                let payload = self.b.variable(ptr, None, fn_sc, None);
                let valid = self.b.variable(self.ptr_fn_bool, None, fn_sc, None);
                self.mesh_exports.insert(target, (payload, valid));
            }
        }

        let live_s: Vec<u8> = self.live.live_sgprs.iter().copied().collect();
        let live_v: Vec<u8> = self.live.live_vgprs.iter().copied().collect();
        for r in &live_s {
            let v = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
            self.sgpr_vars.insert(*r, v);
        }
        for r in &live_v {
            let v = self.b.variable(self.ptr_fn_uint, None, fn_sc, None);
            self.vgpr_vars.insert(*r, v);
        }

        for (&target, &(payload, valid)) in &self.mesh_exports {
            let zero = if target == 20 {
                self.c_u0
            } else {
                self.b.constant_null(self.vec4)
            };
            self.b.store(payload, zero, None, [])?;
            self.b.store(valid, self.c_false, None, [])?;
        }

        // 2. Zero-initialize pruned registers & set active=true, vcc=false, scc=0, pc=0
        self.b.store(self.var_pc, self.c_u0, None, [])?;
        self.b.store(self.var_active, self.c_true, None, [])?;
        self.b.store(self.var_vcc, self.c_false, None, [])?;
        self.b.store(self.var_scc, self.c_u0, None, [])?;
        if let Some(var) = self.in_subgroup_lane {
            self.subgroup_lane = self.b.load(self.uint, None, var, None, [])?;
        }
        if self.stage == ShaderStage::Mesh {
            self.b.store(self.var_mesh_vtx_cnt, self.c_u0, None, [])?;
            self.b.store(self.var_mesh_prim_cnt, self.c_u0, None, [])?;
            let li_var = self.in_local_index.expect("in_local_index");
            self.mesh_local_idx = self.b.load(self.uint, None, li_var, None, [])?;
            let wave_mask = self.const_u32(self.mesh_layout.wave_size.max(32).saturating_sub(1));
            self.mesh_guest_lane =
                self.b
                    .bitwise_and(self.uint, None, self.mesh_local_idx, wave_mask)?;
            let low_lane_mask = self.const_u32(31);
            let low_lane =
                self.b
                    .bitwise_and(self.uint, None, self.mesh_guest_lane, low_lane_mask)?;
            self.mesh_lane_bit = self
                .b
                .shift_left_logical(self.uint, None, self.c_u1, low_lane)?;
        }
        for &r in &live_s {
            let v = self.sgpr_vars[&r];
            self.b.store(v, self.c_u0, None, [])?;
        }
        for &r in &live_v {
            let v = self.vgpr_vars[&r];
            self.b.store(v, self.c_u0, None, [])?;
        }

        if let Some(mem_buf) = self.memory_buf {
            self.mem_words_len = self.b.array_length(self.uint, None, mem_buf, 0)?;
        }
        if let Some(uav_buf) = self.uav_buf {
            self.uav_words_len = self.b.array_length(self.uint, None, uav_buf, 0)?;
        }

        // 2b. Seed live user-data SGPRs from Set 0 Binding 0 RootData UBO when enabled
        if let Some(root_buf) = self.root_buf {
            let limit = if self.stage == ShaderStage::Compute {
                60u8
            } else {
                64u8
            };
            for &r in &live_s {
                if r < limit {
                    let vec_idx = self.const_u32(u32::from(r / 4));
                    let comp_idx = self.const_u32(u32::from(r % 4));
                    let p = self.b.access_chain(
                        self.ptr_uni_uint,
                        None,
                        root_buf,
                        [self.c_u0, vec_idx, comp_idx],
                    )?;
                    let val = self.b.load(self.uint, None, p, None, [])?;
                    self.store_sgpr(r, val)?;
                }
            }
        }

        // 2c. Seed Compute built-ins (`LocalInvocationId` -> v0..v2, `WorkgroupId` -> s60..s62)
        if self.stage == ShaderStage::Compute {
            if let Some(lid_var) = self.in_local_id {
                let lid = self.b.load(self.uvec3, None, lid_var, None, [])?;
                for c in 0..3u8 {
                    if self.vgpr_vars.contains_key(&c) {
                        let comp =
                            self.b
                                .composite_extract(self.uint, None, lid, [u32::from(c)])?;
                        self.store_vgpr_raw(c, comp)?;
                    }
                }
            }
            if let Some(wg_var) = self.in_workgroup_id {
                let wg = self.b.load(self.uvec3, None, wg_var, None, [])?;
                for c in 0..3u8 {
                    let sr = 60 + c;
                    if self.sgpr_vars.contains_key(&sr) {
                        let comp = self
                            .b
                            .composite_extract(self.uint, None, wg, [u32::from(c)])?;
                        self.store_sgpr(sr, comp)?;
                    }
                }
            }
        }

        // 3. Seed Vertex inputs from ISG1 & BuiltIn::VertexIndex
        if self.stage == ShaderStage::Vertex {
            if let Some(vid_var) = self.in_vertex_id {
                let vid = self.b.load(self.uint, None, vid_var, None, [])?;
                self.store_vgpr_raw(0, vid)?;
            }
            for elem in &sig.inputs {
                if elem.system_value_type == 6 || elem.register == u32::MAX || elem.register >= 32 {
                    continue;
                }
                let loc = elem.register as u8;
                if let Some(&in_var) = self.in_locs.get(&loc) {
                    let vec = self.b.load(self.vec4, None, in_var, None, [])?;
                    for c in 0..4u8 {
                        if (elem.mask & (1 << c)) != 0 {
                            let comp =
                                self.b
                                    .composite_extract(self.f32_ty, None, vec, [u32::from(c)])?;
                            let bits = self.b.bitcast(self.uint, None, comp)?;
                            self.store_vgpr_raw(loc * 4 + c, bits)?;
                        }
                    }
                }
            }
        }

        // 4. Structured CFG dispatch (single-block fast path avoids unnecessary loop when 1 block)
        if blocks.len() == 1 && blocks[0].insts.iter().all(|i| i.branch_target.is_none()) {
            for inst in &blocks[0].insts {
                self.emit_inst(inst, None, pc_to_block)?;
            }
        } else {
            let loop_header = self.b.id();
            let switch_header = self.b.id();
            let switch_merge = self.b.id();
            let loop_continue = self.b.id();
            let loop_exit = self.b.id();
            let default_label = self.b.id();
            let case_labels: Vec<(u32, u32)> =
                blocks.iter().map(|bb| (bb.id, self.b.id())).collect();

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

            self.b.begin_block(Some(loop_exit))?;
        }

        if self.deferred_mesh {
            let vtx_count = self
                .b
                .load(self.uint, None, self.var_mesh_vtx_cnt, None, [])?;
            let prim_count = self
                .b
                .load(self.uint, None, self.var_mesh_prim_cnt, None, [])?;
            self.b.set_mesh_outputs_ext(vtx_count, prim_count)?;
            for target in self.mesh_export_targets() {
                let (payload, valid) = self.mesh_exports[&target];
                let ty = if target == 20 { self.uint } else { self.vec4 };
                let payload = self.b.load(ty, None, payload, None, [])?;
                let valid = self.b.load(self.bool_ty, None, valid, None, [])?;
                self.write_mesh_export(target, payload, valid)?;
            }
        }

        // If Fragment shader uses demote, demote any invocation whose final EXEC bit is clear
        if self.stage == ShaderStage::Pixel && self.live.uses_demote {
            let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
            let inactive = self.b.logical_not(self.bool_ty, None, active)?;
            let demote_blk = self.b.id();
            let merge_blk = self.b.id();
            self.b.selection_merge(merge_blk, SelectionControl::NONE)?;
            self.b
                .branch_conditional(inactive, demote_blk, merge_blk, [])?;
            self.b.begin_block(Some(demote_blk))?;
            self.emit_demote_raw()?;
            self.b.branch(merge_blk)?;
            self.b.begin_block(Some(merge_blk))?;
        }

        self.b.ret()?;
        self.b.end_function()?;

        let exec_model = match self.stage {
            ShaderStage::Vertex => ExecutionModel::Vertex,
            ShaderStage::Pixel => ExecutionModel::Fragment,
            ShaderStage::Compute => ExecutionModel::GLCompute,
            ShaderStage::Mesh => ExecutionModel::MeshEXT,
            _ => unreachable!(),
        };
        let iface = self.interface_vars.clone();
        self.b.entry_point(exec_model, entry, "main", iface);

        if self.stage == ShaderStage::Pixel {
            self.b
                .execution_mode(entry, ExecutionMode::OriginUpperLeft, []);
            if self.live.exports_mrtz {
                self.b
                    .execution_mode(entry, ExecutionMode::DepthReplacing, []);
            }
        } else if self.stage == ShaderStage::Compute {
            let [lx, ly, lz] = self.opts.compute_local_size;
            let (lx, ly, lz) = if lx == 0 && ly == 0 && lz == 0 {
                (64, 1, 1)
            } else {
                (lx.max(1), ly.max(1), lz.max(1))
            };
            self.b
                .execution_mode(entry, ExecutionMode::LocalSize, [lx, ly, lz]);
        } else if self.stage == ShaderStage::Mesh {
            let [lx, ly, lz] = self.mesh_layout.local_size;
            self.b.execution_mode(
                entry,
                ExecutionMode::LocalSize,
                [lx.max(1), ly.max(1), lz.max(1)],
            );
            self.b.execution_mode(
                entry,
                ExecutionMode::OutputVertices,
                [self.mesh_layout.max_vertices.max(1)],
            );
            self.b.execution_mode(
                entry,
                ExecutionMode::OutputPrimitivesEXT,
                [self.mesh_layout.max_primitives.max(1)],
            );
            self.b
                .execution_mode(entry, ExecutionMode::OutputTrianglesEXT, []);
        }
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
    // Pruned register load/store helpers
    // -----------------------------------------------------------------------

    fn load_sgpr(&mut self, r: u8) -> Result<u32, SpirvError> {
        if let Some(&var) = self.sgpr_vars.get(&r) {
            Ok(self.b.load(self.uint, None, var, None, [])?)
        } else {
            Ok(self.c_u0)
        }
    }

    fn store_sgpr(&mut self, r: u8, val: u32) -> Result<(), SpirvError> {
        if let Some(&var) = self.sgpr_vars.get(&r) {
            self.b.store(var, val, None, [])?;
        }
        Ok(())
    }

    fn load_vgpr(&mut self, r: u8) -> Result<u32, SpirvError> {
        if let Some(&var) = self.vgpr_vars.get(&r) {
            Ok(self.b.load(self.uint, None, var, None, [])?)
        } else {
            Ok(self.c_u0)
        }
    }

    fn store_vgpr_raw(&mut self, r: u8, val: u32) -> Result<(), SpirvError> {
        if let Some(&var) = self.vgpr_vars.get(&r) {
            self.b.store(var, val, None, [])?;
        }
        Ok(())
    }

    fn store_vgpr_active(&mut self, r: u8, val: u32) -> Result<(), SpirvError> {
        if let Some(&var) = self.vgpr_vars.get(&r) {
            let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
            let old = self.b.load(self.uint, None, var, None, [])?;
            let merged = self.b.select(self.uint, None, active, val, old)?;
            self.b.store(var, merged, None, [])?;
        }
        Ok(())
    }

    fn uses_wave_masks(&self) -> bool {
        self.deferred_mesh || self.in_subgroup_lane.is_some()
    }

    fn mesh_barrier(&mut self) -> Result<(), SpirvError> {
        Ok(self
            .b
            .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_barrier_sem)?)
    }

    fn mesh_mask_words(&mut self, predicate: u32) -> Result<(u32, u32), SpirvError> {
        let shared = self.mesh_collective.expect("single guest wave workgroup");
        let lo = self
            .b
            .access_chain(self.ptr_wg_uint, None, shared, [self.c_u0])?;
        let hi = self
            .b
            .access_chain(self.ptr_wg_uint, None, shared, [self.c_u1])?;
        let leader = self
            .b
            .i_equal(self.bool_ty, None, self.mesh_local_idx, self.c_u0)?;
        let clear = self.b.id();
        let cleared = self.b.id();
        self.b.selection_merge(cleared, SelectionControl::NONE)?;
        self.b.branch_conditional(leader, clear, cleared, [])?;
        self.b.begin_block(Some(clear))?;
        self.b.store(lo, self.c_u0, None, [])?;
        self.b.store(hi, self.c_u0, None, [])?;
        self.b.branch(cleared)?;
        self.b.begin_block(Some(cleared))?;
        self.mesh_barrier()?;

        let contribute = self.b.id();
        let voted = self.b.id();
        self.b.selection_merge(voted, SelectionControl::NONE)?;
        self.b
            .branch_conditional(predicate, contribute, voted, [])?;
        self.b.begin_block(Some(contribute))?;
        let c5 = self.const_u32(5);
        let word = self
            .b
            .shift_right_logical(self.uint, None, self.mesh_guest_lane, c5)?;
        let ptr = self
            .b
            .access_chain(self.ptr_wg_uint, None, shared, [word])?;
        self.b.atomic_or(
            self.uint,
            None,
            ptr,
            self.c_wg_scope,
            self.c_u0,
            self.mesh_lane_bit,
        )?;
        self.b.branch(voted)?;
        self.b.begin_block(Some(voted))?;
        self.mesh_barrier()?;
        let lo = self.b.load(self.uint, None, lo, None, [])?;
        let hi = self.b.load(self.uint, None, hi, None, [])?;
        self.mesh_barrier()?;
        Ok((lo, hi))
    }

    fn mesh_read_first(&mut self, active: u32, value: u32) -> Result<u32, SpirvError> {
        let shared = self.mesh_collective.expect("single guest wave workgroup");
        let slot = self
            .b
            .i_add(self.uint, None, self.mesh_local_idx, self.c_u2)?;
        let ptr = self
            .b
            .access_chain(self.ptr_wg_uint, None, shared, [slot])?;
        self.b.store(ptr, value, None, [])?;
        let (lo, hi) = self.mesh_mask_words(active)?;
        let (lo_nonzero, hi_nonzero) = (
            self.b.i_not_equal(self.bool_ty, None, lo, self.c_u0)?,
            self.b.i_not_equal(self.bool_ty, None, hi, self.c_u0)?,
        );
        let low_lane = self.b.ext_inst(
            self.uint,
            None,
            self.glsl,
            GLOp::FindILsb as u32,
            [Operand::IdRef(lo)],
        )?;
        let high_lane = self.b.ext_inst(
            self.uint,
            None,
            self.glsl,
            GLOp::FindILsb as u32,
            [Operand::IdRef(hi)],
        )?;
        let c32 = self.const_u32(32);
        let high_lane = self.b.i_add(self.uint, None, high_lane, c32)?;
        let high_or_zero = self
            .b
            .select(self.uint, None, hi_nonzero, high_lane, self.c_u0)?;
        let first = self
            .b
            .select(self.uint, None, lo_nonzero, low_lane, high_or_zero)?;
        let slot = self.b.i_add(self.uint, None, first, self.c_u2)?;
        let ptr = self
            .b
            .access_chain(self.ptr_wg_uint, None, shared, [slot])?;
        let scalar = self.b.load(self.uint, None, ptr, None, [])?;
        self.mesh_barrier()?;
        Ok(scalar)
    }

    fn ballot(&mut self, predicate: u32) -> Result<u32, SpirvError> {
        Ok(self
            .b
            .group_non_uniform_ballot(self.uvec4, None, self.subgroup_scope, predicate)?)
    }

    fn mask_words(&mut self, var: u32) -> Result<(u32, u32), SpirvError> {
        let predicate = self.b.load(self.bool_ty, None, var, None, [])?;
        self.predicate_mask_words(predicate)
    }

    fn predicate_mask_words(&mut self, predicate: u32) -> Result<(u32, u32), SpirvError> {
        if self.deferred_mesh {
            return self.mesh_mask_words(predicate);
        }
        let ballot = self.ballot(predicate)?;
        Ok((
            self.b.composite_extract(self.uint, None, ballot, [0])?,
            self.b.composite_extract(self.uint, None, ballot, [1])?,
        ))
    }

    fn mask_nonzero(&mut self, lo: u32, hi: u32) -> Result<u32, SpirvError> {
        let mask = self.b.bitwise_or(self.uint, None, lo, hi)?;
        Ok(self.b.i_not_equal(self.bool_ty, None, mask, self.c_u0)?)
    }

    fn mask_lane(&mut self, lo: u32, hi: u32) -> Result<u32, SpirvError> {
        let c31 = self.const_u32(31);
        let lane_id = if self.deferred_mesh {
            self.mesh_guest_lane
        } else {
            self.subgroup_lane
        };
        let lane = self.b.bitwise_and(self.uint, None, lane_id, c31)?;
        let low = self.b.u_less_than_equal(self.bool_ty, None, lane_id, c31)?;
        let word = self.b.select(self.uint, None, low, lo, hi)?;
        let bit = self
            .b
            .bit_field_u_extract(self.uint, None, word, lane, self.c_u1)?;
        Ok(self.b.i_not_equal(self.bool_ty, None, bit, self.c_u0)?)
    }

    fn load_scalar_b64(&mut self, pc: u32, src: SrcOperand) -> Result<(u32, u32), SpirvError> {
        match src {
            SrcOperand::ExecLo => self.mask_words(self.var_active),
            SrcOperand::VccLo => self.mask_words(self.var_vcc),
            SrcOperand::Sgpr(reg) if reg < 127 => {
                Ok((self.load_sgpr(reg)?, self.load_sgpr(reg + 1)?))
            }
            SrcOperand::InlineInt(value) => {
                let lo = self.const_u32(value as u32);
                let hi = self.const_u32(if value < 0 { u32::MAX } else { 0 });
                Ok((lo, hi))
            }
            _ => Ok((self.load_src_u32(pc, src)?, self.c_u0)),
        }
    }

    fn store_scalar_b64(&mut self, dst: DstOperand, lo: u32, hi: u32) -> Result<(), SpirvError> {
        match dst {
            DstOperand::ExecLo | DstOperand::VccLo => {
                let predicate = self.mask_lane(lo, hi)?;
                let var = if dst == DstOperand::ExecLo {
                    self.var_active
                } else {
                    self.var_vcc
                };
                self.b.store(var, predicate, None, [])?;
                Ok(())
            }
            DstOperand::Sgpr(reg) if reg < 127 => {
                self.store_sgpr(reg, lo)?;
                self.store_sgpr(reg + 1, hi)
            }
            _ => self.store_scalar_dst(dst, lo),
        }
    }

    fn load_src_u32(&mut self, pc: u32, src: SrcOperand) -> Result<u32, SpirvError> {
        if self.uses_wave_masks() {
            match src {
                SrcOperand::ExecLo => return Ok(self.mask_words(self.var_active)?.0),
                SrcOperand::ExecHi => return Ok(self.mask_words(self.var_active)?.1),
                SrcOperand::VccLo => return Ok(self.mask_words(self.var_vcc)?.0),
                SrcOperand::VccHi => return Ok(self.mask_words(self.var_vcc)?.1),
                _ => {}
            }
        }
        match src {
            SrcOperand::Sgpr(n) => self.load_sgpr(n),
            SrcOperand::Vgpr(n) => self.load_vgpr(n),
            SrcOperand::VccLo => {
                let b = self.b.load(self.bool_ty, None, self.var_vcc, None, [])?;
                Ok(self.b.select(self.uint, None, b, self.c_u1, self.c_u0)?)
            }
            SrcOperand::ExecLo => {
                let b = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
                Ok(self.b.select(self.uint, None, b, self.c_u1, self.c_u0)?)
            }
            SrcOperand::VccHi | SrcOperand::ExecHi | SrcOperand::Null => Ok(self.c_u0),
            SrcOperand::M0 => self.load_sgpr(124),
            SrcOperand::Scc => self
                .b
                .load(self.uint, None, self.var_scc, None, [])
                .map_err(Into::into),
            SrcOperand::Vccz => {
                let v = self.b.load(self.bool_ty, None, self.var_vcc, None, [])?;
                let z = self.b.logical_not(self.bool_ty, None, v)?;
                Ok(self.b.select(self.uint, None, z, self.c_u1, self.c_u0)?)
            }
            SrcOperand::Execz => {
                let v = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
                let z = self.b.logical_not(self.bool_ty, None, v)?;
                Ok(self.b.select(self.uint, None, z, self.c_u1, self.c_u0)?)
            }
            SrcOperand::InlineInt(v) => Ok(self.const_u32(v as u32)),
            SrcOperand::InlineFloat(f) => Ok(self.const_u32(f.to_bits())),
            SrcOperand::Literal(bits) => Ok(self.const_u32(bits)),
            SrcOperand::Special(c) => Err(SpirvError::UnsupportedOperand {
                pc,
                detail: format!("special operand {c}"),
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
        let bits = self.load_src_u32(pc, src)?;
        let bits = self
            .alu()
            .apply_f32_sign_mods(bits, mods.abs[slot], mods.neg[slot])?;
        Ok(self.b.bitcast(self.f32_ty, None, bits)?)
    }

    fn apply_out_mods(&mut self, mut val: u32, mods: ValuModifiers) -> Result<u32, SpirvError> {
        val = match mods.omod {
            1 => self.b.f_mul(self.f32_ty, None, val, self.c_f2)?,
            2 => self.b.f_mul(self.f32_ty, None, val, self.c_f4)?,
            3 => self.b.f_mul(self.f32_ty, None, val, self.c_f05)?,
            _ => val,
        };
        if mods.clamp {
            let bits = self.b.bitcast(self.uint, None, val)?;
            let abs_b = self.b.bitwise_and(self.uint, None, bits, self.c_abs_mask)?;
            let is_nan = self
                .b
                .u_greater_than(self.bool_ty, None, abs_b, self.c_exp_mask)?;
            let le_z = self
                .b
                .f_ord_less_than_equal(self.bool_ty, None, val, self.c_f0)?;
            let ge_o = self
                .b
                .f_ord_greater_than_equal(self.bool_ty, None, val, self.c_f1)?;
            let hi = self.b.select(self.f32_ty, None, ge_o, self.c_f1, val)?;
            let zh = self.b.select(self.f32_ty, None, le_z, self.c_f0, hi)?;
            val = self.b.select(self.f32_ty, None, is_nan, self.c_f0, zh)?;
        }
        Ok(val)
    }

    fn store_scc(&mut self, cond: u32) -> Result<(), SpirvError> {
        let w = self.b.select(self.uint, None, cond, self.c_u1, self.c_u0)?;
        self.b.store(self.var_scc, w, None, [])?;
        Ok(())
    }

    fn store_scalar_dst(&mut self, dst: DstOperand, val: u32) -> Result<(), SpirvError> {
        if self.uses_wave_masks() {
            let mask = match dst {
                DstOperand::ExecLo | DstOperand::ExecHi => Some(self.var_active),
                DstOperand::VccLo | DstOperand::VccHi => Some(self.var_vcc),
                _ => None,
            };
            if let Some(var) = mask {
                let (lo, hi) = self.mask_words(var)?;
                return match dst {
                    DstOperand::ExecLo | DstOperand::VccLo => self.store_scalar_b64(dst, val, hi),
                    DstOperand::ExecHi => self.store_scalar_b64(DstOperand::ExecLo, lo, val),
                    DstOperand::VccHi => self.store_scalar_b64(DstOperand::VccLo, lo, val),
                    _ => unreachable!(),
                };
            }
        }
        match dst {
            DstOperand::Sgpr(n) => self.store_sgpr(n, val),
            DstOperand::M0 => self.store_sgpr(124, val),
            DstOperand::VccLo => {
                let b = self.b.i_not_equal(self.bool_ty, None, val, self.c_u0)?;
                self.b.store(self.var_vcc, b, None, [])?;
                Ok(())
            }
            DstOperand::ExecLo => {
                let b = if self.stage == ShaderStage::Mesh {
                    let masked = self
                        .b
                        .bitwise_and(self.uint, None, val, self.mesh_lane_bit)?;
                    self.b.i_not_equal(self.bool_ty, None, masked, self.c_u0)?
                } else {
                    self.b.i_not_equal(self.bool_ty, None, val, self.c_u0)?
                };
                self.b.store(self.var_active, b, None, [])?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // Instruction lowering
    // -----------------------------------------------------------------------

    fn emit_inst(
        &mut self,
        inst: &DecodedInst,
        fallthrough: Option<u32>,
        pc_to_block: &BTreeMap<u32, u32>,
    ) -> Result<bool, SpirvError> {
        if inst.mnemonic == "s_swappc_b64" {
            // Vertex fetch-shader call bypassed because Vulkan already loaded ISG1 inputs.
            return Ok(false);
        }
        if matches!(inst.ext, ValuExt::Sdwa { .. }) {
            sdwa_half_offset(inst)?;
        }
        match inst.encoding {
            Encoding::Sopp => self.emit_sopp(inst, fallthrough, pc_to_block),
            Encoding::Sop1 => {
                self.emit_sop1(inst)?;
                Ok(false)
            }
            Encoding::Sop2 => {
                self.emit_sop2(inst)?;
                Ok(false)
            }
            Encoding::Sopk => {
                self.emit_sopk(inst)?;
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
            Encoding::Vop3p => {
                let DstOperand::Vgpr(dst) = inst.dst else {
                    unreachable!("packed VALU requires a VGPR destination");
                };
                let bits = inst
                    .srcs
                    .iter()
                    .map(|&src| self.load_src_u32(inst.pc, src))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = self.alu().emit_packed_half(inst, &bits)?;
                self.store_vgpr_active(dst, result)?;
                Ok(false)
            }
            Encoding::Vintrp => {
                self.emit_vintrp(inst)?;
                Ok(false)
            }
            Encoding::Export => {
                self.emit_exp(inst)?;
                Ok(false)
            }
            Encoding::Mimg => {
                self.emit_mimg(inst)?;
                Ok(false)
            }
            Encoding::Ds => {
                self.emit_ds(inst)?;
                Ok(false)
            }
            Encoding::Smem | Encoding::Vmem => {
                self.emit_mem_load(inst)?;
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
                // Convergent workgroup barrier — never gated on per-lane `var_active`.
                self.b
                    .control_barrier(self.c_wg_scope, self.c_wg_scope, self.c_wg_barrier_sem)?;
                Ok(false)
            }
            "s_sendmsg" => {
                if self.stage != ShaderStage::Mesh || inst.simm16 != Some(9) {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: inst.mnemonic.clone(),
                    });
                }
                // AMD RDNA2 ISA §12.5.1: MSG_GS_ALLOC_REQ reads m0[9:0] = vertices, m0[22:12] = primitives.
                let m0 = self.load_sgpr(124)?;
                let c_vtx_mask = self.const_u32(0x3FF);
                let c_prim_shift = self.const_u32(12);
                let c_prim_mask = self.const_u32(0x7FF);
                let c_max_v = self.const_u32(self.mesh_layout.max_vertices.max(1));
                let c_max_p = self.const_u32(self.mesh_layout.max_primitives.max(1));
                let raw_vtx = self.b.bitwise_and(self.uint, None, m0, c_vtx_mask)?;
                let shifted = self
                    .b
                    .shift_right_logical(self.uint, None, m0, c_prim_shift)?;
                let raw_prim = self.b.bitwise_and(self.uint, None, shifted, c_prim_mask)?;
                let vtx_cnt = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::UMin as u32,
                    [Operand::IdRef(raw_vtx), Operand::IdRef(c_max_v)],
                )?;
                let prim_cnt = self.b.ext_inst(
                    self.uint,
                    None,
                    self.glsl,
                    GLOp::UMin as u32,
                    [Operand::IdRef(raw_prim), Operand::IdRef(c_max_p)],
                )?;
                self.b.store(self.var_mesh_vtx_cnt, vtx_cnt, None, [])?;
                self.b.store(self.var_mesh_prim_cnt, prim_cnt, None, [])?;
                if !self.deferred_mesh {
                    self.b.set_mesh_outputs_ext(vtx_cnt, prim_cnt)?;
                }
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
                let blk = *pc_to_block
                    .get(&target_pc)
                    .ok_or(SpirvError::InvalidBranchTarget {
                        pc: inst.pc,
                        target: target_pc,
                    })?;
                let c = self.const_u32(blk);
                self.b.store(self.var_pc, c, None, [])?;
                Ok(true)
            }
            "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_vccz" | "s_cbranch_vccnz"
            | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let target_pc = inst.branch_target.ok_or(SpirvError::InvalidBranchTarget {
                    pc: inst.pc,
                    target: 0,
                })?;
                let blk = *pc_to_block
                    .get(&target_pc)
                    .ok_or(SpirvError::InvalidBranchTarget {
                        pc: inst.pc,
                        target: target_pc,
                    })?;
                let ft = fallthrough.unwrap_or(HALT_PC);
                let tc = self.const_u32(blk);
                let fc = self.const_u32(ft);
                let cond = match inst.mnemonic.as_str() {
                    "s_cbranch_scc0" => {
                        let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
                        self.b.i_equal(self.bool_ty, None, scc, self.c_u0)?
                    }
                    "s_cbranch_scc1" => {
                        let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
                        self.b.i_not_equal(self.bool_ty, None, scc, self.c_u0)?
                    }
                    branch => {
                        let var = if branch.starts_with("s_cbranch_vcc") {
                            self.var_vcc
                        } else {
                            self.var_active
                        };
                        let nonzero = if self.uses_wave_masks() {
                            let (lo, hi) = self.mask_words(var)?;
                            self.mask_nonzero(lo, hi)?
                        } else {
                            self.b.load(self.bool_ty, None, var, None, [])?
                        };
                        if branch.ends_with("nz") {
                            nonzero
                        } else {
                            self.b.logical_not(self.bool_ty, None, nonzero)?
                        }
                    }
                };
                let next = self.b.select(self.uint, None, cond, tc, fc)?;
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
        let s0 = inst.srcs[0];
        if matches!(
            inst.mnemonic.as_str(),
            "s_bcnt1_i32_b32" | "s_bcnt1_i32_b64"
        ) {
            let (lo, hi) = if inst.opcode == 0x10 {
                let (lo, hi) = self.load_scalar_b64(inst.pc, s0)?;
                (lo, Some(hi))
            } else {
                (self.load_src_u32(inst.pc, s0)?, None)
            };
            let (result, nonzero) = self.alu().count_bits(lo, hi)?;
            self.store_scalar_dst(inst.dst, result)?;
            return self.store_scc(nonzero);
        }
        if self.uses_wave_masks() && inst.mnemonic.ends_with("_b64") {
            let (lo, hi) = self.load_scalar_b64(inst.pc, s0)?;
            let mnemonic = inst.mnemonic.as_str();
            match mnemonic {
                "s_mov_b64" => return self.store_scalar_b64(inst.dst, lo, hi),
                "s_wqm_b64" => {
                    let lo = self.alu().whole_quad_mask(lo)?;
                    let hi = self.alu().whole_quad_mask(hi)?;
                    self.store_scalar_b64(inst.dst, lo, hi)?;
                    let nonzero = self.mask_nonzero(lo, hi)?;
                    return self.store_scc(nonzero);
                }
                "s_and_saveexec_b64"
                | "s_or_saveexec_b64"
                | "s_xor_saveexec_b64"
                | "s_andn2_saveexec_b64"
                | "s_andn2_wrexec_b64" => {
                    let (exec_lo, exec_hi) = self.mask_words(self.var_active)?;
                    let (new_lo, new_hi) = match mnemonic {
                        "s_and_saveexec_b64" => (
                            self.b.bitwise_and(self.uint, None, lo, exec_lo)?,
                            self.b.bitwise_and(self.uint, None, hi, exec_hi)?,
                        ),
                        "s_or_saveexec_b64" => (
                            self.b.bitwise_or(self.uint, None, lo, exec_lo)?,
                            self.b.bitwise_or(self.uint, None, hi, exec_hi)?,
                        ),
                        "s_xor_saveexec_b64" => (
                            self.b.bitwise_xor(self.uint, None, lo, exec_lo)?,
                            self.b.bitwise_xor(self.uint, None, hi, exec_hi)?,
                        ),
                        _ => {
                            let not_lo = self.b.not(self.uint, None, exec_lo)?;
                            let not_hi = self.b.not(self.uint, None, exec_hi)?;
                            (
                                self.b.bitwise_and(self.uint, None, lo, not_lo)?,
                                self.b.bitwise_and(self.uint, None, hi, not_hi)?,
                            )
                        }
                    };
                    if mnemonic == "s_andn2_wrexec_b64" {
                        self.store_scalar_b64(inst.dst, new_lo, new_hi)?;
                    } else {
                        self.store_scalar_b64(inst.dst, exec_lo, exec_hi)?;
                    }
                    self.store_scalar_b64(DstOperand::ExecLo, new_lo, new_hi)?;
                    let nonzero = self.mask_nonzero(new_lo, new_hi)?;
                    return self.store_scc(nonzero);
                }
                _ => {}
            }
        }
        match inst.mnemonic.as_str() {
            "s_mov_b32" | "s_mov_b64" => {
                let v = self.load_src_u32(inst.pc, s0)?;
                self.store_scalar_dst(inst.dst, v)
            }
            op @ ("s_and_saveexec_b32"
            | "s_or_saveexec_b32"
            | "s_xor_saveexec_b32"
            | "s_andn2_saveexec_b32"
            | "s_andn2_wrexec_b32")
                if self.uses_wave_masks() =>
            {
                let src = self.load_src_u32(inst.pc, s0)?;
                let (exec, _) = self.mask_words(self.var_active)?;
                let new_exec = match op {
                    "s_and_saveexec_b32" => self.b.bitwise_and(self.uint, None, src, exec)?,
                    "s_or_saveexec_b32" => self.b.bitwise_or(self.uint, None, src, exec)?,
                    "s_xor_saveexec_b32" => self.b.bitwise_xor(self.uint, None, src, exec)?,
                    _ => {
                        let inverted = self.b.not(self.uint, None, exec)?;
                        self.b.bitwise_and(self.uint, None, src, inverted)?
                    }
                };
                let dst = if op == "s_andn2_wrexec_b32" {
                    new_exec
                } else {
                    exec
                };
                self.store_scalar_dst(inst.dst, dst)?;
                self.store_scalar_dst(DstOperand::ExecLo, new_exec)?;
                let nonzero = self
                    .b
                    .i_not_equal(self.bool_ty, None, new_exec, self.c_u0)?;
                self.store_scc(nonzero)
            }
            "s_and_saveexec_b32" | "s_and_saveexec_b64" => {
                let s_val = self.load_src_u32(inst.pc, s0)?;
                let s_bool = self.b.i_not_equal(self.bool_ty, None, s_val, self.c_u0)?;
                let old_a = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
                let old_u = self
                    .b
                    .select(self.uint, None, old_a, self.c_u1, self.c_u0)?;
                let new_a = self.b.logical_and(self.bool_ty, None, old_a, s_bool)?;
                self.store_scalar_dst(inst.dst, old_u)?;
                self.b.store(self.var_active, new_a, None, [])?;
                self.store_scc(new_a)
            }
            "s_or_saveexec_b32" | "s_or_saveexec_b64" => {
                let s_val = self.load_src_u32(inst.pc, s0)?;
                let s_bool = self.b.i_not_equal(self.bool_ty, None, s_val, self.c_u0)?;
                let old_a = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
                let old_u = self
                    .b
                    .select(self.uint, None, old_a, self.c_u1, self.c_u0)?;
                let new_a = self.b.logical_or(self.bool_ty, None, old_a, s_bool)?;
                self.store_scalar_dst(inst.dst, old_u)?;
                self.b.store(self.var_active, new_a, None, [])?;
                self.store_scc(new_a)
            }
            "s_wqm_b32" => {
                let src = self.load_src_u32(inst.pc, s0)?;
                let result = self.alu().whole_quad_mask(src)?;
                let nonzero = self.b.i_not_equal(self.bool_ty, None, result, self.c_u0)?;
                self.store_scalar_dst(inst.dst, result)?;
                self.store_scc(nonzero)
            }
            "s_not_b32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                let r = self.b.not(self.uint, None, a)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sopk(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let imm = self.const_u32(i32::from(inst.simm16.unwrap_or(0)) as u32);
        match inst.mnemonic.as_str() {
            "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt" | "s_waitcnt_lgkmcnt" => {
                Ok(())
            }
            "s_movk_i32" => self.store_scalar_dst(inst.dst, imm),
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sop2(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if self.uses_wave_masks() && inst.mnemonic.ends_with("_b64") {
            let (a_lo, a_hi) = self.load_scalar_b64(inst.pc, inst.srcs[0])?;
            let (b_lo, b_hi) = self.load_scalar_b64(inst.pc, inst.srcs[1])?;
            let (lo, hi) = match inst.mnemonic.as_str() {
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
                    let not_lo = self.b.not(self.uint, None, b_lo)?;
                    let not_hi = self.b.not(self.uint, None, b_hi)?;
                    (
                        self.b.bitwise_and(self.uint, None, a_lo, not_lo)?,
                        self.b.bitwise_and(self.uint, None, a_hi, not_hi)?,
                    )
                }
                "s_cselect_b64" => {
                    let scc = self.b.load(self.uint, None, self.var_scc, None, [])?;
                    let predicate = self.b.i_not_equal(self.bool_ty, None, scc, self.c_u0)?;
                    let lo = self.b.select(self.uint, None, predicate, a_lo, b_lo)?;
                    let hi = self.b.select(self.uint, None, predicate, a_hi, b_hi)?;
                    return self.store_scalar_b64(inst.dst, lo, hi);
                }
                _ => {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: inst.mnemonic.clone(),
                    });
                }
            };
            self.store_scalar_b64(inst.dst, lo, hi)?;
            let nonzero = self.mask_nonzero(lo, hi)?;
            return self.store_scc(nonzero);
        }
        let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
        let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
        match inst.mnemonic.as_str() {
            "s_add_u32" | "s_add_i32" => {
                let r = self.b.i_add(self.uint, None, a, b)?;
                self.store_scalar_dst(inst.dst, r)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let r = self.b.i_sub(self.uint, None, a, b)?;
                self.store_scalar_dst(inst.dst, r)
            }
            "s_and_b32" | "s_and_b64" => {
                let r = self.b.bitwise_and(self.uint, None, a, b)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            "s_or_b32" | "s_or_b64" => {
                let r = self.b.bitwise_or(self.uint, None, a, b)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            "s_xor_b32" | "s_xor_b64" => {
                let r = self.b.bitwise_xor(self.uint, None, a, b)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            "s_andn2_b32" | "s_andn2_b64" => {
                let nb = self.b.not(self.uint, None, b)?;
                let r = self.b.bitwise_and(self.uint, None, a, nb)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            "s_bfe_u32" => {
                let result = self.alu().scalar_bfe_u32(a, b)?;
                let nonzero = self.b.i_not_equal(self.bool_ty, None, result, self.c_u0)?;
                self.store_scalar_dst(inst.dst, result)?;
                self.store_scc(nonzero)
            }
            "s_mul_i32" => {
                let r = self.b.i_mul(self.uint, None, a, b)?;
                self.store_scalar_dst(inst.dst, r)
            }
            "s_lshl_b32" => {
                let c31 = self.const_u32(31);
                let sh = self.b.bitwise_and(self.uint, None, b, c31)?;
                let r = self.b.shift_left_logical(self.uint, None, a, sh)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            "s_lshr_b32" => {
                let c31 = self.const_u32(31);
                let sh = self.b.bitwise_and(self.uint, None, b, c31)?;
                let r = self.b.shift_right_logical(self.uint, None, a, sh)?;
                let nz = self.b.i_not_equal(self.bool_ty, None, r, self.c_u0)?;
                self.store_scalar_dst(inst.dst, r)?;
                self.store_scc(nz)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_sopc(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
        let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
        let p = match inst.mnemonic.as_str() {
            "s_cmp_eq_i32" | "s_cmp_eq_u32" => self.b.i_equal(self.bool_ty, None, a, b)?,
            "s_cmp_lg_i32" | "s_cmp_lg_u32" => self.b.i_not_equal(self.bool_ty, None, a, b)?,
            "s_cmp_lt_u32" => self.b.u_less_than(self.bool_ty, None, a, b)?,
            "s_cmp_gt_u32" => self.b.u_greater_than(self.bool_ty, None, a, b)?,
            _ => {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: inst.mnemonic.clone(),
                });
            }
        };
        self.store_scc(p)
    }

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
            let value = self.load_src_u32(inst.pc, inst.srcs[0])?;
            let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
            if self.deferred_mesh {
                let scalar = self.mesh_read_first(active, value)?;
                return self.store_scalar_dst(inst.dst, scalar);
            }
            let ballot = self.ballot(active)?;
            let lo = self.b.composite_extract(self.uint, None, ballot, [0])?;
            let hi = self.b.composite_extract(self.uint, None, ballot, [1])?;
            let has_active = self.mask_nonzero(lo, hi)?;
            // The ISA reads lane zero if EXEC is empty. Broadcast within the
            // EXEC lanes, then publish the scalar result to the whole subgroup.
            let empty = self.b.logical_not(self.bool_ty, None, has_active)?;
            let participate = self.b.logical_or(self.bool_ty, None, active, empty)?;
            let broadcast = self.b.id();
            let skip = self.b.id();
            let merge = self.b.id();
            self.b.selection_merge(merge, SelectionControl::NONE)?;
            self.b
                .branch_conditional(participate, broadcast, skip, [])?;
            self.b.begin_block(Some(broadcast))?;
            let first_value = self.b.group_non_uniform_broadcast_first(
                self.uint,
                None,
                self.subgroup_scope,
                value,
            )?;
            self.b.branch(merge)?;
            self.b.begin_block(Some(skip))?;
            self.b.branch(merge)?;
            self.b.begin_block(Some(merge))?;
            let scalar = self.b.phi(
                self.uint,
                None,
                [(first_value, broadcast), (self.c_u0, skip)],
            )?;
            let first = self.b.group_non_uniform_ballot_find_lsb(
                self.uint,
                None,
                self.subgroup_scope,
                ballot,
            )?;
            let first = self
                .b
                .select(self.uint, None, has_active, first, self.subgroup_lane)?;
            let scalar = self.b.group_non_uniform_shuffle(
                self.uint,
                None,
                self.subgroup_scope,
                scalar,
                first,
            )?;
            return self.store_scalar_dst(inst.dst, scalar);
        }
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };
        let s0 = inst.srcs[0];
        let res = match inst.mnemonic.as_str() {
            "v_mov_b32" => self.load_src_u32(inst.pc, s0)?,
            "v_not_b32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                self.b.not(self.uint, None, a)?
            }
            "v_bfrev_b32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                self.b.bit_reverse(self.uint, None, a)?
            }
            "v_cvt_f32_i32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                let sa = self.b.bitcast(self.sint, None, a)?;
                let f = self.b.convert_s_to_f(self.f32_ty, None, sa)?;
                let f = self.apply_out_mods(f, inst.modifiers)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_cvt_f32_u32" => {
                let a = self.load_src_u32(inst.pc, s0)?;
                let f = self.b.convert_u_to_f(self.f32_ty, None, a)?;
                let f = self.apply_out_mods(f, inst.modifiers)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_cvt_f16_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let f = self.apply_out_mods(f, inst.modifiers)?;
                let bits = self.b.bitcast(self.uint, None, f)?;
                let packed = self
                    .alu()
                    .emit_f32_to_f16(bits, HalfRounding::NearestEven)?;
                let offset = match inst.ext {
                    ValuExt::Sdwa { .. } => sdwa_half_offset(inst)?,
                    _ => 0,
                };
                let offset = self.const_u32(offset);
                let width = self.const_u32(16);
                let old = self.load_vgpr(vdst)?;
                self.b
                    .bit_field_insert(self.uint, None, old, packed, offset, width)?
            }
            "v_cvt_f32_f16" => {
                let mut h16 = if let SrcOperand::InlineFloat(v) = s0 {
                    self.const_u32(u32::from(inline_float_to_f16_bits(v)))
                } else {
                    let raw = self.load_src_u32(inst.pc, s0)?;
                    let c_lo_mask = self.const_u32(0x0000_FFFF);
                    self.b.bitwise_and(self.uint, None, raw, c_lo_mask)?
                };
                if inst.modifiers.abs[0] {
                    let c_abs16 = self.const_u32(0x7FFF);
                    h16 = self.b.bitwise_and(self.uint, None, h16, c_abs16)?;
                }
                if inst.modifiers.neg[0] {
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
                let f = self.apply_out_mods(f, inst.modifiers)?;
                self.b.bitcast(self.uint, None, f)?
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let r = self.b.f_div(self.f32_ty, None, self.c_f1, f)?;
                let r = self.apply_out_mods(r, inst.modifiers)?;
                self.b.bitcast(self.uint, None, r)?
            }
            "v_fract_f32" | "v_trunc_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_floor_f32"
            | "v_exp_f32" | "v_log_f32" | "v_sin_f32" | "v_cos_f32" | "v_rsq_f32"
            | "v_sqrt_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                let r = self.alu().emit_unary(inst.mnemonic.as_str(), f)?;
                let r = self.apply_out_mods(r, inst.modifiers)?;
                self.b.bitcast(self.uint, None, r)?
            }
            "v_cvt_u32_f32" | "v_cvt_i32_f32" => {
                let f = self.load_src_f32(inst.pc, s0, inst.modifiers, 0)?;
                if inst.mnemonic == "v_cvt_u32_f32" {
                    self.alu().emit_cvt_u32_f32(f)?
                } else {
                    self.alu().emit_cvt_i32_f32(f)?
                }
            }
            _ => {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: inst.mnemonic.clone(),
                });
            }
        };
        self.store_vgpr_active(vdst, res)
    }

    fn emit_vop2(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Ok(());
        };
        let res = self.emit_valu_core(inst, vdst)?;
        self.store_vgpr_active(vdst, res)
    }

    fn emit_vop3(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.opcode <= 0x0FF {
            return self.emit_vopc(inst);
        }
        if (0x180..=0x1FF).contains(&inst.opcode) {
            return self.emit_vop1(inst);
        }
        let DstOperand::Vgpr(vdst) = inst.dst else {
            return Ok(());
        };
        let res = self.emit_valu_core(inst, vdst)?;
        self.store_vgpr_active(vdst, res)
    }

    fn emit_valu_core(&mut self, inst: &DecodedInst, vdst: u8) -> Result<u32, SpirvError> {
        let mods = inst.modifiers;
        match inst.mnemonic.as_str() {
            "v_cndmask_b32" => {
                let s0 = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = if let Some(&mask) = inst.srcs.get(2) {
                    if self.uses_wave_masks() {
                        let (lo, hi) = self.load_scalar_b64(inst.pc, mask)?;
                        self.mask_lane(lo, hi)?
                    } else {
                        let mask = self.load_src_u32(inst.pc, mask)?;
                        self.b.i_not_equal(self.bool_ty, None, mask, self.c_u0)?
                    }
                } else {
                    self.b.load(self.bool_ty, None, self.var_vcc, None, [])?
                };
                let s0 = self
                    .alu()
                    .apply_f32_sign_mods(s0, mods.abs[0], mods.neg[0])?;
                let s1 = self
                    .alu()
                    .apply_f32_sign_mods(s1, mods.abs[1], mods.neg[1])?;
                Ok(self.b.select(self.uint, None, c, s1, s0)?)
            }
            "v_min_f32" | "v_max_f32" | "v_min3_f32" | "v_max3_f32" | "v_med3_f32" => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let result = if let Some(&src) = inst.srcs.get(2) {
                    let c = self.load_src_f32(inst.pc, src, mods, 2)?;
                    self.alu()
                        .emit_min_max_med3(inst.mnemonic.as_str(), a, b, c)?
                } else {
                    self.alu()
                        .emit_min_max_f32(a, b, inst.mnemonic == "v_max_f32")?
                };
                let result = self.apply_out_mods(result, mods)?;
                Ok(self.b.bitcast(self.uint, None, result)?)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32" => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let f = match inst.mnemonic.as_str() {
                    "v_add_f32" => self.b.f_add(self.f32_ty, None, a, b)?,
                    "v_sub_f32" => self.b.f_sub(self.f32_ty, None, a, b)?,
                    "v_subrev_f32" => self.b.f_sub(self.f32_ty, None, b, a)?,
                    "v_mul_f32" => self.b.f_mul(self.f32_ty, None, a, b)?,
                    "v_mul_legacy_f32" => {
                        let az = self.b.f_ord_equal(self.bool_ty, None, a, self.c_f0)?;
                        let bz = self.b.f_ord_equal(self.bool_ty, None, b, self.c_f0)?;
                        let any = self.b.logical_or(self.bool_ty, None, az, bz)?;
                        let p = self.b.f_mul(self.f32_ty, None, a, b)?;
                        self.b.select(self.f32_ty, None, any, self.c_f0, p)?
                    }
                    _ => unreachable!(),
                };
                let f = self.apply_out_mods(f, mods)?;
                Ok(self.b.bitcast(self.uint, None, f)?)
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                let x = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                let y = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let z = self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?;
                let result = self.alu().emit_cube_alu(inst.opcode, [x, y, z])?;
                let result = self.apply_out_mods(result, mods)?;
                Ok(self.b.bitcast(self.uint, None, result)?)
            }
            "v_fma_f32" | "v_fmamk_f32" | "v_fmaak_f32" | "v_fmac_f32" | "v_mad_f32"
            | "v_mac_f32" => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let c = if matches!(inst.mnemonic.as_str(), "v_fmac_f32" | "v_mac_f32") {
                    let old = self.load_vgpr(vdst)?;
                    self.b.bitcast(self.f32_ty, None, old)?
                } else {
                    self.load_src_f32(inst.pc, inst.srcs[2], mods, 2)?
                };
                let f = if matches!(inst.mnemonic.as_str(), "v_mad_f32" | "v_mac_f32") {
                    let product = self.b.f_mul(self.f32_ty, None, a, b)?;
                    self.b.decorate(product, Decoration::NoContraction, []);
                    let sum = self.b.f_add(self.f32_ty, None, product, c)?;
                    self.b.decorate(sum, Decoration::NoContraction, []);
                    sum
                } else {
                    self.b.ext_inst(
                        self.f32_ty,
                        None,
                        self.glsl,
                        GLOp::Fma as u32,
                        [Operand::IdRef(a), Operand::IdRef(b), Operand::IdRef(c)],
                    )?
                };
                let f = self.apply_out_mods(f, mods)?;
                Ok(self.b.bitcast(self.uint, None, f)?)
            }
            "v_add_nc_u32" | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_and_b32" | "v_or_b32"
            | "v_xor_b32" | "v_mul_lo_u32" | "v_mul_u32_u24" | "v_lshlrev_b32"
            | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                match inst.mnemonic.as_str() {
                    "v_add_nc_u32" => Ok(self.b.i_add(self.uint, None, a, b)?),
                    "v_sub_nc_u32" => Ok(self.b.i_sub(self.uint, None, a, b)?),
                    "v_subrev_nc_u32" => Ok(self.b.i_sub(self.uint, None, b, a)?),
                    "v_and_b32" => Ok(self.b.bitwise_and(self.uint, None, a, b)?),
                    "v_or_b32" => Ok(self.b.bitwise_or(self.uint, None, a, b)?),
                    "v_xor_b32" => Ok(self.b.bitwise_xor(self.uint, None, a, b)?),
                    "v_mul_lo_u32" => Ok(self.b.i_mul(self.uint, None, a, b)?),
                    "v_mul_u32_u24" => {
                        let c_24 = self.const_u32(0x00FF_FFFF);
                        let a24 = self.b.bitwise_and(self.uint, None, a, c_24)?;
                        let b24 = self.b.bitwise_and(self.uint, None, b, c_24)?;
                        Ok(self.b.i_mul(self.uint, None, a24, b24)?)
                    }
                    "v_lshlrev_b32" => {
                        let c31 = self.const_u32(31);
                        let sh = self.b.bitwise_and(self.uint, None, a, c31)?;
                        Ok(self.b.shift_left_logical(self.uint, None, b, sh)?)
                    }
                    "v_lshrrev_b32" => {
                        let c31 = self.const_u32(31);
                        let sh = self.b.bitwise_and(self.uint, None, a, c31)?;
                        Ok(self.b.shift_right_logical(self.uint, None, b, sh)?)
                    }
                    "v_ashrrev_i32" => {
                        let c31 = self.const_u32(31);
                        let sh = self.b.bitwise_and(self.uint, None, a, c31)?;
                        let sb = self.b.bitcast(self.sint, None, b)?;
                        let sr = self.b.shift_right_arithmetic(self.sint, None, sb, sh)?;
                        Ok(self.b.bitcast(self.uint, None, sr)?)
                    }
                    _ => unreachable!(),
                }
            }
            "v_cvt_pkrtz_f16_f32" => {
                let a_f = self.load_src_f32(inst.pc, inst.srcs[0], mods, 0)?;
                let b_f = self.load_src_f32(inst.pc, inst.srcs[1], mods, 1)?;
                let a_bits = self.b.bitcast(self.uint, None, a_f)?;
                let b_bits = self.b.bitcast(self.uint, None, b_f)?;
                let lo = self
                    .alu()
                    .emit_f32_to_f16(a_bits, HalfRounding::TowardZero)?;
                let hi = self
                    .alu()
                    .emit_f32_to_f16(b_bits, HalfRounding::TowardZero)?;
                let c_16 = self.const_u32(16);
                let hi_sh = self.b.shift_left_logical(self.uint, None, hi, c_16)?;
                Ok(self.b.bitwise_or(self.uint, None, lo, hi_sh)?)
            }
            "v_mad_u32_u24" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let c_24 = self.const_u32(0x00FF_FFFF);
                let a24 = self.b.bitwise_and(self.uint, None, a, c_24)?;
                let b24 = self.b.bitwise_and(self.uint, None, b, c_24)?;
                let prod = self.b.i_mul(self.uint, None, a24, b24)?;
                Ok(self.b.i_add(self.uint, None, prod, c)?)
            }
            "v_add_lshl_u32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let sum = self.b.i_add(self.uint, None, a, b)?;
                let mask = self.const_u32(31);
                let shift = self.b.bitwise_and(self.uint, None, c, mask)?;
                Ok(self.b.shift_left_logical(self.uint, None, sum, shift)?)
            }
            "v_lshl_add_u32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let c31 = self.const_u32(31);
                let sh = self.b.bitwise_and(self.uint, None, b, c31)?;
                let shifted = self.b.shift_left_logical(self.uint, None, a, sh)?;
                Ok(self.b.i_add(self.uint, None, shifted, c)?)
            }
            "v_add3_u32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let sum01 = self.b.i_add(self.uint, None, a, b)?;
                Ok(self.b.i_add(self.uint, None, sum01, c)?)
            }
            "v_lshl_or_b32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let c31 = self.const_u32(31);
                let sh = self.b.bitwise_and(self.uint, None, b, c31)?;
                let shifted = self.b.shift_left_logical(self.uint, None, a, sh)?;
                Ok(self.b.bitwise_or(self.uint, None, shifted, c)?)
            }
            "v_mul_hi_u32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                self.alu().mul_hi_u32(a, b)
            }
            "v_max_u32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let larger = self.b.u_greater_than(self.bool_ty, None, a, b)?;
                Ok(self.b.select(self.uint, None, larger, a, b)?)
            }
            "v_or3_b32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let first = self.b.bitwise_or(self.uint, None, a, b)?;
                Ok(self.b.bitwise_or(self.uint, None, first, c)?)
            }
            "v_and_or_b32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let masked = self.b.bitwise_and(self.uint, None, a, b)?;
                Ok(self.b.bitwise_or(self.uint, None, masked, c)?)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let offset = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let count = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let c31 = self.const_u32(31);
                let c32 = self.const_u32(32);
                let off_m = self.b.bitwise_and(self.uint, None, offset, c31)?;
                let cnt_raw = self.b.bitwise_and(self.uint, None, count, c31)?;
                let max_cnt = self.b.i_sub(self.uint, None, c32, off_m)?;
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
                        .bit_field_u_extract(self.uint, None, a, off_m, cnt_safe)?)
                } else {
                    let s_base = self.b.bitcast(self.sint, None, a)?;
                    let extracted = self
                        .b
                        .bit_field_s_extract(self.sint, None, s_base, off_m, cnt_safe)?;
                    Ok(self.b.bitcast(self.uint, None, extracted)?)
                }
            }
            "v_bfi_b32" => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let c = self.load_src_u32(inst.pc, inst.srcs[2])?;
                let lhs = self.b.bitwise_and(self.uint, None, a, b)?;
                let na = self.b.not(self.uint, None, a)?;
                let rhs = self.b.bitwise_and(self.uint, None, na, c)?;
                Ok(self.b.bitwise_or(self.uint, None, lhs, rhs)?)
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                let s0 = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let s1 = self.load_src_u32(inst.pc, inst.srcs[1])?;
                let below_mask = if self.stage == ShaderStage::Mesh {
                    let low = self
                        .b
                        .i_sub(self.uint, None, self.mesh_lane_bit, self.c_u1)?;
                    let c32 = self.const_u32(32);
                    let high_lane = self.b.u_greater_than_equal(
                        self.bool_ty,
                        None,
                        self.mesh_guest_lane,
                        c32,
                    )?;
                    if inst.mnemonic == "v_mbcnt_hi_u32_b32" {
                        self.b.select(self.uint, None, high_lane, low, self.c_u0)?
                    } else {
                        let all = self.const_u32(u32::MAX);
                        self.b.select(self.uint, None, high_lane, all, low)?
                    }
                } else {
                    self.c_u0
                };
                let masked = self.b.bitwise_and(self.uint, None, s0, below_mask)?;
                let cnt = self.b.bit_count(self.uint, None, masked)?;
                Ok(self.b.i_add(self.uint, None, cnt, s1)?)
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }

    fn emit_vopc(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let op = inst.opcode;
        let pred = match op {
            0x00..=0x1F => {
                let a = self.load_src_f32(inst.pc, inst.srcs[0], inst.modifiers, 0)?;
                let b = self.load_src_f32(inst.pc, inst.srcs[1], inst.modifiers, 1)?;
                self.alu().eval_f32_cmp_pred(op & 0xF, a, b)?
            }
            0x80..=0x87 | 0x90..=0x97 | 0xC0..=0xC7 | 0xD0..=0xD7 => {
                let a = self.load_src_u32(inst.pc, inst.srcs[0])?;
                let b = self.load_src_u32(inst.pc, inst.srcs[1])?;
                self.alu().eval_int_cmp_pred(op & 7, a, b, op < 0xC0)?
            }
            _ => {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: inst.mnemonic.clone(),
                });
            }
        };
        let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
        let res = self.b.logical_and(self.bool_ty, None, active, pred)?;
        if matches!(inst.dst, DstOperand::VccLo) {
            self.b.store(self.var_vcc, res, None, [])?;
        } else if matches!(inst.dst, DstOperand::ExecLo) {
            self.b.store(self.var_active, res, None, [])?;
        } else if self.uses_wave_masks() {
            let (lo, hi) = self.predicate_mask_words(res)?;
            self.store_scalar_b64(inst.dst, lo, hi)?;
        } else {
            let u = self.b.select(self.uint, None, res, self.c_u1, self.c_u0)?;
            self.store_scalar_dst(inst.dst, u)?;
        }
        Ok(())
    }

    fn emit_vintrp(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let InstExtra::Vintrp {
            op,
            vdst,
            attr,
            chan,
            ..
        } = inst.extra
        else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };
        // In Vulkan Fragment shaders, hardware already interpolated `Input Location attr`.
        // Lower canonical p1+p2 (and mov) cleanly to the interpolated component value.
        if op == 0 {
            // p1 is followed by p2 into the same vdst; load immediately so even standalone p1 is valid
            if let Some(&in_var) = self.in_locs.get(&attr) {
                let vec = self.b.load(self.vec4, None, in_var, None, [])?;
                let comp = self
                    .b
                    .composite_extract(self.f32_ty, None, vec, [u32::from(chan)])?;
                let bits = self.b.bitcast(self.uint, None, comp)?;
                self.store_vgpr_active(vdst, bits)?;
            }
            return Ok(());
        }
        if let Some(&in_var) = self.in_locs.get(&attr) {
            let vec = self.b.load(self.vec4, None, in_var, None, [])?;
            let comp = self
                .b
                .composite_extract(self.f32_ty, None, vec, [u32::from(chan)])?;
            let bits = self.b.bitcast(self.uint, None, comp)?;
            self.store_vgpr_active(vdst, bits)?;
        }
        Ok(())
    }

    fn emit_exp(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let InstExtra::Exp {
            target,
            en,
            compr,
            vm,
            vsrc,
            ..
        } = inst.extra
        else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };

        let read_vec4 = |ctx: &mut Self| -> Result<u32, SpirvError> {
            if compr {
                let (r, g) = if (en & 0x3) != 0 {
                    let bits0 = ctx.load_vgpr(vsrc[0])?;
                    let rg = ctx.b.ext_inst(
                        ctx.vec2,
                        None,
                        ctx.glsl,
                        GLOp::UnpackHalf2x16 as u32,
                        [Operand::IdRef(bits0)],
                    )?;
                    (
                        ctx.b.composite_extract(ctx.f32_ty, None, rg, [0])?,
                        ctx.b.composite_extract(ctx.f32_ty, None, rg, [1])?,
                    )
                } else {
                    (ctx.c_f0, ctx.c_f0)
                };
                let (b_, a) = if (en & 0xC) != 0 {
                    let bits1 = ctx.load_vgpr(vsrc[1])?;
                    let ba = ctx.b.ext_inst(
                        ctx.vec2,
                        None,
                        ctx.glsl,
                        GLOp::UnpackHalf2x16 as u32,
                        [Operand::IdRef(bits1)],
                    )?;
                    (
                        ctx.b.composite_extract(ctx.f32_ty, None, ba, [0])?,
                        ctx.b.composite_extract(ctx.f32_ty, None, ba, [1])?,
                    )
                } else {
                    (ctx.c_f0, ctx.c_f1)
                };
                Ok(ctx.b.composite_construct(ctx.vec4, None, [r, g, b_, a])?)
            } else {
                let mut comps = [ctx.c_f0, ctx.c_f0, ctx.c_f0, ctx.c_f1];
                for c in 0..4 {
                    if (en & (1 << c)) != 0 {
                        let bits = ctx.load_vgpr(vsrc[c])?;
                        comps[c] = ctx.b.bitcast(ctx.f32_ty, None, bits)?;
                    }
                }
                Ok(ctx.b.composite_construct(ctx.vec4, None, comps)?)
            }
        };

        if self.stage == ShaderStage::Mesh {
            if compr || vm {
                return Err(SpirvError::UnsupportedInstruction {
                    pc: inst.pc,
                    mnemonic: format!("unsupported mesh EXP flags on {}", inst.mnemonic),
                });
            }
            let payload = match target {
                12 | 32..=63 => read_vec4(self)?,
                20 if en == 1 => self.load_vgpr(vsrc[0])?,
                _ => {
                    return Err(SpirvError::UnsupportedInstruction {
                        pc: inst.pc,
                        mnemonic: format!(
                            "unsupported mesh export {} with en=0x{en:x}",
                            inst.mnemonic
                        ),
                    });
                }
            };
            let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
            if let Some(&(var, valid)) = self.mesh_exports.get(&target) {
                let store = self.b.id();
                let merge = self.b.id();
                self.b.selection_merge(merge, SelectionControl::NONE)?;
                self.b.branch_conditional(active, store, merge, [])?;
                self.b.begin_block(Some(store))?;
                self.b.store(var, payload, None, [])?;
                self.b.store(valid, self.c_true, None, [])?;
                self.b.branch(merge)?;
                self.b.begin_block(Some(merge))?;
                return Ok(());
            }
            return self.write_mesh_export(target, payload, active);
        }

        match target {
            12 => {
                // exp pos0 -> BuiltIn Position
                if let Some(pos_var) = self.out_pos0 {
                    let mut v = read_vec4(self)?;
                    if self.opts.flip_y {
                        let x = self.b.composite_extract(self.f32_ty, None, v, [0])?;
                        let y = self.b.composite_extract(self.f32_ty, None, v, [1])?;
                        let z = self.b.composite_extract(self.f32_ty, None, v, [2])?;
                        let w = self.b.composite_extract(self.f32_ty, None, v, [3])?;
                        let ny = self.b.f_negate(self.f32_ty, None, y)?;
                        v = self.b.composite_construct(self.vec4, None, [x, ny, z, w])?;
                    }
                    self.b.store(pos_var, v, None, [])?;
                }
            }
            32..=63 => {
                let idx = target - 32;
                if let Some(&param_var) = self.out_params.get(&idx) {
                    let v = read_vec4(self)?;
                    self.b.store(param_var, v, None, [])?;
                }
            }
            0..=7 => {
                if let Some(&mrt_var) = self.out_mrts.get(&target) {
                    let v = read_vec4(self)?;
                    self.b.store(mrt_var, v, None, [])?;
                }
            }
            8 => {
                // exp mrtz -> BuiltIn FragDepth
                if let Some(z_var) = self.out_mrtz {
                    let bits = self.load_vgpr(vsrc[0])?;
                    let z = self.b.bitcast(self.f32_ty, None, bits)?;
                    self.b.store(z_var, z, None, [])?;
                }
            }
            9 => {
                // exp null -> pixel discard via OpDemoteToHelperInvocation
                if self.stage == ShaderStage::Pixel {
                    self.b.store(self.var_active, self.c_false, None, [])?;
                    self.emit_demote_raw()?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn mesh_export_targets(&self) -> Vec<u8> {
        self.out_mesh_pos
            .map(|_| 12)
            .into_iter()
            .chain(self.out_mesh_prim_indices.map(|_| 20))
            .chain(self.out_mesh_params.keys().map(|loc| loc + 32))
            .collect()
    }

    fn write_mesh_export(
        &mut self,
        target: u8,
        payload: u32,
        active: u32,
    ) -> Result<(), SpirvError> {
        let count_var = if target == 20 {
            self.var_mesh_prim_cnt
        } else {
            self.var_mesh_vtx_cnt
        };
        let count = self.b.load(self.uint, None, count_var, None, [])?;
        let in_range = self
            .b
            .u_less_than(self.bool_ty, None, self.mesh_local_idx, count)?;
        let enabled = self.b.logical_and(self.bool_ty, None, active, in_range)?;
        let store = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, SelectionControl::NONE)?;
        self.b.branch_conditional(enabled, store, merge, [])?;
        self.b.begin_block(Some(store))?;
        match target {
            12 => {
                if let Some(var) = self.out_mesh_pos {
                    let value = if self.opts.flip_y {
                        let y = self.b.composite_extract(self.f32_ty, None, payload, [1])?;
                        let neg_y = self.b.f_negate(self.f32_ty, None, y)?;
                        self.b
                            .composite_insert(self.vec4, None, neg_y, payload, [1])?
                    } else {
                        payload
                    };
                    let ptr = self.b.access_chain(
                        self.ptr_out_vec4,
                        None,
                        var,
                        [self.mesh_local_idx, self.c_u0],
                    )?;
                    self.b.store(ptr, value, None, [])?;
                }
            }
            20 => {
                if let (Some(indices), Some(cull)) =
                    (self.out_mesh_prim_indices, self.out_mesh_prim_cull)
                {
                    let null_bit =
                        self.b
                            .bitwise_and(self.uint, None, payload, self.c_sign_mask)?;
                    let is_null = self
                        .b
                        .i_not_equal(self.bool_ty, None, null_bit, self.c_u0)?;
                    let c9 = self.const_u32(9);
                    let mut vertices = [self.c_u0; 3];
                    for (index, vertex) in vertices.iter_mut().enumerate() {
                        let offset = self.const_u32(index as u32 * 10);
                        let value = self
                            .b
                            .bit_field_u_extract(self.uint, None, payload, offset, c9)?;
                        *vertex = self.b.select(self.uint, None, is_null, self.c_u0, value)?;
                    }
                    let triangle = self.b.composite_construct(self.uvec3, None, vertices)?;
                    let ptr = self.b.access_chain(
                        self.ptr_out_uvec3,
                        None,
                        indices,
                        [self.mesh_local_idx],
                    )?;
                    self.b.store(ptr, triangle, None, [])?;
                    let ptr = self.b.access_chain(
                        self.ptr_out_bool,
                        None,
                        cull,
                        [self.mesh_local_idx],
                    )?;
                    // Keep bool output writes literal. Some software Vulkan
                    // drivers miscompile runtime boolean mesh output stores.
                    let culled = self.b.id();
                    let visible = self.b.id();
                    let cull_merge = self.b.id();
                    self.b
                        .selection_merge(cull_merge, SelectionControl::DONT_FLATTEN)?;
                    self.b.branch_conditional(is_null, culled, visible, [])?;
                    self.b.begin_block(Some(culled))?;
                    self.b.store(ptr, self.c_true, None, [])?;
                    self.b.branch(cull_merge)?;
                    self.b.begin_block(Some(visible))?;
                    self.b.store(ptr, self.c_false, None, [])?;
                    self.b.branch(cull_merge)?;
                    self.b.begin_block(Some(cull_merge))?;
                }
            }
            _ => {
                if let Some(&var) = self.out_mesh_params.get(&(target - 32)) {
                    let ptr =
                        self.b
                            .access_chain(self.ptr_out_vec4, None, var, [self.mesh_local_idx])?;
                    self.b.store(ptr, payload, None, [])?;
                }
            }
        }
        self.b.branch(merge)?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }

    fn emit_quad_swizzle(&mut self, vdst: u8, vsrc: u8, offset: u16) -> Result<(), SpirvError> {
        let value = self.load_vgpr(vsrc)?;
        let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
        let value = self.b.select(self.uint, None, active, value, self.c_u0)?;
        let lane = self
            .b
            .bitwise_and(self.uint, None, self.subgroup_lane, self.c_u3)?;
        let mut result = self.c_u0;
        for src in 0..4 {
            let dst_mask = (0..4)
                .filter(|&j| ((offset >> (2 * j)) & 3) == src)
                .fold(0, |mask, j| mask | (1 << j));
            if dst_mask == 0 {
                continue;
            }
            let index = self.const_u32(u32::from(src));
            let fetched = self.b.group_non_uniform_quad_broadcast(
                self.uint,
                None,
                self.subgroup_scope,
                value,
                index,
            )?;
            if dst_mask == 15 {
                result = fetched;
            } else {
                let mask = self.const_u32(dst_mask);
                let bit = self
                    .b
                    .bit_field_u_extract(self.uint, None, mask, lane, self.c_u1)?;
                let selected = self.b.i_not_equal(self.bool_ty, None, bit, self.c_u0)?;
                result = self.b.select(self.uint, None, selected, fetched, result)?;
            }
        }
        self.store_vgpr_active(vdst, result)
    }

    fn emit_ds(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if self.stage == ShaderStage::Pixel
            && let InstExtra::DsSwizzle {
                offset: offset @ 0x8000..=0x80ff,
                vsrc,
            } = inst.extra
            && let DstOperand::Vgpr(vdst) = inst.dst
        {
            return self.emit_quad_swizzle(vdst, vsrc, offset);
        }
        let InstExtra::DsLds {
            addr,
            data0,
            data1,
            vdst,
            offset0,
            offset1,
            gds,
            is_write,
            paired,
            ..
        } = inst.extra
        else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };
        if gds {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        }
        let Some(lds_var) = self.lds_buf else {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        };
        let base_addr = self.load_vgpr(addr)?;
        let first = if paired {
            u32::from(offset0) * 4
        } else {
            (u32::from(offset1) << 8) | u32::from(offset0)
        };
        let write = is_write.then(|| MemoryWrite::for_inst(inst));
        let result = self.emit_lds_word(lds_var, base_addr, first, write, data0)?;
        if paired && (!is_write || offset0 != offset1) {
            let next = if offset0 == offset1 {
                result
            } else {
                self.emit_lds_word(lds_var, base_addr, u32::from(offset1) * 4, write, data1)?
            };
            if !is_write {
                self.store_vgpr_active(vdst.wrapping_add(1), next)?;
            }
        }
        if let Some(dst) = write.map_or(Some(vdst), MemoryWrite::result) {
            self.store_vgpr_active(dst, result)?;
        }
        Ok(())
    }

    fn emit_lds_word(
        &mut self,
        lds_var: u32,
        base_addr: u32,
        byte_offset: u32,
        write: Option<MemoryWrite>,
        data: u8,
    ) -> Result<u32, SpirvError> {
        let c_off = self.const_u32(byte_offset);
        let c_len = self.const_u32(self.lds_dwords);
        let byte_addr = self.b.i_add(self.uint, None, base_addr, c_off)?;
        let word_idx = self
            .b
            .shift_right_logical(self.uint, None, byte_addr, self.c_u2)?;
        let in_bounds = self.b.u_less_than(self.bool_ty, None, word_idx, c_len)?;
        let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
        let guard = self.b.logical_and(self.bool_ty, None, active, in_bounds)?;

        let no_overflow = self
            .b
            .u_greater_than_equal(self.bool_ty, None, byte_addr, base_addr)?;
        let guard = self.b.logical_and(self.bool_ty, None, guard, no_overflow)?;
        let rem = self.b.bitwise_and(self.uint, None, byte_addr, self.c_u3)?;
        let aligned = self.b.i_equal(self.bool_ty, None, rem, self.c_u0)?;
        let guard = self.b.logical_and(self.bool_ty, None, guard, aligned)?;
        let reads_value = write.is_none_or(|op| op.result().is_some());
        if reads_value {
            self.b.store(self.var_mem_tmp, self.c_u0, None, [])?;
        }
        let value = if write.is_some() {
            Some(self.load_vgpr(data)?)
        } else {
            None
        };
        let access = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, SelectionControl::NONE)?;
        self.b.branch_conditional(guard, access, merge, [])?;
        self.b.begin_block(Some(access))?;
        let p = self
            .b
            .access_chain(self.ptr_wg_uint, None, lds_var, [word_idx])?;
        if let Some((op, value)) = write.zip(value) {
            op.emit(
                &mut self.b,
                self.uint,
                p,
                [value, self.c_u0],
                Scope::Workgroup,
                self.var_mem_tmp,
            )?;
        } else {
            let loaded = self.b.load(self.uint, None, p, None, [])?;
            self.b.store(self.var_mem_tmp, loaded, None, [])?;
        }
        self.b.branch(merge)?;
        self.b.begin_block(Some(merge))?;
        if reads_value {
            Ok(self.b.load(self.uint, None, self.var_mem_tmp, None, [])?)
        } else {
            Ok(self.c_u0)
        }
    }

    fn emit_mimg(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        let fragment = self.stage == ShaderStage::Pixel;
        let i = Mimg::parse(inst, fragment)?;
        let binding = self.image_bindings.binding(i);
        let addr = i
            .address_registers()
            .map(|reg| self.load_vgpr(reg))
            .collect::<Result<Vec<_>, _>>()?;
        if i.is_store() {
            let data = (0..i.dmask.count_ones() as u8)
                .map(|offset| self.load_vgpr(i.vdata.wrapping_add(offset)))
                .collect::<Result<Vec<_>, _>>()?;
            let active = self.b.load(self.bool_ty, None, self.var_active, None, [])?;
            self.alu()
                .emit_image_store(i, binding, &addr, &data, active)?;
        } else {
            let sampler = self.image_bindings.sampler(i);
            let data = self
                .alu()
                .emit_image_read(i, binding, sampler, &addr, fragment)?;
            for (offset, bits) in data.into_iter().enumerate() {
                self.store_vgpr_active(i.vdata.wrapping_add(offset as u8), bits)?;
            }
        }
        Ok(())
    }

    fn emit_mem_load(&mut self, inst: &DecodedInst) -> Result<(), SpirvError> {
        if inst.atomic_op().is_some()
            || (matches!(inst.extra, InstExtra::Mubuf { .. }) && inst.opcode <= 7)
            || matches!(
                inst.mnemonic.as_str(),
                "buffer_load_ushort" | "buffer_store_short"
            )
        {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            });
        }
        let load_word = |ctx: &mut Self, byte_addr: u32| -> Result<u32, SpirvError> {
            let mem_buf = ctx.memory_buf.expect("memory_buf");
            let word_idx = ctx
                .b
                .shift_right_logical(ctx.uint, None, byte_addr, ctx.c_u2)?;
            let in_bounds = ctx
                .b
                .u_less_than(ctx.bool_ty, None, word_idx, ctx.mem_words_len)?;
            ctx.b.store(ctx.var_mem_tmp, ctx.c_u0, None, [])?;
            let l_blk = ctx.b.id();
            let m_blk = ctx.b.id();
            ctx.b.selection_merge(m_blk, SelectionControl::NONE)?;
            ctx.b.branch_conditional(in_bounds, l_blk, m_blk, [])?;
            ctx.b.begin_block(Some(l_blk))?;
            let p = ctx
                .b
                .access_chain(ctx.ptr_sb_uint, None, mem_buf, [ctx.c_u0, word_idx])?;
            let v = ctx.b.load(ctx.uint, None, p, None, [])?;
            ctx.b.store(ctx.var_mem_tmp, v, None, [])?;
            ctx.b.branch(m_blk)?;
            ctx.b.begin_block(Some(m_blk))?;
            Ok(ctx.b.load(ctx.uint, None, ctx.var_mem_tmp, None, [])?)
        };

        let store_word = |ctx: &mut Self, byte_addr: u32, val: u32| -> Result<(), SpirvError> {
            let uav_buf = ctx.uav_buf.expect("uav_buf");
            let word_idx = ctx
                .b
                .shift_right_logical(ctx.uint, None, byte_addr, ctx.c_u2)?;
            let in_bounds = ctx
                .b
                .u_less_than(ctx.bool_ty, None, word_idx, ctx.uav_words_len)?;
            let active = ctx.b.load(ctx.bool_ty, None, ctx.var_active, None, [])?;
            let do_store = ctx.b.logical_and(ctx.bool_ty, None, active, in_bounds)?;
            let s_blk = ctx.b.id();
            let m_blk = ctx.b.id();
            ctx.b.selection_merge(m_blk, SelectionControl::NONE)?;
            ctx.b.branch_conditional(do_store, s_blk, m_blk, [])?;
            ctx.b.begin_block(Some(s_blk))?;
            let p = ctx
                .b
                .access_chain(ctx.ptr_sb_uint, None, uav_buf, [ctx.c_u0, word_idx])?;
            ctx.b.store(p, val, None, [])?;
            ctx.b.branch(m_blk)?;
            ctx.b.begin_block(Some(m_blk))?;
            Ok(())
        };

        match inst.extra {
            InstExtra::Smem {
                sbase,
                sdata,
                ioffset,
                dwords,
                ..
            } => {
                let base = self.load_sgpr(sbase)?;
                let off = self.const_u32(ioffset as u32);
                let addr0 = self.b.i_add(self.uint, None, base, off)?;
                let mut vals = Vec::new();
                for j in 0..dwords {
                    let cj = self.const_u32(u32::from(j) * 4);
                    let a = self.b.i_add(self.uint, None, addr0, cj)?;
                    vals.push(load_word(self, a)?);
                }
                for (j, val) in vals.into_iter().enumerate() {
                    self.store_sgpr(sdata.wrapping_add(j as u8), val)?;
                }
                Ok(())
            }
            InstExtra::Mubuf {
                vaddr,
                vdata,
                srsrc,
                offset,
                offen,
                dwords,
                is_store,
                ..
            } => {
                let base = self.load_sgpr(srsrc)?;
                let voff = if offen {
                    self.load_vgpr(vaddr)?
                } else {
                    self.c_u0
                };
                let c_off = self.const_u32(u32::from(offset));
                let a0 = self.b.i_add(self.uint, None, base, voff)?;
                let a1 = self.b.i_add(self.uint, None, a0, c_off)?;
                if is_store {
                    for j in 0..dwords {
                        let cj = self.const_u32(u32::from(j) * 4);
                        let a = self.b.i_add(self.uint, None, a1, cj)?;
                        let v = self.load_vgpr(vdata.wrapping_add(j))?;
                        store_word(self, a, v)?;
                    }
                } else {
                    let mut vals = Vec::new();
                    for j in 0..dwords {
                        let cj = self.const_u32(u32::from(j) * 4);
                        let a = self.b.i_add(self.uint, None, a1, cj)?;
                        vals.push(load_word(self, a)?);
                    }
                    for (j, val) in vals.into_iter().enumerate() {
                        self.store_vgpr_active(vdata.wrapping_add(j as u8), val)?;
                    }
                }
                Ok(())
            }
            InstExtra::Flat {
                addr,
                data,
                vdst,
                saddr,
                offset,
                dwords,
                is_store,
                ..
            } => {
                let base = if saddr == 0x7D {
                    self.load_vgpr(addr)?
                } else {
                    let sb = self.load_sgpr(saddr)?;
                    let vo = self.load_vgpr(addr)?;
                    self.b.i_add(self.uint, None, sb, vo)?
                };
                let c_off = self.const_u32(i32::from(offset) as u32);
                let a0 = self.b.i_add(self.uint, None, base, c_off)?;
                if is_store {
                    for j in 0..dwords {
                        let cj = self.const_u32(u32::from(j) * 4);
                        let a = self.b.i_add(self.uint, None, a0, cj)?;
                        let v = self.load_vgpr(data.wrapping_add(j))?;
                        store_word(self, a, v)?;
                    }
                } else {
                    let mut vals = Vec::new();
                    for j in 0..dwords {
                        let cj = self.const_u32(u32::from(j) * 4);
                        let a = self.b.i_add(self.uint, None, a0, cj)?;
                        vals.push(load_word(self, a)?);
                    }
                    for (j, val) in vals.into_iter().enumerate() {
                        self.store_vgpr_active(vdst.wrapping_add(j as u8), val)?;
                    }
                }
                Ok(())
            }
            _ => Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: inst.mnemonic.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{SignatureElement, build_signature_chunk_body, parse_dxbc_signature};
    use crate::decode::disassemble;
    use crate::encode as asm;
    #[cfg(not(feature = "spirv-val"))]
    use crate::spirv::is_spirv_val_available;
    use crate::spirv::validate_spirv_vulkan13;

    fn assert_valid(words: &[u32]) {
        assert_eq!(words[0], 0x0723_0203);
        assert_eq!(words[1], 0x0001_0600);
        #[cfg(feature = "spirv-val")]
        validate_spirv_vulkan13(words).expect("graphics SPIR-V must pass spirv-val");
        #[cfg(not(feature = "spirv-val"))]
        if is_spirv_val_available() {
            validate_spirv_vulkan13(words).expect("graphics SPIR-V must pass spirv-val");
        }
    }

    #[test]
    fn preserves_four_distinct_fragment_export_locations() {
        let mut a = asm::Asm::new();
        for target in 0..4 {
            a.push2(asm::exp_mrt(target, [0, 0, 0, 0], false, target == 3, true));
        }
        a.push(asm::s_endpgm());
        let insts = disassemble(a.words()).unwrap();
        let sig = StageSignature::default();
        let live = analyze_live_registers(ShaderStage::Pixel, &insts, &sig);
        assert_eq!(live.mrt_locations, BTreeSet::from([0, 1, 2, 3]));

        let words = compile_graphics_spv(
            ShaderStage::Pixel,
            &insts,
            &sig,
            &GraphicsCompileOptions::default(),
        )
        .unwrap();
        assert_valid(&words);
        let module = rspirv::dr::load_words(words).unwrap();
        let locations: BTreeSet<_> = module
            .annotations
            .iter()
            .filter_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(_),
                    Operand::Decoration(Decoration::Location),
                    Operand::LiteralBit32(loc),
                ] => Some(*loc),
                _ => None,
            })
            .collect();
        assert_eq!(locations, BTreeSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn mesh_psv_layout_and_readfirstlane_registers() {
        let mut blob = vec![0u8; 44 + 52];
        blob[..4].copy_from_slice(b"DXBC");
        let len = (blob.len() as u32).to_le_bytes();
        blob[24..28].copy_from_slice(&len);
        blob[28..32].copy_from_slice(&1u32.to_le_bytes());
        blob[32..36].copy_from_slice(&36u32.to_le_bytes());
        blob[36..40].copy_from_slice(b"PSV0");
        blob[40..44].copy_from_slice(&52u32.to_le_bytes());
        let runtime = &mut blob[44..];
        for (offset, value) in [
            (0, 48u32),
            (4, 2048),
            (16, (5 << 16) | 8),
            (40, 8),
            (44, 2),
            (48, 1),
        ] {
            runtime[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        runtime[31] = 2;
        assert_eq!(
            MeshLayout::from_dxbc(&blob, false).unwrap(),
            MeshLayout {
                local_size: [8, 2, 1],
                wave_size: 64,
                max_vertices: 8,
                max_primitives: 5,
                lds_bytes: 2048,
            }
        );
        blob[75] = 1;
        assert!(MeshLayout::from_dxbc(&blob, false).is_err());
        blob[28..32].fill(0);
        assert_eq!(
            MeshLayout::from_dxbc(&blob, true).unwrap(),
            MeshLayout::default()
        );

        let insts = disassemble(&[asm::v_readfirstlane_b32(12, 45), asm::s_endpgm()]).unwrap();
        let sig = StageSignature::default();
        let live = analyze_live_registers(ShaderStage::Pixel, &insts, &sig);
        assert_eq!(live.live_sgprs, BTreeSet::from([12]));
        assert_eq!(live.live_vgprs, BTreeSet::from([45]));
        let words =
            compile_graphics_spv(ShaderStage::Pixel, &insts, &sig, &Default::default()).unwrap();
        assert_valid(&words);
    }

    #[test]
    fn parses_isg1_and_isgn_signatures_roundtrip() {
        let elems = vec![
            SignatureElement {
                stream: 0,
                semantic_name: "POSITION".into(),
                semantic_index: 0,
                system_value_type: 0,
                component_type: 3,
                register: 0,
                mask: 0xF,
                rw_mask: 0xF,
                min_precision: 0,
            },
            SignatureElement {
                stream: 0,
                semantic_name: "COLOR".into(),
                semantic_index: 0,
                system_value_type: 0,
                component_type: 3,
                register: 1,
                mask: 0xF,
                rw_mask: 0xF,
                min_precision: 0,
            },
        ];
        for ext in [true, false] {
            let chunk = build_signature_chunk_body(&elems, ext);
            let parsed = parse_dxbc_signature(&chunk, ext).expect("parse signature");
            assert_eq!(parsed, elems);
        }
    }

    #[test]
    fn prunes_dead_registers_and_validates_vs_and_ps_modules() {
        let sig = StageSignature {
            inputs: vec![
                SignatureElement {
                    stream: 0,
                    semantic_name: "POSITION".into(),
                    semantic_index: 0,
                    system_value_type: 0,
                    component_type: 3,
                    register: 0,
                    mask: 0xF,
                    rw_mask: 0xF,
                    min_precision: 0,
                },
                SignatureElement {
                    stream: 0,
                    semantic_name: "COLOR".into(),
                    semantic_index: 0,
                    system_value_type: 0,
                    component_type: 3,
                    register: 1,
                    mask: 0xF,
                    rw_mask: 0xF,
                    min_precision: 0,
                },
            ],
            outputs: Vec::new(),
        };

        let mut vs = asm::Asm::new();
        vs.push(asm::s_swappc_b64(4, 20)); // fetch-shader call cleanly bypassed
        vs.push2(asm::exp_pos0([0, 1, 2, 3], false));
        vs.push2(asm::exp_param(0, [4, 5, 6, 7], true));
        vs.push(asm::s_endpgm());
        let vs_insts = disassemble(vs.words()).expect("disasm vs");

        // Verify live-register pruning allocates ONLY v0..v7 and 0 SGPRs (not 256 VGPRs + 128 SGPRs!)
        let live = analyze_live_registers(ShaderStage::Vertex, &vs_insts, &sig);
        assert_eq!(live.live_vgprs.len(), 8);
        assert!(live.live_sgprs.is_empty());

        let vs_spv = compile_graphics_spv(
            ShaderStage::Vertex,
            &vs_insts,
            &sig,
            &GraphicsCompileOptions::default(),
        )
        .expect("compile VS");
        assert_valid(&vs_spv);
        assert!(
            vs_spv.len() < 600,
            "pruned VS SPIR-V should be compact (got {} words)",
            vs_spv.len()
        );

        // Fragment shader with VINTRP, image_sample, discard (OpDemoteToHelperInvocation), mrtz, and mrt0
        let mut ps = asm::Asm::new();
        ps.push(asm::v_interp_p1_f32(0, 0, 0, 0));
        ps.push(asm::v_interp_p2_f32(0, 1, 0, 0));
        ps.push(asm::v_interp_p1_f32(1, 0, 0, 1));
        ps.push(asm::v_interp_p2_f32(1, 1, 0, 1));
        ps.push2(asm::image_sample_2d(4, 0, 0, 4, 0xF));
        ps.push(asm::v_cmp_gt_f32(asm::v(0), 1));
        ps.push(asm::s_and_saveexec_b32(0, asm::VCC_LO));
        ps.push(asm::v_mov_b32(8, asm::F_0_5));
        ps.push2(asm::exp_mrtz(8, false, true));
        ps.push2(asm::exp_mrt(0, [4, 5, 6, 7], false, true, true));
        ps.push(asm::s_endpgm());
        let ps_insts = disassemble(ps.words()).expect("disasm ps");

        let ps_spv = compile_graphics_spv(
            ShaderStage::Pixel,
            &ps_insts,
            &StageSignature::default(),
            &GraphicsCompileOptions::default(),
        )
        .expect("compile PS");
        assert_valid(&ps_spv);
    }

    #[test]
    fn validates_d3d12_root_bindings_graphics_and_compute_spv() {
        // Compute shader: reads root constant s0, reads CBV word[0] via s_load_dword,
        // adds local invocation id v0, writes to UAV buffer via buffer_store_dword.
        let mut cs = asm::Asm::new();
        cs.push2(asm::s_load_dword(8, 4, 0x7D, 0));
        cs.push(asm::v_add_nc_u32(1, asm::s(0), 0));
        cs.push(asm::v_add_nc_u32(1, asm::s(8), 1));
        cs.push2(asm::v_mul_lo_u32(2, asm::v(0), asm::inline_i(4)));
        cs.push2(asm::buffer_store_dword(
            1,
            2,
            12,
            asm::inline_i(0),
            0,
            true,
            false,
        ));
        cs.push(asm::s_endpgm());
        let cs_insts = disassemble(cs.words()).expect("disasm cs");
        let cs_spv = compile_d3d12_compute_spv(&cs_insts, [64, 1, 1], true, false)
            .expect("compile D3D12 compute SPV");
        assert_valid(&cs_spv);

        // Fragment shader reading root constant s0..s3 color tint + CBV override
        let mut ps = asm::Asm::new();
        ps.push(asm::v_mov_b32(0, asm::s(0)));
        ps.push(asm::v_mov_b32(1, asm::s(1)));
        ps.push(asm::v_mov_b32(2, asm::s(2)));
        ps.push(asm::v_mov_b32(3, asm::s(3)));
        ps.push2(asm::exp_mrt(0, [0, 1, 2, 3], false, true, false));
        ps.push(asm::s_endpgm());
        let ps_insts = disassemble(ps.words()).expect("disasm ps");
        let ps_spv = compile_graphics_spv(
            ShaderStage::Pixel,
            &ps_insts,
            &StageSignature::default(),
            &GraphicsCompileOptions {
                d3d12_root_bindings: true,
                strict_float_controls: true,
                ..Default::default()
            },
        )
        .expect("compile root-bound PS");
        assert_valid(&ps_spv);
    }

    #[test]
    fn validates_rdna2_mesh_shader_spirv_and_rejects_missing_alloc() {
        // Exact LLVM 22.1.8 gfx1030 assembled RDNA2 triangle mesh shader:
        // m0 = 0x1003 (3 verts, 1 prim), MSG_GS_ALLOC_REQ, exp prim with edge flags 0x20280600,
        // v_mbcnt_lo_u32_b32 lane ID, exp param0 (yellow) + exp pos0.
        let ms_words: &[u32] = &[
            0xbefc03ff, 0x00001003, 0xbf900009, 0x7e1202ff, 0x20280600, 0xbefe0381, 0xf8000941,
            0x00000009, 0xbefe0387, 0xd7650000, 0x000100c1, 0x7e020280, 0x7e0402ff, 0xbf400000,
            0x7e0602ff, 0x3f400000, 0x7d840081, 0x02020701, 0x7d840082, 0x02020501, 0x7d840080,
            0x02040503, 0x7e0802f0, 0x7e0a02f2, 0x7e0c0280, 0xf800020f, 0x05060505, 0xf80008cf,
            0x05040201, 0xbf810000,
        ];
        let ms_insts = disassemble(ms_words).expect("disasm mesh shader");
        let ms_spv = compile_mesh_spv(
            &ms_insts,
            &StageSignature::default(),
            &GraphicsCompileOptions::default(),
            MeshLayout {
                local_size: [32, 1, 1],
                max_vertices: 3,
                max_primitives: 1,
                wave_size: 32,
                lds_bytes: 0,
            },
        )
        .expect("compile MeshEXT SPIR-V");
        assert_valid(&ms_spv);

        // Missing MSG_GS_ALLOC_REQ must be rejected explicitly.
        let mut bad = asm::Asm::new();
        bad.push2(asm::exp_pos0([0, 1, 2, 3], true));
        bad.push(asm::s_endpgm());
        let bad_insts = disassemble(bad.words()).expect("disasm bad mesh");
        assert!(
            compile_mesh_spv(
                &bad_insts,
                &StageSignature::default(),
                &GraphicsCompileOptions::default(),
                MeshLayout::default(),
            )
            .is_err()
        );
    }
}
