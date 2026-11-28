//! Direct `RDNA2 ISA -> Binary SPIR-V 1.6` shader container unpacker, disassembler,
//! test assembler, compute & graphics compiler (`rspirv`), and optional Vulkan 1.3 GPU runner (`ash`).
//!
//! Emits binary SPIR-V 1.6 directly without HLSL text emitters, DXIL, Windows DXC, or
//! `vkd3d-proton`.

pub mod container;
pub mod decode;
pub mod encode;
#[cfg(feature = "gpu-tests")]
pub mod gpu_runner;
pub mod graphics;
mod image;
pub mod spirv;

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel};
use thiserror::Error;

pub use container::{
    CodeSegment, ContainerError, ContainerKind, FssmVariant, HwStage, ParsedContainer, SegmentKind,
    ShaderStage, SignatureElement, StageSignature, classify, parse_container, parse_dxbc_signature,
};
pub use decode::{
    DecodeError, DecodedInst, DstOperand, Encoding, InstExtra, SrcOperand, ValuExt, ValuModifiers,
    disassemble, disassemble_bytes, disassemble_reachable, format_disassembly,
};
#[cfg(feature = "gpu-tests")]
pub use gpu_runner::{
    DrawCall, GpuDeviceInfo, GpuRunner, GpuRunnerError, ImageInput, MeshRenderRequest,
    RenderRequest, Texture2D, Vertex4x2, WaveState,
};
pub use graphics::{
    GraphicsCompileOptions, LiveRegisters, MeshLayout, analyze_live_registers,
    compile_d3d12_compute_spv, compile_graphics_spv, compile_mesh_spv,
};
pub use spirv::{
    CompileOptions, CrossLaneMode, LDS_DWORDS, Rdna2SpirvConverter, SpirvError, WAVE_STATE_STRIDE,
    WaveSize, compile_compute_spv, compile_compute_spv_with_options, is_spirv_val_available,
    spirv_bytes_to_words, spirv_words_to_bytes, validate_spirv_vulkan13,
};

/// Top-level errors raised while unpacking, decoding, or lowering console shaders.
#[derive(Debug, Error)]
pub enum ShaderError {
    #[error("unsupported shader stage or instruction: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Container(#[from] ContainerError),
    #[error(transparent)]
    Decode(#[from] DecodeError),
    #[error(transparent)]
    Spirv(#[from] SpirvError),
}

/// Emit a minimal valid SPIR-V 1.6 compute shader stub using `rspirv` (M0 pipeline smoke test).
pub fn emit_empty_compute_spirv() -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 6);
    b.capability(Capability::Shader);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let void = b.type_void();
    let void_fn = b.type_function(void, vec![]);
    let func = b
        .begin_function(void, None, rspirv::spirv::FunctionControl::NONE, void_fn)
        .expect("begin entry function");
    b.begin_block(None).expect("begin entry block");
    b.ret().expect("emit OpReturn");
    b.end_function().expect("end entry function");

    b.entry_point(ExecutionModel::GLCompute, func, "main", vec![]);
    b.execution_mode(func, rspirv::spirv::ExecutionMode::LocalSize, vec![1, 1, 1]);

    b.module().assemble()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_spirv_1_6_magic_header() {
        let words = emit_empty_compute_spirv();
        assert!(words.len() >= 5);
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic number");
        assert_eq!(words[1], 0x0001_0600, "SPIR-V version 1.6");
        #[cfg(feature = "spirv-val")]
        validate_spirv_vulkan13(&words).expect("empty compute shader must pass spirv-val");
    }
}
