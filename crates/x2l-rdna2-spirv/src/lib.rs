//! Direct `RDNA2 ISA → SSA IR → Binary SPIR-V 1.6` shader compiler (`rspirv`).
//!
//! Emits binary SPIR-V 1.6 directly from SSA IR without HLSL text emitters or Windows DXC.

use rspirv::binary::Assemble;
use rspirv::dr::Builder;
use rspirv::spirv::{AddressingModel, Capability, ExecutionModel, MemoryModel};
use thiserror::Error;

/// Errors raised while decoding RDNA2 shaders or emitting binary SPIR-V 1.6.
#[derive(Debug, Error)]
pub enum ShaderError {
    #[error("unsupported shader stage or instruction: {0}")]
    Unsupported(&'static str),
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
    }
}
