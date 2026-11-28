// Clean-room tests for GFX10.3 image and cube instruction encoding and lowering.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

use rspirv::dr::{Operand, load_words};
use rspirv::spirv::{Capability, Decoration, Dim, ImageFormat, Op};
use x2l_rdna2_spirv::{
    GraphicsCompileOptions, InstExtra, ShaderStage, StageSignature, compile_compute_spv,
    compile_d3d12_compute_spv, compile_graphics_spv, disassemble, encode as asm,
};

#[test]
fn decodes_llvm_gfx1030_cube_alu_and_nsa_address_order() {
    let [x, y, z] = [asm::v(1), asm::v(2), asm::v(3)];
    let encoders = [
        (asm::v_cubeid_f32(0, x, y, z), "v_cubeid_f32"),
        (asm::v_cubesc_f32(0, x, y, z), "v_cubesc_f32"),
        (asm::v_cubetc_f32(0, x, y, z), "v_cubetc_f32"),
        (asm::v_cubema_f32(0, x, y, z), "v_cubema_f32"),
    ];
    for (offset, (encoded, mnemonic)) in encoders.into_iter().enumerate() {
        assert_eq!(
            encoded,
            [0xd544_0000 + ((offset as u32) << 16), 0x040e_0501]
        );
        assert_eq!(disassemble(&encoded).unwrap()[0].mnemonic, mnemonic);
    }

    let sample = asm::mimg_nsa(asm::mimg(0x20, 4, 0, 0, 8, 0xf, 3, false), [2, 4, 0, 0]);
    assert_eq!(sample, [0xf080_0f1a, 0x0040_0400, 0x0000_0402]);
    let inst = &disassemble(&sample).unwrap()[0];
    assert_eq!(inst.word_len, 3);
    let InstExtra::Mimg {
        dim,
        dmask,
        is_nsa,
        addrs,
        ..
    } = inst.extra
    else {
        panic!("expected MIMG");
    };
    assert!(is_nsa);
    assert_eq!((dim, dmask, &addrs[..3]), (3, 0xf, &[0, 2, 4][..]));
}

#[test]
fn compute_and_fragment_declare_distinct_image_types_and_access() {
    let mut a = asm::Asm::new();
    a.push2(asm::mimg(0x20, 8, 0, 0, 16, 0xf, 3, false))
        .push2(asm::mimg(0x20, 8, 0, 0, 16, 0xf, 5, false))
        .push2(asm::mimg(0x00, 8, 0, 0, 0, 0xf, 2, true))
        .push2(asm::mimg(0x08, 8, 0, 8, 0, 0xf, 1, true))
        .push2(asm::mimg(0x08, 8, 0, 8, 0, 0xf, 2, true))
        .push2(asm::mimg(0x00, 8, 0, 8, 0, 0xf, 2, true))
        .push(asm::s_endpgm());
    let insts = disassemble(a.words()).unwrap();
    for words in [
        compile_compute_spv(&insts),
        compile_d3d12_compute_spv(&insts, [1, 1, 1], false, false),
        compile_graphics_spv(
            ShaderStage::Pixel,
            &insts,
            &StageSignature::default(),
            &GraphicsCompileOptions::default(),
        ),
    ] {
        let words = words.unwrap();
        #[cfg(feature = "spirv-val")]
        x2l_rdna2_spirv::validate_spirv_vulkan13(&words).unwrap();
        let module = load_words(&words).unwrap();
        let count = module
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::TypeImage)
            .count();
        assert_eq!(count, 5);
        for (dim, arrayed, sampled) in [
            (Dim::DimCube, 0, 1),
            (Dim::Dim2D, 1, 1),
            (Dim::Dim3D, 0, 1),
            (Dim::Dim2D, 0, 2),
            (Dim::Dim3D, 0, 2),
        ] {
            assert!(module.types_global_values.iter().any(|i| {
                i.class.opcode == Op::TypeImage
                    && i.operands[1] == Operand::Dim(dim)
                    && i.operands[3] == Operand::LiteralBit32(arrayed)
                    && i.operands[5] == Operand::LiteralBit32(sampled)
                    && i.operands[6] == Operand::ImageFormat(ImageFormat::Unknown)
            }));
        }
        for capability in [
            Capability::StorageImageWriteWithoutFormat,
            Capability::StorageImageReadWithoutFormat,
        ] {
            assert!(
                module
                    .capabilities
                    .iter()
                    .any(|i| i.operands == [Operand::Capability(capability)])
            );
        }
        assert_eq!(
            module
                .annotations
                .iter()
                .filter(|i| i
                    .operands
                    .contains(&Operand::Decoration(Decoration::NonReadable)))
                .count(),
            1
        );
    }
}

#[test]
fn image_dimensions_validate_with_mip_queries_and_multisample_fetch() {
    let mut a = asm::Asm::new();
    for dim in 0..8 {
        a.push2(asm::mimg(0x00, 8, 0, dim * 4, 0, 0xf, dim, true));
        a.push2(asm::mimg(0x0e, 8, 0, dim * 4, 0, 0xf, dim, false));
    }
    a.push(asm::s_endpgm());
    let words = compile_compute_spv(&disassemble(a.words()).unwrap()).unwrap();
    #[cfg(feature = "spirv-val")]
    x2l_rdna2_spirv::validate_spirv_vulkan13(&words).unwrap();
    assert_eq!(
        load_words(&words).unwrap().types_global_values.iter()
            .filter(|i| i.class.opcode == Op::TypeImage && i.operands[4] == Operand::LiteralBit32(1))
            .count(),
        2,
    );

    let unknown = asm::mimg(0xff, 8, 0, 0, 0, 0xf, 2, false);
    let insts = disassemble(&unknown).unwrap();
    assert!(matches!(
        compile_compute_spv(&insts),
        Err(x2l_rdna2_spirv::SpirvError::UnsupportedInstruction { pc: 0, mnemonic })
            if mnemonic.contains("mimg_0xff")
    ));
}
