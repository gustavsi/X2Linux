// Clean-room synthetic Vulkan image instruction tests.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use ash::vk::ImageViewType;
use std::sync::OnceLock;
use x2l_rdna2_spirv::{GpuRunner, ImageInput, WaveSize, WaveState, encode as asm};

fn runner() -> &'static GpuRunner {
    static RUNNER: OnceLock<GpuRunner> = OnceLock::new();
    RUNNER.get_or_init(|| {
        let runner = GpuRunner::new().expect("initialize Vulkan 1.3");
        println!("[gpu-image-tests] {}", runner.device_info());
        runner
    })
}

const DIRECTIONS: [[f32; 3]; 12] = [
    [4.0, 1.0, 2.0],
    [-4.0, 2.0, -1.0],
    [1.0, 4.0, 2.0],
    [-2.0, -4.0, -1.0],
    [2.0, -1.0, 4.0],
    [-1.0, -2.0, -4.0],
    [3.0, -3.0, 1.0],
    [-4.0, 1.0, -4.0],
    [-1.0, -2.0, 2.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [-0.0, -0.0, -0.0],
];

fn cube_reference([x, y, z]: [f32; 3]) -> [f32; 4] {
    let (axis, major) = [x, y, z]
        .into_iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .unwrap();
    let negative = major < 0.0;
    let (s, t) = match (axis, negative) {
        (0, false) => (-z, -y),
        (0, true) => (z, -y),
        (1, false) => (x, z),
        (1, true) => (x, -z),
        (2, false) => (x, -y),
        (2, true) => (-x, -y),
        _ => unreachable!(),
    };
    [(axis * 2 + usize::from(negative)) as f32, s, t, 2.0 * major]
}

fn cube_alu(a: &mut asm::Asm) {
    let [x, y, z] = [asm::v(0), asm::v(1), asm::v(2)];
    a.push2(asm::v_cubeid_f32(8, x, y, z))
        .push2(asm::v_cubesc_f32(6, x, y, z))
        .push2(asm::v_cubetc_f32(7, x, y, z))
        .push2(asm::v_cubema_f32(9, x, y, z));
}

#[test]
fn cube_alu_matches_reference_on_all_faces_and_axis_ties() {
    let mut a = asm::Asm::new();
    cube_alu(&mut a);
    a.push(asm::s_endpgm());
    let mut init = WaveState::new_wave32();
    for (lane, xyz) in DIRECTIONS.into_iter().enumerate() {
        init.vgpr[lane][..3].copy_from_slice(&xyz.map(f32::to_bits));
    }
    let (state, _) = runner()
        .run_rdna2(a.words(), WaveSize::Wave32, &init, &[])
        .unwrap();
    for (lane, xyz) in DIRECTIONS.into_iter().enumerate() {
        let actual = [8, 6, 7, 9].map(|reg| state.vgpr[lane][reg]);
        assert_eq!(actual, cube_reference(xyz).map(f32::to_bits), "{xyz:?}");
    }
}

#[test]
fn array_and_cube_samples_fetches_and_queries_use_typed_views() {
    let mut a = asm::Asm::new();
    a.push2(asm::mimg(0x24, 16, 3, 0, 8, 0xf, 5, false));
    cube_alu(&mut a);
    // Reciprocal of |ma|, then project to biased face-local texture coordinates.
    a.push2(asm::vop3(0x1aa, 10, asm::v(9), 0, 0, 1, 0, 0, false))
        .push2_lit(
            asm::v_fma_f32(6, asm::v(6), asm::v(10), asm::SRC_LIT),
            1.5_f32.to_bits(),
        )
        .push2_lit(
            asm::v_fma_f32(7, asm::v(7), asm::v(10), asm::SRC_LIT),
            1.5_f32.to_bits(),
        );
    for word in asm::mimg_nsa(asm::mimg(0x27, 20, 6, 0, 8, 0xf, 3, false), [7, 8, 0, 0]) {
        a.push(word);
    }
    a.push2(asm::mimg(0x0e, 24, 11, 0, 0, 0xf, 5, false))
        .push2(asm::mimg(0x0e, 28, 11, 0, 0, 0xf, 3, false))
        .push2(asm::mimg(0x00, 32, 12, 0, 0, 0xf, 3, true))
        .push2(asm::mimg(0x01, 36, 40, 0, 0, 0xf, 5, true))
        .push(asm::s_endpgm());

    let layer_texels = [
        [0.25, 0.5, 0.75, 1.0],
        [0.75, 0.25, 0.5, 1.0],
        [0.0, 1.0, 0.25, 1.0],
    ];
    let face_texels: [[f32; 4]; 24] =
        std::array::from_fn(|i| [i as f32, (i / 4) as f32, (i % 4) as f32, 1.0]);
    let inputs = [
        ImageInput {
            binding: 3,
            view_type: ImageViewType::TYPE_2D_ARRAY,
            extent: [1, 1, 1],
            layers: 3,
            texels: &layer_texels,
            storage: false,
        },
        ImageInput {
            binding: 6,
            view_type: ImageViewType::CUBE,
            extent: [2, 2, 1],
            layers: 6,
            texels: &face_texels,
            storage: false,
        },
        ImageInput {
            binding: 7,
            view_type: ImageViewType::TYPE_2D_ARRAY,
            extent: [2, 2, 1],
            layers: 6,
            texels: &face_texels,
            storage: false,
        },
    ];
    let layers = [0.49_f32, 0.5, 0.51, 1.49, 1.5, 1.51];
    let expected_layers = [0, 0, 1, 1, 2, 2];
    let mut init = WaveState::new_wave32();
    init.exec[0] = 0x3f;
    for (lane, (xyz, layer)) in DIRECTIONS.into_iter().zip(layers).enumerate() {
        let vgpr = &mut init.vgpr[lane];
        vgpr[..3].copy_from_slice(&xyz.map(f32::to_bits));
        vgpr[3..6].copy_from_slice(&[0.5, 0.5, layer].map(f32::to_bits));
        vgpr[14] = lane as u32;
        vgpr[42] = expected_layers[lane] as u32;
    }
    let (state, _) = runner()
        .run_rdna2_with_images(a.words(), &init, &inputs, Some(4))
        .unwrap();
    for (lane, layer) in expected_layers.into_iter().enumerate() {
        let vgpr = &state.vgpr[lane];
        assert_eq!(
            vgpr[16..20],
            layer_texels[layer].map(f32::to_bits),
            "array sample lane {lane}"
        );
        let [_, s, t, ma] = cube_reference(DIRECTIONS[lane]);
        let corner = usize::from(s >= 0.0) + 2 * usize::from(t >= 0.0);
        assert_ne!(ma, 0.0);
        assert_eq!(
            vgpr[20..24],
            face_texels[4 * lane + corner].map(f32::to_bits),
            "cube sample lane {lane}"
        );
        assert_eq!(vgpr[24..28], [1, 1, 3, 1]);
        assert_eq!(vgpr[28..32], [2, 2, 1, 1]);
        assert_eq!(
            vgpr[32..36],
            face_texels[4 * lane].map(f32::to_bits),
            "cube fetch lane {lane}"
        );
        assert_eq!(vgpr[36..40], layer_texels[layer].map(f32::to_bits));
    }
}

#[test]
fn volume_load_and_sample_round_trip_through_2d_and_3d_uavs() {
    let mut a = asm::Asm::new();
    a.push2(asm::mimg(0x00, 8, 0, 8, 0, 0xf, 2, true))
        .push2(asm::mimg(0x08, 8, 3, 0, 0, 0xf, 1, true))
        .push2(asm::mimg(0x00, 32, 0, 4, 0, 0xf, 2, true))
        .push2(asm::mimg(0x08, 8, 0, 4, 0, 0x7, 2, true))
        .push2(asm::mimg(0x0e, 12, 5, 8, 0, 0xf, 2, false))
        .push2(asm::mimg(0x0e, 16, 6, 8, 0, 0xf, 2, false))
        .push2(asm::mimg(0x20, 20, 24, 8, 16, 0xf, 2, false))
        .push2(asm::mimg(0x0e, 28, 5, 4, 0, 0xf, 2, false))
        .push(asm::s_endpgm());
    let texels: [[f32; 4]; 8] =
        std::array::from_fn(|i| [i as f32 - 4.0, i as f32 + 1.25, i as f32 + 0.5, 1.0]);
    let untouched = [-9.0; 4];
    let empty = [untouched; 8];
    let inputs = [
        ImageInput {
            binding: 3,
            view_type: ImageViewType::TYPE_3D,
            extent: [2, 2, 2],
            layers: 1,
            texels: &texels,
            storage: false,
        },
        ImageInput {
            binding: 5,
            view_type: ImageViewType::TYPE_2D,
            extent: [2, 4, 1],
            layers: 1,
            texels: &empty,
            storage: true,
        },
        ImageInput {
            binding: 6,
            view_type: ImageViewType::TYPE_3D,
            extent: [2, 2, 2],
            layers: 1,
            texels: &empty,
            storage: true,
        },
    ];
    let mut init = WaveState::new_wave32();
    init.exec[0] = 0x7f;
    for lane in 0..8u32 {
        let [x, y, z] = [lane & 1, (lane >> 1) & 1, lane >> 2];
        let vgpr = &mut init.vgpr[lane as usize];
        vgpr[..7].copy_from_slice(&[x, y, z, x, y + 2 * z, 0, 2]);
        vgpr[24..27].copy_from_slice(&[x, y, z].map(|c| (0.25_f32 + 0.5 * c as f32).to_bits()));
    }
    let (state, outputs) = runner()
        .run_rdna2_with_images(a.words(), &init, &inputs, Some(4))
        .unwrap();
    for lane in 0..7 {
        let vgpr = &state.vgpr[lane];
        assert_eq!(vgpr[12..16], [2, 2, 2, 1]);
        assert_eq!(vgpr[16..20], [0, 0, 0, 1]);
        assert_eq!(vgpr[20..24], outputs[0][lane].map(f32::to_bits));
        assert_eq!(vgpr[28..32], [2, 2, 2, 1]);
        assert_eq!(vgpr[32..36], untouched.map(f32::to_bits));
    }
    let mut expected = texels;
    expected[7] = untouched;
    assert_eq!(outputs[0], expected);
    expected[..7].iter_mut().for_each(|c| c[3] = 0.0);
    assert_eq!(outputs[1], expected);
}
