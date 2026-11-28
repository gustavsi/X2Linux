// Clean-room headless Vulkan 1.3 offscreen graphics & compute LDS silicon tests.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use std::sync::OnceLock;
use x2l_rdna2_spirv::encode as asm;
use x2l_rdna2_spirv::{
    DrawCall, GpuRunner, RenderRequest, SignatureElement, StageSignature, Texture2D, Vertex4x2,
    WaveSize, WaveState,
};

fn shared_runner() -> &'static GpuRunner {
    static RUNNER: OnceLock<GpuRunner> = OnceLock::new();
    RUNNER.get_or_init(|| {
        let runner = GpuRunner::new().expect("failed to initialize Vulkan 1.3 GpuRunner");
        println!(
            "[gpu-graphics] Active Vulkan device: {}",
            runner.device_info()
        );
        runner
    })
}

fn pos_attr_sig() -> StageSignature {
    StageSignature {
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
                semantic_name: "TEXCOORD".into(),
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
    }
}

fn passthrough_vs() -> Vec<u32> {
    let mut a = asm::Asm::new();
    // Console VS fetch-shader call bypassed cleanly via ISG1 inputs in v0..v7
    a.push(asm::s_swappc_b64(4, 20));
    a.push2(asm::exp_pos0([0, 1, 2, 3], false));
    a.push2(asm::exp_param(0, [4, 5, 6, 7], true));
    a.push(asm::s_endpgm());
    a.into_words()
}

fn fullscreen_quad(z: f32) -> [Vertex4x2; 6] {
    [
        Vertex4x2 {
            pos: [-1.0, -1.0, z, 1.0],
            attr: [0.0, 0.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [1.0, -1.0, z, 1.0],
            attr: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [1.0, 1.0, z, 1.0],
            attr: [1.0, 1.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [-1.0, -1.0, z, 1.0],
            attr: [0.0, 0.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [1.0, 1.0, z, 1.0],
            attr: [1.0, 1.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [-1.0, 1.0, z, 1.0],
            attr: [0.0, 1.0, 0.0, 1.0],
        },
    ]
}

/// Encode two f32 values (0.0 or 1.0) into IEEE binary16x2 for `exp mrt0 compr`.
const fn pack_f16x2_exact(lo: f32, hi: f32) -> u32 {
    const fn f32_to_f16_simple(x: f32) -> u16 {
        if x == 0.0 {
            0x0000
        } else if x == 0.5 {
            0x3800
        } else if x == 1.0 {
            0x3C00
        } else {
            panic!("unsupported simple f16 constant")
        }
    }
    (f32_to_f16_simple(lo) as u32) | ((f32_to_f16_simple(hi) as u32) << 16)
}

#[test]
fn renders_colored_triangle_from_rdna2_vs_ps() {
    let runner = shared_runner();
    let sig = pos_attr_sig();
    let vs = passthrough_vs();

    // 1. Standard uncompressed exp mrt0 PS reading interpolated RGB color from param0
    let mut ps = asm::Asm::new();
    for ch in 0..4u8 {
        ps.push(asm::v_interp_p1_f32(ch, 0, 0, ch));
        ps.push(asm::v_interp_p2_f32(ch, 1, 0, ch));
    }
    ps.push2(asm::exp_mrt(0, [0, 1, 2, 3], false, true, true));
    ps.push(asm::s_endpgm());

    // Triangle covering bottom-center to top-center of 64x64 target with solid green (0, 255, 0, 255)
    let verts = [
        Vertex4x2 {
            pos: [0.0, -0.75, 0.5, 1.0],
            attr: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [0.75, 0.75, 0.5, 1.0],
            attr: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex4x2 {
            pos: [-0.75, 0.75, 0.5, 1.0],
            attr: [0.0, 1.0, 0.0, 1.0],
        },
    ];

    let pixels = runner
        .render_offscreen(&RenderRequest {
            width: 64,
            height: 64,
            clear_color: [0.0, 0.0, 1.0, 1.0], // clear = blue
            clear_depth: 1.0,
            depth_test: false,
            draws: &[DrawCall {
                vs_words: &vs,
                ps_words: ps.words(),
                vs_sig: &sig,
                vertices: &verts,
                texture: None,
                flip_y: false,
            }],
        })
        .expect("render colored triangle");

    assert_eq!(pixels.len(), 64 * 64);
    // Corner (2, 2) is outside the triangle -> exact clear color (blue)
    assert_eq!(
        pixels[2 * 64 + 2],
        [0, 0, 255, 255],
        "outside triangle retains clear color"
    );
    // Center (32, 36) is well inside the triangle -> exact solid green
    assert_eq!(
        pixels[36 * 64 + 32],
        [0, 255, 0, 255],
        "inside triangle rendered green"
    );

    // 2. Compressed f16x2 (`COMPR=1`) exp mrt0 variant outputting magenta (1.0, 0.0, 1.0, 1.0)
    let mut ps_compr = asm::Asm::new();
    ps_compr.push_lit(asm::v_mov_b32(0, asm::SRC_LIT), pack_f16x2_exact(1.0, 0.0)); // RG = (1.0, 0.0)
    ps_compr.push_lit(asm::v_mov_b32(1, asm::SRC_LIT), pack_f16x2_exact(1.0, 1.0)); // BA = (1.0, 1.0)
    ps_compr.push2(asm::exp_mrt(0, [0, 1, 0, 0], true, true, true));
    ps_compr.push(asm::s_endpgm());

    let pixels_compr = runner
        .render_offscreen(&RenderRequest {
            width: 64,
            height: 64,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            clear_depth: 1.0,
            depth_test: false,
            draws: &[DrawCall {
                vs_words: &vs,
                ps_words: ps_compr.words(),
                vs_sig: &sig,
                vertices: &verts,
                texture: None,
                flip_y: false,
            }],
        })
        .expect("render COMPR f16x2 triangle");

    assert_eq!(pixels_compr[2 * 64 + 2], [0, 0, 0, 255]);
    assert_eq!(
        pixels_compr[36 * 64 + 32],
        [255, 0, 255, 255],
        "COMPR=1 f16x2 unpacked magenta accurately"
    );
}

#[test]
fn renders_textured_quad_with_image_sample() {
    let runner = shared_runner();
    let sig = pos_attr_sig();
    let vs = passthrough_vs();
    let quad = fullscreen_quad(0.5);

    // 4x4 RGBA checkerboard with 4 distinct quadrant colors:
    //   top-left (u<0.5, v<0.5): RED
    //   top-right (u>=0.5, v<0.5): GREEN
    //   bottom-left (u<0.5, v>=0.5): BLUE
    //   bottom-right (u>=0.5, v>=0.5): YELLOW
    let mut rgba = Vec::with_capacity(4 * 4 * 4);
    for y in 0..4 {
        for x in 0..4 {
            let color: [u8; 4] = match (x >= 2, y >= 2) {
                (false, false) => [255, 0, 0, 255],
                (true, false) => [0, 255, 0, 255],
                (false, true) => [0, 0, 255, 255],
                (true, true) => [255, 255, 0, 255],
            };
            rgba.extend_from_slice(&color);
        }
    }
    let tex = Texture2D {
        width: 4,
        height: 4,
        rgba,
    };

    // Test both `image_sample` (implicit LOD) and `image_sample_l` (explicit LOD = 0.0)
    for use_explicit_lod in [false, true] {
        let mut ps = asm::Asm::new();
        // Interpolate UV into v0, v1
        ps.push(asm::v_interp_p1_f32(0, 0, 0, 0));
        ps.push(asm::v_interp_p2_f32(0, 1, 0, 0));
        ps.push(asm::v_interp_p1_f32(1, 0, 0, 1));
        ps.push(asm::v_interp_p2_f32(1, 1, 0, 1));
        if use_explicit_lod {
            ps.push(asm::v_mov_b32(2, asm::inline_i(0))); // lod = 0.0f
            ps.push2(asm::image_sample_l_2d(4, 0, 0, 4, 0xF));
        } else {
            ps.push2(asm::image_sample_2d(4, 0, 0, 4, 0xF));
        }
        ps.push2(asm::exp_mrt(0, [4, 5, 6, 7], false, true, true));
        ps.push(asm::s_endpgm());

        let pixels = runner
            .render_offscreen(&RenderRequest {
                width: 64,
                height: 64,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                clear_depth: 1.0,
                depth_test: false,
                draws: &[DrawCall {
                    vs_words: &vs,
                    ps_words: ps.words(),
                    vs_sig: &sig,
                    vertices: &quad,
                    texture: Some(&tex),
                    flip_y: false,
                }],
            })
            .expect("render textured quad");

        assert_eq!(
            pixels[16 * 64 + 16],
            [255, 0, 0, 255],
            "top-left RED (lod={use_explicit_lod})"
        );
        assert_eq!(
            pixels[16 * 64 + 48],
            [0, 255, 0, 255],
            "top-right GREEN (lod={use_explicit_lod})"
        );
        assert_eq!(
            pixels[48 * 64 + 16],
            [0, 0, 255, 255],
            "bottom-left BLUE (lod={use_explicit_lod})"
        );
        assert_eq!(
            pixels[48 * 64 + 48],
            [255, 255, 0, 255],
            "bottom-right YELLOW (lod={use_explicit_lod})"
        );
    }
}

#[test]
fn pixel_shader_discard_and_mrtz_depth() {
    let runner = shared_runner();
    let sig = pos_attr_sig();
    let vs = passthrough_vs();
    let quad_front = fullscreen_quad(0.8); // vertex z = 0.8, overridden by exp mrtz = 0.25!
    let quad_mid = fullscreen_quad(0.5); // second pass at z = 0.5

    // PS1: interpolate u into v0; discard left half (u < 0.5) via s_and_saveexec -> OpDemoteToHelperInvocation;
    // on surviving right half, export RED and custom depth 0.25 via exp mrtz!
    let mut ps1 = asm::Asm::new();
    ps1.push(asm::v_interp_p1_f32(0, 0, 0, 0));
    ps1.push(asm::v_interp_p2_f32(0, 1, 0, 0));
    ps1.push(asm::v_mov_b32(1, asm::F_0_5)); // v1 = 0.5
    ps1.push(asm::v_cmp_gt_f32(asm::v(0), 1)); // vcc = (u > 0.5)
    ps1.push(asm::s_and_saveexec_b32(0, asm::VCC_LO)); // left half becomes inactive -> demoted!
    ps1.push_lit(asm::v_mov_b32(2, asm::SRC_LIT), 0.25_f32.to_bits()); // custom depth = 0.25
    ps1.push(asm::v_mov_b32(4, asm::F_1_0)); // R = 1
    ps1.push(asm::v_mov_b32(5, asm::inline_i(0))); // G = 0
    ps1.push(asm::v_mov_b32(6, asm::inline_i(0))); // B = 0
    ps1.push(asm::v_mov_b32(7, asm::F_1_0)); // A = 1
    ps1.push2(asm::exp_mrtz(2, false, true));
    ps1.push2(asm::exp_mrt(0, [4, 5, 6, 7], false, true, true));
    ps1.push(asm::s_endpgm());

    // 1. Single-draw check: left half discarded -> exact clear color (green); right half -> RED
    let pixels_discard = runner
        .render_offscreen(&RenderRequest {
            width: 64,
            height: 64,
            clear_color: [0.0, 1.0, 0.0, 1.0],
            clear_depth: 1.0,
            depth_test: true,
            draws: &[DrawCall {
                vs_words: &vs,
                ps_words: ps1.words(),
                vs_sig: &sig,
                vertices: &quad_front,
                texture: None,
                flip_y: false,
            }],
        })
        .expect("render discard + mrtz single draw");

    assert_eq!(
        pixels_discard[32 * 64 + 10],
        [0, 255, 0, 255],
        "discarded left pixel must retain clear color"
    );
    assert_eq!(
        pixels_discard[32 * 64 + 54],
        [255, 0, 0, 255],
        "surviving right pixel must render RED"
    );

    // 2. Two-draw depth test check:
    //    Draw 1 wrote depth 0.25 on right half (vertex z was 0.8) and discarded left half (depth stays 1.0).
    //    Draw 2 draws solid BLUE at z = 0.5 with LESS depth test:
    //      - Left half (0.5 < 1.0): passes -> BLUE (proving discarded pixels didn't write depth!)
    //      - Right half (0.5 < 0.25): fails -> stays RED (proving exp mrtz=0.25 overrode vertex z=0.8!)
    let mut ps2 = asm::Asm::new();
    ps2.push(asm::v_mov_b32(4, asm::inline_i(0)));
    ps2.push(asm::v_mov_b32(5, asm::inline_i(0)));
    ps2.push(asm::v_mov_b32(6, asm::F_1_0));
    ps2.push(asm::v_mov_b32(7, asm::F_1_0));
    ps2.push2(asm::exp_mrt(0, [4, 5, 6, 7], false, true, true));
    ps2.push(asm::s_endpgm());

    let pixels_depth = runner
        .render_offscreen(&RenderRequest {
            width: 64,
            height: 64,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            clear_depth: 1.0,
            depth_test: true,
            draws: &[
                DrawCall {
                    vs_words: &vs,
                    ps_words: ps1.words(),
                    vs_sig: &sig,
                    vertices: &quad_front,
                    texture: None,
                    flip_y: false,
                },
                DrawCall {
                    vs_words: &vs,
                    ps_words: ps2.words(),
                    vs_sig: &sig,
                    vertices: &quad_mid,
                    texture: None,
                    flip_y: false,
                },
            ],
        })
        .expect("render two-pass mrtz depth test");

    assert_eq!(
        pixels_depth[32 * 64 + 10],
        [0, 0, 255, 255],
        "left half (discarded in draw 1) must accept draw 2 BLUE at z=0.5"
    );
    assert_eq!(
        pixels_depth[32 * 64 + 54],
        [255, 0, 0, 255],
        "right half (exp mrtz=0.25 in draw 1) must occlude draw 2 BLUE at z=0.5"
    );
}

#[test]
fn compute_shader_lds_ds_read_write_barrier() {
    let runner = shared_runner();

    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let lanes = wave_size.width() as usize;
        let mut a = asm::Asm::new();

        // Each lane writes v1 (0x5000 + lane) to LDS byte address v0 (lane * 4) via ds_write_b32,
        // and writes paired (v2, v3) to LDS at offset0=64, offset1=128 via ds_write2_b32.
        a.push2(asm::ds_write_b32(0, 1, 0));
        a.push2(asm::ds_write2_b32(0, 2, 3, 64, 128));
        a.push(asm::s_barrier());
        // Read reversed neighbor lane address v4 ((lanes - 1 - lane) * 4) via ds_read_b32 & ds_read2_b32!
        a.push2(asm::ds_read_b32(5, 4, 0));
        a.push2(asm::ds_read2_b32(6, 4, 64, 128));
        a.push(asm::s_endpgm());

        let mut init = match wave_size {
            WaveSize::Wave32 => WaveState::new_wave32(),
            WaveSize::Wave64 => WaveState::new_wave64(),
        };
        init.fill_vgpr(0, lanes, |l| (l * 4) as u32);
        init.fill_vgpr(1, lanes, |l| 0x5000 + l as u32);
        init.fill_vgpr(2, lanes, |l| 0x6000 + l as u32);
        init.fill_vgpr(3, lanes, |l| 0x7000 + l as u32);
        init.fill_vgpr(4, lanes, |l| ((lanes - 1 - l) * 4) as u32);

        let (out, _) = runner
            .run_rdna2(a.words(), wave_size, &init, &[])
            .expect("compute LDS dispatch");

        for (lane, bank) in out.vgpr.iter().enumerate().take(lanes) {
            let rev = (lanes - 1 - lane) as u32;
            assert_eq!(bank[5], 0x5000 + rev, "ds_read_b32 reversed lane {lane}");
            assert_eq!(
                bank[6],
                0x6000 + rev,
                "ds_read2_b32[0] reversed lane {lane}"
            );
            assert_eq!(
                bank[7],
                0x7000 + rev,
                "ds_read2_b32[1] reversed lane {lane}"
            );
        }
    }
}
