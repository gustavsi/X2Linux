// Clean-room headless Vulkan 1.3 offscreen graphics & compute LDS silicon tests.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use std::sync::OnceLock;
use x2l_rdna2_spirv::encode as asm;
use x2l_rdna2_spirv::{
    DrawCall, GpuRunner, MeshLayout, MeshRenderRequest, RenderRequest, SignatureElement,
    StageSignature, Texture2D, Vertex4x2, WaveSize, WaveState,
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

    for mode in ["implicit", "explicit", "biased", "zero"] {
        let mut ps = asm::Asm::new();
        // Interpolate UV into v0, v1
        ps.push(asm::v_interp_p1_f32(0, 0, 0, 0));
        ps.push(asm::v_interp_p2_f32(0, 1, 0, 0));
        ps.push(asm::v_interp_p1_f32(1, 0, 0, 1));
        ps.push(asm::v_interp_p2_f32(1, 1, 0, 1));
        match mode {
            "explicit" => {
                ps.push(asm::v_mov_b32(2, asm::inline_i(0)));
                ps.push2(asm::image_sample_l_2d(4, 0, 0, 4, 0xF));
            }
            "biased" => {
                ps.push(asm::v_mov_b32(2, asm::v(1)));
                ps.push(asm::v_mov_b32(1, asm::v(0)));
                ps.push(asm::v_mov_b32(0, asm::F_1_0));
                ps.push2(asm::image_sample_b_2d(4, 0, 0, 4, 0xF));
            }
            "zero" => {
                ps.push2(asm::image_sample_lz_2d(4, 0, 0, 4, 0xF));
            }
            _ => {
                ps.push2(asm::image_sample_2d(4, 0, 0, 4, 0xF));
            }
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
            "top-left RED ({mode})"
        );
        assert_eq!(
            pixels[16 * 64 + 48],
            [0, 255, 0, 255],
            "top-right GREEN ({mode})"
        );
        assert_eq!(
            pixels[48 * 64 + 16],
            [0, 0, 255, 255],
            "bottom-left BLUE ({mode})"
        );
        assert_eq!(
            pixels[48 * 64 + 48],
            [255, 255, 0, 255],
            "bottom-right YELLOW ({mode})"
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

#[test]
fn renders_triangle_from_rdna2_mesh_shader_and_ps() {
    let runner = shared_runner();
    if !runner.device_info().mesh_shader_enabled {
        println!(
            "[gpu-mesh] Skipping VK_EXT_mesh_shader execution: active device {} does not expose VK_EXT_mesh_shader",
            runner.device_info()
        );
        return;
    }

    // 1. Primary LLVM 22.1.8 (`-mcpu=gfx1030`) assembled RDNA2 Mesh Shader + Pixel Shader:
    //    Allocates 3 vertices & 1 primitive via `s_mov_b32 m0, 0x1003; s_sendmsg MSG_GS_ALLOC_REQ`,
    //    exports primitive connectivity `0x20280600` (indices (0, 1, 2) on lane 0 with `exec_lo = 1`),
    //    and exports a solid yellow triangle across lanes 0..2 (`exec_lo = 7`) via `v_mbcnt_lo_u32_b32`.
    const MS_WORDS: [u32; 30] = [
        0xbefc_03ff,
        0x0000_1003,
        0xbf90_0009,
        0x7e12_02ff,
        0x2028_0600,
        0xbefe_0381,
        0xf800_0941,
        0x0000_0009,
        0xbefe_0387,
        0xd765_0000,
        0x0001_00c1,
        0x7e02_0280,
        0x7e04_02ff,
        0xbf40_0000,
        0x7e06_02ff,
        0x3f40_0000,
        0x7d84_0081,
        0x0202_0701,
        0x7d84_0082,
        0x0202_0501,
        0x7d84_0080,
        0x0204_0503,
        0x7e08_02f0,
        0x7e0a_02f2,
        0x7e0c_0280,
        0xf800_020f,
        0x0506_0505,
        0xf800_08cf,
        0x0504_0201,
        0xbf81_0000,
    ];
    const PS_WORDS: [u32; 11] = [
        0xc810_0000,
        0xc811_0001,
        0xc814_0100,
        0xc815_0101,
        0xc818_0200,
        0xc819_0201,
        0xc81c_0300,
        0xc81d_0301,
        0xf800_180f,
        0x0706_0504,
        0xbf81_0000,
    ];

    let layout = MeshLayout {
        local_size: [32, 1, 1],
        max_vertices: 3,
        max_primitives: 1,
        ..MeshLayout::default()
    };
    let clear = [0.0, 0.0, 32.0 / 255.0, 1.0];

    let pixels = runner
        .render_mesh_offscreen(&MeshRenderRequest {
            width: 64,
            height: 64,
            clear_color: clear,
            ms_words: &MS_WORDS,
            ps_words: &PS_WORDS,
            mesh_layout: layout,
            group_count: [1, 1, 1],
            flip_y: false,
        })
        .expect("render RDNA2 mesh shader triangle");

    assert_eq!(pixels.len(), 64 * 64);
    assert_eq!(
        pixels[36 * 64 + 32],
        [255, 255, 0, 255],
        "mesh triangle interior must render yellow"
    );
    assert_eq!(
        pixels[2 * 64 + 2],
        [0, 0, 32, 255],
        "corner outside mesh triangle must retain clear color"
    );
    let yellow_count = pixels.iter().filter(|&&p| p == [255, 255, 0, 255]).count();
    let bg_count = pixels.iter().filter(|&&p| p == [0, 0, 32, 255]).count();
    assert_eq!(yellow_count, 1152, "exact yellow pixel coverage");
    assert_eq!(bg_count, 2944, "exact background pixel coverage");

    // 2. Cull-bit variant: setting bit 31 (`0x8000_0000`) in `exp prim` sets `CullPrimitiveEXT = true`,
    //    so zero pixels are rasterized and every pixel stays dark-blue clear color.
    let mut ms_culled = MS_WORDS;
    ms_culled[4] = 0xa028_0600;
    let culled_pixels = runner
        .render_mesh_offscreen(&MeshRenderRequest {
            width: 64,
            height: 64,
            clear_color: clear,
            ms_words: &ms_culled,
            ps_words: &PS_WORDS,
            mesh_layout: layout,
            group_count: [1, 1, 1],
            flip_y: false,
        })
        .expect("render culled RDNA2 mesh shader triangle");
    let culled_bg_count = culled_pixels
        .iter()
        .filter(|&&p| p == [0, 0, 32, 255])
        .count();
    assert_eq!(
        culled_bg_count, 4096,
        "cull bit 31 must cull the primitive cleanly"
    );

    // 3. Workgroup LDS (`ds_write_b32` + `s_barrier` + `ds_read_b32`) + EXEC-mask isolation test:
    //    Lane 0 writes solid green (0.0, 1.0, 0.0, 1.0) into LDS[0..16], synchronizes with `s_barrier`,
    //    exports primitive (0, 1, 2) under `exec_lo = 1`, then lanes 0..2 (`exec_lo = 7`) read the shared
    //    green color from LDS and export vertices while inactive lanes carry poisoned values.
    let mut ms_lds = asm::Asm::new();
    ms_lds.push_lit(asm::s_mov_b32(asm::M0, asm::SRC_LIT), 0x0000_1003);
    ms_lds.push(asm::s_sendmsg_gs_alloc_req());
    // Poison v1..v3 across all lanes; only active lanes 0..2 will overwrite and export.
    ms_lds.push_lit(asm::v_mov_b32(1, asm::SRC_LIT), 99.0_f32.to_bits());
    ms_lds.push_lit(asm::v_mov_b32(2, asm::SRC_LIT), 99.0_f32.to_bits());
    // Lane 0 writes green RGBA to LDS[0..12] and exports prim (0, 1, 2)
    ms_lds.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(1)));
    ms_lds.push(asm::v_mov_b32(10, asm::inline_i(0))); // byte addr 0
    ms_lds.push(asm::v_mov_b32(11, asm::inline_i(0))); // 0.0f
    ms_lds.push(asm::v_mov_b32(12, asm::F_1_0)); // 1.0f
    ms_lds.push2(asm::ds_write_b32(10, 11, 0)); // R = 0.0
    ms_lds.push2(asm::ds_write_b32(10, 11, 4));
    ms_lds.push2(asm::v_med3_f32(13, asm::F_2_0, asm::F_NEG_1_0, asm::v(12)));
    ms_lds.push2(asm::ds_add_u32(10, 13, 4)); // G = 1.0
    ms_lds.push2(asm::ds_write_b32(10, 11, 8)); // B = 0.0
    ms_lds.push2(asm::ds_write_b32(10, 12, 12)); // A = 1.0
    ms_lds.push_lit(asm::v_mov_b32(9, asm::SRC_LIT), 0x0020_0400); // prim (0, 1, 2)
    ms_lds.push2(asm::exp_prim(9, true));
    // Re-enable full wave for workgroup barrier
    ms_lds.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(-1)));
    ms_lds.push(asm::s_barrier());
    // Activate lanes 0..2 (`exec_lo = 7`) and compute vertex positions + read color from LDS
    ms_lds.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(7)));
    ms_lds.push2(asm::v_mbcnt_lo_u32_b32(
        0,
        asm::inline_i(-1),
        asm::inline_i(0),
    ));
    ms_lds.push(asm::v_mov_b32(10, asm::inline_i(0)));
    ms_lds.push2(asm::ds_read_b32(4, 10, 0)); // R
    ms_lds.push2(asm::ds_read_b32(5, 10, 4)); // G
    ms_lds.push2(asm::ds_read_b32(6, 10, 8)); // B
    ms_lds.push2(asm::ds_read_b32(7, 10, 12)); // A
    // Positions: lane 0 -> (0.0, -0.75), lane 1 -> (-0.75, 0.75), lane 2 -> (0.75, 0.75)
    ms_lds.push(asm::v_mov_b32(1, asm::inline_i(0))); // default x = 0.0
    ms_lds.push_lit(asm::v_mov_b32(2, asm::SRC_LIT), (-0.75_f32).to_bits()); // default y = -0.75
    ms_lds.push_lit(asm::v_mov_b32(3, asm::SRC_LIT), 0.75_f32.to_bits()); // +0.75
    ms_lds.push_lit(asm::v_mov_b32(13, asm::SRC_LIT), (-0.75_f32).to_bits()); // -0.75
    ms_lds.push2(asm::vop3(
        0xC2,
        16,
        asm::inline_i(1),
        asm::v(0),
        0,
        0,
        0,
        0,
        false,
    ));
    ms_lds.push2(asm::vop3(
        0x101,
        1,
        asm::v(1),
        asm::v(13),
        asm::s(16),
        0,
        0,
        0,
        false,
    ));
    ms_lds.push2(asm::vop3(
        0x101,
        2,
        asm::v(2),
        asm::v(3),
        asm::s(16),
        0,
        0,
        0,
        false,
    ));
    ms_lds.push(asm::v_cmp_eq_u32(asm::inline_i(2), 0));
    ms_lds.push(asm::v_cndmask_b32(1, asm::v(1), 3)); // lane 2: x = +0.75
    ms_lds.push(asm::v_cndmask_b32(2, asm::v(2), 3)); // lane 2: y = +0.75
    ms_lds.push(asm::v_mov_b32(8, asm::F_0_5)); // z = 0.5
    ms_lds.push(asm::v_mov_b32(14, asm::F_1_0)); // w = 1.0
    ms_lds.push2(asm::exp_param(0, [4, 5, 6, 7], false));
    ms_lds.push2(asm::exp_pos0([1, 2, 8, 14], true));
    ms_lds.push(asm::s_endpgm());

    let lds_pixels = runner
        .render_mesh_offscreen(&MeshRenderRequest {
            width: 64,
            height: 64,
            clear_color: clear,
            ms_words: ms_lds.words(),
            ps_words: &PS_WORDS,
            mesh_layout: layout,
            group_count: [1, 1, 1],
            flip_y: false,
        })
        .expect("render LDS+barrier RDNA2 mesh shader triangle");

    assert_eq!(
        lds_pixels[36 * 64 + 32],
        [0, 255, 0, 255],
        "LDS+barrier mesh triangle interior must render green"
    );
    assert_eq!(
        lds_pixels[2 * 64 + 2],
        [0, 0, 32, 255],
        "corner outside LDS+barrier mesh triangle must retain clear color"
    );
}
