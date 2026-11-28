// Clean-room bit-exact Vulkan 1.3 GPU instruction test suite for x2l-rdna2-spirv.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use std::sync::OnceLock;
use x2l_rdna2_spirv::encode as asm;
use x2l_rdna2_spirv::{CompileOptions, CrossLaneMode, GpuRunner, WaveSize, WaveState};

fn shared_runner() -> &'static GpuRunner {
    static RUNNER: OnceLock<GpuRunner> = OnceLock::new();
    RUNNER.get_or_init(|| {
        let runner = GpuRunner::new().expect("failed to initialize Vulkan 1.3 GpuRunner");
        println!("[gpu-tests] Active Vulkan device: {}", runner.device_info());
        runner
    })
}

fn wave_state(wave_size: WaveSize) -> WaveState {
    match wave_size {
        WaveSize::Wave32 => WaveState::new_wave32(),
        WaveSize::Wave64 => WaveState::new_wave64(),
    }
}

#[test]
fn gpu_scalar_and_vector_alu_legacy_mul_fma_mad_and_vop3_modifiers() {
    let runner = shared_runner();

    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let lanes = wave_size.width() as usize;
        let mut a = asm::Asm::new();

        // Scalar ALU: carry, borrow, bitwise, mul
        a.push_lit(asm::s_mov_b32(0, asm::SRC_LIT), 0xFFFF_FFFE);
        a.push(asm::s_add_u32(1, 0, asm::inline_i(5))); // s1 = 3, scc = 1 (carry)
        a.push(asm::s_mov_b32(2, asm::SRC_SCC)); // s2 = 1
        a.push(asm::s_sub_u32(3, asm::inline_i(2), asm::inline_i(5))); // s3 = -3u, scc = 1 (borrow)
        a.push(asm::s_mov_b32(4, asm::SRC_SCC)); // s4 = 1
        a.push_lit(asm::s_andn2_b32(5, asm::SRC_LIT, asm::inline_i(0x0F)), 0xFF); // s5 = 0xF0
        a.push(asm::s_mul_i32(6, asm::inline_i(7), asm::inline_i(-6))); // s6 = -42

        // Vector ALU & IEEE / legacy / console gfx10.1 float ops
        // Inputs in v0, v1, v2 per lane:
        //   lane 0: 0.0, qNaN, 3.0
        //   lane 1: -0.0, +Inf, -4.0
        //   lane 2: -3.0, 2.0, 0.5
        //   lane 3: 4.0, 9.0, -0.0
        a.push(asm::v_mul_legacy_f32(10, asm::v(0), 1)); // v10 = legacy_mul(v0, v1)
        a.push(asm::v_mul_f32(11, asm::v(0), 1)); // v11 = ieee mul
        a.push2(asm::v_fma_f32(12, asm::v(0), asm::v(1), asm::v(2))); // v12 = fma
        a.push2(asm::v_mad_f32(13, asm::v(0), asm::v(1), asm::v(2))); // v13 = gfx10.1 mad
        a.push(asm::v_mov_b32(14, asm::v(2)));
        a.push(asm::v_fmac_f32(14, asm::v(0), 1)); // v14 = fmac
        a.push(asm::v_mov_b32(15, asm::v(2)));
        a.push(asm::v_mac_f32(15, asm::v(0), 1)); // v15 = gfx10.1 mac

        // VOP3 modifiers: v16 = clamp((|v0| + -v1) * 2.0)  (abs0=1, neg1=2, omod=1 (*2), clamp=true)
        a.push2(asm::vop3(
            0x103,
            16,
            asm::v(0),
            asm::v(1),
            0,
            0b001,
            0b010,
            1,
            true,
        ));
        // v_min_f32 / v_max_f32 (NaN & signed-zero handling)
        a.push(asm::v_min_f32(17, asm::v(0), 1));
        a.push(asm::v_max_f32(18, asm::v(0), 1));

        // Unary math & conversions on v3 (exact powers of 2 / edge values)
        a.push(asm::v_rcp_f32(19, asm::v(3)));
        a.push(asm::v_sqrt_f32(20, asm::v(4)));
        a.push(asm::v_rsq_f32(21, asm::v(3)));
        a.push(asm::v_cvt_i32_f32(22, asm::v(5)));
        a.push(asm::v_cvt_u32_f32(23, asm::v(5)));
        a.push(asm::v_cvt_f32_i32(24, asm::v(6)));
        a.push(asm::v_cvt_f32_u32(25, asm::v(6)));
        a.push(asm::s_endpgm());

        let mut init = wave_state(wave_size);
        let qnan = 0x7FC0_0000u32;
        init.vgpr[0][0] = 0.0_f32.to_bits();
        init.vgpr[0][1] = qnan;
        init.vgpr[0][2] = 3.0_f32.to_bits();

        init.vgpr[1][0] = (-0.0_f32).to_bits();
        init.vgpr[1][1] = f32::INFINITY.to_bits();
        init.vgpr[1][2] = (-4.0_f32).to_bits();

        init.vgpr[2][0] = (-3.0_f32).to_bits();
        init.vgpr[2][1] = 2.0_f32.to_bits();
        init.vgpr[2][2] = 0.5_f32.to_bits();

        // Lane 3: -0.0 vs +0.0 for min/max
        init.vgpr[3][0] = (-0.0_f32).to_bits();
        init.vgpr[3][1] = 0.0_f32.to_bits();
        init.vgpr[3][2] = 1.0_f32.to_bits();

        // Lane 4: v0=NaN, v1=5.0 (tests min/max with NaN -> 5.0 and clamp(NaN) -> +0.0)
        init.vgpr[4][0] = qnan;
        init.vgpr[4][1] = 5.0_f32.to_bits();
        init.vgpr[4][2] = 0.0_f32.to_bits();

        // Lane 5: (-1.0) vs (-2.0): (|-1.0| + -(-2.0))*2 = 6.0 -> clamp 1.0; and in-range test on lane 6
        init.vgpr[6][0] = (-0.75_f32).to_bits();
        init.vgpr[6][1] = 0.5_f32.to_bits(); // (0.75 - 0.5) * 2 = 0.5

        // Unary math & cvt inputs
        for bank in init.vgpr.iter_mut().take(lanes) {
            bank[3] = 4.0_f32.to_bits();
            bank[4] = 9.0_f32.to_bits();
            bank[6] = (-7i32) as u32;
        }
        init.vgpr[0][5] = (-12.75_f32).to_bits();
        init.vgpr[1][5] = qnan;
        init.vgpr[2][5] = 1.0e20_f32.to_bits();
        init.vgpr[3][5] = (-1.0e20_f32).to_bits();

        let (out, _) = runner
            .run_rdna2(a.words(), wave_size, &init, &[])
            .expect("GPU ALU dispatch");

        // Scalar assertions
        assert_eq!(out.sgpr[1], 3);
        assert_eq!(out.sgpr[2], 1, "s_add_u32 carry SCC");
        assert_eq!(out.sgpr[3], (-3i32) as u32);
        assert_eq!(out.sgpr[4], 1, "s_sub_u32 borrow SCC");
        assert_eq!(out.sgpr[5], 0xF0);
        assert_eq!(out.sgpr[6], (-42i32) as u32);

        // v_mul_legacy_f32: 0.0 * NaN = +0.0, -0.0 * +Inf = +0.0, -3.0 * 2.0 = -6.0
        assert_eq!(out.vgpr[0][10], 0.0_f32.to_bits(), "0.0 * NaN == +0.0");
        assert_eq!(out.vgpr[1][10], 0.0_f32.to_bits(), "-0.0 * +Inf == +0.0");
        assert_eq!(out.vgpr[2][10], (-6.0_f32).to_bits());

        // FMA / MAD / FMAC / MAC on lane 2: (-3.0)*2.0 + 0.5 = -5.5
        assert_eq!(out.vgpr[2][12], (-5.5_f32).to_bits());
        assert_eq!(out.vgpr[2][13], (-5.5_f32).to_bits());
        assert_eq!(out.vgpr[2][14], (-5.5_f32).to_bits());
        assert_eq!(out.vgpr[2][15], (-5.5_f32).to_bits());

        // VOP3 abs + neg + omod(*2) + clamp
        assert_eq!(
            out.vgpr[2][16],
            1.0_f32.to_bits(),
            "(|-3| - 2)*2 = 2 -> clamp 1.0"
        );
        assert_eq!(
            out.vgpr[4][16],
            0.0_f32.to_bits(),
            "DX10 clamp(NaN) == +0.0"
        );
        assert_eq!(
            out.vgpr[6][16],
            0.5_f32.to_bits(),
            "(|-0.75| - 0.5)*2 == 0.5"
        );

        // v_min_f32 / v_max_f32 signed zero & NaN
        assert_eq!(out.vgpr[3][17], (-0.0_f32).to_bits(), "min(-0, +0) == -0");
        assert_eq!(out.vgpr[3][18], 0.0_f32.to_bits(), "max(-0, +0) == +0");
        assert_eq!(out.vgpr[4][17], 5.0_f32.to_bits(), "min(NaN, 5.0) == 5.0");
        assert_eq!(out.vgpr[4][18], 5.0_f32.to_bits(), "max(NaN, 5.0) == 5.0");

        // Unary math & saturating float-to-int conversions
        assert_eq!(out.vgpr[0][19], 0.25_f32.to_bits(), "rcp(4.0) == 0.25");
        assert_eq!(out.vgpr[0][20], 3.0_f32.to_bits(), "sqrt(9.0) == 3.0");
        assert_eq!(out.vgpr[0][21], 0.5_f32.to_bits(), "rsq(4.0) == 0.5");

        assert_eq!(out.vgpr[0][22], (-12i32) as u32);
        assert_eq!(out.vgpr[1][22], 0, "cvt_i32(NaN) == 0");
        assert_eq!(
            out.vgpr[2][22],
            i32::MAX as u32,
            "cvt_i32(+1e20) == INT_MAX"
        );
        assert_eq!(
            out.vgpr[3][22],
            i32::MIN as u32,
            "cvt_i32(-1e20) == INT_MIN"
        );

        assert_eq!(out.vgpr[0][23], 0, "cvt_u32(-12.75) == 0");
        assert_eq!(out.vgpr[1][23], 0, "cvt_u32(NaN) == 0");
        assert_eq!(out.vgpr[2][23], u32::MAX, "cvt_u32(+1e20) == UINT_MAX");
        assert_eq!(out.vgpr[0][24], (-7.0_f32).to_bits());
        assert_eq!(out.vgpr[0][25], (((-7i32) as u32) as f32).to_bits());
    }
}

#[test]
fn gpu_exec_divergence_half_wave_execz_skip_and_loop() {
    let runner = shared_runner();

    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let lanes = wave_size.width() as usize;
        let is_w64 = wave_size == WaveSize::Wave64;
        let mut a = asm::Asm::new();

        // v_cmp_lt_u32 vcc, v0, 16  (NOTE: vopc src0=v0, vsrc1=v10 where v10 holds 16)
        a.push(asm::v_cmp_lt_u32(asm::v(0), 10));

        // Save EXEC to s[0:1] (or s0) and mask to lanes < 16
        if is_w64 {
            a.push(asm::s_and_saveexec_b64(0, asm::VCC_LO));
        } else {
            a.push(asm::s_and_saveexec_b32(0, asm::VCC_LO));
        }
        // Only lanes 0..15 execute this write into v1
        a.push(asm::v_add_nc_u32(1, asm::v(0), 11)); // v1 = v0 + 100

        // Switch EXEC to the else-half: exec = saved & ~exec
        if is_w64 {
            a.push(asm::s_andn2_b64(asm::EXEC_LO, 0, asm::EXEC_LO));
        } else {
            a.push(asm::s_andn2_b32(asm::EXEC_LO, 0, asm::EXEC_LO));
        }
        // Only lanes 16..W-1 execute this write into v2
        a.push(asm::v_add_nc_u32(2, asm::v(0), 12)); // v2 = v0 + 200

        // Clear EXEC to 0 and verify s_cbranch_execz skips the poison block
        if is_w64 {
            a.push(asm::s_mov_b64(asm::EXEC_LO, asm::inline_i(0)));
        } else {
            a.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(0)));
        }
        let skip_poison = a.branch_placeholder(0x08); // s_cbranch_execz
        a.push(asm::s_movk_i32(10, 999)); // MUST BE SKIPPED
        a.push(asm::v_mov_b32(3, asm::inline_i(7)));
        a.patch_branch_here(skip_poison);

        // Restore full EXEC from s[0:1] / s0
        if is_w64 {
            a.push(asm::s_mov_b64(asm::EXEC_LO, 0));
        } else {
            a.push(asm::s_mov_b32(asm::EXEC_LO, 0));
        }

        // 5-iteration loop: s5 += s6 (s6 = 5..1 => 15), v4 += v0 (=> 5 * lane)
        a.push(asm::s_movk_i32(6, 5));
        a.push(asm::s_movk_i32(5, 0));
        let loop_pc = a.pc();
        a.push(asm::s_add_u32(5, 5, 6));
        a.push(asm::v_add_nc_u32(4, asm::v(0), 4));
        a.push(asm::s_sub_u32(6, 6, asm::inline_i(1)));
        a.push(asm::s_cmp_lg_u32(6, asm::inline_i(0)));
        a.branch_to_pc(0x05, loop_pc); // s_cbranch_scc1
        a.push(asm::s_endpgm());

        let mut init = wave_state(wave_size);
        init.fill_vgpr(0, lanes, |l| l as u32);
        init.fill_vgpr(1, lanes, |_| 0xDEAD_BEEF);
        init.fill_vgpr(2, lanes, |_| 0x1111_1111);
        init.fill_vgpr(3, lanes, |_| 0x2222_2222);
        init.fill_vgpr(4, lanes, |_| 0);
        init.fill_vgpr(10, lanes, |_| 16);
        init.fill_vgpr(11, lanes, |_| 100);
        init.fill_vgpr(12, lanes, |_| 200);

        let (out, _) = runner
            .run_rdna2(a.words(), wave_size, &init, &[])
            .expect("GPU divergence dispatch");

        assert_eq!(out.vcc[0], 0x0000_FFFF, "vcc_lo should mark lanes 0..15");
        if is_w64 {
            assert_eq!(out.vcc[1], 0, "vcc_hi should be 0 for lanes < 16");
        }
        assert_eq!(out.sgpr[10], 0, "s_cbranch_execz must skip poison block");
        assert_eq!(out.sgpr[5], 15, "loop sum 5+4+3+2+1 == 15");

        for (lane, bank) in out.vgpr.iter().enumerate().take(lanes) {
            if lane < 16 {
                assert_eq!(bank[1], 100 + lane as u32, "then-branch lane {lane}");
                assert_eq!(bank[2], 0x1111_1111, "else-branch untouched lane {lane}");
            } else {
                assert_eq!(bank[1], 0xDEAD_BEEF, "then-branch untouched lane {lane}");
                assert_eq!(bank[2], 200 + lane as u32, "else-branch lane {lane}");
            }
            assert_eq!(bank[3], 0x2222_2222, "poison vector write skipped");
            assert_eq!(bank[4], 5 * lane as u32, "loop accumulation lane {lane}");
        }
    }
}

#[test]
fn gpu_cross_lane_readfirstlane_readlane_writelane_swizzle_permlane_and_dpp() {
    let runner = shared_runner();

    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let lanes = wave_size.width() as usize;
        let is_w64 = wave_size == WaveSize::Wave64;
        let mut a = asm::Asm::new();

        // 1. Disable lanes 0, 1, 2 -> first active lane is 3
        a.push_lit(asm::s_mov_b32(asm::EXEC_LO, asm::SRC_LIT), 0xFFFF_FFF8);
        a.push(asm::v_readfirstlane_b32(0, 0)); // s0 = v0[3] = 0x1003
        a.push2(asm::v_readlane_b32(1, 0, asm::inline_i(1))); // s1 = v0[1] = 0x1001 (inactive lane!)
        a.push2(asm::v_writelane_b32(1, asm::s(2), asm::inline_i(1))); // v1[1] = s2 (inactive lane!)

        // 2. Restore full EXEC for full-wave shuffles
        a.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(-1)));
        if is_w64 {
            a.push(asm::s_mov_b32(asm::EXEC_HI, asm::inline_i(-1)));
        }

        // ds_swizzle_b32 quad permute 0x801B (reverse quad: [3,2,1,0])
        a.push2(asm::ds_swizzle_b32(2, 0, 0x801B));
        // ds_swizzle_b32 bitmode 0x041F (and=31, or=0, xor=1 -> swap even/odd)
        a.push2(asm::ds_swizzle_b32(3, 0, 0x041F));

        // v_permlane16_b32 & v_permlanex16_b32 with reverse-16 nibble tables in s4, s5
        a.push2(asm::v_permlane16(
            false,
            4,
            0,
            asm::s(4),
            asm::s(5),
            true,
            false,
        ));
        a.push2(asm::v_permlane16(
            true,
            5,
            0,
            asm::s(4),
            asm::s(5),
            true,
            false,
        ));

        // DPP16 quad permute 0xB1 ([1,0,3,2] -> lane ^ 1)
        a.push_lit(
            asm::v_mov_b32(6, asm::SRC_DPP16),
            asm::dpp16_word(0, 0xB1, true, true, 0xF, 0xF),
        );
        // DPP16 row_shl:1 (0x101) with bc=1 (lane 15 in each row gets 0)
        a.push_lit(
            asm::v_mov_b32(7, asm::SRC_DPP16),
            asm::dpp16_word(0, 0x101, true, true, 0xF, 0xF),
        );
        // DPP8 rotate-by-1 within each 8-lane group: [1,2,3,4,5,6,7,0]
        a.push_lit(
            asm::v_mov_b32(8, asm::SRC_DPP8),
            asm::dpp8_word(0, [1, 2, 3, 4, 5, 6, 7, 0]),
        );
        a.push(asm::s_endpgm());

        let mut init = wave_state(wave_size);
        init.fill_vgpr(0, lanes, |l| 0x1000 + l as u32);
        init.fill_vgpr(1, lanes, |l| 0xAAAA_0000 + l as u32);
        init.sgpr[2] = 0xCAFE_BABE;
        // Reverse nibbles for 0..15:
        //   k=0..7  -> 15,14,13,12,11,10,9,8 => 0x89AB_CDEF
        //   k=8..15 -> 7,6,5,4,3,2,1,0       => 0x0123_4567
        init.sgpr[4] = 0x89AB_CDEF;
        init.sgpr[5] = 0x0123_4567;

        let (out, _) = runner
            .run_rdna2(a.words(), wave_size, &init, &[])
            .expect("GPU cross-lane dispatch");

        assert_eq!(out.sgpr[0], 0x1003, "v_readfirstlane_b32 picks lane 3");
        assert_eq!(out.sgpr[1], 0x1001, "v_readlane_b32 reads inactive lane 1");
        assert_eq!(
            out.vgpr[1][1], 0xCAFE_BABE,
            "v_writelane_b32 writes inactive lane 1"
        );
        assert_eq!(
            out.vgpr[0][1], 0xAAAA_0000,
            "v_writelane_b32 preserves other lanes"
        );

        for (lane, bank) in out.vgpr.iter().enumerate().take(lanes) {
            assert_eq!(
                bank[2],
                0x1000 + (lane as u32 ^ 3),
                "ds_swizzle quad lane {lane}"
            );
            assert_eq!(
                bank[3],
                0x1000 + (lane as u32 ^ 1),
                "ds_swizzle bitmode lane {lane}"
            );
            assert_eq!(
                bank[4],
                0x1000 + (lane as u32 ^ 15),
                "v_permlane16 lane {lane}"
            );
            assert_eq!(
                bank[5],
                0x1000 + ((lane as u32 ^ 16) ^ 15),
                "v_permlanex16 lane {lane}"
            );
            assert_eq!(
                bank[6],
                0x1000 + (lane as u32 ^ 1),
                "DPP16 quad lane {lane}"
            );

            let expected_shl = if (lane & 15) == 15 {
                0
            } else {
                0x1000 + (lane as u32) + 1
            };
            assert_eq!(bank[7], expected_shl, "DPP16 row_shl:1 lane {lane}");

            let expected_dpp8 = 0x1000 + ((lane & !7) + ((lane + 1) & 7)) as u32;
            assert_eq!(bank[8], expected_dpp8, "DPP8 rotate lane {lane}");
        }
    }
}

#[test]
fn gpu_memory_smem_mubuf_global_and_vsharp_oob_clamping() {
    let runner = shared_runner();

    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let mut a = asm::Asm::new();

        // 1. SMEM s_load_dwordx4 from byte 16 (words 4..8) into s[8..12]
        a.push2(asm::s_load_dwordx4(8, 0, 0x7D, 0));
        // 2. SMEM s_buffer_load_dwordx2 from V# s[4..8] (base=32, num_records=32) + ioffset=8 -> words 10, 11 into s[12..14]
        a.push2(asm::s_buffer_load_dwordx2(12, 4, 0x7D, 8));
        // 3. MUBUF buffer_load_dwordx4 v[4..8], v0, s[4..8], soffset=0 offen (tests in-bounds, partial OOB, full OOB)
        a.push2(asm::buffer_load_dwordx4(
            4,
            0,
            4,
            asm::inline_i(0),
            0,
            true,
            false,
        ));
        // 4. MUBUF buffer_store_dword v8, v1, s[4..8], soffset=0 offen (lane 0 writes word 8, lane 1 OOB dropped!)
        a.push2(asm::buffer_store_dword(
            8,
            1,
            4,
            asm::inline_i(0),
            0,
            true,
            false,
        ));
        // 5. GLOBAL load & store via s[16..18] (base=128 = word 32)
        //    global_load_dword v10, v2, s16, offset=+4 (lane 0 reads word 33, lane 1 OOB byte 512 -> 0)
        a.push2(asm::global_load_dword(10, 2, 16, 4));
        //    global_store_dword v3, v11, s16, offset=0 (lane 0 writes word 32, lane 1 OOB byte 512 -> dropped)
        a.push2(asm::global_store_dword(3, 11, 16, 0));
        a.push(asm::s_endpgm());

        // Seed 256-byte (64-word) test MemoryBuffer
        let mut mem = Vec::with_capacity(256);
        for i in 0u32..64 {
            mem.extend_from_slice(&(0xA000_0000 + i).to_le_bytes());
        }

        let mut init = wave_state(wave_size);
        // Only activate lanes 0, 1, 2 so other lanes don't overwrite test addresses
        init.exec = [0b111, 0];
        // s[0:1] = 64-bit byte address 16 (word 4)
        init.sgpr[0] = 16;
        init.sgpr[1] = 0;
        // s[4:7] = raw V# at base=32 (word 8), stride=0, num_records=32 bytes (words 8..16), oob_select=3
        init.set_vsharp(4, asm::make_vsharp(32, 0, 32, 3, false));
        // s[16:17] = 64-bit byte address 128 (word 32)
        init.sgpr[16] = 128;
        init.sgpr[17] = 0;

        // MUBUF load offsets (v0): lane 0 = 0 (all 4 in bounds), lane 1 = 24 (2 in bounds + 2 OOB), lane 2 = 64 (all 4 OOB)
        init.vgpr[0][0] = 0;
        init.vgpr[1][0] = 24;
        init.vgpr[2][0] = 64;

        // MUBUF store offsets (v1) & values (v8): lane 0 writes word 8, lanes 1 & 2 are OOB (offset 32 & 64)
        init.vgpr[0][1] = 0;
        init.vgpr[0][8] = 0x5555_0001;
        init.vgpr[1][1] = 32; // OOB of num_records=32 (would hit word 16 if unclamped!)
        init.vgpr[1][8] = 0xBAD0_BAD0;
        init.vgpr[2][1] = 64; // OOB
        init.vgpr[2][8] = 0xBAD1_BAD1;

        // GLOBAL load offsets (v2): lane 0 = 0 (+4 => byte 132 = word 33), lane 1 = 512 (OOB => 0), lane 2 = 8 (+4 => word 35)
        init.vgpr[0][2] = 0;
        init.vgpr[1][2] = 512;
        init.vgpr[2][2] = 8;

        // GLOBAL store offsets (v3) & values (v11): lane 0 writes word 32, lane 1 OOB, lane 2 writes word 34
        init.vgpr[0][3] = 0;
        init.vgpr[0][11] = 0x7777_8888;
        init.vgpr[1][3] = 512; // OOB beyond 256-byte MemoryBuffer
        init.vgpr[1][11] = 0xDEAD_DEAD;
        init.vgpr[2][3] = 8;
        init.vgpr[2][11] = 0x9999_AAAA;

        let (out, out_mem) = runner
            .run_rdna2(a.words(), wave_size, &init, &mem)
            .expect("GPU memory dispatch");

        // Check SMEM s_load_dwordx4
        assert_eq!(out.sgpr[8], 0xA000_0004);
        assert_eq!(out.sgpr[9], 0xA000_0005);
        assert_eq!(out.sgpr[10], 0xA000_0006);
        assert_eq!(out.sgpr[11], 0xA000_0007);

        // Check SMEM s_buffer_load_dwordx2
        assert_eq!(out.sgpr[12], 0xA000_000A);
        assert_eq!(out.sgpr[13], 0xA000_000B);

        // Check MUBUF buffer_load_dwordx4 (in-bounds, partial OOB, full OOB)
        assert_eq!(
            [
                out.vgpr[0][4],
                out.vgpr[0][5],
                out.vgpr[0][6],
                out.vgpr[0][7]
            ],
            [0xA000_0008, 0xA000_0009, 0xA000_000A, 0xA000_000B],
            "lane 0 in-bounds buffer_load_dwordx4"
        );
        assert_eq!(
            [
                out.vgpr[1][4],
                out.vgpr[1][5],
                out.vgpr[1][6],
                out.vgpr[1][7]
            ],
            [0xA000_000E, 0xA000_000F, 0, 0],
            "lane 1 partial OOB buffer_load_dwordx4 clamps trailing words to 0"
        );
        assert_eq!(
            [
                out.vgpr[2][4],
                out.vgpr[2][5],
                out.vgpr[2][6],
                out.vgpr[2][7]
            ],
            [0, 0, 0, 0],
            "lane 2 full OOB buffer_load_dwordx4 returns zeros"
        );

        // Check GLOBAL load
        assert_eq!(out.vgpr[0][10], 0xA000_0021, "global_load word 33");
        assert_eq!(out.vgpr[1][10], 0, "global_load OOB returns 0");
        assert_eq!(out.vgpr[2][10], 0xA000_0023, "global_load word 35");

        // Check MemoryBuffer mutations & OOB write drop
        let read_word = |idx: usize| -> u32 {
            u32::from_le_bytes(out_mem[idx * 4..idx * 4 + 4].try_into().unwrap())
        };
        assert_eq!(
            read_word(8),
            0x5555_0001,
            "in-bounds MUBUF store wrote word 8"
        );
        assert_eq!(
            read_word(16),
            0xA000_0010,
            "OOB MUBUF store at offset 32 MUST NOT clobber word 16"
        );
        assert_eq!(
            read_word(32),
            0x7777_8888,
            "in-bounds GLOBAL store wrote word 32"
        );
        assert_eq!(
            read_word(34),
            0x9999_AAAA,
            "in-bounds GLOBAL store wrote word 34"
        );
    }
}

#[test]
fn gpu_subgroup_and_workgroup_lowering_produce_bit_identical_results() {
    let runner = shared_runner();
    let info = runner.device_info();
    println!("[gpu-tests] Parity check on: {info}");

    let mut a = asm::Asm::new();
    a.push(asm::v_cmp_lt_u32(asm::v(0), 1));
    a.push(asm::v_readfirstlane_b32(0, 0));
    a.push2(asm::ds_swizzle_b32(2, 0, 0x801B));
    a.push(asm::s_endpgm());

    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let mut init = wave_state(wave_size);
        let lanes = wave_size.width() as usize;
        init.fill_vgpr(0, lanes, |l| l as u32);
        init.fill_vgpr(1, lanes, |_| 13);

        let wg_opts = CompileOptions {
            wave_size,
            cross_lane_mode: CrossLaneMode::WorkgroupShared,
            strict_float_controls: info.shader_rounding_mode_rte_f32
                && info.shader_signed_zero_inf_nan_preserve_f32,
            denorm_preserve: info.shader_denorm_preserve_f32,
        };
        let (out_wg, _) = runner
            .run_rdna2_with_options(a.words(), wg_opts, &init, &[])
            .expect("WorkgroupShared execution");

        if info.supports_required_subgroup_size(wave_size.width()) {
            let sg_opts = CompileOptions {
                cross_lane_mode: CrossLaneMode::Subgroup,
                ..wg_opts
            };
            let (out_sg, _) = runner
                .run_rdna2_with_options(a.words(), sg_opts, &init, &[])
                .expect("Subgroup execution");
            assert_eq!(
                out_wg, out_sg,
                "WorkgroupShared and native Subgroup execution must be bit-identical for {wave_size:?}"
            );
            println!(
                "[gpu-tests] Verified native Subgroup == WorkgroupShared parity for {wave_size:?}"
            );
        } else {
            println!(
                "[gpu-tests] Device subgroup range [{}..{}] does not pin full {wave_size:?}; verified WorkgroupShared path",
                info.min_subgroup_size, info.max_subgroup_size
            );
        }
    }
}

#[test]
fn gpu_gfx103_waitcnt_f16_and_bit_ops_are_bit_exact() {
    let runner = shared_runner();
    let mut a = asm::Asm::new();
    // Scheduling / waitcnt hints must be clean no-ops
    a.push(asm::s_inst_prefetch(3))
        .push(asm::s_clause(1))
        .push(asm::s_waitcnt_depctr(0))
        .push(asm::s_waitcnt_vscnt(asm::NULL, 0))
        .push(asm::s_waitcnt_vmcnt(asm::NULL, 0))
        .push(asm::s_waitcnt_expcnt(asm::NULL, 0))
        .push(asm::s_waitcnt_lgkmcnt(asm::NULL, 0))
        // v10 = v_cvt_pkrtz_f16_f32(v0, v1)
        .push(asm::v_cvt_pkrtz_f16_f32(10, asm::v(0), 1))
        // v11 = v_cvt_f16_f32(v2) preserving high 16 bits of v11
        .push(asm::v_cvt_f16_f32(11, asm::v(2)))
        // v12 = v_cvt_f32_f16(v10) (unpack low half of v10)
        .push(asm::v_cvt_f32_f16(12, asm::v(10)))
        // v13 = v_cvt_f32_f16(0.5) (inline binary16 constant selector)
        .push(asm::v_cvt_f32_f16(13, asm::F_0_5))
        // v14 = v_lshl_add_u32(v3, v4, v5)
        .push2(asm::v_lshl_add_u32(14, asm::v(3), asm::v(4), asm::v(5)))
        // v15 = v_add3_u32(v3, v4, v5)
        .push2(asm::v_add3_u32(15, asm::v(3), asm::v(4), asm::v(5)))
        // v16 = v_lshl_or_b32(v3, v4, v5)
        .push2(asm::v_lshl_or_b32(16, asm::v(3), asm::v(4), asm::v(5)))
        // v17 = v_and_or_b32(v3, v4, v5)
        .push2(asm::v_and_or_b32(17, asm::v(3), asm::v(4), asm::v(5)))
        // v18 = v_bfe_u32(v6, v7, v8)
        .push2(asm::v_bfe_u32(18, asm::v(6), asm::v(7), asm::v(8)))
        // v19 = v_bfe_i32(v6, v7, v8)
        .push2(asm::v_bfe_i32(19, asm::v(6), asm::v(7), asm::v(8)))
        // v20 = v_bfi_b32(v3, v4, v5)
        .push2(asm::v_bfi_b32(20, asm::v(3), asm::v(4), asm::v(5)))
        // v21 = v_mul_u32_u24(v3, v4)
        .push(asm::v_mul_u32_u24(21, asm::v(3), 4))
        // v22 = v_mad_u32_u24(v3, v4, v5)
        .push2(asm::v_mad_u32_u24(22, asm::v(3), asm::v(4), asm::v(5)))
        // v23 = v_lshlrev_b32(v4, v3)
        .push(asm::v_lshlrev_b32(23, asm::v(4), 3))
        // v24 = v_lshrrev_b32(v4, v3)
        .push(asm::v_lshrrev_b32(24, asm::v(4), 3))
        // v25 = v_ashrrev_i32(v4, v6)
        .push(asm::v_ashrrev_i32(25, asm::v(4), 6))
        .push(asm::s_endpgm());

    let mut init = wave_state(WaveSize::Wave32);
    // Lane 0: RTZ test (1.0009764 -> 0x3C00 not 0x3C01, 65520.0 -> 0x7BFF not +Inf)
    init.vgpr[0][0] = 0x3F80_1FFF;
    init.vgpr[0][1] = 65520.0_f32.to_bits();
    init.vgpr[0][2] = 1.5_f32.to_bits();
    init.vgpr[0][11] = 0xCAFE_0000;
    // Lane 1: subnormal 2^-24 -> 0x0001, -0.0 -> 0x8000
    init.vgpr[1][0] = f32::from_bits(0x3380_0000).to_bits();
    init.vgpr[1][1] = (-0.0_f32).to_bits();
    init.vgpr[1][2] = (-2.0_f32).to_bits();
    init.vgpr[1][11] = 0x1234_FFFF;
    // Lane 2: +Inf -> 0x7C00, 100000.0 -> 0x7BFF
    init.vgpr[2][0] = f32::INFINITY.to_bits();
    init.vgpr[2][1] = 100000.0_f32.to_bits();
    init.vgpr[2][2] = 0.0_f32.to_bits();
    init.vgpr[2][11] = 0xBEEF_0000;

    // Bit & 24-bit integer lanes
    // Lane 0: normal BFE & shift/add/or/and/u24 (top 8 bits of v3/v4 must be masked off by u24!)
    init.vgpr[0][3] = 0xFF00_0015; // low 24 = 21
    init.vgpr[0][4] = 0xAA00_0004; // low 24 = 4, shift = 4
    init.vgpr[0][5] = 0x0000_0003;
    init.vgpr[0][6] = 0xF080_AB12;
    init.vgpr[0][7] = 8; // offset
    init.vgpr[0][8] = 8; // count -> 0xAB (unsigned 0xAB, signed -85 = 0xFFFF_FFAB)

    // Lane 1: BFE crossing bit 31 (offset=28, count=12 -> clamped to 4 bits of 0xF080_AB12 = 0xF / -1)
    init.vgpr[1][3] = 0x0F0F_0F0F;
    init.vgpr[1][4] = 3;
    init.vgpr[1][5] = 0x1010_1010;
    init.vgpr[1][6] = 0xF080_AB12;
    init.vgpr[1][7] = 28;
    init.vgpr[1][8] = 12;

    // Lane 2: BFE with count == 0 must return 0
    init.vgpr[2][6] = 0xFFFF_FFFF;
    init.vgpr[2][7] = 5;
    init.vgpr[2][8] = 0;

    let (out, _) = runner
        .run_rdna2(a.words(), WaveSize::Wave32, &init, &[])
        .expect("GFX10.3 instruction execution");

    assert_eq!(
        out.vgpr[0][10], 0x7BFF_3C00,
        "v_cvt_pkrtz_f16_f32 must RTZ 1.0009764->0x3C00 and 65520->0x7BFF"
    );
    assert_eq!(out.vgpr[1][10], 0x8000_0001, "subnormal 2^-24 and -0.0");
    assert_eq!(out.vgpr[2][10], 0x7BFF_7C00, "+Inf and overflow RTZ");

    assert_eq!(
        out.vgpr[0][11], 0xCAFE_3E00,
        "v_cvt_f16_f32 must preserve upper 16 bits of vdst"
    );
    assert_eq!(out.vgpr[1][11], 0x1234_C000);
    assert_eq!(out.vgpr[0][12], 1.0_f32.to_bits());
    assert_eq!(out.vgpr[0][13], 0.5_f32.to_bits());

    let s3 = init.vgpr[0][3];
    let s4 = init.vgpr[0][4];
    let s5 = init.vgpr[0][5];
    assert_eq!(out.vgpr[0][14], (s3 << (s4 & 31)).wrapping_add(s5));
    assert_eq!(out.vgpr[0][15], s3.wrapping_add(s4).wrapping_add(s5));
    assert_eq!(out.vgpr[0][16], (s3 << (s4 & 31)) | s5);
    assert_eq!(out.vgpr[0][17], (s3 & s4) | s5);
    assert_eq!(out.vgpr[0][18], 0x0000_00AB);
    assert_eq!(out.vgpr[0][19], (-85i32) as u32);
    assert_eq!(out.vgpr[0][20], (s3 & s4) | (!s3 & s5));
    assert_eq!(out.vgpr[0][21], 21 * 4);
    assert_eq!(out.vgpr[0][22], 21 * 4 + 3);
    assert_eq!(out.vgpr[0][23], s3 << (s4 & 31));
    assert_eq!(out.vgpr[0][24], s3 >> (s4 & 31));
    assert_eq!(
        out.vgpr[0][25],
        ((init.vgpr[0][6] as i32) >> (s4 & 31)) as u32
    );

    assert_eq!(
        out.vgpr[1][18], 0x0000_000F,
        "v_bfe_u32 crossing bit 31 must extract remaining top 4 bits cleanly"
    );
    assert_eq!(
        out.vgpr[1][19], 0xFFFF_FFFF,
        "v_bfe_i32 crossing bit 31 must sign-extend top nibble 0xF to -1"
    );
    assert_eq!(out.vgpr[2][18], 0, "v_bfe_u32 count=0 -> 0");
    assert_eq!(out.vgpr[2][19], 0, "v_bfe_i32 count=0 -> 0");
}

#[test]
fn gpu_extended_float_alu_rounds_reduces_turns_and_handles_nan() {
    let runner = shared_runner();
    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let mut shader = asm::Asm::new();
        for (reg, op) in [
            (10, asm::v_fract_f32 as fn(u8, u16) -> u32),
            (11, asm::v_trunc_f32),
            (12, asm::v_ceil_f32),
            (13, asm::v_rndne_f32),
            (14, asm::v_floor_f32),
        ] {
            shader.push(op(reg, asm::v(0)));
        }
        shader.push(asm::v_exp_f32(15, asm::v(1)));
        shader.push(asm::v_log_f32(16, asm::v(2)));
        shader.push(asm::v_sin_f32(17, asm::v(3)));
        shader.push(asm::v_cos_f32(18, asm::v(3)));
        shader.push2(asm::v_min3_f32(19, asm::v(4), asm::v(5), asm::v(6)));
        shader.push2(asm::v_max3_f32(20, asm::v(4), asm::v(5), asm::v(6)));
        shader.push2(asm::v_med3_f32(21, asm::v(4), asm::v(5), asm::v(6)));
        shader.push2(asm::vop3(0x1A4, 22, asm::v(0), 0, 0, 1, 0, 1, false));
        shader.push(asm::s_endpgm());
        let mut init = wave_state(wave_size);
        for (lane, values) in [
            [-1.5, 3.0, 8.0, 2.25, 2.0, 4.0, -1.0],
            [2.5, -2.0, 0.0, -0.0, f32::NAN, 4.0, 2.0],
            [-3.25, 0.0, -1.0, 33_554_432.0, 4.0, 2.0, f32::NAN],
        ]
        .into_iter()
        .enumerate()
        {
            for (dst, value) in init.vgpr[lane].iter_mut().zip(values) {
                *dst = value.to_bits();
            }
        }
        let (out, _) = runner
            .run_rdna2(shader.words(), wave_size, &init, &[])
            .unwrap();
        let bits = |values: [f32; 13]| values.map(f32::to_bits);
        assert_eq!(
            out.vgpr[0][10..23],
            bits([
                0.5, -1.0, -1.0, -2.0, -2.0, 8.0, 3.0, 1.0, 0.0, -1.0, 4.0, 2.0, 2.0,
            ])
        );
        assert_eq!(
            out.vgpr[1][10..23],
            bits([
                0.5,
                2.0,
                3.0,
                2.0,
                2.0,
                0.25,
                f32::NEG_INFINITY,
                -0.0,
                1.0,
                2.0,
                4.0,
                2.0,
                4.0,
            ])
        );
        assert_eq!(out.vgpr[2][16], 0xffc0_0000);
        assert_eq!(
            out.vgpr[2][17..22],
            [0.0_f32, 1.0, 2.0, 4.0, 2.0].map(f32::to_bits)
        );
    }
}

#[test]
fn gpu_wave_counts_carry_and_compare_exec_masks() {
    let runner = shared_runner();
    for wave_size in [WaveSize::Wave32, WaveSize::Wave64] {
        let mut shader = asm::Asm::new();
        shader.push2(asm::v_mbcnt_lo_u32_b32(
            4,
            asm::inline_i(-1),
            asm::inline_i(0),
        ));
        shader.push2(asm::v_mbcnt_hi_u32_b32(4, asm::inline_i(-1), asm::v(4)));
        shader.push2(asm::v_add_co_u32(5, 8, asm::v(0), asm::v(1)));
        shader.push2(asm::v_add_co_ci_u32_e64(
            6,
            10,
            asm::v(1),
            asm::v(2),
            asm::s(8),
        ));
        shader.push(asm::v_add_co_ci_u32(7, asm::v(1), 2));
        shader.push(asm::v_mul_hi_u32_u24(8, asm::v(0), 0));
        shader.push2(asm::v_add_lshl_u32(
            9,
            asm::v(1),
            asm::v(1),
            asm::inline_i(34),
        ));
        shader.push(asm::v_bfrev_b32(10, asm::v(1)));
        shader.push(asm::s_ff1_i32_b64(12, 2));
        shader.push(asm::v_cmpx_eq_i32(asm::inline_i(1), 3));
        shader.push(asm::v_mov_b32(11, asm::inline_i(7)));
        shader.push(asm::s_endpgm());

        let mut init = wave_state(wave_size);
        init.vcc = [0xAAAA_AAAA; 2];
        init.sgpr[3] = 1 << 8;
        for (lane, v) in init.vgpr.iter_mut().enumerate() {
            v[0] = if lane % 2 == 0 { u32::MAX } else { 0 };
            v[1] = 1;
            v[2] = u32::MAX - 1;
            v[3] = u32::from(lane % 2 == 1);
            v[11] = u32::MAX;
        }
        let (out, _) = runner
            .run_rdna2(shader.words(), wave_size, &init, &[])
            .unwrap();
        let active_hi = if wave_size == WaveSize::Wave64 {
            0xAAAA_AAAA
        } else {
            0
        };
        assert_eq!(out.exec, [0xAAAA_AAAA, active_hi]);
        assert_eq!(out.vcc, [0xAAAA_AAAA; 2]);
        assert_eq!(out.sgpr[8], 0x5555_5555);
        assert_eq!(out.sgpr[10], 0x5555_5555);
        assert_eq!(out.sgpr[12], 40);
        for lane in 0..wave_size.width() {
            let v = &out.vgpr[lane as usize];
            let odd = lane % 2;
            assert_eq!(
                v[4..12],
                [
                    lane,
                    odd,
                    if odd == 0 { 0 } else { u32::MAX },
                    if odd == 1 { 0 } else { u32::MAX },
                    if odd == 0 { 0xFFFF } else { 0 },
                    8,
                    0x8000_0000,
                    if odd == 1 { 7 } else { u32::MAX },
                ],
                "lane {lane}"
            );
        }
    }
}

#[test]
fn gpu_atomic_adds_return_unique_values_and_preserve_inactive_lanes() {
    let runner = shared_runner();
    let mut shader = asm::Asm::new();
    shader.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(1)));
    shader.push2(asm::ds_write_b32(0, 0, 0));
    shader.push(asm::s_mov_b32(asm::EXEC_LO, asm::inline_i(15)));
    shader.push(asm::s_barrier());
    shader.push2(asm::ds_add_u32(0, 1, 0));
    shader.push(asm::s_barrier());
    shader.push2(asm::ds_add_rtn_u32(4, 0, 1, 0));
    shader.push(asm::s_barrier());
    shader.push2(asm::ds_read_b32(5, 0, 0));
    shader.push2(asm::buffer_atomic_add(
        1,
        0,
        4,
        asm::inline_i(0),
        0,
        true,
        false,
        true,
    ));
    shader.push2(asm::global_atomic_add(8, 2, 0x7D, 4, Some(6)));
    shader.push2(asm::global_atomic_add(8, 2, 0x7D, 8, None));
    shader.push(asm::s_endpgm());

    let mut init = WaveState::new_wave32();
    init.set_vsharp(4, asm::make_vsharp(0, 0, 12, 3, false));
    for v in &mut init.vgpr {
        v[1] = 1;
        v[2] = 1;
        v[4] = u32::MAX;
        v[6] = u32::MAX;
    }
    let mut mem = [0; 12];
    mem[..4].copy_from_slice(&10_u32.to_le_bytes());
    mem[4..8].copy_from_slice(&20_u32.to_le_bytes());
    let (out, mem) = runner
        .run_rdna2(shader.words(), WaveSize::Wave32, &init, &mem)
        .unwrap();
    for (reg, expected) in [
        (4, [4, 5, 6, 7]),
        (1, [10, 11, 12, 13]),
        (6, [20, 21, 22, 23]),
    ] {
        let mut previous = [0; 4];
        for (dst, lane) in previous.iter_mut().zip(&out.vgpr) {
            *dst = lane[reg];
            assert_eq!(lane[5], 8);
        }
        previous.sort_unstable();
        assert_eq!(previous, expected);
    }
    assert_eq!(out.vgpr[4][4], u32::MAX);
    assert_eq!(out.vgpr[4][6], u32::MAX);
    let (word_bytes, tail) = mem.as_chunks::<4>();
    assert!(tail.is_empty());
    let words: Vec<_> = word_bytes.iter().copied().map(u32::from_le_bytes).collect();
    assert_eq!(words, [14, 24, 4]);
}

#[test]
fn gpu_halfword_writes_keep_the_other_half_and_scalar_loads_copy_eight_words() {
    let runner = shared_runner();
    let mut shader = asm::Asm::new();
    shader.push2(asm::s_load_dwordx8(8, 0, 0x7D, 0));
    shader.push2(asm::s_buffer_load_dwordx8(16, 4, 0x7D, 0));
    shader.push2(asm::buffer_load_ushort(
        2,
        0,
        4,
        asm::inline_i(0),
        0,
        true,
        false,
    ));
    shader.push_lit(asm::v_mul_f16(3, asm::SRC_LIT, 2), 0x3e00);
    shader.push2(asm::buffer_store_short(
        3,
        1,
        4,
        asm::inline_i(0),
        0,
        true,
        false,
    ));
    shader.push(asm::s_endpgm());
    let mut init = WaveState::new_wave32();
    init.exec = [0b111, 0];
    init.set_vsharp(4, asm::make_vsharp(0, 0, 32, 3, false));
    for (lane, (src, dst)) in [(0, 4), (2, 6), (32, 32)].into_iter().enumerate() {
        init.vgpr[lane][0] = src;
        init.vgpr[lane][1] = dst;
        init.vgpr[lane][3] = 0x4321_ffff;
    }
    let words = [0x4000_3c00_u32, 0xdead_beef, 2, 3, 4, 5, 6, 7];
    let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
    let (out, mem) = runner
        .run_rdna2(shader.words(), WaveSize::Wave32, &init, &bytes)
        .unwrap();
    assert_eq!(out.sgpr[8..16], words);
    assert_eq!(out.sgpr[16..24], words);
    assert_eq!(out.vgpr[0][3], 0x4321_3e00);
    assert_eq!(out.vgpr[1][3], 0x4321_4200);
    assert_eq!(out.vgpr[2][2], 0);
    let expected = [0x4200_3e00_u32, 2, 3, 4, 5, 6, 7];
    assert_eq!(
        mem[4..],
        expected
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}
