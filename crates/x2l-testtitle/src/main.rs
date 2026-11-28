//! Clean-room synthetic console PE32+ end-to-end test title (`x86_64-pc-windows-gnu`).
//!
//! Verifies all M1 & M2.4 compatibility invariants under `x2linux run`:
//! 1. `kernelx!GetConsoleType` & `TitleMemoryStatus`
//! 2. `kernelx!XMemAlloc` + memory read/write + `XMemFree`
//! 3. `xgameruntime!XGameRuntimeInitialize`, offline `XStoreCreateContext`, `XGameRuntimeUninitialize`
//! 4. `d3d12_xs!D3D12CreateDevice` + spare vtable probe slot 45 + `Release` (and when
//!    `--render-vulkan`, `X2L_TEST_RENDER=1`, or `X2L_GPU_BACKEND=vulkan` is active: full
//!    64x64 Vulkan 1.3 triangle draw via `ID3D12DeviceX`, `ID3D12CommandQueueX` + `PresentX`,
//!    `ID3D12CommandAllocator`, `ID3D12GraphicsCommandListX`, `ID3D12Fence`, `ID3D12Resource`,
//!    `ID3D12DescriptorHeap`, and `ID3D12PipelineState` with embedded synthetic RDNA2 VS+PS blobs!)
//! 5. Unknown ordinal `kernelx.dll!#999` auto-stub call without crashing (telemetry in `report.json`)
//! 6. Multi-threaded static TLS (`thread_local!`) + panic/unwind handling
//! 7. Copy-on-write OverlayFS reads, overwrite, create, and whiteout delete inside `G:\`

use std::cell::Cell;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::thread;

thread_local! {
    static SYNTHETIC_TLS_MARKER: Cell<u64> = const { Cell::new(0x5832_4C5F_544C_5331) };
}

#[cfg(target_os = "windows")]
mod console_imports {
    use std::ffi::c_void;
    use std::mem::size_of;

    #[repr(C)]
    #[derive(Default)]
    pub struct TitleMemoryStatus {
        pub dw_length: u32,
        pub dw_reserved: u32,
        pub ull_total_mem: u64,
        pub ull_avail_mem: u64,
        pub ull_legacy_used: u64,
        pub ull_legacy_peak: u64,
        pub ull_legacy_avail: u64,
        pub ull_title_used: u64,
        pub ull_title_avail: u64,
        pub ull_legacy_page_faults: u64,
        pub ull_title_page_faults: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12CommandQueueDesc {
        pub r#type: u32,
        pub priority: i32,
        pub flags: u32,
        pub node_mask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12HeapProperties {
        pub r#type: u32,
        pub cpu_page_property: u32,
        pub memory_pool_preference: u32,
        pub creation_node_mask: u32,
        pub visible_node_mask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct DxgiSampleDesc {
        pub count: u32,
        pub quality: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12ResourceDesc {
        pub dimension: u32,
        pub alignment: u64,
        pub width: u64,
        pub height: u32,
        pub depth_or_array_size: u16,
        pub mip_levels: u16,
        pub format: u32,
        pub sample_desc: DxgiSampleDesc,
        pub layout: u32,
        pub flags: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12DescriptorHeapDesc {
        pub r#type: u32,
        pub num_descriptors: u32,
        pub flags: u32,
        pub node_mask: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12CpuDescriptorHandle {
        pub ptr: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12ShaderBytecode {
        pub p_shader_bytecode: *const c_void,
        pub bytecode_length: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12Viewport {
        pub top_left_x: f32,
        pub top_left_y: f32,
        pub width: f32,
        pub height: f32,
        pub min_depth: f32,
        pub max_depth: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12Rect {
        pub left: i32,
        pub top: i32,
        pub right: i32,
        pub bottom: i32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12ResourceTransitionBarrier {
        pub p_resource: *mut c_void,
        pub subresource: u32,
        pub state_before: u32,
        pub state_after: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub union D3D12ResourceBarrierUnion {
        pub transition: D3D12ResourceTransitionBarrier,
        pub raw: [u64; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12ResourceBarrier {
        pub r#type: u32,
        pub flags: u32,
        pub u: D3D12ResourceBarrierUnion,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12StreamOutputDesc {
        pub p_so_declaration: *const c_void,
        pub num_entries: u32,
        pub p_buffer_strides: *const u32,
        pub num_strides: u32,
        pub rasterized_stream: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12InputLayoutDesc {
        pub p_input_element_descs: *const c_void,
        pub num_elements: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12CachedPipelineState {
        pub p_cached_blob: *const c_void,
        pub cached_blob_size_in_bytes: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12GraphicsPipelineStateDesc {
        pub p_root_signature: *mut c_void,
        pub vs: D3D12ShaderBytecode,
        pub ps: D3D12ShaderBytecode,
        pub ds: D3D12ShaderBytecode,
        pub hs: D3D12ShaderBytecode,
        pub gs: D3D12ShaderBytecode,
        pub stream_output: D3D12StreamOutputDesc,
        pub blend_state: [u32; 82],
        pub sample_mask: u32,
        pub rasterizer_state: [u32; 11],
        pub depth_stencil_state: [u32; 13],
        pub input_layout: D3D12InputLayoutDesc,
        pub ib_strip_cut_value: u32,
        pub primitive_topology_type: u32,
        pub num_render_targets: u32,
        pub rtv_formats: [u32; 8],
        pub dsv_format: u32,
        pub sample_desc: DxgiSampleDesc,
        pub node_mask: u32,
        pub cached_pso: D3D12CachedPipelineState,
        pub flags: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12XboxPresentPlaneParameters {
        pub token: u64,
        pub resource_count: u32,
        pub _pad0: u32,
        pub pp_resources: *const *mut c_void,
        pub _reserved: [u64; 6],
    }

    // Compile-time ABI checks matching Win64 DirectX-Headers & x2l-gpu-vk::com.
    const _: () = assert!(size_of::<D3D12CommandQueueDesc>() == 16);
    const _: () = assert!(size_of::<D3D12HeapProperties>() == 20);
    const _: () = assert!(size_of::<D3D12ResourceDesc>() == 56);
    const _: () = assert!(size_of::<D3D12DescriptorHeapDesc>() == 16);
    const _: () = assert!(size_of::<D3D12CpuDescriptorHandle>() == 8);
    const _: () = assert!(size_of::<D3D12GraphicsPipelineStateDesc>() == 656);
    const _: () = assert!(size_of::<D3D12XboxPresentPlaneParameters>() == 72);

    #[link(name = "kernelx", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void;
        pub fn XMemFree(ptr: *mut c_void, attrs: u64);
        pub fn GetConsoleType() -> u32;
        pub fn TitleMemoryStatus(out: *mut TitleMemoryStatus) -> i32;
        #[link_ordinal(999)]
        pub fn x2l_synthetic_ordinal_probe() -> u64;
    }

    #[link(name = "xgameruntime", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XGameRuntimeInitialize() -> i32;
        pub fn XGameRuntimeUninitialize();
        pub fn XStoreCreateContext(user: *mut c_void, out_ctx: *mut *mut c_void) -> i32;
    }

    #[link(name = "d3d12_xs", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn D3D12CreateDevice(
            adapter: *mut c_void,
            min_feature_level: u32,
            riid: *const c_void,
            out_device: *mut *mut c_void,
        ) -> i32;
    }

    // Minimal clean-room GFX10.3 RDNA2 encoders for synthetic triangle VS & PS blobs.
    const F_0_5: u16 = 240;
    const F_NEG_0_5: u16 = 241;
    const F_1_0: u16 = 242;
    const INLINE_0: u16 = 128;
    const INLINE_1: u16 = 129;

    const fn v_reg(n: u8) -> u16 {
        256 + n as u16
    }

    const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
        (0x3Fu32 << 25) | ((vdst as u32) << 17) | (0x01u32 << 9) | ((src0 as u32) & 0x1FF)
    }

    const fn v_cndmask_b32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (0x01u32 << 25) | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | ((src0 as u32) & 0x1FF)
    }

    const fn v_cmp_eq_u32(src0: u16, vsrc1: u8) -> u32 {
        (0x3Eu32 << 25) | (0xC2u32 << 17) | ((vsrc1 as u32) << 9) | ((src0 as u32) & 0x1FF)
    }

    const fn exp(target: u8, en: u8, compr: bool, done: bool, vm: bool, vsrc: [u8; 4]) -> [u32; 2] {
        let w0 = (0x3Eu32 << 26)
            | ((vm as u32) << 12)
            | ((done as u32) << 11)
            | ((compr as u32) << 10)
            | (((target as u32) & 0x3F) << 4)
            | ((en as u32) & 0xF);
        let w1 = (vsrc[0] as u32)
            | ((vsrc[1] as u32) << 8)
            | ((vsrc[2] as u32) << 16)
            | ((vsrc[3] as u32) << 24);
        [w0, w1]
    }

    const fn vintrp(op: u8, vdst: u8, attr: u8, chan: u8, vsrc: u8) -> u32 {
        (0x32u32 << 26)
            | ((vdst as u32) << 18)
            | (((op as u32) & 0x3) << 16)
            | (((attr as u32) & 0x3F) << 10)
            | (((chan as u32) & 0x3) << 8)
            | (vsrc as u32)
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// Synthetic RDNA2 procedural triangle Vertex Shader (`SV_VertexID` in `v0` -> `pos0` & yellow `param0`).
    pub fn synthetic_triangle_vs_bytes() -> Vec<u8> {
        let pos0 = exp(12, 0xF, false, false, false, [1, 2, 3, 4]);
        let param0 = exp(32, 0xF, false, true, false, [5, 6, 7, 8]);
        let words: &[u32] = &[
            v_mov_b32(1, F_NEG_0_5),
            v_mov_b32(2, F_NEG_0_5),
            v_mov_b32(3, INLINE_0),
            v_mov_b32(4, F_1_0),
            v_mov_b32(9, F_0_5),
            v_mov_b32(10, INLINE_0),
            v_cmp_eq_u32(INLINE_1, 0),
            v_cndmask_b32(1, v_reg(1), 9),
            v_cmp_eq_u32(INLINE_0, 0),
            v_cndmask_b32(1, v_reg(1), 10),
            v_cndmask_b32(2, v_reg(2), 9),
            v_mov_b32(5, F_1_0),
            v_mov_b32(6, F_1_0),
            v_mov_b32(7, INLINE_0),
            v_mov_b32(8, F_1_0),
            pos0[0],
            pos0[1],
            param0[0],
            param0[1],
            S_ENDPGM,
        ];
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Synthetic RDNA2 Pixel Shader reading interpolated `param0` via `VINTRP` and exporting `mrt0`.
    pub fn synthetic_triangle_ps_bytes() -> Vec<u8> {
        let mrt0 = exp(0, 0xF, false, true, false, [0, 1, 2, 3]);
        let words: &[u32] = &[
            vintrp(0, 0, 0, 0, 0),
            vintrp(1, 0, 0, 0, 1),
            vintrp(0, 1, 0, 1, 0),
            vintrp(1, 1, 0, 1, 1),
            vintrp(0, 2, 0, 2, 0),
            vintrp(1, 2, 0, 2, 1),
            vintrp(0, 3, 0, 3, 0),
            vintrp(1, 3, 0, 3, 1),
            mrt0[0],
            mrt0[1],
            S_ENDPGM,
        ];
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }
}

fn check(name: &str, ok: bool, failures: &mut u32) {
    if ok {
        println!("[PASS] {name}");
    } else {
        eprintln!("[FAIL] {name}");
        *failures += 1;
    }
}

fn should_render_vulkan() -> bool {
    if std::env::args().any(|a| a == "--render-vulkan") {
        return true;
    }
    if std::env::var("X2L_TEST_RENDER")
        .ok()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    {
        return true;
    }
    std::env::var("X2L_GPU_BACKEND")
        .ok()
        .is_some_and(|v| v.eq_ignore_ascii_case("vulkan"))
}

fn main() -> ExitCode {
    let mut failures = 0u32;

    #[cfg(target_os = "windows")]
    run_console_api_checks(&mut failures, should_render_vulkan());

    // 6. Multi-threaded static TLS + panic/unwind handling
    let tls_handles: Vec<_> = (0u64..4)
        .map(|tid| {
            thread::spawn(move || {
                SYNTHETIC_TLS_MARKER.with(|slot| {
                    let initial = slot.get();
                    slot.set(initial.wrapping_add(tid + 1));
                    (initial, slot.get())
                })
            })
        })
        .collect();
    let mut tls_ok = true;
    for (idx, handle) in tls_handles.into_iter().enumerate() {
        match handle.join() {
            Ok((initial, updated)) => {
                if initial != 0x5832_4C5F_544C_5331
                    || updated != 0x5832_4C5F_544C_5331u64.wrapping_add(idx as u64 + 1)
                {
                    tls_ok = false;
                }
            }
            Err(_) => tls_ok = false,
        }
    }
    check("multi-threaded static TLS isolation", tls_ok, &mut failures);

    let unwind_ok = std::panic::catch_unwind(|| {
        panic!("synthetic-unwind-probe");
    })
    .is_err();
    check("SEH / Rust panic unwind handling", unwind_ok, &mut failures);

    // 7. OverlayFS copy-on-write & whiteout checks (active when fixture files exist)
    run_overlay_fixture_checks(&mut failures);

    if failures == 0 {
        println!("[x2l-testtitle] ALL CHECKS PASSED");
        ExitCode::SUCCESS
    } else {
        eprintln!("[x2l-testtitle] {failures} CHECK(S) FAILED");
        ExitCode::from(1)
    }
}

#[cfg(target_os = "windows")]
fn run_console_api_checks(failures: &mut u32, render_vulkan: bool) {
    use console_imports::*;
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::ptr;

    // 1. GetConsoleType & TitleMemoryStatus
    // SAFETY: Exported by `kernelx.dll` PE shim.
    let console_type = unsafe { GetConsoleType() };
    check(
        &format!("kernelx!GetConsoleType returned 0x{console_type:04x}"),
        console_type == 0x0007 || console_type == 0x0006,
        failures,
    );

    let mut mem_status = TitleMemoryStatus::default();
    // SAFETY: `mem_status` is a valid writable `TITLEMEMORYSTATUS` struct.
    let status_ok = unsafe { TitleMemoryStatus(&mut mem_status) } != 0
        && mem_status.ull_total_mem >= 8_000_000_000;
    check(
        &format!(
            "kernelx!TitleMemoryStatus (total={} B)",
            mem_status.ull_total_mem
        ),
        status_ok,
        failures,
    );

    // 2. XMemAlloc + write/read + XMemFree
    // SAFETY: Valid 4096-byte allocation from `kernelx!XMemAlloc`.
    let xmem_ok = unsafe {
        let ptr = XMemAlloc(4096, 0x1000) as *mut u8;
        if ptr.is_null() {
            false
        } else {
            let slice = std::slice::from_raw_parts_mut(ptr, 4096);
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i & 0xFF) as u8;
            }
            let verified = slice
                .iter()
                .enumerate()
                .all(|(i, &b)| b == (i & 0xFF) as u8);
            XMemFree(ptr as *mut c_void, 0);
            verified
        }
    };
    check(
        "kernelx!XMemAlloc + write/read + XMemFree",
        xmem_ok,
        failures,
    );

    // 3. XGameRuntimeInitialize / offline XStore / XGameRuntimeUninitialize
    // SAFETY: Offline XGameRuntime shim calls.
    let xgr_init_hr = unsafe { XGameRuntimeInitialize() };
    check(
        &format!("xgameruntime!XGameRuntimeInitialize (hr=0x{xgr_init_hr:08x})"),
        xgr_init_hr == 0,
        failures,
    );

    let mut store_ctx: *mut c_void = ptr::null_mut();
    // SAFETY: Verifies XStore stays offline (E_NOTIMPL = 0x80004001) and never spoofs licenses.
    let store_hr = unsafe { XStoreCreateContext(ptr::null_mut(), &mut store_ctx) };
    check(
        &format!("xgameruntime!XStoreCreateContext offline refusal (hr=0x{store_hr:08x})"),
        store_hr == -2147467263,
        failures,
    );

    // SAFETY: Matches successful `XGameRuntimeInitialize`.
    unsafe { XGameRuntimeUninitialize() };
    check("xgameruntime!XGameRuntimeUninitialize", true, failures);

    // 4. D3D12CreateDevice + spare vtable probe slot 45 (+ optional Vulkan M2.4 triangle draw)
    let mut dev: *mut c_void = ptr::null_mut();
    // SAFETY: `dev` is a writable COM output pointer.
    let d3d_hr = unsafe { D3D12CreateDevice(ptr::null_mut(), 0xb000, ptr::null(), &mut dev) };
    let mut d3d_ok = d3d_hr == 0 && !dev.is_null();
    if d3d_ok {
        // SAFETY: `dev` points to a COM `ID3D12DeviceX` whose first field is `*const usize`
        // vtable with 128 Win64 slots (`slot 2 = Release`, `slot 45 = spare probe thunk`).
        unsafe {
            let vtbl = *(dev as *const *const usize);
            let probe_45: extern "win64" fn(*mut c_void) -> u64 =
                std::mem::transmute(*vtbl.add(45));
            let probe_ret = probe_45(dev);
            d3d_ok = probe_ret == 0;

            if d3d_ok && render_vulkan {
                let draw_ok = run_vulkan_triangle_draw(dev);
                check(
                    "d3d12_xs!Vulkan 64x64 triangle draw + fence + PresentX",
                    draw_ok,
                    failures,
                );
            }

            let release: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vtbl.add(2));
            let rem_refs = release(dev);
            d3d_ok = d3d_ok && rem_refs == 0;
        }
    }
    check(
        "d3d12_xs!D3D12CreateDevice + spare vtable probe[45] + Release",
        d3d_ok,
        failures,
    );

    // 5. Unknown ordinal kernelx.dll!#999 auto-stub
    // SAFETY: Exported via `kernelx.def` NONAME ordinal `#999` auto-stub thunk.
    let ord_ret = unsafe { x2l_synthetic_ordinal_probe() };
    check(
        &format!("kernelx.dll!#999 unknown-ordinal auto-stub (ret=0x{ord_ret:x})"),
        ord_ret == 0,
        failures,
    );

    // Avoid unused warning on MaybeUninit import when not called directly in this scope.
    let _ = MaybeUninit::<u8>::uninit();
}

#[cfg(target_os = "windows")]
unsafe fn run_vulkan_triangle_draw(dev: *mut std::ffi::c_void) -> bool {
    use console_imports::*;
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::ptr;

    let vs_bytes = synthetic_triangle_vs_bytes();
    let ps_bytes = synthetic_triangle_ps_bytes();

    // SAFETY: Caller passes a live Vulkan-backed COM `ID3D12DeviceX` from `d3d12_xs.dll`
    // implementing the exact Win64 COM vtable slot indices verified in `x2l-gpu-vk::com`.
    unsafe {
        let dev_vt = *(dev as *const *const usize);

        let create_queue: extern "win64" fn(
            *mut c_void,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(8));
        let create_alloc: extern "win64" fn(
            *mut c_void,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(9));
        let create_pso: extern "win64" fn(
            *mut c_void,
            *const D3D12GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(10));
        let create_list: extern "win64" fn(
            *mut c_void,
            u32,
            u32,
            *mut c_void,
            *mut c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
        let create_heap: extern "win64" fn(
            *mut c_void,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(14));
        let create_rtv: extern "win64" fn(
            *mut c_void,
            *mut c_void,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(20));
        let create_res: extern "win64" fn(
            *mut c_void,
            *const D3D12HeapProperties,
            u32,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(27));
        let create_fence: extern "win64" fn(
            *mut c_void,
            u64,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(36));

        let q_desc = D3D12CommandQueueDesc::default();
        let mut queue: *mut c_void = ptr::null_mut();
        if create_queue(dev, &q_desc, ptr::null(), &mut queue) != 0 || queue.is_null() {
            return false;
        }

        let mut allocator: *mut c_void = ptr::null_mut();
        if create_alloc(dev, 0, ptr::null(), &mut allocator) != 0 || allocator.is_null() {
            return false;
        }

        let heap_props = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let rt_desc = D3D12ResourceDesc {
            dimension: 3, // TEXTURE2D
            alignment: 0,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28, // DXGI_FORMAT_R8G8B8A8_UNORM
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            layout: 0,
            flags: 1, // ALLOW_RENDER_TARGET
        };
        let mut rt: *mut c_void = ptr::null_mut();
        if create_res(
            dev,
            &heap_props,
            0,
            &rt_desc,
            0,
            ptr::null(),
            ptr::null(),
            &mut rt,
        ) != 0
            || rt.is_null()
        {
            return false;
        }

        let heap_desc = D3D12DescriptorHeapDesc {
            r#type: 2, // RTV
            num_descriptors: 1,
            flags: 0,
            node_mask: 0,
        };
        let mut rtv_heap: *mut c_void = ptr::null_mut();
        if create_heap(dev, &heap_desc, ptr::null(), &mut rtv_heap) != 0 || rtv_heap.is_null() {
            return false;
        }

        let heap_vt = *(rtv_heap as *const *const usize);
        let get_cpu_start: extern "win64" fn(
            *mut c_void,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*heap_vt.add(9));
        let mut rtv_handle = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        let ret_ptr = get_cpu_start(rtv_heap, rtv_handle.as_mut_ptr());
        if ret_ptr != rtv_handle.as_mut_ptr() {
            return false;
        }
        let rtv_handle = rtv_handle.assume_init();
        create_rtv(dev, rt, ptr::null(), rtv_handle);

        let mut pso_desc: D3D12GraphicsPipelineStateDesc = std::mem::zeroed();
        pso_desc.vs = D3D12ShaderBytecode {
            p_shader_bytecode: vs_bytes.as_ptr().cast::<c_void>(),
            bytecode_length: vs_bytes.len(),
        };
        pso_desc.ps = D3D12ShaderBytecode {
            p_shader_bytecode: ps_bytes.as_ptr().cast::<c_void>(),
            bytecode_length: ps_bytes.len(),
        };
        pso_desc.num_render_targets = 1;
        pso_desc.rtv_formats[0] = 28; // DXGI_FORMAT_R8G8B8A8_UNORM
        pso_desc.sample_desc = DxgiSampleDesc {
            count: 1,
            quality: 0,
        };
        let mut pso: *mut c_void = ptr::null_mut();
        if create_pso(dev, &pso_desc, ptr::null(), &mut pso) != 0 || pso.is_null() {
            return false;
        }

        let mut list: *mut c_void = ptr::null_mut();
        if create_list(dev, 0, 0, allocator, pso, ptr::null(), &mut list) != 0 || list.is_null() {
            return false;
        }

        let mut fence: *mut c_void = ptr::null_mut();
        if create_fence(dev, 0, 0, ptr::null(), &mut fence) != 0 || fence.is_null() {
            return false;
        }

        let list_vt = *(list as *const *const usize);
        let list_close: extern "win64" fn(*mut c_void) -> i32 =
            std::mem::transmute(*list_vt.add(9));
        let list_draw: extern "win64" fn(*mut c_void, u32, u32, u32, u32) =
            std::mem::transmute(*list_vt.add(12));
        let list_topo: extern "win64" fn(*mut c_void, u32) = std::mem::transmute(*list_vt.add(20));
        let list_vp: extern "win64" fn(*mut c_void, u32, *const D3D12Viewport) =
            std::mem::transmute(*list_vt.add(21));
        let list_sc: extern "win64" fn(*mut c_void, u32, *const D3D12Rect) =
            std::mem::transmute(*list_vt.add(22));
        let list_set_pso: extern "win64" fn(*mut c_void, *mut c_void) =
            std::mem::transmute(*list_vt.add(25));
        let list_barrier: extern "win64" fn(*mut c_void, u32, *const D3D12ResourceBarrier) =
            std::mem::transmute(*list_vt.add(26));
        let list_om_rt: extern "win64" fn(
            *mut c_void,
            u32,
            *const D3D12CpuDescriptorHandle,
            i32,
            *const D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*list_vt.add(50));
        let list_clear_rtv: extern "win64" fn(
            *mut c_void,
            D3D12CpuDescriptorHandle,
            *const f32,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*list_vt.add(52));

        let barrier = D3D12ResourceBarrier {
            r#type: 0,
            flags: 0,
            u: D3D12ResourceBarrierUnion {
                transition: D3D12ResourceTransitionBarrier {
                    p_resource: rt,
                    subresource: u32::MAX,
                    state_before: 0,
                    state_after: 4,
                },
            },
        };
        list_barrier(list, 1, &barrier);

        let clear_color = [0.0f32, 0.0, 64.0 / 255.0, 1.0];
        list_clear_rtv(list, rtv_handle, clear_color.as_ptr(), 0, ptr::null());

        let vp = D3D12Viewport {
            top_left_x: 0.0,
            top_left_y: 0.0,
            width: 64.0,
            height: 64.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let sc = D3D12Rect {
            left: 0,
            top: 0,
            right: 64,
            bottom: 64,
        };
        list_vp(list, 1, &vp);
        list_sc(list, 1, &sc);
        list_om_rt(list, 1, &rtv_handle, 0, ptr::null());
        list_set_pso(list, pso);
        list_topo(list, 4); // TRIANGLELIST
        list_draw(list, 3, 1, 0, 0);
        if list_close(list) != 0 {
            return false;
        }

        let q_vt = *(queue as *const *const usize);
        let q_exec: extern "win64" fn(*mut c_void, u32, *const *mut c_void) =
            std::mem::transmute(*q_vt.add(10));
        let q_signal: extern "win64" fn(*mut c_void, *mut c_void, u64) -> i32 =
            std::mem::transmute(*q_vt.add(14));
        let q_present_x: extern "win64" fn(
            *mut c_void,
            u32,
            *const D3D12XboxPresentPlaneParameters,
            *const c_void,
        ) -> i32 = std::mem::transmute(*q_vt.add(30));

        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());
        if q_signal(queue, fence, 1) != 0 {
            return false;
        }

        let f_vt = *(fence as *const *const usize);
        let f_completed: extern "win64" fn(*mut c_void) -> u64 = std::mem::transmute(*f_vt.add(8));
        let f_wait: extern "win64" fn(*mut c_void, u64, *mut c_void) -> i32 =
            std::mem::transmute(*f_vt.add(9));
        if f_wait(fence, 1, ptr::null_mut()) != 0 || f_completed(fence) < 1 {
            return false;
        }

        let rt_list = [rt];
        let plane = D3D12XboxPresentPlaneParameters {
            token: 0,
            resource_count: 1,
            _pad0: 0,
            pp_resources: rt_list.as_ptr(),
            _reserved: [0; 6],
        };
        if q_present_x(queue, 1, &plane, ptr::null()) != 0 {
            return false;
        }

        // Release all child COM objects cleanly.
        let release_obj = |obj: *mut c_void| -> bool {
            let vt = *(obj as *const *const usize);
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            rel(obj) == 0
        };
        let released = release_obj(fence)
            && release_obj(list)
            && release_obj(pso)
            && release_obj(rtv_heap)
            && release_obj(rt)
            && release_obj(allocator)
            && release_obj(queue);
        if !released {
            return false;
        }

        // If X2L_FRAME_DUMP is set, verify the dumped P6 PPM pixels directly!
        if let Ok(dump_path) = std::env::var("X2L_FRAME_DUMP")
            && !dump_path.is_empty()
        {
            let win_path = if dump_path.starts_with('/') {
                format!("Z:{}", dump_path.replace('/', "\\"))
            } else {
                dump_path
            };
            if let Ok(ppm) = fs::read(&win_path) {
                let header = b"P6\n64 64\n255\n";
                if !ppm.starts_with(header) || ppm.len() != header.len() + 64 * 64 * 3 {
                    return false;
                }
                let rgb = &ppm[header.len()..];
                let px = |x: usize, y: usize| -> [u8; 3] {
                    let i = (y * 64 + x) * 3;
                    [rgb[i], rgb[i + 1], rgb[i + 2]]
                };
                let corner = px(4, 4);
                let center = px(32, 32);
                println!(
                    "[PASS] verified dumped frame pixels: corner(4,4)={corner:?} center(32,32)={center:?}"
                );
                if corner != [0, 0, 64] || center != [255, 255, 0] {
                    return false;
                }
            }
        }

        true
    }
}

fn run_overlay_fixture_checks(failures: &mut u32) {
    let keep = Path::new("fixture_keep.txt");
    if !keep.is_file() {
        println!("[SKIP] OverlayFS G:\\ fixture files not present in working directory");
        return;
    }

    let read_ok = fs::read_to_string(keep)
        .map(|s| s.trim() == "immutable-keep-payload")
        .unwrap_or(false);
    check("overlay read fixture_keep.txt", read_ok, failures);

    let modify = Path::new("fixture_modify.txt");
    let cow_ok = fs::write(modify, "overwritten-inside-overlay\n").is_ok()
        && fs::read_to_string(modify)
            .map(|s| s == "overwritten-inside-overlay\n")
            .unwrap_or(false);
    check("overlay CoW overwrite fixture_modify.txt", cow_ok, failures);

    let created = Path::new("fixture_created.txt");
    let create_ok = fs::write(created, "created-inside-overlay\n").is_ok()
        && fs::read_to_string(created)
            .map(|s| s == "created-inside-overlay\n")
            .unwrap_or(false);
    check("overlay create fixture_created.txt", create_ok, failures);

    let delete_target = Path::new("fixture_delete.txt");
    let delete_ok = fs::remove_file(delete_target).is_ok() && !delete_target.exists();
    check(
        "overlay whiteout delete fixture_delete.txt",
        delete_ok,
        failures,
    );
}
