// Clean-room end-to-end headless Vulkan 1.3 + Win64 COM draw test for `x2l-gpu-vk`.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use x2l_gpu_vk::com::{
    ComCommandAllocator, ComCommandQueue, ComDescriptorHeap, ComDevice, ComFence,
    ComGraphicsCommandList, ComPipelineState, ComResource, D3D12CommandQueueDesc,
    D3D12CpuDescriptorHandle, D3D12DescriptorHeapDesc, D3D12GraphicsPipelineStateDesc,
    D3D12HeapProperties, D3D12Rect, D3D12ResourceBarrier, D3D12ResourceBarrierUnion,
    D3D12ResourceDesc, D3D12ResourceTransitionBarrier, D3D12ShaderBytecode, D3D12Viewport,
    D3D12XboxPresentPlaneParameters, DxgiSampleDesc, S_OK, create_vulkan_com_device,
};
use x2l_gpu_vk::vulkan::{
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    DXGI_FORMAT_R8G8B8A8_UNORM, VulkanBackend, VulkanBackendConfig,
};
use x2l_rdna2_spirv::encode as asm;

static PROBE_LAST_SLOT: AtomicU32 = AtomicU32::new(0);
static PROBE_LAST_RIP: AtomicU64 = AtomicU64::new(0);

extern "win64" fn test_com_probe(slot: u32, rip: u64) -> u64 {
    PROBE_LAST_SLOT.store(slot, Ordering::SeqCst);
    PROBE_LAST_RIP.store(rip, Ordering::SeqCst);
    0
}

/// Build a procedural RDNA2 Vertex Shader (`SV_VertexID` in `v0`) emitting a centered triangle:
/// - v0 == 0: ( 0.0,  0.5, 0.0, 1.0)
/// - v0 == 1: ( 0.5, -0.5, 0.0, 1.0)
/// - v0 == 2: (-0.5, -0.5, 0.0, 1.0)
///
/// Exports vivid yellow `(1.0, 1.0, 0.0, 1.0)` in `param0`.
fn synthetic_triangle_vs_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    // Default = vertex 2: (-0.5, -0.5, 0.0, 1.0)
    a.push(asm::v_mov_b32(1, asm::F_NEG_0_5)); // x
    a.push(asm::v_mov_b32(2, asm::F_NEG_0_5)); // y
    a.push(asm::v_mov_b32(3, asm::inline_i(0))); // z = 0.0 (bits 0x00000000)
    a.push(asm::v_mov_b32(4, asm::F_1_0)); // w = 1.0
    a.push(asm::v_mov_b32(9, asm::F_0_5)); // +0.5 helper VGPR
    a.push(asm::v_mov_b32(10, asm::inline_i(0))); // 0.0 helper VGPR

    // If v0 == 1 -> x = +0.5
    a.push(asm::v_cmp_eq_u32(asm::inline_i(1), 0));
    a.push(asm::v_cndmask_b32(1, asm::v(1), 9));

    // If v0 == 0 -> x = 0.0, y = +0.5
    a.push(asm::v_cmp_eq_u32(asm::inline_i(0), 0));
    a.push(asm::v_cndmask_b32(1, asm::v(1), 10));
    a.push(asm::v_cndmask_b32(2, asm::v(2), 9));

    // Export yellow color (1.0, 1.0, 0.0, 1.0) via param0
    a.push(asm::v_mov_b32(5, asm::F_1_0));
    a.push(asm::v_mov_b32(6, asm::F_1_0));
    a.push(asm::v_mov_b32(7, asm::inline_i(0)));
    a.push(asm::v_mov_b32(8, asm::F_1_0));

    a.push2(asm::exp_pos0([1, 2, 3, 4], false));
    a.push2(asm::exp_param(0, [5, 6, 7, 8], true));
    a.push(asm::s_endpgm());

    let words = a.into_words();
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Build an RDNA2 Pixel Shader reading interpolated `param0` RGB color via `VINTRP` and
/// exporting it to `mrt0`.
fn synthetic_triangle_ps_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    for ch in 0..4u8 {
        a.push(asm::v_interp_p1_f32(ch, 0, 0, ch));
        a.push(asm::v_interp_p2_f32(ch, 1, 0, ch));
    }
    a.push2(asm::exp_mrt(0, [0, 1, 2, 3], false, true, false));
    a.push(asm::s_endpgm());

    let words = a.into_words();
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

#[test]
fn com_d3d12x_renders_verified_triangle_and_preserves_slot_45_probe() {
    let tmp_dir = std::env::temp_dir().join(format!("x2l-gpu-com-test-{}", std::process::id()));
    let cache_dir = tmp_dir.join("spirv");
    let ppm_path = tmp_dir.join("frame.ppm");

    let mut cfg = VulkanBackendConfig::from_env();
    cfg.spirv_cache_dir = Some(cache_dir.clone());
    cfg.frame_dump_path = Some(ppm_path.clone());

    let backend = Arc::new(VulkanBackend::new(cfg).expect("init VulkanBackend"));
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_com_probe)
        .expect("create COM ID3D12DeviceX");

    let vs_bytes = synthetic_triangle_vs_bytes();
    let ps_bytes = synthetic_triangle_ps_bytes();

    // SAFETY: All COM pointers and vtable slot indices match `com.rs` exact Win64 ABI layouts.
    unsafe {
        let dev_vt = (*dev).vtbl;

        // 1. Verify M1 unimplemented spare slot 45 still invokes the W^X probe callback!
        let probe_45: extern "win64" fn(*mut ComDevice) -> u64 =
            std::mem::transmute(*dev_vt.add(45));
        assert_eq!(probe_45(dev), 0);
        assert_eq!(PROBE_LAST_SLOT.load(Ordering::SeqCst), 45);
        assert_ne!(PROBE_LAST_RIP.load(Ordering::SeqCst), 0);

        // 2. Create CommandQueue (slot 8)
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(8));
        let q_desc = D3D12CommandQueueDesc::default();
        let mut queue_raw: *mut c_void = ptr::null_mut();
        assert_eq!(
            create_queue(dev, &q_desc, ptr::null(), &mut queue_raw),
            S_OK
        );
        let queue = queue_raw.cast::<ComCommandQueue>();

        // 3. Create CommandAllocator (slot 9)
        let create_alloc: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(9));
        let mut alloc_raw: *mut c_void = ptr::null_mut();
        assert_eq!(create_alloc(dev, 0, ptr::null(), &mut alloc_raw), S_OK);
        let allocator = alloc_raw.cast::<ComCommandAllocator>();

        // 4. Create 64x64 R8G8B8A8_UNORM Render Target (slot 27)
        let create_res: extern "win64" fn(
            *mut ComDevice,
            *const D3D12HeapProperties,
            u32,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(27));
        let heap_props = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let rt_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            alignment: 0,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            layout: 0,
            flags: 1, // ALLOW_RENDER_TARGET
        };
        let mut rt_raw: *mut c_void = ptr::null_mut();
        assert_eq!(
            create_res(
                dev,
                &heap_props,
                0,
                &rt_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut rt_raw
            ),
            S_OK
        );
        let rt = rt_raw.cast::<ComResource>();

        // Also test a committed buffer resource with Map/Unmap/GetGPUVirtualAddress.
        let buf_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut buf_raw: *mut c_void = ptr::null_mut();
        assert_eq!(
            create_res(
                dev,
                &heap_props,
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut buf_raw
            ),
            S_OK
        );
        let buf_res = buf_raw.cast::<ComResource>();
        let buf_vt = (*buf_res).vtbl;
        let res_map: extern "win64" fn(
            *mut ComResource,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*buf_vt.add(8));
        let res_unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*buf_vt.add(9));
        let res_get_va: extern "win64" fn(*mut ComResource) -> u64 =
            std::mem::transmute(*buf_vt.add(11));
        assert_ne!(res_get_va(buf_res), 0);
        let mut mapped_ptr: *mut c_void = ptr::null_mut();
        assert_eq!(res_map(buf_res, 0, ptr::null(), &mut mapped_ptr), S_OK);
        assert!(!mapped_ptr.is_null());
        ptr::write_volatile(mapped_ptr.cast::<u8>(), 0x5A);
        res_unmap(buf_res, 0, ptr::null());

        // 5. Create RTV Descriptor Heap (slot 14) & get CPU handle (MSVC struct-return ABI!)
        let create_heap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(14));
        let heap_desc = D3D12DescriptorHeapDesc {
            r#type: 2, // RTV
            num_descriptors: 1,
            flags: 0,
            node_mask: 0,
        };
        let mut heap_raw: *mut c_void = ptr::null_mut();
        assert_eq!(
            create_heap(dev, &heap_desc, ptr::null(), &mut heap_raw),
            S_OK
        );
        let heap = heap_raw.cast::<ComDescriptorHeap>();
        let heap_vt = (*heap).vtbl;
        let get_cpu_start: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*heap_vt.add(9));
        let mut rtv_handle = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        let ret_ptr = get_cpu_start(heap, rtv_handle.as_mut_ptr());
        assert_eq!(ret_ptr, rtv_handle.as_mut_ptr());
        let rtv_handle = rtv_handle.assume_init();
        assert_ne!(rtv_handle.ptr, 0);

        // 6. CreateRenderTargetView (slot 20)
        let create_rtv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(20));
        create_rtv(dev, rt, ptr::null(), rtv_handle);

        // 7. CreateGraphicsPipelineState (slot 10)
        let create_pso: extern "win64" fn(
            *mut ComDevice,
            *const D3D12GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(10));
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
        pso_desc.rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.sample_desc = DxgiSampleDesc {
            count: 1,
            quality: 0,
        };
        let mut pso_raw: *mut c_void = ptr::null_mut();
        assert_eq!(create_pso(dev, &pso_desc, ptr::null(), &mut pso_raw), S_OK);
        let pso = pso_raw.cast::<ComPipelineState>();

        // 8. CreateCommandList (slot 12) & CreateFence (slot 36)
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
        let mut list_raw: *mut c_void = ptr::null_mut();
        assert_eq!(
            create_list(dev, 0, 0, allocator, pso, ptr::null(), &mut list_raw),
            S_OK
        );
        let list = list_raw.cast::<ComGraphicsCommandList>();

        let create_fence: extern "win64" fn(
            *mut ComDevice,
            u64,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(36));
        let mut fence_raw: *mut c_void = ptr::null_mut();
        assert_eq!(create_fence(dev, 0, 0, ptr::null(), &mut fence_raw), S_OK);
        let fence = fence_raw.cast::<ComFence>();

        // 9. Record commands: Barrier, Clear (dark blue), Viewport, Scissor, OMSetRenderTargets,
        //    SetPSO, Topology, DrawInstanced(3,1,0,0), Close.
        let list_vt = (*list).vtbl;
        let list_close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
            std::mem::transmute(*list_vt.add(9));
        let list_draw: extern "win64" fn(*mut ComGraphicsCommandList, u32, u32, u32, u32) =
            std::mem::transmute(*list_vt.add(12));
        let list_topo: extern "win64" fn(*mut ComGraphicsCommandList, u32) =
            std::mem::transmute(*list_vt.add(20));
        let list_vp: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const D3D12Viewport) =
            std::mem::transmute(*list_vt.add(21));
        let list_sc: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const D3D12Rect) =
            std::mem::transmute(*list_vt.add(22));
        let list_set_pso: extern "win64" fn(*mut ComGraphicsCommandList, *mut ComPipelineState) =
            std::mem::transmute(*list_vt.add(25));
        let list_barrier: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            *const D3D12ResourceBarrier,
        ) = std::mem::transmute(*list_vt.add(26));
        let list_om_rt: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            *const D3D12CpuDescriptorHandle,
            i32,
            *const D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*list_vt.add(50));
        let list_clear_rtv: extern "win64" fn(
            *mut ComGraphicsCommandList,
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
                    state_after: 4, // RENDER_TARGET
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
        list_topo(list, 4); // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
        list_draw(list, 3, 1, 0, 0);
        assert_eq!(list_close(list), S_OK);

        // 10. ExecuteCommandLists (slot 10), Signal (slot 14), Wait on Fence, PresentX (slot 30)
        let q_vt = (*queue).vtbl;
        let q_exec: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const *mut ComGraphicsCommandList,
        ) = std::mem::transmute(*q_vt.add(10));
        let q_signal: extern "win64" fn(*mut ComCommandQueue, *mut ComFence, u64) -> i32 =
            std::mem::transmute(*q_vt.add(14));
        let q_present_x: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const D3D12XboxPresentPlaneParameters,
            *const c_void,
        ) -> i32 = std::mem::transmute(*q_vt.add(30));

        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());
        assert_eq!(q_signal(queue, fence, 1), S_OK);

        let f_vt = (*fence).vtbl;
        let f_completed: extern "win64" fn(*mut ComFence) -> u64 =
            std::mem::transmute(*f_vt.add(8));
        let f_wait: extern "win64" fn(*mut ComFence, u64, *mut c_void) -> i32 =
            std::mem::transmute(*f_vt.add(9));
        assert_eq!(f_wait(fence, 1, ptr::null_mut()), S_OK);
        assert!(f_completed(fence) >= 1);

        let rt_list = [rt];
        let plane = D3D12XboxPresentPlaneParameters {
            token: 0,
            resource_count: 1,
            _pad0: 0,
            pp_resources: rt_list.as_ptr(),
            _reserved: [0; 6],
        };
        assert_eq!(q_present_x(queue, 1, &plane, ptr::null()), S_OK);

        // 11. Verify rendered pixels, PPM dump, SPIR-V cache files, and telemetry!
        let frame = backend
            .last_presented_frame()
            .expect("PresentX must capture frame");
        assert_eq!((frame.width, frame.height), (64, 64));
        let corner = frame.pixel(4, 4).expect("corner pixel");
        let center = frame.pixel(32, 32).expect("center pixel");
        println!(
            "[gpu-com-draw] corner(4,4)={corner:?} center(32,32)={center:?} telemetry={:?}",
            backend.telemetry_snapshot()
        );
        assert_eq!(
            corner,
            [0, 0, 64, 255],
            "outside triangle must be dark blue"
        );
        assert_eq!(
            center,
            [255, 255, 0, 255],
            "inside triangle must be vivid yellow"
        );

        let ppm = std::fs::read(&ppm_path).expect("PPM dump file written");
        assert!(ppm.starts_with(b"P6\n64 64\n255\n"));
        assert_eq!(ppm.len(), b"P6\n64 64\n255\n".len() + 64 * 64 * 3);

        let snap = backend.telemetry_snapshot();
        assert_eq!(snap.shaders_compiled, 2);
        assert_eq!(snap.frames_presented, 1);

        // Release all COM objects cleanly.
        let release_obj = |obj: *mut c_void, vt: *const usize| {
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            assert_eq!(rel(obj), 0);
        };
        release_obj(fence.cast(), (*fence).vtbl);
        release_obj(list.cast(), (*list).vtbl);
        release_obj(pso.cast(), (*pso).vtbl);
        release_obj(heap.cast(), (*heap).vtbl);
        release_obj(buf_res.cast(), (*buf_res).vtbl);
        release_obj(rt.cast(), (*rt).vtbl);
        release_obj(allocator.cast(), (*allocator).vtbl);
        release_obj(queue.cast(), (*queue).vtbl);
        release_obj(dev.cast(), (*dev).vtbl);
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
