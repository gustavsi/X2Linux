// Clean-room end-to-end headless Vulkan 1.3 + Win64 COM tests for M3.3 expanded D3D12.X surface:
// - Indexed draws + InputLayout + root 32-bit constants + inline CBV
// - D32_FLOAT depth buffer + ClearDepthStencilView + depth occlusion
// - Placed resources (CreateHeap + CreatePlacedResource) + CopyBufferRegion + CopyTextureRegion
// - Compute pipeline (CreateComputePipelineState + Dispatch + UAV descriptor table + readback)
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use x2l_gpu_vk::com::{
    ComCommandAllocator, ComCommandQueue, ComDescriptorHeap, ComDevice, ComFence,
    ComGraphicsCommandList, ComHeap, ComPipelineState, ComResource, ComRootSignature,
    D3D12CommandQueueDesc, D3D12ComputePipelineStateDesc, D3D12CpuDescriptorHandle,
    D3D12DepthStencilViewDesc, D3D12DescriptorHeapDesc, D3D12GpuDescriptorHandle,
    D3D12GraphicsPipelineStateDesc, D3D12HeapDesc, D3D12HeapProperties, D3D12IndexBufferView,
    D3D12InputElementDesc, D3D12InputLayoutDesc, D3D12PlacedSubresourceFootprint, D3D12Rect,
    D3D12ResourceDesc, D3D12SamplerDesc, D3D12ShaderBytecode, D3D12ShaderResourceViewDesc,
    D3D12TextureCopyLocation, D3D12TextureCopyLocationUnion, D3D12UnorderedAccessViewDesc,
    D3D12VertexBufferView, D3D12Viewport, D3D12XboxPresentPlaneParameters, DxgiSampleDesc, S_OK,
    create_vulkan_com_device,
};
use x2l_gpu_vk::d3d12x_ext::{
    ComD3DBlob, D3D12RootParameterRaw, D3D12VersionedRootSignatureDesc,
    serialize_versioned_root_signature,
};
use x2l_gpu_vk::vulkan::{
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_D32_FLOAT,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT,
    VulkanBackend, VulkanBackendConfig,
};
use x2l_rdna2_spirv::encode as asm;

extern "win64" fn test_probe(_slot: u32, _rip: u64) -> u64 {
    0
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Pass-through VS: takes POSITION vec4 in v0..v3, reads root constants s0..s2 (R, G, B) +
/// CBV word 0 into s8 (used for blue channel) and exports (s0, s1, s8, 1.0) in param0.
fn indexed_root_cbv_vs_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    // Load CBV word 0 (bound via inline CBV at root param 1 -> s4:s5 = 0) into s8.
    a.push2(asm::s_load_dword(8, 4, 0x7D, 0));
    a.push(asm::v_mov_b32(4, asm::s(0))); // R from root const 0
    a.push(asm::v_mov_b32(5, asm::s(1))); // G from root const 1
    a.push(asm::v_mov_b32(6, asm::s(8))); // B from CBV word 0
    a.push(asm::v_mov_b32(7, asm::F_1_0)); // A = 1.0
    a.push2(asm::exp_pos0([0, 1, 2, 3], false));
    a.push2(asm::exp_param(0, [4, 5, 6, 7], true));
    a.push(asm::s_endpgm());
    words_to_bytes(a.words())
}

/// Depth-test VS: takes POSITION vec4 in v0..v3, reads RGBA color from root constants s0..s3,
/// exports pos0 and param0.
fn depth_test_vs_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    a.push(asm::v_mov_b32(4, asm::s(0)));
    a.push(asm::v_mov_b32(5, asm::s(1)));
    a.push(asm::v_mov_b32(6, asm::s(2)));
    a.push(asm::v_mov_b32(7, asm::s(3)));
    a.push2(asm::exp_pos0([0, 1, 2, 3], false));
    a.push2(asm::exp_param(0, [4, 5, 6, 7], true));
    a.push(asm::s_endpgm());
    words_to_bytes(a.words())
}

fn passthrough_param0_ps_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    for ch in 0..4u8 {
        a.push(asm::v_interp_p1_f32(ch, 0, 0, ch));
        a.push(asm::v_interp_p2_f32(ch, 1, 0, ch));
    }
    a.push2(asm::exp_mrt(0, [0, 1, 2, 3], false, true, false));
    a.push(asm::s_endpgm());
    words_to_bytes(a.words())
}

/// Compute shader:
/// - Root param 0 (s0, s1): base magic + stride
/// - Root param 1 (inline CBV @ s2:s3): word 0 = offset
/// - Root param 2 (UAV table): writes `s0 + s8 + v0 * s1` to byte offset `v0 * 4`.
fn compute_signature_cs_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    // s8 = CBV[0]
    a.push2(asm::s_load_dword(8, 2, 0x7D, 0));
    // v1 = v0 * s1
    a.push2(asm::v_mul_lo_u32(1, asm::v(0), asm::s(1)));
    // v1 = v1 + s0
    a.push(asm::v_add_nc_u32(1, asm::s(0), 1));
    // v1 = v1 + s8
    a.push(asm::v_add_nc_u32(1, asm::s(8), 1));
    // v2 = v0 * 4 (byte offset)
    a.push2(asm::v_mul_lo_u32(2, asm::v(0), asm::inline_i(4)));
    // store v1 to UAV[v2]
    a.push2(asm::buffer_store_dword(
        1,
        2,
        12,
        asm::inline_i(0),
        0,
        true,
        false,
    ));
    a.push(asm::s_endpgm());
    words_to_bytes(a.words())
}

unsafe fn serialize_and_create_root_sig(
    dev: *mut ComDevice,
    params: &[D3D12RootParameterRaw],
) -> *mut ComRootSignature {
    let desc = D3D12VersionedRootSignatureDesc {
        version: 2,
        _pad04: 0,
        num_parameters: params.len() as u32,
        _pad0c: 0,
        p_parameters: if params.is_empty() {
            ptr::null()
        } else {
            params.as_ptr().cast()
        },
        num_static_samplers: 0,
        _pad1c: 0,
        p_static_samplers: ptr::null(),
        flags: 1,
        _pad2c: 0,
    };
    let mut blob_raw: *mut c_void = ptr::null_mut();
    // SAFETY: Caller passes valid `dev` and stack `params`.
    unsafe {
        assert_eq!(
            serialize_versioned_root_signature(&desc, &mut blob_raw, ptr::null_mut()),
            S_OK
        );
        let blob = blob_raw.cast::<ComD3DBlob>();
        let bvt = (*blob).vtbl;
        let get_ptr: extern "win64" fn(*mut ComD3DBlob) -> *mut c_void =
            std::mem::transmute(*bvt.add(3));
        let get_sz: extern "win64" fn(*mut ComD3DBlob) -> usize = std::mem::transmute(*bvt.add(4));
        let rel: extern "win64" fn(*mut ComD3DBlob) -> u32 = std::mem::transmute(*bvt.add(2));

        let dev_vt = (*dev).vtbl;
        let create_rs: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            usize,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(16));
        let mut rs_raw: *mut c_void = ptr::null_mut();
        assert_eq!(
            create_rs(
                dev,
                0,
                get_ptr(blob),
                get_sz(blob),
                ptr::null(),
                &mut rs_raw
            ),
            S_OK
        );
        assert_eq!(rel(blob), 0);
        rs_raw.cast::<ComRootSignature>()
    }
}

#[test]
fn indexed_draw_with_placed_buffers_copy_root_constants_and_cbv() {
    let cfg = VulkanBackendConfig::from_env();
    let backend = Arc::new(VulkanBackend::new(cfg).expect("init VulkanBackend"));
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe).expect("create ComDevice");

    // SAFETY: All COM calls follow verified Win64 console D3D12.X ABI.
    unsafe {
        let dev_vt = (*dev).vtbl;
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(8));
        let create_alloc: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(9));
        let create_pso: extern "win64" fn(
            *mut ComDevice,
            *const D3D12GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(10));
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
        let create_dheap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(14));
        let create_rtv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(20));
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
        let create_heap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12HeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(28));
        let create_placed: extern "win64" fn(
            *mut ComDevice,
            *mut ComHeap,
            u64,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(29));

        // 1. Root signature: param 0 = 4 root 32-bit constants (s0..s3), param 1 = inline CBV (s4..s5)
        let rs_params = [
            D3D12RootParameterRaw {
                parameter_type: 1, // 32BIT_CONSTANTS
                _pad04: 0,
                data: [0, 0, 4, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 2, // CBV
                _pad04: 0,
                data: [1, 0, 0, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
        ];
        let root_sig = serialize_and_create_root_sig(dev, &rs_params);

        // 2. Queue, Allocator, RTV heap, 64x64 RT
        let mut queue_raw = ptr::null_mut();
        assert_eq!(
            create_queue(
                dev,
                &D3D12CommandQueueDesc::default(),
                ptr::null(),
                &mut queue_raw
            ),
            S_OK
        );
        let queue = queue_raw.cast::<ComCommandQueue>();

        let mut alloc_raw = ptr::null_mut();
        assert_eq!(create_alloc(dev, 0, ptr::null(), &mut alloc_raw), S_OK);
        let allocator = alloc_raw.cast::<ComCommandAllocator>();

        let mut rt_raw = ptr::null_mut();
        let rt_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            flags: 1,
            ..Default::default()
        };
        assert_eq!(
            create_res(
                dev,
                &D3D12HeapProperties {
                    r#type: 1,
                    ..Default::default()
                },
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

        let mut rtv_heap_raw = ptr::null_mut();
        assert_eq!(
            create_dheap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 2,
                    num_descriptors: 1,
                    ..Default::default()
                },
                ptr::null(),
                &mut rtv_heap_raw
            ),
            S_OK
        );
        let rtv_heap = rtv_heap_raw.cast::<ComDescriptorHeap>();
        let get_cpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*(*rtv_heap).vtbl.add(9));
        let mut rtv_h = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        get_cpu(rtv_heap, rtv_h.as_mut_ptr());
        let rtv_h = rtv_h.assume_init();
        create_rtv(dev, rt, ptr::null(), rtv_h);

        // 3. CreateHeap (128 KiB) + Placed VB (@0) + Placed IB (@65536) + Committed Upload & CBV
        let mut heap_raw = ptr::null_mut();
        let heap_desc = D3D12HeapDesc {
            size_in_bytes: 131_072,
            properties: D3D12HeapProperties {
                r#type: 1,
                ..Default::default()
            },
            alignment: 65_536,
            ..Default::default()
        };
        assert_eq!(
            create_heap(dev, &heap_desc, ptr::null(), &mut heap_raw),
            S_OK
        );
        let gpu_heap = heap_raw.cast::<ComHeap>();

        // Verify ID3D12Heap::GetDesc (slot 8) MSVC struct-return ABI!
        let heap_get_desc: extern "win64" fn(
            *mut ComHeap,
            *mut D3D12HeapDesc,
        ) -> *mut D3D12HeapDesc = std::mem::transmute(*(*gpu_heap).vtbl.add(8));
        let mut queried_hd = MaybeUninit::<D3D12HeapDesc>::uninit();
        assert_eq!(
            heap_get_desc(gpu_heap, queried_hd.as_mut_ptr()),
            queried_hd.as_mut_ptr()
        );
        assert_eq!(queried_hd.assume_init().size_in_bytes, 131_072);

        let buf_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut vb_raw = ptr::null_mut();
        let mut ib_raw = ptr::null_mut();
        let mut upload_raw = ptr::null_mut();
        let mut cbv_raw = ptr::null_mut();
        assert_eq!(
            create_placed(
                dev,
                gpu_heap,
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut vb_raw
            ),
            S_OK
        );
        assert_eq!(
            create_placed(
                dev,
                gpu_heap,
                65_536,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut ib_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &D3D12HeapProperties {
                    r#type: 2,
                    ..Default::default()
                },
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut upload_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &D3D12HeapProperties {
                    r#type: 2,
                    ..Default::default()
                },
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut cbv_raw
            ),
            S_OK
        );
        let vb = vb_raw.cast::<ComResource>();
        let ib = ib_raw.cast::<ComResource>();
        let upload = upload_raw.cast::<ComResource>();
        let cbv = cbv_raw.cast::<ComResource>();

        let res_map: extern "win64" fn(
            *mut ComResource,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*(*upload).vtbl.add(8));
        let res_unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*(*upload).vtbl.add(9));
        let res_va: extern "win64" fn(*mut ComResource) -> u64 =
            std::mem::transmute(*(*upload).vtbl.add(11));

        // Upload 3 vertices (with dummy vertex 0 skipped via indices [1, 2, 3]) + u16 indices at +256.
        let vertices: [[f32; 4]; 4] = [
            [9.0, 9.0, 0.0, 1.0],   // unused vertex 0
            [0.0, 0.6, 0.0, 1.0],   // top
            [0.6, -0.6, 0.0, 1.0],  // bottom-right
            [-0.6, -0.6, 0.0, 1.0], // bottom-left
        ];
        let indices: [u16; 3] = [1, 2, 3];
        let mut up_ptr = ptr::null_mut();
        assert_eq!(res_map(upload, 0, ptr::null(), &mut up_ptr), S_OK);
        ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            up_ptr.cast::<u8>(),
            std::mem::size_of_val(&vertices),
        );
        ptr::copy_nonoverlapping(
            indices.as_ptr().cast::<u8>(),
            up_ptr.cast::<u8>().add(256),
            std::mem::size_of_val(&indices),
        );
        res_unmap(upload, 0, ptr::null());

        // CBV holds blue = 1.0f32 in word 0.
        let mut cbv_ptr = ptr::null_mut();
        assert_eq!(res_map(cbv, 0, ptr::null(), &mut cbv_ptr), S_OK);
        ptr::write_volatile(cbv_ptr.cast::<u32>(), 1.0f32.to_bits());
        res_unmap(cbv, 0, ptr::null());

        // 4. Graphics PSO with InputLayout POSITION (R32G32B32A32_FLOAT) & RootSignature
        let vs_bytes = indexed_root_cbv_vs_bytes();
        let ps_bytes = passthrough_param0_ps_bytes();
        let input_elems = [D3D12InputElementDesc {
            semantic_name: c"POSITION".as_ptr(),
            semantic_index: 0,
            format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            input_slot: 0,
            aligned_byte_offset: 0,
            input_slot_class: 0,
            instance_data_step_rate: 0,
        }];
        let mut pso_desc: D3D12GraphicsPipelineStateDesc = std::mem::zeroed();
        pso_desc.p_root_signature = root_sig.cast();
        pso_desc.vs = D3D12ShaderBytecode {
            p_shader_bytecode: vs_bytes.as_ptr().cast(),
            bytecode_length: vs_bytes.len(),
        };
        pso_desc.ps = D3D12ShaderBytecode {
            p_shader_bytecode: ps_bytes.as_ptr().cast(),
            bytecode_length: ps_bytes.len(),
        };
        pso_desc.input_layout = D3D12InputLayoutDesc {
            p_input_element_descs: input_elems.as_ptr().cast(),
            num_elements: 1,
        };
        pso_desc.num_render_targets = 1;
        pso_desc.rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.sample_desc = DxgiSampleDesc {
            count: 1,
            quality: 0,
        };
        let mut pso_raw = ptr::null_mut();
        assert_eq!(create_pso(dev, &pso_desc, ptr::null(), &mut pso_raw), S_OK);
        let pso = pso_raw.cast::<ComPipelineState>();

        // 5. Record command list: CopyBufferRegion -> Clear -> Bind RootSig/Constants/CBV/VB/IB -> DrawIndexedInstanced
        let mut list_raw = ptr::null_mut();
        assert_eq!(
            create_list(dev, 0, 0, allocator, pso, ptr::null(), &mut list_raw),
            S_OK
        );
        let list = list_raw.cast::<ComGraphicsCommandList>();
        let lvt = (*list).vtbl;
        let list_close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
            std::mem::transmute(*lvt.add(9));
        let list_draw_idx: extern "win64" fn(*mut ComGraphicsCommandList, u32, u32, u32, i32, u32) =
            std::mem::transmute(*lvt.add(13));
        let list_copy_buf: extern "win64" fn(
            *mut ComGraphicsCommandList,
            *mut ComResource,
            u64,
            *mut ComResource,
            u64,
            u64,
        ) = std::mem::transmute(*lvt.add(15));
        let list_vp: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const D3D12Viewport) =
            std::mem::transmute(*lvt.add(21));
        let list_sc: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const D3D12Rect) =
            std::mem::transmute(*lvt.add(22));
        let list_set_gfx_rs: extern "win64" fn(*mut ComGraphicsCommandList, *mut ComRootSignature) =
            std::mem::transmute(*lvt.add(34));
        let list_set_gfx_consts: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            u32,
            *const c_void,
            u32,
        ) = std::mem::transmute(*lvt.add(40));
        let list_set_gfx_cbv: extern "win64" fn(*mut ComGraphicsCommandList, u32, u64) =
            std::mem::transmute(*lvt.add(42));
        let list_ia_ib: extern "win64" fn(
            *mut ComGraphicsCommandList,
            *const D3D12IndexBufferView,
        ) = std::mem::transmute(*lvt.add(47));
        let list_ia_vb: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            u32,
            *const D3D12VertexBufferView,
        ) = std::mem::transmute(*lvt.add(48));
        let list_om_rt: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            *const D3D12CpuDescriptorHandle,
            i32,
            *const D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*lvt.add(50));
        let list_clear_rtv: extern "win64" fn(
            *mut ComGraphicsCommandList,
            D3D12CpuDescriptorHandle,
            *const f32,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*lvt.add(52));

        list_copy_buf(list, vb, 0, upload, 0, 64);
        list_copy_buf(list, ib, 0, upload, 256, 6);
        let clear = [0.0f32, 0.0, 64.0 / 255.0, 1.0];
        list_clear_rtv(list, rtv_h, clear.as_ptr(), 0, ptr::null());
        let vp = D3D12Viewport {
            width: 64.0,
            height: 64.0,
            max_depth: 1.0,
            ..Default::default()
        };
        let sc = D3D12Rect {
            right: 64,
            bottom: 64,
            ..Default::default()
        };
        list_vp(list, 1, &vp);
        list_sc(list, 1, &sc);
        list_om_rt(list, 1, &rtv_h, 0, ptr::null());
        list_set_gfx_rs(list, root_sig);
        // Root constants: R=0.0, G=1.0 -> combined with CBV B=1.0 -> vivid cyan (0, 255, 255, 255)!
        let root_consts: [u32; 4] = [0.0f32.to_bits(), 1.0f32.to_bits(), 0, 0];
        list_set_gfx_consts(list, 0, 4, root_consts.as_ptr().cast(), 0);
        list_set_gfx_cbv(list, 1, res_va(cbv));

        let vbv = D3D12VertexBufferView {
            buffer_location: res_va(vb),
            size_in_bytes: 64,
            stride_in_bytes: 16,
        };
        let ibv = D3D12IndexBufferView {
            buffer_location: res_va(ib),
            size_in_bytes: 6,
            format: DXGI_FORMAT_R16_UINT,
        };
        list_ia_vb(list, 0, 1, &vbv);
        list_ia_ib(list, &ibv);
        list_draw_idx(list, 3, 1, 0, 0, 0);
        assert_eq!(list_close(list), S_OK);

        let qvt = (*queue).vtbl;
        let q_exec: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const *mut ComGraphicsCommandList,
        ) = std::mem::transmute(*qvt.add(10));
        let q_present: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const D3D12XboxPresentPlaneParameters,
            *const c_void,
        ) -> i32 = std::mem::transmute(*qvt.add(30));

        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());
        let rt_list = [rt];
        let plane = D3D12XboxPresentPlaneParameters {
            token: 0,
            resource_count: 1,
            _pad0: 0,
            pp_resources: rt_list.as_ptr(),
            _reserved: [0; 6],
        };
        assert_eq!(q_present(queue, 1, &plane, ptr::null()), S_OK);

        let frame = backend.last_presented_frame().expect("captured frame");
        assert_eq!(frame.pixel(4, 4), Some([0, 0, 64, 255]));
        assert_eq!(
            frame.pixel(32, 32),
            Some([0, 255, 255, 255]),
            "indexed draw must blend root const G=1.0 + CBV B=1.0 into cyan"
        );
    }
}

#[test]
fn depth_buffer_d32_float_occludes_far_geometry_behind_near_triangle() {
    let cfg = VulkanBackendConfig::from_env();
    let backend = Arc::new(VulkanBackend::new(cfg).expect("init VulkanBackend"));
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe).expect("create ComDevice");

    // SAFETY: All COM calls follow verified Win64 console D3D12.X ABI.
    unsafe {
        let dev_vt = (*dev).vtbl;
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(8));
        let create_alloc: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(9));
        let create_pso: extern "win64" fn(
            *mut ComDevice,
            *const D3D12GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(10));
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
        let create_dheap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(14));
        let create_rtv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(20));
        let create_dsv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const D3D12DepthStencilViewDesc,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(21));
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

        let rs_params = [D3D12RootParameterRaw {
            parameter_type: 1,
            _pad04: 0,
            data: [0, 0, 4, 0],
            shader_visibility: 0,
            _pad1c: 0,
        }];
        let root_sig = serialize_and_create_root_sig(dev, &rs_params);

        let mut queue_raw = ptr::null_mut();
        assert_eq!(
            create_queue(
                dev,
                &D3D12CommandQueueDesc::default(),
                ptr::null(),
                &mut queue_raw
            ),
            S_OK
        );
        let queue = queue_raw.cast::<ComCommandQueue>();
        let mut alloc_raw = ptr::null_mut();
        assert_eq!(create_alloc(dev, 0, ptr::null(), &mut alloc_raw), S_OK);
        let allocator = alloc_raw.cast::<ComCommandAllocator>();

        let mut rt_raw = ptr::null_mut();
        let mut ds_raw = ptr::null_mut();
        let mut vb_raw = ptr::null_mut();
        let hp = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &D3D12ResourceDesc {
                    dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    width: 64,
                    height: 64,
                    depth_or_array_size: 1,
                    mip_levels: 1,
                    format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    sample_desc: DxgiSampleDesc {
                        count: 1,
                        quality: 0
                    },
                    flags: 1,
                    ..Default::default()
                },
                0,
                ptr::null(),
                ptr::null(),
                &mut rt_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &D3D12ResourceDesc {
                    dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    width: 64,
                    height: 64,
                    depth_or_array_size: 1,
                    mip_levels: 1,
                    format: DXGI_FORMAT_D32_FLOAT,
                    sample_desc: DxgiSampleDesc {
                        count: 1,
                        quality: 0
                    },
                    flags: 2, // ALLOW_DEPTH_STENCIL
                    ..Default::default()
                },
                0,
                ptr::null(),
                ptr::null(),
                &mut ds_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &D3D12ResourceDesc {
                    dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    width: 4096,
                    height: 1,
                    depth_or_array_size: 1,
                    mip_levels: 1,
                    ..Default::default()
                },
                0,
                ptr::null(),
                ptr::null(),
                &mut vb_raw
            ),
            S_OK
        );
        let rt = rt_raw.cast::<ComResource>();
        let ds = ds_raw.cast::<ComResource>();
        let vb = vb_raw.cast::<ComResource>();

        // Write 2 triangles into VB:
        //   0..3 = near green center triangle at z = 0.25
        //   3..6 = far red full-screen triangle at z = 0.75
        let verts: [[f32; 4]; 6] = [
            [0.0, 0.5, 0.25, 1.0],
            [0.5, -0.5, 0.25, 1.0],
            [-0.5, -0.5, 0.25, 1.0],
            [-1.0, 3.0, 0.75, 1.0],
            [3.0, -1.0, 0.75, 1.0],
            [-1.0, -1.0, 0.75, 1.0],
        ];
        let res_map: extern "win64" fn(
            *mut ComResource,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*(*vb).vtbl.add(8));
        let res_unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*(*vb).vtbl.add(9));
        let res_va: extern "win64" fn(*mut ComResource) -> u64 =
            std::mem::transmute(*(*vb).vtbl.add(11));
        let mut p = ptr::null_mut();
        assert_eq!(res_map(vb, 0, ptr::null(), &mut p), S_OK);
        ptr::copy_nonoverlapping(
            verts.as_ptr().cast::<u8>(),
            p.cast::<u8>(),
            std::mem::size_of_val(&verts),
        );
        res_unmap(vb, 0, ptr::null());

        let mut rtv_heap_raw = ptr::null_mut();
        let mut dsv_heap_raw = ptr::null_mut();
        assert_eq!(
            create_dheap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 2,
                    num_descriptors: 1,
                    ..Default::default()
                },
                ptr::null(),
                &mut rtv_heap_raw
            ),
            S_OK
        );
        assert_eq!(
            create_dheap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 3, // DSV
                    num_descriptors: 1,
                    ..Default::default()
                },
                ptr::null(),
                &mut dsv_heap_raw
            ),
            S_OK
        );
        let rtv_heap = rtv_heap_raw.cast::<ComDescriptorHeap>();
        let dsv_heap = dsv_heap_raw.cast::<ComDescriptorHeap>();
        let get_cpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*(*rtv_heap).vtbl.add(9));
        let mut rtv_h = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        let mut dsv_h = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        get_cpu(rtv_heap, rtv_h.as_mut_ptr());
        get_cpu(dsv_heap, dsv_h.as_mut_ptr());
        let rtv_h = rtv_h.assume_init();
        let dsv_h = dsv_h.assume_init();
        create_rtv(dev, rt, ptr::null(), rtv_h);
        create_dsv(dev, ds, ptr::null(), dsv_h);

        let vs_bytes = depth_test_vs_bytes();
        let ps_bytes = passthrough_param0_ps_bytes();
        let input_elems = [D3D12InputElementDesc {
            semantic_name: c"POSITION".as_ptr(),
            semantic_index: 0,
            format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            input_slot: 0,
            aligned_byte_offset: 0,
            input_slot_class: 0,
            instance_data_step_rate: 0,
        }];
        let mut pso_desc: D3D12GraphicsPipelineStateDesc = std::mem::zeroed();
        pso_desc.p_root_signature = root_sig.cast();
        pso_desc.vs = D3D12ShaderBytecode {
            p_shader_bytecode: vs_bytes.as_ptr().cast(),
            bytecode_length: vs_bytes.len(),
        };
        pso_desc.ps = D3D12ShaderBytecode {
            p_shader_bytecode: ps_bytes.as_ptr().cast(),
            bytecode_length: ps_bytes.len(),
        };
        pso_desc.depth_stencil_state[0] = 1; // DepthEnable = TRUE
        pso_desc.depth_stencil_state[1] = 1; // DepthWriteMask = ALL
        pso_desc.depth_stencil_state[2] = 2; // DepthFunc = LESS
        pso_desc.input_layout = D3D12InputLayoutDesc {
            p_input_element_descs: input_elems.as_ptr().cast(),
            num_elements: 1,
        };
        pso_desc.num_render_targets = 1;
        pso_desc.rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.dsv_format = DXGI_FORMAT_D32_FLOAT;
        pso_desc.sample_desc = DxgiSampleDesc {
            count: 1,
            quality: 0,
        };
        let mut pso_raw = ptr::null_mut();
        assert_eq!(create_pso(dev, &pso_desc, ptr::null(), &mut pso_raw), S_OK);
        let pso = pso_raw.cast::<ComPipelineState>();

        let mut list_raw = ptr::null_mut();
        assert_eq!(
            create_list(dev, 0, 0, allocator, pso, ptr::null(), &mut list_raw),
            S_OK
        );
        let list = list_raw.cast::<ComGraphicsCommandList>();
        let lvt = (*list).vtbl;
        let list_close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
            std::mem::transmute(*lvt.add(9));
        let list_draw: extern "win64" fn(*mut ComGraphicsCommandList, u32, u32, u32, u32) =
            std::mem::transmute(*lvt.add(12));
        let list_set_gfx_rs: extern "win64" fn(*mut ComGraphicsCommandList, *mut ComRootSignature) =
            std::mem::transmute(*lvt.add(34));
        let list_set_gfx_consts: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            u32,
            *const c_void,
            u32,
        ) = std::mem::transmute(*lvt.add(40));
        let list_ia_vb: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            u32,
            *const D3D12VertexBufferView,
        ) = std::mem::transmute(*lvt.add(48));
        let list_om_rt: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            *const D3D12CpuDescriptorHandle,
            i32,
            *const D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*lvt.add(50));
        let list_clear_dsv: extern "win64" fn(
            *mut ComGraphicsCommandList,
            D3D12CpuDescriptorHandle,
            u32,
            f32,
            u8,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*lvt.add(51));
        let list_clear_rtv: extern "win64" fn(
            *mut ComGraphicsCommandList,
            D3D12CpuDescriptorHandle,
            *const f32,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*lvt.add(52));

        let black = [0.0f32, 0.0, 0.0, 1.0];
        list_clear_rtv(list, rtv_h, black.as_ptr(), 0, ptr::null());
        list_clear_dsv(list, dsv_h, 1, 1.0, 0, 0, ptr::null());
        list_om_rt(list, 1, &rtv_h, 0, &dsv_h);
        list_set_gfx_rs(list, root_sig);
        let vbv = D3D12VertexBufferView {
            buffer_location: res_va(vb),
            size_in_bytes: 96,
            stride_in_bytes: 16,
        };
        list_ia_vb(list, 0, 1, &vbv);

        // Draw 1: near green triangle (z = 0.25)
        let green = [
            0.0f32.to_bits(),
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
        ];
        list_set_gfx_consts(list, 0, 4, green.as_ptr().cast(), 0);
        list_draw(list, 3, 1, 0, 0);

        // Draw 2: far red full-screen triangle (z = 0.75) — must NOT overwrite near green pixels!
        let red = [
            1.0f32.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits(),
            1.0f32.to_bits(),
        ];
        list_set_gfx_consts(list, 0, 4, red.as_ptr().cast(), 0);
        list_draw(list, 3, 1, 3, 0);
        assert_eq!(list_close(list), S_OK);

        let qvt = (*queue).vtbl;
        let q_exec: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const *mut ComGraphicsCommandList,
        ) = std::mem::transmute(*qvt.add(10));
        let q_present: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const D3D12XboxPresentPlaneParameters,
            *const c_void,
        ) -> i32 = std::mem::transmute(*qvt.add(30));

        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());
        let rt_list = [rt];
        let plane = D3D12XboxPresentPlaneParameters {
            token: 0,
            resource_count: 1,
            _pad0: 0,
            pp_resources: rt_list.as_ptr(),
            _reserved: [0; 6],
        };
        assert_eq!(q_present(queue, 1, &plane, ptr::null()), S_OK);

        let frame = backend.last_presented_frame().expect("captured frame");
        assert_eq!(
            frame.pixel(32, 32),
            Some([0, 255, 0, 255]),
            "near z=0.25 green triangle must occlude far z=0.75 red triangle"
        );
        assert_eq!(
            frame.pixel(4, 4),
            Some([255, 0, 0, 255]),
            "uncovered corner must receive far z=0.75 red triangle"
        );
    }
}

#[test]
fn compute_dispatch_with_root_constants_cbv_and_uav_table_reads_back_exact_signature() {
    let cfg = VulkanBackendConfig::from_env();
    let backend = Arc::new(VulkanBackend::new(cfg).expect("init VulkanBackend"));
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe).expect("create ComDevice");

    // SAFETY: All COM calls follow verified Win64 console D3D12.X ABI.
    unsafe {
        let dev_vt = (*dev).vtbl;
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(8));
        let create_alloc: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(9));
        let create_cpso: extern "win64" fn(
            *mut ComDevice,
            *const D3D12ComputePipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(11));
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
        let create_dheap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(14));
        let create_uav: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *mut ComResource,
            *const D3D12UnorderedAccessViewDesc,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(19));
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
        let create_fence: extern "win64" fn(
            *mut ComDevice,
            u64,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(36));

        // 1. Create COMPUTE queue (type = 2)
        let mut queue_raw = ptr::null_mut();
        assert_eq!(
            create_queue(
                dev,
                &D3D12CommandQueueDesc {
                    r#type: 2,
                    ..Default::default()
                },
                ptr::null(),
                &mut queue_raw
            ),
            S_OK
        );
        let queue = queue_raw.cast::<ComCommandQueue>();
        let mut alloc_raw = ptr::null_mut();
        assert_eq!(create_alloc(dev, 2, ptr::null(), &mut alloc_raw), S_OK);
        let allocator = alloc_raw.cast::<ComCommandAllocator>();

        // 2. Root signature: param 0 = 2 root constants, param 1 = inline CBV, param 2 = UAV descriptor table
        #[repr(C)]
        struct Range1 {
            range_type: u32,
            num_descriptors: u32,
            base_shader_register: u32,
            register_space: u32,
            flags: u32,
            offset_from_start: u32,
        }
        let ranges = [Range1 {
            range_type: 1, // UAV
            num_descriptors: 1,
            base_shader_register: 0,
            register_space: 0,
            flags: 0,
            offset_from_start: 0,
        }];
        let rptr = ranges.as_ptr() as usize as u64;
        let rs_params = [
            D3D12RootParameterRaw {
                parameter_type: 1, // 32BIT_CONSTANTS (2 dwords -> s0..s1)
                _pad04: 0,
                data: [0, 0, 2, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 2, // CBV (2 dwords -> s2..s3)
                _pad04: 0,
                data: [1, 0, 0, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 0, // DESCRIPTOR_TABLE (UAV u0)
                _pad04: 0,
                data: [1, 0, rptr as u32, (rptr >> 32) as u32],
                shader_visibility: 0,
                _pad1c: 0,
            },
        ];
        let root_sig = serialize_and_create_root_sig(dev, &rs_params);

        // 3. Buffers: CBV, UAV, Readback
        let bdesc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut cbv_raw = ptr::null_mut();
        let mut uav_raw = ptr::null_mut();
        let mut rb_raw = ptr::null_mut();
        let hp = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &bdesc,
                0,
                ptr::null(),
                ptr::null(),
                &mut cbv_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &bdesc,
                0,
                ptr::null(),
                ptr::null(),
                &mut uav_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &bdesc,
                0,
                ptr::null(),
                ptr::null(),
                &mut rb_raw
            ),
            S_OK
        );
        let cbv = cbv_raw.cast::<ComResource>();
        let uav = uav_raw.cast::<ComResource>();
        let rb = rb_raw.cast::<ComResource>();

        let res_map: extern "win64" fn(
            *mut ComResource,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*(*cbv).vtbl.add(8));
        let res_unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*(*cbv).vtbl.add(9));
        let res_va: extern "win64" fn(*mut ComResource) -> u64 =
            std::mem::transmute(*(*cbv).vtbl.add(11));

        let mut p = ptr::null_mut();
        assert_eq!(res_map(cbv, 0, ptr::null(), &mut p), S_OK);
        ptr::write_volatile(p.cast::<u32>(), 0x0000_0055);
        res_unmap(cbv, 0, ptr::null());

        // 4. Shader-visible CBV_SRV_UAV descriptor heap + UAV view
        let mut dheap_raw = ptr::null_mut();
        assert_eq!(
            create_dheap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 0,
                    num_descriptors: 4,
                    flags: 1, // SHADER_VISIBLE
                    node_mask: 0,
                },
                ptr::null(),
                &mut dheap_raw
            ),
            S_OK
        );
        let dheap = dheap_raw.cast::<ComDescriptorHeap>();
        let get_cpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*(*dheap).vtbl.add(9));
        let get_gpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12GpuDescriptorHandle,
        ) -> *mut D3D12GpuDescriptorHandle = std::mem::transmute(*(*dheap).vtbl.add(10));
        let mut cpu_h = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        let mut gpu_h = MaybeUninit::<D3D12GpuDescriptorHandle>::uninit();
        get_cpu(dheap, cpu_h.as_mut_ptr());
        get_gpu(dheap, gpu_h.as_mut_ptr());
        let cpu_h = cpu_h.assume_init();
        let gpu_h = gpu_h.assume_init();
        create_uav(dev, uav, ptr::null_mut(), ptr::null(), cpu_h);

        // 5. Compute PSO
        let cs_bytes = compute_signature_cs_bytes();
        let mut cpso_desc: D3D12ComputePipelineStateDesc = std::mem::zeroed();
        cpso_desc.p_root_signature = root_sig.cast();
        cpso_desc.cs = D3D12ShaderBytecode {
            p_shader_bytecode: cs_bytes.as_ptr().cast(),
            bytecode_length: cs_bytes.len(),
        };
        let mut pso_raw = ptr::null_mut();
        assert_eq!(
            create_cpso(dev, &cpso_desc, ptr::null(), &mut pso_raw),
            S_OK
        );
        let pso = pso_raw.cast::<ComPipelineState>();

        // 6. Record compute list: SetDescriptorHeaps, SetComputeRootSignature, Constants, CBV, Table, Dispatch(1,1,1), CopyBufferRegion
        let mut list_raw = ptr::null_mut();
        assert_eq!(
            create_list(dev, 0, 2, allocator, pso, ptr::null(), &mut list_raw),
            S_OK
        );
        let list = list_raw.cast::<ComGraphicsCommandList>();
        let lvt = (*list).vtbl;
        let list_close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
            std::mem::transmute(*lvt.add(9));
        let list_dispatch: extern "win64" fn(*mut ComGraphicsCommandList, u32, u32, u32) =
            std::mem::transmute(*lvt.add(14));
        let list_copy_buf: extern "win64" fn(
            *mut ComGraphicsCommandList,
            *mut ComResource,
            u64,
            *mut ComResource,
            u64,
            u64,
        ) = std::mem::transmute(*lvt.add(15));
        let list_set_heaps: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            *const *mut ComDescriptorHeap,
        ) = std::mem::transmute(*lvt.add(32));
        let list_set_cmp_rs: extern "win64" fn(*mut ComGraphicsCommandList, *mut ComRootSignature) =
            std::mem::transmute(*lvt.add(33));
        let list_set_cmp_tbl: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            D3D12GpuDescriptorHandle,
        ) = std::mem::transmute(*lvt.add(35));
        let list_set_cmp_consts: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            u32,
            *const c_void,
            u32,
        ) = std::mem::transmute(*lvt.add(39));
        let list_set_cmp_cbv: extern "win64" fn(*mut ComGraphicsCommandList, u32, u64) =
            std::mem::transmute(*lvt.add(41));

        let heaps = [dheap];
        list_set_heaps(list, 1, heaps.as_ptr());
        list_set_cmp_rs(list, root_sig);
        let root_consts: [u32; 2] = [0xCAFE_0000, 7];
        list_set_cmp_consts(list, 0, 2, root_consts.as_ptr().cast(), 0);
        list_set_cmp_cbv(list, 1, res_va(cbv));
        list_set_cmp_tbl(list, 2, gpu_h);
        list_dispatch(list, 1, 1, 1);
        list_copy_buf(list, rb, 0, uav, 0, 64 * 4);
        assert_eq!(list_close(list), S_OK);

        // Execute + Fence Signal + Wait
        let mut fence_raw = ptr::null_mut();
        assert_eq!(create_fence(dev, 0, 0, ptr::null(), &mut fence_raw), S_OK);
        let fence = fence_raw.cast::<ComFence>();

        let qvt = (*queue).vtbl;
        let q_exec: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const *mut ComGraphicsCommandList,
        ) = std::mem::transmute(*qvt.add(10));
        let q_signal: extern "win64" fn(*mut ComCommandQueue, *mut ComFence, u64) -> i32 =
            std::mem::transmute(*qvt.add(14));
        let q_wait: extern "win64" fn(*mut ComCommandQueue, *mut ComFence, u64) -> i32 =
            std::mem::transmute(*qvt.add(15));
        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());
        assert_eq!(q_signal(queue, fence, 1), S_OK);
        assert_eq!(q_wait(queue, fence, 1), S_OK);

        // Read back 64 DWORDs from readback buffer and verify exact compute signature!
        let mut rb_ptr = ptr::null_mut();
        assert_eq!(res_map(rb, 0, ptr::null(), &mut rb_ptr), S_OK);
        let words = std::slice::from_raw_parts(rb_ptr.cast::<u32>(), 64);
        for (i, &w) in words.iter().enumerate() {
            assert_eq!(
                w,
                0xCAFE_0055u32 + (i as u32) * 7,
                "compute UAV word[{i}] mismatch"
            );
        }
        res_unmap(rb, 0, ptr::null());
    }
}

#[test]
fn copy_texture_region_roundtrips_buffer_to_texture_and_back() {
    let cfg = VulkanBackendConfig::from_env();
    let backend = Arc::new(VulkanBackend::new(cfg).expect("init VulkanBackend"));
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe).expect("create ComDevice");

    // SAFETY: All COM calls follow verified Win64 console D3D12.X ABI.
    unsafe {
        let dev_vt = (*dev).vtbl;
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(8));
        let create_alloc: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(9));
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
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
        let get_footprints: extern "win64" fn(
            *mut ComDevice,
            *const D3D12ResourceDesc,
            u32,
            u32,
            u64,
            *mut D3D12PlacedSubresourceFootprint,
            *mut u32,
            *mut u64,
            *mut u64,
        ) = std::mem::transmute(*dev_vt.add(38));

        let mut queue_raw = ptr::null_mut();
        assert_eq!(
            create_queue(
                dev,
                &D3D12CommandQueueDesc {
                    r#type: 3, // COPY queue
                    ..Default::default()
                },
                ptr::null(),
                &mut queue_raw
            ),
            S_OK
        );
        let queue = queue_raw.cast::<ComCommandQueue>();
        let mut alloc_raw = ptr::null_mut();
        assert_eq!(create_alloc(dev, 3, ptr::null(), &mut alloc_raw), S_OK);
        let allocator = alloc_raw.cast::<ComCommandAllocator>();

        let tex_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            ..Default::default()
        };
        let mut fp = D3D12PlacedSubresourceFootprint::default();
        let mut total_bytes = 0u64;
        get_footprints(
            dev,
            &tex_desc,
            0,
            1,
            0,
            &mut fp,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut total_bytes,
        );
        assert_eq!(fp.footprint.row_pitch, 256);
        assert_eq!(total_bytes, 64 * 256);

        let hp = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let buf_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: total_bytes,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut tex_raw = ptr::null_mut();
        let mut up_raw = ptr::null_mut();
        let mut rb_raw = ptr::null_mut();
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &tex_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut tex_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut up_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut rb_raw
            ),
            S_OK
        );
        let tex = tex_raw.cast::<ComResource>();
        let up = up_raw.cast::<ComResource>();
        let rb = rb_raw.cast::<ComResource>();

        let res_map: extern "win64" fn(
            *mut ComResource,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*(*up).vtbl.add(8));
        let res_unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*(*up).vtbl.add(9));

        let mut up_ptr = ptr::null_mut();
        assert_eq!(res_map(up, 0, ptr::null(), &mut up_ptr), S_OK);
        let up_slice = std::slice::from_raw_parts_mut(up_ptr.cast::<u8>(), total_bytes as usize);
        for (i, b) in up_slice.iter_mut().enumerate() {
            *b = ((i * 31 + 7) & 0xFF) as u8;
        }
        res_unmap(up, 0, ptr::null());

        let mut list_raw = ptr::null_mut();
        assert_eq!(
            create_list(
                dev,
                0,
                3,
                allocator,
                ptr::null_mut(),
                ptr::null(),
                &mut list_raw
            ),
            S_OK
        );
        let list = list_raw.cast::<ComGraphicsCommandList>();
        let lvt = (*list).vtbl;
        let list_close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
            std::mem::transmute(*lvt.add(9));
        let list_copy_tex: extern "win64" fn(
            *mut ComGraphicsCommandList,
            *const D3D12TextureCopyLocation,
            u32,
            u32,
            u32,
            *const D3D12TextureCopyLocation,
            *const c_void,
        ) = std::mem::transmute(*lvt.add(16));

        let loc_up = D3D12TextureCopyLocation {
            p_resource: up,
            r#type: 1,
            _pad12: 0,
            u: D3D12TextureCopyLocationUnion {
                placed_footprint: fp,
            },
        };
        let loc_tex = D3D12TextureCopyLocation {
            p_resource: tex,
            r#type: 0,
            _pad12: 0,
            u: D3D12TextureCopyLocationUnion {
                subresource_index: 0,
            },
        };
        let loc_rb = D3D12TextureCopyLocation {
            p_resource: rb,
            r#type: 1,
            _pad12: 0,
            u: D3D12TextureCopyLocationUnion {
                placed_footprint: fp,
            },
        };
        list_copy_tex(list, &loc_tex, 0, 0, 0, &loc_up, ptr::null());
        list_copy_tex(list, &loc_rb, 0, 0, 0, &loc_tex, ptr::null());
        assert_eq!(list_close(list), S_OK);

        let q_exec: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const *mut ComGraphicsCommandList,
        ) = std::mem::transmute(*(*queue).vtbl.add(10));
        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());

        let mut rb_ptr = ptr::null_mut();
        assert_eq!(res_map(rb, 0, ptr::null(), &mut rb_ptr), S_OK);
        let rb_slice = std::slice::from_raw_parts(rb_ptr.cast::<u8>(), total_bytes as usize);
        assert_eq!(rb_slice.len(), total_bytes as usize);
        for (i, &b) in rb_slice.iter().enumerate() {
            assert_eq!(b, ((i * 31 + 7) & 0xFF) as u8, "texel byte {i} mismatch");
        }
        res_unmap(rb, 0, ptr::null());
    }
}

#[test]
fn sampled_texture2d_srv_quad_renders_four_quadrant_colors() {
    // 1. Pass-through VS: Location 0 (`v0..v3`) = clip position, Location 1 (`v4..v7`) = UV.
    let mut vs_asm = asm::Asm::new();
    vs_asm.push2(asm::exp_pos0([0, 1, 2, 3], false));
    vs_asm.push2(asm::exp_param(0, [4, 5, 6, 7], true));
    vs_asm.push(asm::s_endpgm());
    let vs_bytes = words_to_bytes(vs_asm.words());

    // 2. Sampled PS: interpolate UV from param0 into (v0, v1), execute `image_sample_2d`, export `mrt0`.
    let mut ps_asm = asm::Asm::new();
    ps_asm.push(asm::v_interp_p1_f32(0, 0, 0, 0));
    ps_asm.push(asm::v_interp_p2_f32(0, 1, 0, 0));
    ps_asm.push(asm::v_interp_p1_f32(1, 0, 0, 1));
    ps_asm.push(asm::v_interp_p2_f32(1, 1, 0, 1));
    ps_asm.push2(asm::image_sample_2d(4, 0, 0, 4, 0xF));
    ps_asm.push2(asm::exp_mrt(0, [4, 5, 6, 7], false, true, true));
    ps_asm.push(asm::s_endpgm());
    let ps_bytes = words_to_bytes(ps_asm.words());

    let backend = Arc::new(
        VulkanBackend::new(VulkanBackendConfig {
            enable_validation: true,
            ..VulkanBackendConfig::from_env()
        })
        .expect("VulkanBackend::new"),
    );
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe)
        .expect("create_vulkan_com_device");

    // SAFETY: Valid COM objects and Win64 vtable indices matching `x2l_gpu_vk::com`.
    unsafe {
        let dvt = (*dev).vtbl;
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(8));
        let create_alloc: extern "win64" fn(
            *mut ComDevice,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(9));
        let create_pso: extern "win64" fn(
            *mut ComDevice,
            *const D3D12GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(10));
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(12));
        let create_desc_heap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(14));
        let create_srv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const D3D12ShaderResourceViewDesc,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dvt.add(18));
        let create_rtv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dvt.add(20));
        let create_sampler: extern "win64" fn(
            *mut ComDevice,
            *const D3D12SamplerDesc,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dvt.add(22));
        let create_res: extern "win64" fn(
            *mut ComDevice,
            *const D3D12HeapProperties,
            u32,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(27));
        let get_footprints: extern "win64" fn(
            *mut ComDevice,
            *const D3D12ResourceDesc,
            u32,
            u32,
            u64,
            *mut D3D12PlacedSubresourceFootprint,
            *mut u32,
            *mut u64,
            *mut u64,
        ) = std::mem::transmute(*dvt.add(38));

        let mut queue_raw = ptr::null_mut();
        let mut alloc_raw = ptr::null_mut();
        assert_eq!(
            create_queue(
                dev,
                &D3D12CommandQueueDesc::default(),
                ptr::null(),
                &mut queue_raw
            ),
            S_OK
        );
        assert_eq!(create_alloc(dev, 0, ptr::null(), &mut alloc_raw), S_OK);
        let queue = queue_raw.cast::<ComCommandQueue>();
        let allocator = alloc_raw.cast::<ComCommandAllocator>();

        // Create 64x64 RT and 4x4 sampled Texture2D (quadrants: TL=RED, TR=GREEN, BL=BLUE, BR=YELLOW).
        let hp = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let rt_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            ..Default::default()
        };
        let tex_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 4,
            height: 4,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            ..Default::default()
        };
        let mut fp = D3D12PlacedSubresourceFootprint::default();
        let mut tex_upload_bytes = 0u64;
        get_footprints(
            dev,
            &tex_desc,
            0,
            1,
            0,
            &mut fp,
            ptr::null_mut(),
            ptr::null_mut(),
            &mut tex_upload_bytes,
        );
        let up_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: tex_upload_bytes,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        // Fullscreen quad: 6 vertices of [pos: f32; 4, uv: f32; 4] (flip_y is true on Vulkan VS,
        // so D3D12 y=+1 maps to top of screen y_vk=-1 -> UV v=0 at top, v=1 at bottom).
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct QuadVert {
            pos: [f32; 4],
            uv: [f32; 4],
        }
        let quad: [QuadVert; 6] = [
            QuadVert {
                pos: [-1.0, 1.0, 0.5, 1.0],
                uv: [0.0, 0.0, 0.0, 1.0],
            },
            QuadVert {
                pos: [1.0, 1.0, 0.5, 1.0],
                uv: [1.0, 0.0, 0.0, 1.0],
            },
            QuadVert {
                pos: [1.0, -1.0, 0.5, 1.0],
                uv: [1.0, 1.0, 0.0, 1.0],
            },
            QuadVert {
                pos: [-1.0, 1.0, 0.5, 1.0],
                uv: [0.0, 0.0, 0.0, 1.0],
            },
            QuadVert {
                pos: [1.0, -1.0, 0.5, 1.0],
                uv: [1.0, 1.0, 0.0, 1.0],
            },
            QuadVert {
                pos: [-1.0, -1.0, 0.5, 1.0],
                uv: [0.0, 1.0, 0.0, 1.0],
            },
        ];
        let vb_bytes_len = std::mem::size_of_val(&quad) as u64;
        let vb_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: vb_bytes_len,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };

        let mut rt_raw = ptr::null_mut();
        let mut tex_raw = ptr::null_mut();
        let mut up_raw = ptr::null_mut();
        let mut vb_raw = ptr::null_mut();
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &rt_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut rt_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &tex_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut tex_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &up_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut up_raw
            ),
            S_OK
        );
        assert_eq!(
            create_res(
                dev,
                &hp,
                0,
                &vb_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut vb_raw
            ),
            S_OK
        );
        let rt = rt_raw.cast::<ComResource>();
        let tex = tex_raw.cast::<ComResource>();
        let up = up_raw.cast::<ComResource>();
        let vb = vb_raw.cast::<ComResource>();

        let res_map: extern "win64" fn(
            *mut ComResource,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*(*up).vtbl.add(8));
        let res_unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*(*up).vtbl.add(9));
        let res_gpu_va: extern "win64" fn(*mut ComResource) -> u64 =
            std::mem::transmute(*(*vb).vtbl.add(11));

        // Upload 4x4 4-quadrant RGBA8 image into upload buffer using `row_pitch = 256`.
        let mut up_ptr = ptr::null_mut();
        assert_eq!(res_map(up, 0, ptr::null(), &mut up_ptr), S_OK);
        let up_slice =
            std::slice::from_raw_parts_mut(up_ptr.cast::<u8>(), tex_upload_bytes as usize);
        up_slice.fill(0);
        for y in 0..4usize {
            for x in 0..4usize {
                let px: [u8; 4] = match (x >= 2, y >= 2) {
                    (false, false) => [255, 0, 0, 255],
                    (true, false) => [0, 255, 0, 255],
                    (false, true) => [0, 0, 255, 255],
                    (true, true) => [255, 255, 0, 255],
                };
                let off = y * (fp.footprint.row_pitch as usize) + x * 4;
                up_slice[off..off + 4].copy_from_slice(&px);
            }
        }
        res_unmap(up, 0, ptr::null());

        let mut vb_ptr = ptr::null_mut();
        assert_eq!(res_map(vb, 0, ptr::null(), &mut vb_ptr), S_OK);
        ptr::copy_nonoverlapping(
            quad.as_ptr().cast::<u8>(),
            vb_ptr.cast::<u8>(),
            vb_bytes_len as usize,
        );
        res_unmap(vb, 0, ptr::null());

        // Descriptor heaps: RTV, SRV, and Sampler.
        let mut rtv_heap_raw = ptr::null_mut();
        let mut srv_heap_raw = ptr::null_mut();
        let mut samp_heap_raw = ptr::null_mut();
        assert_eq!(
            create_desc_heap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 2,
                    num_descriptors: 1,
                    flags: 0,
                    node_mask: 0
                },
                ptr::null(),
                &mut rtv_heap_raw
            ),
            S_OK
        );
        assert_eq!(
            create_desc_heap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 0,
                    num_descriptors: 1,
                    flags: 1,
                    node_mask: 0
                },
                ptr::null(),
                &mut srv_heap_raw
            ),
            S_OK
        );
        assert_eq!(
            create_desc_heap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 1,
                    num_descriptors: 1,
                    flags: 1,
                    node_mask: 0
                },
                ptr::null(),
                &mut samp_heap_raw
            ),
            S_OK
        );
        let rtv_heap = rtv_heap_raw.cast::<ComDescriptorHeap>();
        let srv_heap = srv_heap_raw.cast::<ComDescriptorHeap>();
        let samp_heap = samp_heap_raw.cast::<ComDescriptorHeap>();

        let get_cpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*(*rtv_heap).vtbl.add(9));
        let get_gpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12GpuDescriptorHandle,
        ) -> *mut D3D12GpuDescriptorHandle = std::mem::transmute(*(*srv_heap).vtbl.add(10));

        let mut rtv_cpu = D3D12CpuDescriptorHandle::default();
        let mut srv_cpu = D3D12CpuDescriptorHandle::default();
        let mut srv_gpu = D3D12GpuDescriptorHandle::default();
        let mut samp_cpu = D3D12CpuDescriptorHandle::default();
        let mut samp_gpu = D3D12GpuDescriptorHandle::default();
        get_cpu(rtv_heap, &mut rtv_cpu);
        get_cpu(srv_heap, &mut srv_cpu);
        get_gpu(srv_heap, &mut srv_gpu);
        get_cpu(samp_heap, &mut samp_cpu);
        get_gpu(samp_heap, &mut samp_gpu);

        create_rtv(dev, rt, ptr::null(), rtv_cpu);
        let mut srv_desc: D3D12ShaderResourceViewDesc = std::mem::zeroed();
        srv_desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv_desc.view_dimension = 4; // TEXTURE2D
        srv_desc.shader_4_component_mapping = 0x1688;
        create_srv(dev, tex, &srv_desc, srv_cpu);
        let samp_desc = D3D12SamplerDesc {
            filter: 0,    // MIN_MAG_MIP_POINT
            address_u: 3, // CLAMP
            address_v: 3,
            address_w: 3,
            max_lod: f32::MAX,
            ..Default::default()
        };
        create_sampler(dev, &samp_desc, samp_cpu);

        let elems = [
            D3D12InputElementDesc {
                semantic_name: c"POSITION".as_ptr(),
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 0,
                input_slot_class: 0,
                instance_data_step_rate: 0,
            },
            D3D12InputElementDesc {
                semantic_name: c"TEXCOORD".as_ptr(),
                semantic_index: 0,
                format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                input_slot: 0,
                aligned_byte_offset: 16,
                input_slot_class: 0,
                instance_data_step_rate: 0,
            },
        ];
        let mut pso_desc: D3D12GraphicsPipelineStateDesc = std::mem::zeroed();
        pso_desc.vs = D3D12ShaderBytecode {
            p_shader_bytecode: vs_bytes.as_ptr().cast(),
            bytecode_length: vs_bytes.len(),
        };
        pso_desc.ps = D3D12ShaderBytecode {
            p_shader_bytecode: ps_bytes.as_ptr().cast(),
            bytecode_length: ps_bytes.len(),
        };
        pso_desc.input_layout = D3D12InputLayoutDesc {
            p_input_element_descs: elems.as_ptr().cast(),
            num_elements: 2,
        };
        pso_desc.num_render_targets = 1;
        pso_desc.rtv_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.sample_desc = DxgiSampleDesc {
            count: 1,
            quality: 0,
        };
        let mut pso_raw = ptr::null_mut();
        assert_eq!(create_pso(dev, &pso_desc, ptr::null(), &mut pso_raw), S_OK);
        let pso = pso_raw.cast::<ComPipelineState>();

        let mut list_raw = ptr::null_mut();
        assert_eq!(
            create_list(dev, 0, 0, allocator, pso, ptr::null(), &mut list_raw),
            S_OK
        );
        let list = list_raw.cast::<ComGraphicsCommandList>();
        let lvt = (*list).vtbl;
        let list_close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
            std::mem::transmute(*lvt.add(9));
        let list_draw: extern "win64" fn(*mut ComGraphicsCommandList, u32, u32, u32, u32) =
            std::mem::transmute(*lvt.add(12));
        let list_copy_tex: extern "win64" fn(
            *mut ComGraphicsCommandList,
            *const D3D12TextureCopyLocation,
            u32,
            u32,
            u32,
            *const D3D12TextureCopyLocation,
            *const c_void,
        ) = std::mem::transmute(*lvt.add(16));
        let list_set_vp: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const D3D12Viewport) =
            std::mem::transmute(*lvt.add(21));
        let list_set_sc: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const D3D12Rect) =
            std::mem::transmute(*lvt.add(22));
        let list_set_gfx_tbl: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            D3D12GpuDescriptorHandle,
        ) = std::mem::transmute(*lvt.add(36));
        let list_set_vb: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            u32,
            *const D3D12VertexBufferView,
        ) = std::mem::transmute(*lvt.add(48));
        let list_om_set_rt: extern "win64" fn(
            *mut ComGraphicsCommandList,
            u32,
            *const D3D12CpuDescriptorHandle,
            i32,
            *const D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*lvt.add(50));
        let list_clear_rtv: extern "win64" fn(
            *mut ComGraphicsCommandList,
            D3D12CpuDescriptorHandle,
            *const f32,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*lvt.add(52));

        let loc_up = D3D12TextureCopyLocation {
            p_resource: up,
            r#type: 1,
            _pad12: 0,
            u: D3D12TextureCopyLocationUnion {
                placed_footprint: fp,
            },
        };
        let loc_tex = D3D12TextureCopyLocation {
            p_resource: tex,
            r#type: 0,
            _pad12: 0,
            u: D3D12TextureCopyLocationUnion {
                subresource_index: 0,
            },
        };
        list_copy_tex(list, &loc_tex, 0, 0, 0, &loc_up, ptr::null());
        list_clear_rtv(list, rtv_cpu, [0.0, 0.0, 0.0, 1.0].as_ptr(), 0, ptr::null());
        list_set_vp(
            list,
            1,
            &D3D12Viewport {
                top_left_x: 0.0,
                top_left_y: 0.0,
                width: 64.0,
                height: 64.0,
                min_depth: 0.0,
                max_depth: 1.0,
            },
        );
        list_set_sc(
            list,
            1,
            &D3D12Rect {
                left: 0,
                top: 0,
                right: 64,
                bottom: 64,
            },
        );
        list_om_set_rt(list, 1, &rtv_cpu, 1, ptr::null());
        list_set_gfx_tbl(list, 0, srv_gpu);
        list_set_gfx_tbl(list, 1, samp_gpu);
        list_set_vb(
            list,
            0,
            1,
            &D3D12VertexBufferView {
                buffer_location: res_gpu_va(vb),
                size_in_bytes: vb_bytes_len as u32,
                stride_in_bytes: std::mem::size_of::<QuadVert>() as u32,
            },
        );
        list_draw(list, 6, 1, 0, 0);
        assert_eq!(list_close(list), S_OK);

        let q_exec: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const *mut ComGraphicsCommandList,
        ) = std::mem::transmute(*(*queue).vtbl.add(10));
        let q_present: extern "win64" fn(
            *mut ComCommandQueue,
            u32,
            *const D3D12XboxPresentPlaneParameters,
            *const c_void,
        ) -> i32 = std::mem::transmute(*(*queue).vtbl.add(30));
        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());
        let rt_list = [rt];
        let plane = D3D12XboxPresentPlaneParameters {
            token: 0,
            resource_count: 1,
            _pad0: 0,
            pp_resources: rt_list.as_ptr(),
            _reserved: [0; 6],
        };
        assert_eq!(q_present(queue, 1, &plane, ptr::null()), S_OK);

        let frame = backend.last_presented_frame().expect("presented frame");
        assert_eq!(frame.pixel(16, 16), Some([255, 0, 0, 255]), "top-left RED");
        assert_eq!(
            frame.pixel(48, 16),
            Some([0, 255, 0, 255]),
            "top-right GREEN"
        );
        assert_eq!(
            frame.pixel(16, 48),
            Some([0, 0, 255, 255]),
            "bottom-left BLUE"
        );
        assert_eq!(
            frame.pixel(48, 48),
            Some([255, 255, 0, 255]),
            "bottom-right YELLOW"
        );
    }
}
