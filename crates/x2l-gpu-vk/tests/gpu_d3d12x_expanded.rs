// Clean-room end-to-end headless Vulkan 1.3 + Win64 COM tests for M3.3 expanded D3D12.X surface:
// - Indexed draws + InputLayout + root 32-bit constants + inline CBV
// - D32_FLOAT depth buffer + ClearDepthStencilView + depth occlusion
// - Placed resources (CreateHeap + CreatePlacedResource) + CopyBufferRegion + CopyTextureRegion
// - Compute pipeline (CreateComputePipelineState + Dispatch + UAV descriptor table + readback)
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

#![cfg(feature = "gpu-tests")]

use std::cell::Cell;
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use x2l_gpu_vk::com::{
    ComCommandAllocator, ComCommandQueue, ComDescriptorHeap, ComDevice, ComFence,
    ComGraphicsCommandList, ComHeap, ComPipelineState, ComResource, ComRootSignature,
    D3D12CommandListXDesc, D3D12CommandQueueDesc, D3D12CommandSignatureDesc,
    D3D12ConstantBufferViewDesc, D3D12CpuDescriptorHandle, D3D12DepthStencilViewDesc,
    D3D12DescriptorHeapDesc, D3D12GpuDescriptorHandle, D3D12GraphicsPipelineStateDesc,
    D3D12HeapDesc, D3D12HeapProperties, D3D12IndexBufferView, D3D12IndirectArgumentDesc,
    D3D12InputElementDesc, D3D12InputLayoutDesc, D3D12PlacedSubresourceFootprint,
    D3D12QueryHeapDesc, D3D12Rect, D3D12ResourceAllocationInfo, D3D12ResourceAllocationInfo1,
    D3D12ResourceDesc, D3D12ResourceDesc1, D3D12SamplerDesc, D3D12ShaderBytecode,
    D3D12ShaderResourceViewDesc, D3D12TextureCopyLocation, D3D12TextureCopyLocationUnion,
    D3D12UnorderedAccessViewDesc, D3D12VertexBufferView, D3D12Viewport,
    D3D12XboxPresentPlaneParameters, DxgiSampleDesc, E_INVALIDARG, E_NOINTERFACE, S_OK,
    create_vulkan_com_device,
};
use x2l_gpu_vk::d3d12x_ext::{
    ComD3DBlob, D3D12PipelineStateStreamDesc, D3D12RootParameterRaw,
    D3D12VersionedRootSignatureDesc, serialize_versioned_root_signature,
};
use x2l_gpu_vk::decode_d3d12_vtable_probe;
use x2l_gpu_vk::vulkan::{
    D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_BC7_UNORM,
    DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R16_UINT,
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT, DescriptorEntry, VulkanBackend,
    VulkanBackendConfig,
};
use x2l_rdna2_spirv::encode as asm;

type CopyDescriptorsSimple =
    extern "win64" fn(*mut ComDevice, u32, D3D12CpuDescriptorHandle, D3D12CpuDescriptorHandle, u32);
type CopyDescriptorRanges = extern "win64" fn(
    *mut ComDevice,
    u32,
    *const D3D12CpuDescriptorHandle,
    *const u32,
    u32,
    *const D3D12CpuDescriptorHandle,
    *const u32,
    u32,
);

thread_local! {
    static PROBE_HIT: Cell<Option<(u32, u64)>> = const { Cell::new(None) };
}

extern "win64" fn test_probe(slot: u32, rip: u64) -> u64 {
    PROBE_HIT.set(Some((slot, rip)));
    0
}

#[repr(C, align(8))]
struct StreamSubobject<T> {
    kind: u32,
    payload: T,
}

fn subobject<T>(kind: u32, payload: T) -> StreamSubobject<T> {
    StreamSubobject { kind, payload }
}

#[repr(C)]
struct StreamRenderTargetFormats {
    formats: [u32; 8],
    count: u32,
}

fn stream_desc<T>(stream: &T) -> D3D12PipelineStateStreamDesc {
    D3D12PipelineStateStreamDesc {
        size_in_bytes: std::mem::size_of_val(stream),
        stream: ptr::from_ref(stream).cast(),
    }
}

fn shader_bytecode(bytes: &[u8]) -> D3D12ShaderBytecode {
    D3D12ShaderBytecode {
        p_shader_bytecode: bytes.as_ptr().cast(),
        bytecode_length: bytes.len(),
    }
}

type CreateStreamPso = extern "win64" fn(
    *mut ComDevice,
    *const D3D12PipelineStateStreamDesc,
    *const c_void,
    *mut *mut c_void,
) -> i32;
type CreateCommandListX = extern "win64" fn(
    *mut ComDevice,
    *const D3D12CommandListXDesc,
    *mut ComCommandAllocator,
    *mut ComPipelineState,
    *const c_void,
    *mut *mut c_void,
) -> i32;

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
/// - Root param 1 (SRV table): word 0 = offset
/// - Root param 2 (UAV table): writes `s0 + s8 + v0 * s1` to byte offset `v0 * 4`.
fn compute_signature_cs_bytes() -> Vec<u8> {
    let mut a = asm::Asm::new();
    // s8 = SRV[0]
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
        let create_pso: CreateStreamPso = std::mem::transmute(*dev_vt.add(86));
        let create_list: CreateCommandListX = std::mem::transmute(*dev_vt.add(73));
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
                    num_descriptors: 4096,
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
        let mut dsv_h = dsv_h.assume_init();
        create_rtv(dev, rt, ptr::null(), rtv_h);
        dsv_h.ptr += 4095 * 32;
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
        #[repr(C)]
        struct GraphicsStream {
            root: StreamSubobject<*const c_void>,
            vs: StreamSubobject<D3D12ShaderBytecode>,
            ps: StreamSubobject<D3D12ShaderBytecode>,
            input: StreamSubobject<D3D12InputLayoutDesc>,
            topology: StreamSubobject<u32>,
            rtvs: StreamSubobject<StreamRenderTargetFormats>,
            dsv: StreamSubobject<u32>,
        }
        let stream = GraphicsStream {
            root: subobject(0, root_sig.cast()),
            vs: subobject(1, shader_bytecode(&vs_bytes)),
            ps: subobject(2, shader_bytecode(&ps_bytes)),
            input: subobject(
                12,
                D3D12InputLayoutDesc {
                    p_input_element_descs: input_elems.as_ptr().cast(),
                    num_elements: 1,
                },
            ),
            topology: subobject(14, 3),
            rtvs: subobject(
                15,
                StreamRenderTargetFormats {
                    formats: [DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0, 0, 0, 0, 0, 0],
                    count: 1,
                },
            ),
            dsv: subobject(16, DXGI_FORMAT_D32_FLOAT),
        };
        let mut pso_raw = ptr::null_mut();
        assert_eq!(
            create_pso(dev, &stream_desc(&stream), ptr::null(), &mut pso_raw),
            S_OK
        );
        assert!(!pso_raw.is_null());
        let pso = pso_raw.cast::<ComPipelineState>();
        let params = D3D12CommandListXDesc {
            list_type: 0,
            flags: 0,
            node_mask: 1,
            barrier_batch: 0,
        };
        let mut list_raw = ptr::null_mut();
        assert_eq!(
            create_list(dev, &params, allocator, pso, ptr::null(), &mut list_raw),
            S_OK
        );
        assert!(!list_raw.is_null());
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
fn compute_dispatch_uses_copied_srv_and_uav_from_large_descriptor_ring() {
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
        let create_cpso: CreateStreamPso = std::mem::transmute(*dev_vt.add(86));
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

        // 2. Root signature: param 0 = 2 root constants, param 1 = SRV descriptor table, param 2 = UAV descriptor table
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
        let srv_ranges = [Range1 {
            range_type: 0,
            num_descriptors: 1,
            base_shader_register: 0,
            register_space: 0,
            flags: 0,
            offset_from_start: 0,
        }];
        let rptr = ranges.as_ptr() as usize as u64;
        let srv_ptr = srv_ranges.as_ptr() as usize as u64;
        let rs_params = [
            D3D12RootParameterRaw {
                parameter_type: 1, // 32BIT_CONSTANTS (2 dwords -> s0..s1)
                _pad04: 0,
                data: [0, 0, 2, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 0, // DESCRIPTOR_TABLE (SRV t0)
                _pad04: 0,
                data: [1, 0, srv_ptr as u32, (srv_ptr >> 32) as u32],
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

        // 3. Buffers for the SRV, UAV and readback.
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

        // 4. Stage a ring in CPU descriptors, then copy it to the end of a 1M-entry GPU heap.
        const HEAP_COUNT: u32 = 1_000_000;
        const RING_COUNT: u32 = 4096;
        let increment: extern "win64" fn(*mut ComDevice, u32) -> u32 =
            std::mem::transmute(*dev_vt.add(15));
        let copy_ranges: CopyDescriptorRanges = std::mem::transmute(*dev_vt.add(23));
        let copy_simple: CopyDescriptorsSimple = std::mem::transmute(*dev_vt.add(24));
        let create_cbv: extern "win64" fn(
            *mut ComDevice,
            *const D3D12ConstantBufferViewDesc,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(17));
        let inc = increment(dev, 0) as usize;
        assert_eq!(inc, 32);
        let mut dheap_raw = ptr::null_mut();
        let mut staging_raw = ptr::null_mut();
        for (r#type, num_descriptors, flags, out) in [
            (0, HEAP_COUNT, 1, &mut dheap_raw),
            (4, RING_COUNT, 0, &mut staging_raw),
        ] {
            assert_eq!(
                create_dheap(
                    dev,
                    &D3D12DescriptorHeapDesc {
                        r#type,
                        num_descriptors,
                        flags,
                        node_mask: 0,
                    },
                    ptr::null(),
                    out
                ),
                S_OK
            );
        }
        let dheap = dheap_raw.cast::<ComDescriptorHeap>();
        let staging = staging_raw.cast::<ComDescriptorHeap>();
        let get_cpu: extern "win64" fn(
            *mut ComDescriptorHeap,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*(*dheap).vtbl.add(9));
        let get_cpu_reversed: extern "win64" fn(
            *mut D3D12CpuDescriptorHandle,
            *mut ComDescriptorHeap,
        ) -> *mut D3D12CpuDescriptorHandle = std::mem::transmute(*(*dheap).vtbl.add(9));
        let get_gpu_reversed: extern "win64" fn(
            *mut D3D12GpuDescriptorHandle,
            *mut ComDescriptorHeap,
        ) -> *mut D3D12GpuDescriptorHandle = std::mem::transmute(*(*dheap).vtbl.add(10));
        let mut cpu = D3D12CpuDescriptorHandle::default();
        let mut src = D3D12CpuDescriptorHandle::default();
        let mut gpu = D3D12GpuDescriptorHandle::default();
        let dst_ret = get_cpu_reversed(&mut cpu, dheap);
        assert_eq!(dst_ret, &raw mut cpu);
        get_cpu(staging, &mut src);
        let gpu_ret = get_gpu_reversed(&mut gpu, dheap);
        assert_eq!(gpu_ret, &raw mut gpu);
        assert!(src.ptr > cpu.ptr + HEAP_COUNT as usize * inc);
        assert_ne!(gpu.ptr, cpu.ptr as u64);
        let mut staging_gpu = D3D12GpuDescriptorHandle { ptr: u64::MAX };
        get_gpu_reversed(&mut staging_gpu, staging);
        assert_eq!(staging_gpu.ptr, 0);

        let at = |handle: D3D12CpuDescriptorHandle, index| D3D12CpuDescriptorHandle {
            ptr: handle.ptr + index as usize * inc,
        };
        let cbv_desc = D3D12ConstantBufferViewDesc {
            buffer_location: res_va(cbv),
            size_in_bytes: 256,
            _pad12: 0,
        };
        for index in 0..RING_COUNT {
            create_cbv(dev, &cbv_desc, at(src, index));
        }
        create_uav(dev, uav, ptr::null_mut(), ptr::null(), at(src, 0));
        let create_srv: extern "win64" fn(
            *mut ComDevice,
            *mut ComResource,
            *const D3D12ShaderResourceViewDesc,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(18));
        create_srv(dev, cbv, ptr::null(), at(src, 2));

        let ring_index = HEAP_COUNT - RING_COUNT;
        let ring = at(cpu, ring_index);
        let copied = at(cpu, ring_index - RING_COUNT);
        copy_simple(dev, RING_COUNT, copied, src, 4);
        let dst_ranges = [ring, at(ring, 1024)];
        let src_ranges = [copied, at(copied, 2048)];
        copy_ranges(
            dev,
            2,
            dst_ranges.as_ptr(),
            [1024, 3072].as_ptr(),
            2,
            src_ranges.as_ptr(),
            [2048, 2048].as_ptr(),
            0,
        );
        assert!(matches!(
            backend.resolve_descriptor(ring.ptr),
            Some(DescriptorEntry::Uav { .. })
        ));
        assert!(matches!(
            backend.resolve_descriptor(at(cpu, HEAP_COUNT - 1).ptr),
            Some(DescriptorEntry::Cbv { gpu_va, .. }) if gpu_va == cbv_desc.buffer_location
        ));

        // Shift forward through an overlapping range. The final range then copies the original
        // UAV and SRV into the final three slots using a null source size array.
        copy_simple(dev, RING_COUNT - 1, at(ring, 1), ring, 0);
        assert!(matches!(
            backend.resolve_descriptor(at(ring, 1).ptr),
            Some(DescriptorEntry::Uav { .. })
        ));
        assert!(matches!(
            backend.resolve_descriptor(at(ring, 2).ptr),
            Some(DescriptorEntry::Cbv { .. })
        ));
        let last = at(cpu, HEAP_COUNT - 1);
        let source_ranges = [at(copied, 0), at(copied, 2), at(copied, 0)];
        let destination_ranges = [at(ring, RING_COUNT - 3), last];
        copy_ranges(
            dev,
            2,
            destination_ranges.as_ptr(),
            [2, 1].as_ptr(),
            3,
            source_ranges.as_ptr(),
            ptr::null(),
            0,
        );
        let final_offset = u64::from(HEAP_COUNT - 1) * inc as u64;
        let gpu_h = D3D12GpuDescriptorHandle {
            ptr: gpu.ptr + final_offset,
        };
        assert!(matches!(
            backend.resolve_descriptor(gpu_h.ptr as usize),
            Some(DescriptorEntry::Uav { .. })
        ));

        // Reject invalid ranges and type mismatches without changing any destination descriptor.
        copy_simple(dev, 2, last, at(src, 1), 0);
        copy_simple(
            dev,
            1,
            last,
            D3D12CpuDescriptorHandle { ptr: src.ptr + 1 },
            0,
        );
        copy_simple(dev, 1, last, at(src, 1), 1);
        assert!(matches!(
            backend.resolve_descriptor(gpu_h.ptr as usize),
            Some(DescriptorEntry::Uav { .. })
        ));
        assert!(
            backend
                .resolve_descriptor(at(cpu, HEAP_COUNT).ptr)
                .is_none()
        );

        let srv_gpu = D3D12GpuDescriptorHandle {
            ptr: gpu_h.ptr - inc as u64,
        };
        assert!(matches!(
            backend.resolve_descriptor(srv_gpu.ptr as usize),
            Some(DescriptorEntry::Srv { .. })
        ));

        // 5. Compute PSO
        let cs_bytes = compute_signature_cs_bytes();
        #[repr(C)]
        struct ComputeStream {
            cs: StreamSubobject<D3D12ShaderBytecode>,
            root: StreamSubobject<*const c_void>,
        }
        let stream = ComputeStream {
            cs: subobject(6, shader_bytecode(&cs_bytes)),
            root: subobject(0, root_sig.cast()),
        };
        let mut pso_raw = ptr::null_mut();
        assert_eq!(
            create_cpso(dev, &stream_desc(&stream), ptr::null(), &mut pso_raw),
            S_OK
        );
        assert!(!pso_raw.is_null());
        let pso = pso_raw.cast::<ComPipelineState>();

        // 6. Record compute list: SetDescriptorHeaps, SetComputeRootSignature, Constants, SRV and UAV tables, Dispatch(1,1,1), CopyBufferRegion
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

        let heaps = [dheap];
        list_set_heaps(list, 1, heaps.as_ptr());
        list_set_cmp_rs(list, root_sig);
        let root_consts: [u32; 2] = [0xCAFE_0000, 7];
        list_set_cmp_consts(list, 0, 2, root_consts.as_ptr().cast(), 0);
        list_set_cmp_tbl(list, 1, srv_gpu);
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
        assert_eq!(release_object(staging_raw), 0);
        assert_eq!(release_object(dheap_raw), 0);
        assert!(backend.resolve_descriptor(gpu_h.ptr as usize).is_none());
        for object in [
            cbv_raw,
            uav_raw,
            rb_raw,
            fence_raw,
            list_raw,
            pso_raw,
            root_sig.cast(),
            alloc_raw,
            queue_raw,
            dev.cast(),
        ] {
            assert_eq!(release_object(object), 0);
        }
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
                    num_descriptors: 2048,
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
                    num_descriptors: 8192,
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
                    num_descriptors: 2048,
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

        rtv_cpu.ptr += 2047 * 32;
        samp_cpu.ptr += 2047 * 16;
        samp_gpu.ptr += 2047 * 16;
        create_rtv(dev, rt, ptr::null(), rtv_cpu);
        let mut srv_desc: D3D12ShaderResourceViewDesc = std::mem::zeroed();
        srv_desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv_desc.view_dimension = 4; // TEXTURE2D
        srv_desc.shader_4_component_mapping = 0x1688;
        create_srv(dev, tex, &srv_desc, srv_cpu);
        let copy_simple: CopyDescriptorsSimple = std::mem::transmute(*dvt.add(24));
        let copied_srv = D3D12CpuDescriptorHandle {
            ptr: srv_cpu.ptr + 4096 * 32,
        };
        copy_simple(dev, 4096, copied_srv, srv_cpu, 0);
        srv_gpu.ptr += 4096 * 32;
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

#[test]
fn audited_id3d12_device_x_console_slots_and_null_device_safety() {
    use x2l_gpu_vk::com::{D3D12Luid, D3D12ResourceAllocationInfo};
    use x2l_gpu_vk::{build_null_device_vtable, create_null_device};

    let backend = Arc::new(
        VulkanBackend::new(VulkanBackendConfig {
            enable_validation: true,
            ..VulkanBackendConfig::from_env()
        })
        .expect("VulkanBackend::new"),
    );
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe)
        .expect("create_vulkan_com_device");

    // SAFETY: Exercises verified Win64 console ID3D12DeviceX slots 7, 8, 13, 15, 25, 26, 36, 37,
    // 38, 43, 48, 78..=81 and queue GetTimestampFrequency (16, 18).
    unsafe {
        let dvt = (*dev).vtbl;
        let get_node_count: extern "win64" fn(*mut ComDevice) -> u32 =
            std::mem::transmute(*dvt.add(7));
        let create_q: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(8));
        let check_feat: extern "win64" fn(*mut ComDevice, u32, *mut c_void, u32) -> i32 =
            std::mem::transmute(*dvt.add(13));
        let get_inc: extern "win64" fn(*mut ComDevice, u32) -> u32 =
            std::mem::transmute(*dvt.add(15));
        let get_alloc_this_first: extern "win64" fn(
            *mut ComDevice,
            *mut D3D12ResourceAllocationInfo,
            u32,
            u32,
            *const D3D12ResourceDesc,
        ) -> *mut D3D12ResourceAllocationInfo = std::mem::transmute(*dvt.add(25));
        let get_alloc_ret_first: extern "win64" fn(
            *mut D3D12ResourceAllocationInfo,
            *mut ComDevice,
            u32,
            u32,
            *const D3D12ResourceDesc,
        ) -> *mut D3D12ResourceAllocationInfo = std::mem::transmute(*dvt.add(25));
        let get_custom_this_first: extern "win64" fn(
            *mut ComDevice,
            *mut D3D12HeapProperties,
            u32,
            u32,
        ) -> *mut D3D12HeapProperties = std::mem::transmute(*dvt.add(26));
        let get_custom_ret_first: extern "win64" fn(
            *mut D3D12HeapProperties,
            *mut ComDevice,
            u32,
            u32,
        ) -> *mut D3D12HeapProperties = std::mem::transmute(*dvt.add(26));
        let create_fence: extern "win64" fn(
            *mut ComDevice,
            u64,
            u32,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(36));
        let removed_reason: extern "win64" fn(*mut ComDevice) -> i32 =
            std::mem::transmute(*dvt.add(37));
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
        let get_luid_this_first: extern "win64" fn(
            *mut ComDevice,
            *mut D3D12Luid,
        ) -> *mut D3D12Luid = std::mem::transmute(*dvt.add(43));
        let get_luid_ret_first: extern "win64" fn(
            *mut D3D12Luid,
            *mut ComDevice,
        ) -> *mut D3D12Luid = std::mem::transmute(*dvt.add(43));
        let create_q_x: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dvt.add(48));
        let set_frame_int: extern "win64" fn(*mut ComDevice, *mut c_void, u32, u32, u32) -> i32 =
            std::mem::transmute(*dvt.add(78));
        let sched_frame: extern "win64" fn(*mut ComDevice, u32, u32, *const c_void, u32) -> i32 =
            std::mem::transmute(*dvt.add(79));
        let wait_frame: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *const c_void,
            u32,
            *mut u64,
        ) -> i32 = std::mem::transmute(*dvt.add(80));
        let dev_rel: extern "win64" fn(*mut ComDevice) -> u32 = std::mem::transmute(*dvt.add(2));

        assert_eq!(get_node_count(dev), 1);
        assert_eq!(get_inc(dev, 0), 32);
        assert_eq!(
            get_inc(dev, 1),
            16,
            "sampler heap increment must be 16 bytes"
        );
        assert_eq!(removed_reason(dev), S_OK);

        let mut arch = [0u32; 4];
        assert_eq!(check_feat(dev, 1, arch.as_mut_ptr().cast(), 16), S_OK);
        assert_eq!(arch[2], 1, "UMA must be TRUE");
        assert_eq!(arch[3], 1, "CacheCoherentUMA must be TRUE");
        let mut sm = 0x67u32;
        assert_eq!(check_feat(dev, 7, (&mut sm as *mut u32).cast(), 4), S_OK);
        assert_eq!(sm, 0x66);

        let dma_desc = D3D12CommandQueueDesc {
            r#type: 0x10,
            ..Default::default()
        };
        let mut q_dma = ptr::null_mut();
        assert_eq!(create_q(dev, &dma_desc, ptr::null(), &mut q_dma), S_OK);
        assert!(!q_dma.is_null());
        let mut q_x = ptr::null_mut();
        assert_eq!(
            create_q_x(
                dev,
                &D3D12CommandQueueDesc::default(),
                ptr::null(),
                &mut q_x
            ),
            S_OK
        );
        assert!(!q_x.is_null());

        let qvt = (*q_x.cast::<ComCommandQueue>()).vtbl;
        let get_ts16: extern "win64" fn(*mut ComCommandQueue, *mut u64) -> i32 =
            std::mem::transmute(*qvt.add(16));
        let get_ts18: extern "win64" fn(*mut ComCommandQueue, *mut u64) -> i32 =
            std::mem::transmute(*qvt.add(18));
        let q_rel: extern "win64" fn(*mut ComCommandQueue) -> u32 =
            std::mem::transmute(*qvt.add(2));
        let mut freq16 = 0u64;
        let mut freq18 = 0u64;
        assert_eq!(get_ts16(q_x.cast(), &mut freq16), S_OK);
        assert_eq!(get_ts18(q_x.cast(), &mut freq18), S_OK);
        assert_eq!(freq16, 100_000_000);
        assert_eq!(freq18, 100_000_000);
        assert_eq!(q_rel(q_x.cast()), 0);
        assert_eq!(q_rel(q_dma.cast()), 0);

        let buf_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut ai1 = D3D12ResourceAllocationInfo::default();
        let mut ai2 = D3D12ResourceAllocationInfo::default();
        assert_eq!(
            get_alloc_this_first(dev, &mut ai1, 0, 1, &buf_desc),
            &mut ai1 as *mut _
        );
        assert_eq!(
            get_alloc_ret_first(&mut ai2, dev, 0, 1, &buf_desc),
            &mut ai2 as *mut _
        );
        assert_eq!(ai1.size_in_bytes, 65536);
        assert_eq!(ai2.size_in_bytes, 65536);

        let mut hp1 = D3D12HeapProperties::default();
        let mut hp2 = D3D12HeapProperties::default();
        assert_eq!(
            get_custom_this_first(dev, &mut hp1, 0, 3),
            &mut hp1 as *mut _
        );
        assert_eq!(
            get_custom_ret_first(&mut hp2, dev, 0, 2),
            &mut hp2 as *mut _
        );
        assert_eq!(hp1.r#type, 4);
        assert_eq!(hp1.cpu_page_property, 2, "READBACK -> WRITE_BACK");
        assert_eq!(hp2.cpu_page_property, 1, "UPLOAD -> WRITE_COMBINE");

        let mut l1 = D3D12Luid::default();
        let mut l2 = D3D12Luid::default();
        assert_eq!(get_luid_this_first(dev, &mut l1), &mut l1 as *mut _);
        assert_eq!(get_luid_ret_first(&mut l2, dev), &mut l2 as *mut _);
        assert_eq!(l1, l2);
        assert_ne!(l1.low_part, 0);

        let mut fence = ptr::null_mut();
        assert_eq!(create_fence(dev, 42, 0, ptr::null(), &mut fence), S_OK);
        assert!(!fence.is_null());
        let fvt = (*fence.cast::<ComFence>()).vtbl;
        let f_val: extern "win64" fn(*mut ComFence) -> u64 = std::mem::transmute(*fvt.add(8));
        let f_rel: extern "win64" fn(*mut ComFence) -> u32 = std::mem::transmute(*fvt.add(2));
        assert_eq!(f_val(fence.cast()), 42);
        assert_eq!(f_rel(fence.cast()), 0);

        let mip_tex = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 8,
            height: 8,
            depth_or_array_size: 1,
            mip_levels: 2,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ..Default::default()
        };
        let mut fps = [D3D12PlacedSubresourceFootprint::default(); 2];
        let mut rows = [0u32; 2];
        let mut total = 0u64;
        get_footprints(
            dev,
            &mip_tex,
            0,
            2,
            0,
            fps.as_mut_ptr(),
            rows.as_mut_ptr(),
            ptr::null_mut(),
            &mut total,
        );
        assert_eq!(fps[0].footprint.width, 8);
        assert_eq!(fps[1].footprint.width, 4);
        assert_eq!(rows, [8, 4]);
        assert!(total >= 256 * 12);

        let mut tok = 0u64;
        assert_eq!(set_frame_int(dev, ptr::null_mut(), 16_667, 1, 0), S_OK);
        assert_eq!(sched_frame(dev, 0, 0, ptr::null(), 0), S_OK);
        assert_eq!(wait_frame(dev, 0, 0, ptr::null(), 0, &mut tok), S_OK);
        assert!(tok >= 1);
        assert_eq!(dev_rel(dev), 0);

        let nvt = build_null_device_vtable(test_probe).expect("null vtable");
        let ndev = create_null_device(nvt);
        let n_create_q: extern "win64" fn(
            *mut c_void,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*(*ndev).vtbl.add(8));
        let n_get_node: extern "win64" fn(*mut c_void) -> u32 =
            std::mem::transmute(*(*ndev).vtbl.add(7));
        let n_rel: extern "win64" fn(*mut c_void) -> u32 =
            std::mem::transmute(*(*ndev).vtbl.add(2));
        let mut out_q = 0xDEADusize as *mut c_void;
        assert_eq!(n_get_node(ndev.cast()), 1);
        assert_eq!(
            n_create_q(
                ndev.cast(),
                &D3D12CommandQueueDesc::default(),
                ptr::null(),
                &mut out_q
            ),
            -2147467263,
            "Null device CreateCommandQueue must return E_NOTIMPL"
        );
        assert!(
            out_q.is_null(),
            "Null device must clear *out_q on E_NOTIMPL"
        );
        assert_eq!(n_rel(ndev.cast()), 0);
    }
}

type CreateQueryHeap = extern "win64" fn(
    *mut c_void,
    *const D3D12QueryHeapDesc,
    *const c_void,
    *mut *mut c_void,
) -> i32;
type CreateCommandSignature = extern "win64" fn(
    *mut c_void,
    *const D3D12CommandSignatureDesc,
    *const ComRootSignature,
    *const c_void,
    *mut *mut c_void,
) -> i32;
type CreateCommittedResource = extern "win64" fn(
    *mut c_void,
    *const D3D12HeapProperties,
    u32,
    *const D3D12ResourceDesc,
    u32,
    *const c_void,
    *const c_void,
    *mut *mut c_void,
) -> i32;

const IID_IUNKNOWN: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 0xc0, 0, 0, 0, 0, 0, 0, 0x46];
const IID_QUERY_HEAP: [u8; 16] = [
    0xae, 0x58, 0x96, 0x0d, 0x45, 0xed, 0x9e, 0x46, 0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4,
];
const IID_COMMAND_SIGNATURE: [u8; 16] = [
    0x7c, 0x79, 0x6a, 0xc3, 0x80, 0xec, 0x0a, 0x4f, 0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1,
];

/// # Safety
/// `object` is a live COM object with one reference held by the caller.
unsafe fn release_object(object: *mut c_void) -> u32 {
    // SAFETY: IUnknown Release is the third entry of every tested COM object.
    unsafe {
        let vtable = *object.cast::<*const usize>();
        let release: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vtable.add(2));
        release(object)
    }
}

/// # Safety
/// `object` holds one reference to a live device child, which this helper releases.
unsafe fn check_device_child(
    object: *mut c_void,
    parent: *mut ComDevice,
    own_iid: &[u8; 16],
    wrong_iid: &[u8; 16],
) {
    assert!(!object.is_null());
    // SAFETY: Device child and parent have live COM vtables. All outputs are writable locals.
    unsafe {
        let vtable = *object.cast::<*const usize>();
        type Query = extern "win64" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32;
        let query: Query = std::mem::transmute(*vtable);
        let add_ref: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vtable.add(1));
        let get_device: Query = std::mem::transmute(*vtable.add(7));
        let mut out = ptr::dangling_mut();
        assert_eq!(
            query(object, wrong_iid.as_ptr().cast(), &mut out),
            E_NOINTERFACE
        );
        assert!(out.is_null());
        assert_eq!(query(object, own_iid.as_ptr().cast(), &mut out), S_OK);
        assert_eq!(out, object);
        assert_eq!(release_object(out), 1);
        assert_eq!(query(object, IID_IUNKNOWN.as_ptr().cast(), &mut out), S_OK);
        assert_eq!(out, object);
        assert_eq!(add_ref(object), 3);
        assert_eq!(release_object(object), 2);
        assert_eq!(release_object(out), 1);
        assert_eq!(
            get_device(object, IID_IUNKNOWN.as_ptr().cast(), &mut out),
            S_OK
        );
        assert_eq!(out, parent.cast());
        assert_eq!(release_object(out), 2);
        assert_eq!(release_object(object), 0);
    }
}

#[test]
fn expanded_device_creates_query_heaps_command_signatures_resources_and_samplers() {
    let backend = Arc::new(
        VulkanBackend::new(VulkanBackendConfig {
            enable_validation: true,
            ..VulkanBackendConfig::from_env()
        })
        .expect("Vulkan backend"),
    );
    let dev = create_vulkan_com_device(Arc::clone(&backend), test_probe).expect("COM device");

    // SAFETY: All device slots use their Win64 signatures, with live inputs and writable outputs.
    unsafe {
        let vtable = (*dev).vtbl;
        let create_query: CreateQueryHeap = std::mem::transmute(*vtable.add(39));
        let create_signature: CreateCommandSignature = std::mem::transmute(*vtable.add(41));
        let create_resource: CreateCommittedResource = std::mem::transmute(*vtable.add(76));
        let create_sampler: extern "win64" fn(*mut ComDevice, *const u8, D3D12CpuDescriptorHandle) =
            std::mem::transmute(*vtable.add(61));
        let (_, dest, _) = backend.create_descriptor_heap(1, 1).expect("sampler heap");
        let mut sampler = [0u32; 27];
        sampler[0x18 / 4..=0x20 / 4].fill(3);
        sampler[0x28 / 4] = 4.0f32.to_bits();
        sampler[0x2c / 4] = 0.25f32.to_bits();
        sampler[0x34 / 4] = 8;
        create_sampler(
            dev,
            sampler.as_ptr().cast(),
            D3D12CpuDescriptorHandle { ptr: dest },
        );
        assert!(matches!(
            backend.resolve_descriptor(dest),
            Some(x2l_gpu_vk::vulkan::DescriptorEntry::Sampler { sampler_id }) if sampler_id != 0
        ));
        let mut out = ptr::dangling_mut();
        let mut query_desc = D3D12QueryHeapDesc {
            heap_type: 0,
            count: 0,
            node_mask: 0,
        };
        let query_iid = IID_QUERY_HEAP.as_ptr().cast();
        assert_eq!(
            create_query(dev.cast(), &query_desc, query_iid, &mut out),
            E_INVALIDARG
        );
        assert!(out.is_null());
        query_desc.count = 4;
        for heap_type in [0, 1, 2, 5] {
            query_desc.heap_type = heap_type;
            assert_eq!(
                create_query(dev.cast(), &query_desc, query_iid, &mut out),
                S_OK
            );
            check_device_child(out, dev, &IID_QUERY_HEAP, &IID_COMMAND_SIGNATURE);
        }
        out = ptr::dangling_mut();
        assert_eq!(
            create_query(
                dev.cast(),
                &query_desc,
                IID_COMMAND_SIGNATURE.as_ptr().cast(),
                &mut out
            ),
            E_NOINTERFACE
        );
        assert!(out.is_null());

        let arguments = [D3D12IndirectArgumentDesc {
            argument_type: 2,
            payload: [0; 3],
        }];
        let mut signature_desc = D3D12CommandSignatureDesc {
            byte_stride: 8,
            num_argument_descs: 1,
            argument_descs: arguments.as_ptr(),
            node_mask: 1,
        };
        let signature_iid = IID_COMMAND_SIGNATURE.as_ptr().cast();
        out = ptr::dangling_mut();
        assert_eq!(
            create_signature(
                dev.cast(),
                &signature_desc,
                ptr::null(),
                signature_iid,
                &mut out
            ),
            E_INVALIDARG
        );
        assert!(out.is_null());
        signature_desc.byte_stride = 16;
        assert_eq!(
            create_signature(
                dev.cast(),
                &signature_desc,
                ptr::null(),
                signature_iid,
                &mut out
            ),
            S_OK
        );
        let bad_arguments = [D3D12IndirectArgumentDesc {
            argument_type: u32::MAX,
            payload: [0; 3],
        }];
        signature_desc.argument_descs = bad_arguments.as_ptr();
        assert_eq!(
            create_signature(
                dev.cast(),
                &signature_desc,
                ptr::null(),
                signature_iid,
                &mut ptr::null_mut()
            ),
            E_INVALIDARG
        );
        check_device_child(out, dev, &IID_COMMAND_SIGNATURE, &IID_QUERY_HEAP);

        let res_desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            ..D3D12ResourceDesc::default()
        };
        out = ptr::dangling_mut();
        assert_ne!(
            create_resource(
                dev.cast(),
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            S_OK
        );
        assert!(out.is_null());
        assert_eq!(
            create_resource(
                dev.cast(),
                ptr::null(),
                0,
                &res_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            S_OK
        );
        assert!(!out.is_null());
        assert_eq!(release_object(out), 0);
        assert_eq!(release_object(dev.cast()), 0);
    }
}

#[test]
fn pipeline_stream_rejects_invalid_descriptors_and_creates_mesh_pipeline() {
    let backend = Arc::new(
        VulkanBackend::new(VulkanBackendConfig {
            enable_validation: true,
            ..VulkanBackendConfig::from_env()
        })
        .expect("Vulkan backend"),
    );
    let supports_mesh = backend.mesh_shader_supported();
    let dev = create_vulkan_com_device(backend, test_probe).expect("COM device");

    // SAFETY: Device slots use the Win64 signatures. Descriptors remain readable through each call.
    unsafe {
        let create_pipeline: CreateStreamPso = std::mem::transmute(*(*dev).vtbl.add(86));
        let create_list: CreateCommandListX = std::mem::transmute(*(*dev).vtbl.add(73));
        let mut out = ptr::dangling_mut();
        assert_eq!(
            create_list(
                dev,
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                &mut out
            ),
            E_INVALIDARG
        );
        assert!(out.is_null());
        let code = compute_signature_cs_bytes();
        let shader = shader_bytecode(&code);
        let compute = subobject(6, shader);
        let mut truncated = stream_desc(&compute);
        truncated.size_in_bytes -= 8;
        let duplicate = [subobject(6, shader), subobject(6, shader)];
        let mixed = [subobject(6, shader), subobject(1, shader)];
        let unknown = subobject(u32::MAX, 0u32);
        let mut unaligned = stream_desc(&compute);
        unaligned.stream = unaligned.stream.cast::<u8>().add(1).cast();
        let empty_bytecode = subobject(
            6,
            D3D12ShaderBytecode {
                p_shader_bytecode: ptr::null(),
                bytecode_length: 4,
            },
        );

        for desc in [
            truncated,
            stream_desc(&duplicate),
            stream_desc(&mixed),
            stream_desc(&unknown),
            unaligned,
            stream_desc(&empty_bytecode),
        ] {
            out = ptr::dangling_mut();
            assert_eq!(
                create_pipeline(dev, &desc, ptr::null(), &mut out),
                E_INVALIDARG
            );
            assert!(out.is_null());
        }

        let mut mesh = asm::Asm::new();
        mesh.push(asm::s_mov_b32(124, asm::inline_i(0))); // m0: zero vertices and primitives
        mesh.push(asm::s_sendmsg_gs_alloc_req());
        mesh.push(asm::s_endpgm());
        let mesh = words_to_bytes(mesh.words());
        let mesh_stream = subobject(25, shader_bytecode(&mesh));
        let mesh_with_as = [subobject(24, shader), subobject(25, shader_bytecode(&mesh))];
        let not_implemented = 0x8000_4001u32 as i32;
        out = ptr::dangling_mut();
        assert_eq!(
            create_pipeline(dev, &stream_desc(&mesh_with_as), ptr::null(), &mut out),
            not_implemented
        );
        assert!(out.is_null());

        let hr = create_pipeline(dev, &stream_desc(&mesh_stream), ptr::null(), &mut out);
        if supports_mesh {
            assert_eq!(hr, S_OK);
            assert!(!out.is_null());
            assert_eq!(release_object(out), 0);
        } else {
            assert_eq!(hr, not_implemented);
            assert!(out.is_null());
        }
        assert_eq!(release_object(dev.cast()), 0);
    }
}

#[test]
fn null_device_expanded_creation_slots_clear_outputs() {
    let vtable = x2l_gpu_vk::build_null_device_vtable(test_probe).expect("null vtable");
    let device = x2l_gpu_vk::create_null_device(vtable).cast::<c_void>();
    let not_implemented = 0x8000_4001u32 as i32;

    // SAFETY: Each slot is invoked with its corresponding Win64 signature and writable COM out.
    unsafe {
        let query: CreateQueryHeap = std::mem::transmute(vtable[39]);
        let signature: CreateCommandSignature = std::mem::transmute(vtable[41]);
        let list: CreateCommandListX = std::mem::transmute(vtable[73]);
        let resource: CreateCommittedResource = std::mem::transmute(vtable[76]);
        let pso: CreateStreamPso = std::mem::transmute(vtable[86]);
        let mut out = ptr::dangling_mut();
        assert_eq!(
            query(device, ptr::null(), ptr::null(), &mut out),
            not_implemented
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        assert_eq!(
            signature(device, ptr::null(), ptr::null(), ptr::null(), &mut out),
            not_implemented
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        assert_eq!(
            list(
                device.cast(),
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                &mut out
            ),
            not_implemented
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        assert_eq!(
            resource(
                device,
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            not_implemented
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        assert_eq!(
            pso(device.cast(), ptr::null(), ptr::null(), &mut out),
            not_implemented
        );
        assert!(out.is_null());
        assert_eq!(release_object(device), 0);
    }
}

type GetAllocationInfo2 = extern "win64" fn(
    *mut c_void,
    *mut c_void,
    u32,
    u32,
    *const D3D12ResourceDesc1,
    *mut D3D12ResourceAllocationInfo1,
) -> *mut D3D12ResourceAllocationInfo;
type CreateCommittedResource3 = extern "win64" fn(
    *mut c_void,
    *const D3D12HeapProperties,
    u32,
    *const D3D12ResourceDesc1,
    u32,
    *const c_void,
    *mut c_void,
    u32,
    *const u32,
    *const c_void,
    *mut *mut c_void,
) -> i32;
type CreatePlacedResourceLayoutX = extern "win64" fn(
    *mut c_void,
    u64,
    *const D3D12ResourceDesc1,
    u32,
    *const c_void,
    *const c_void,
    *mut *mut c_void,
) -> i32;

fn extended_buffer_desc(width: u64) -> D3D12ResourceDesc1 {
    D3D12ResourceDesc1 {
        desc: D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            layout: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn expanded_device_resource_slots_allocate_and_alias_vulkan_memory() {
    let backend = Arc::new(VulkanBackend::new(VulkanBackendConfig::from_env()).expect("Vulkan"));
    let device = create_vulkan_com_device(backend, test_probe).expect("Vulkan COM device");
    let buffer = extended_buffer_desc(128 * 1024);
    let texture = D3D12ResourceDesc1 {
        desc: D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 32,
            height: 16,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let heap = D3D12HeapProperties {
        r#type: 1,
        ..Default::default()
    };

    // SAFETY: Each slot is called with the verified Win64 ABI and live input and output storage.
    unsafe {
        let vt = (*device).vtbl;
        let device = device.cast();
        let query: GetAllocationInfo2 = std::mem::transmute(*vt.add(108));
        let commit: CreateCommittedResource3 = std::mem::transmute(*vt.add(127));
        let place: CreatePlacedResourceLayoutX = std::mem::transmute(*vt.add(133));
        let mut aggregate = MaybeUninit::<D3D12ResourceAllocationInfo>::uninit();
        let mut parts = [MaybeUninit::<D3D12ResourceAllocationInfo1>::uninit(); 2];
        let output = aggregate.as_mut_ptr();
        let descriptions = [buffer, texture];

        assert_eq!(
            query(
                output.cast(),
                device,
                0,
                2,
                descriptions.as_ptr(),
                parts.as_mut_ptr().cast()
            ),
            output
        );
        let aggregate = aggregate.assume_init();
        let [first, second] = parts.map(|part| part.assume_init());
        assert_eq!(first.offset, 0);
        assert_eq!(first.size_in_bytes, buffer.desc.width);
        assert!(first.alignment.is_power_of_two());
        assert!(second.offset >= first.size_in_bytes);
        assert!(second.offset.is_multiple_of(second.alignment));
        assert!(second.size_in_bytes >= 32 * 16 * 4);
        assert!(aggregate.size_in_bytes >= second.offset + second.size_in_bytes);
        assert!(aggregate.size_in_bytes.is_multiple_of(aggregate.alignment));

        let mut result = MaybeUninit::<D3D12ResourceAllocationInfo>::uninit();
        assert_eq!(
            query(
                device,
                result.as_mut_ptr().cast(),
                0,
                1,
                &buffer,
                ptr::null_mut()
            ),
            result.as_mut_ptr()
        );
        assert_eq!(result.assume_init().size_in_bytes, buffer.desc.width);

        let mut result = MaybeUninit::<D3D12ResourceAllocationInfo>::uninit();
        query(
            device,
            result.as_mut_ptr().cast(),
            0,
            2,
            ptr::null(),
            parts.as_mut_ptr().cast(),
        );
        assert_eq!(result.assume_init().size_in_bytes, u64::MAX);
        for part in parts {
            assert_eq!(part.assume_init().size_in_bytes, u64::MAX);
        }

        let mut out = ptr::dangling_mut();
        assert_eq!(
            commit(
                device,
                &heap,
                0,
                &buffer,
                0,
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            E_INVALIDARG
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        let unsupported_iid = [0xffu8; 16];
        assert_eq!(
            commit(
                device,
                &heap,
                0,
                &buffer,
                u32::MAX,
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::null(),
                unsupported_iid.as_ptr().cast(),
                &mut out
            ),
            E_NOINTERFACE
        );
        assert!(out.is_null());
        for (desc, layout) in [(&texture, 0), (&buffer, u32::MAX)] {
            assert_eq!(
                commit(
                    device,
                    &heap,
                    0,
                    desc,
                    layout,
                    ptr::null(),
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut out
                ),
                S_OK
            );
            assert!(!out.is_null());
            if layout == 0 {
                assert_eq!(release_object(out), 0);
                out = ptr::null_mut();
            }
        }
        let resource = out.cast::<ComResource>();
        let res_vt = (*resource).vtbl;
        let get_va: extern "win64" fn(*mut ComResource) -> u64 =
            std::mem::transmute(*res_vt.add(11));
        let map: extern "win64" fn(*mut ComResource, u32, *const c_void, *mut *mut c_void) -> i32 =
            std::mem::transmute(*res_vt.add(8));
        let unmap: extern "win64" fn(*mut ComResource, u32, *const c_void) =
            std::mem::transmute(*res_vt.add(9));

        let offset = 65536;
        let base = get_va(resource);
        assert_ne!(base, 0);
        let view = extended_buffer_desc(4096);
        out = ptr::dangling_mut();
        assert_eq!(
            place(
                device,
                base + buffer.desc.width - 1,
                &view,
                u32::MAX,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            E_INVALIDARG
        );
        assert!(out.is_null());
        assert_eq!(
            place(
                device,
                base + offset,
                &view,
                u32::MAX,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            S_OK
        );
        assert!(!out.is_null());
        let alias = out.cast::<ComResource>();
        assert_eq!(get_va(alias), base + offset);
        let mut original_ptr = ptr::null_mut();
        let mut alias_ptr = ptr::null_mut();
        assert_eq!(map(resource, 0, ptr::null(), &mut original_ptr), S_OK);
        assert_eq!(map(alias, 0, ptr::null(), &mut alias_ptr), S_OK);
        assert!(!original_ptr.is_null() && !alias_ptr.is_null());
        assert_eq!(
            alias_ptr,
            original_ptr.cast::<u8>().add(offset as usize).cast()
        );
        alias_ptr.cast::<u32>().write_unaligned(0x1234_5678);
        assert_eq!(
            original_ptr
                .cast::<u8>()
                .add(offset as usize)
                .cast::<u32>()
                .read_unaligned(),
            0x1234_5678
        );
        unmap(alias, 0, ptr::null());
        unmap(resource, 0, ptr::null());
        assert_eq!(release_object(alias.cast()), 0);
        assert_eq!(release_object(resource.cast()), 0);
        assert_eq!(release_object(device), 0);
    }
}

unsafe fn assert_probe(object: *mut c_void, slot: u32, interface: &str) {
    // SAFETY: The caller identifies a live COM object and an unassigned slot in its vtable.
    let entry = unsafe { *(*object.cast::<*const usize>()).add(slot as usize) };
    assert_ne!(entry, 0);
    let mappings = std::fs::read_to_string("/proc/self/maps").expect("process mappings");
    let permissions = mappings.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        let (start, end) = fields.next()?.split_once('-')?;
        let start = usize::from_str_radix(start, 16).ok()?;
        let end = usize::from_str_radix(end, 16).ok()?;
        (start..end)
            .contains(&entry)
            .then(|| fields.next())
            .flatten()
    });
    assert!(permissions.is_some_and(|perms| perms.starts_with("r-x")));
    // SAFETY: The unassigned slots contain W^X Win64 probe thunks.
    let invoke: extern "win64" fn(*mut c_void) -> u64 = unsafe { std::mem::transmute(entry) };
    PROBE_HIT.set(None);
    assert_eq!(invoke(object), 0);
    let (probe_id, caller_rip) = PROBE_HIT.take().expect("callback invoked");
    assert_eq!(decode_d3d12_vtable_probe(probe_id), (interface, slot));
    assert_ne!(caller_rip, 0);
}

#[test]
fn extended_device_queue_and_list_probes_report_their_interface() {
    let backend = Arc::new(VulkanBackend::new(VulkanBackendConfig::from_env()).expect("Vulkan"));
    let device = create_vulkan_com_device(backend, test_probe).expect("Vulkan COM device");

    // SAFETY: The known creation/lifecycle slots are invoked with their Win64 prototypes.
    // The probe calls below target slots that are unassigned in these tables.
    unsafe {
        for slot in [
            45, 97, 107, 109, 126, 128, 130, 131, 132, 134, 159, 160, 165,
        ] {
            assert_probe(device.cast(), slot, "ID3D12DeviceX");
        }
        let vt = (*device).vtbl;
        let create_queue: extern "win64" fn(
            *mut ComDevice,
            *const D3D12CommandQueueDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*vt.add(48));
        let create_list: extern "win64" fn(
            *mut ComDevice,
            u32,
            u32,
            *mut ComCommandAllocator,
            *mut ComPipelineState,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*vt.add(12));
        let mut queue = ptr::null_mut();
        let mut list = ptr::null_mut();
        assert_eq!(
            create_queue(
                device,
                &D3D12CommandQueueDesc::default(),
                ptr::null(),
                &mut queue
            ),
            S_OK
        );
        assert!(!queue.is_null());
        assert_eq!(
            create_list(
                device,
                0,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                &mut list
            ),
            S_OK
        );
        assert!(!list.is_null());
        let close: extern "win64" fn(*mut c_void) -> i32 =
            std::mem::transmute(*(*list.cast::<*const usize>()).add(9));
        assert_eq!(close(list), S_OK);
        for slot in [35, 63] {
            assert_probe(queue, slot, "ID3D12CommandQueueX");
        }
        for slot in [148, 159] {
            assert_probe(list, slot, "ID3D12GraphicsCommandListX");
        }
        assert_eq!(release_object(list), 0);
        assert_eq!(release_object(queue), 0);
        assert_eq!(release_object(device.cast()), 0);
    }
}

#[test]
fn expanded_null_device_rejects_resources_and_preserves_high_probes() {
    let vt = x2l_gpu_vk::build_null_device_vtable(test_probe).expect("Null device table");
    let device = x2l_gpu_vk::create_null_device(vt).cast();
    let buffer = extended_buffer_desc(65536);
    let mut info = MaybeUninit::<D3D12ResourceAllocationInfo>::uninit();
    let mut part = MaybeUninit::<D3D12ResourceAllocationInfo1>::uninit();
    let not_implemented = 0x8000_4001u32 as i32;

    // SAFETY: The typed Win64 methods receive live descriptors and writable outputs.
    // The high slots invoked through assert_probe are W^X probes in this Null vtable.
    unsafe {
        let query: GetAllocationInfo2 = std::mem::transmute(vt[108]);
        let commit: CreateCommittedResource3 = std::mem::transmute(vt[127]);
        let place: CreatePlacedResourceLayoutX = std::mem::transmute(vt[133]);
        assert_eq!(
            query(
                info.as_mut_ptr().cast(),
                device,
                0,
                1,
                &buffer,
                part.as_mut_ptr()
            ),
            info.as_mut_ptr()
        );
        assert_eq!(info.assume_init().size_in_bytes, u64::MAX);
        assert_eq!(part.assume_init().size_in_bytes, u64::MAX);
        let mut out = ptr::dangling_mut();
        assert_eq!(
            commit(
                device,
                ptr::null(),
                0,
                &buffer,
                u32::MAX,
                ptr::null(),
                ptr::null_mut(),
                0,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            not_implemented
        );
        assert!(out.is_null());
        out = ptr::dangling_mut();
        assert_eq!(
            place(
                device,
                65536,
                &buffer,
                u32::MAX,
                ptr::null(),
                ptr::null(),
                &mut out
            ),
            not_implemented
        );
        assert!(out.is_null());
        for slot in [45, 131, 132, 165] {
            assert_probe(device, slot, "ID3D12DeviceX");
        }
        assert_eq!(release_object(device), 0);
    }
}

thread_local! {
    static REJECTED_QI: std::cell::RefCell<Option<(&'static str, String)>> =
        const { std::cell::RefCell::new(None) };
}

fn capture_rejected_qi(interface: &'static str, iid: &str) {
    REJECTED_QI.replace(Some((interface, iid.to_owned())));
}

fn iid_bytes(guid: &str) -> [u8; 16] {
    let mut bytes = [0; 16];
    let mut digits = guid.bytes().filter(|&byte| byte != b'-');
    for byte in &mut bytes {
        let hex = |value: u8| (value as char).to_digit(16).unwrap() as u8;
        *byte = (hex(digits.next().unwrap()) << 4) | hex(digits.next().unwrap());
    }
    assert!(digits.next().is_none());
    bytes[..4].reverse();
    bytes[4..6].reverse();
    bytes[6..8].reverse();
    bytes
}

#[test]
fn expanded_device_checks_the_public_feature_layouts() {
    let backend = Arc::new(VulkanBackend::new(VulkanBackendConfig::from_env()).expect("Vulkan"));
    let device = create_vulkan_com_device(backend, test_probe).expect("Vulkan COM device");
    // SAFETY: Device slot 13 accepts the Win64 feature ID, a writable buffer and its size.
    unsafe {
        let query: extern "win64" fn(*mut ComDevice, u32, *mut c_void, u32) -> i32 =
            std::mem::transmute(*(*device).vtbl.add(13));
        let check = |feature, words: &mut [u32]| {
            query(
                device,
                feature,
                words.as_mut_ptr().cast(),
                std::mem::size_of_val(words) as u32,
            )
        };
        let mut options = [u32::MAX; 15];
        assert_eq!(check(0, &mut options), S_OK);
        assert_eq!(options[3..5], [3, 3]); // TiledResourcesTier, ResourceBindingTier
        assert_eq!(options[14], 2); // ResourceHeapTier
        let mut options1 = [0; 6];
        assert_eq!(check(8, &mut options1), S_OK);
        assert_eq!(options1[..3], [1, 32, 64]);
        let mut mesh = [u32::MAX; 2];
        assert_eq!(check(32, &mut mesh), S_OK);
        assert_eq!(mesh, [10, 0]);
        let mut enhanced = [0; 3];
        assert_eq!(check(41, &mut enhanced), S_OK);
        assert_eq!(enhanced, [u32::MAX, 1, 0]);
        let mut sm = [0x67];
        assert_eq!(check(7, &mut sm), S_OK);
        assert_eq!(sm, [0x66]);
        let mut root = [3];
        assert_eq!(check(12, &mut root), S_OK);
        assert_eq!(root, [2]);
        let mut arch1 = [0, u32::MAX, 0, 0, u32::MAX];
        assert_eq!(check(16, &mut arch1), S_OK);
        assert_eq!(arch1, [0, 0, 1, 1, 0]);

        for format in [2, 10, 24, 28, 29, 41, 42, 87, 71, 74, 77, 80, 83, 95, 98] {
            let mut support = [format, 0, 0];
            assert_eq!(check(3, &mut support), S_OK);
            assert_eq!(support[0], format);
            assert_eq!(support[1] & 0x120, 0x120); // TEXTURE2D | SHADER_LOAD
            if matches!(format, 29 | 87 | 71 | 74 | 77 | 80 | 83 | 95 | 98) {
                assert_eq!(support[2], 0);
            }
        }
        for format in [40, 45] {
            let mut support = [format, 0, u32::MAX];
            assert_eq!(check(3, &mut support), S_OK);
            assert_ne!(support[1] & 0x1_0000, 0); // DEPTH_STENCIL
            assert_eq!(support[2], 0);
        }
        let mut sentinel = [0xbad0_cafe; 16];
        assert_eq!(check(0, &mut sentinel[..14]), E_INVALIDARG);
        assert_eq!(check(0, &mut sentinel), E_INVALIDARG);
        assert_eq!(check(0xffff_ffff, &mut sentinel), E_INVALIDARG);
        assert_eq!(sentinel, [0xbad0_cafe; 16]);
        assert_eq!(release_object(device.cast()), 0);
    }
}

#[test]
fn expanded_command_list_qi_and_console_slots_execute_on_direct_and_compute() {
    use x2l_gpu_vk::com::E_POINTER;
    use x2l_gpu_vk::set_d3d12_rejected_qi_callback;

    let backend = Arc::new(VulkanBackend::new(VulkanBackendConfig::from_env()).expect("Vulkan"));
    let dev = create_vulkan_com_device(backend, test_probe).expect("Vulkan COM device");

    type Query = extern "win64" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32;
    type CreateAllocator =
        extern "win64" fn(*mut ComDevice, u32, *const c_void, *mut *mut c_void) -> i32;
    type CreateQueue = extern "win64" fn(
        *mut ComDevice,
        *const D3D12CommandQueueDesc,
        *const c_void,
        *mut *mut c_void,
    ) -> i32;
    type RawCall = extern "win64" fn(*mut ComGraphicsCommandList, usize, usize, usize) -> u64;

    let device_versions = [
        "77acce80-638e-4e65-8895-c1f23386863e",
        "30baa41e-b15b-475c-a0bb-1af5c5b64328",
        "81dadc15-2bad-4392-93c5-101345c4aa98",
        "e865df17-a9ee-46f9-a463-3098315aa2e5",
        "8b4f173b-2fea-4b80-8f58-4307191ab95d",
        "c70b221b-40e4-4a17-89af-025a0727a6dc",
        "5c014b53-68a1-4b9b-8bd1-dd6046b9358b",
        "9218e6bb-f944-4f7e-a75c-b1b2c7b701f3",
        "4c80e962-f032-4f60-bc9e-ebc2cfa1d83c",
        "517f8718-aa66-49f9-b02b-a7ab89c06031",
        "5405c344-d457-444e-b4dd-2366e45aee39",
        "5af5c532-4c91-4cd0-b541-15a405395fc5",
        "14eecffc-4df8-40f7-a118-5c816f45695e",
        "5f6e592d-d895-44c2-8e4a-88ad4926d323",
    ];
    let list_versions = [
        "553103fb-1fe7-4557-bb38-946d7d0e7ca7",
        "38c3e585-ff17-412c-9150-4fc6f9d72a28",
        "6fda83a7-b84c-4e38-9ac8-c7bd22016b3d",
        "8754318e-d3a9-4541-98cf-645b50dc4874",
        "55050859-4024-474c-87f5-6472eaee44ea",
        "c3827890-e548-4cfa-96cf-5689a9370f80",
        "dd171223-8b61-4769-90e3-160ccde4e2c1",
        "ee936ef9-599d-4d28-938e-23c4ad05ce51",
        "34ed2808-ffe6-4c2b-b11a-cabd2b0c59e1",
        "7013c015-d161-4b63-a08c-238552dd8acc",
    ];

    // SAFETY: All COM objects stay live through submission, all output storage is writable,
    // and the tested slots use their documented Win64 ABI or the raw probe register layout.
    unsafe {
        let vt = (*dev).vtbl;
        let device_query: Query = std::mem::transmute(*vt);
        let mut alias = ptr::null_mut();
        let mut identity = ptr::null_mut();
        for guid in device_versions {
            let iid = iid_bytes(guid);
            assert_eq!(
                device_query(dev.cast(), iid.as_ptr().cast(), &mut alias),
                S_OK
            );
            assert_eq!(
                device_query(alias, IID_IUNKNOWN.as_ptr().cast(), &mut identity),
                S_OK
            );
            assert_eq!(identity, dev.cast());
            assert_eq!(release_object(identity), 2);
            assert_eq!(release_object(alias), 1);
        }

        let create_allocator: CreateAllocator = std::mem::transmute(*vt.add(9));
        let create_queue: CreateQueue = std::mem::transmute(*vt.add(48));
        let create_list: CreateCommandListX = std::mem::transmute(*vt.add(73));
        let create_buffer: CreateCommittedResource = std::mem::transmute(*vt.add(27));
        let buffer_desc = extended_buffer_desc(65_536);
        let heap = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let queue_iid = iid_bytes("0ec870a6-5d7e-4c22-8cfc-5baae07616ed");
        let alloc_iid = iid_bytes("6102dee4-af59-4b09-b999-b44d73f09b24");
        let resource_iid = iid_bytes("696442be-a72e-4059-bc79-5b5c98040fad");
        let list_iid = iid_bytes("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455");

        for list_type in [0, 2] {
            let mut q = ptr::null_mut();
            let mut alloc = ptr::null_mut();
            let mut object = ptr::null_mut();
            assert_eq!(
                create_queue(
                    dev,
                    &D3D12CommandQueueDesc {
                        r#type: list_type,
                        ..Default::default()
                    },
                    queue_iid.as_ptr().cast(),
                    &mut q
                ),
                S_OK
            );
            assert_eq!(
                create_allocator(dev, list_type, alloc_iid.as_ptr().cast(), &mut alloc),
                S_OK
            );
            assert_eq!(
                create_list(
                    dev,
                    &D3D12CommandListXDesc {
                        list_type,
                        flags: 0,
                        node_mask: 0,
                        barrier_batch: 0
                    },
                    alloc.cast(),
                    ptr::null_mut(),
                    list_iid.as_ptr().cast(),
                    &mut object
                ),
                S_OK
            );
            let list = object.cast::<ComGraphicsCommandList>();
            let lv = (*list).vtbl;
            let list_query: Query = std::mem::transmute(*lv);
            for guid in list_versions {
                let iid = iid_bytes(guid);
                assert_eq!(list_query(object, iid.as_ptr().cast(), &mut alias), S_OK);
                assert_eq!(
                    list_query(alias, IID_IUNKNOWN.as_ptr().cast(), &mut identity),
                    S_OK
                );
                assert_eq!(identity, object);
                assert_eq!(release_object(identity), 2);
                assert_eq!(release_object(alias), 1);
            }

            let wrong = iid_bytes("12345678-9abc-def0-0123-456789abcdef");
            set_d3d12_rejected_qi_callback(Some(capture_rejected_qi));
            alias = ptr::dangling_mut();
            let rejected = list_query(object, wrong.as_ptr().cast(), &mut alias);
            let null_output = list_query(object, wrong.as_ptr().cast(), ptr::null_mut());
            set_d3d12_rejected_qi_callback(None);
            assert_eq!(rejected, E_NOINTERFACE);
            assert!(alias.is_null());
            assert_eq!(null_output, E_POINTER);
            assert_eq!(
                REJECTED_QI.take(),
                Some((
                    "ID3D12GraphicsCommandListX",
                    "{12345678-9abc-def0-0123-456789abcdef}".to_owned()
                ))
            );

            for slot in [132, 131] {
                let invoke: RawCall = std::mem::transmute(*lv.add(slot));
                PROBE_HIT.set(None);
                assert_eq!(invoke(list, 0x1111, 0x2222, 0x3333), 0);
                let (tag, rip) = PROBE_HIT.take().expect("compatibility call recorded");
                assert_eq!(
                    decode_d3d12_vtable_probe(tag),
                    ("ID3D12GraphicsCommandListX", slot as u32)
                );
                assert_ne!(rip, 0);
            }

            let mut payload = ptr::null_mut();
            let mut target = ptr::null_mut();
            let mut readback = ptr::null_mut();
            for out in [&mut payload, &mut target, &mut readback] {
                assert_eq!(
                    create_buffer(
                        dev.cast(),
                        &heap,
                        0,
                        &buffer_desc.desc,
                        0,
                        ptr::null(),
                        resource_iid.as_ptr().cast(),
                        out
                    ),
                    S_OK
                );
            }
            let rv = (*payload.cast::<ComResource>()).vtbl;
            let gpu_va: extern "win64" fn(*mut c_void) -> u64 = std::mem::transmute(*rv.add(11));
            let source = gpu_va(payload);
            let dest = gpu_va(target);
            assert_ne!(source, 0);
            assert_ne!(dest, 0);

            let copy: extern "win64" fn(*mut ComGraphicsCommandList, u64, u64, u64, u32) =
                std::mem::transmute(*lv.add(60));
            let fill32: extern "win64" fn(*mut ComGraphicsCommandList, u64, u64, u32) =
                std::mem::transmute(*lv.add(61));
            let fill64: extern "win64" fn(*mut ComGraphicsCommandList, u64, u64, u64) =
                std::mem::transmute(*lv.add(62));
            let fill128: extern "win64" fn(*mut ComGraphicsCommandList, u64, u64, *const [u32; 4]) =
                std::mem::transmute(*lv.add(63));
            let copy_resource: extern "win64" fn(
                *mut ComGraphicsCommandList,
                *mut c_void,
                *mut c_void,
                u32,
            ) = std::mem::transmute(*lv.add(84));
            let barrier: extern "win64" fn(*mut ComGraphicsCommandList, u32, *const c_void) =
                std::mem::transmute(*lv.add(146));
            let pattern = [10, 20, 30, 40];
            fill32(list, source, 8, 0xaabb_ccdd);
            fill64(list, source + 8, 8, 0x8877_6655_4433_2211);
            fill128(list, source + 16, 16, &pattern);
            barrier(list, 0, ptr::null());
            copy(list, dest, source, 32, 0);
            copy_resource(list, readback, target, 0);

            let close: extern "win64" fn(*mut ComGraphicsCommandList) -> i32 =
                std::mem::transmute(*lv.add(9));
            let qv = (*q.cast::<ComCommandQueue>()).vtbl;
            let execute: extern "win64" fn(*mut c_void, u32, *const *mut ComGraphicsCommandList) =
                std::mem::transmute(*qv.add(10));
            assert_eq!(close(list), S_OK);
            execute(q, 1, &list);

            let map: extern "win64" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> i32 =
                std::mem::transmute(*rv.add(8));
            let unmap: extern "win64" fn(*mut c_void, u32, *const c_void) =
                std::mem::transmute(*rv.add(9));
            let mut mapped = ptr::null_mut();
            assert_eq!(map(readback, 0, ptr::null(), &mut mapped), S_OK);
            assert_eq!(
                mapped.cast::<[u32; 8]>().read_unaligned(),
                [
                    0xaabb_ccdd,
                    0xaabb_ccdd,
                    0x4433_2211,
                    0x8877_6655,
                    10,
                    20,
                    30,
                    40
                ]
            );
            unmap(readback, 0, ptr::null());
            assert_probe(object, 165, "ID3D12GraphicsCommandListX");
            for obj in [readback, target, payload, object, alloc, q] {
                assert_eq!(release_object(obj), 0);
            }
        }
        assert_eq!(release_object(dev.cast()), 0);
    }
}

type DxgiQuery = extern "win64" fn(*mut c_void, *const [u8; 16], *mut *mut c_void) -> i32;
type DxgiGetObject = extern "win64" fn(*mut c_void, *mut *mut c_void) -> i32;
type DxgiEnumObject = extern "win64" fn(*mut c_void, u32, *mut *mut c_void) -> i32;

#[repr(align(8))]
struct DxgiBytes<const N: usize>([u8; N]);

impl<const N: usize> DxgiBytes<N> {
    fn u32_at(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
    }

    fn u64_at(&self, at: usize) -> u64 {
        u64::from_le_bytes(self.0[at..at + 8].try_into().unwrap())
    }

    fn utf16_at(&self, len: usize) -> String {
        let chars: Vec<u16> = self.0[..len]
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .take_while(|&ch| ch != 0)
            .collect();
        String::from_utf16(&chars).expect("UTF-16 name")
    }
}

unsafe fn dxgi_slot(object: *mut c_void, slot: usize) -> usize {
    // SAFETY: The caller supplies a live COM object whose vtable contains the requested slot.
    unsafe { (*object.cast::<*const usize>()).add(slot).read() }
}

fn exercise_dxgi(owner: *mut c_void, expected_name: &str, expected_null_defaults: bool) {
    const NOT_FOUND: i32 = 0x887A_0002_u32 as i32;
    const MORE_DATA: i32 = 0x887A_0003_u32 as i32;
    const UNSUPPORTED: i32 = 0x887A_0004_u32 as i32;
    let iid_device = iid_bytes("54ec77fa-1377-44e6-8c32-88fd5f44c84c");
    let iid_adapter1 = iid_bytes("29038f61-3839-4626-91fd-086879011a05");
    let iid_adapter3 = iid_bytes("645967a4-1392-4310-a798-8053ce3e93fd");
    let iid_factory6 = iid_bytes("c1b6694f-ff09-44a9-b03c-77900a0a1d17");
    let iid_d3d12 = iid_bytes("189819f1-1db6-4b57-be54-1821339b85f7");
    // SAFETY: Every slot is called with its public Win64 DXGI signature and live objects.
    // Byte buffers independently exercise the public x64 output sizes and offsets.
    unsafe {
        let qi: DxgiQuery = std::mem::transmute(dxgi_slot(owner, 0));
        let mut device = ptr::null_mut();
        assert_eq!(qi(owner, &iid_device, &mut device), S_OK);
        assert_ne!(device, owner);
        let dxgi_qi: DxgiQuery = std::mem::transmute(dxgi_slot(device, 0));
        for iid in [
            "aec22fb8-76f3-4639-9be0-28eb43a67a2e", // IDXGIObject
            "77db970f-6276-48ba-ba28-070143b4392c",
            "05008617-fbfd-4051-a790-144884b4f6a9",
            "6007896c-3244-4afd-bf18-a6d3beda5023",
            "95b4f95f-d8da-4ca4-9ee6-3b76d5968a10",
        ] {
            let mut same = ptr::null_mut();
            let iid = iid_bytes(iid);
            assert_eq!(qi(owner, &iid, &mut same), S_OK);
            assert_eq!(same, device);
            assert_eq!(release_object(same), 1);
        }
        let mut back = ptr::null_mut();
        for iid in [iid_d3d12, iid_bytes("00000000-0000-0000-c000-000000000046")] {
            assert_eq!(dxgi_qi(device, &iid, &mut back), S_OK);
            assert_eq!(back, owner);
            assert_eq!(release_object(back), 2);
        }

        let get_adapter: DxgiGetObject = std::mem::transmute(dxgi_slot(device, 7));
        let mut adapter = ptr::null_mut();
        assert_eq!(get_adapter(device, &mut adapter), S_OK);
        let device_parent: DxgiQuery = std::mem::transmute(dxgi_slot(device, 6));
        let mut parent = ptr::null_mut();
        assert_eq!(device_parent(device, &iid_adapter1, &mut parent), S_OK);
        assert_eq!(parent, adapter);
        assert_eq!(release_object(parent), 1);
        let set_priority: extern "win64" fn(*mut c_void, i32) -> i32 =
            std::mem::transmute(dxgi_slot(device, 10));
        let get_priority: extern "win64" fn(*mut c_void, *mut i32) -> i32 =
            std::mem::transmute(dxgi_slot(device, 11));
        let set_latency: extern "win64" fn(*mut c_void, u32) -> i32 =
            std::mem::transmute(dxgi_slot(device, 12));
        let get_latency: extern "win64" fn(*mut c_void, *mut u32) -> i32 =
            std::mem::transmute(dxgi_slot(device, 13));
        let mut priority = -1;
        let mut latency = 0;
        assert_eq!(set_priority(device, 7), S_OK);
        assert_eq!(set_priority(device, 8), E_INVALIDARG);
        assert_eq!(get_priority(device, &mut priority), S_OK);
        assert_eq!(priority, 7);
        assert_eq!(get_latency(device, &mut latency), S_OK);
        assert_eq!(latency, 3);
        assert_eq!(set_latency(device, 16), S_OK);
        assert_eq!(set_latency(device, 17), E_INVALIDARG);
        assert_eq!(get_latency(device, &mut latency), S_OK);
        assert_eq!(latency, 16);
        assert_eq!(set_latency(device, 0), S_OK);
        assert_eq!(get_latency(device, &mut latency), S_OK);
        assert_eq!(latency, 3);

        let adapter_qi: DxgiQuery = std::mem::transmute(dxgi_slot(adapter, 0));
        for iid in [
            iid_adapter1,
            iid_adapter3,
            iid_bytes("3c8d99d1-4fbf-4181-a82c-af66bf7bd24e"),
        ] {
            let mut upgraded = ptr::null_mut();
            assert_eq!(adapter_qi(adapter, &iid, &mut upgraded), S_OK);
            assert_eq!(upgraded, adapter);
            assert_eq!(release_object(upgraded), 1);
        }
        let get_desc1: extern "win64" fn(*mut c_void, *mut u8) -> i32 =
            std::mem::transmute(dxgi_slot(adapter, 10));
        let mut desc = DxgiBytes([0; 312]);
        assert_eq!(get_desc1(adapter, desc.0.as_mut_ptr()), S_OK);
        assert_eq!(desc.utf16_at(256), expected_name);
        assert_ne!(desc.u32_at(256), 0);
        assert_ne!(desc.u64_at(272), 0);
        assert_ne!(desc.u64_at(288), 0);
        assert_eq!(desc.u64_at(296), 0x5832_4c01);
        eprintln!(
            "DXGI adapter {expected_name}: vendor={:#06x}, device={:#06x}, dedicated={}, shared={}, flags={:#x}",
            desc.u32_at(256),
            desc.u32_at(260),
            desc.u64_at(272),
            desc.u64_at(288),
            desc.u32_at(304)
        );

        if expected_null_defaults {
            assert_eq!(desc.u32_at(256), 0x1002);
            assert_eq!(desc.u32_at(260), 0x73bf);
            assert_eq!(desc.u64_at(272), 10 << 30);
            assert_eq!(desc.u64_at(288), 6 << 30);
            assert_eq!(desc.u32_at(304), 2);
        }
        for (slot, size) in [(8, 304), (11, 320), (18, 320)] {
            let get_desc: extern "win64" fn(*mut c_void, *mut u8) -> i32 =
                std::mem::transmute(dxgi_slot(adapter, slot));
            let mut other = DxgiBytes([0xb2; 328]);
            assert_eq!(get_desc(adapter, other.0.as_mut_ptr()), S_OK);
            assert_eq!(&other.0[..304], &desc.0[..304]);
            assert_eq!(&other.0[size..], &[0xb2; 328][size..]);
        }
        let supported: extern "win64" fn(*mut c_void, *const [u8; 16], *mut i64) -> i32 =
            std::mem::transmute(dxgi_slot(adapter, 9));
        assert_eq!(supported(adapter, &iid_d3d12, ptr::null_mut()), S_OK);
        let query_memory: extern "win64" fn(*mut c_void, u32, u32, *mut u8) -> i32 =
            std::mem::transmute(dxgi_slot(adapter, 14));
        for (segment, bytes) in [(0, desc.u64_at(272)), (1, desc.u64_at(288))] {
            let mut memory = DxgiBytes([0; 32]);
            assert_eq!(
                query_memory(adapter, 0, segment, memory.0.as_mut_ptr()),
                S_OK
            );
            assert!(memory.u64_at(0) > 0 && memory.u64_at(0) <= bytes);
            if expected_null_defaults {
                assert_eq!(memory.u64_at(0), bytes);
                assert_eq!(memory.u64_at(8), 0);
            }
            assert_eq!(
                query_memory(adapter, 1, segment, memory.0.as_mut_ptr()),
                E_INVALIDARG
            );
        }

        let enum_outputs: DxgiEnumObject = std::mem::transmute(dxgi_slot(adapter, 7));
        let mut output = ptr::null_mut();
        assert_eq!(enum_outputs(adapter, 0, &mut output), S_OK);
        let mut missing = output;
        assert_eq!(enum_outputs(adapter, 1, &mut missing), NOT_FOUND);
        assert!(missing.is_null());
        let output_desc: extern "win64" fn(*mut c_void, *mut u8) -> i32 =
            std::mem::transmute(dxgi_slot(output, 7));
        let mut display = DxgiBytes([0; 96]);
        assert_eq!(output_desc(output, display.0.as_mut_ptr()), S_OK);
        assert_eq!(display.utf16_at(64), "X2Linux Display");
        assert_eq!(display.u64_at(64), 0);
        assert_eq!(display.u32_at(72), 3840);
        assert_eq!(display.u32_at(76), 2160);
        assert_eq!(display.u32_at(80), 1);
        assert_eq!(display.u32_at(84), 1);

        let modes: extern "win64" fn(*mut c_void, u32, u32, *mut u32, *mut u32) -> i32 =
            std::mem::transmute(dxgi_slot(output, 8));
        for format in [24, 28, 87] {
            let mut count = 0;
            assert_eq!(modes(output, format, 0, &mut count, ptr::null_mut()), S_OK);
            assert_eq!(count, 6);
            let mut entries = [[0u32; 7]; 6];
            let mut capacity = 1;
            assert_eq!(
                modes(
                    output,
                    format,
                    0,
                    &mut capacity,
                    entries.as_mut_ptr().cast()
                ),
                MORE_DATA
            );
            assert_eq!(capacity, count);
            assert_eq!(
                modes(output, format, 0, &mut count, entries.as_mut_ptr().cast()),
                S_OK
            );
            assert_eq!(
                entries,
                [
                    [1920, 1080, 60, 1, format, 1, 0],
                    [1920, 1080, 120, 1, format, 1, 0],
                    [2560, 1440, 60, 1, format, 1, 0],
                    [2560, 1440, 120, 1, format, 1, 0],
                    [3840, 2160, 60, 1, format, 1, 0],
                    [3840, 2160, 120, 1, format, 1, 0],
                ]
            );
        }
        let closest: extern "win64" fn(*mut c_void, *const u32, *mut u32, *mut c_void) -> i32 =
            std::mem::transmute(dxgi_slot(output, 9));
        let request = [2510, 1400, 119_880, 1000, 28, 0, 0];
        let mut matched = [0u32; 7];
        assert_eq!(
            closest(
                output,
                request.as_ptr(),
                matched.as_mut_ptr(),
                ptr::null_mut()
            ),
            S_OK
        );
        assert_eq!(matched, [2560, 1440, 120, 1, 28, 1, 0]);
        let vblank: extern "win64" fn(*mut c_void) -> i32 =
            std::mem::transmute(dxgi_slot(output, 10));
        let before = std::time::Instant::now();
        assert_eq!(vblank(output), S_OK);
        assert!(before.elapsed() >= std::time::Duration::from_millis(15));

        let adapter_parent: DxgiQuery = std::mem::transmute(dxgi_slot(adapter, 6));
        let mut factory = ptr::null_mut();
        assert_eq!(adapter_parent(adapter, &iid_factory6, &mut factory), S_OK);
        for slot in [7, 12] {
            let enum_adapters: DxgiEnumObject = std::mem::transmute(dxgi_slot(factory, slot));
            assert_eq!(enum_adapters(factory, 0, &mut parent), S_OK);
            assert_eq!(parent, adapter);
            assert_eq!(release_object(parent), 1);
            assert_eq!(enum_adapters(factory, 1, &mut parent), NOT_FOUND);
            assert!(parent.is_null());
        }
        let swap: extern "win64" fn(
            *mut c_void,
            *mut c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(dxgi_slot(factory, 10));
        let mut swapchain = factory;
        assert_eq!(
            swap(factory, owner, ptr::null(), &mut swapchain),
            UNSUPPORTED
        );
        assert!(swapchain.is_null());
        let (probe, rip) = PROBE_HIT.get().expect("unsupported DXGI method probe");
        assert_eq!(decode_d3d12_vtable_probe(probe), ("IDXGIFactory", 10));
        assert_ne!(rip, 0);
        let swap_hwnd: extern "win64" fn(
            *mut c_void,
            usize,
            usize,
            usize,
            usize,
            usize,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(dxgi_slot(factory, 15));
        swapchain = factory;
        assert_eq!(
            swap_hwnd(factory, 0, 0, 0, 0, 0, &mut swapchain),
            UNSUPPORTED
        );
        assert!(swapchain.is_null());

        assert_eq!(release_object(factory), 0);
        assert_eq!(release_object(owner), 3);
        assert_eq!(release_object(device), 0);
        assert_eq!(release_object(adapter), 0);
        assert_eq!(output_desc(output, display.0.as_mut_ptr()), S_OK);
        assert_eq!(release_object(output), 0);
    }
}

#[test]
fn dxgi_vulkan_device_adapter_output_and_factory_win64_abi() {
    let backend = Arc::new(VulkanBackend::new(VulkanBackendConfig::from_env()).expect("Vulkan"));
    let telemetry = backend.telemetry_snapshot();
    eprintln!(
        "DXGI Vulkan device {} via {}",
        telemetry.device_name, telemetry.driver_info
    );
    let device = create_vulkan_com_device(backend, test_probe).expect("Vulkan COM device");
    exercise_dxgi(device.cast(), &telemetry.device_name, false);
}

#[test]
fn dxgi_null_device_adapter_output_and_factory_win64_abi() {
    let vtable = x2l_gpu_vk::build_null_device_vtable(test_probe).expect("null vtable");
    let device = x2l_gpu_vk::create_null_device(vtable);
    exercise_dxgi(device.cast(), "X2Linux Null Adapter", true);
}

#[test]
fn common_dxgi_texture_formats_create_committed_and_placed_resources() {
    let backend = Arc::new(VulkanBackend::new(VulkanBackendConfig::from_env()).expect("Vulkan"));
    let device = create_vulkan_com_device(backend, test_probe).expect("Vulkan COM device");
    // SAFETY: COM methods use the typed Win64 signatures and live descriptors and outputs.
    unsafe {
        let vtable = (*device).vtbl;
        let create_committed: extern "win64" fn(
            *mut ComDevice,
            *const D3D12HeapProperties,
            u32,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*vtable.add(27));
        let create_heap: extern "win64" fn(
            *mut ComDevice,
            *const D3D12HeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*vtable.add(28));
        let create_placed: extern "win64" fn(
            *mut ComDevice,
            *mut ComHeap,
            u64,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*vtable.add(29));
        let props = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let mut heap = ptr::null_mut();
        assert_eq!(
            create_heap(
                device,
                &D3D12HeapDesc {
                    size_in_bytes: 2 << 20,
                    properties: props,
                    ..Default::default()
                },
                ptr::null(),
                &mut heap
            ),
            S_OK
        );
        let mut desc = D3D12ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            width: 64,
            height: 64,
            depth_or_array_size: 1,
            mip_levels: 1,
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            ..Default::default()
        };
        for (format, flags) in [
            (DXGI_FORMAT_BC7_UNORM, 0),
            (DXGI_FORMAT_R16G16B16A16_FLOAT, 1),
            (DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 2),
            (DXGI_FORMAT_R11G11B10_FLOAT, 1),
            (DXGI_FORMAT_D24_UNORM_S8_UINT, 2),
        ] {
            desc.format = format;
            desc.flags = flags;
            let mut resource = ptr::null_mut();
            assert_eq!(
                create_committed(
                    device,
                    &props,
                    0,
                    &desc,
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut resource
                ),
                S_OK,
                "committed format {format}",
            );
            assert_eq!(release_object(resource), 0);
            assert_eq!(
                create_placed(
                    device,
                    heap.cast(),
                    65536,
                    &desc,
                    0,
                    ptr::null(),
                    ptr::null(),
                    &mut resource
                ),
                S_OK,
                "placed format {format}",
            );
            assert_eq!(release_object(resource), 0);
        }
        desc.format = u32::MAX;
        let mut bad = heap;
        assert_eq!(
            create_committed(
                device,
                &props,
                0,
                &desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut bad
            ),
            E_INVALIDARG,
        );
        assert!(bad.is_null());
        bad = heap;
        assert_eq!(
            create_placed(
                device,
                heap.cast(),
                0,
                &desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut bad
            ),
            E_INVALIDARG,
        );
        assert!(bad.is_null());
        assert_eq!(release_object(heap), 0);
        assert_eq!(release_object(device.cast()), 0);
    }
}
