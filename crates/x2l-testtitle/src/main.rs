//! Clean-room synthetic console PE32+ end-to-end test title (`x86_64-pc-windows-gnu`).
//!
//! Verifies all M1, M2.4, and M3.1 compatibility invariants under `x2linux run`:
//! 1. `kernelx!GetConsoleType`, `TitleMemoryStatus`, `XMemAlloc`/`XMemFree`, unknown ordinal `#999`
//! 2. `xgameruntime!XGameRuntimeInitialize`, offline `XStoreCreateContext`, `XGameRuntimeUninitialize`
//! 3. `d3d12_xs`: `D3D12ConfigureVirtualMemoryX`, `D3D12SerializeVersionedRootSignature` (`ID3DBlob`),
//!    `D3D12XboxCreateDevice` + spare vtable probe slot 45 (+ 64x64 Vulkan triangle draw when active)
//! 4. `xmem.dll`: `XMemAllocDefault`/`XMemFreeDefault`, `XMemAllocatePhysicalPages`,
//!    `XMemVirtualAlloc` + `XMemMapPhysicalPagesScatter` + `XMemVirtualQuery`,
//!    `XMemFreePhysicalPages`, `XMemGetWorkingSetStatistics`, `XMemIsToolsMemAvailable`
//! 5. `xg_xs.dll`: `XGSuggestSwizzleMode` & `XGCreateTextureComputer` (`GetResourceLayout`)
//! 6. `XCRDAPI.dll`: `DStorageGetFactory`, `CreateQueue`, `CreateStatusArray`, `OpenFile` + read on `G:\`
//! 7. `GameInput.dll`: ordinal `#1` (`GameInputCreate`) + `IGameInput` timestamp/reading/callback lifecycle
//! 8. `AcpHal.dll`, `xapu.dll`, `chad.dll`, & `api-ms-win-core-psm-appnotify-l1-1-0.dll` lifecycles
//! 9. Multi-threaded static TLS (`thread_local!`), SEH/panic unwind, and OverlayFS CoW checks

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
    pub struct XMemBasicInformation {
        pub base_address: u64,
        pub allocation_base: u64,
        pub allocation_protect: u32,
        pub partition_id: u16,
        pub _pad0: u16,
        pub region_size: u64,
        pub state: u32,
        pub protect: u32,
        pub mem_type: u32,
        pub _pad1: u32,
        pub flags: u64,
        pub tag_code: u32,
        pub tag: [u8; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12ConfigureVirtualMemoryXParams {
        pub flags_or_mode: u32,
        pub field_04: u32,
        pub field_08: u32,
        pub _pad0c: u32,
        pub low_address_out: u64,
        pub high_address_out: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12VersionedRootSignatureDesc {
        pub version: u32,
        pub _pad04: u32,
        pub num_parameters: u32,
        pub _pad0c: u32,
        pub p_parameters: *const c_void,
        pub num_static_samplers: u32,
        pub _pad1c: u32,
        pub p_static_samplers: *const c_void,
        pub flags: u32,
        pub _pad2c: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12XboxCreateDeviceParameters {
        pub version: u32,
        pub process_debug_flags: u32,
        pub graphics_command_queue_ring_size_bytes: u32,
        pub graphics_scratch_memory_size_bytes: u32,
        pub compute_scratch_memory_size_bytes: u32,
        pub reserved: [u32; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct DStorageRequest {
        pub options: u64,
        pub dst_ptr: u64,
        pub dst_page_offset: u64,
        pub dst_size: u32,
        pub _pad1c: u32,
        pub src_ptr: u64,
        pub file_offset: u64,
        pub src_size: u32,
        pub intermediate_size: u32,
        pub cancellation_tag: u64,
        pub name: *const u8,
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
    #[derive(Clone, Copy, Default)]
    pub struct D3D12GpuDescriptorHandle {
        pub ptr: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12RootParameterRaw {
        pub parameter_type: u32,
        pub _pad04: u32,
        pub data: [u32; 4],
        pub shader_visibility: u32,
        pub _pad1c: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12DescriptorRange1 {
        pub range_type: u32,
        pub num_descriptors: u32,
        pub base_shader_register: u32,
        pub register_space: u32,
        pub flags: u32,
        pub offset_from_start: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12HeapDesc {
        pub size_in_bytes: u64,
        pub properties: D3D12HeapProperties,
        pub _pad28: u32,
        pub alignment: u64,
        pub flags: u32,
        pub _pad44: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12VertexBufferView {
        pub buffer_location: u64,
        pub size_in_bytes: u32,
        pub stride_in_bytes: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct D3D12IndexBufferView {
        pub buffer_location: u64,
        pub size_in_bytes: u32,
        pub format: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12InputElementDesc {
        pub semantic_name: *const u8,
        pub semantic_index: u32,
        pub format: u32,
        pub input_slot: u32,
        pub aligned_byte_offset: u32,
        pub input_slot_class: u32,
        pub instance_data_step_rate: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct D3D12ComputePipelineStateDesc {
        pub p_root_signature: *mut c_void,
        pub cs: D3D12ShaderBytecode,
        pub node_mask: u32,
        pub _pad28: u32,
        pub cached_pso: D3D12CachedPipelineState,
        pub flags: u32,
        pub _pad52: u32,
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

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct MemoryBasicInformation {
        pub base_address: u64,
        pub allocation_base: u64,
        pub allocation_protect: u32,
        pub partition_id: u16,
        pub _pad0: u16,
        pub region_size: u64,
        pub state: u32,
        pub protect: u32,
        pub mem_type: u32,
        pub _pad1: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct GameInputGamepadState {
        pub buttons: u32,
        pub left_trigger: f32,
        pub right_trigger: f32,
        pub left_thumbstick_x: f32,
        pub left_thumbstick_y: f32,
        pub right_thumbstick_x: f32,
        pub right_thumbstick_y: f32,
    }

    // Compile-time ABI checks matching Win64 DirectX-Headers & x2l-gpu-vk::com / x2l-hle-kernelx.
    const _: () = assert!(size_of::<MemoryBasicInformation>() == 48);
    const _: () = assert!(size_of::<GameInputGamepadState>() == 28);
    const _: () = assert!(size_of::<XMemBasicInformation>() == 64);
    const _: () = assert!(size_of::<D3D12ConfigureVirtualMemoryXParams>() == 32);
    const _: () = assert!(size_of::<D3D12VersionedRootSignatureDesc>() == 48);
    const _: () = assert!(size_of::<DStorageRequest>() == 0x48);
    const _: () = assert!(size_of::<D3D12CommandQueueDesc>() == 16);
    const _: () = assert!(size_of::<D3D12HeapProperties>() == 20);
    const _: () = assert!(size_of::<D3D12ResourceDesc>() == 56);
    const _: () = assert!(size_of::<D3D12DescriptorHeapDesc>() == 16);
    const _: () = assert!(size_of::<D3D12CpuDescriptorHandle>() == 8);
    const _: () = assert!(size_of::<D3D12GraphicsPipelineStateDesc>() == 656);
    const _: () = assert!(size_of::<D3D12XboxPresentPlaneParameters>() == 72);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn GetLastError() -> u32;
        pub fn VirtualAlloc(
            addr: *mut c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut c_void;
        pub fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        pub fn VirtualQueryEx(
            process: *mut c_void,
            addr: *const c_void,
            buf: *mut MemoryBasicInformation,
            len: usize,
        ) -> usize;
    }

    #[link(name = "kernelx", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XMemAlloc(size: usize, attrs: u64) -> *mut c_void;
        pub fn XMemFree(ptr: *mut c_void, attrs: u64);
        pub fn GetConsoleType() -> u32;
        pub fn TitleMemoryStatus(out: *mut TitleMemoryStatus) -> i32;
        #[link_name = "VirtualAlloc"]
        pub fn KernelxVirtualAlloc(
            addr: *mut c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut c_void;
        #[link_name = "VirtualAllocFromApp"]
        pub fn KernelxVirtualAllocFromApp(
            addr: *mut c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut c_void;
        #[link_name = "VirtualFree"]
        pub fn KernelxVirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        #[link_name = "VirtualQueryEx"]
        pub fn KernelxVirtualQueryEx(
            process: *mut c_void,
            addr: *const c_void,
            buf: *mut MemoryBasicInformation,
            len: usize,
        ) -> usize;
        #[link_ordinal(999)]
        pub fn x2l_synthetic_ordinal_probe() -> u64;
    }

    #[link(name = "xmem", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XMemAllocDefault(size: usize, attrs: u64) -> *mut c_void;
        pub fn XMemFreeDefault(ptr: *mut c_void, attrs: u64);
        pub fn XMemAllocatePhysicalPages(
            flags: u64,
            count: *mut usize,
            out_pfns: *mut usize,
        ) -> i32;
        pub fn XMemFreePhysicalPages(count: usize, pfns: *const usize) -> i32;
        pub fn XMemMapPhysicalPages(
            base: *mut c_void,
            count: usize,
            pfns: *const usize,
        ) -> *mut c_void;
        pub fn XMemMapPhysicalPagesScatter(
            addresses: *const *mut c_void,
            count: usize,
            pfns: *const usize,
        ) -> i32;
        pub fn XMemVirtualAlloc(
            base: *mut c_void,
            size: usize,
            alloc_type: u32,
            flags: u64,
            protect: u32,
        ) -> *mut c_void;
        pub fn XMemVirtualQuery(
            process: *mut c_void,
            addr: *const c_void,
            out_info: *mut XMemBasicInformation,
        ) -> i32;
        pub fn XMemGetWorkingSetStatistics(flags: u32, out_buf: *mut u8) -> i32;
        pub fn XMemIsToolsMemAvailable() -> i32;
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
        pub fn D3D12XboxCreateDevice(
            adapter: *mut c_void,
            params: *const D3D12XboxCreateDeviceParameters,
            riid: *const c_void,
            out_device: *mut *mut c_void,
        ) -> i32;
        pub fn D3D12ConfigureVirtualMemoryX(config: *mut D3D12ConfigureVirtualMemoryXParams)
        -> i32;
        pub fn D3D12SerializeVersionedRootSignature(
            root_sig_desc: *const D3D12VersionedRootSignatureDesc,
            out_blob: *mut *mut c_void,
            out_error_blob: *mut *mut c_void,
        ) -> i32;
    }

    #[link(name = "xg_xs", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XGCreateTextureComputer(
            desc: *const D3D12ResourceDesc,
            out_computer: *mut *mut c_void,
        ) -> i32;
        pub fn XGSuggestSwizzleMode(
            desc: *const D3D12ResourceDesc,
            flags: u32,
            out_layout: *mut u32,
        ) -> i32;
    }

    #[link(name = "xcrdapi", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn DStorageGetFactory(riid: *const c_void, ppv: *mut *mut c_void) -> i32;
    }

    #[link(name = "gameinput", kind = "raw-dylib")]
    unsafe extern "system" {
        #[link_ordinal(1)]
        pub fn GameInputCreate(out_gameinput: *mut *mut c_void) -> i32;
    }

    #[link(name = "acphal", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn AcpHalCreate(out_acp: *mut *mut c_void) -> i32;
        pub fn AcpHalAllocateShapeContexts(contexts: *mut c_void) -> i32;
        pub fn AcpHalReleaseShapeContexts() -> i32;
        pub fn ApuCreateHeap(cached_bytes: u32, noncached_bytes: u32) -> i32;
        pub fn ApuAlloc(
            out_va: *mut *mut c_void,
            out_apu_addr: *mut u32,
            size: u32,
            alignment: u32,
            flags: u32,
        ) -> i32;
        pub fn ApuFree(va: *mut c_void) -> i32;
        pub fn ApuMapApuAddress(apu_addr: u32) -> *mut c_void;
        pub fn ApuIsVirtualAddressValid(va: *const c_void, size: u32) -> i32;
    }

    #[link(name = "xapu", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XApuConnect(
            input: *const c_void,
            output: *mut c_void,
            out_conn: *mut *mut c_void,
        ) -> i32;
        pub fn XApuDisconnect(conn: *mut c_void) -> i32;
        pub fn XApuEnqueueCommand(
            conn: *mut c_void,
            cmd: *const c_void,
            out_ticket: *mut u32,
        ) -> i32;
        pub fn XApuDequeueResult(conn: *mut c_void, out_result: *mut c_void) -> i32;
        pub fn XApuGetResultCount(conn: *mut c_void, out_count: *mut u32) -> u32;
    }

    #[link(name = "chad", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn XDspConnectWithMaximumStreamLimit(
            ctx: *mut c_void,
            max_streams: u32,
            a2: u32,
            a3: u32,
            a4: u32,
            out_dsp: *mut *mut c_void,
        ) -> i32;
        pub fn XDspDisconnect(dsp: *mut c_void) -> i32;
        pub fn XDspActivate(
            dsp: *mut c_void,
            stream_desc: *const c_void,
            out_status: *mut *const c_void,
            out_stream: *mut *mut c_void,
        ) -> i32;
        pub fn XDspDeactivate(stream: *mut c_void) -> i32;
        pub fn XDspSubmitCommand(
            stream: *mut c_void,
            cmd: *const c_void,
            out_ticket: *mut u32,
        ) -> i32;
    }

    #[link(name = "api-ms-win-core-psm-appnotify-l1-1-0", kind = "raw-dylib")]
    unsafe extern "system" {
        pub fn RegisterAppStateChangeNotification(
            callback: *const c_void,
            context: *mut c_void,
            out_registration: *mut *mut c_void,
        ) -> u32;
        pub fn UnregisterAppStateChangeNotification(registration: *mut c_void);
        pub fn RegisterAppConstrainedChangeNotification(
            callback: *const c_void,
            context: *mut c_void,
            out_registration: *mut *mut c_void,
        ) -> u32;
        pub fn UnregisterAppConstrainedChangeNotification(registration: *mut c_void);
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

    const fn s_load_dword(sdata: u8, sbase: u8, soffset: u8, ioffset: i32) -> [u32; 2] {
        let w0 = (0x3Du32 << 26)
            | (((sdata as u32) & 0x7F) << 6)
            | (((sbase as u32) >> 1) & 0x3F);
        let w1 = (((soffset as u32) & 0x7F) << 25) | ((ioffset as u32) & 0x1F_FFFF);
        [w0, w1]
    }

    const fn v_add_nc_u32(vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (0x25u32 << 25) | ((vdst as u32) << 17) | ((vsrc1 as u32) << 9) | ((src0 as u32) & 0x1FF)
    }

    const fn v_mul_lo_u32(vdst: u8, s0: u16, s1: u16) -> [u32; 2] {
        let w0 = (0x35u32 << 26) | (0x169u32 << 16) | (vdst as u32);
        let w1 = (((s1 as u32) & 0x1FF) << 9) | ((s0 as u32) & 0x1FF);
        [w0, w1]
    }

    const fn buffer_store_dword(
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: u16,
        offset: u16,
        offen: bool,
    ) -> [u32; 2] {
        let w0 = (0x38u32 << 26)
            | (0x1Cu32 << 18)
            | ((offen as u32) << 12)
            | ((offset as u32) & 0xFFF);
        let w1 = (((soffset as u32) & 0xFF) << 24)
            | ((((srsrc as u32) >> 2) & 0x1F) << 16)
            | ((vdata as u32) << 8)
            | (vaddr as u32);
        [w0, w1]
    }

    /// Pass-through VS taking `POSITION` in `v0..v3`, root constants `s0, s1` (R, G), and
    /// inline CBV word 0 at `s4:s5` (B) -> exports `pos0` and `param0`.
    pub fn synthetic_indexed_depth_vs_bytes() -> Vec<u8> {
        let cbv_ld = s_load_dword(8, 4, 0x7D, 0);
        let pos0 = exp(12, 0xF, false, false, false, [0, 1, 2, 3]);
        let param0 = exp(32, 0xF, false, true, false, [4, 5, 6, 7]);
        let words: &[u32] = &[
            cbv_ld[0],
            cbv_ld[1],
            v_mov_b32(4, 0), // s0 = R
            v_mov_b32(5, 1), // s1 = G
            v_mov_b32(6, 8), // s8 = CBV[0] = B
            v_mov_b32(7, F_1_0),
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

    /// Compute shader: reads root consts `(s0, s1)`, CBV word 0 (`s2:s3 -> s8`), computes
    /// `s0 + s8 + v0 * s1`, and stores to UAV at byte offset `v0 * 4`.
    pub fn synthetic_compute_signature_cs_bytes() -> Vec<u8> {
        let cbv_ld = s_load_dword(8, 2, 0x7D, 0);
        let mul1 = v_mul_lo_u32(1, v_reg(0), 1); // v1 = v0 * s1
        let mul2 = v_mul_lo_u32(2, v_reg(0), 128 + 4); // v2 = v0 * 4
        let st = buffer_store_dword(1, 2, 12, INLINE_0, 0, true);
        let words: &[u32] = &[
            cbv_ld[0],
            cbv_ld[1],
            mul1[0],
            mul1[1],
            v_add_nc_u32(1, 0, 1), // v1 += s0
            v_add_nc_u32(1, 8, 1), // v1 += s8
            mul2[0],
            mul2[1],
            st[0],
            st[1],
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

fn requested_frame_count() -> u32 {
    let args: Vec<String> = std::env::args().collect();
    for pair in args.windows(2) {
        if pair[0] == "--frames"
            && let Ok(n) = pair[1].parse::<u32>()
            && n > 0
        {
            return n;
        }
    }
    1
}

fn main() -> ExitCode {
    let mut failures = 0u32;

    #[cfg(target_os = "windows")]
    run_console_api_checks(
        &mut failures,
        should_render_vulkan(),
        requested_frame_count(),
    );

    // Multi-threaded static TLS + panic/unwind handling
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

    // OverlayFS copy-on-write & whiteout checks (active when fixture files exist)
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
fn run_console_api_checks(failures: &mut u32, render_vulkan: bool, frame_count: u32) {
    use console_imports::*;
    use std::ffi::c_void;
    use std::ptr;

    // 1. kernelx!GetConsoleType & TitleMemoryStatus & XMemAlloc/XMemFree
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

    // 2. XGameRuntimeInitialize / offline XStore / XGameRuntimeUninitialize
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

    // 3. d3d12_xs: D3D12CreateDevice smoke + D3D12ConfigureVirtualMemoryX +
    //    D3D12SerializeVersionedRootSignature + D3D12XboxCreateDevice + probe[45] (+ Vulkan draw)
    let mut legacy_dev: *mut c_void = ptr::null_mut();
    // SAFETY: Verifies base `D3D12CreateDevice` remains intact alongside `D3D12XboxCreateDevice`.
    let legacy_ok = unsafe {
        if D3D12CreateDevice(ptr::null_mut(), 0xb000, ptr::null(), &mut legacy_dev) == 0
            && !legacy_dev.is_null()
        {
            let vt = *(legacy_dev as *const *const usize);
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            rel(legacy_dev) == 0
        } else {
            false
        }
    };
    check("d3d12_xs!D3D12CreateDevice + Release", legacy_ok, failures);

    let mut vm_cfg = D3D12ConfigureVirtualMemoryXParams::default();
    // SAFETY: `vm_cfg` is a stack-owned 32-byte struct.
    let vm_ok = unsafe { D3D12ConfigureVirtualMemoryX(&mut vm_cfg) } == 0
        && vm_cfg.low_address_out > 0
        && vm_cfg.high_address_out > vm_cfg.low_address_out;
    check("d3d12_xs!D3D12ConfigureVirtualMemoryX", vm_ok, failures);

    let rs_desc = D3D12VersionedRootSignatureDesc {
        version: 2,
        _pad04: 0,
        num_parameters: 0,
        _pad0c: 0,
        p_parameters: ptr::null(),
        num_static_samplers: 0,
        _pad1c: 0,
        p_static_samplers: ptr::null(),
        flags: 0,
        _pad2c: 0,
    };
    let mut rs_blob: *mut c_void = ptr::null_mut();
    // SAFETY: `rs_desc` and `rs_blob` are valid stack variables.
    let rs_ok = unsafe {
        let hr = D3D12SerializeVersionedRootSignature(&rs_desc, &mut rs_blob, ptr::null_mut());
        if hr != 0 || rs_blob.is_null() {
            false
        } else {
            let vt = *(rs_blob as *const *const usize);
            let get_ptr: extern "win64" fn(*mut c_void) -> *mut c_void =
                std::mem::transmute(*vt.add(3));
            let get_size: extern "win64" fn(*mut c_void) -> usize = std::mem::transmute(*vt.add(4));
            let release: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            let ptr_b = get_ptr(rs_blob);
            let sz = get_size(rs_blob);
            let valid = !ptr_b.is_null()
                && sz >= 68
                && std::slice::from_raw_parts(ptr_b.cast::<u8>(), 4) == b"DXBC";
            let rem = release(rs_blob);
            valid && rem == 0
        }
    };
    check(
        "d3d12_xs!D3D12SerializeVersionedRootSignature (ID3DBlob DXBC/RTS0)",
        rs_ok,
        failures,
    );

    let dev_params = D3D12XboxCreateDeviceParameters {
        version: 1,
        ..Default::default()
    };
    let mut dev: *mut c_void = ptr::null_mut();
    // SAFETY: `dev_params` and `dev` are valid pointers.
    let d3d_hr =
        unsafe { D3D12XboxCreateDevice(ptr::null_mut(), &dev_params, ptr::null(), &mut dev) };
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
                let expanded_ok = run_vulkan_expanded_d3d12x_checks(dev);
                check(
                    "d3d12_xs!M3.3 placed VB/IB + CopyBufferRegion + depth-tested DrawIndexedInstanced + Compute Dispatch",
                    expanded_ok,
                    failures,
                );
                let draw_ok = run_vulkan_triangle_draw(dev, frame_count);
                check(
                    &format!(
                        "d3d12_xs!Vulkan 64x64 triangle draw ({frame_count} frame(s)) + fence + PresentX"
                    ),
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
        "d3d12_xs!D3D12XboxCreateDevice + spare vtable probe[45] + Release",
        d3d_ok,
        failures,
    );

    // 4. xmem.dll: XMemAllocDefault, physical pages + scatter map + XMemVirtualQuery + stats
    // SAFETY: Exercises Linux `memfd_create`/`mmap` physical & scatter arena through `xmem.dll`.
    let xmem_dll_ok = unsafe {
        let p = XMemAllocDefault(256, 12u64 << 24) as *mut u8;
        if p.is_null() {
            false
        } else {
            ptr::write_volatile(p, 0x5A);
            let ok_heap = ptr::read_volatile(p) == 0x5A;
            XMemFreeDefault(p.cast(), 0);

            let mut count = 2usize;
            let mut pfns = [0usize; 2];
            let alloc_ok = XMemAllocatePhysicalPages(0, &mut count, pfns.as_mut_ptr()) != 0
                && count == 2
                && pfns[0] != 0
                && pfns[1] != 0;

            let va = XMemVirtualAlloc(ptr::null_mut(), 2 * 65536, 0x2000, 0, 0x04);
            let mut scatter_ok = false;
            if alloc_ok && !va.is_null() {
                let addrs = [va, (va as usize + 65536) as *mut c_void];
                let rev_pfns = [pfns[1], pfns[0]];
                if XMemMapPhysicalPagesScatter(addrs.as_ptr(), 2, rev_pfns.as_ptr()) != 0 {
                    ptr::write_volatile(va as *mut u32, 0xCAFE_BABE);
                    ptr::write_volatile((va as usize + 65536 + 8) as *mut u32, 0x1337_C0DE);
                    let rd_ok = ptr::read_volatile(va as *const u32) == 0xCAFE_BABE
                        && ptr::read_volatile((va as usize + 65536 + 8) as *const u32)
                            == 0x1337_C0DE;

                    let mut q = XMemBasicInformation::default();
                    let q_ok = XMemVirtualQuery(ptr::null_mut(), va, &mut q) != 0
                        && q.state == 0x1000
                        && q.mem_type == 0x40000
                        && q.region_size >= 65536
                        && (q.flags & 0b11) == 0b11;

                    let unmap_ok = !XMemMapPhysicalPages(va, 2, ptr::null()).is_null();
                    scatter_ok = rd_ok && q_ok && unmap_ok;
                }
            }
            let free_ok = XMemFreePhysicalPages(2, pfns.as_ptr()) != 0;

            let mut ws = [0u8; 80];
            let ws_ok = XMemGetWorkingSetStatistics(0, ws.as_mut_ptr()) != 0
                && u64::from_le_bytes(ws[0..8].try_into().unwrap()) >= 8_000_000_000;
            let tools_ok = XMemIsToolsMemAvailable() == 0;

            ok_heap && alloc_ok && scatter_ok && free_ok && ws_ok && tools_ok
        }
    };
    check(
        "xmem.dll!AllocDefault + PhysicalPages + ScatterMap + VirtualQuery + WorkingSet",
        xmem_dll_ok,
        failures,
    );

    // 5. xg_xs.dll: XGSuggestSwizzleMode + XGCreateTextureComputer + GetResourceLayout
    // SAFETY: Exercises `xg_xs.dll` backed by `x2l-formats::texture`.
    let xg_ok = unsafe {
        let mut tex_desc = D3D12ResourceDesc {
            dimension: 3, // TEXTURE2D
            alignment: 0,
            width: 256,
            height: 256,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28, // R8G8B8A8_UNORM
            sample_desc: DxgiSampleDesc {
                count: 1,
                quality: 0,
            },
            layout: 0,
            flags: 0,
        };
        let mut suggested = 0u32;
        if XGSuggestSwizzleMode(&tex_desc, 0, &mut suggested) != 0 || (suggested & 0x100) == 0 {
            false
        } else {
            tex_desc.layout = suggested;
            let mut comp: *mut c_void = ptr::null_mut();
            if XGCreateTextureComputer(&tex_desc, &mut comp) != 0 || comp.is_null() {
                false
            } else {
                let vt = *(comp as *const *const usize);
                // IXGTextureAddressComputer: 0=AddRef, 1=Release, 2=GetResourceLayout, 3=GetResourceSizeBytes
                let get_layout: extern "win64" fn(*mut c_void, *mut u8) -> i32 =
                    std::mem::transmute(*vt.add(2));
                let get_size: extern "win64" fn(*mut c_void) -> u64 =
                    std::mem::transmute(*vt.add(3));
                let release: extern "win64" fn(*mut c_void) -> u32 =
                    std::mem::transmute(*vt.add(1));

                let mut layout_buf = vec![0u8; 0x1B40];
                let hr = get_layout(comp, layout_buf.as_mut_ptr());
                let size_bytes = u64::from_le_bytes(layout_buf[0..8].try_into().unwrap());
                let align_bytes = u64::from_le_bytes(layout_buf[8..16].try_into().unwrap());
                let direct_size = get_size(comp);
                let rem = release(comp);
                hr == 0
                    && size_bytes > 0
                    && size_bytes == direct_size
                    && align_bytes >= 4096
                    && rem == 0
            }
        }
    };
    check(
        "xg_xs.dll!XGSuggestSwizzleMode + XGCreateTextureComputer::GetResourceLayout",
        xg_ok,
        failures,
    );

    // 6. XCRDAPI.dll: DStorageGetFactory + CreateQueue + OpenFile/Read on G:\fixture_keep.txt
    // SAFETY: Exercises `XCRDAPI.dll` DirectStorage COM interfaces.
    let dstorage_ok = unsafe {
        let mut fac: *mut c_void = ptr::null_mut();
        if DStorageGetFactory(ptr::null(), &mut fac) != 0 || fac.is_null() {
            false
        } else {
            let fvt = *(fac as *const *const usize);
            let create_q: extern "win64" fn(
                *mut c_void,
                *const c_void,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(*fvt.add(3));
            let open_file: extern "win64" fn(
                *mut c_void,
                *const u16,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(*fvt.add(4));
            let create_status: extern "win64" fn(
                *mut c_void,
                u32,
                *const u8,
                *const c_void,
                *mut *mut c_void,
            ) -> i32 = std::mem::transmute(*fvt.add(5));
            let fac_rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*fvt.add(2));

            let mut queue: *mut c_void = ptr::null_mut();
            let mut status_arr: *mut c_void = ptr::null_mut();
            let mut ok = create_q(fac, ptr::null(), ptr::null(), &mut queue) == 0
                && !queue.is_null()
                && create_status(fac, 4, ptr::null(), ptr::null(), &mut status_arr) == 0
                && !status_arr.is_null();

            if ok && Path::new("fixture_keep.txt").is_file() {
                let path_u16: Vec<u16> = "G:\\fixture_keep.txt\0".encode_utf16().collect();
                let mut ds_file: *mut c_void = ptr::null_mut();
                if open_file(fac, path_u16.as_ptr(), ptr::null(), &mut ds_file) == 0
                    && !ds_file.is_null()
                {
                    let qvt = *(queue as *const *const usize);
                    let enqueue_req: extern "win64" fn(*mut c_void, *const DStorageRequest) =
                        std::mem::transmute(*qvt.add(3));
                    let enqueue_st: extern "win64" fn(*mut c_void, *mut c_void, u32) =
                        std::mem::transmute(*qvt.add(4));
                    let submit: extern "win64" fn(*mut c_void) = std::mem::transmute(*qvt.add(6));

                    let mut dst_buf = [0u8; 22];
                    let req = DStorageRequest {
                        options: 0,
                        dst_ptr: dst_buf.as_mut_ptr() as usize as u64,
                        dst_page_offset: 0,
                        dst_size: dst_buf.len() as u32,
                        _pad1c: 0,
                        src_ptr: ds_file as usize as u64,
                        file_offset: 0,
                        src_size: dst_buf.len() as u32,
                        intermediate_size: 0,
                        cancellation_tag: 0,
                        name: ptr::null(),
                    };
                    enqueue_req(queue, &req);
                    enqueue_st(queue, status_arr, 0);
                    submit(queue);

                    let svt = *(status_arr as *const *const usize);
                    let is_complete: extern "win64" fn(*mut c_void, u32) -> bool =
                        std::mem::transmute(*svt.add(3));
                    let get_hr: extern "win64" fn(*mut c_void, u32) -> i32 =
                        std::mem::transmute(*svt.add(4));
                    ok = is_complete(status_arr, 0)
                        && get_hr(status_arr, 0) == 0
                        && &dst_buf == b"immutable-keep-payload";

                    let file_vt = *(ds_file as *const *const usize);
                    let file_close: extern "win64" fn(*mut c_void) =
                        std::mem::transmute(*file_vt.add(3));
                    let file_rel: extern "win64" fn(*mut c_void) -> u32 =
                        std::mem::transmute(*file_vt.add(2));
                    file_close(ds_file);
                    ok = ok && file_rel(ds_file) == 0;
                } else {
                    ok = false;
                }
            }

            if !status_arr.is_null() {
                let svt = *(status_arr as *const *const usize);
                let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*svt.add(2));
                ok = ok && rel(status_arr) == 0;
            }
            if !queue.is_null() {
                let qvt = *(queue as *const *const usize);
                let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*qvt.add(2));
                ok = ok && rel(queue) == 0;
            }
            ok && fac_rel(fac) == 0
        }
    };
    check(
        "XCRDAPI.dll!DStorageGetFactory + OpenFile + Queue Submit roundtrip",
        dstorage_ok,
        failures,
    );

    // 7. GameInput.dll ordinal #1 (GameInputCreate)
    // SAFETY: Imports ordinal `#1` from `gameinput.dll` and exercises `IGameInput` vtable.
    let gi_ok = unsafe {
        let mut gi: *mut c_void = ptr::null_mut();
        if GameInputCreate(&mut gi) != 0 || gi.is_null() {
            false
        } else {
            let vt = *(gi as *const *const usize);
            let get_ts: extern "win64" fn(*mut c_void) -> u64 = std::mem::transmute(*vt.add(3));
            let get_cur: extern "win64" fn(*mut c_void, u32, *mut c_void, *mut *mut c_void) -> i32 =
                std::mem::transmute(*vt.add(4));
            let reg_dev: extern "win64" fn(
                *mut c_void,
                *mut c_void,
                u32,
                u32,
                u32,
                *mut c_void,
                *mut c_void,
                *mut u64,
            ) -> i32 = std::mem::transmute(*vt.add(9));
            let unreg: extern "win64" fn(*mut c_void, u64, u64) -> bool =
                std::mem::transmute(*vt.add(13));
            let release: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));

            let _ts = get_ts(gi);
            let mut reading: *mut c_void = ptr::null_mut();
            let hr_read = get_cur(gi, 0x0002_0000, ptr::null_mut(), &mut reading);
            let read_ok = if hr_read == 0 && !reading.is_null() {
                // Live Linux `evdev` gamepad attached: verify `IGameInputReading` vtable.
                let r_vt = *(reading as *const *const usize);
                let r_kind: extern "win64" fn(*mut c_void) -> u32 =
                    std::mem::transmute(*r_vt.add(3));
                let r_gp: extern "win64" fn(*mut c_void, *mut GameInputGamepadState) -> bool =
                    std::mem::transmute(*r_vt.add(22));
                let r_rel: extern "win64" fn(*mut c_void) -> u32 =
                    std::mem::transmute(*r_vt.add(2));
                let mut gp = GameInputGamepadState::default();
                let ok = (r_kind(reading) & 0x0002_0000) != 0 && r_gp(reading, &mut gp);
                let rem_r = r_rel(reading);
                ok && rem_r == 0
            } else {
                hr_read == -2088108025 && reading.is_null()
            };
            let mut tok = 0u64;
            let hr_reg = reg_dev(
                gi,
                ptr::null_mut(),
                0x0002_0000,
                1,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut tok,
            );
            let unreg_ok = tok != 0 && unreg(gi, tok, 0);
            let rem = release(gi);
            read_ok && hr_reg == 0 && unreg_ok && rem == 0
        }
    };
    check(
        "GameInput.dll!#1 GameInputCreate + IGameInput GetCurrentReading lifecycle",
        gi_ok,
        failures,
    );

    // 7b. Safe `VirtualAlloc` console flag-stripping (both IAT-patched kernel32 & explicit kernelx)
    // SAFETY: Exercises IAT-patched `kernel32!VirtualAlloc` and `kernelx!VirtualAlloc*` + `VirtualQueryEx` + `VirtualFree`.
    let valloc_ok = unsafe {
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RESERVE: u32 = 0x2000;
        const MEM_RELEASE: u32 = 0x8000;
        const MEM_FREE: u32 = 0x10000;
        const MEM_PRIVATE: u32 = 0x20000;
        const MEM_GRAPHICS: u32 = 0x1000_0000;
        const MEM_64K_PAGES: u32 = 0x2040_0000;
        const MEM_TITLE: u32 = 0x4000_0000;
        const MEM_4MB_PAGES: u32 = 0x8000_0000;
        const PAGE_READWRITE: u32 = 0x04;
        const PAGE_EXECUTE_READWRITE: u32 = 0x40;
        const PAGE_GRAPHICS_READWRITE: u32 = 0x2000;
        const PAGE_GRAPHICS_COHERENT: u32 = 0x20000;

        let proc = GetCurrentProcess();

        // Route A: static `kernel32!VirtualAlloc` intercepted by startup PE32+ IAT patcher
        let p_iat = VirtualAlloc(
            ptr::null_mut(),
            65536,
            MEM_COMMIT | MEM_RESERVE | MEM_GRAPHICS | MEM_TITLE | MEM_4MB_PAGES,
            PAGE_READWRITE | PAGE_GRAPHICS_READWRITE | PAGE_GRAPHICS_COHERENT,
        );
        let mut iat_ok = false;
        if !p_iat.is_null() {
            *(p_iat as *mut u64) = 0x5832_4C5F_5641_4C4C;
            let val = *(p_iat as *const u64);
            let mut mbi = MemoryBasicInformation::default();
            let q1 = VirtualQueryEx(
                proc,
                p_iat,
                &mut mbi,
                std::mem::size_of::<MemoryBasicInformation>(),
            );
            let freed = VirtualFree(p_iat, 0, MEM_RELEASE) != 0;
            let mut mbi_after = MemoryBasicInformation::default();
            let q2 = VirtualQueryEx(
                proc,
                p_iat,
                &mut mbi_after,
                std::mem::size_of::<MemoryBasicInformation>(),
            );
            iat_ok = val == 0x5832_4C5F_5641_4C4C
                && q1 == 48
                && mbi.state == MEM_COMMIT
                && mbi.mem_type == MEM_PRIVATE
                && mbi.protect == PAGE_READWRITE
                && freed
                && q2 == 48
                && mbi_after.state == MEM_FREE;
        }

        // Route B: explicit `kernelx!VirtualAlloc` with 64 KiB composite + graphics flags
        let p_kx = KernelxVirtualAlloc(
            ptr::null_mut(),
            65536,
            MEM_COMMIT | MEM_RESERVE | MEM_GRAPHICS | MEM_64K_PAGES,
            PAGE_READWRITE | PAGE_GRAPHICS_READWRITE,
        );
        let mut kx_ok = false;
        if !p_kx.is_null() {
            *(p_kx as *mut u32) = 0xCAFE_BABE;
            let val_kx = *(p_kx as *const u32);
            let mut mbi = MemoryBasicInformation::default();
            let q = KernelxVirtualQueryEx(proc, p_kx, &mut mbi, 48);
            let freed = KernelxVirtualFree(p_kx, 0, MEM_RELEASE) != 0;
            kx_ok = val_kx == 0xCAFE_BABE && q == 48 && mbi.state == MEM_COMMIT && freed;
        }

        // Route C: `kernelx!VirtualAllocFromApp` rejects executable CPU protect with error 87
        let p_exec = KernelxVirtualAllocFromApp(
            ptr::null_mut(),
            4096,
            MEM_COMMIT | MEM_RESERVE | MEM_GRAPHICS,
            PAGE_EXECUTE_READWRITE,
        );
        let exec_rejected = p_exec.is_null() && GetLastError() == 87;
        let p_app = KernelxVirtualAllocFromApp(
            ptr::null_mut(),
            4096,
            MEM_COMMIT | MEM_RESERVE | MEM_GRAPHICS,
            PAGE_READWRITE | PAGE_GRAPHICS_COHERENT,
        );
        let app_ok = !p_app.is_null() && VirtualFree(p_app, 0, MEM_RELEASE) != 0;

        // Control: plain unmarked Win32 allocation still routes cleanly through Wine
        let p_plain = VirtualAlloc(
            ptr::null_mut(),
            65536,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        let plain_ok = !p_plain.is_null() && VirtualFree(p_plain, 0, MEM_RELEASE) != 0;

        let all_ok = iat_ok && kx_ok && exec_rejected && app_ok && plain_ok;
        if !all_ok {
            eprintln!(
                "[DIAG] VirtualAlloc details: iat_ok={iat_ok} kx_ok={kx_ok} exec_rejected={exec_rejected} app_ok={app_ok} plain_ok={plain_ok}"
            );
        }
        all_ok
    };
    check(
        "VirtualAlloc console flag-stripping (IAT + kernelx + VirtualQueryEx + VirtualFree)",
        valloc_ok,
        failures,
    );

    // 8. Audio co-processor shims (AcpHal, xapu, chad) & PSM AppNotify
    // SAFETY: Exercises AcpHal/xapu/chad/appnotify PE shims.
    unsafe {
        let mut acp: *mut c_void = ptr::null_mut();
        let mut va: *mut c_void = ptr::null_mut();
        let mut apu = 0u32;
        let acp_ok = ApuCreateHeap(4 * 1024 * 1024, 1024 * 1024) == 0
            && AcpHalAllocateShapeContexts(ptr::null_mut()) == 0
            && AcpHalCreate(&mut acp) == 0
            && !acp.is_null()
            && ApuAlloc(&mut va, &mut apu, 512, 64, 0) == 0
            && !va.is_null()
            && apu != 0
            && ApuIsVirtualAddressValid(va, 512) != 0
            && ApuMapApuAddress(apu) == va
            && ApuFree(va) == 0
            && AcpHalReleaseShapeContexts() == 0;
        let acp_rel_ok = if !acp.is_null() {
            let vt = *(acp as *const *const usize);
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            rel(acp) == 0
        } else {
            false
        };

        let mut xapu_conn: *mut c_void = ptr::null_mut();
        let mut ticket = 0u32;
        let mut res_count = 0u32;
        let mut res_val = 0u32;
        // Synthetic stereo PCM frame packet (`magic = 0x58415055`, `sample_count = 128`, 128 f32 zeros)
        let mut xapu_pkt = [0u32; 2 + 128];
        xapu_pkt[0] = 0x5841_5055;
        xapu_pkt[1] = 128;
        let xapu_ok = XApuConnect(ptr::null(), ptr::null_mut(), &mut xapu_conn) == 0
            && !xapu_conn.is_null()
            && XApuEnqueueCommand(xapu_conn, xapu_pkt.as_ptr().cast(), &mut ticket) == 0
            && XApuGetResultCount(xapu_conn, &mut res_count) == 1
            && res_count == 1
            && XApuDequeueResult(xapu_conn, (&mut res_val as *mut u32).cast()) == 0
            && res_val == ticket
            && XApuDisconnect(xapu_conn) == 0;

        let mut dsp: *mut c_void = ptr::null_mut();
        let mut stream: *mut c_void = ptr::null_mut();
        let mut status_ptr: *const c_void = ptr::null();
        let mut dsp_ticket = 0u32;
        let chad_ok = XDspConnectWithMaximumStreamLimit(ptr::null_mut(), 4, 0, 0, 0, &mut dsp) == 0
            && !dsp.is_null()
            && XDspActivate(dsp, ptr::null(), &mut status_ptr, &mut stream) == 0
            && !stream.is_null()
            && XDspSubmitCommand(stream, ptr::null(), &mut dsp_ticket) == 0
            && dsp_ticket == 1
            && XDspDeactivate(stream) == 0
            && XDspDisconnect(dsp) == 0;

        let mut reg_state: *mut c_void = ptr::null_mut();
        let mut reg_constr: *mut c_void = ptr::null_mut();
        let psm_ok =
            RegisterAppStateChangeNotification(ptr::null(), ptr::null_mut(), &mut reg_state) == 0
                && !reg_state.is_null()
                && RegisterAppConstrainedChangeNotification(
                    ptr::null(),
                    ptr::null_mut(),
                    &mut reg_constr,
                ) == 0
                && !reg_constr.is_null();
        if !reg_state.is_null() {
            UnregisterAppStateChangeNotification(reg_state);
        }
        if !reg_constr.is_null() {
            UnregisterAppConstrainedChangeNotification(reg_constr);
        }

        check(
            "AcpHal.dll!ApuCreateHeap + AcpHalCreate + ApuAlloc/Map/Free",
            acp_ok && acp_rel_ok,
            failures,
        );
        check(
            "xapu.dll!XApuConnect + Enqueue/Dequeue + Disconnect",
            xapu_ok,
            failures,
        );
        check(
            "chad.dll!XDspConnect + Activate + SubmitCommand + Disconnect",
            chad_ok,
            failures,
        );
        check(
            "api-ms-win-core-psm-appnotify!Register/Unregister lifecycle",
            psm_ok,
            failures,
        );
    };

    // 9. Unknown ordinal kernelx.dll!#999 auto-stub
    // SAFETY: Exported via `kernelx.def` NONAME ordinal `#999` auto-stub thunk.
    let ord_ret = unsafe { x2l_synthetic_ordinal_probe() };
    check(
        &format!("kernelx.dll!#999 unknown-ordinal auto-stub (ret=0x{ord_ret:x})"),
        ord_ret == 0,
        failures,
    );
}

#[cfg(target_os = "windows")]
unsafe fn run_vulkan_triangle_draw(dev: *mut std::ffi::c_void, frame_count: u32) -> bool {
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
        let f_vt = *(fence as *const *const usize);
        let f_completed: extern "win64" fn(*mut c_void) -> u64 = std::mem::transmute(*f_vt.add(8));
        let f_wait: extern "win64" fn(*mut c_void, u64, *mut c_void) -> i32 =
            std::mem::transmute(*f_vt.add(9));

        let rt_list = [rt];
        let plane = D3D12XboxPresentPlaneParameters {
            token: 0,
            resource_count: 1,
            _pad0: 0,
            pp_resources: rt_list.as_ptr(),
            _reserved: [0; 6],
        };

        let frames = frame_count.max(1);
        for frame_idx in 1..=frames {
            q_exec(queue, 1, lists.as_ptr());
            if q_signal(queue, fence, u64::from(frame_idx)) != 0 {
                return false;
            }
            if f_wait(fence, u64::from(frame_idx), ptr::null_mut()) != 0
                || f_completed(fence) < u64::from(frame_idx)
            {
                return false;
            }
            if q_present_x(queue, 1, &plane, ptr::null()) != 0 {
                return false;
            }
        }

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

/// M3.3 Win64 COM Vulkan checks inside Wine:
/// 1. `CreateRootSignature` + `CreateHeap` + `CreatePlacedResource` + VB/IB upload + `CopyBufferRegion`
/// 2. Depth-tested (`D32_FLOAT` + `ClearDepthStencilView`) `DrawIndexedInstanced` with root constants & CBV
/// 3. `CreateComputePipelineState` + `Dispatch(1, 1, 1)` + UAV table + `CopyBufferRegion` readback verification
#[cfg(target_os = "windows")]
unsafe fn run_vulkan_expanded_d3d12x_checks(dev: *mut std::ffi::c_void) -> bool {
    use console_imports::*;
    use std::ffi::c_void;
    use std::mem::MaybeUninit;
    use std::ptr;

    // SAFETY: `dev` is a valid `ID3D12DeviceX` COM pointer created by `D3D12CreateDevice`,
    // and all COM vtable indices and descriptor layouts match X2Linux's Win64 D3D12.X ABI.
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
        let create_gpso: extern "win64" fn(
            *mut c_void,
            *const D3D12GraphicsPipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(10));
        let create_cpso: extern "win64" fn(
            *mut c_void,
            *const D3D12ComputePipelineStateDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(11));
        let create_list: extern "win64" fn(
            *mut c_void,
            u32,
            u32,
            *mut c_void,
            *mut c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(12));
        let create_dheap: extern "win64" fn(
            *mut c_void,
            *const D3D12DescriptorHeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(14));
        let create_rs: extern "win64" fn(
            *mut c_void,
            u32,
            *const c_void,
            usize,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(16));
        let create_uav: extern "win64" fn(
            *mut c_void,
            *mut c_void,
            *mut c_void,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(19));
        let create_rtv: extern "win64" fn(
            *mut c_void,
            *mut c_void,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(20));
        let create_dsv: extern "win64" fn(
            *mut c_void,
            *mut c_void,
            *const c_void,
            D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*dev_vt.add(21));
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
        let create_heap: extern "win64" fn(
            *mut c_void,
            *const D3D12HeapDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(28));
        let create_placed: extern "win64" fn(
            *mut c_void,
            *mut c_void,
            u64,
            *const D3D12ResourceDesc,
            u32,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> i32 = std::mem::transmute(*dev_vt.add(29));

        let serialize_and_make_rs = |params: &[D3D12RootParameterRaw]| -> *mut c_void {
            let desc = D3D12VersionedRootSignatureDesc {
                version: 2,
                _pad04: 0,
                num_parameters: params.len() as u32,
                _pad0c: 0,
                p_parameters: params.as_ptr().cast(),
                num_static_samplers: 0,
                _pad1c: 0,
                p_static_samplers: ptr::null(),
                flags: 1,
                _pad2c: 0,
            };
            let mut blob: *mut c_void = ptr::null_mut();
            if D3D12SerializeVersionedRootSignature(&desc, &mut blob, ptr::null_mut()) != 0
                || blob.is_null()
            {
                return ptr::null_mut();
            }
            let bvt = *(blob as *const *const usize);
            let get_ptr: extern "win64" fn(*mut c_void) -> *mut c_void =
                std::mem::transmute(*bvt.add(3));
            let get_sz: extern "win64" fn(*mut c_void) -> usize = std::mem::transmute(*bvt.add(4));
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*bvt.add(2));
            let mut rs = ptr::null_mut();
            let hr = create_rs(dev, 0, get_ptr(blob), get_sz(blob), ptr::null(), &mut rs);
            let _ = rel(blob);
            if hr == 0 { rs } else { ptr::null_mut() }
        };

        // --- Part A: Placed VB/IB + CopyBufferRegion + Depth-tested DrawIndexedInstanced ---
        let gfx_params = [
            D3D12RootParameterRaw {
                parameter_type: 1, // 4 root constants
                _pad04: 0,
                data: [0, 0, 4, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 2, // inline CBV
                _pad04: 0,
                data: [1, 0, 0, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
        ];
        let gfx_rs = serialize_and_make_rs(&gfx_params);
        if gfx_rs.is_null() {
            return false;
        }

        let mut queue = ptr::null_mut();
        let mut alloc = ptr::null_mut();
        if create_queue(
            dev,
            &D3D12CommandQueueDesc::default(),
            ptr::null(),
            &mut queue,
        ) != 0
            || create_alloc(dev, 0, ptr::null(), &mut alloc) != 0
        {
            return false;
        }

        let hp = D3D12HeapProperties {
            r#type: 1,
            ..Default::default()
        };
        let mut gpu_heap = ptr::null_mut();
        if create_heap(
            dev,
            &D3D12HeapDesc {
                size_in_bytes: 131_072,
                properties: hp,
                alignment: 65_536,
                ..Default::default()
            },
            ptr::null(),
            &mut gpu_heap,
        ) != 0
        {
            return false;
        }

        let buf_desc = D3D12ResourceDesc {
            dimension: 1,
            width: 4096,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            ..Default::default()
        };
        let mut vb = ptr::null_mut();
        let mut ib = ptr::null_mut();
        let mut upload = ptr::null_mut();
        let mut cbv = ptr::null_mut();
        let mut rt = ptr::null_mut();
        let mut ds = ptr::null_mut();
        if create_placed(
            dev,
            gpu_heap,
            0,
            &buf_desc,
            0,
            ptr::null(),
            ptr::null(),
            &mut vb,
        ) != 0
            || create_placed(
                dev,
                gpu_heap,
                65_536,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut ib,
            ) != 0
            || create_res(
                dev,
                &hp,
                0,
                &buf_desc,
                0,
                ptr::null(),
                ptr::null(),
                &mut upload,
            ) != 0
            || create_res(dev, &hp, 0, &buf_desc, 0, ptr::null(), ptr::null(), &mut cbv) != 0
            || create_res(
                dev,
                &hp,
                0,
                &D3D12ResourceDesc {
                    dimension: 3,
                    width: 64,
                    height: 64,
                    depth_or_array_size: 1,
                    mip_levels: 1,
                    format: 28,
                    sample_desc: DxgiSampleDesc {
                        count: 1,
                        quality: 0,
                    },
                    flags: 1,
                    ..Default::default()
                },
                0,
                ptr::null(),
                ptr::null(),
                &mut rt,
            ) != 0
            || create_res(
                dev,
                &hp,
                0,
                &D3D12ResourceDesc {
                    dimension: 3,
                    width: 64,
                    height: 64,
                    depth_or_array_size: 1,
                    mip_levels: 1,
                    format: 40, // D32_FLOAT
                    sample_desc: DxgiSampleDesc {
                        count: 1,
                        quality: 0,
                    },
                    flags: 2,
                    ..Default::default()
                },
                0,
                ptr::null(),
                ptr::null(),
                &mut ds,
            ) != 0
        {
            return false;
        }

        let res_vt = *(upload as *const *const usize);
        let res_map: extern "win64" fn(*mut c_void, u32, *const c_void, *mut *mut c_void) -> i32 =
            std::mem::transmute(*res_vt.add(8));
        let res_unmap: extern "win64" fn(*mut c_void, u32, *const c_void) =
            std::mem::transmute(*res_vt.add(9));
        let res_va: extern "win64" fn(*mut c_void) -> u64 = std::mem::transmute(*res_vt.add(11));

        // Near triangle (0..3, z=0.25) + far triangle (3..6, z=0.75)
        let verts: [[f32; 4]; 6] = [
            [0.0, 0.5, 0.25, 1.0],
            [0.5, -0.5, 0.25, 1.0],
            [-0.5, -0.5, 0.25, 1.0],
            [0.0, 0.4, 0.75, 1.0],
            [0.4, -0.4, 0.75, 1.0],
            [-0.4, -0.4, 0.75, 1.0],
        ];
        let idxs: [u16; 6] = [0, 1, 2, 3, 4, 5];
        let mut p = ptr::null_mut();
        if res_map(upload, 0, ptr::null(), &mut p) != 0 {
            return false;
        }
        ptr::copy_nonoverlapping(
            verts.as_ptr().cast::<u8>(),
            p.cast::<u8>(),
            std::mem::size_of_val(&verts),
        );
        ptr::copy_nonoverlapping(
            idxs.as_ptr().cast::<u8>(),
            p.cast::<u8>().add(256),
            std::mem::size_of_val(&idxs),
        );
        res_unmap(upload, 0, ptr::null());

        let mut cp = ptr::null_mut();
        if res_map(cbv, 0, ptr::null(), &mut cp) != 0 {
            return false;
        }
        ptr::write_volatile(cp.cast::<u32>(), 0.0f32.to_bits());
        res_unmap(cbv, 0, ptr::null());

        let mut rtv_heap = ptr::null_mut();
        let mut dsv_heap = ptr::null_mut();
        if create_dheap(
            dev,
            &D3D12DescriptorHeapDesc {
                r#type: 2,
                num_descriptors: 1,
                ..Default::default()
            },
            ptr::null(),
            &mut rtv_heap,
        ) != 0
            || create_dheap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 3,
                    num_descriptors: 1,
                    ..Default::default()
                },
                ptr::null(),
                &mut dsv_heap,
            ) != 0
        {
            return false;
        }
        let get_cpu: extern "win64" fn(
            *mut c_void,
            *mut D3D12CpuDescriptorHandle,
        ) -> *mut D3D12CpuDescriptorHandle =
            std::mem::transmute(*(*(rtv_heap as *const *const usize)).add(9));
        let mut rtv_h = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        let mut dsv_h = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        get_cpu(rtv_heap, rtv_h.as_mut_ptr());
        get_cpu(dsv_heap, dsv_h.as_mut_ptr());
        let rtv_h = rtv_h.assume_init();
        let dsv_h = dsv_h.assume_init();
        create_rtv(dev, rt, ptr::null(), rtv_h);
        create_dsv(dev, ds, ptr::null(), dsv_h);

        let vs_bytes = synthetic_indexed_depth_vs_bytes();
        let ps_bytes = synthetic_triangle_ps_bytes();
        let in_elem = [D3D12InputElementDesc {
            semantic_name: b"POSITION\0".as_ptr(),
            semantic_index: 0,
            format: 2, // R32G32B32A32_FLOAT
            input_slot: 0,
            aligned_byte_offset: 0,
            input_slot_class: 0,
            instance_data_step_rate: 0,
        }];
        let mut gpso_desc: D3D12GraphicsPipelineStateDesc = std::mem::zeroed();
        gpso_desc.p_root_signature = gfx_rs;
        gpso_desc.vs = D3D12ShaderBytecode {
            p_shader_bytecode: vs_bytes.as_ptr().cast(),
            bytecode_length: vs_bytes.len(),
        };
        gpso_desc.ps = D3D12ShaderBytecode {
            p_shader_bytecode: ps_bytes.as_ptr().cast(),
            bytecode_length: ps_bytes.len(),
        };
        gpso_desc.depth_stencil_state[0] = 1;
        gpso_desc.depth_stencil_state[1] = 1;
        gpso_desc.depth_stencil_state[2] = 2; // LESS
        gpso_desc.input_layout = D3D12InputLayoutDesc {
            p_input_element_descs: in_elem.as_ptr().cast(),
            num_elements: 1,
        };
        gpso_desc.num_render_targets = 1;
        gpso_desc.rtv_formats[0] = 28;
        gpso_desc.dsv_format = 40;
        gpso_desc.sample_desc = DxgiSampleDesc {
            count: 1,
            quality: 0,
        };
        let mut gpso = ptr::null_mut();
        if create_gpso(dev, &gpso_desc, ptr::null(), &mut gpso) != 0 {
            return false;
        }

        let mut list = ptr::null_mut();
        if create_list(dev, 0, 0, alloc, gpso, ptr::null(), &mut list) != 0 {
            return false;
        }
        let lvt = *(list as *const *const usize);
        let list_close: extern "win64" fn(*mut c_void) -> i32 = std::mem::transmute(*lvt.add(9));
        let list_draw_idx: extern "win64" fn(*mut c_void, u32, u32, u32, i32, u32) =
            std::mem::transmute(*lvt.add(13));
        let list_dispatch: extern "win64" fn(*mut c_void, u32, u32, u32) =
            std::mem::transmute(*lvt.add(14));
        let list_copy_buf: extern "win64" fn(*mut c_void, *mut c_void, u64, *mut c_void, u64, u64) =
            std::mem::transmute(*lvt.add(15));
        let list_set_pso: extern "win64" fn(*mut c_void, *mut c_void) =
            std::mem::transmute(*lvt.add(25));
        let list_set_heaps: extern "win64" fn(*mut c_void, u32, *const *mut c_void) =
            std::mem::transmute(*lvt.add(32));
        let list_set_cmp_rs: extern "win64" fn(*mut c_void, *mut c_void) =
            std::mem::transmute(*lvt.add(33));
        let list_set_gfx_rs: extern "win64" fn(*mut c_void, *mut c_void) =
            std::mem::transmute(*lvt.add(34));
        let list_set_cmp_tbl: extern "win64" fn(*mut c_void, u32, D3D12GpuDescriptorHandle) =
            std::mem::transmute(*lvt.add(35));
        let list_set_cmp_consts: extern "win64" fn(*mut c_void, u32, u32, *const c_void, u32) =
            std::mem::transmute(*lvt.add(39));
        let list_set_gfx_consts: extern "win64" fn(*mut c_void, u32, u32, *const c_void, u32) =
            std::mem::transmute(*lvt.add(40));
        let list_set_cmp_cbv: extern "win64" fn(*mut c_void, u32, u64) =
            std::mem::transmute(*lvt.add(41));
        let list_set_gfx_cbv: extern "win64" fn(*mut c_void, u32, u64) =
            std::mem::transmute(*lvt.add(42));
        let list_ia_ib: extern "win64" fn(*mut c_void, *const D3D12IndexBufferView) =
            std::mem::transmute(*lvt.add(47));
        let list_ia_vb: extern "win64" fn(*mut c_void, u32, u32, *const D3D12VertexBufferView) =
            std::mem::transmute(*lvt.add(48));
        let list_om_rt: extern "win64" fn(
            *mut c_void,
            u32,
            *const D3D12CpuDescriptorHandle,
            i32,
            *const D3D12CpuDescriptorHandle,
        ) = std::mem::transmute(*lvt.add(50));
        let list_clear_dsv: extern "win64" fn(
            *mut c_void,
            D3D12CpuDescriptorHandle,
            u32,
            f32,
            u8,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*lvt.add(51));
        let list_clear_rtv: extern "win64" fn(
            *mut c_void,
            D3D12CpuDescriptorHandle,
            *const f32,
            u32,
            *const D3D12Rect,
        ) = std::mem::transmute(*lvt.add(52));

        list_copy_buf(list, vb, 0, upload, 0, 96);
        list_copy_buf(list, ib, 0, upload, 256, 12);
        let clear_bg = [0.0f32, 0.0, 64.0 / 255.0, 1.0];
        list_clear_rtv(list, rtv_h, clear_bg.as_ptr(), 0, ptr::null());
        list_clear_dsv(list, dsv_h, 1, 1.0, 0, 0, ptr::null());
        list_om_rt(list, 1, &rtv_h, 0, &dsv_h);
        list_set_gfx_rs(list, gfx_rs);
        list_set_gfx_cbv(list, 1, res_va(cbv));
        let vbv = D3D12VertexBufferView {
            buffer_location: res_va(vb),
            size_in_bytes: 96,
            stride_in_bytes: 16,
        };
        let ibv = D3D12IndexBufferView {
            buffer_location: res_va(ib),
            size_in_bytes: 12,
            format: 57, // R16_UINT
        };
        list_ia_vb(list, 0, 1, &vbv);
        list_ia_ib(list, &ibv);

        let yellow = [1.0f32.to_bits(), 1.0f32.to_bits(), 0, 0];
        list_set_gfx_consts(list, 0, 4, yellow.as_ptr().cast(), 0);
        list_draw_idx(list, 3, 1, 0, 0, 0);
        let red = [1.0f32.to_bits(), 0, 0, 0];
        list_set_gfx_consts(list, 0, 4, red.as_ptr().cast(), 0);
        list_draw_idx(list, 3, 1, 3, 0, 0);

        // --- Part B: Compute PSO + Dispatch + UAV Descriptor Table + Readback Verification ---
        let uav_range = [D3D12DescriptorRange1 {
            range_type: 1,
            num_descriptors: 1,
            base_shader_register: 0,
            register_space: 0,
            flags: 0,
            offset_from_start: 0,
        }];
        let rp = uav_range.as_ptr() as usize as u64;
        let cmp_params = [
            D3D12RootParameterRaw {
                parameter_type: 1,
                _pad04: 0,
                data: [0, 0, 2, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 2,
                _pad04: 0,
                data: [1, 0, 0, 0],
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 0,
                _pad04: 0,
                data: [1, 0, rp as u32, (rp >> 32) as u32],
                shader_visibility: 0,
                _pad1c: 0,
            },
        ];
        let cmp_rs = serialize_and_make_rs(&cmp_params);
        if cmp_rs.is_null() {
            return false;
        }
        let mut cmp_cbv = ptr::null_mut();
        let mut uav = ptr::null_mut();
        let mut rb = ptr::null_mut();
        let mut srv_heap = ptr::null_mut();
        if create_res(
            dev,
            &hp,
            0,
            &buf_desc,
            0,
            ptr::null(),
            ptr::null(),
            &mut cmp_cbv,
        ) != 0
            || create_res(dev, &hp, 0, &buf_desc, 0, ptr::null(), ptr::null(), &mut uav) != 0
            || create_res(dev, &hp, 0, &buf_desc, 0, ptr::null(), ptr::null(), &mut rb) != 0
            || create_dheap(
                dev,
                &D3D12DescriptorHeapDesc {
                    r#type: 0,
                    num_descriptors: 2,
                    flags: 1,
                    node_mask: 0,
                },
                ptr::null(),
                &mut srv_heap,
            ) != 0
        {
            return false;
        }
        let mut ccp = ptr::null_mut();
        if res_map(cmp_cbv, 0, ptr::null(), &mut ccp) != 0 {
            return false;
        }
        ptr::write_volatile(ccp.cast::<u32>(), 0x0000_0055);
        res_unmap(cmp_cbv, 0, ptr::null());

        let get_gpu: extern "win64" fn(
            *mut c_void,
            *mut D3D12GpuDescriptorHandle,
        ) -> *mut D3D12GpuDescriptorHandle =
            std::mem::transmute(*(*(srv_heap as *const *const usize)).add(10));
        let mut uav_cpu = MaybeUninit::<D3D12CpuDescriptorHandle>::uninit();
        let mut uav_gpu = MaybeUninit::<D3D12GpuDescriptorHandle>::uninit();
        get_cpu(srv_heap, uav_cpu.as_mut_ptr());
        get_gpu(srv_heap, uav_gpu.as_mut_ptr());
        let uav_cpu = uav_cpu.assume_init();
        let uav_gpu = uav_gpu.assume_init();
        create_uav(dev, uav, ptr::null_mut(), ptr::null(), uav_cpu);

        let cs_bytes = synthetic_compute_signature_cs_bytes();
        let mut cpso_desc: D3D12ComputePipelineStateDesc = std::mem::zeroed();
        cpso_desc.p_root_signature = cmp_rs;
        cpso_desc.cs = D3D12ShaderBytecode {
            p_shader_bytecode: cs_bytes.as_ptr().cast(),
            bytecode_length: cs_bytes.len(),
        };
        let mut cpso = ptr::null_mut();
        if create_cpso(dev, &cpso_desc, ptr::null(), &mut cpso) != 0 {
            return false;
        }

        let heaps = [srv_heap];
        list_set_heaps(list, 1, heaps.as_ptr());
        list_set_pso(list, cpso);
        list_set_cmp_rs(list, cmp_rs);
        let cmp_consts: [u32; 2] = [0x5832_0000, 11];
        list_set_cmp_consts(list, 0, 2, cmp_consts.as_ptr().cast(), 0);
        list_set_cmp_cbv(list, 1, res_va(cmp_cbv));
        list_set_cmp_tbl(list, 2, uav_gpu);
        list_dispatch(list, 1, 1, 1);
        list_copy_buf(list, rb, 0, uav, 0, 64 * 4);
        if list_close(list) != 0 {
            return false;
        }

        let qvt = *(queue as *const *const usize);
        let q_exec: extern "win64" fn(*mut c_void, u32, *const *mut c_void) =
            std::mem::transmute(*qvt.add(10));
        let lists = [list];
        q_exec(queue, 1, lists.as_ptr());

        // Verify exact 64-DWORD compute readback signature!
        let mut rbp = ptr::null_mut();
        if res_map(rb, 0, ptr::null(), &mut rbp) != 0 {
            return false;
        }
        let out_words = std::slice::from_raw_parts(rbp.cast::<u32>(), 64);
        let sig_ok = out_words
            .iter()
            .enumerate()
            .all(|(i, &w)| w == 0x5832_0055u32 + (i as u32) * 11);
        res_unmap(rb, 0, ptr::null());

        let release_obj = |obj: *mut c_void| -> bool {
            let vt = *(obj as *const *const usize);
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            rel(obj) == 0
        };
        let released = release_obj(cpso)
            && release_obj(srv_heap)
            && release_obj(rb)
            && release_obj(uav)
            && release_obj(cmp_cbv)
            && release_obj(cmp_rs)
            && release_obj(list)
            && release_obj(gpso)
            && release_obj(dsv_heap)
            && release_obj(rtv_heap)
            && release_obj(ds)
            && release_obj(rt)
            && release_obj(cbv)
            && release_obj(upload)
            && release_obj(ib)
            && release_obj(vb)
            && release_obj(gpu_heap)
            && release_obj(alloc)
            && release_obj(queue)
            && release_obj(gfx_rs);

        sig_ok && released
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
