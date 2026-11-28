// Win64 COM vtable slot mappings and console `PresentX` plane stride informed by
// Marvisak/VIctory (https://github.com/Marvisak/VIctory, commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-gpu/src/{device,queue,list,pipeline}.rs)
// and public MIT Microsoft DirectX-Headers (https://github.com/microsoft/DirectX-Headers, d3d12.h).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: clean-room Win64 COM objects backed directly by
// `VulkanBackend` with W^X spare-slot probe thunks on all unimplemented slots.

//! Win64 `extern "win64"` COM ABI objects for `D3D12.X` backed by [`VulkanBackend`].
//!
//! Every interface keeps M1's RW→RX W^X probe thunks on all unimplemented vtable slots (including
//! `ID3D12DeviceX[45]`) so guest calls to unimplemented methods never crash silently and always
//! log exact caller RVAs in `report.json`.
//!
//! Public COM struct-return methods (`GetCPUDescriptorHandleForHeapStart`,
//! `GetGPUDescriptorHandleForHeapStart`, `GetDesc`) follow MSVC x86-64 COM ABI:
//! `extern "win64" fn(this: *mut Obj, out: *mut T) -> *mut T` (`RCX=this`, `RDX=out`, `RAX=out`).

use crate::vulkan::{
    DescriptorEntry, GraphicsPipelineDescInfo, InputElementInfo, RecordedCommand, ResourceDescInfo,
    VulkanBackend,
};
use crate::{GpuError, VtableProbeCallback, build_probe_vtable_table};
use ash::vk;
use std::ffi::{CStr, c_void};
use std::mem::{align_of, offset_of, size_of};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

// ============================================================================
// Native Linux GPU Worker Bridge (Wine Syscall User Dispatch isolation)
// ============================================================================
//
// Under Wine, guest PE threads run with Linux `PR_SET_SYSCALL_USER_DISPATCH` (SUD) active,
// which traps raw Linux syscalls like `SYS_userfaultfd` (323) or Vulkan driver ioctls as NT
// syscalls. When `start_native_gpu_worker()` is called from `libx2l_core.so`'s `.init_array`
// constructor (before Wine enables SUD), a dedicated native Linux pthread is spawned without
// SUD. All Vulkan + `userfaultfd` + XDG cache work from Win64 COM methods executes on that
// native thread via a zero-allocation, syscall-free atomic mailbox.

const MAILBOX_IDLE: u32 = 0;
const MAILBOX_WORK_READY: u32 = 1;
const MAILBOX_WORK_DONE: u32 = 2;

struct NativeWorkerMailbox {
    caller_lock: AtomicBool,
    state: AtomicU32,
    trampoline: AtomicUsize,
    data_ptr: AtomicUsize,
}

static NATIVE_WORKER: OnceLock<&'static NativeWorkerMailbox> = OnceLock::new();

/// Spawn the SUD-free native Linux worker thread. Must be called from ELF `.init_array`
/// before Wine enables `PR_SET_SYSCALL_USER_DISPATCH` on guest threads.
pub fn start_native_gpu_worker() {
    if NATIVE_WORKER.get().is_some() {
        return;
    }
    let mailbox: &'static NativeWorkerMailbox = Box::leak(Box::new(NativeWorkerMailbox {
        caller_lock: AtomicBool::new(false),
        state: AtomicU32::new(MAILBOX_IDLE),
        trampoline: AtomicUsize::new(0),
        data_ptr: AtomicUsize::new(0),
    }));
    let spawn_res = std::thread::Builder::new()
        .name("x2l-native-worker".into())
        .spawn(move || {
            let mut idle_spins = 0u32;
            loop {
                if mailbox.state.load(Ordering::Acquire) == MAILBOX_WORK_READY {
                    idle_spins = 0;
                    let fn_addr = mailbox.trampoline.load(Ordering::Relaxed);
                    let arg_addr = mailbox.data_ptr.load(Ordering::Relaxed);
                    if fn_addr != 0 {
                        // SAFETY: `fn_addr` and `arg_addr` are valid for the duration of this
                        // synchronous call while the requester spins waiting for `MAILBOX_WORK_DONE`.
                        unsafe {
                            let call: fn(*mut c_void) = std::mem::transmute(fn_addr);
                            call(arg_addr as *mut c_void);
                        }
                    }
                    mailbox.state.store(MAILBOX_WORK_DONE, Ordering::Release);
                } else if idle_spins < 4000 {
                    idle_spins += 1;
                    std::hint::spin_loop();
                } else {
                    std::thread::sleep(Duration::from_micros(200));
                }
            }
        });
    if spawn_res.is_ok() {
        let _ = NATIVE_WORKER.set(mailbox);
    }
}

/// Execute `f` on the SUD-free native Linux worker thread when active, or inline in native tests.
pub fn exec_native<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let Some(&mailbox) = NATIVE_WORKER.get() else {
        return f();
    };

    struct Payload<F, R> {
        func: Option<F>,
        result: Option<R>,
    }

    fn invoke<F: FnOnce() -> R, R>(raw: *mut c_void) {
        // SAFETY: `raw` points to a live stack-pinned `Payload<F, R>` owned by `exec_native`.
        unsafe {
            let payload = &mut *(raw as *mut Payload<F, R>);
            if let Some(func) = payload.func.take() {
                payload.result = Some(func());
            }
        }
    }

    // Acquire caller slot using pure userspace atomics (zero Linux futex syscalls on Wine thread).
    while mailbox
        .caller_lock
        .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        std::hint::spin_loop();
    }

    let mut payload = Payload {
        func: Some(f),
        result: None,
    };
    mailbox
        .trampoline
        .store(invoke::<F, R> as *const () as usize, Ordering::Relaxed);
    mailbox.data_ptr.store(
        (&mut payload as *mut Payload<F, R>) as usize,
        Ordering::Relaxed,
    );
    mailbox.state.store(MAILBOX_WORK_READY, Ordering::Release);

    while mailbox.state.load(Ordering::Acquire) != MAILBOX_WORK_DONE {
        std::hint::spin_loop();
    }

    mailbox.trampoline.store(0, Ordering::Relaxed);
    mailbox.data_ptr.store(0, Ordering::Relaxed);
    mailbox.state.store(MAILBOX_IDLE, Ordering::Release);
    mailbox.caller_lock.store(false, Ordering::Release);

    payload.result.expect("native worker must populate result")
}

pub const S_OK: i32 = 0;
pub const E_POINTER: i32 = -2147467261; // 0x80004003
pub const E_INVALIDARG: i32 = -2147024809; // 0x80070057
pub const E_FAIL: i32 = -2147467259; // 0x80004005

// ============================================================================
// Exact Win64 `#[repr(C)]` D3D12 / D3D12.X ABI records
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12CommandQueueDesc {
    pub r#type: u32,
    pub priority: i32,
    pub flags: u32,
    pub node_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12HeapProperties {
    pub r#type: u32,
    pub cpu_page_property: u32,
    pub memory_pool_preference: u32,
    pub creation_node_mask: u32,
    pub visible_node_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DxgiSampleDesc {
    pub count: u32,
    pub quality: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
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
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12DescriptorHeapDesc {
    pub r#type: u32,
    pub num_descriptors: u32,
    pub flags: u32,
    pub node_mask: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12CpuDescriptorHandle {
    pub ptr: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12GpuDescriptorHandle {
    pub ptr: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct D3D12ShaderBytecode {
    pub p_shader_bytecode: *const c_void,
    pub bytecode_length: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12Viewport {
    pub top_left_x: f32,
    pub top_left_y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12ResourceTransitionBarrier {
    pub p_resource: *mut ComResource,
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

/// Exact 656-byte Win64 `D3D12_GRAPHICS_PIPELINE_STATE_DESC` layout (matching Microsoft
/// `DirectX-Headers` `d3d12.h`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12GraphicsPipelineStateDesc {
    pub p_root_signature: *mut c_void,        // @0
    pub vs: D3D12ShaderBytecode,              // @8
    pub ps: D3D12ShaderBytecode,              // @24
    pub ds: D3D12ShaderBytecode,              // @40
    pub hs: D3D12ShaderBytecode,              // @56
    pub gs: D3D12ShaderBytecode,              // @72
    pub stream_output: D3D12StreamOutputDesc, // @88 (32 bytes)
    pub blend_state: [u32; 82],               // @120 (328 bytes)
    pub sample_mask: u32,                     // @448
    pub rasterizer_state: [u32; 11],          // @452 (44 bytes)
    pub depth_stencil_state: [u32; 13],       // @496 (52 bytes)
    pub input_layout: D3D12InputLayoutDesc,   // @552 (16 bytes)
    pub ib_strip_cut_value: u32,              // @568
    pub primitive_topology_type: u32,         // @572
    pub num_render_targets: u32,              // @576
    pub rtv_formats: [u32; 8],                // @580
    pub dsv_format: u32,                      // @612
    pub sample_desc: DxgiSampleDesc,          // @616
    pub node_mask: u32,                       // @624
    pub cached_pso: D3D12CachedPipelineState, // @632 (16 bytes)
    pub flags: u32,                           // @648
}

/// Console `D3D12XBOX_PRESENT_PLANE_PARAMETERS` (72 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12XboxPresentPlaneParameters {
    pub token: u64,                            // @0
    pub resource_count: u32,                   // @8
    pub _pad0: u32,                            // @12
    pub pp_resources: *const *mut ComResource, // @16
    pub _reserved: [u64; 6],                   // @24..72
}

/// Win64 `D3D12_COMPUTE_PIPELINE_STATE_DESC` (56 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12ComputePipelineStateDesc {
    pub p_root_signature: *mut c_void,        // @0
    pub cs: D3D12ShaderBytecode,              // @8
    pub node_mask: u32,                       // @24
    pub _pad28: u32,                          // @28
    pub cached_pso: D3D12CachedPipelineState, // @32
    pub flags: u32,                           // @48
    pub _pad52: u32,                          // @52
}

/// Win64 `D3D12_HEAP_DESC` (48 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12HeapDesc {
    pub size_in_bytes: u64,              // @0
    pub properties: D3D12HeapProperties, // @8 (20 bytes)
    pub _pad28: u32,                     // @28
    pub alignment: u64,                  // @32
    pub flags: u32,                      // @40
    pub _pad44: u32,                     // @44
}

/// Win64 `D3D12_VERTEX_BUFFER_VIEW` (16 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12VertexBufferView {
    pub buffer_location: u64,
    pub size_in_bytes: u32,
    pub stride_in_bytes: u32,
}

/// Win64 `D3D12_INDEX_BUFFER_VIEW` (16 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12IndexBufferView {
    pub buffer_location: u64,
    pub size_in_bytes: u32,
    pub format: u32,
}

/// Win64 `D3D12_CONSTANT_BUFFER_VIEW_DESC` (16 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12ConstantBufferViewDesc {
    pub buffer_location: u64,
    pub size_in_bytes: u32,
    pub _pad12: u32,
}

/// Win64 `D3D12_BUFFER_SRV` (24 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12BufferSrv {
    pub first_element: u64,
    pub num_elements: u32,
    pub structure_byte_stride: u32,
    pub flags: u32,
    pub _pad20: u32,
}

/// Win64 `D3D12_SHADER_RESOURCE_VIEW_DESC` (40 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12ShaderResourceViewDesc {
    pub format: u32,
    pub view_dimension: u32,
    pub shader_4_component_mapping: u32,
    pub _pad12: u32,
    pub buffer: D3D12BufferSrv,
}

/// Win64 `D3D12_BUFFER_UAV` (32 bytes, align 8 — console 40-byte UAV desc layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12BufferUav {
    pub first_element: u64,
    pub num_elements: u32,
    pub structure_byte_stride: u32,
    pub counter_offset_in_bytes: u64,
    pub flags: u32,
    pub _pad28: u32,
}

/// Win64 console `D3D12_UNORDERED_ACCESS_VIEW_DESC` (40 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12UnorderedAccessViewDesc {
    pub format: u32,
    pub view_dimension: u32,
    pub buffer: D3D12BufferUav,
}

/// Win64 `D3D12_DEPTH_STENCIL_VIEW_DESC` (24 bytes, align 4).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12DepthStencilViewDesc {
    pub format: u32,
    pub view_dimension: u32,
    pub flags: u32,
    pub mip_slice: u32,
    pub first_array_slice: u32,
    pub array_size: u32,
}

/// Win64 `D3D12_SUBRESOURCE_FOOTPRINT` (20 bytes, align 4).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12SubresourceFootprint {
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub row_pitch: u32,
}

/// Win64 `D3D12_PLACED_SUBRESOURCE_FOOTPRINT` (32 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12PlacedSubresourceFootprint {
    pub offset: u64,
    pub footprint: D3D12SubresourceFootprint,
    pub _pad28: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union D3D12TextureCopyLocationUnion {
    pub placed_footprint: D3D12PlacedSubresourceFootprint,
    pub subresource_index: u32,
}

/// Win64 `D3D12_TEXTURE_COPY_LOCATION` (48 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12TextureCopyLocation {
    pub p_resource: *mut ComResource,
    pub r#type: u32, // 0 = SUBRESOURCE_INDEX, 1 = PLACED_FOOTPRINT
    pub _pad12: u32,
    pub u: D3D12TextureCopyLocationUnion,
}

/// Win64 `D3D12_BOX` (24 bytes, align 4).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12Box {
    pub left: u32,
    pub top: u32,
    pub front: u32,
    pub right: u32,
    pub bottom: u32,
    pub back: u32,
}

/// Win64 `D3D12_INPUT_ELEMENT_DESC` (32 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12InputElementDesc {
    pub semantic_name: *const std::ffi::c_char,
    pub semantic_index: u32,
    pub format: u32,
    pub input_slot: u32,
    pub aligned_byte_offset: u32,
    pub input_slot_class: u32,
    pub instance_data_step_rate: u32,
}

/// Win64 `LUID` (8 bytes, align 4).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct D3D12Luid {
    pub low_part: u32,
    pub high_part: i32,
}

const _: () = assert!(size_of::<D3D12Luid>() == 8);

/// Win64 `D3D12_RESOURCE_ALLOCATION_INFO` (16 bytes, align 8).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12ResourceAllocationInfo {
    pub size_in_bytes: u64,
    pub alignment: u64,
}

// Compile-time Win64 ABI size, alignment, and field-offset assertions matching
// Microsoft DirectX-Headers `d3d12.h` and pinned console ABI.
const _: () = assert!(size_of::<D3D12CommandQueueDesc>() == 16);
const _: () = assert!(size_of::<D3D12HeapProperties>() == 20);
const _: () = assert!(size_of::<D3D12ResourceDesc>() == 56);
const _: () = assert!(size_of::<D3D12DescriptorHeapDesc>() == 16);
const _: () = assert!(size_of::<D3D12CpuDescriptorHandle>() == 8);
const _: () = assert!(size_of::<D3D12GpuDescriptorHandle>() == 8);
const _: () = assert!(size_of::<D3D12ShaderBytecode>() == 16);
const _: () = assert!(size_of::<D3D12Viewport>() == 24);
const _: () = assert!(size_of::<D3D12Rect>() == 16);
const _: () = assert!(size_of::<D3D12ResourceBarrier>() == 32);
const _: () = assert!(size_of::<D3D12GraphicsPipelineStateDesc>() == 656);
const _: () = assert!(size_of::<D3D12XboxPresentPlaneParameters>() == 72);
const _: () = assert!(size_of::<D3D12ComputePipelineStateDesc>() == 56);
const _: () = assert!(align_of::<D3D12ComputePipelineStateDesc>() == 8);
const _: () = assert!(offset_of!(D3D12ComputePipelineStateDesc, cs) == 8);
const _: () = assert!(offset_of!(D3D12ComputePipelineStateDesc, cached_pso) == 32);
const _: () = assert!(offset_of!(D3D12ComputePipelineStateDesc, flags) == 48);
const _: () = assert!(size_of::<D3D12HeapDesc>() == 48);
const _: () = assert!(offset_of!(D3D12HeapDesc, alignment) == 32);
const _: () = assert!(offset_of!(D3D12HeapDesc, flags) == 40);
const _: () = assert!(size_of::<D3D12VertexBufferView>() == 16);
const _: () = assert!(size_of::<D3D12IndexBufferView>() == 16);
const _: () = assert!(size_of::<D3D12ConstantBufferViewDesc>() == 16);
const _: () = assert!(size_of::<D3D12ShaderResourceViewDesc>() == 40);
const _: () = assert!(size_of::<D3D12UnorderedAccessViewDesc>() == 40);
const _: () = assert!(size_of::<D3D12DepthStencilViewDesc>() == 24);
const _: () = assert!(size_of::<D3D12PlacedSubresourceFootprint>() == 32);
const _: () = assert!(size_of::<D3D12TextureCopyLocation>() == 48);
const _: () = assert!(offset_of!(D3D12TextureCopyLocation, u) == 16);
const _: () = assert!(size_of::<D3D12Box>() == 24);
const _: () = assert!(size_of::<D3D12InputElementDesc>() == 32);
const _: () = assert!(size_of::<D3D12ResourceAllocationInfo>() == 16);

// ============================================================================
// Process-lifetime vtable registry (each interface preserves W^X spare probes)
// ============================================================================

struct ComVtables {
    device: &'static [usize],
    queue: &'static [usize],
    allocator: &'static [usize],
    cmd_list: &'static [usize],
    fence: &'static [usize],
    resource: &'static [usize],
    desc_heap: &'static [usize],
    pso: &'static [usize],
    root_sig: &'static [usize],
    heap: &'static [usize],
}

static COM_VTABLES: OnceLock<ComVtables> = OnceLock::new();
static KNOWN_DEVICE_VTBL_VK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static KNOWN_DEVICE_VTBL_NULL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static NEXT_FRAME_TOKEN: AtomicU64 = AtomicU64::new(1);
static FRAME_INTERVAL_US: AtomicU32 = AtomicU32::new(16_667);

/// Register a known device vtable address so dual-ABI struct-return methods (`25`, `26`, `43`)
/// can distinguish `RCX=this, RDX=ret` from `RCX=ret, RDX=this`.
pub(crate) fn register_device_vtable_ptr(ptr: usize) {
    let _ = KNOWN_DEVICE_VTBL_NULL.compare_exchange(0, ptr, Ordering::AcqRel, Ordering::Acquire);
}

fn is_known_device_object(ptr: *mut c_void) -> bool {
    if ptr.is_null() || !(ptr as usize).is_multiple_of(align_of::<usize>()) {
        return false;
    }
    let vk = KNOWN_DEVICE_VTBL_VK.load(Ordering::Acquire);
    let null = KNOWN_DEVICE_VTBL_NULL.load(Ordering::Acquire);
    // SAFETY: `ptr` is non-null, 8-byte aligned, and points to either a live COM device object
    // or a caller-allocated struct-return buffer of at least 8 bytes.
    let first_word = unsafe { *(ptr as *const usize) };
    (vk != 0 && first_word == vk) || (null != 0 && first_word == null)
}

fn init_vtables(probe_cb: VtableProbeCallback) -> Result<&'static ComVtables, GpuError> {
    if let Some(vt) = COM_VTABLES.get() {
        return Ok(vt);
    }

    let device = build_probe_vtable_table(
        128,
        probe_cb,
        &[
            (0, dev_query_interface as *const () as usize),
            (1, dev_add_ref as *const () as usize),
            (2, dev_release as *const () as usize),
            (7, dev_get_node_count as *const () as usize),
            (8, dev_create_command_queue as *const () as usize),
            (9, dev_create_command_allocator as *const () as usize),
            (10, dev_create_graphics_pipeline_state as *const () as usize),
            (11, dev_create_compute_pipeline_state as *const () as usize),
            (12, dev_create_command_list as *const () as usize),
            (13, dev_check_feature_support as *const () as usize),
            (14, dev_create_descriptor_heap as *const () as usize),
            (
                15,
                dev_get_descriptor_handle_increment_size as *const () as usize,
            ),
            (16, dev_create_root_signature as *const () as usize),
            (17, dev_create_constant_buffer_view as *const () as usize),
            (18, dev_create_shader_resource_view as *const () as usize),
            (19, dev_create_unordered_access_view as *const () as usize),
            (20, dev_create_render_target_view as *const () as usize),
            (21, dev_create_depth_stencil_view as *const () as usize),
            (22, dev_create_sampler as *const () as usize),
            (23, dev_copy_descriptors as *const () as usize),
            (24, dev_copy_descriptors_simple as *const () as usize),
            (25, dev_get_resource_allocation_info as *const () as usize),
            (26, dev_get_custom_heap_properties as *const () as usize),
            (27, dev_create_committed_resource as *const () as usize),
            (28, dev_create_heap as *const () as usize),
            (29, dev_create_placed_resource as *const () as usize),
            (36, dev_create_fence as *const () as usize),
            (37, dev_get_device_removed_reason as *const () as usize),
            (38, dev_get_copyable_footprints as *const () as usize),
            (43, dev_get_adapter_luid as *const () as usize),
            (44, dev_create_placed_resource_x as *const () as usize),
            // Slot 45 has no verified console assignment and stays a W^X probe thunk.
            (48, dev_create_command_queue_x as *const () as usize),
            (52, dev_set_debug_flags_x as *const () as usize),
            (
                60,
                dev_create_graphics_pipeline_state_x as *const () as usize,
            ),
            (
                62,
                dev_create_compute_pipeline_state_x as *const () as usize,
            ),
            (78, dev_set_frame_interval_x as *const () as usize),
            (79, dev_schedule_frame_event_x as *const () as usize),
            (80, dev_wait_frame_event_x as *const () as usize),
            (81, dev_get_frame_statistics_x as *const () as usize),
        ],
    )?;
    KNOWN_DEVICE_VTBL_VK.store(device.as_ptr() as usize, Ordering::Release);

    let queue = build_probe_vtable_table(
        64,
        probe_cb,
        &[
            (0, queue_query_interface as *const () as usize),
            (1, queue_add_ref as *const () as usize),
            (2, queue_release as *const () as usize),
            (10, queue_execute_command_lists as *const () as usize),
            (14, queue_signal as *const () as usize),
            (15, queue_wait as *const () as usize),
            (16, queue_get_timestamp_frequency as *const () as usize),
            (18, queue_get_timestamp_frequency as *const () as usize),
            (30, queue_present_x as *const () as usize),
        ],
    )?;

    let allocator = build_probe_vtable_table(
        32,
        probe_cb,
        &[
            (0, alloc_query_interface as *const () as usize),
            (1, alloc_add_ref as *const () as usize),
            (2, alloc_release as *const () as usize),
            (8, alloc_reset as *const () as usize),
        ],
    )?;

    let cmd_list = build_probe_vtable_table(
        160,
        probe_cb,
        &[
            (0, list_query_interface as *const () as usize),
            (1, list_add_ref as *const () as usize),
            (2, list_release as *const () as usize),
            (8, list_get_type as *const () as usize),
            (9, list_close as *const () as usize),
            (10, list_reset as *const () as usize),
            (12, list_draw_instanced as *const () as usize),
            (13, list_draw_indexed_instanced as *const () as usize),
            (14, list_dispatch as *const () as usize),
            (15, list_copy_buffer_region as *const () as usize),
            (16, list_copy_texture_region as *const () as usize),
            (17, list_copy_resource as *const () as usize),
            (20, list_ia_set_primitive_topology as *const () as usize),
            (21, list_rs_set_viewports as *const () as usize),
            (22, list_rs_set_scissor_rects as *const () as usize),
            (25, list_set_pipeline_state as *const () as usize),
            (26, list_resource_barrier as *const () as usize),
            // Console ID3D12GraphicsCommandListX layout (+4 vs desktop from slot 32 onward):
            (32, list_set_descriptor_heaps as *const () as usize),
            (33, list_set_compute_root_signature as *const () as usize),
            (34, list_set_graphics_root_signature as *const () as usize),
            (
                35,
                list_set_compute_root_descriptor_table as *const () as usize,
            ),
            (
                36,
                list_set_graphics_root_descriptor_table as *const () as usize,
            ),
            (
                37,
                list_set_compute_root_32bit_constant as *const () as usize,
            ),
            (
                38,
                list_set_graphics_root_32bit_constant as *const () as usize,
            ),
            (
                39,
                list_set_compute_root_32bit_constants as *const () as usize,
            ),
            (
                40,
                list_set_graphics_root_32bit_constants as *const () as usize,
            ),
            (
                41,
                list_set_compute_root_constant_buffer_view as *const () as usize,
            ),
            (
                42,
                list_set_graphics_root_constant_buffer_view as *const () as usize,
            ),
            (
                43,
                list_set_compute_root_shader_resource_view as *const () as usize,
            ),
            (
                44,
                list_set_graphics_root_shader_resource_view as *const () as usize,
            ),
            (
                45,
                list_set_compute_root_unordered_access_view as *const () as usize,
            ),
            (
                46,
                list_set_graphics_root_unordered_access_view as *const () as usize,
            ),
            (47, list_ia_set_index_buffer as *const () as usize),
            (48, list_ia_set_vertex_buffers as *const () as usize),
            (50, list_om_set_render_targets as *const () as usize),
            (51, list_clear_depth_stencil_view as *const () as usize),
            (52, list_clear_render_target_view as *const () as usize),
            (98, list_dispatch_mesh_x as *const () as usize),
            (127, list_dispatch_mesh as *const () as usize),
        ],
    )?;

    let fence = build_probe_vtable_table(
        32,
        probe_cb,
        &[
            (0, fence_query_interface as *const () as usize),
            (1, fence_add_ref as *const () as usize),
            (2, fence_release as *const () as usize),
            (8, fence_get_completed_value as *const () as usize),
            (9, fence_set_event_on_completion as *const () as usize),
            (10, fence_signal as *const () as usize),
        ],
    )?;

    let resource = build_probe_vtable_table(
        32,
        probe_cb,
        &[
            (0, res_query_interface as *const () as usize),
            (1, res_add_ref as *const () as usize),
            (2, res_release as *const () as usize),
            (8, res_map as *const () as usize),
            (9, res_unmap as *const () as usize),
            (10, res_get_desc as *const () as usize),
            (11, res_get_gpu_virtual_address as *const () as usize),
        ],
    )?;

    let desc_heap = build_probe_vtable_table(
        32,
        probe_cb,
        &[
            (0, heap_query_interface as *const () as usize),
            (1, heap_add_ref as *const () as usize),
            (2, heap_release as *const () as usize),
            (8, heap_get_desc as *const () as usize),
            (
                9,
                heap_get_cpu_descriptor_handle_for_heap_start as *const () as usize,
            ),
            (
                10,
                heap_get_gpu_descriptor_handle_for_heap_start as *const () as usize,
            ),
        ],
    )?;

    let pso = build_probe_vtable_table(
        32,
        probe_cb,
        &[
            (0, pso_query_interface as *const () as usize),
            (1, pso_add_ref as *const () as usize),
            (2, pso_release as *const () as usize),
        ],
    )?;

    let root_sig = build_probe_vtable_table(
        16,
        probe_cb,
        &[
            (0, rsig_query_interface as *const () as usize),
            (1, rsig_add_ref as *const () as usize),
            (2, rsig_release as *const () as usize),
        ],
    )?;

    let heap = build_probe_vtable_table(
        16,
        probe_cb,
        &[
            (0, mem_heap_query_interface as *const () as usize),
            (1, mem_heap_add_ref as *const () as usize),
            (2, mem_heap_release as *const () as usize),
            (8, mem_heap_get_desc as *const () as usize),
        ],
    )?;

    let _ = COM_VTABLES.set(ComVtables {
        device,
        queue,
        allocator,
        cmd_list,
        fence,
        resource,
        desc_heap,
        pso,
        root_sig,
        heap,
    });
    Ok(COM_VTABLES.get().expect("vtables initialized"))
}

// ============================================================================
// COM objects (`#[repr(C)]` with vtable pointer at offset 0)
// ============================================================================

#[repr(C)]
pub struct ComDevice {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    backend: Arc<VulkanBackend>,
    vtables: &'static ComVtables,
}

#[repr(C)]
pub struct ComCommandQueue {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    queue_type: u32,
    backend: Arc<VulkanBackend>,
}

#[repr(C)]
pub struct ComCommandAllocator {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    list_type: u32,
    generation: AtomicU64,
}

struct CmdListInner {
    closed: bool,
    commands: Vec<RecordedCommand>,
}

#[repr(C)]
pub struct ComGraphicsCommandList {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    list_type: u32,
    backend: Arc<VulkanBackend>,
    inner: Mutex<CmdListInner>,
}

#[repr(C)]
pub struct ComFence {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    backend: Arc<VulkanBackend>,
    fence_id: u64,
}

#[repr(C)]
pub struct ComResource {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    backend: Arc<VulkanBackend>,
    resource_id: u64,
    desc: D3D12ResourceDesc,
}

#[repr(C)]
pub struct ComDescriptorHeap {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    heap_id: u64,
    desc: D3D12DescriptorHeapDesc,
    base_cpu_ptr: usize,
}

#[repr(C)]
pub struct ComPipelineState {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    pso_id: u64,
}

#[repr(C)]
pub struct ComRootSignature {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    root_sig_id: u64,
}

#[repr(C)]
pub struct ComHeap {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    heap_id: u64,
    desc: D3D12HeapDesc,
}

/// Allocate a real Vulkan-backed Win64 COM `ID3D12DeviceX` object with refcount = 1.
pub fn create_vulkan_com_device(
    backend: Arc<VulkanBackend>,
    probe_cb: VtableProbeCallback,
) -> Result<*mut ComDevice, GpuError> {
    let vtables = exec_native(|| init_vtables(probe_cb))?;
    Ok(Box::into_raw(Box::new(ComDevice {
        vtbl: vtables.device.as_ptr(),
        ref_count: AtomicU32::new(1),
        backend,
        vtables,
    })))
}

macro_rules! impl_iunknown {
    ($ty:ty, $qi:ident, $addref:ident, $release:ident) => {
        extern "win64" fn $qi(
            this: *mut $ty,
            _riid: *const c_void,
            out_obj: *mut *mut c_void,
        ) -> i32 {
            if this.is_null() || out_obj.is_null() {
                return E_POINTER;
            }
            // SAFETY: Caller passed a live COM object pointer and writable `out_obj`.
            unsafe {
                (*this).ref_count.fetch_add(1, Ordering::AcqRel);
                *out_obj = this as *mut c_void;
            }
            S_OK
        }

        extern "win64" fn $addref(this: *mut $ty) -> u32 {
            if this.is_null() {
                return 0;
            }
            // SAFETY: `this` is a live COM object allocated via `Box::into_raw`.
            unsafe { (*this).ref_count.fetch_add(1, Ordering::AcqRel) + 1 }
        }

        extern "win64" fn $release(this: *mut $ty) -> u32 {
            if this.is_null() {
                return 0;
            }
            // SAFETY: `this` was allocated via `Box::into_raw` and is freed exactly once at 0.
            unsafe {
                let prev = (*this).ref_count.fetch_sub(1, Ordering::AcqRel);
                if prev <= 1 {
                    drop(Box::from_raw(this));
                    0
                } else {
                    prev - 1
                }
            }
        }
    };
}

impl_iunknown!(ComDevice, dev_query_interface, dev_add_ref, dev_release);
impl_iunknown!(
    ComCommandQueue,
    queue_query_interface,
    queue_add_ref,
    queue_release
);
impl_iunknown!(
    ComCommandAllocator,
    alloc_query_interface,
    alloc_add_ref,
    alloc_release
);
impl_iunknown!(
    ComGraphicsCommandList,
    list_query_interface,
    list_add_ref,
    list_release
);
impl_iunknown!(
    ComFence,
    fence_query_interface,
    fence_add_ref,
    fence_release
);
impl_iunknown!(ComResource, res_query_interface, res_add_ref, res_release);
impl_iunknown!(
    ComDescriptorHeap,
    heap_query_interface,
    heap_add_ref,
    heap_release
);
impl_iunknown!(
    ComPipelineState,
    pso_query_interface,
    pso_add_ref,
    pso_release
);
impl_iunknown!(
    ComRootSignature,
    rsig_query_interface,
    rsig_add_ref,
    rsig_release
);
impl_iunknown!(
    ComHeap,
    mem_heap_query_interface,
    mem_heap_add_ref,
    mem_heap_release
);

extern "win64" fn mem_heap_get_desc(
    this: *mut ComHeap,
    out_desc: *mut D3D12HeapDesc,
) -> *mut D3D12HeapDesc {
    if !this.is_null() && !out_desc.is_null() {
        // SAFETY: MSVC COM struct-return ABI writes `*out_desc` and returns `out_desc`.
        unsafe {
            *out_desc = (*this).desc;
        }
    }
    out_desc
}

// ============================================================================
// ID3D12DeviceX methods
// ============================================================================

extern "win64" fn dev_get_node_count(_this: *mut ComDevice) -> u32 {
    1
}

extern "win64" fn dev_create_command_queue(
    this: *mut ComDevice,
    desc: *const D3D12CommandQueueDesc,
    _riid: *const c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if !out_queue.is_null() {
        // SAFETY: Clear output pointer before validation.
        unsafe { *out_queue = ptr::null_mut() };
    }
    if this.is_null() || desc.is_null() || out_queue.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComDevice`; `desc` is readable and `out_queue` is writable.
    unsafe {
        let dev = &*this;
        let q_type = (*desc).r#type;
        // Accept Direct (0), Compute (2), Copy (3), and console DMA (0x10).
        if !matches!(q_type, 0 | 2 | 3 | 0x10) {
            return E_INVALIDARG;
        }
        let q = Box::into_raw(Box::new(ComCommandQueue {
            vtbl: dev.vtables.queue.as_ptr(),
            ref_count: AtomicU32::new(1),
            queue_type: q_type,
            backend: Arc::clone(&dev.backend),
        }));
        *out_queue = q as *mut c_void;
    }
    S_OK
}

extern "win64" fn dev_create_command_queue_x(
    this: *mut ComDevice,
    desc: *const D3D12CommandQueueDesc,
    _riid: *const c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if !out_queue.is_null() {
        // SAFETY: Clear output pointer before validation.
        unsafe { *out_queue = ptr::null_mut() };
    }
    if this.is_null() || desc.is_null() || out_queue.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComDevice`; `desc` is a 16-byte console queue descriptor.
    unsafe {
        let dev = &*this;
        let raw_type = (*desc).r#type;
        let q_type = match raw_type {
            0 | 2 | 3 | 0x10 => raw_type,
            _ => 0,
        };
        let q = Box::into_raw(Box::new(ComCommandQueue {
            vtbl: dev.vtables.queue.as_ptr(),
            ref_count: AtomicU32::new(1),
            queue_type: q_type,
            backend: Arc::clone(&dev.backend),
        }));
        *out_queue = q as *mut c_void;
    }
    S_OK
}

extern "win64" fn dev_create_command_allocator(
    this: *mut ComDevice,
    list_type: u32,
    _riid: *const c_void,
    out_alloc: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_alloc.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComDevice` and `out_alloc` is writable.
    unsafe {
        let dev = &*this;
        let alloc = Box::into_raw(Box::new(ComCommandAllocator {
            vtbl: dev.vtables.allocator.as_ptr(),
            ref_count: AtomicU32::new(1),
            list_type,
            generation: AtomicU64::new(1),
        }));
        *out_alloc = alloc as *mut c_void;
    }
    S_OK
}

extern "win64" fn dev_create_graphics_pipeline_state(
    this: *mut ComDevice,
    desc: *const D3D12GraphicsPipelineStateDesc,
    _riid: *const c_void,
    out_pso: *mut *mut c_void,
) -> i32 {
    if this.is_null() || desc.is_null() || out_pso.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComDevice`, `desc` points to an initialized PSO descriptor,
    // and shader bytecode pointers/lengths are validated before slicing.
    unsafe {
        let dev = &*this;
        let d = &*desc;
        if d.vs.p_shader_bytecode.is_null()
            || d.vs.bytecode_length == 0
            || d.ps.p_shader_bytecode.is_null()
            || d.ps.bytecode_length == 0
        {
            return E_INVALIDARG;
        }
        let vs_vec =
            std::slice::from_raw_parts(d.vs.p_shader_bytecode.cast::<u8>(), d.vs.bytecode_length)
                .to_vec();
        let ps_vec =
            std::slice::from_raw_parts(d.ps.p_shader_bytecode.cast::<u8>(), d.ps.bytecode_length)
                .to_vec();
        let rtv_fmt = if d.num_render_targets > 0 {
            d.rtv_formats[0]
        } else {
            crate::vulkan::DXGI_FORMAT_R8G8B8A8_UNORM
        };
        let mut input_elements = Vec::new();
        if d.input_layout.num_elements > 0 && !d.input_layout.p_input_element_descs.is_null() {
            let elems = std::slice::from_raw_parts(
                d.input_layout
                    .p_input_element_descs
                    .cast::<D3D12InputElementDesc>(),
                d.input_layout.num_elements as usize,
            );
            for elem in elems {
                let name = if elem.semantic_name.is_null() {
                    "TEXCOORD".to_owned()
                } else {
                    CStr::from_ptr(elem.semantic_name)
                        .to_string_lossy()
                        .into_owned()
                };
                input_elements.push(InputElementInfo {
                    semantic_name: name,
                    semantic_index: elem.semantic_index,
                    format: elem.format,
                    input_slot: elem.input_slot,
                    aligned_byte_offset: elem.aligned_byte_offset,
                });
            }
        }
        // D3D12_DEPTH_STENCIL_DESC @496: [0]=DepthEnable, [1]=DepthWriteMask, [2]=DepthFunc
        let depth_enable = d.depth_stencil_state[0] != 0;
        let depth_write_enable = d.depth_stencil_state[1] != 0;
        let depth_func = if d.depth_stencil_state[2] == 0 {
            2 // LESS
        } else {
            d.depth_stencil_state[2]
        };
        let info = GraphicsPipelineDescInfo {
            vs_blob: vs_vec,
            ps_blob: ps_vec,
            rtv_format: rtv_fmt,
            dsv_format: d.dsv_format,
            depth_enable,
            depth_write_enable,
            depth_func,
            input_elements,
            has_root_signature: !d.p_root_signature.is_null(),
        };
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || backend.create_graphics_pipeline_ex(&info)) {
            Ok(pso_id) => {
                let obj = Box::into_raw(Box::new(ComPipelineState {
                    vtbl: dev.vtables.pso.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    pso_id,
                }));
                *out_pso = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn dev_create_compute_pipeline_state(
    this: *mut ComDevice,
    desc: *const D3D12ComputePipelineStateDesc,
    _riid: *const c_void,
    out_pso: *mut *mut c_void,
) -> i32 {
    if this.is_null() || desc.is_null() || out_pso.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live and `desc` points to a readable 56-byte compute PSO descriptor.
    unsafe {
        let dev = &*this;
        let d = &*desc;
        if d.cs.p_shader_bytecode.is_null() || d.cs.bytecode_length == 0 {
            return E_INVALIDARG;
        }
        let cs_vec =
            std::slice::from_raw_parts(d.cs.p_shader_bytecode.cast::<u8>(), d.cs.bytecode_length)
                .to_vec();
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || backend.create_compute_pipeline(&cs_vec)) {
            Ok(pso_id) => {
                let obj = Box::into_raw(Box::new(ComPipelineState {
                    vtbl: dev.vtables.pso.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    pso_id,
                }));
                *out_pso = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn dev_create_command_list(
    this: *mut ComDevice,
    _node_mask: u32,
    list_type: u32,
    _allocator: *mut ComCommandAllocator,
    initial_pso: *mut ComPipelineState,
    _riid: *const c_void,
    out_list: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_list.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComDevice` and `out_list` is writable.
    unsafe {
        let dev = &*this;
        let mut commands = Vec::new();
        if !initial_pso.is_null() {
            commands.push(RecordedCommand::SetPipelineState((*initial_pso).pso_id));
        }
        let obj = Box::into_raw(Box::new(ComGraphicsCommandList {
            vtbl: dev.vtables.cmd_list.as_ptr(),
            ref_count: AtomicU32::new(1),
            list_type,
            backend: Arc::clone(&dev.backend),
            inner: Mutex::new(CmdListInner {
                closed: false,
                commands,
            }),
        }));
        *out_list = obj as *mut c_void;
    }
    S_OK
}

extern "win64" fn dev_create_descriptor_heap(
    this: *mut ComDevice,
    desc: *const D3D12DescriptorHeapDesc,
    _riid: *const c_void,
    out_heap: *mut *mut c_void,
) -> i32 {
    if this.is_null() || desc.is_null() || out_heap.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `desc` and `out_heap` are valid guest pointers.
    unsafe {
        let dev = &*this;
        let d = *desc;
        match dev
            .backend
            .create_descriptor_heap(d.r#type, d.num_descriptors)
        {
            Ok((heap_id, base_cpu_ptr, _inc)) => {
                let obj = Box::into_raw(Box::new(ComDescriptorHeap {
                    vtbl: dev.vtables.desc_heap.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    heap_id,
                    desc: d,
                    base_cpu_ptr,
                }));
                *out_heap = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

pub(crate) fn fill_check_feature_support(
    feature: u32,
    data: *mut c_void,
    data_size: u32,
    mesh_shader_supported: bool,
) -> i32 {
    if data.is_null() || data_size == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: Caller provides a writable `data[..data_size]` buffer for `CheckFeatureSupport`.
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(data.cast::<u8>(), data_size as usize);
        match feature {
            0 => {
                // D3D12_FEATURE_D3D12_OPTIONS
                bytes.fill(0);
                if bytes.len() >= 28 {
                    let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), bytes.len() / 4);
                    words[2] = 3; // ResourceBindingTier = TIER_3
                    words[3] = 3; // TiledResourcesTier = TIER_3
                    words[4] = 3; // ConservativeRasterizationTier = TIER_3
                    words[6] = 2; // ResourceHeapTier = TIER_2
                }
            }
            1 | 16 => {
                // D3D12_FEATURE_ARCHITECTURE / ARCHITECTURE1 (UMA + CacheCoherentUMA)
                if bytes.len() >= 16 {
                    let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), bytes.len() / 4);
                    words[1] = 0; // TileBasedRenderer = FALSE
                    words[2] = 1; // UMA = TRUE
                    words[3] = 1; // CacheCoherentUMA = TRUE
                }
            }
            2 => {
                // D3D12_FEATURE_FEATURE_LEVELS: [u32 NumFeatureLevels, pad, ptr, u32 MaxSupported]
                if bytes.len() >= 20 {
                    let max_ptr = data.cast::<u8>().add(16).cast::<u32>();
                    *max_ptr = 0xC100; // D3D_FEATURE_LEVEL_12_1
                }
            }
            3 => {
                // D3D12_FEATURE_FORMAT_SUPPORT: [u32 Format, u32 Support1, u32 Support2]
                if bytes.len() >= 12 {
                    let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), 3);
                    words[1] = 0x00FF_FFFF;
                    words[2] = 0x0000_03FF;
                }
            }
            7 => {
                // D3D12_FEATURE_SHADER_MODEL
                if bytes.len() >= 4 {
                    let sm = data.cast::<u32>();
                    *sm = (*sm).min(0x66); // D3D_SHADER_MODEL_6_6
                }
            }
            12 => {
                // D3D12_FEATURE_ROOT_SIGNATURE
                if bytes.len() >= 4 {
                    let ver = data.cast::<u32>();
                    *ver = (*ver).min(2); // D3D_ROOT_SIGNATURE_VERSION_1_1
                }
            }
            18 => {
                // D3D12_FEATURE_D3D12_OPTIONS1 (WaveOps=1, WaveLaneCountMin=32, Max=64, Total=64)
                bytes.fill(0);
                if bytes.len() >= 24 {
                    let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), 6);
                    words[0] = 1;
                    words[1] = 32;
                    words[2] = 64;
                    words[3] = 64;
                    words[4] = 1;
                    words[5] = 1;
                }
            }
            19 => {
                // D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT (40-bit resource, 44-bit process)
                if bytes.len() >= 8 {
                    let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), 2);
                    words[0] = 40;
                    words[1] = 44;
                }
            }
            30 => {
                // D3D12_FEATURE_D3D12_OPTIONS5
                bytes.fill(0);
                if bytes.len() >= 12 {
                    let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), 3);
                    words[1] = 2; // RenderPassesTier = TIER_2
                }
            }
            32 => {
                // D3D12_FEATURE_D3D12_OPTIONS7 (MeshShaderTier = TIER_1 = 10)
                bytes.fill(0);
                if bytes.len() >= 4 {
                    *data.cast::<u32>() = if mesh_shader_supported { 10 } else { 0 };
                }
            }
            _ => {
                bytes.fill(0);
            }
        }
    }
    S_OK
}

extern "win64" fn dev_check_feature_support(
    this: *mut ComDevice,
    feature: u32,
    data: *mut c_void,
    data_size: u32,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    fill_check_feature_support(feature, data, data_size, true)
}

extern "win64" fn dev_get_descriptor_handle_increment_size(
    _this: *mut ComDevice,
    heap_type: u32,
) -> u32 {
    // Console D3D12.X descriptor strides: 16 bytes for Sampler (1), 32 bytes for CBV/SRV/UAV/RTV/DSV.
    if heap_type == 1 { 16 } else { 32 }
}

extern "win64" fn dev_create_root_signature(
    this: *mut ComDevice,
    _node_mask: u32,
    p_blob: *const c_void,
    blob_len: usize,
    _riid: *const c_void,
    out_root_sig: *mut *mut c_void,
) -> i32 {
    if this.is_null() || p_blob.is_null() || out_root_sig.is_null() {
        return E_POINTER;
    }
    if blob_len == 0 {
        return E_INVALIDARG;
    }
    // SAFETY: `this` is live and `[p_blob .. p_blob + blob_len]` is readable.
    unsafe {
        let dev = &*this;
        let bytes = std::slice::from_raw_parts(p_blob.cast::<u8>(), blob_len);
        match dev.backend.create_root_signature(bytes) {
            Ok(root_sig_id) => {
                let obj = Box::into_raw(Box::new(ComRootSignature {
                    vtbl: dev.vtables.root_sig.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    root_sig_id,
                }));
                *out_root_sig = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_INVALIDARG,
        }
    }
}

extern "win64" fn dev_create_constant_buffer_view(
    this: *mut ComDevice,
    desc: *const D3D12ConstantBufferViewDesc,
    dest_descriptor: D3D12CpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live; `desc` may be null for null CBV descriptor.
    unsafe {
        let dev = &*this;
        let entry = if desc.is_null() {
            DescriptorEntry::Empty
        } else {
            let d = *desc;
            DescriptorEntry::Cbv {
                gpu_va: d.buffer_location,
                size_in_bytes: d.size_in_bytes,
            }
        };
        let _ = dev.backend.write_descriptor(dest_descriptor.ptr, entry);
    }
}

extern "win64" fn dev_create_shader_resource_view(
    this: *mut ComDevice,
    resource: *mut ComResource,
    desc: *const D3D12ShaderResourceViewDesc,
    dest_descriptor: D3D12CpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live; `resource` and `desc` may be null for null descriptors.
    unsafe {
        let dev = &*this;
        let entry = if resource.is_null() {
            DescriptorEntry::Empty
        } else {
            let (first_element, num_elements, stride) = if desc.is_null() {
                (0, 0, 0)
            } else {
                let d = &*desc;
                (
                    d.buffer.first_element,
                    d.buffer.num_elements,
                    d.buffer.structure_byte_stride,
                )
            };
            DescriptorEntry::Srv {
                resource_id: (*resource).resource_id,
                first_element,
                num_elements,
                stride,
            }
        };
        let _ = dev.backend.write_descriptor(dest_descriptor.ptr, entry);
    }
}

extern "win64" fn dev_create_unordered_access_view(
    this: *mut ComDevice,
    resource: *mut ComResource,
    _counter_resource: *mut ComResource,
    desc: *const D3D12UnorderedAccessViewDesc,
    dest_descriptor: D3D12CpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live; `resource` and `desc` may be null for null descriptors.
    unsafe {
        let dev = &*this;
        let entry = if resource.is_null() {
            DescriptorEntry::Empty
        } else {
            let (first_element, num_elements, stride) = if desc.is_null() {
                (0, 0, 0)
            } else {
                let d = &*desc;
                (
                    d.buffer.first_element,
                    d.buffer.num_elements,
                    d.buffer.structure_byte_stride,
                )
            };
            DescriptorEntry::Uav {
                resource_id: (*resource).resource_id,
                first_element,
                num_elements,
                stride,
            }
        };
        let _ = dev.backend.write_descriptor(dest_descriptor.ptr, entry);
    }
}

extern "win64" fn dev_create_render_target_view(
    this: *mut ComDevice,
    resource: *mut ComResource,
    _desc: *const c_void,
    dest_descriptor: D3D12CpuDescriptorHandle,
) {
    if this.is_null() || resource.is_null() {
        return;
    }
    // SAFETY: `this` and `resource` are live COM objects created by this backend.
    unsafe {
        let dev = &*this;
        let res_id = (*resource).resource_id;
        let _ = dev
            .backend
            .create_render_target_view(res_id, dest_descriptor.ptr);
    }
}

extern "win64" fn dev_create_depth_stencil_view(
    this: *mut ComDevice,
    resource: *mut ComResource,
    desc: *const D3D12DepthStencilViewDesc,
    dest_descriptor: D3D12CpuDescriptorHandle,
) {
    if this.is_null() || resource.is_null() {
        return;
    }
    // SAFETY: `this` and `resource` are live COM objects; `desc` is optional.
    unsafe {
        let dev = &*this;
        let flags = if desc.is_null() { 0 } else { (*desc).flags };
        let _ = dev.backend.create_depth_stencil_view(
            (*resource).resource_id,
            flags,
            dest_descriptor.ptr,
        );
    }
}

/// `D3D12_SAMPLER_DESC` (52 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12SamplerDesc {
    pub filter: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison_func: u32,
    pub border_color: [f32; 4],
    pub min_lod: f32,
    pub max_lod: f32,
}

const _: () = assert!(std::mem::size_of::<D3D12SamplerDesc>() == 52);

extern "win64" fn dev_create_sampler(
    this: *mut ComDevice,
    desc: *const D3D12SamplerDesc,
    dest_descriptor: D3D12CpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live; `desc` is an optional readable 52-byte `D3D12_SAMPLER_DESC`.
    unsafe {
        let dev = &*this;
        let info = if desc.is_null() {
            None
        } else {
            let d = *desc;
            Some(crate::vulkan::SamplerDescInfo {
                filter: d.filter,
                address_u: d.address_u,
                address_v: d.address_v,
                address_w: d.address_w,
                min_lod: d.min_lod,
                max_lod: d.max_lod,
            })
        };
        let backend = Arc::clone(&dev.backend);
        let sampler_id = exec_native(move || backend.create_sampler(info)).unwrap_or(0);
        let _ = dev
            .backend
            .write_descriptor(dest_descriptor.ptr, DescriptorEntry::Sampler { sampler_id });
    }
}

#[allow(clippy::too_many_arguments)]
extern "win64" fn dev_copy_descriptors(
    this: *mut ComDevice,
    num_dest_ranges: u32,
    p_dest_starts: *const D3D12CpuDescriptorHandle,
    p_dest_sizes: *const u32,
    num_src_ranges: u32,
    p_src_starts: *const D3D12CpuDescriptorHandle,
    p_src_sizes: *const u32,
    heap_type: u32,
) {
    if this.is_null()
        || num_dest_ranges == 0
        || num_src_ranges == 0
        || p_dest_starts.is_null()
        || p_src_starts.is_null()
    {
        return;
    }
    let inc = if heap_type == 1 { 16usize } else { 32usize };
    // SAFETY: Caller supplies valid range arrays matching `num_dest_ranges` and `num_src_ranges`.
    unsafe {
        let dev = &*this;
        let mut src_handles = Vec::new();
        for i in 0..(num_src_ranges as usize) {
            let start = (*p_src_starts.add(i)).ptr;
            let count = if p_src_sizes.is_null() {
                1u32
            } else {
                *p_src_sizes.add(i)
            };
            for j in 0..(count as usize) {
                let entry = dev
                    .backend
                    .resolve_descriptor(start.saturating_add(j * inc))
                    .unwrap_or(DescriptorEntry::Empty);
                src_handles.push(entry);
            }
        }
        let mut src_idx = 0usize;
        for i in 0..(num_dest_ranges as usize) {
            let start = (*p_dest_starts.add(i)).ptr;
            let count = if p_dest_sizes.is_null() {
                1u32
            } else {
                *p_dest_sizes.add(i)
            };
            for j in 0..(count as usize) {
                if src_idx >= src_handles.len() {
                    return;
                }
                let _ = dev
                    .backend
                    .write_descriptor(start.saturating_add(j * inc), src_handles[src_idx]);
                src_idx += 1;
            }
        }
    }
}

extern "win64" fn dev_copy_descriptors_simple(
    this: *mut ComDevice,
    num_descriptors: u32,
    dest_start: D3D12CpuDescriptorHandle,
    src_start: D3D12CpuDescriptorHandle,
    heap_type: u32,
) {
    if this.is_null() || num_descriptors == 0 {
        return;
    }
    // SAFETY: `this` is a live `ComDevice`.
    unsafe {
        (*this).backend.copy_descriptors_simple(
            num_descriptors,
            dest_start.ptr,
            src_start.ptr,
            heap_type,
        );
    }
}

pub(crate) fn compute_resource_allocation_info_dual_abi(
    a0: *mut c_void,
    a1: *mut c_void,
    num_resource_descs: u32,
    p_resource_descs: *const D3D12ResourceDesc,
) -> *mut D3D12ResourceAllocationInfo {
    let out_info = if is_known_device_object(a0) {
        a1.cast::<D3D12ResourceAllocationInfo>()
    } else {
        a0.cast::<D3D12ResourceAllocationInfo>()
    };
    if !out_info.is_null() {
        let mut total_size = 0u64;
        if num_resource_descs > 0 && !p_resource_descs.is_null() {
            // SAFETY: Caller supplies `num_resource_descs` readable `D3D12ResourceDesc` entries.
            let descs = unsafe {
                std::slice::from_raw_parts(p_resource_descs, num_resource_descs as usize)
            };
            for d in descs {
                let sz = if d.dimension == crate::vulkan::D3D12_RESOURCE_DIMENSION_BUFFER {
                    d.width.max(1).next_multiple_of(65536)
                } else {
                    (d.width.max(1) * u64::from(d.height.max(1)) * 4).next_multiple_of(65536)
                };
                total_size = total_size.saturating_add(sz);
            }
        }
        // SAFETY: Win64 hidden struct-return buffer `out_info` is writable for 16 bytes.
        unsafe {
            *out_info = D3D12ResourceAllocationInfo {
                size_in_bytes: total_size.max(65536),
                alignment: 65536,
            };
        }
    }
    out_info
}

extern "win64" fn dev_get_resource_allocation_info(
    a0: *mut c_void,
    a1: *mut c_void,
    _visible_mask: u32,
    num_resource_descs: u32,
    p_resource_descs: *const D3D12ResourceDesc,
) -> *mut D3D12ResourceAllocationInfo {
    compute_resource_allocation_info_dual_abi(a0, a1, num_resource_descs, p_resource_descs)
}

pub(crate) fn compute_custom_heap_properties_dual_abi(
    a0: *mut c_void,
    a1: *mut c_void,
    node_mask: u32,
    heap_type: u32,
) -> *mut D3D12HeapProperties {
    let out_props = if is_known_device_object(a0) {
        a1.cast::<D3D12HeapProperties>()
    } else {
        a0.cast::<D3D12HeapProperties>()
    };
    if !out_props.is_null() {
        let mask = if node_mask == 0 { 1 } else { node_mask };
        // Console UMA custom heap mapping: READBACK (3) -> WRITE_BACK (2), else WRITE_COMBINE (1).
        let cpu_prop = if heap_type == 3 { 2 } else { 1 };
        // SAFETY: Win64 hidden struct-return buffer `out_props` is writable for 20 bytes.
        unsafe {
            *out_props = D3D12HeapProperties {
                r#type: 4, // D3D12_HEAP_TYPE_CUSTOM
                cpu_page_property: cpu_prop,
                memory_pool_preference: 1, // D3D12_MEMORY_POOL_L0 (UMA)
                creation_node_mask: mask,
                visible_node_mask: mask,
            };
        }
    }
    out_props
}

extern "win64" fn dev_get_custom_heap_properties(
    a0: *mut c_void,
    a1: *mut c_void,
    node_mask: u32,
    heap_type: u32,
) -> *mut D3D12HeapProperties {
    compute_custom_heap_properties_dual_abi(a0, a1, node_mask, heap_type)
}

pub(crate) fn compute_adapter_luid_dual_abi(a0: *mut c_void, a1: *mut c_void) -> *mut D3D12Luid {
    let out_luid = if is_known_device_object(a0) {
        a1.cast::<D3D12Luid>()
    } else {
        a0.cast::<D3D12Luid>()
    };
    if !out_luid.is_null() {
        // SAFETY: Win64 hidden struct-return buffer `out_luid` is writable for 8 bytes.
        unsafe {
            *out_luid = D3D12Luid {
                low_part: 0x5832_4C01,
                high_part: 0,
            };
        }
    }
    out_luid
}

extern "win64" fn dev_get_adapter_luid(a0: *mut c_void, a1: *mut c_void) -> *mut D3D12Luid {
    compute_adapter_luid_dual_abi(a0, a1)
}

extern "win64" fn dev_get_device_removed_reason(_this: *mut ComDevice) -> i32 {
    S_OK
}

extern "win64" fn dev_create_placed_resource_x(
    this: *mut ComDevice,
    _gpu_va: u64,
    desc: *const D3D12ResourceDesc,
    initial_state: u32,
    optimized_clear_value: *const c_void,
    riid: *const c_void,
    out_resource: *mut *mut c_void,
) -> i32 {
    dev_create_committed_resource(
        this,
        ptr::null(),
        0,
        desc,
        initial_state,
        optimized_clear_value,
        riid,
        out_resource,
    )
}

extern "win64" fn dev_set_debug_flags_x(_this: *mut ComDevice, _flags: u32) -> i32 {
    S_OK
}

extern "win64" fn dev_create_graphics_pipeline_state_x(
    this: *mut ComDevice,
    desc: *const D3D12GraphicsPipelineStateDesc,
    _ext_count: u32,
    _extensions: *const c_void,
    riid: *const c_void,
    out_pso: *mut *mut c_void,
) -> i32 {
    dev_create_graphics_pipeline_state(this, desc, riid, out_pso)
}

extern "win64" fn dev_create_compute_pipeline_state_x(
    this: *mut ComDevice,
    desc: *const D3D12ComputePipelineStateDesc,
    _ext_count: u32,
    _extensions: *const c_void,
    riid: *const c_void,
    out_pso: *mut *mut c_void,
) -> i32 {
    dev_create_compute_pipeline_state(this, desc, riid, out_pso)
}

extern "win64" fn dev_set_frame_interval_x(
    _this: *mut ComDevice,
    _output: *mut c_void,
    interval_us: u32,
    _periodicity: u32,
    _flags: u32,
) -> i32 {
    if interval_us > 0 {
        FRAME_INTERVAL_US.store(interval_us, Ordering::Relaxed);
    }
    S_OK
}

extern "win64" fn dev_schedule_frame_event_x(
    _this: *mut ComDevice,
    _event_type: u32,
    _offset_us: u32,
    _ancillary: *const c_void,
    _flags: u32,
) -> i32 {
    S_OK
}

extern "win64" fn dev_wait_frame_event_x(
    _this: *mut ComDevice,
    _event_type: u32,
    _timeout_ms: u32,
    _ancillary: *const c_void,
    _flags: u32,
    token_in_out: *mut u64,
) -> i32 {
    if !token_in_out.is_null() {
        // SAFETY: Caller supplied writable `token_in_out`.
        unsafe { *token_in_out = NEXT_FRAME_TOKEN.fetch_add(1, Ordering::Relaxed) };
    }
    S_OK
}

extern "win64" fn dev_get_frame_statistics_x(
    _this: *mut ComDevice,
    _token: u64,
    _event_type: u32,
    count_in_out: *mut u32,
    _stats: *mut c_void,
) -> i32 {
    if !count_in_out.is_null() {
        // SAFETY: Report zero pending stats records safely.
        unsafe { *count_in_out = 0 };
    }
    S_OK
}

extern "win64" fn dev_create_heap(
    this: *mut ComDevice,
    desc: *const D3D12HeapDesc,
    _riid: *const c_void,
    out_heap: *mut *mut c_void,
) -> i32 {
    if this.is_null() || desc.is_null() || out_heap.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `desc` is readable and `out_heap` is writable.
    unsafe {
        let dev = &*this;
        let d = *desc;
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || {
            backend.create_heap(d.size_in_bytes, d.properties.r#type, d.alignment, d.flags)
        }) {
            Ok(heap_id) => {
                let obj = Box::into_raw(Box::new(ComHeap {
                    vtbl: dev.vtables.heap.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    heap_id,
                    desc: d,
                }));
                *out_heap = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn dev_create_placed_resource(
    this: *mut ComDevice,
    heap: *mut ComHeap,
    heap_offset: u64,
    desc: *const D3D12ResourceDesc,
    _initial_state: u32,
    _optimized_clear_value: *const c_void,
    _riid: *const c_void,
    out_resource: *mut *mut c_void,
) -> i32 {
    if this.is_null() || heap.is_null() || desc.is_null() || out_resource.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` and `heap` are live COM objects; `desc` and `out_resource` are valid.
    unsafe {
        let dev = &*this;
        let heap_id = (*heap).heap_id;
        let d = *desc;
        let info = ResourceDescInfo {
            dimension: d.dimension,
            width: d.width,
            height: d.height,
            depth_or_array_size: d.depth_or_array_size,
            mip_levels: d.mip_levels,
            format: d.format,
            flags: d.flags,
        };
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || backend.create_placed_resource(heap_id, heap_offset, info)) {
            Ok(resource_id) => {
                let obj = Box::into_raw(Box::new(ComResource {
                    vtbl: dev.vtables.resource.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    backend: Arc::clone(&dev.backend),
                    resource_id,
                    desc: d,
                }));
                *out_resource = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_INVALIDARG,
        }
    }
}

#[allow(clippy::too_many_arguments)]
extern "win64" fn dev_get_copyable_footprints(
    _this: *mut ComDevice,
    p_resource_desc: *const D3D12ResourceDesc,
    first_subresource: u32,
    num_subresources: u32,
    base_offset: u64,
    p_layouts: *mut D3D12PlacedSubresourceFootprint,
    p_num_rows: *mut u32,
    p_row_size_in_bytes: *mut u64,
    p_total_bytes: *mut u64,
) {
    if p_resource_desc.is_null() || num_subresources == 0 {
        return;
    }
    // SAFETY: Caller supplies readable `p_resource_desc` and optional writable arrays of length `num_subresources`.
    unsafe {
        let d = &*p_resource_desc;
        let mips = u32::from(d.mip_levels.max(1));
        let bpp = 4u32;
        let mut cursor = base_offset;
        for i in 0..num_subresources {
            let sub = first_subresource + i;
            let mip = sub % mips;
            let w = ((d.width as u32) >> mip).max(1);
            let h = (d.height >> mip).max(1);
            let row_bytes = u64::from(w) * u64::from(bpp);
            let row_pitch = (w * bpp).next_multiple_of(256);
            let sub_bytes = u64::from(row_pitch) * u64::from(h);
            let aligned_offset = if i == 0 {
                cursor
            } else {
                cursor.next_multiple_of(512)
            };
            if !p_layouts.is_null() {
                *p_layouts.add(i as usize) = D3D12PlacedSubresourceFootprint {
                    offset: aligned_offset,
                    footprint: D3D12SubresourceFootprint {
                        format: d.format,
                        width: w,
                        height: h,
                        depth: 1,
                        row_pitch,
                    },
                    _pad28: 0,
                };
            }
            if !p_num_rows.is_null() {
                *p_num_rows.add(i as usize) = h;
            }
            if !p_row_size_in_bytes.is_null() {
                *p_row_size_in_bytes.add(i as usize) = row_bytes;
            }
            cursor = aligned_offset.saturating_add(sub_bytes);
        }
        if !p_total_bytes.is_null() {
            *p_total_bytes = cursor.saturating_sub(base_offset);
        }
    }
}

extern "win64" fn dev_create_committed_resource(
    this: *mut ComDevice,
    _heap_props: *const D3D12HeapProperties,
    _heap_flags: u32,
    desc: *const D3D12ResourceDesc,
    _initial_state: u32,
    _optimized_clear_value: *const c_void,
    _riid: *const c_void,
    out_resource: *mut *mut c_void,
) -> i32 {
    if this.is_null() || desc.is_null() || out_resource.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `desc` and `out_resource` are valid pointers.
    unsafe {
        let dev = &*this;
        let d = *desc;
        let info = ResourceDescInfo {
            dimension: d.dimension,
            width: d.width,
            height: d.height,
            depth_or_array_size: d.depth_or_array_size,
            mip_levels: d.mip_levels,
            format: d.format,
            flags: d.flags,
        };
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || backend.create_committed_resource(info)) {
            Ok(resource_id) => {
                let obj = Box::into_raw(Box::new(ComResource {
                    vtbl: dev.vtables.resource.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    backend: Arc::clone(&dev.backend),
                    resource_id,
                    desc: d,
                }));
                *out_resource = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn dev_create_fence(
    this: *mut ComDevice,
    initial_value: u64,
    _flags: u32,
    _riid: *const c_void,
    out_fence: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_fence.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live and `out_fence` is writable.
    unsafe {
        let dev = &*this;
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || backend.create_fence(initial_value)) {
            Ok(fence_id) => {
                let obj = Box::into_raw(Box::new(ComFence {
                    vtbl: dev.vtables.fence.as_ptr(),
                    ref_count: AtomicU32::new(1),
                    backend: Arc::clone(&dev.backend),
                    fence_id,
                }));
                *out_fence = obj as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

// ============================================================================
// ID3D12CommandQueueX methods (legacy console layout: Signal@14, PresentX@30)
// ============================================================================

extern "win64" fn queue_execute_command_lists(
    this: *mut ComCommandQueue,
    num_lists: u32,
    pp_lists: *const *mut ComGraphicsCommandList,
) {
    if this.is_null() || num_lists == 0 || pp_lists.is_null() {
        return;
    }
    // SAFETY: Caller passes `num_lists` pointers to live `ComGraphicsCommandList` objects.
    unsafe {
        let q = &*this;
        let ptrs = std::slice::from_raw_parts(pp_lists, num_lists as usize);
        let mut snapshots = Vec::with_capacity(ptrs.len());
        for &list_ptr in ptrs {
            if !list_ptr.is_null()
                && let Ok(guard) = (*list_ptr).inner.lock()
            {
                snapshots.push(guard.commands.clone());
            }
        }
        let backend = Arc::clone(&q.backend);
        let _ = exec_native(move || backend.execute_command_lists(&snapshots));
    }
}

extern "win64" fn queue_signal(
    this: *mut ComCommandQueue,
    fence: *mut ComFence,
    value: u64,
) -> i32 {
    if this.is_null() || fence.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` and `fence` are live COM objects created by this backend.
    unsafe {
        let _ = (*this).queue_type;
        let backend = Arc::clone(&(*this).backend);
        let fence_id = (*fence).fence_id;
        match exec_native(move || backend.queue_signal_fence(fence_id, value)) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn queue_wait(this: *mut ComCommandQueue, fence: *mut ComFence, value: u64) -> i32 {
    if this.is_null() || fence.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` and `fence` are live COM objects created by this backend.
    unsafe {
        let backend = Arc::clone(&(*this).backend);
        let fence_id = (*fence).fence_id;
        match exec_native(move || backend.wait_fence_value(fence_id, value)) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn queue_get_timestamp_frequency(
    this: *mut ComCommandQueue,
    out_freq: *mut u64,
) -> i32 {
    if this.is_null() || out_freq.is_null() {
        return E_POINTER;
    }
    // Console GPU timestamp frequency: 100 MHz (100_000_000 ticks/sec).
    // SAFETY: `out_freq` was checked non-null.
    unsafe { *out_freq = 100_000_000 };
    S_OK
}

extern "win64" fn queue_present_x(
    this: *mut ComCommandQueue,
    plane_count: u32,
    planes: *const D3D12XboxPresentPlaneParameters,
    _params: *const c_void,
) -> i32 {
    if this.is_null() || plane_count == 0 || planes.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: Caller supplies at least `plane_count` initialized `D3D12XboxPresentPlaneParameters`.
    unsafe {
        let q = &*this;
        let plane = &*planes;
        if plane.resource_count == 0 || plane.pp_resources.is_null() {
            return E_INVALIDARG;
        }
        let res_ptr = *plane.pp_resources;
        if res_ptr.is_null() {
            return E_POINTER;
        }
        let backend = Arc::clone(&q.backend);
        let res_id = (*res_ptr).resource_id;
        match exec_native(move || backend.present_resource(res_id)) {
            Ok(_) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

// ============================================================================
// ID3D12CommandAllocator methods
// ============================================================================

extern "win64" fn alloc_reset(this: *mut ComCommandAllocator) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComCommandAllocator`.
    unsafe {
        let _ = (*this).list_type;
        (*this).generation.fetch_add(1, Ordering::AcqRel);
    }
    S_OK
}

// ============================================================================
// ID3D12GraphicsCommandListX methods
// ============================================================================

extern "win64" fn list_get_type(this: *mut ComGraphicsCommandList) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe { (*this).list_type }
}

extern "win64" fn list_close(this: *mut ComGraphicsCommandList) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe {
        match (*this).inner.lock() {
            Ok(mut guard) => {
                guard.closed = true;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn list_reset(
    this: *mut ComGraphicsCommandList,
    _allocator: *mut ComCommandAllocator,
    initial_pso: *mut ComPipelineState,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe {
        match (*this).inner.lock() {
            Ok(mut guard) => {
                guard.closed = false;
                guard.commands.clear();
                if !initial_pso.is_null() {
                    guard
                        .commands
                        .push(RecordedCommand::SetPipelineState((*initial_pso).pso_id));
                }
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn list_draw_instanced(
    this: *mut ComGraphicsCommandList,
    vertex_count: u32,
    instance_count: u32,
    start_vertex: u32,
    start_instance: u32,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::DrawInstanced {
                vertex_count,
                instance_count,
                start_vertex,
                start_instance,
            });
        }
    }
}

extern "win64" fn list_draw_indexed_instanced(
    this: *mut ComGraphicsCommandList,
    index_count: u32,
    instance_count: u32,
    start_index: u32,
    base_vertex: i32,
    start_instance: u32,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::DrawIndexedInstanced {
                index_count,
                instance_count,
                start_index,
                base_vertex,
                start_instance,
            });
        }
    }
}

extern "win64" fn list_dispatch(this: *mut ComGraphicsCommandList, x: u32, y: u32, z: u32) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::Dispatch { x, y, z });
        }
    }
}

extern "win64" fn list_copy_buffer_region(
    this: *mut ComGraphicsCommandList,
    dst: *mut ComResource,
    dst_offset: u64,
    src: *mut ComResource,
    src_offset: u64,
    num_bytes: u64,
) {
    if this.is_null() || dst.is_null() || src.is_null() {
        return;
    }
    // SAFETY: `this`, `dst`, and `src` are live COM objects.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::CopyBufferRegion {
                dst_res_id: (*dst).resource_id,
                dst_offset,
                src_res_id: (*src).resource_id,
                src_offset,
                num_bytes,
            });
        }
    }
}

extern "win64" fn list_copy_texture_region(
    this: *mut ComGraphicsCommandList,
    dst: *const D3D12TextureCopyLocation,
    dst_x: u32,
    dst_y: u32,
    _dst_z: u32,
    src: *const D3D12TextureCopyLocation,
    src_box: *const D3D12Box,
) {
    if this.is_null() || dst.is_null() || src.is_null() {
        return;
    }
    // SAFETY: `this`, `dst`, and `src` are valid; `src_box` is optional.
    unsafe {
        let d = &*dst;
        let s = &*src;
        if d.p_resource.is_null() || s.p_resource.is_null() {
            return;
        }
        let dst_is_fp = d.r#type == 1;
        let src_is_fp = s.r#type == 1;
        let (dst_offset, dst_row_pitch, dst_w, dst_h) = if dst_is_fp {
            let fp = d.u.placed_footprint;
            (
                fp.offset,
                fp.footprint.row_pitch,
                fp.footprint.width,
                fp.footprint.height,
            )
        } else {
            (0, 0, 0, 0)
        };
        let (src_offset, src_row_pitch, src_w, src_h) = if src_is_fp {
            let fp = s.u.placed_footprint;
            (
                fp.offset,
                fp.footprint.row_pitch,
                fp.footprint.width,
                fp.footprint.height,
            )
        } else {
            (0, 0, 0, 0)
        };
        let (width, height) = if !src_box.is_null() {
            let b = &*src_box;
            (
                b.right.saturating_sub(b.left),
                b.bottom.saturating_sub(b.top),
            )
        } else if src_w > 0 && src_h > 0 {
            (src_w, src_h)
        } else {
            (dst_w, dst_h)
        };
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::CopyTextureRegion {
                dst_res_id: (*d.p_resource).resource_id,
                dst_is_footprint: dst_is_fp,
                dst_offset,
                dst_row_pitch,
                dst_x,
                dst_y,
                src_res_id: (*s.p_resource).resource_id,
                src_is_footprint: src_is_fp,
                src_offset,
                src_row_pitch,
                width,
                height,
            });
        }
    }
}

extern "win64" fn list_copy_resource(
    this: *mut ComGraphicsCommandList,
    dst: *mut ComResource,
    src: *mut ComResource,
) {
    if this.is_null() || dst.is_null() || src.is_null() {
        return;
    }
    // SAFETY: `this`, `dst`, and `src` are live COM objects.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::CopyResource {
                dst_res_id: (*dst).resource_id,
                src_res_id: (*src).resource_id,
            });
        }
    }
}

extern "win64" fn list_ia_set_primitive_topology(this: *mut ComGraphicsCommandList, topology: u32) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is a live `ComGraphicsCommandList`.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetPrimitiveTopology(topology));
        }
    }
}

extern "win64" fn list_rs_set_viewports(
    this: *mut ComGraphicsCommandList,
    num_viewports: u32,
    p_viewports: *const D3D12Viewport,
) {
    if this.is_null() || num_viewports == 0 || p_viewports.is_null() {
        return;
    }
    // SAFETY: `p_viewports` points to `num_viewports` initialized `D3D12Viewport` entries.
    unsafe {
        let slice = std::slice::from_raw_parts(p_viewports, num_viewports as usize);
        let vps = slice
            .iter()
            .map(|v| vk::Viewport {
                x: v.top_left_x,
                y: v.top_left_y,
                width: v.width,
                height: v.height,
                min_depth: v.min_depth,
                max_depth: v.max_depth,
            })
            .collect();
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::SetViewports(vps));
        }
    }
}

extern "win64" fn list_rs_set_scissor_rects(
    this: *mut ComGraphicsCommandList,
    num_rects: u32,
    p_rects: *const D3D12Rect,
) {
    if this.is_null() || num_rects == 0 || p_rects.is_null() {
        return;
    }
    // SAFETY: `p_rects` points to `num_rects` initialized `D3D12Rect` entries.
    unsafe {
        let slice = std::slice::from_raw_parts(p_rects, num_rects as usize);
        let rects = slice
            .iter()
            .map(|r| vk::Rect2D {
                offset: vk::Offset2D {
                    x: r.left,
                    y: r.top,
                },
                extent: vk::Extent2D {
                    width: r.right.saturating_sub(r.left).max(0) as u32,
                    height: r.bottom.saturating_sub(r.top).max(0) as u32,
                },
            })
            .collect();
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::SetScissors(rects));
        }
    }
}

extern "win64" fn list_set_pipeline_state(
    this: *mut ComGraphicsCommandList,
    pso: *mut ComPipelineState,
) {
    if this.is_null() || pso.is_null() {
        return;
    }
    // SAFETY: `this` and `pso` are live COM objects.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetPipelineState((*pso).pso_id));
        }
    }
}

extern "win64" fn list_resource_barrier(
    this: *mut ComGraphicsCommandList,
    num_barriers: u32,
    p_barriers: *const D3D12ResourceBarrier,
) -> u32 {
    if this.is_null() || num_barriers == 0 || p_barriers.is_null() {
        return 0;
    }
    // SAFETY: `p_barriers` points to `num_barriers` initialized `D3D12ResourceBarrier` entries.
    unsafe {
        let slice = std::slice::from_raw_parts(p_barriers, num_barriers as usize);
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            for b in slice {
                if b.r#type == 0 {
                    let tr = b.u.transition;
                    if !tr.p_resource.is_null() {
                        guard.commands.push(RecordedCommand::ResourceBarrier {
                            resource_id: (*tr.p_resource).resource_id,
                            state_before: tr.state_before,
                            state_after: tr.state_after,
                        });
                    }
                }
            }
        }
    }
    0
}

extern "win64" fn list_set_descriptor_heaps(
    this: *mut ComGraphicsCommandList,
    num_heaps: u32,
    pp_heaps: *const *mut ComDescriptorHeap,
) {
    if this.is_null() || num_heaps == 0 || pp_heaps.is_null() {
        return;
    }
    // SAFETY: `pp_heaps` points to `num_heaps` live `ComDescriptorHeap` pointers.
    unsafe {
        let slice = std::slice::from_raw_parts(pp_heaps, num_heaps as usize);
        let ids: Vec<u64> = slice
            .iter()
            .filter(|&&h| !h.is_null())
            .map(|&h| (*h).heap_id)
            .collect();
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetDescriptorHeaps(ids));
        }
    }
}

extern "win64" fn list_set_compute_root_signature(
    this: *mut ComGraphicsCommandList,
    root_sig: *mut ComRootSignature,
) {
    if this.is_null() || root_sig.is_null() {
        return;
    }
    // SAFETY: `this` and `root_sig` are live COM objects.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRootSignature(
                    (*root_sig).root_sig_id,
                ));
        }
    }
}

extern "win64" fn list_set_graphics_root_signature(
    this: *mut ComGraphicsCommandList,
    root_sig: *mut ComRootSignature,
) {
    if this.is_null() || root_sig.is_null() {
        return;
    }
    // SAFETY: `this` and `root_sig` are live COM objects.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRootSignature(
                    (*root_sig).root_sig_id,
                ));
        }
    }
}

extern "win64" fn list_set_compute_root_descriptor_table(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    base_descriptor: D3D12GpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRootDescriptorTable {
                    root_param_index,
                    gpu_handle: base_descriptor.ptr,
                });
        }
    }
}

extern "win64" fn list_set_graphics_root_descriptor_table(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    base_descriptor: D3D12GpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRootDescriptorTable {
                    root_param_index,
                    gpu_handle: base_descriptor.ptr,
                });
        }
    }
}

extern "win64" fn list_set_compute_root_32bit_constant(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    src_data: u32,
    dest_offset: u32,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRoot32BitConstants {
                    root_param_index,
                    values: vec![src_data],
                    dest_offset,
                });
        }
    }
}

extern "win64" fn list_set_graphics_root_32bit_constant(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    src_data: u32,
    dest_offset: u32,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRoot32BitConstants {
                    root_param_index,
                    values: vec![src_data],
                    dest_offset,
                });
        }
    }
}

extern "win64" fn list_set_compute_root_32bit_constants(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    num_32bit_values: u32,
    p_src_data: *const c_void,
    dest_offset: u32,
) {
    if this.is_null() || num_32bit_values == 0 || p_src_data.is_null() {
        return;
    }
    // SAFETY: `p_src_data` points to `num_32bit_values` readable `u32` words.
    unsafe {
        let vals = std::slice::from_raw_parts(p_src_data.cast::<u32>(), num_32bit_values as usize)
            .to_vec();
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRoot32BitConstants {
                    root_param_index,
                    values: vals,
                    dest_offset,
                });
        }
    }
}

extern "win64" fn list_set_graphics_root_32bit_constants(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    num_32bit_values: u32,
    p_src_data: *const c_void,
    dest_offset: u32,
) {
    if this.is_null() || num_32bit_values == 0 || p_src_data.is_null() {
        return;
    }
    // SAFETY: `p_src_data` points to `num_32bit_values` readable `u32` words.
    unsafe {
        let vals = std::slice::from_raw_parts(p_src_data.cast::<u32>(), num_32bit_values as usize)
            .to_vec();
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRoot32BitConstants {
                    root_param_index,
                    values: vals,
                    dest_offset,
                });
        }
    }
}

extern "win64" fn list_set_compute_root_constant_buffer_view(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    gpu_va: u64,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRootConstantBufferView {
                    root_param_index,
                    gpu_va,
                });
        }
    }
}

extern "win64" fn list_set_graphics_root_constant_buffer_view(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    gpu_va: u64,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRootConstantBufferView {
                    root_param_index,
                    gpu_va,
                });
        }
    }
}

extern "win64" fn list_set_compute_root_shader_resource_view(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    gpu_va: u64,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRootShaderResourceView {
                    root_param_index,
                    gpu_va,
                });
        }
    }
}

extern "win64" fn list_set_graphics_root_shader_resource_view(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    gpu_va: u64,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRootShaderResourceView {
                    root_param_index,
                    gpu_va,
                });
        }
    }
}

extern "win64" fn list_set_compute_root_unordered_access_view(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    gpu_va: u64,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetComputeRootUnorderedAccessView {
                    root_param_index,
                    gpu_va,
                });
        }
    }
}

extern "win64" fn list_set_graphics_root_unordered_access_view(
    this: *mut ComGraphicsCommandList,
    root_param_index: u32,
    gpu_va: u64,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetGraphicsRootUnorderedAccessView {
                    root_param_index,
                    gpu_va,
                });
        }
    }
}

extern "win64" fn list_ia_set_index_buffer(
    this: *mut ComGraphicsCommandList,
    view: *const D3D12IndexBufferView,
) {
    if this.is_null() || view.is_null() {
        return;
    }
    // SAFETY: `this` is live and `view` points to a readable 16-byte `D3D12IndexBufferView`.
    unsafe {
        let v = *view;
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::IASetIndexBuffer {
                gpu_va: v.buffer_location,
                size_in_bytes: v.size_in_bytes,
                format: v.format,
            });
        }
    }
}

extern "win64" fn list_ia_set_vertex_buffers(
    this: *mut ComGraphicsCommandList,
    start_slot: u32,
    num_views: u32,
    p_views: *const D3D12VertexBufferView,
) {
    if this.is_null() || num_views == 0 || p_views.is_null() {
        return;
    }
    // SAFETY: `p_views` points to `num_views` readable `D3D12VertexBufferView` entries.
    unsafe {
        let slice = std::slice::from_raw_parts(p_views, num_views as usize);
        let views = slice
            .iter()
            .map(|v| (v.buffer_location, v.size_in_bytes, v.stride_in_bytes))
            .collect();
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::IASetVertexBuffers { start_slot, views });
        }
    }
}

extern "win64" fn list_om_set_render_targets(
    this: *mut ComGraphicsCommandList,
    num_rtvs: u32,
    p_rtv_descriptors: *const D3D12CpuDescriptorHandle,
    rts_single_handle_to_descriptor_range: i32,
    p_dsv_descriptor: *const D3D12CpuDescriptorHandle,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live; `p_rtv_descriptors` and `p_dsv_descriptor` are optional pointers.
    unsafe {
        let list = &*this;
        let mut rts = Vec::with_capacity(num_rtvs as usize);
        if num_rtvs > 0 && !p_rtv_descriptors.is_null() {
            if rts_single_handle_to_descriptor_range != 0 {
                let base = (*p_rtv_descriptors).ptr;
                for i in 0..(num_rtvs as usize) {
                    if let Some(res_id) = list.backend.resolve_rtv(base + i * 32) {
                        rts.push(res_id);
                    }
                }
            } else {
                let handles = std::slice::from_raw_parts(p_rtv_descriptors, num_rtvs as usize);
                for h in handles {
                    if let Some(res_id) = list.backend.resolve_rtv(h.ptr) {
                        rts.push(res_id);
                    }
                }
            }
        }
        let dsv = if !p_dsv_descriptor.is_null() {
            list.backend.resolve_dsv((*p_dsv_descriptor).ptr)
        } else {
            None
        };
        if let Ok(mut guard) = list.inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::SetRenderTargets { rtvs: rts, dsv });
        }
    }
}

extern "win64" fn list_clear_depth_stencil_view(
    this: *mut ComGraphicsCommandList,
    dsv: D3D12CpuDescriptorHandle,
    clear_flags: u32,
    depth: f32,
    _stencil: u8,
    _num_rects: u32,
    _p_rects: *const D3D12Rect,
) {
    if this.is_null() || (clear_flags & 1) == 0 {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        let list = &*this;
        if let Some(resource_id) = list.backend.resolve_dsv(dsv.ptr)
            && let Ok(mut guard) = list.inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::ClearDepthStencilView { resource_id, depth });
        }
    }
}

extern "win64" fn list_dispatch_mesh(this: *mut ComGraphicsCommandList, x: u32, y: u32, z: u32) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::DispatchMesh { x, y, z });
        }
    }
}

extern "win64" fn list_dispatch_mesh_x(
    this: *mut ComGraphicsCommandList,
    x: u32,
    flags: u64,
    indirect: u64,
    payload: u64,
) {
    if this.is_null() || flags != 0 || indirect != 0 || payload != 0 {
        return;
    }
    // SAFETY: `this` is live; verified direct shape `(x, 0, 0, 0)` maps to `(x, 1, 1)`.
    unsafe {
        if let Ok(mut guard) = (*this).inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::DispatchMesh { x, y: 1, z: 1 });
        }
    }
}

extern "win64" fn list_clear_render_target_view(
    this: *mut ComGraphicsCommandList,
    rtv: D3D12CpuDescriptorHandle,
    color_rgba: *const f32,
    _num_rects: u32,
    _p_rects: *const D3D12Rect,
) {
    if this.is_null() || color_rgba.is_null() {
        return;
    }
    // SAFETY: `this` is live and `color_rgba` points to 4 `f32` channels.
    unsafe {
        let list = &*this;
        let rgba = std::slice::from_raw_parts(color_rgba, 4);
        let color = [rgba[0], rgba[1], rgba[2], rgba[3]];
        if let Some(resource_id) = list.backend.resolve_rtv(rtv.ptr)
            && let Ok(mut guard) = list.inner.lock()
            && !guard.closed
        {
            guard
                .commands
                .push(RecordedCommand::ClearRenderTargetView { resource_id, color });
        }
    }
}

// ============================================================================
// ID3D12Fence methods (backed 1:1 by VK_KHR_timeline_semaphore)
// ============================================================================

extern "win64" fn fence_get_completed_value(this: *mut ComFence) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComFence`.
    unsafe {
        let backend = Arc::clone(&(*this).backend);
        let fence_id = (*this).fence_id;
        exec_native(move || backend.fence_completed_value(fence_id).unwrap_or(0))
    }
}

extern "win64" fn fence_set_event_on_completion(
    this: *mut ComFence,
    value: u64,
    _h_event: *mut c_void,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComFence`.
    unsafe {
        let backend = Arc::clone(&(*this).backend);
        let fence_id = (*this).fence_id;
        match exec_native(move || backend.wait_fence_value(fence_id, value)) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn fence_signal(this: *mut ComFence, value: u64) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComFence`.
    unsafe {
        let backend = Arc::clone(&(*this).backend);
        let fence_id = (*this).fence_id;
        match exec_native(move || backend.cpu_signal_fence(fence_id, value)) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

// ============================================================================
// ID3D12Resource methods
// ============================================================================

extern "win64" fn res_map(
    this: *mut ComResource,
    _subresource: u32,
    _p_read_range: *const c_void,
    pp_data: *mut *mut c_void,
) -> i32 {
    if this.is_null() || pp_data.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live and `pp_data` is writable.
    unsafe {
        let backend = Arc::clone(&(*this).backend);
        let res_id = (*this).resource_id;
        match exec_native(move || backend.map_resource(res_id).map(|p| p as usize)) {
            Ok(addr) => {
                *pp_data = addr as *mut c_void;
                S_OK
            }
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn res_unmap(
    this: *mut ComResource,
    _subresource: u32,
    _p_written_range: *const c_void,
) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        let backend = Arc::clone(&(*this).backend);
        let res_id = (*this).resource_id;
        let _ = exec_native(move || backend.unmap_resource(res_id));
    }
}

extern "win64" fn res_get_desc(
    this: *mut ComResource,
    out_desc: *mut D3D12ResourceDesc,
) -> *mut D3D12ResourceDesc {
    if !this.is_null() && !out_desc.is_null() {
        // SAFETY: MSVC COM struct-return ABI writes `*out_desc` and returns `out_desc`.
        unsafe {
            *out_desc = (*this).desc;
        }
    }
    out_desc
}

extern "win64" fn res_get_gpu_virtual_address(this: *mut ComResource) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is live.
    unsafe { (*this).backend.resource_gpu_va((*this).resource_id) }
}

// ============================================================================
// ID3D12DescriptorHeap methods (MSVC COM struct-return ABI: RCX=this, RDX=out, RAX=out)
// ============================================================================

extern "win64" fn heap_get_desc(
    this: *mut ComDescriptorHeap,
    out_desc: *mut D3D12DescriptorHeapDesc,
) -> *mut D3D12DescriptorHeapDesc {
    if !this.is_null() && !out_desc.is_null() {
        // SAFETY: MSVC COM struct-return ABI writes `*out_desc` and returns `out_desc`.
        unsafe {
            *out_desc = (*this).desc;
        }
    }
    out_desc
}

extern "win64" fn heap_get_cpu_descriptor_handle_for_heap_start(
    this: *mut ComDescriptorHeap,
    out_handle: *mut D3D12CpuDescriptorHandle,
) -> *mut D3D12CpuDescriptorHandle {
    if !this.is_null() && !out_handle.is_null() {
        // SAFETY: MSVC COM struct-return ABI writes `*out_handle` and returns `out_handle`.
        unsafe {
            *out_handle = D3D12CpuDescriptorHandle {
                ptr: (*this).base_cpu_ptr,
            };
        }
    }
    out_handle
}

extern "win64" fn heap_get_gpu_descriptor_handle_for_heap_start(
    this: *mut ComDescriptorHeap,
    out_handle: *mut D3D12GpuDescriptorHandle,
) -> *mut D3D12GpuDescriptorHandle {
    if !this.is_null() && !out_handle.is_null() {
        // SAFETY: MSVC COM struct-return ABI writes `*out_handle` and returns `out_handle`.
        unsafe {
            *out_handle = D3D12GpuDescriptorHandle {
                ptr: (*this).base_cpu_ptr as u64,
            };
        }
    }
    out_handle
}
