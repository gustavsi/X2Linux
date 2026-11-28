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

use crate::vulkan::{RecordedCommand, ResourceDescInfo, VulkanBackend};
use crate::{GpuError, VtableProbeCallback, build_probe_vtable_table};
use ash::vk;
use std::ffi::c_void;
use std::mem::size_of;
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

/// Spawn the SUD-free native Linux GPU worker thread. Must be called from ELF `.init_array`
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
    if NATIVE_WORKER.set(mailbox).is_err() {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("x2l-vk-native".into())
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

// Compile-time Win64 ABI size assertions matching Microsoft DirectX-Headers `d3d12.h`.
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
}

static COM_VTABLES: OnceLock<ComVtables> = OnceLock::new();

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
            (8, dev_create_command_queue as *const () as usize),
            (9, dev_create_command_allocator as *const () as usize),
            (10, dev_create_graphics_pipeline_state as *const () as usize),
            (12, dev_create_command_list as *const () as usize),
            (14, dev_create_descriptor_heap as *const () as usize),
            (
                15,
                dev_get_descriptor_handle_increment_size as *const () as usize,
            ),
            (20, dev_create_render_target_view as *const () as usize),
            (27, dev_create_committed_resource as *const () as usize),
            (36, dev_create_fence as *const () as usize),
            // Slot 45 intentionally remains a W^X probe thunk (verified by M1 & M2.4 tests).
        ],
    )?;

    let queue = build_probe_vtable_table(
        64,
        probe_cb,
        &[
            (0, queue_query_interface as *const () as usize),
            (1, queue_add_ref as *const () as usize),
            (2, queue_release as *const () as usize),
            (10, queue_execute_command_lists as *const () as usize),
            (14, queue_signal as *const () as usize),
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
        96,
        probe_cb,
        &[
            (0, list_query_interface as *const () as usize),
            (1, list_add_ref as *const () as usize),
            (2, list_release as *const () as usize),
            (8, list_get_type as *const () as usize),
            (9, list_close as *const () as usize),
            (10, list_reset as *const () as usize),
            (12, list_draw_instanced as *const () as usize),
            (20, list_ia_set_primitive_topology as *const () as usize),
            (21, list_rs_set_viewports as *const () as usize),
            (22, list_rs_set_scissor_rects as *const () as usize),
            (25, list_set_pipeline_state as *const () as usize),
            (26, list_resource_barrier as *const () as usize),
            (50, list_om_set_render_targets as *const () as usize),
            (52, list_clear_render_target_view as *const () as usize),
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

    let _ = COM_VTABLES.set(ComVtables {
        device,
        queue,
        allocator,
        cmd_list,
        fence,
        resource,
        desc_heap,
        pso,
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
    desc: D3D12DescriptorHeapDesc,
    base_cpu_ptr: usize,
}

#[repr(C)]
pub struct ComPipelineState {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    pso_id: u64,
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

// ============================================================================
// ID3D12DeviceX methods
// ============================================================================

extern "win64" fn dev_create_command_queue(
    this: *mut ComDevice,
    desc: *const D3D12CommandQueueDesc,
    _riid: *const c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if this.is_null() || desc.is_null() || out_queue.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComDevice` and `out_queue` is writable.
    unsafe {
        let dev = &*this;
        let q = Box::into_raw(Box::new(ComCommandQueue {
            vtbl: dev.vtables.queue.as_ptr(),
            ref_count: AtomicU32::new(1),
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
        let has_vertex_inputs = d.input_layout.num_elements > 0;
        let backend = Arc::clone(&dev.backend);
        match exec_native(move || {
            backend.create_graphics_pipeline(&vs_vec, &ps_vec, rtv_fmt, has_vertex_inputs)
        }) {
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
            Ok((_id, base_cpu_ptr, _inc)) => {
                let obj = Box::into_raw(Box::new(ComDescriptorHeap {
                    vtbl: dev.vtables.desc_heap.as_ptr(),
                    ref_count: AtomicU32::new(1),
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

extern "win64" fn dev_get_descriptor_handle_increment_size(
    _this: *mut ComDevice,
    _heap_type: u32,
) -> u32 {
    32
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
        let backend = Arc::clone(&(*this).backend);
        let fence_id = (*fence).fence_id;
        match exec_native(move || backend.queue_signal_fence(fence_id, value)) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
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
) {
    if this.is_null() || num_barriers == 0 || p_barriers.is_null() {
        return;
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
}

extern "win64" fn list_om_set_render_targets(
    this: *mut ComGraphicsCommandList,
    num_rtvs: u32,
    p_rtv_descriptors: *const D3D12CpuDescriptorHandle,
    rts_single_handle_to_descriptor_range: i32,
    _p_dsv_descriptor: *const D3D12CpuDescriptorHandle,
) {
    if this.is_null() || num_rtvs == 0 || p_rtv_descriptors.is_null() {
        return;
    }
    // SAFETY: `this` is live; `p_rtv_descriptors` is valid for either 1 or `num_rtvs` handles.
    unsafe {
        let list = &*this;
        let mut rts = Vec::with_capacity(num_rtvs as usize);
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
        if let Ok(mut guard) = list.inner.lock()
            && !guard.closed
        {
            guard.commands.push(RecordedCommand::SetRenderTargets(rts));
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
