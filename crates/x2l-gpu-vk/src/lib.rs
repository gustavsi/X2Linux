//! Direct `D3D12.X` → Vulkan 1.3/1.4 translation backend (`ash`) plus M1 Null COM `ID3D12DeviceX`
//! vtable probe & shader capture support.
//!
//! Targets `VK_KHR_buffer_device_address`, `VK_EXT_descriptor_buffer`, `VK_EXT_mesh_shader`,
//! `VK_EXT_graphics_pipeline_library`, `VK_EXT_external_memory_host`, `VK_KHR_timeline_semaphore`,
//! and Linux `userfaultfd` write-protect dirty-page tracking without `vkd3d-proton`.

pub mod com;
pub mod d3d12x_ext;
pub mod memory;
pub mod vulkan;
pub mod wsi;

use std::ffi::c_void;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use thiserror::Error;
use x2l_common::GpuBackend;

/// Errors raised while initializing or driving the Vulkan 1.3/1.4 or Null M1 backend.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("Vulkan backend error: {0}")]
    Vulkan(String),
    #[error("executable thunk allocation failed: {0}")]
    ThunkAlloc(String),
}

/// Minimum target Vulkan API version (`Vulkan 1.3`).
pub const TARGET_VULKAN_API_VERSION: u32 = ash::vk::API_VERSION_1_3;

/// Required Vulkan device extensions for full M2 `D3D12.X` translation.
pub const REQUIRED_VULKAN_EXTENSIONS: &[&str] = &[
    "VK_KHR_buffer_device_address",
    "VK_EXT_descriptor_buffer",
    "VK_EXT_mesh_shader",
    "VK_EXT_graphics_pipeline_library",
    "VK_KHR_timeline_semaphore",
];

/// Shared vtable extent for native and Null devices, including unassigned extension slots.
pub const DEVICE_VTABLE_LEN: usize = 166;
pub const NULL_DEVICE_VTABLE_SLOTS: usize = DEVICE_VTABLE_LEN;

/// Summary of active GPU backend configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRuntimeState {
    pub backend: GpuBackend,
    pub validation: bool,
}

/// Callback invoked for unmapped COM methods. Use [`decode_d3d12_vtable_probe`] on the first argument.
pub type VtableProbeCallback = extern "win64" fn(slot_index: u32, caller_rip: u64) -> u64;

const QUEUE_PROBE_BASE: u32 = 1 << 16;
const LIST_PROBE_BASE: u32 = 2 << 16;

/// Returns the interface name and vtable index carried by a D3D12.X probe.
pub fn decode_d3d12_vtable_probe(probe_id: u32) -> (&'static str, u32) {
    match probe_id & 0xffff_0000 {
        QUEUE_PROBE_BASE => ("ID3D12CommandQueueX", probe_id & 0xffff),
        LIST_PROBE_BASE => ("ID3D12GraphicsCommandListX", probe_id & 0xffff),
        _ => ("ID3D12DeviceX", probe_id),
    }
}

fn trace_d3d12_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(x2l_common::bridge::ENV_TRACE_HLE)
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

fn log_d3d12_trace(call: std::fmt::Arguments<'_>) {
    let line = format!("[D3D12X] {call}\n");
    // The Wine process does not install the CLI's tracing subscriber. Use the same
    // append-only log file as the core runtime and keep file syscalls on its native worker.
    com::exec_native(move || {
        tracing::info!("{}", line.trim_end());
        static LOG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
        if let Some(path) = LOG_PATH.get_or_init(|| {
            std::env::var_os("X2L_LOG_PATH")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
        }) && let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path)
        {
            let _ = file.write_all(line.as_bytes());
        }
    });
}

/// Win64 COM-layout dummy `ID3D12DeviceX` object returned by Null-mode `D3D12CreateDevice`.
#[repr(C)]
pub struct NullD3D12DeviceX {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
}

// SAFETY: The vtable table pointer points to an immutable process-lifetime array of W^X
// executable probe thunks and `ref_count` is synchronized atomically.
unsafe impl Send for NullD3D12DeviceX {}
unsafe impl Sync for NullD3D12DeviceX {}

impl NullD3D12DeviceX {
    pub fn ref_count(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }
}

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: isize,
    ) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const PROT_EXEC: i32 = 0x4;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

/// Size in bytes of each emitted x86-64 Win64 vtable probe tail-call thunk (padded to 32 B).
const THUNK_STRIDE: usize = 32;

/// Emit a W^X-protected COM vtable of `slot_count` entries where every slot defaults to an
/// x86-64 Win64 probe tail-call thunk invoking `probe_cb(slot_index, caller_rip)` unless
/// overridden in `overrides`.
pub fn build_probe_vtable_table(
    slot_count: usize,
    probe_cb: VtableProbeCallback,
    overrides: &[(usize, usize)],
) -> Result<&'static [usize], GpuError> {
    build_tagged_probe_vtable_table(slot_count, 0, probe_cb, overrides)
}

fn build_tagged_probe_vtable_table(
    slot_count: usize,
    probe_base: u32,
    probe_cb: VtableProbeCallback,
    overrides: &[(usize, usize)],
) -> Result<&'static [usize], GpuError> {
    if slot_count == 0 {
        return Err(GpuError::ThunkAlloc("slot_count must be non-zero".into()));
    }
    u32::try_from(slot_count - 1)
        .ok()
        .and_then(|last_slot| probe_base.checked_add(last_slot))
        .ok_or_else(|| GpuError::ThunkAlloc("vtable probe index overflow".into()))?;
    let page_len = slot_count
        .checked_mul(THUNK_STRIDE)
        .and_then(|bytes| bytes.checked_next_multiple_of(4096))
        .ok_or_else(|| GpuError::ThunkAlloc("vtable mapping length overflow".into()))?;

    // SAFETY: Anonymous private read/write mapping (W^X: never writable and executable at once).
    let raw = unsafe {
        mmap(
            std::ptr::null_mut(),
            page_len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if raw == MAP_FAILED || raw.is_null() {
        return Err(GpuError::ThunkAlloc("mmap RW page failed".into()));
    }

    let mut vtbl = vec![0usize; slot_count].into_boxed_slice();
    let dispatch_addr = probe_cb as usize as u64;

    for (slot, entry) in vtbl.iter_mut().enumerate() {
        let offset = slot * THUNK_STRIDE;
        // SAFETY: `offset + THUNK_STRIDE <= page_len` within the fresh mmap allocation.
        let dst =
            unsafe { std::slice::from_raw_parts_mut((raw as *mut u8).add(offset), THUNK_STRIDE) };
        dst.fill(0xCC);
        encode_win64_vtable_probe_thunk(dst, probe_base + slot as u32, dispatch_addr);
        *entry = (raw as usize) + offset;
    }

    for &(slot, fn_ptr) in overrides {
        if slot < slot_count {
            vtbl[slot] = fn_ptr;
        }
    }

    // SAFETY: Transition page from RW to RX (strict W^X) before exposing function pointers.
    let rc = unsafe { mprotect(raw, page_len, PROT_READ | PROT_EXEC) };
    if rc != 0 {
        // SAFETY: The private mapping is owned here and no thunk address has been published.
        unsafe { munmap(raw, page_len) };
        return Err(GpuError::ThunkAlloc("mprotect RX transition failed".into()));
    }

    Ok(Box::leak(vtbl))
}

/// Emit a W^X-protected table of COM vtable function pointers for `NullD3D12DeviceX`:
/// - Slot 0 (`QueryInterface`), Slot 1 (`AddRef`), Slot 2 (`Release`): real `IUnknown` methods.
/// - Verified queries use typed return values. Verified creations clear the output pointer
///   (when present) and return `E_NOTIMPL`.
/// - Unmapped slots (including slot `45`) use dynamically emitted x86-64 Win64 tail-call thunks
///   that capture the exact caller return address `[rsp]` and invoke `probe_cb(slot_index, caller_rip)`.
pub fn build_null_device_vtable(
    probe_cb: VtableProbeCallback,
) -> Result<&'static [usize; NULL_DEVICE_VTABLE_SLOTS], GpuError> {
    let slice = build_probe_vtable_table(
        NULL_DEVICE_VTABLE_SLOTS,
        probe_cb,
        &[
            (0, null_query_interface as *const () as usize),
            (1, null_add_ref as *const () as usize),
            (2, null_release as *const () as usize),
            (7, null_get_node_count as *const () as usize),
            (8, null_create_out3 as *const () as usize),
            (9, null_create_out3 as *const () as usize),
            (10, null_create_out3 as *const () as usize),
            (11, null_create_out3 as *const () as usize),
            (12, null_create_out6 as *const () as usize),
            (13, null_check_feature_support as *const () as usize),
            (14, null_create_out3 as *const () as usize),
            (
                15,
                null_get_descriptor_handle_increment_size as *const () as usize,
            ),
            (16, null_create_out5 as *const () as usize),
            (25, null_get_resource_allocation_info as *const () as usize),
            (26, null_get_custom_heap_properties as *const () as usize),
            (27, null_create_out7 as *const () as usize),
            (28, null_create_out3 as *const () as usize),
            (29, null_create_out7 as *const () as usize),
            (36, null_create_out4 as *const () as usize),
            (37, null_get_device_removed_reason as *const () as usize),
            (39, null_create_out3 as *const () as usize),
            (41, null_create_out4 as *const () as usize),
            (43, null_get_adapter_luid as *const () as usize),
            (44, null_create_out6 as *const () as usize),
            (48, null_create_out3 as *const () as usize),
            (60, null_create_out5 as *const () as usize),
            (62, null_create_out5 as *const () as usize),
            (73, null_create_out5 as *const () as usize),
            (76, null_create_out7 as *const () as usize),
            (86, null_create_out3 as *const () as usize),
            (
                108,
                com::null_get_resource_allocation_info2 as *const () as usize,
            ),
            (127, null_create_out10 as *const () as usize),
            (133, null_create_out6 as *const () as usize),
        ],
    )?;
    let arr: &'static [usize; NULL_DEVICE_VTABLE_SLOTS] = slice
        .try_into()
        .map_err(|_| GpuError::ThunkAlloc("vtable length mismatch".into()))?;
    Ok(arr)
}

/// Encode a 21-byte Win64 ABI tail-call probe thunk:
/// ```asm
/// mov rdx, [rsp]          ; 48 8B 14 24       (arg1 = caller return address)
/// mov ecx, <slot_index>   ; B9 <u32>          (arg0 = slot_index)
/// mov rax, <dispatch_fn>  ; 48 B8 <u64>       (target callback)
/// jmp rax                 ; FF E0             (tail call; preserves [rsp] & alignment)
/// ```
fn encode_win64_vtable_probe_thunk(buf: &mut [u8], slot_index: u32, dispatch_addr: u64) {
    buf[0..4].copy_from_slice(&[0x48, 0x8B, 0x14, 0x24]);
    buf[4] = 0xB9;
    buf[5..9].copy_from_slice(&slot_index.to_le_bytes());
    buf[9..11].copy_from_slice(&[0x48, 0xB8]);
    buf[11..19].copy_from_slice(&dispatch_addr.to_le_bytes());
    buf[19..21].copy_from_slice(&[0xFF, 0xE0]);
}

/// Allocate a new Null COM `ID3D12DeviceX` instance with refcount = 1.
pub fn create_null_device(
    vtbl: &'static [usize; NULL_DEVICE_VTABLE_SLOTS],
) -> *mut NullD3D12DeviceX {
    let device = Box::into_raw(Box::new(NullD3D12DeviceX {
        vtbl: vtbl.as_ptr(),
        ref_count: AtomicU32::new(1),
    }));
    com::register_device_object(device as usize, false);
    device
}

impl Drop for NullD3D12DeviceX {
    fn drop(&mut self) {
        com::unregister_device_object(std::ptr::from_mut(self) as usize);
    }
}

extern "win64" fn null_query_interface(
    this: *mut NullD3D12DeviceX,
    _riid: *const c_void,
    out_obj: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_obj.is_null() {
        return -2147467261; // E_POINTER (0x80004003)
    }
    // SAFETY: Caller supplied valid COM `this` and writable `out_obj` pointer.
    unsafe {
        (*this).ref_count.fetch_add(1, Ordering::AcqRel);
        *out_obj = this as *mut c_void;
    }
    0 // S_OK
}

extern "win64" fn null_add_ref(this: *mut NullD3D12DeviceX) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `NullD3D12DeviceX` allocated by `create_null_device`.
    unsafe { (*this).ref_count.fetch_add(1, Ordering::AcqRel) + 1 }
}

extern "win64" fn null_release(this: *mut NullD3D12DeviceX) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` was allocated via `Box::into_raw` in `create_null_device`; dropping on
    // zero refcount matches COM ownership rules.
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

const E_NOTIMPL_HR: i32 = -2147467263; // 0x80004001

extern "win64" fn null_get_node_count(_this: *mut NullD3D12DeviceX) -> u32 {
    1
}

extern "win64" fn null_get_descriptor_handle_increment_size(
    _this: *mut NullD3D12DeviceX,
    heap_type: u32,
) -> u32 {
    if heap_type == 1 { 16 } else { 32 }
}

extern "win64" fn null_get_device_removed_reason(_this: *mut NullD3D12DeviceX) -> i32 {
    0 // S_OK
}

extern "win64" fn null_check_feature_support(
    _this: *mut NullD3D12DeviceX,
    feature: u32,
    data: *mut c_void,
    data_size: u32,
) -> i32 {
    crate::com::fill_check_feature_support(feature, data, data_size, false)
}

extern "win64" fn null_get_resource_allocation_info(
    a0: *mut c_void,
    a1: *mut c_void,
    _visible_mask: u32,
    num_descs: u32,
    descs: *const crate::com::D3D12ResourceDesc,
) -> *mut crate::com::D3D12ResourceAllocationInfo {
    crate::com::compute_resource_allocation_info_dual_abi(a0, a1, num_descs, descs)
}

extern "win64" fn null_get_custom_heap_properties(
    a0: *mut c_void,
    a1: *mut c_void,
    node_mask: u32,
    heap_type: u32,
) -> *mut crate::com::D3D12HeapProperties {
    crate::com::compute_custom_heap_properties_dual_abi(a0, a1, node_mask, heap_type)
}

extern "win64" fn null_get_adapter_luid(
    a0: *mut c_void,
    a1: *mut c_void,
) -> *mut crate::com::D3D12Luid {
    crate::com::compute_adapter_luid_dual_abi(a0, a1)
}

extern "win64" fn null_create_out3(
    _this: *mut NullD3D12DeviceX,
    _a1: usize,
    _a2: usize,
    out_obj: *mut *mut c_void,
) -> i32 {
    if !out_obj.is_null() {
        // SAFETY: Clear output pointer before returning `E_NOTIMPL` on Null device.
        unsafe { *out_obj = std::ptr::null_mut() };
    }
    E_NOTIMPL_HR
}

extern "win64" fn null_create_out4(
    _this: *mut NullD3D12DeviceX,
    _a1: usize,
    _a2: usize,
    _a3: usize,
    out_obj: *mut *mut c_void,
) -> i32 {
    if !out_obj.is_null() {
        // SAFETY: Clear output pointer before returning `E_NOTIMPL` on Null device.
        unsafe { *out_obj = std::ptr::null_mut() };
    }
    E_NOTIMPL_HR
}

extern "win64" fn null_create_out5(
    _this: *mut NullD3D12DeviceX,
    _a1: usize,
    _a2: usize,
    _a3: usize,
    _a4: usize,
    out_obj: *mut *mut c_void,
) -> i32 {
    if !out_obj.is_null() {
        // SAFETY: Clear output pointer before returning `E_NOTIMPL` on Null device.
        unsafe { *out_obj = std::ptr::null_mut() };
    }
    E_NOTIMPL_HR
}

#[allow(clippy::too_many_arguments)]
extern "win64" fn null_create_out6(
    _this: *mut NullD3D12DeviceX,
    _a1: usize,
    _a2: usize,
    _a3: usize,
    _a4: usize,
    _a5: usize,
    out_obj: *mut *mut c_void,
) -> i32 {
    if !out_obj.is_null() {
        // SAFETY: Clear output pointer before returning `E_NOTIMPL` on Null device.
        unsafe { *out_obj = std::ptr::null_mut() };
    }
    E_NOTIMPL_HR
}

#[allow(clippy::too_many_arguments)]
extern "win64" fn null_create_out7(
    _this: *mut NullD3D12DeviceX,
    _a1: usize,
    _a2: usize,
    _a3: usize,
    _a4: usize,
    _a5: usize,
    _a6: usize,
    out_obj: *mut *mut c_void,
) -> i32 {
    if !out_obj.is_null() {
        // SAFETY: Clear output pointer before returning `E_NOTIMPL` on Null device.
        unsafe { *out_obj = std::ptr::null_mut() };
    }
    E_NOTIMPL_HR
}

#[allow(clippy::too_many_arguments)]
extern "win64" fn null_create_out10(
    _this: *mut NullD3D12DeviceX,
    _a1: usize,
    _a2: u32,
    _a3: usize,
    _a4: u32,
    _a5: usize,
    _a6: usize,
    _a7: u32,
    _a8: usize,
    _a9: usize,
    out_obj: *mut *mut c_void,
) -> i32 {
    if !out_obj.is_null() {
        // SAFETY: A non-null COM output is writable.
        unsafe { out_obj.write(std::ptr::null_mut()) };
    }
    E_NOTIMPL_HR
}

static SHADER_CAPTURE_SEQ: AtomicUsize = AtomicUsize::new(0);

/// Persist a raw console shader blob into `shaders_raw/` for offline M2 compiler analysis.
pub fn capture_raw_shader_blob(shaders_raw_dir: &Path, blob: &[u8]) -> std::io::Result<PathBuf> {
    fs::create_dir_all(shaders_raw_dir)?;
    let seq = SHADER_CAPTURE_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = shaders_raw_dir.join(format!("shader_{seq:06}.blob"));
    fs::write(&path, blob)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static LAST_SLOT: AtomicU32 = AtomicU32::new(0);
    static LAST_RIP: AtomicU64 = AtomicU64::new(0);

    extern "win64" fn test_probe(slot: u32, rip: u64) -> u64 {
        LAST_SLOT.store(slot, Ordering::SeqCst);
        LAST_RIP.store(rip, Ordering::SeqCst);
        0
    }

    #[test]
    fn null_device_vtable_probe_executes_and_releases() {
        let vtbl = build_null_device_vtable(test_probe).expect("build vtable");
        let dev = create_null_device(vtbl);
        assert!(!dev.is_null());

        // SAFETY: `dev` is a valid NullD3D12DeviceX with 128 vtable slots.
        unsafe {
            let vtable_ptr = (*dev).vtbl;
            let add_ref: extern "win64" fn(*mut NullD3D12DeviceX) -> u32 =
                std::mem::transmute(*vtable_ptr.add(1));
            let release: extern "win64" fn(*mut NullD3D12DeviceX) -> u32 =
                std::mem::transmute(*vtable_ptr.add(2));
            let probe_45: extern "win64" fn(*mut NullD3D12DeviceX) -> u64 =
                std::mem::transmute(*vtable_ptr.add(45));

            assert_eq!(add_ref(dev), 2);
            assert_eq!(probe_45(dev), 0);
            assert_eq!(LAST_SLOT.load(Ordering::SeqCst), 45);
            assert_ne!(LAST_RIP.load(Ordering::SeqCst), 0);
            assert_eq!(release(dev), 1);
            assert_eq!(release(dev), 0);
        }
    }
}
