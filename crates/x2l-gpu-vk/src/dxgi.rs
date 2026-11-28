// SPDX-License-Identifier: GPL-3.0-or-later

//! Win64 DXGI interfaces based on the public DXGI interface and structure definitions.

use crate::com::{E_FAIL, E_INVALIDARG, E_NOINTERFACE, E_POINTER, S_OK, exec_native};
use crate::com_guid::{format_guid, guid, notify_rejected};
use crate::vulkan::VulkanBackend;
use crate::{
    DXGI_ADAPTER_PROBE_BASE, DXGI_DEVICE_PROBE_BASE, DXGI_FACTORY_PROBE_BASE,
    DXGI_OUTPUT_PROBE_BASE, GpuError, VtableProbeCallback, build_tagged_probe_vtable_table,
    decode_d3d12_vtable_probe, log_com_trace, trace_d3d12_enabled,
};
use std::ffi::c_void;
use std::mem::{offset_of, size_of, transmute};
use std::ptr;
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

type Iid = [u8; 16];

const UNKNOWN: Iid = guid(0, 0, 0, 0xc000_0000_0000_0046);
const OBJECT: Iid = guid(0xaec22fb8, 0x76f3, 0x4639, 0x9be0_28eb_43a6_7a2e);
const DEVICES: [Iid; 5] = [
    guid(0x54ec77fa, 0x1377, 0x44e6, 0x8c32_88fd_5f44_c84c),
    guid(0x77db970f, 0x6276, 0x48ba, 0xba28_0701_43b4_392c),
    guid(0x05008617, 0xfbfd, 0x4051, 0xa790_1448_84b4_f6a9),
    guid(0x6007896c, 0x3244, 0x4afd, 0xbf18_a6d3_beda_5023),
    guid(0x95b4f95f, 0xd8da, 0x4ca4, 0x9ee6_3b76_d596_8a10),
];
const ADAPTERS: [Iid; 5] = [
    guid(0x2411e7e1, 0x12ac, 0x4ccf, 0xbd14_9798_e853_4dc0),
    guid(0x29038f61, 0x3839, 0x4626, 0x91fd_0868_7901_1a05),
    guid(0x0aa1ae0a, 0xfa0e, 0x4b84, 0x8644_e05f_f8e5_acb5),
    guid(0x645967a4, 0x1392, 0x4310, 0xa798_8053_ce3e_93fd),
    guid(0x3c8d99d1, 0x4fbf, 0x4181, 0xa82c_af66_bf7b_d24e),
];
const OUTPUTS: [Iid; 1] = [guid(0xae02eedb, 0xc735, 0x4690, 0x8d52_5a8d_c202_13aa)];
const FACTORIES: [Iid; 7] = [
    guid(0x7b7166ec, 0x21c7, 0x44ae, 0xb21a_c9ae_321a_e369),
    guid(0x770aae78, 0xf26f, 0x4dba, 0xa829_253c_83d1_b387),
    guid(0x50c83a1c, 0xe072, 0x4c48, 0x87b0_3630_fa36_a6d0),
    guid(0x25483823, 0xcd46, 0x4c7d, 0x86ca_47aa_95b8_37bd),
    guid(0x1bc6ea02, 0xef36, 0x464f, 0xbf0c_21ca_39e5_168a),
    guid(0x7632e1f5, 0xee65, 0x4dca, 0x87fd_84cd_75f8_838d),
    guid(0xc1b6694f, 0xff09, 0x44a9, 0xb03c_7790_0a0a_1d17),
];

const NOT_FOUND: i32 = 0x887a_0002_u32 as i32;
const MORE_DATA: i32 = 0x887a_0003_u32 as i32;
const UNSUPPORTED: i32 = 0x887a_0004_u32 as i32;
pub(crate) const ADAPTER_LUID_LOW: u32 = 0x5832_4c01;

#[repr(C)]
#[derive(Clone, Copy)]
struct AdapterDesc {
    description: [u16; 128],
    vendor_id: u32,
    device_id: u32,
    subsystem_id: u32,
    revision: u32,
    dedicated_video_memory: u64,
    dedicated_system_memory: u64,
    shared_system_memory: u64,
    luid_low: u32,
    luid_high: i32,
}

#[repr(C)]
struct AdapterDesc1 {
    base: AdapterDesc,
    flags: u32,
}

#[repr(C)]
struct AdapterDesc2 {
    base: AdapterDesc,
    flags: u32,
    graphics_preemption_granularity: u32,
    compute_preemption_granularity: u32,
}

// DESC3 changes the flag enum, whose x64 representation is still 32 bits.
type AdapterDesc3 = AdapterDesc2;

#[repr(C)]
struct OutputDesc {
    device_name: [u16; 32],
    desktop_coordinates: [i32; 4],
    attached_to_desktop: i32,
    rotation: u32,
    monitor: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ModeDesc {
    width: u32,
    height: u32,
    refresh_numerator: u32,
    refresh_denominator: u32,
    format: u32,
    scanline_ordering: u32,
    scaling: u32,
}

#[repr(C)]
pub(crate) struct VideoMemoryInfo {
    pub(crate) budget: u64,
    pub(crate) current_usage: u64,
    available_for_reservation: u64,
    current_reservation: u64,
}

impl VideoMemoryInfo {
    pub(crate) fn new(budget: u64, current_usage: u64) -> Self {
        Self {
            budget,
            current_usage,
            available_for_reservation: 0,
            current_reservation: 0,
        }
    }
}

const _: () = {
    assert!(size_of::<AdapterDesc>() == 304);
    assert!(size_of::<AdapterDesc1>() == 312);
    assert!(size_of::<AdapterDesc2>() == 320);
    assert!(size_of::<AdapterDesc3>() == 320);
    assert!(size_of::<OutputDesc>() == 96);
    assert!(size_of::<ModeDesc>() == 28);
    assert!(size_of::<VideoMemoryInfo>() == 32);
    assert!(offset_of!(AdapterDesc, dedicated_video_memory) == 272);
    assert!(offset_of!(AdapterDesc, shared_system_memory) == 288);
    assert!(offset_of!(AdapterDesc, luid_low) == 296);
    assert!(offset_of!(AdapterDesc1, flags) == 304);
    assert!(offset_of!(AdapterDesc2, graphics_preemption_granularity) == 308);
    assert!(offset_of!(OutputDesc, desktop_coordinates) == 64);
    assert!(offset_of!(OutputDesc, monitor) == 88);
};

fn wide<const N: usize>(s: &str) -> [u16; N] {
    let mut result = [0; N];
    let mut end = 0;
    for ch in s.chars() {
        let len = ch.len_utf16();
        if end + len >= N {
            break;
        }
        ch.encode_utf16(&mut result[end..end + len]);
        end += len;
    }
    result
}

pub(crate) struct AdapterInfo {
    desc: AdapterDesc,
    flags: u32,
    backend: Option<Arc<VulkanBackend>>,
}

impl Default for AdapterInfo {
    fn default() -> Self {
        Self {
            desc: AdapterDesc {
                description: wide("X2Linux Null Adapter"),
                vendor_id: 0x1002,
                device_id: 0x73bf,
                subsystem_id: 0,
                revision: 0,
                dedicated_video_memory: 10 << 30,
                dedicated_system_memory: 0,
                shared_system_memory: 6 << 30,
                luid_low: ADAPTER_LUID_LOW,
                luid_high: 0,
            },
            flags: 2, // DXGI_ADAPTER_FLAG_SOFTWARE
            backend: None,
        }
    }
}

impl AdapterInfo {
    pub(crate) fn vulkan(backend: Arc<VulkanBackend>) -> Self {
        let (vendor, device, software) = backend.dxgi_adapter_ids();
        let (local, non_local) = backend.dxgi_memory_sizes();
        Self {
            desc: AdapterDesc {
                description: wide(backend.device_name()),
                vendor_id: vendor,
                device_id: device,
                subsystem_id: 0,
                revision: 0,
                dedicated_video_memory: local,
                dedicated_system_memory: 0,
                shared_system_memory: non_local,
                luid_low: ADAPTER_LUID_LOW,
                luid_high: 0,
            },
            flags: if software { 2 } else { 0 },
            backend: Some(backend),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Device,
    Adapter,
    Output,
    Factory,
}

impl Kind {
    fn iids(self) -> &'static [Iid] {
        match self {
            Self::Device => &DEVICES,
            Self::Adapter => &ADAPTERS,
            Self::Output => &OUTPUTS,
            Self::Factory => &FACTORIES,
        }
    }

    fn probe_base(self) -> u32 {
        match self {
            Self::Device => DXGI_DEVICE_PROBE_BASE,
            Self::Adapter => DXGI_ADAPTER_PROBE_BASE,
            Self::Output => DXGI_OUTPUT_PROBE_BASE,
            Self::Factory => DXGI_FACTORY_PROBE_BASE,
        }
    }
}

#[repr(C)]
struct Object {
    vtable: usize,
    kind: Kind,
    references: AtomicU32,
    interfaces: usize,
}

// Each outstanding reference to an endpoint owns a reference to the D3D12 device.
// The D3D12 device owns this allocation and supplies the controlling IUnknown.
#[repr(C)]
pub(crate) struct Interfaces {
    objects: [Object; 4],
    owner: usize,
    probe: Option<VtableProbeCallback>,
    adapter: AdapterInfo,
    gpu_priority: AtomicI32,
    frame_latency: AtomicU32,
}

type Query = extern "win64" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i32;
type Ref = extern "win64" fn(*mut c_void) -> u32;

impl Object {
    fn interfaces(&self) -> &Interfaces {
        // SAFETY: This points to the owning allocation, kept live by the caller's COM reference.
        unsafe { &*(self.interfaces as *const Interfaces) }
    }

    fn give(&self, out: &mut *mut c_void) {
        add_ref(ptr::from_ref(self).cast_mut());
        *out = ptr::from_ref(self).cast_mut().cast();
    }
}

impl Interfaces {
    fn owner_query(&self, iid: &Iid, out: &mut *mut c_void) -> i32 {
        let owner = self.owner as *mut c_void;
        // SAFETY: Owner is a live D3D12 COM device, whose slot 0 is IUnknown::QueryInterface.
        let query: Query = unsafe { transmute(**owner.cast::<*const usize>()) };
        query(owner, iid.as_ptr().cast(), out)
    }

    fn owner_ref(&self, slot: usize) -> (Ref, *mut c_void) {
        let owner = self.owner as *mut c_void;
        // SAFETY: The only callers select IUnknown slots 1 (AddRef) and 2 (Release).
        let method: Ref = unsafe { transmute((*owner.cast::<*const usize>()).add(slot).read()) };
        (method, owner)
    }
}

extern "win64" fn add_ref(this: *mut Object) -> u32 {
    // SAFETY: A non-null this refers to a live endpoint held by the caller.
    let Some(object) = (unsafe { this.as_ref() }) else {
        return 0;
    };
    let (add, owner) = object.interfaces().owner_ref(1);
    add(owner);
    object.references.fetch_add(1, Ordering::AcqRel) + 1
}

extern "win64" fn release(this: *mut Object) -> u32 {
    // SAFETY: A non-null this refers to a live endpoint held by the caller.
    let Some(object) = (unsafe { this.as_ref() }) else {
        return 0;
    };
    let (release, owner) = object.interfaces().owner_ref(2);
    let remaining = object.references.fetch_sub(1, Ordering::AcqRel) - 1;
    release(owner);
    remaining
}

// # Safety
// The IID and output must be valid COM pointers when non-null.
pub(crate) unsafe fn query_device(
    owner: *mut c_void,
    iid: *const c_void,
    out: *mut *mut c_void,
    cached: &OnceLock<Box<Interfaces>>,
    probe: Option<VtableProbeCallback>,
    backend: Option<&Arc<VulkanBackend>>,
) -> Option<i32> {
    // SAFETY: The caller supplies a readable IID when non-null.
    let iid = unsafe { iid.cast::<Iid>().as_ref() }?;
    if iid != &OBJECT && !DEVICES.contains(iid) {
        return None;
    }
    // SAFETY: The caller supplies a writable output when non-null.
    if out.is_null() {
        return Some(E_POINTER);
    }
    // SAFETY: Initialize the output before borrowing it.
    unsafe { out.write(ptr::null_mut()) };
    let Ok(vtables) = exec_native(|| VTABLES.get_or_init(build_vtables)) else {
        return Some(E_FAIL);
    };
    let owner = owner as usize;
    let interfaces = exec_native(|| {
        cached.get_or_init(|| {
            let mut interfaces = Box::new(Interfaces {
                objects: vtables.map(|(kind, vtable)| Object {
                    vtable: vtable.as_ptr() as usize,
                    kind,
                    references: AtomicU32::new(0),
                    interfaces: 0,
                }),
                owner,
                probe,
                adapter: backend.map_or_else(AdapterInfo::default, |backend| {
                    AdapterInfo::vulkan(Arc::clone(backend))
                }),
                gpu_priority: AtomicI32::new(0),
                frame_latency: AtomicU32::new(3),
            });
            let address = ptr::from_mut(&mut *interfaces) as usize;
            for object in &mut interfaces.objects {
                object.interfaces = address;
            }
            interfaces
        })
    });
    if trace_d3d12_enabled() {
        log_com_trace(
            "DXGI",
            format_args!("ID3D12DeviceX::QueryInterface({})", format_guid(iid)),
        );
    }
    // SAFETY: The output is writable and initialized above.
    interfaces.objects[Kind::Device as usize].give(unsafe { &mut *out });
    Some(S_OK)
}

extern "win64" fn query_interface(
    this: *mut Object,
    iid: *const Iid,
    out: *mut *mut c_void,
) -> i32 {
    // SAFETY: COM supplies a live this, readable IID and writable output when non-null.
    let Some((object, iid, out)) = (unsafe { com_args(this, iid, out) }) else {
        return E_POINTER;
    };
    if object.kind == Kind::Device && iid != &OBJECT && !DEVICES.contains(iid) {
        return object.interfaces().owner_query(iid, out);
    }
    query_object(object, iid, out)
}

fn query_object(object: &Object, iid: &Iid, out: &mut *mut c_void) -> i32 {
    if iid == &UNKNOWN || iid == &OBJECT || object.kind.iids().contains(iid) {
        object.give(out);
        S_OK
    } else {
        reject(object, iid)
    }
}

fn reject(object: &Object, iid: &Iid) -> i32 {
    let (interface, _) = decode_d3d12_vtable_probe(object.kind.probe_base());
    notify_rejected(interface, &format_guid(iid));
    E_NOINTERFACE
}

unsafe fn com_args<'a, T>(
    this: *mut Object,
    input: *const T,
    out: *mut *mut c_void,
) -> Option<(&'a Object, &'a T, &'a mut *mut c_void)> {
    // SAFETY: The caller supplies writable output and readable object and input when non-null.
    unsafe {
        if out.is_null() {
            return None;
        }
        out.write(ptr::null_mut());
        Some((this.as_ref()?, input.as_ref()?, &mut *out))
    }
}

extern "win64" fn get_parent(this: *mut Object, iid: *const Iid, out: *mut *mut c_void) -> i32 {
    // SAFETY: COM supplies a live this, readable IID and writable output when non-null.
    let Some((object, iid, out)) = (unsafe { com_args(this, iid, out) }) else {
        return E_POINTER;
    };
    let kind = match object.kind {
        Kind::Device | Kind::Output => Kind::Adapter,
        Kind::Adapter => Kind::Factory,
        Kind::Factory => return reject(object, iid),
    };
    query_object(&object.interfaces().objects[kind as usize], iid, out)
}

unsafe fn related(this: *mut Object, index: u32, out: *mut *mut c_void, kind: Kind) -> i32 {
    // SAFETY: The caller supplies writable output and a live this when non-null.
    let Some((object, _, out)) = (unsafe { com_args(this, &index, out) }) else {
        return E_POINTER;
    };
    if index != 0 {
        return NOT_FOUND;
    }
    object.interfaces().objects[kind as usize].give(out);
    S_OK
}

extern "win64" fn get_adapter(this: *mut Object, out: *mut *mut c_void) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { related(this, 0, out, Kind::Adapter) }
}

extern "win64" fn enum_adapters(this: *mut Object, index: u32, out: *mut *mut c_void) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { related(this, index, out, Kind::Adapter) }
}

extern "win64" fn enum_outputs(this: *mut Object, index: u32, out: *mut *mut c_void) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { related(this, index, out, Kind::Output) }
}

extern "win64" fn set_gpu_priority(this: *mut Object, value: i32) -> i32 {
    // SAFETY: COM supplies a live this when non-null.
    let Some(object) = (unsafe { this.as_ref() }) else {
        return E_POINTER;
    };
    if !(-7..=7).contains(&value) {
        return E_INVALIDARG;
    }
    object
        .interfaces()
        .gpu_priority
        .store(value, Ordering::Relaxed);
    S_OK
}

extern "win64" fn set_frame_latency(this: *mut Object, value: u32) -> i32 {
    // SAFETY: COM supplies a live this when non-null.
    let Some(object) = (unsafe { this.as_ref() }) else {
        return E_POINTER;
    };
    if value > 16 {
        return E_INVALIDARG;
    }
    object
        .interfaces()
        .frame_latency
        .store(if value == 0 { 3 } else { value }, Ordering::Relaxed);
    S_OK
}

// Writes to potentially uninitialized caller storage without first forming a Rust reference to it.
unsafe fn value<T>(this: *mut Object, out: *mut T, get: impl FnOnce(&Interfaces) -> T) -> i32 {
    // SAFETY: The caller supplies a live this when non-null.
    let Some(object) = (unsafe { this.as_ref() }) else {
        return E_POINTER;
    };
    if out.is_null() {
        return E_POINTER;
    }
    // SAFETY: The caller supplies writable output storage for T.
    unsafe { out.write(get(object.interfaces())) };
    S_OK
}

extern "win64" fn get_gpu_priority(this: *mut Object, out: *mut i32) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { value(this, out, |i| i.gpu_priority.load(Ordering::Relaxed)) }
}

extern "win64" fn get_frame_latency(this: *mut Object, out: *mut u32) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { value(this, out, |i| i.frame_latency.load(Ordering::Relaxed)) }
}

extern "win64" fn get_adapter_desc(this: *mut Object, out: *mut AdapterDesc) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { value(this, out, |i| i.adapter.desc) }
}

extern "win64" fn get_adapter_desc1(this: *mut Object, out: *mut AdapterDesc1) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe {
        value(this, out, |i| AdapterDesc1 {
            base: i.adapter.desc,
            flags: i.adapter.flags,
        })
    }
}

extern "win64" fn get_adapter_desc2(this: *mut Object, out: *mut AdapterDesc2) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe {
        value(this, out, |i| AdapterDesc2 {
            base: i.adapter.desc,
            flags: i.adapter.flags,
            graphics_preemption_granularity: 0,
            compute_preemption_granularity: 0,
        })
    }
}

extern "win64" fn check_interface(this: *mut Object, iid: *const Iid, version: *mut i64) -> i32 {
    // SAFETY: COM supplies readable this and IID when non-null.
    let Some((object, iid)) = (unsafe { this.as_ref().zip(iid.as_ref()) }) else {
        return E_POINTER;
    };
    if !version.is_null() {
        // SAFETY: COM supplies optional writable driver-version storage.
        unsafe { version.write(0) };
    }
    if crate::com_guid::DEVICE.supports(iid) || object.kind.iids().contains(iid) {
        S_OK
    } else {
        UNSUPPORTED
    }
}

extern "win64" fn query_video_memory(
    this: *mut Object,
    node: u32,
    segment: u32,
    out: *mut VideoMemoryInfo,
) -> i32 {
    if node != 0 || segment > 1 {
        return E_INVALIDARG;
    }
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe {
        value(this, out, |i| {
            let local = segment == 0;
            if let Some(backend) = &i.adapter.backend {
                let backend = Arc::clone(backend);
                return exec_native(move || backend.dxgi_video_memory_info(local));
            }
            let desc = i.adapter.desc;
            VideoMemoryInfo::new(
                if local {
                    desc.dedicated_video_memory
                } else {
                    desc.shared_system_memory
                },
                0,
            )
        })
    }
}

extern "win64" fn output_desc(this: *mut Object, out: *mut OutputDesc) -> i32 {
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe {
        value(this, out, |_| OutputDesc {
            device_name: wide("X2Linux Display"),
            desktop_coordinates: [0, 0, 3840, 2160],
            attached_to_desktop: 1,
            rotation: 1, // DXGI_MODE_ROTATION_IDENTITY
            monitor: 0,
        })
    }
}

fn display_modes(format: u32) -> impl Iterator<Item = ModeDesc> {
    [(1920, 1080), (2560, 1440), (3840, 2160)]
        .into_iter()
        .flat_map(move |(width, height)| {
            [60, 120].map(|hz| ModeDesc {
                width,
                height,
                refresh_numerator: hz,
                refresh_denominator: 1,
                format,
                scanline_ordering: 1, // DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE
                scaling: 0,
            })
        })
        .filter(move |_| matches!(format, 24 | 28 | 87))
}

extern "win64" fn get_display_modes(
    this: *mut Object,
    format: u32,
    _flags: u32,
    count: *mut u32,
    modes: *mut ModeDesc,
) -> i32 {
    if this.is_null() || count.is_null() {
        return E_POINTER;
    }
    let required = display_modes(format).count() as u32;
    // SAFETY: count is writable. A fill call initializes it with the caller's array length.
    let capacity = unsafe {
        let capacity = if modes.is_null() { 0 } else { count.read() };
        count.write(required);
        capacity
    };
    if modes.is_null() {
        return S_OK;
    }
    if capacity < required {
        return MORE_DATA;
    }
    for (idx, mode) in display_modes(format).enumerate() {
        // SAFETY: COM supplies backing storage for capacity entries and idx < required <= capacity.
        unsafe { modes.add(idx).write(mode) };
    }
    S_OK
}

extern "win64" fn find_closest_mode(
    this: *mut Object,
    requested: *const ModeDesc,
    out: *mut ModeDesc,
    device: *mut c_void,
) -> i32 {
    // SAFETY: COM supplies a readable mode description when non-null.
    let Some(request) = (unsafe { requested.as_ref() }) else {
        return E_POINTER;
    };
    if (request.width == 0) != (request.height == 0)
        || (request.refresh_numerator != 0 && request.refresh_denominator == 0)
        || (request.format == 0 && device.is_null())
    {
        return E_INVALIDARG;
    }
    let format = if request.format == 0 {
        28
    } else {
        request.format
    };
    let best = display_modes(format).min_by_key(|mode| {
        let distance = if request.width == 0 {
            0
        } else {
            u64::from(request.width.abs_diff(mode.width))
                + u64::from(request.height.abs_diff(mode.height))
        };
        let refresh = if request.refresh_numerator == 0 {
            0
        } else {
            (u64::from(mode.refresh_numerator) * u64::from(request.refresh_denominator))
                .abs_diff(u64::from(request.refresh_numerator))
        };
        (distance, refresh)
    });
    let Some(best) = best else {
        return NOT_FOUND;
    };
    // SAFETY: COM supplies a live this and writable output when non-null.
    unsafe { value(this, out, |_| best) }
}

extern "win64" fn wait_for_vblank(this: *mut Object) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    exec_native(|| std::thread::sleep(Duration::from_micros(16_667)));
    S_OK
}

extern "win64" fn is_current(this: *mut Object) -> i32 {
    i32::from(!this.is_null())
}

extern "win64" fn no_stereo(_this: *mut Object) -> i32 {
    0
}

extern "win64" fn creation_flags(_this: *mut Object) -> u32 {
    0
}

extern "win64" fn enum_adapter_by_luid(
    this: *mut Object,
    luid: u64,
    iid: *const Iid,
    out: *mut *mut c_void,
) -> i32 {
    // SAFETY: COM supplies a live this, readable IID and writable output when non-null.
    let Some((object, iid, out)) = (unsafe { com_args(this, iid, out) }) else {
        return E_POINTER;
    };
    if luid != u64::from(ADAPTER_LUID_LOW) {
        return NOT_FOUND;
    }
    query_object(
        &object.interfaces().objects[Kind::Adapter as usize],
        iid,
        out,
    )
}

extern "win64" fn enum_adapter_by_preference(
    this: *mut Object,
    index: u32,
    preference: u32,
    iid: *const Iid,
    out: *mut *mut c_void,
) -> i32 {
    // SAFETY: COM supplies a live this, readable IID and writable output when non-null.
    let Some((object, iid, out)) = (unsafe { com_args(this, iid, out) }) else {
        return E_POINTER;
    };
    if preference > 2 {
        return E_INVALIDARG;
    }
    if index != 0 {
        return NOT_FOUND;
    }
    query_object(
        &object.interfaces().objects[Kind::Adapter as usize],
        iid,
        out,
    )
}

type Vtables = [(Kind, &'static [usize]); 4];
static VTABLES: OnceLock<Result<Vtables, GpuError>> = OnceLock::new();

fn build_vtables() -> Result<Vtables, GpuError> {
    macro_rules! table {
        ($kind:ident, $length:literal, $($slot:literal => $method:ident),* $(,)?) => {{
            let kind = Kind::$kind;
            (
                kind,
                build_tagged_probe_vtable_table(
                    $length,
                    kind.probe_base(),
                    unused_probe,
                    &[
                        (0, query_interface as *const () as usize),
                        (1, add_ref as *const () as usize),
                        (2, release as *const () as usize),
                        (6, get_parent as *const () as usize),
                        $(($slot, $method as *const () as usize)),*
                    ],
                )?,
            )
        }};
    }
    Ok([
        table!(Device, 20,
            7 => get_adapter,
            10 => set_gpu_priority,
            11 => get_gpu_priority,
            12 => set_frame_latency,
            13 => get_frame_latency,
        ),
        table!(Adapter, 19,
            7 => enum_outputs,
            8 => get_adapter_desc,
            9 => check_interface,
            10 => get_adapter_desc1,
            11 => get_adapter_desc2,
            14 => query_video_memory,
            18 => get_adapter_desc2,
        ),
        table!(Output, 19,
            7 => output_desc,
            8 => get_display_modes,
            9 => find_closest_mode,
            10 => wait_for_vblank,
        ),
        table!(Factory, 30,
            7 => enum_adapters,
            12 => enum_adapters,
            13 => is_current,
            14 => no_stereo,
            25 => creation_flags,
            26 => enum_adapter_by_luid,
            29 => enum_adapter_by_preference,
        ),
    ])
}

// DXGI thunks dispatch through the instance's probe callback, which can differ by backend.
extern "win64" fn unused_probe(_slot: u32, _caller: u64) -> u64 {
    0
}

pub(crate) fn encode_probe(code: &mut [u8], slot: u32) {
    let mut start = 0;
    if let Some(arg) = output_arg(slot) {
        // Load the object output pointer into RAX, then clear it when non-null.
        let load = match arg {
            2 => [0x4c, 0x89, 0xc0, 0x90, 0x90], // mov rax, r8; nop; nop
            3 => [0x4c, 0x89, 0xc8, 0x90, 0x90], // mov rax, r9; nop; nop
            _ => [0x48, 0x8b, 0x44, 0x24, (arg + 1) * 8], // mov rax, [rsp + (arg+1)*8]
        };
        code[..5].copy_from_slice(&load);
        code[5..17].copy_from_slice(&[
            0x48, 0x85, 0xc0, 0x74, 0x07, // test rax, rax; jz +7
            0x48, 0xc7, 0x00, 0, 0, 0, 0, // mov qword [rax], 0
        ]);
        start = 17;
    }
    // RCX keeps this. RDX receives the guest return address and R8D the probe slot.
    let code = &mut code[start..];
    code[..6].copy_from_slice(&[0x48, 0x8b, 0x14, 0x24, 0x41, 0xb8]);
    code[6..10].copy_from_slice(&slot.to_le_bytes());
    code[10..12].copy_from_slice(&[0x48, 0xb8]);
    code[12..20].copy_from_slice(&(unsupported as *const () as u64).to_le_bytes());
    code[20..22].copy_from_slice(&[0xff, 0xe0]);
}

fn output_arg(slot: u32) -> Option<u8> {
    let (kind, index) = (slot & 0xffff_0000, slot & 0xffff);
    match (kind, index) {
        (DXGI_DEVICE_PROBE_BASE, 8) => Some(5),   // CreateSurface
        (DXGI_FACTORY_PROBE_BASE, 10) => Some(3), // CreateSwapChain
        (DXGI_FACTORY_PROBE_BASE, 11 | 27) => Some(2), // CreateSoftwareAdapter, EnumWarpAdapter
        (DXGI_FACTORY_PROBE_BASE, 15) => Some(6),
        (DXGI_FACTORY_PROBE_BASE, 16) => Some(5),
        (DXGI_FACTORY_PROBE_BASE, 24) => Some(4),
        _ => None,
    }
}

extern "win64" fn unsupported(this: *mut Object, caller: u64, slot: u32) -> u64 {
    // SAFETY: The thunk forwards the original COM this pointer.
    if let Some(probe) = unsafe { this.as_ref() }.and_then(|obj| obj.interfaces().probe) {
        probe(slot, caller);
    }
    let (interface, slot) = decode_d3d12_vtable_probe(slot);
    log_com_trace(
        "DXGI",
        format_args!("{interface}[{slot}] caller=0x{caller:x} -> DXGI_ERROR_UNSUPPORTED"),
    );
    u64::from(UNSUPPORTED as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_description_is_terminated_without_splitting_a_surrogate_pair() {
        assert_eq!(wide::<3>("a🦊"), [u16::from(b'a'), 0, 0]);
        assert_eq!(wide::<4>("a🦊"), [u16::from(b'a'), 0xd83e, 0xdd8a, 0]);
    }

    #[test]
    fn public_dxgi_iids_use_win64_memory_order() {
        assert_eq!(
            DEVICES[0],
            [
                0xfa, 0x77, 0xec, 0x54, 0x77, 0x13, 0xe6, 0x44, 0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44,
                0xc8, 0x4c
            ]
        );
        assert_eq!(
            format_guid(&FACTORIES[6]),
            "{c1b6694f-ff09-44a9-b03c-77900a0a1d17}"
        );
    }
}
