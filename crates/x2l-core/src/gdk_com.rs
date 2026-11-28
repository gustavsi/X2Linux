// Win64 COM vtable layouts and struct offsets informed by Marvisak/VIctory
// (https://github.com/Marvisak/VIctory, commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-hle/src/modules/{xg,dstorage,gameinput,lifecycle}.rs), xcl-dev/xcl (commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/{xg,dstorage,gameinput}.rs),
// and public Microsoft GDK DirectStorage / GameInput / AppNotify specifications.
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: native Linux Win64 COM objects backed by `x2l-formats`,
// `x2l-input`, and W^X spare-slot probe thunks (`x2l-gpu-vk`).

// Modified 2026-11-28: typed XG constructors, layout metadata, and audio completions.

//! Win64 COM objects for `xg_xs.dll` (`IXGTextureAddressComputer`), `XCRDAPI.dll`
//! (`IDStorageFactoryX`, `IDStorageQueueX`, `IDStorageFileX`, `IDStorageStatusArrayX`),
//! `GameInput.dll` (`IGameInput`), `AcpHal.dll` (`IAcpHal`), and PSM AppNotify cookies.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use x2l_audio::coproc::{
    AcpCommandRing, AcpMessage, XAPU_INVALID_COMMAND, XAPU_PENDING_RESULTS, XApuCommand,
    XApuCommandId, XApuResult,
};
use x2l_audio::{AudioOutput, DEFAULT_SAMPLE_RATE_HZ};
use x2l_formats::texture::{Dimension, PipeProfile, Swizzle, Texture, TextureDesc};
use x2l_formats::{bcpack, decompress_zlib_bounded};
use x2l_gpu_vk::com::{D3D12ResourceDesc, E_FAIL, E_INVALIDARG, E_POINTER, S_OK, exec_native};
use x2l_gpu_vk::{VtableProbeCallback, build_probe_vtable_table};
use x2l_input::{
    GAMEINPUT_E_DEVICE_NOT_FOUND, GAMEINPUT_E_READING_NOT_FOUND, GameInputGamepadState,
    GameInputState,
};

// ============================================================================
// Shared Win64 COM vtable registry (preserves W^X probe thunks on spare slots)
// ============================================================================

struct GdkComVtables {
    xg_computer: &'static [usize],
    dstorage_factory: &'static [usize],
    dstorage_queue: &'static [usize],
    dstorage_file: &'static [usize],
    dstorage_status: &'static [usize],
    gameinput: &'static [usize],
    gameinput_reading: &'static [usize],
    acphal: &'static [usize],
    acphal_connected: &'static [usize],
}

static GDK_VTABLES: OnceLock<GdkComVtables> = OnceLock::new();

fn init_gdk_vtables(probe_cb: VtableProbeCallback) -> Option<&'static GdkComVtables> {
    if let Some(vt) = GDK_VTABLES.get() {
        return Some(vt);
    }
    let built = exec_native(|| -> Result<GdkComVtables, x2l_gpu_vk::GpuError> {
        // IXGTextureAddressComputer starts at slot 0 = AddRef, slot 1 = Release (no QueryInterface).
        let xg_computer = build_probe_vtable_table(
            16,
            probe_cb,
            &[
                (0, xg_add_ref as *const () as usize),
                (1, xg_release as *const () as usize),
                (2, xg_get_resource_layout as *const () as usize),
                (3, xg_get_resource_size_bytes as *const () as usize),
                (
                    4,
                    xg_get_resource_base_alignment_bytes as *const () as usize,
                ),
                (5, xg_get_mip_level_offset_bytes as *const () as usize),
                (6, xg_get_texel_element_offset_bytes as *const () as usize),
                (8, xg_copy_into_subresource as *const () as usize),
                (9, xg_copy_from_subresource as *const () as usize),
            ],
        )?;

        let dstorage_factory = build_probe_vtable_table(
            24,
            probe_cb,
            &[
                (0, ds_fac_qi as *const () as usize),
                (1, ds_fac_add_ref as *const () as usize),
                (2, ds_fac_release as *const () as usize),
                (3, ds_fac_create_queue as *const () as usize),
                (4, ds_fac_open_file as *const () as usize),
                (5, ds_fac_create_status_array as *const () as usize),
                (6, ds_fac_set_cpu_affinity as *const () as usize),
                (7, ds_fac_set_debug_flags as *const () as usize),
                (8, ds_fac_set_staging_buffer_size as *const () as usize),
                (9, ds_fac_get_stats as *const () as usize),
            ],
        )?;

        let dstorage_queue = build_probe_vtable_table(
            24,
            probe_cb,
            &[
                (0, ds_q_qi as *const () as usize),
                (1, ds_q_add_ref as *const () as usize),
                (2, ds_q_release as *const () as usize),
                (3, ds_q_enqueue_request as *const () as usize),
                (4, ds_q_enqueue_status as *const () as usize),
                (5, ds_q_enqueue_signal as *const () as usize),
                (6, ds_q_submit as *const () as usize),
                (7, ds_q_cancel_with_tag as *const () as usize),
                (8, ds_q_close as *const () as usize),
                (9, ds_q_get_error_event as *const () as usize),
                (10, ds_q_retrieve_error_record as *const () as usize),
                (11, ds_q_query as *const () as usize),
            ],
        )?;

        let dstorage_file = build_probe_vtable_table(
            16,
            probe_cb,
            &[
                (0, ds_file_qi as *const () as usize),
                (1, ds_file_add_ref as *const () as usize),
                (2, ds_file_release as *const () as usize),
                (3, ds_file_close as *const () as usize),
                (4, ds_file_get_handle as *const () as usize),
                (5, ds_file_get_file_information as *const () as usize),
            ],
        )?;

        let dstorage_status = build_probe_vtable_table(
            16,
            probe_cb,
            &[
                (0, ds_st_qi as *const () as usize),
                (1, ds_st_add_ref as *const () as usize),
                (2, ds_st_release as *const () as usize),
                (3, ds_st_is_complete as *const () as usize),
                (4, ds_st_get_hresult as *const () as usize),
            ],
        )?;

        let gameinput = build_probe_vtable_table(
            32,
            probe_cb,
            &[
                (0, gi_qi as *const () as usize),
                (1, gi_add_ref as *const () as usize),
                (2, gi_release as *const () as usize),
                (3, gi_get_current_timestamp as *const () as usize),
                (4, gi_get_current_reading as *const () as usize),
                (5, gi_get_next_reading as *const () as usize),
                (6, gi_get_previous_reading as *const () as usize),
                (7, gi_get_temporal_reading as *const () as usize),
                (8, gi_register_reading_cb as *const () as usize),
                (9, gi_register_device_cb as *const () as usize),
                (10, gi_register_sysbtn_cb as *const () as usize),
                (11, gi_register_kbd_cb as *const () as usize),
                (12, gi_stop_callback as *const () as usize),
                (13, gi_unregister_callback as *const () as usize),
                (16, gi_find_device_not_found as *const () as usize),
                (17, gi_find_device_not_found as *const () as usize),
                (18, gi_find_device_not_found as *const () as usize),
                (19, gi_find_device_not_found as *const () as usize),
                (20, gi_enable_oem_device_support as *const () as usize),
                (21, gi_set_focus_policy as *const () as usize),
            ],
        )?;

        let gameinput_reading = build_probe_vtable_table(
            26,
            probe_cb,
            &[
                (0, gi_read_qi as *const () as usize),
                (1, gi_read_add_ref as *const () as usize),
                (2, gi_read_release as *const () as usize),
                (3, gi_read_get_input_kind as *const () as usize),
                (4, gi_read_get_sequence_number as *const () as usize),
                (5, gi_read_get_timestamp as *const () as usize),
                (6, gi_read_get_device as *const () as usize),
                (7, gi_read_ret_false as *const () as usize),
                (8, gi_read_ret_zero_u32 as *const () as usize),
                (9, gi_read_ret_zero_u32 as *const () as usize),
                (10, gi_read_ret_zero_u32 as *const () as usize),
                (11, gi_read_ret_zero_u32 as *const () as usize),
                (12, gi_read_ret_zero_u32 as *const () as usize),
                (13, gi_read_ret_zero_u32 as *const () as usize),
                (14, gi_read_ret_zero_u32 as *const () as usize),
                (15, gi_read_ret_zero_u32 as *const () as usize),
                (16, gi_read_ret_false as *const () as usize),
                (17, gi_read_ret_zero_u32 as *const () as usize),
                (18, gi_read_ret_zero_u32 as *const () as usize),
                (19, gi_read_ret_false as *const () as usize),
                (20, gi_read_ret_false as *const () as usize),
                (21, gi_read_ret_false as *const () as usize),
                (22, gi_read_get_gamepad_state as *const () as usize),
                (23, gi_read_ret_false as *const () as usize),
                (24, gi_read_ret_false as *const () as usize),
            ],
        )?;

        let acphal = build_probe_vtable_table(
            24,
            probe_cb,
            &[
                (0, acp_connect as *const () as usize),
                (1, acp_disconnect as *const () as usize),
                (2, acp_release_legacy as *const () as usize),
                (3, acp_pop_message as *const () as usize),
                (4, acp_message_count as *const () as usize),
                (5, acp_release as *const () as usize),
            ],
        )?;
        let acphal_connected = build_probe_vtable_table(
            24,
            probe_cb,
            &[
                (0, acp_connect as *const () as usize),
                (1, acp_disconnect as *const () as usize),
                (2, acp_submit as *const () as usize),
                (3, acp_pop_message as *const () as usize),
                (4, acp_message_count as *const () as usize),
                (5, acp_release as *const () as usize),
            ],
        )?;

        Ok(GdkComVtables {
            xg_computer,
            dstorage_factory,
            dstorage_queue,
            dstorage_file,
            dstorage_status,
            gameinput,
            gameinput_reading,
            acphal,
            acphal_connected,
        })
    })
    .ok()?;
    let _ = GDK_VTABLES.set(built);
    GDK_VTABLES.get()
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
            // SAFETY: `this` is a live COM object and `out_obj` is writable.
            unsafe {
                (*this).ref_count.fetch_add(1, Ordering::AcqRel);
                *out_obj = this.cast::<c_void>();
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
            // SAFETY: `this` was allocated via `Box::into_raw` and is freed once at refcount 0.
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

// ============================================================================
// 1. xg_xs.dll — IXGTextureAddressComputer & XGSuggestSwizzleMode
// ============================================================================

/// `XG_RESOURCE_LAYOUT` wire size (`0x1B40` = 6976 bytes, align 8).
pub const XG_RESOURCE_LAYOUT_SIZE: usize = 0x1B40;
const XG_PLANE_OFFSET: usize = 0x18;
const XG_PLANE_STRIDE: usize = 0x6C8;
const XG_MIP_OFFSET_IN_PLANE: usize = 0x38;
const XG_MIP_STRIDE: usize = 0x70;
const XG_MAX_LAYOUT_MIPS: usize = (XG_PLANE_STRIDE - XG_MIP_OFFSET_IN_PLANE) / XG_MIP_STRIDE;
pub const OP_XG_CREATE_TYPED_COMPUTER: u32 = 0x0402;
pub const OP_XG_COMPUTE_OPTIMAL_TILE_MODE: u32 = 0x0403;

#[repr(C)]
struct XgTexture2dDesc {
    width: u32,
    height: u32,
    mip_levels: u32,
    array_size: u32,
    format: u32,
    samples: u32,
    sample_quality: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access_flags: u32,
    misc_flags: u32,
    swizzle: u32,
    pitch: u32,
}

#[repr(C)]
struct XgTexture3dDesc {
    width: u32,
    height: u32,
    depth: u32,
    mip_levels: u32,
    format: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access_flags: u32,
    misc_flags: u32,
    swizzle: u32,
    pitch: u32,
}

const _: () = assert!(size_of::<XgTexture2dDesc>() == 52);
const _: () = assert!(size_of::<XgTexture3dDesc>() == 44);

fn choose_swizzle(
    dimension: u32,
    format: u32,
    samples: u32,
    flags: u32,
    extent: [u32; 3],
) -> Option<Swizzle> {
    if matches!(dimension, 1 | 2) {
        return (samples == 1).then_some(Swizzle::LINEAR);
    }
    if !matches!(dimension, 3 | 4)
        || !matches!(samples, 1 | 2 | 4 | 8)
        || (dimension == 4 && samples != 1)
    {
        return None;
    }
    let format = x2l_formats::texture::format(format)?;
    if format.block > 1 && (samples > 1 || flags & 3 != 0) {
        return None;
    }
    let [width, height, depth] = extent;
    let volume_depth = if dimension == 4 { depth } else { 1 };
    let bytes = u64::from(width.div_ceil(format.block))
        .saturating_mul(u64::from(height.div_ceil(format.block)))
        .saturating_mul(u64::from(volume_depth))
        .saturating_mul(u64::from(format.bytes));
    Some(if flags & 2 != 0 {
        Swizzle::SW_64KB_Z_X
    } else if flags & 1 != 0 || samples > 1 || (flags & 4 != 0 && bytes > 65_536) {
        Swizzle::SW_64KB_R_X
    } else if flags & 4 != 0 {
        Swizzle::SW_4KB_S_X
    } else {
        Swizzle::SW_4KB_S
    })
}

fn resource_flags(bind_flags: u32) -> u32 {
    (bind_flags >> 5) & 7
}

pub fn xg_compute_optimal_tile_mode(
    dimension: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    samples: u32,
    bind: u32,
) -> u32 {
    if width == 0 || height == 0 || depth == 0 {
        return u32::MAX;
    }
    choose_swizzle(
        dimension,
        format,
        samples,
        resource_flags(bind),
        [width, height, depth],
    )
    .map_or(u32::MAX, |swizzle| swizzle.0)
}

#[repr(C)]
pub struct ComXgTextureComputer {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    dimension: u32,
    depth: bool,
    texture: Texture,
}

/// `XGSuggestSwizzleMode(desc, flags, out_layout) -> HRESULT`.
///
/// # Safety
/// `desc` must be readable and `out_layout` must be writable if non-null.
pub unsafe fn xg_suggest_swizzle_mode(
    desc: *const D3D12ResourceDesc,
    _flags: u32,
    out_layout: *mut u32,
) -> i32 {
    if desc.is_null() || out_layout.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller passed a live 56-byte `D3D12ResourceDesc` and writable `u32*`.
    let d = unsafe { desc.read_unaligned() };
    let Ok(width) = u32::try_from(d.width) else {
        return E_INVALIDARG;
    };
    let Some(swizzle) = choose_swizzle(
        d.dimension,
        if d.format == 0 { 28 } else { d.format },
        d.sample_desc.count.max(1),
        d.flags,
        [width, d.height, u32::from(d.depth_or_array_size)],
    ) else {
        return E_INVALIDARG;
    };
    let mode = if d.dimension == 1 {
        0
    } else {
        0x100 | swizzle.0
    };
    // SAFETY: `out_layout` was checked non-null.
    unsafe {
        *out_layout = mode;
    }
    S_OK
}

/// `XGCreateTextureComputer(desc, out_computer) -> HRESULT` backed by `x2l_formats::texture`.
///
/// # Safety
/// `desc` must be readable and `out_computer` must be writable.
pub unsafe fn xg_create_texture_computer(
    desc: *const D3D12ResourceDesc,
    out_computer: *mut *mut c_void,
    probe_cb: VtableProbeCallback,
) -> i32 {
    if desc.is_null() || out_computer.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller provides writable output storage and a readable descriptor.
    let (d, out) = unsafe { (desc.read_unaligned(), &mut *out_computer) };
    *out = ptr::null_mut();
    let dim = match d.dimension {
        2 => Dimension::Tex1d,
        3 => Dimension::Tex2d,
        4 => Dimension::Tex3d,
        _ => return E_INVALIDARG,
    };
    let format = if d.format == 0 { 28 } else { d.format };
    let samples = d.sample_desc.count.max(1);
    let depth = d.flags & 2 != 0;
    let Ok(width) = u32::try_from(d.width) else {
        return E_INVALIDARG;
    };
    let swizzle = match d.layout {
        0 => match choose_swizzle(
            d.dimension,
            format,
            samples,
            d.flags,
            [width, d.height, u32::from(d.depth_or_array_size)],
        ) {
            Some(mode) => mode,
            None => return E_INVALIDARG,
        },
        1 => Swizzle::LINEAR,
        2 if dim == Dimension::Tex2d && samples == 1 && !depth => Swizzle::SW_64KB_D,
        n if n & !0xff == 0x100 => Swizzle(n & 0xff),
        _ => return E_INVALIDARG,
    };
    let tex_desc = TextureDesc {
        dimension: dim,
        width,
        height: if dim == Dimension::Tex1d { 1 } else { d.height },
        depth_or_array: u32::from(d.depth_or_array_size),
        mips: u32::from(d.mip_levels),
        format,
        samples,
        swizzle,
    };
    create_xg_computer(&tex_desc, depth, out, probe_cb)
}

/// # Safety
/// The descriptor matches the dimension and the output pointer is writable.
pub unsafe fn xg_create_typed_computer(
    dimension: u32,
    desc: *const c_void,
    out_computer: *mut *mut c_void,
    probe_cb: VtableProbeCallback,
) -> i32 {
    if desc.is_null() || out_computer.is_null() {
        return E_POINTER;
    }
    // SAFETY: The bridge forwards the caller's writable output pointer.
    let out = unsafe { &mut *out_computer };
    *out = ptr::null_mut();
    let (mut texture, bind, pitch) = match dimension {
        3 => {
            // SAFETY: The 2D export supplies a readable XG_TEXTURE2D_DESC.
            let d = unsafe { desc.cast::<XgTexture2dDesc>().read_unaligned() };
            (
                TextureDesc {
                    dimension: Dimension::Tex2d,
                    width: d.width,
                    height: d.height,
                    depth_or_array: d.array_size,
                    mips: d.mip_levels,
                    format: d.format,
                    samples: d.samples,
                    swizzle: Swizzle(d.swizzle),
                },
                d.bind_flags,
                d.pitch,
            )
        }
        4 => {
            // SAFETY: The 3D export supplies a readable XG_TEXTURE3D_DESC.
            let d = unsafe { desc.cast::<XgTexture3dDesc>().read_unaligned() };
            (
                TextureDesc {
                    dimension: Dimension::Tex3d,
                    width: d.width,
                    height: d.height,
                    depth_or_array: d.depth,
                    mips: d.mip_levels,
                    format: d.format,
                    samples: 1,
                    swizzle: Swizzle(d.swizzle),
                },
                d.bind_flags,
                d.pitch,
            )
        }
        _ => return E_INVALIDARG,
    };
    let flags = resource_flags(bind);
    if texture.swizzle.0 == u32::MAX {
        let Some(mode) = choose_swizzle(
            dimension,
            texture.format,
            texture.samples,
            flags,
            [texture.width, texture.height, texture.depth_or_array],
        ) else {
            return E_INVALIDARG;
        };
        texture.swizzle = mode;
    }
    if pitch != 0 || texture.samples == 0 {
        return E_INVALIDARG;
    }
    create_xg_computer(&texture, flags & 2 != 0, out, probe_cb)
}

fn create_xg_computer(
    desc: &TextureDesc,
    depth: bool,
    out: &mut *mut c_void,
    probe_cb: VtableProbeCallback,
) -> i32 {
    let Ok(mut texture) = Texture::new(desc) else {
        return E_INVALIDARG;
    };
    if texture.planes[0].layout.mips.len() > XG_MAX_LAYOUT_MIPS {
        return E_INVALIDARG;
    }
    let profile =
        if super::runtime().kernelx.device_type() == x2l_hle_kernelx::XSYSTEM_DEVICE_SERIES_S {
            PipeProfile::SeriesS
        } else {
            PipeProfile::SeriesX
        };
    for plane in &mut texture.planes {
        plane.layout.set_pipe_profile(profile);
    }
    let Some(vtables) = init_gdk_vtables(probe_cb) else {
        return E_FAIL;
    };
    let dimension = match desc.dimension {
        Dimension::Tex1d => 2,
        Dimension::Tex2d => 3,
        Dimension::Tex3d => 4,
    };
    *out = Box::into_raw(Box::new(ComXgTextureComputer {
        vtbl: vtables.xg_computer.as_ptr(),
        ref_count: AtomicU32::new(1),
        dimension,
        depth,
        texture,
    }))
    .cast();
    S_OK
}

extern "win64" fn xg_add_ref(this: *mut ComXgTextureComputer) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComXgTextureComputer`.
    unsafe { (*this).ref_count.fetch_add(1, Ordering::AcqRel) + 1 }
}

extern "win64" fn xg_release(this: *mut ComXgTextureComputer) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` was allocated via `Box::into_raw`.
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

extern "win64" fn xg_get_resource_layout(
    this: *mut ComXgTextureComputer,
    out_layout: *mut u8,
) -> i32 {
    if this.is_null() || out_layout.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller passes a writable `0x1B40`-byte `XG_RESOURCE_LAYOUT` buffer.
    unsafe {
        let comp = &*this;
        let buf = std::slice::from_raw_parts_mut(out_layout, XG_RESOURCE_LAYOUT_SIZE);
        buf.fill(0);

        let write_u32 = |b: &mut [u8], off: usize, v: u32| {
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };
        let write_u64 = |b: &mut [u8], off: usize, v: u64| {
            b[off..off + 8].copy_from_slice(&v.to_le_bytes());
        };

        let mips_count = comp
            .texture
            .planes
            .first()
            .map(|p| p.layout.mips.len() as u32)
            .unwrap_or(1);
        write_u64(buf, 0x00, comp.texture.size);
        write_u64(buf, 0x08, comp.texture.alignment);
        write_u32(buf, 0x10, mips_count);
        write_u32(buf, 0x14, comp.texture.planes.len().min(4) as u32);

        for (pi, plane) in comp.texture.planes.iter().take(4).enumerate() {
            let p_off = XG_PLANE_OFFSET + pi * XG_PLANE_STRIDE;
            let lay = &plane.layout;
            let depth = comp.depth && (comp.texture.planes.len() > 1 || mips_count == 1);
            write_u32(
                buf,
                p_off,
                if plane.stencil {
                    8
                } else if depth {
                    7
                } else {
                    1
                },
            );
            write_u64(buf, p_off + 0x08, lay.size);
            write_u64(buf, p_off + 0x10, lay.slice_size);
            write_u64(buf, p_off + 0x18, plane.offset);
            write_u64(buf, p_off + 0x20, lay.alignment);
            write_u32(buf, p_off + 0x28, lay.desc.element_bytes);
            write_u32(buf, p_off + 0x2C, lay.first_tail_mip);
            write_u32(
                buf,
                p_off + 0x30,
                (lay.mips.len() as u32).saturating_sub(lay.first_tail_mip),
            );
            for (mi, mip) in lay.mips.iter().enumerate() {
                let m_off = p_off + XG_MIP_OFFSET_IN_PLANE + mi * XG_MIP_STRIDE;
                let three_d = comp.dimension == 4;
                let row_bytes = mip.pitch * lay.desc.element_bytes;
                let mut slice_bytes =
                    u64::from(row_bytes) * u64::from(mip.rows) * u64::from(lay.desc.samples);
                let mut offset = mip.offset;
                let mut alignment = lay.alignment as u32;
                let mut swizzle = lay.desc.swizzle.0;
                if let Some(tail) = mip.tail {
                    alignment = 256;
                    swizzle = match swizzle {
                        2 | 6 | 10 | 18 | 22 | 26 => Swizzle::SW_256B_D.0,
                        23 | 27 => 3,
                        24 => swizzle,
                        _ => Swizzle::SW_256B_S.0,
                    };
                    if lay.thick {
                        slice_bytes = lay.alignment / u64::from(lay.block[2]);
                        write_u32(buf, m_off + 0x68, tail.x | (tail.y << 11));
                    } else {
                        offset += u64::from(tail.offset);
                    }
                    write_u32(buf, m_off + 0x64, 1);
                }
                write_u64(buf, m_off, slice_bytes * u64::from(mip.depth_pitch));
                write_u64(buf, m_off + 0x08, offset);
                write_u64(buf, m_off + 0x10, slice_bytes);
                write_u32(buf, m_off + 0x18, mip.pitch * lay.desc.block);
                write_u32(buf, m_off + 0x1C, row_bytes);
                write_u32(buf, m_off + 0x20, alignment);
                write_u32(buf, m_off + 0x24, mip.pitch);
                write_u32(buf, m_off + 0x28, mip.rows);
                write_u32(
                    buf,
                    m_off + 0x2C,
                    if three_d {
                        mip.depth_pitch
                    } else {
                        lay.desc.depth
                    },
                );
                write_u32(buf, m_off + 0x30, mip.width);
                write_u32(buf, m_off + 0x34, mip.height);
                write_u32(buf, m_off + 0x38, (lay.desc.width >> mi).max(1));
                write_u32(buf, m_off + 0x3C, (lay.desc.height >> mi).max(1));
                write_u32(
                    buf,
                    m_off + 0x40,
                    if three_d { mip.depth } else { lay.desc.depth },
                );
                write_u32(buf, m_off + 0x44, lay.desc.samples);
                write_u32(buf, m_off + 0x48, swizzle);
                write_u32(buf, m_off + 0x60, lay.block[2]);
            }
        }
        write_u32(buf, 0x1B38, comp.dimension);
    }
    S_OK
}

extern "win64" fn xg_get_resource_size_bytes(this: *mut ComXgTextureComputer) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComXgTextureComputer`.
    unsafe { (*this).texture.size }
}

extern "win64" fn xg_get_resource_base_alignment_bytes(this: *mut ComXgTextureComputer) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComXgTextureComputer`.
    unsafe { (*this).texture.alignment }
}

extern "win64" fn xg_get_mip_level_offset_bytes(
    this: *mut ComXgTextureComputer,
    plane: u32,
    mip: u32,
) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComXgTextureComputer`.
    unsafe {
        let this_ref = &*this;
        let Some(p) = this_ref.texture.planes.get(plane as usize) else {
            return 0;
        };
        let Some(m) = p.layout.mips.get(mip as usize) else {
            return 0;
        };
        p.offset
            + m.offset
            + m.tail
                .filter(|_| !p.layout.thick)
                .map_or(0, |t| u64::from(t.offset))
    }
}

extern "win64" fn xg_get_texel_element_offset_bytes(
    this: *mut ComXgTextureComputer,
    plane: u32,
    mip: u32,
    x: u64,
    y: u32,
    z_or_slice: u32,
    sample: u32,
    bit_offset: *mut u32,
) -> u64 {
    if !bit_offset.is_null() {
        // SAFETY: Caller supplied writable optional `bit_offset`.
        unsafe { *bit_offset = 0 };
    }
    let Ok(x) = u32::try_from(x) else {
        return u64::MAX;
    };
    if this.is_null() || sample != 0 {
        return u64::MAX;
    }
    // SAFETY: The caller owns a live COM reference.
    let texture = unsafe { &(*this).texture };
    texture
        .planes
        .get(plane as usize)
        .and_then(|p| {
            p.layout
                .offset(mip, x, y, z_or_slice)
                .ok()
                .map(|off| p.offset + off)
        })
        .unwrap_or(u64::MAX)
}

extern "win64" fn xg_copy_into_subresource(
    this: *mut ComXgTextureComputer,
    tiled: *mut c_void,
    plane: u32,
    subresource: u32,
    linear: *const c_void,
    row_pitch: u32,
    slice_pitch: u32,
) -> i32 {
    if this.is_null() || tiled.is_null() || linear.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller supplies valid tiled and linear buffers matching this texture's layout.
    unsafe {
        let this_ref = &*this;
        let Some(p) = this_ref.texture.planes.get(plane as usize) else {
            return E_INVALIDARG;
        };
        let mips = p.layout.mips.len() as u32;
        let mip = subresource % mips;
        let slice = subresource / mips;
        let Ok(linear_len) =
            p.layout
                .linear_subresource_len(mip, slice, row_pitch as usize, slice_pitch as usize)
        else {
            return E_INVALIDARG;
        };
        if !xg_copy_spans_disjoint(p.offset, p.layout.size, tiled, linear, linear_len) {
            return E_INVALIDARG;
        }
        let tiled_slice = std::slice::from_raw_parts_mut(
            tiled.cast::<u8>().add(p.offset as usize),
            p.layout.size as usize,
        );
        let linear_slice = std::slice::from_raw_parts(linear.cast::<u8>(), linear_len);
        match p.layout.tile(
            tiled_slice,
            mip,
            slice,
            linear_slice,
            row_pitch as usize,
            slice_pitch as usize,
        ) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

extern "win64" fn xg_copy_from_subresource(
    this: *mut ComXgTextureComputer,
    linear: *mut c_void,
    plane: u32,
    subresource: u32,
    sample: u32,
    tiled: *const c_void,
    row_pitch: u32,
    slice_pitch: u32,
) -> i32 {
    if this.is_null() || tiled.is_null() || linear.is_null() {
        return E_POINTER;
    }
    if sample != 0 {
        return E_INVALIDARG;
    }
    // SAFETY: Caller supplies valid tiled source and linear destination buffers.
    unsafe {
        let this_ref = &*this;
        let Some(p) = this_ref.texture.planes.get(plane as usize) else {
            return E_INVALIDARG;
        };
        let mips = p.layout.mips.len() as u32;
        let mip = subresource % mips;
        let slice = subresource / mips;
        let Ok(linear_len) =
            p.layout
                .linear_subresource_len(mip, slice, row_pitch as usize, slice_pitch as usize)
        else {
            return E_INVALIDARG;
        };
        if !xg_copy_spans_disjoint(p.offset, p.layout.size, tiled, linear, linear_len) {
            return E_INVALIDARG;
        }
        let tiled_slice = std::slice::from_raw_parts(
            tiled.cast::<u8>().add(p.offset as usize),
            p.layout.size as usize,
        );
        let linear_slice = std::slice::from_raw_parts_mut(linear.cast::<u8>(), linear_len);
        match p.layout.detile(
            tiled_slice,
            mip,
            slice,
            linear_slice,
            row_pitch as usize,
            slice_pitch as usize,
        ) {
            Ok(()) => S_OK,
            Err(_) => E_FAIL,
        }
    }
}

fn xg_copy_spans_disjoint(
    plane_offset: u64,
    plane_size: u64,
    tiled: *const c_void,
    linear: *const c_void,
    linear_len: usize,
) -> bool {
    let Some(plane_start) = tiled.addr().checked_add(plane_offset as usize) else {
        return false;
    };
    let (Some(plane_end), Some(linear_end)) = (
        plane_start.checked_add(plane_size as usize),
        linear.addr().checked_add(linear_len),
    ) else {
        return false;
    };
    linear_len <= isize::MAX as usize && (plane_end <= linear.addr() || linear_end <= plane_start)
}

// ============================================================================
// 2. XCRDAPI.dll — DirectStorage (`IDStorageFactoryX`, Queue, File, StatusArray)
// ============================================================================

/// Exact `0x48` (72)-byte Win64 `DSTORAGE_REQUEST` layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DStorageRequest {
    pub options: u64,           // @0x00
    pub dst_ptr: u64,           // @0x08
    pub dst_page_offset: u64,   // @0x10
    pub dst_size: u32,          // @0x18
    pub _pad1c: u32,            // @0x1C
    pub src_ptr: u64,           // @0x20 (IDStorageFileX* or memory ptr)
    pub file_offset: u64,       // @0x28
    pub src_size: u32,          // @0x30
    pub intermediate_size: u32, // @0x34
    pub cancellation_tag: u64,  // @0x38
    pub name: *const u8,        // @0x40
}

const _: () = assert!(size_of::<DStorageRequest>() == 0x48);

#[repr(C)]
pub struct ComDStorageFactory {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    vtables: &'static GdkComVtables,
    merged_root: Option<PathBuf>,
    saves_root: Option<PathBuf>,
    staging_bytes: AtomicU32,
}

#[repr(C)]
pub struct ComDStorageFile {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    path: PathBuf,
    closed: AtomicBool,
}

#[repr(C)]
pub struct ComDStorageStatusArray {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    entries: Mutex<Vec<(bool, i32)>>,
}

enum QueuedDsOp {
    Read {
        options: u64,
        dst_ptr: usize,
        dst_size: usize,
        src_file_path: Option<PathBuf>,
        src_mem_ptr: usize,
        file_offset: u64,
        src_size: usize,
        cancellation_tag: u64,
    },
    Status {
        status_obj: usize,
        index: u32,
    },
    Signal {
        fence_obj: usize,
        value: u64,
    },
}

#[repr(C)]
pub struct ComDStorageQueue {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    pending: Mutex<Vec<QueuedDsOp>>,
    closed: AtomicBool,
}

impl Drop for ComDStorageQueue {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.pending.lock() {
            for op in std::mem::take(&mut *guard) {
                match op {
                    QueuedDsOp::Status { status_obj, .. } => {
                        ds_st_release(status_obj as *mut ComDStorageStatusArray);
                    }
                    QueuedDsOp::Signal { fence_obj, .. } => {
                        // SAFETY: `fence_obj` is an `ID3D12Fence` COM object `AddRef`'d in `EnqueueSignal`.
                        unsafe {
                            let vt = *(fence_obj as *const *const usize);
                            let rel: extern "win64" fn(*mut c_void) -> u32 =
                                std::mem::transmute(*vt.add(2));
                            let _ = rel(fence_obj as *mut c_void);
                        }
                    }
                    QueuedDsOp::Read { .. } => {}
                }
            }
        }
    }
}

impl_iunknown!(
    ComDStorageFactory,
    ds_fac_qi,
    ds_fac_add_ref,
    ds_fac_release
);
impl_iunknown!(ComDStorageQueue, ds_q_qi, ds_q_add_ref, ds_q_release);
impl_iunknown!(
    ComDStorageFile,
    ds_file_qi,
    ds_file_add_ref,
    ds_file_release
);
impl_iunknown!(
    ComDStorageStatusArray,
    ds_st_qi,
    ds_st_add_ref,
    ds_st_release
);

/// `DStorageGetFactory(riid, ppv) -> HRESULT`.
///
/// # Safety
/// `out_factory` must be writable.
pub unsafe fn dstorage_get_factory(
    out_factory: *mut *mut c_void,
    probe_cb: VtableProbeCallback,
    merged_root: Option<PathBuf>,
    saves_root: Option<PathBuf>,
) -> i32 {
    if out_factory.is_null() {
        return E_POINTER;
    }
    let Some(vtables) = init_gdk_vtables(probe_cb) else {
        return E_FAIL;
    };
    let obj = Box::into_raw(Box::new(ComDStorageFactory {
        vtbl: vtables.dstorage_factory.as_ptr(),
        ref_count: AtomicU32::new(1),
        vtables,
        merged_root,
        saves_root,
        staging_bytes: AtomicU32::new(32 * 1024 * 1024),
    }));
    // SAFETY: `out_factory` was checked non-null.
    unsafe {
        *out_factory = obj.cast::<c_void>();
    }
    S_OK
}

extern "win64" fn ds_fac_create_queue(
    this: *mut ComDStorageFactory,
    _desc: *const c_void,
    _riid: *const c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_queue.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live and `out_queue` is writable.
    unsafe {
        let fac = &*this;
        let q = Box::into_raw(Box::new(ComDStorageQueue {
            vtbl: fac.vtables.dstorage_queue.as_ptr(),
            ref_count: AtomicU32::new(1),
            pending: Mutex::new(Vec::new()),
            closed: AtomicBool::new(false),
        }));
        *out_queue = q.cast::<c_void>();
    }
    S_OK
}

fn translate_console_path(
    utf16_path: &str,
    merged_root: Option<&PathBuf>,
    saves_root: Option<&PathBuf>,
) -> PathBuf {
    let trimmed = utf16_path
        .trim()
        .trim_start_matches("\\\\?\\")
        .replace('\\', "/");
    if let Some(rest) = trimmed
        .strip_prefix("G:/")
        .or_else(|| trimmed.strip_prefix("g:/"))
        .or_else(|| trimmed.strip_prefix("G:"))
        .or_else(|| trimmed.strip_prefix("g:"))
        && let Some(root) = merged_root
    {
        return root.join(rest.trim_start_matches('/'));
    }
    if let Some(rest) = trimmed
        .strip_prefix("S:/")
        .or_else(|| trimmed.strip_prefix("s:/"))
        && let Some(root) = saves_root
    {
        return root.join(rest.trim_start_matches('/'));
    }
    if let Some(rest) = trimmed
        .strip_prefix("Z:/")
        .or_else(|| trimmed.strip_prefix("z:/"))
    {
        return PathBuf::from("/").join(rest);
    }
    if let Some(root) = merged_root {
        let candidate = root.join(trimmed.trim_start_matches('/'));
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(trimmed)
}

extern "win64" fn ds_fac_open_file(
    this: *mut ComDStorageFactory,
    path_utf16: *const u16,
    _riid: *const c_void,
    out_file: *mut *mut c_void,
) -> i32 {
    if this.is_null() || path_utf16.is_null() || out_file.is_null() {
        return E_POINTER;
    }
    // SAFETY: `path_utf16` is a NUL-terminated UTF-16 `PCWSTR` bounded to 4096 code units.
    let utf16_str = unsafe {
        let mut len = 0usize;
        while len < 4096 && *path_utf16.add(len) != 0 {
            len += 1;
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(path_utf16, len))
    };
    // SAFETY: `this` is a live `ComDStorageFactory`.
    let fac = unsafe { &*this };
    let host_path = translate_console_path(
        &utf16_str,
        fac.merged_root.as_ref(),
        fac.saves_root.as_ref(),
    );
    let exists = exec_native({
        let p = host_path.clone();
        move || p.is_file()
    });
    if !exists {
        return -2147024894; // HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) 0x80070002
    }
    let obj = Box::into_raw(Box::new(ComDStorageFile {
        vtbl: fac.vtables.dstorage_file.as_ptr(),
        ref_count: AtomicU32::new(1),
        path: host_path,
        closed: AtomicBool::new(false),
    }));
    // SAFETY: `out_file` is validated non-null.
    unsafe {
        *out_file = obj.cast::<c_void>();
    }
    S_OK
}

extern "win64" fn ds_fac_create_status_array(
    this: *mut ComDStorageFactory,
    capacity: u32,
    _name: *const u8,
    _riid: *const c_void,
    out_arr: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_arr.is_null() {
        return E_POINTER;
    }
    let cap = (capacity as usize).clamp(1, 65536);
    // SAFETY: `this` is live and `out_arr` is writable.
    unsafe {
        let fac = &*this;
        let obj = Box::into_raw(Box::new(ComDStorageStatusArray {
            vtbl: fac.vtables.dstorage_status.as_ptr(),
            ref_count: AtomicU32::new(1),
            entries: Mutex::new(vec![(true, S_OK); cap]),
        }));
        *out_arr = obj.cast::<c_void>();
    }
    S_OK
}

extern "win64" fn ds_fac_set_cpu_affinity(_this: *mut ComDStorageFactory, _affinity: u64) -> i32 {
    S_OK
}

extern "win64" fn ds_fac_set_debug_flags(_this: *mut ComDStorageFactory, _flags: u32) {}

extern "win64" fn ds_fac_set_staging_buffer_size(this: *mut ComDStorageFactory, size: u32) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live.
    unsafe {
        (*this).staging_bytes.store(size, Ordering::Relaxed);
    }
    S_OK
}

extern "win64" fn ds_fac_get_stats(_this: *mut ComDStorageFactory, out_stats: *mut u8) {
    if !out_stats.is_null() {
        // SAFETY: Zero 64 bytes of optional stats buffer if provided.
        unsafe {
            ptr::write_bytes(out_stats, 0, 64);
        }
    }
}

extern "win64" fn ds_file_close(this: *mut ComDStorageFile) {
    if !this.is_null() {
        // SAFETY: `this` is live.
        unsafe {
            (*this).closed.store(true, Ordering::Release);
        }
    }
}

extern "win64" fn ds_file_get_handle(_this: *mut ComDStorageFile) -> *mut c_void {
    ptr::null_mut()
}

extern "win64" fn ds_file_get_file_information(
    this: *mut ComDStorageFile,
    out_info: *mut u32,
) -> i32 {
    if this.is_null() || out_info.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `BY_HANDLE_FILE_INFORMATION` is 52 bytes (13 u32s; sizeHigh@8, sizeLow@9).
    unsafe {
        let p = (*this).path.clone();
        let len = exec_native(move || fs::metadata(&p).map(|m| m.len()).unwrap_or(0));
        let words = std::slice::from_raw_parts_mut(out_info, 13);
        words.fill(0);
        words[8] = (len >> 32) as u32;
        words[9] = len as u32;
    }
    S_OK
}

extern "win64" fn ds_st_is_complete(this: *mut ComDStorageStatusArray, index: u32) -> bool {
    if this.is_null() {
        return false;
    }
    // SAFETY: `this` is live.
    unsafe {
        (*this)
            .entries
            .lock()
            .ok()
            .and_then(|g| g.get(index as usize).map(|e| e.0))
            .unwrap_or(false)
    }
}

extern "win64" fn ds_st_get_hresult(this: *mut ComDStorageStatusArray, index: u32) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live.
    unsafe {
        (*this)
            .entries
            .lock()
            .ok()
            .and_then(|g| g.get(index as usize).map(|e| e.1))
            .unwrap_or(E_INVALIDARG)
    }
}

extern "win64" fn ds_q_enqueue_request(
    this: *mut ComDStorageQueue,
    request: *const DStorageRequest,
) {
    if this.is_null() || request.is_null() {
        return;
    }
    // SAFETY: `this` is live and `request` points to an initialized 72-byte `DStorageRequest`.
    unsafe {
        let q = &*this;
        if q.closed.load(Ordering::Acquire) {
            return;
        }
        let r = *request;
        let source_is_memory = ((r.options >> 22) & 1) != 0;
        let (src_file_path, src_mem_ptr) = if source_is_memory {
            (None, r.src_ptr as usize)
        } else if r.src_ptr != 0 {
            let f = &*(r.src_ptr as *const ComDStorageFile);
            (Some(f.path.clone()), 0)
        } else {
            (None, 0)
        };
        if let Ok(mut guard) = q.pending.lock() {
            guard.push(QueuedDsOp::Read {
                options: r.options,
                dst_ptr: r.dst_ptr as usize,
                dst_size: r.dst_size as usize,
                src_file_path,
                src_mem_ptr,
                file_offset: r.file_offset,
                src_size: r.src_size as usize,
                cancellation_tag: r.cancellation_tag,
            });
        }
    }
}

extern "win64" fn ds_q_enqueue_status(
    this: *mut ComDStorageQueue,
    status_array: *mut ComDStorageStatusArray,
    index: u32,
) {
    if this.is_null() || status_array.is_null() {
        return;
    }
    // SAFETY: `this` and `status_array` are live COM objects; retain `status_array` until Submit.
    unsafe {
        (*status_array).ref_count.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut entries) = (*status_array).entries.lock()
            && let Some(slot) = entries.get_mut(index as usize)
        {
            *slot = (false, S_OK);
        }
        if let Ok(mut guard) = (*this).pending.lock() {
            guard.push(QueuedDsOp::Status {
                status_obj: status_array as usize,
                index,
            });
        }
    }
}

extern "win64" fn ds_q_enqueue_signal(this: *mut ComDStorageQueue, fence: *mut c_void, value: u64) {
    if this.is_null() || fence.is_null() {
        return;
    }
    // SAFETY: `this` is a live `ComDStorageQueue` and `fence` is a live `ID3D12Fence` COM object
    // with `IUnknown::AddRef` at vtable slot 1; retain a reference until `Submit` signals & releases.
    unsafe {
        let q = &*this;
        if q.closed.load(Ordering::Acquire) {
            return;
        }
        let vt = *(fence as *const *const usize);
        let add_ref: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(1));
        let _ = add_ref(fence);
        if let Ok(mut guard) = q.pending.lock() {
            guard.push(QueuedDsOp::Signal {
                fence_obj: fence as usize,
                value,
            });
        } else {
            let rel: extern "win64" fn(*mut c_void) -> u32 = std::mem::transmute(*vt.add(2));
            let _ = rel(fence);
        }
    }
}

extern "win64" fn ds_q_submit(this: *mut ComDStorageQueue) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is a live `ComDStorageQueue`.
    let ops = unsafe {
        let Ok(mut guard) = (*this).pending.lock() else {
            return;
        };
        std::mem::take(&mut *guard)
    };

    let mut last_hr = S_OK;
    for op in ops {
        match op {
            QueuedDsOp::Read {
                options,
                dst_ptr,
                dst_size,
                src_file_path,
                src_mem_ptr,
                file_offset,
                src_size,
                ..
            } => {
                if dst_ptr == 0 || dst_size == 0 {
                    last_hr = E_INVALIDARG;
                    continue;
                }
                let raw_src: Result<Vec<u8>, i32> = if let Some(path) = src_file_path {
                    exec_native(move || -> Result<Vec<u8>, i32> {
                        let mut f = File::open(&path).map_err(|_| E_FAIL)?;
                        f.seek(SeekFrom::Start(file_offset)).map_err(|_| E_FAIL)?;
                        let read_len = if src_size == 0 { dst_size } else { src_size };
                        let mut buf = vec![0u8; read_len];
                        let n = f.read(&mut buf).map_err(|_| E_FAIL)?;
                        buf.truncate(n);
                        Ok(buf)
                    })
                } else if src_mem_ptr != 0 {
                    let read_len = if src_size == 0 { dst_size } else { src_size };
                    // SAFETY: Caller queued a memory-source DirectStorage request of `read_len` bytes.
                    Ok(unsafe {
                        std::slice::from_raw_parts(src_mem_ptr as *const u8, read_len).to_vec()
                    })
                } else {
                    Err(E_INVALIDARG)
                };

                let Ok(src_bytes) = raw_src else {
                    last_hr = E_FAIL;
                    continue;
                };

                let use_zlib = (options & 1) != 0;
                let bcpack_mode = ((options >> 1) & 0xF) as u8;
                let decoded: Result<Vec<u8>, i32> = if use_zlib {
                    decompress_zlib_bounded(&src_bytes, dst_size).map_err(|_| E_FAIL)
                } else if bcpack_mode != 0 {
                    bcpack::Tables::builtin()
                        .ok()
                        .and_then(|tables| {
                            let mut out = vec![0u8; dst_size];
                            bcpack::decode(bcpack_mode, &src_bytes, &mut out, &tables)
                                .ok()
                                .map(|()| out)
                        })
                        .ok_or(E_FAIL)
                } else {
                    Ok(src_bytes)
                };

                match decoded {
                    Ok(bytes) => {
                        let copy_len = bytes.len().min(dst_size);
                        // SAFETY: Caller queued a writable destination buffer `(dst_ptr, dst_size)`.
                        unsafe {
                            ptr::copy_nonoverlapping(bytes.as_ptr(), dst_ptr as *mut u8, copy_len);
                        }
                        last_hr = S_OK;
                    }
                    Err(hr) => last_hr = hr,
                }
            }
            QueuedDsOp::Status { status_obj, index } => {
                let st = status_obj as *mut ComDStorageStatusArray;
                // SAFETY: `status_obj` was `AddRef`'d in `EnqueueStatus`; mark complete then release.
                unsafe {
                    if let Ok(mut entries) = (*st).entries.lock()
                        && let Some(slot) = entries.get_mut(index as usize)
                    {
                        *slot = (true, last_hr);
                    }
                    ds_st_release(st);
                }
            }
            QueuedDsOp::Signal { fence_obj, value } => {
                let fence = fence_obj as *mut c_void;
                // SAFETY: `fence_obj` is a live `ID3D12Fence` COM object (`AddRef`'d in `EnqueueSignal`)
                // implementing `Signal(u64)` at vtable slot 10 and `IUnknown::Release` at slot 2.
                unsafe {
                    let vt = *(fence as *const *const usize);
                    let sig: extern "win64" fn(*mut c_void, u64) -> i32 =
                        std::mem::transmute(*vt.add(10));
                    let _ = sig(fence, value);
                    let rel: extern "win64" fn(*mut c_void) -> u32 =
                        std::mem::transmute(*vt.add(2));
                    let _ = rel(fence);
                }
            }
        }
    }
}

extern "win64" fn ds_q_cancel_with_tag(this: *mut ComDStorageQueue, mask: u64, value: u64) {
    if this.is_null() {
        return;
    }
    // SAFETY: `this` is live.
    unsafe {
        if let Ok(mut guard) = (*this).pending.lock() {
            guard.retain(|op| match op {
                QueuedDsOp::Read {
                    cancellation_tag, ..
                } => (cancellation_tag & mask) != value,
                QueuedDsOp::Status { .. } | QueuedDsOp::Signal { .. } => true,
            });
        }
    }
}

extern "win64" fn ds_q_close(this: *mut ComDStorageQueue) {
    if !this.is_null() {
        // SAFETY: `this` is live.
        unsafe {
            (*this).closed.store(true, Ordering::Release);
        }
    }
}

extern "win64" fn ds_q_get_error_event(_this: *mut ComDStorageQueue) -> *mut c_void {
    ptr::null_mut()
}

extern "win64" fn ds_q_retrieve_error_record(_this: *mut ComDStorageQueue, out_record: *mut u8) {
    if !out_record.is_null() {
        // SAFETY: Zero 88-byte error record header safely.
        unsafe {
            ptr::write_bytes(out_record, 0, 88);
        }
    }
}

extern "win64" fn ds_q_query(_this: *mut ComDStorageQueue, out_info: *mut u8) {
    if !out_info.is_null() {
        // SAFETY: Zero 32-byte queue info record safely.
        unsafe {
            ptr::write_bytes(out_info, 0, 32);
        }
    }
}

// ============================================================================
// 3. GameInput.dll — Offline `IGameInput` v0 COM object (ordinal #1)
// ============================================================================

#[repr(C)]
pub struct ComGameInput {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    state: Arc<GameInputState>,
}

impl_iunknown!(ComGameInput, gi_qi, gi_add_ref, gi_release);

/// `GameInputCreate(out_gameinput) -> HRESULT` (exported at ordinal `#1`).
///
/// # Safety
/// `out_gi` must be writable.
pub unsafe fn gameinput_create(
    state: Arc<GameInputState>,
    out_gi: *mut *mut c_void,
    probe_cb: VtableProbeCallback,
) -> i32 {
    if out_gi.is_null() {
        return E_POINTER;
    }
    let Some(vtables) = init_gdk_vtables(probe_cb) else {
        return E_FAIL;
    };
    let obj = Box::into_raw(Box::new(ComGameInput {
        vtbl: vtables.gameinput.as_ptr(),
        ref_count: AtomicU32::new(1),
        state,
    }));
    // SAFETY: `out_gi` was checked non-null.
    unsafe {
        *out_gi = obj.cast::<c_void>();
    }
    S_OK
}

extern "win64" fn gi_get_current_timestamp(this: *mut ComGameInput) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComGameInput`.
    unsafe { (*this).state.current_timestamp_us() }
}

const GAMEINPUT_KIND_GAMEPAD: u32 = 0x0002_0000;

#[repr(C)]
struct ComGameInputReading {
    vtbl: *const usize,
    ref_count: AtomicU32,
    timestamp_us: u64,
    sequence: u64,
    gamepad: GameInputGamepadState,
}

impl_iunknown!(
    ComGameInputReading,
    gi_read_qi,
    gi_read_add_ref,
    gi_read_release
);

static GI_READING_SEQ: AtomicU64 = AtomicU64::new(1);

extern "win64" fn gi_read_get_input_kind(_this: *mut ComGameInputReading) -> u32 {
    GAMEINPUT_KIND_GAMEPAD
}

extern "win64" fn gi_read_get_sequence_number(this: *mut ComGameInputReading, _kind: u32) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComGameInputReading`.
    unsafe { (*this).sequence }
}

extern "win64" fn gi_read_get_timestamp(this: *mut ComGameInputReading) -> u64 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComGameInputReading`.
    unsafe { (*this).timestamp_us }
}

extern "win64" fn gi_read_get_device(
    _this: *mut ComGameInputReading,
    out_device: *mut *mut c_void,
) {
    if !out_device.is_null() {
        // SAFETY: Caller passed writable `IGameInputDevice**`.
        unsafe { *out_device = ptr::null_mut() };
    }
}

extern "win64" fn gi_read_ret_false(_this: *mut ComGameInputReading, _out: *mut c_void) -> bool {
    false
}

extern "win64" fn gi_read_ret_zero_u32(_this: *mut ComGameInputReading) -> u32 {
    0
}

extern "win64" fn gi_read_get_gamepad_state(
    this: *mut ComGameInputReading,
    out_state: *mut GameInputGamepadState,
) -> bool {
    if this.is_null() || out_state.is_null() {
        return false;
    }
    // SAFETY: `this` is live and `out_state` is caller's writable `GameInputGamepadState*`.
    unsafe {
        *out_state = (*this).gamepad;
    }
    true
}

extern "win64" fn gi_get_current_reading(
    this: *mut ComGameInput,
    kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if out_reading.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller provided writable output reading pointer.
    unsafe { *out_reading = ptr::null_mut() };
    if this.is_null() {
        return E_POINTER;
    }
    if kind != 0 && (kind & GAMEINPUT_KIND_GAMEPAD) == 0 {
        return GAMEINPUT_E_READING_NOT_FOUND;
    }
    // SAFETY: `this` is a live `ComGameInput` holding `Arc<GameInputState>`.
    let state = unsafe { Arc::clone(&(*this).state) };
    let polled = exec_native(move || {
        let gp = state.current_gamepad_state()?;
        Some((gp, state.current_timestamp_us()))
    });
    let Some((gamepad, timestamp_us)) = polled else {
        return GAMEINPUT_E_READING_NOT_FOUND;
    };
    let Some(vtables) = GDK_VTABLES.get() else {
        return E_FAIL;
    };
    let seq = GI_READING_SEQ.fetch_add(1, Ordering::Relaxed);
    let obj = Box::into_raw(Box::new(ComGameInputReading {
        vtbl: vtables.gameinput_reading.as_ptr(),
        ref_count: AtomicU32::new(1),
        timestamp_us,
        sequence: seq,
        gamepad,
    }));
    // SAFETY: `out_reading` was validated non-null.
    unsafe {
        *out_reading = obj.cast::<c_void>();
    }
    S_OK
}

extern "win64" fn gi_get_next_reading(
    _this: *mut ComGameInput,
    _ref_reading: *mut c_void,
    _kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if !out_reading.is_null() {
        // SAFETY: Caller provided writable output reading pointer.
        unsafe { *out_reading = ptr::null_mut() };
    }
    GAMEINPUT_E_READING_NOT_FOUND
}

extern "win64" fn gi_get_previous_reading(
    _this: *mut ComGameInput,
    _ref_reading: *mut c_void,
    _kind: u32,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if !out_reading.is_null() {
        // SAFETY: Caller provided writable output reading pointer.
        unsafe { *out_reading = ptr::null_mut() };
    }
    GAMEINPUT_E_READING_NOT_FOUND
}

extern "win64" fn gi_get_temporal_reading(
    _this: *mut ComGameInput,
    _timestamp: u64,
    _device: *mut c_void,
    out_reading: *mut *mut c_void,
) -> i32 {
    if !out_reading.is_null() {
        // SAFETY: Caller provided writable output reading pointer.
        unsafe { *out_reading = ptr::null_mut() };
    }
    GAMEINPUT_E_READING_NOT_FOUND
}

extern "win64" fn gi_register_reading_cb(
    this: *mut ComGameInput,
    _device: *mut c_void,
    _kind: u32,
    _threshold: f32,
    _ctx: *mut c_void,
    _cb: *mut c_void,
    out_token: *mut u64,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `out_token` is optional writable output.
    unsafe {
        let tok = (*this).state.register_callback_token();
        if !out_token.is_null() {
            *out_token = tok;
        }
    }
    S_OK
}

extern "win64" fn gi_register_device_cb(
    this: *mut ComGameInput,
    _device: *mut c_void,
    _kind: u32,
    _status: u32,
    _enum_kind: u32,
    _ctx: *mut c_void,
    _cb: *mut c_void,
    out_token: *mut u64,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `out_token` is optional writable output.
    unsafe {
        let tok = (*this).state.register_callback_token();
        if !out_token.is_null() {
            *out_token = tok;
        }
    }
    S_OK
}

extern "win64" fn gi_register_sysbtn_cb(
    this: *mut ComGameInput,
    _device: *mut c_void,
    _buttons: u32,
    _ctx: *mut c_void,
    _cb: *mut c_void,
    out_token: *mut u64,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `out_token` is optional writable output.
    unsafe {
        let tok = (*this).state.register_callback_token();
        if !out_token.is_null() {
            *out_token = tok;
        }
    }
    S_OK
}

extern "win64" fn gi_register_kbd_cb(
    this: *mut ComGameInput,
    _device: *mut c_void,
    _ctx: *mut c_void,
    _cb: *mut c_void,
    out_token: *mut u64,
) -> i32 {
    if this.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is live; `out_token` is optional writable output.
    unsafe {
        let tok = (*this).state.register_callback_token();
        if !out_token.is_null() {
            *out_token = tok;
        }
    }
    S_OK
}

extern "win64" fn gi_stop_callback(this: *mut ComGameInput, token: u64) {
    if !this.is_null() {
        // SAFETY: `this` is live.
        unsafe {
            let _ = (*this).state.unregister_callback_token(token);
        }
    }
}

extern "win64" fn gi_unregister_callback(
    this: *mut ComGameInput,
    token: u64,
    _timeout_us: u64,
) -> bool {
    if this.is_null() {
        return false;
    }
    // SAFETY: `this` is live.
    unsafe { (*this).state.unregister_callback_token(token) }
}

extern "win64" fn gi_find_device_not_found(
    _this: *mut ComGameInput,
    _key: *const c_void,
    out_dev: *mut *mut c_void,
) -> i32 {
    if !out_dev.is_null() {
        // SAFETY: Caller provided writable output device pointer.
        unsafe { *out_dev = ptr::null_mut() };
    }
    GAMEINPUT_E_DEVICE_NOT_FOUND
}

extern "win64" fn gi_enable_oem_device_support(
    _this: *mut ComGameInput,
    _vid: u16,
    _pid: u16,
    _iface: u8,
    _coll: u8,
) -> i32 {
    S_OK
}

extern "win64" fn gi_set_focus_policy(this: *mut ComGameInput, policy: u32) {
    if !this.is_null() {
        // SAFETY: `this` is live.
        unsafe { (*this).state.set_focus_policy(policy) };
    }
}

// ============================================================================
// 4. AcpHal COM object & PSM AppNotify cookie registry
// ============================================================================

pub const OP_CHAD_DECOMPRESS: u32 = 0x0725;
pub const OP_XAPU_CONNECT_DECODER: u32 = 0x0715;
pub const OP_XAPU_DISCONNECT_DECODER: u32 = 0x0716;
pub const OP_XAPU_SUBMIT: u32 = 0x0717;
pub const OP_XAPU_POP: u32 = 0x0718;
pub const OP_XAPU_RESULT_COUNTS: u32 = 0x0719;
pub const OP_CHAD_CONNECT_STREAM_LIMIT: u32 = 0x0726;
pub const OP_CHAD_ACTIVATE_STATUS: u32 = 0x0727;
pub const OP_CHAD_SUBMIT_STATUS: u32 = 0x0728;

#[repr(C)]
pub struct ChadDecompressCommand {
    codec: u32,
    source_len: u32,
    source: *const u8,
    destination_len: u32,
    destination: *mut u8,
    bytes_written: *mut u32,
}

fn decompress_chad(codec: u32, source: &[u8], destination_len: usize) -> Result<Vec<u8>, i32> {
    match codec {
        0 => decompress_zlib_bounded(source, destination_len).map_err(|_| E_FAIL),
        1 | 3..=7 => {
            static TABLES: OnceLock<Result<bcpack::Tables, bcpack::Error>> = OnceLock::new();
            let tables = TABLES
                .get_or_init(bcpack::Tables::builtin)
                .as_ref()
                .map_err(|_| E_FAIL)?;
            let mut destination = vec![0; destination_len];
            bcpack::decode(codec as u8, source, &mut destination, tables).map_err(|_| E_FAIL)?;
            Ok(destination)
        }
        _ => Err(E_INVALIDARG),
    }
}

/// # Safety
/// The command and its input and output buffers remain live through synchronous completion.
pub unsafe fn chad_decompress(
    audio: &x2l_audio::AudioCoprocState,
    stream: u64,
    command: *const ChadDecompressCommand,
    out_ticket: *mut u32,
) -> i32 {
    if command.is_null() {
        return E_POINTER;
    }
    // SAFETY: The extension forwards the caller's readable wire descriptor.
    let command = unsafe { command.read_unaligned() };
    let Some((ticket, result)) = audio.chad_execute(stream, || {
        const MAX_BYTES: u32 = 64 * 1024 * 1024;
        if !command.bytes_written.is_null() {
            // SAFETY: The descriptor supplies an optional writable count field.
            unsafe { command.bytes_written.write_unaligned(0) };
        }
        if command.source.is_null() || command.destination.is_null() {
            return E_POINTER;
        }
        if command.source_len == 0
            || command.source_len > MAX_BYTES
            || command.destination_len > MAX_BYTES
        {
            return E_INVALIDARG;
        }
        // SAFETY: The caller supplies a live source range of the declared length.
        let source =
            unsafe { std::slice::from_raw_parts(command.source, command.source_len as usize) };
        let decoded = match decompress_chad(command.codec, source, command.destination_len as usize)
        {
            Ok(decoded) => decoded,
            Err(hr) => return hr,
        };
        // SAFETY: The decoded output is owned separately and fits the caller's destination.
        unsafe { ptr::copy_nonoverlapping(decoded.as_ptr(), command.destination, decoded.len()) };
        if !command.bytes_written.is_null() {
            // SAFETY: The optional count output is writable.
            unsafe { command.bytes_written.write_unaligned(decoded.len() as u32) };
        }
        S_OK
    }) else {
        return E_INVALIDARG;
    };
    if !out_ticket.is_null() {
        // SAFETY: The extension supplies a writable ticket pointer.
        unsafe { out_ticket.write_unaligned(ticket) };
    }
    result
}

const PCM_PACKET: u32 = 0x5841_5055;
const MAX_PCM_SAMPLES: u32 = 16_384;

#[repr(C)]
struct XApuConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    memory: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C)]
struct XApuConnectOutput {
    memory: usize,
    result_event: usize,
}

#[repr(C, packed)]
struct XApuProcess {
    id: XApuCommandId,
    pitch_ramp: f32,
    pitch: f32,
    first_frame: u32,
    frame_count: u32,
    input_length: u32,
    input: *const u8,
    output_length: u32,
    output: *mut u8,
}

type XApuSignal = extern "win64" fn(usize);

/// # Safety
/// Non-null input and output point to the XApu connection descriptors and handle output.
pub unsafe fn xapu_connect(
    audio: &x2l_audio::AudioCoprocState,
    input: *const c_void,
    output: *mut c_void,
    handle: *mut u64,
    event: usize,
) -> i32 {
    if handle.is_null() {
        return E_POINTER;
    }
    // SAFETY: The live handle output is writable.
    unsafe { handle.write_unaligned(0) };
    if input.is_null() && output.is_null() {
        if let Some(id) = audio.xapu_connect() {
            // SAFETY: The legacy caller provides the handle output.
            unsafe { handle.write_unaligned(id) };
            return S_OK;
        }
        return E_FAIL;
    }
    if input.is_null() || output.is_null() {
        return E_POINTER;
    }
    // SAFETY: The typed connection input is readable for this call.
    let input = unsafe { input.cast::<XApuConnectInput>().read_unaligned() };
    let output = output.cast::<XApuConnectOutput>();
    // SAFETY: The caller supplies the connection output descriptor.
    unsafe {
        output.write_unaligned(XApuConnectOutput {
            memory: 0,
            result_event: 0,
        })
    };
    if input.process != 1 {
        return super::HRESULT_E_NOTIMPL as i32;
    }
    if input.frames_per_output != 0 || (input.options & 8 != 0 && event == 0) {
        return E_INVALIDARG;
    }
    let Some((id, memory)) = audio.xapu_connect_decoder(
        input.streams,
        input.commands_per_stream,
        input.memory,
        event,
        input.options & 4 != 0,
    ) else {
        return E_INVALIDARG;
    };
    // SAFETY: Both caller outputs are initialized with the live connection.
    unsafe {
        output.write_unaligned(XApuConnectOutput {
            memory,
            result_event: event,
        });
        handle.write_unaligned(id);
    }
    S_OK
}

/// # Safety
/// The native command supplies an ID and the payload implied by its kind. The output
/// counter and event callback, when supplied, remain valid through completion.
pub unsafe fn xapu_enqueue(
    audio: &x2l_audio::AudioCoprocState,
    conn: u64,
    packet: *const c_void,
    out_total: *mut u32,
    wake_event: Option<XApuSignal>,
) -> i32 {
    let mut request_wake = false;
    let result = match audio.xapu_is_decoder(conn) {
        // SAFETY: This connection accepts the legacy PCM packet or a null marker.
        Some(false) => unsafe {
            xapu_submit_packet(audio, conn, packet.cast()).ok_or(E_INVALIDARG)
        },
        Some(true) if !packet.is_null() => {
            // SAFETY: The native command starts with the packed command ID.
            let id = unsafe { packet.cast::<XApuCommandId>().read_unaligned() };
            request_wake = id.options & 1 != 0;
            match id.kind {
                1 => {
                    // SAFETY: Activate places its channel count after the packed command ID.
                    let channels = unsafe {
                        packet
                            .cast::<u8>()
                            .add(size_of::<XApuCommandId>())
                            .cast::<u32>()
                            .read_unaligned()
                    };
                    exec_native(|| {
                        audio.xapu_submit_decoder(conn, id, XApuCommand::Activate(channels))
                    })
                }
                2 => exec_native(|| audio.xapu_submit_decoder(conn, id, XApuCommand::Deactivate)),
                // SAFETY: Process carries the packet and both live buffer ranges in its descriptor.
                3 => unsafe { xapu_enqueue_process(audio, conn, id, packet.cast()) },
                4 => audio.xapu_submit_decoder(conn, id, XApuCommand::Notify),
                5 => exec_native(|| audio.xapu_submit_decoder(conn, id, XApuCommand::Reset)),
                _ => Err(XAPU_INVALID_COMMAND),
            }
        }
        _ => Err(E_INVALIDARG),
    };
    match result {
        Ok(total) => {
            if !out_total.is_null() {
                // SAFETY: The caller supplies an optional writable counter.
                unsafe { out_total.write_unaligned(total) };
            }
            if request_wake && let Some(wake_event) = wake_event {
                audio.xapu_signal_decoder(conn, |event| wake_event(event));
            }
            S_OK
        }
        Err(hr) => hr,
    }
}

/// # Safety
/// The process descriptor and ranges are readable/writable for this synchronous command.
unsafe fn xapu_enqueue_process(
    audio: &x2l_audio::AudioCoprocState,
    conn: u64,
    id: XApuCommandId,
    process: *const XApuProcess,
) -> Result<u32, i32> {
    // SAFETY: Command kind 3 guarantees the packed process descriptor is readable.
    let process = unsafe { process.read_unaligned() };
    let input_length = process.input_length;
    if process.input.is_null() || process.output.is_null() || !(1..=65_536).contains(&input_length)
    {
        return Err(E_INVALIDARG);
    }
    // SAFETY: The caller supplies the encoded packet. Copy it before borrowing the output,
    // since the two guest regions can overlap.
    let input =
        unsafe { std::slice::from_raw_parts(process.input, input_length as usize) }.to_vec();
    // SAFETY: The output is writable for the declared length. Opus can write at most 120 ms stereo.
    let output = unsafe {
        std::slice::from_raw_parts_mut(process.output, process.output_length.min(5760 * 8) as usize)
    };
    let input_address = process.input as usize;
    let first_frame = process.first_frame;
    let frames = process.frame_count;
    exec_native(move || {
        audio.xapu_submit_decoder(
            conn,
            id,
            XApuCommand::Process {
                input_address,
                input: &input,
                output,
                first_frame,
                frames,
            },
        )
    })
}

/// # Safety
/// A non-null output is writable for the connection's result representation.
pub unsafe fn xapu_dequeue(
    audio: &x2l_audio::AudioCoprocState,
    conn: u64,
    output: *mut c_void,
) -> i32 {
    match audio.xapu_is_decoder(conn) {
        Some(true) if !output.is_null() => {
            let Some(result) = audio.xapu_dequeue_decoder(conn) else {
                return XAPU_PENDING_RESULTS;
            };
            // SAFETY: A native connection supplies the packed XApuResult buffer.
            unsafe { output.cast::<XApuResult>().write_unaligned(result) };
            S_OK
        }
        Some(false) => {
            let Some(ticket) = audio.xapu_dequeue_result(conn) else {
                return 1;
            };
            if !output.is_null() {
                // SAFETY: A legacy connection supplies an optional u32 output.
                unsafe { output.cast::<u32>().write_unaligned(ticket) };
            }
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

fn audio_output() -> &'static AudioOutput {
    super::runtime().audio_output.get_or_init(|| {
        let prefer_pipewire = std::env::var("X2L_AUDIO_BACKEND")
            .map(|v| !v.eq_ignore_ascii_case("null"))
            .unwrap_or(true);
        exec_native(move || AudioOutput::open(DEFAULT_SAMPLE_RATE_HZ, prefer_pipewire))
    })
}

/// # Safety
/// A non-null packet provides a readable command header and, for a PCM tag, its sample payload.
unsafe fn xapu_submit_packet(
    audio: &x2l_audio::AudioCoprocState,
    conn: u64,
    packet: *const u32,
) -> Option<u32> {
    if packet.is_null() {
        return audio.xapu_enqueue_command(conn);
    }
    // SAFETY: Non-null commands start with a readable tag.
    if unsafe { packet.read_unaligned() } != PCM_PACKET {
        return None;
    }
    // SAFETY: The tag identifies a PCM packet with a count and interleaved sample data.
    let samples = unsafe { read_pcm_packet(packet) }?;
    audio.xapu_submit_pcm(conn, audio_output().ring(), &samples)
}

/// # Safety
/// The tagged packet provides the two-word header and declared sample payload.
unsafe fn read_pcm_packet(packet: *const u32) -> Option<Vec<f32>> {
    // SAFETY: The tagged packet has a readable sample count after its tag.
    let count = unsafe { packet.add(1).read_unaligned() };
    if count == 0 || count > MAX_PCM_SAMPLES || !count.is_multiple_of(2) {
        return None;
    }
    Some(
        (0..count as usize)
            .map(|i| {
                // SAFETY: The packet supplies the validated count of sample words.
                unsafe { f32::from_bits(packet.add(2 + i).read_unaligned()) }
            })
            .collect(),
    )
}

#[repr(C)]
pub struct ComAcpHal {
    pub vtbl: std::sync::atomic::AtomicPtr<usize>,
    ring: Mutex<AcpCommandRing>,
}

extern "win64" fn acp_connect(this: *mut ComAcpHal, commands: u32, messages: u32) -> i32 {
    // SAFETY: The caller owns the live interface until Release.
    let Some(this) = (unsafe { this.as_ref() }) else {
        return E_POINTER;
    };
    let Ok(mut ring) = this.ring.lock() else {
        return E_FAIL;
    };
    if !ring.connect(commands, messages) {
        return E_INVALIDARG;
    }
    // The earlier synthetic fixture releases an unused interface via slot 2.
    let vtables = GDK_VTABLES.get().expect("initialized in AcpHalCreate");
    this.vtbl.store(
        vtables.acphal_connected.as_ptr().cast_mut(),
        Ordering::Release,
    );
    S_OK
}

extern "win64" fn acp_disconnect(this: *mut ComAcpHal) -> i32 {
    // SAFETY: The caller owns the live interface until Release.
    let Some(this) = (unsafe { this.as_ref() }) else {
        return E_POINTER;
    };
    let Ok(mut ring) = this.ring.lock() else {
        return E_FAIL;
    };
    if ring.disconnect() { S_OK } else { E_FAIL }
}

extern "win64" fn acp_submit(
    this: *mut ComAcpHal,
    kind: u32,
    id: u64,
    _frame: u32,
    data: *const c_void,
    notification: u32,
) -> i32 {
    // SAFETY: The caller owns the live interface until Release.
    let Some(this) = (unsafe { this.as_ref() }) else {
        return E_POINTER;
    };
    let Ok(mut ring) = this.ring.lock() else {
        return E_FAIL;
    };
    let word = if let 1..=3 = kind {
        if data.is_null() {
            return E_POINTER;
        }
        // SAFETY: Register/unregister carry one word. LoadFlowgraph carries its ID in word one.
        unsafe {
            data.cast::<u32>()
                .add(usize::from(kind == 1))
                .read_unaligned()
        }
    } else {
        0
    };
    if !ring.is_connected() {
        return E_FAIL;
    }
    if kind == PCM_PACKET {
        if data.is_null() {
            return E_POINTER;
        }
        // SAFETY: The PCM command supplies a tagged sample packet.
        let Some(samples) = (unsafe { read_pcm_packet(data.cast()) }) else {
            return E_INVALIDARG;
        };
        if super::runtime()
            .audio_coproc
            .acp_submit_pcm(audio_output().ring(), &samples)
            .is_none()
        {
            return E_FAIL;
        }
    }
    ring.submit(kind, id, word);
    if notification != 0 {
        super::runtime().audio_coproc.apu_notify(notification);
    }
    S_OK
}

extern "win64" fn acp_pop_message(this: *mut ComAcpHal, out: *mut AcpMessage) -> bool {
    if out.is_null() {
        return false;
    }
    // SAFETY: The caller owns the live interface until Release.
    let message = unsafe { this.as_ref() }.and_then(|this| this.ring.lock().ok()?.pop());
    if let Some(message) = message {
        // SAFETY: The caller supplies an AcpMessage output buffer.
        unsafe { out.write_unaligned(message) };
        return true;
    }
    false
}

extern "win64" fn acp_message_count(this: *mut ComAcpHal) -> u32 {
    // SAFETY: The caller owns the live interface until Release.
    unsafe { this.as_ref() }
        .and_then(|this| Some(this.ring.lock().ok()?.count()))
        .unwrap_or(0)
}

extern "win64" fn acp_release(this: *mut ComAcpHal) {
    if !this.is_null() {
        // SAFETY: The caller releases the Box from AcpHalCreate exactly once.
        unsafe { drop(Box::from_raw(this)) };
    }
}

extern "win64" fn acp_release_legacy(this: *mut ComAcpHal) -> u32 {
    acp_release(this);
    0
}

/// `AcpHalCreate(out_acp) -> HRESULT`.
///
/// # Safety
/// `out_acp` must be writable.
pub unsafe fn acphal_create(out_acp: *mut *mut c_void, probe_cb: VtableProbeCallback) -> i32 {
    if out_acp.is_null() {
        return E_POINTER;
    }
    let Some(vtables) = init_gdk_vtables(probe_cb) else {
        return E_FAIL;
    };
    let obj = Box::into_raw(Box::new(ComAcpHal {
        vtbl: std::sync::atomic::AtomicPtr::new(vtables.acphal.as_ptr().cast_mut()),
        ring: Mutex::new(AcpCommandRing::default()),
    }));
    // SAFETY: `out_acp` was validated non-null.
    unsafe {
        *out_acp = obj.cast::<c_void>();
    }
    S_OK
}

/// Process-local PSM AppNotify registration cookie tracker.
#[derive(Default)]
pub struct AppNotifyRegistry {
    next_cookie: AtomicU64,
    active: Mutex<BTreeMap<u64, (usize, usize)>>,
}

impl AppNotifyRegistry {
    pub fn new() -> Self {
        Self {
            next_cookie: AtomicU64::new(0x1000),
            active: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn register(&self, callback: usize, context: usize) -> u64 {
        let cookie = self.next_cookie.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut guard) = self.active.lock() {
            guard.insert(cookie, (callback, context));
        }
        cookie
    }

    pub fn unregister(&self, cookie: u64) -> bool {
        if let Ok(mut guard) = self.active.lock() {
            guard.remove(&cookie).is_some()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "win64" fn probe(_: u32, _: u64) -> u64 {
        panic!("test called an unimplemented slot");
    }

    fn u32_at(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn xapu_decoder_wire_ring_signals_and_echoes_native_results() {
        static SIGNALLED: AtomicU64 = AtomicU64::new(0);
        extern "win64" fn signal(event: usize) {
            SIGNALLED.store(event as u64, Ordering::Release);
        }
        let audio = x2l_audio::AudioCoprocState::new();
        let input = XApuConnectInput {
            process: 1,
            streams: 1,
            commands_per_stream: 1,
            memory: 4096,
            frames_per_output: 0,
            options: 8,
        };
        let mut output = XApuConnectOutput {
            memory: 0,
            result_event: 0,
        };
        let mut conn = 0;
        // SAFETY: The fixture passes both complete connection descriptors and a handle output.
        assert_eq!(
            unsafe {
                xapu_connect(
                    &audio,
                    (&raw const input).cast(),
                    (&raw mut output).cast(),
                    &mut conn,
                    123,
                )
            },
            S_OK
        );
        assert_eq!(output.memory % 65_536, 0);
        assert_ne!(output.memory, 0);
        assert_eq!(output.result_event, 123);
        let mut id = XApuCommandId {
            kind: 4,
            stream: 0,
            sequence: 25,
            options: 1,
            context: 42,
        };
        let mut writes = 0;
        // SAFETY: Notify consists of a command ID only and writes the counter.
        assert_eq!(
            unsafe {
                xapu_enqueue(
                    &audio,
                    conn,
                    (&raw const id).cast(),
                    &mut writes,
                    Some(signal),
                )
            },
            S_OK
        );
        assert_eq!(SIGNALLED.load(Ordering::Acquire), 123);
        assert_eq!(writes, 1);
        assert_eq!(audio.xapu_result_counts(conn), (1, 0));
        // SAFETY: The ID and counter remain live when testing backpressure.
        assert_eq!(
            unsafe { xapu_enqueue(&audio, conn, (&raw const id).cast(), &mut writes, None) },
            x2l_audio::coproc::XAPU_QUEUE_FULL
        );
        let mut result = std::mem::MaybeUninit::<XApuResult>::uninit();
        // SAFETY: The destination is storage for one complete native result.
        assert_eq!(
            unsafe { xapu_dequeue(&audio, conn, result.as_mut_ptr().cast()) },
            S_OK
        );
        // SAFETY: S_OK means the dequeue initialized the complete result.
        let result = unsafe { result.assume_init() };
        let (seq, ctx, error) = (result.id.sequence, result.id.context, result.error);
        assert_eq!((seq, ctx, error), (25, 42, 0));
        assert_eq!(audio.xapu_result_counts(conn), (0, 1));
        id.sequence += 1;
        // SAFETY: A dequeued slot is reusable and the command and output are still live.
        assert_eq!(
            unsafe { xapu_enqueue(&audio, conn, (&raw const id).cast(), &mut writes, None) },
            S_OK
        );
        assert_eq!(writes, 2);
        assert_eq!(audio.xapu_disconnect_with_event(conn), Some(123));
        assert!(!audio.apu_is_virtual_address_valid(output.memory, 1));
    }

    #[test]
    fn acphal_connect_selects_command_vtable_and_keeps_unused_legacy_release() {
        let mut raw = ptr::null_mut();
        // SAFETY: The fixture provides writable COM output storage.
        assert_eq!(unsafe { acphal_create(&mut raw, probe) }, S_OK);
        let this = raw.cast::<ComAcpHal>();
        // SAFETY: An unused object supports the legacy fixture's slot 2 release.
        let old_release = unsafe {
            std::mem::transmute::<usize, extern "win64" fn(*mut ComAcpHal) -> u32>(
                *(*this).vtbl.load(Ordering::Acquire).add(2),
            )
        };
        assert_eq!(old_release(this), 0);
        // SAFETY: The previous object was released and raw can receive a fresh one.
        assert_eq!(unsafe { acphal_create(&mut raw, probe) }, S_OK);
        let this = raw.cast::<ComAcpHal>();
        assert_eq!(acp_connect(this, 2, 2), S_OK);
        // SAFETY: After Connect, slot 2 is the command submission method.
        let submit = unsafe {
            std::mem::transmute::<
                usize,
                extern "win64" fn(*mut ComAcpHal, u32, u64, u32, *const c_void, u32) -> i32,
            >(*(*this).vtbl.load(Ordering::Acquire).add(2))
        };
        let mask = 0x10u32;
        assert_eq!(submit(this, 2, 0, 0, (&raw const mask).cast(), 0), S_OK);
        assert_eq!(submit(this, 1, 3, 0, [0u32, 17].as_ptr().cast(), 0), S_OK);
        assert_eq!(acp_message_count(this), 1);
        let mut result = [0u8; size_of::<AcpMessage>()];
        assert!(acp_pop_message(this, result.as_mut_ptr().cast()));
        assert_eq!(u32_at(&result, 0), 0x10);
        assert_eq!(u32_at(&result, 12), 1);
        assert_eq!(u64_at(&result, 16), 3);
        assert_eq!(acp_message_count(this), 0);
        assert_eq!(acp_disconnect(this), S_OK);
        acp_release(this);
    }

    #[test]
    fn xg_2d_win64_layout_mips_and_copy_roundtrips() {
        for (format, swizzle, bind_flags, element_bytes, block, planes) in [
            (28, Swizzle::SW_64KB_S, 8, 4, 1, 1),
            (98, Swizzle::SW_64KB_D, 8, 16, 4, 1),
            (40, Swizzle::LINEAR, 0x40, 4, 1, 1),
            (40, Swizzle::SW_64KB_Z_X, 0x40, 4, 1, 1),
            (20, Swizzle::SW_64KB_D, 0x40, 4, 1, 2),
            (10, Swizzle::SW_64KB_R_X, 0x20, 8, 1, 1),
        ] {
            let desc = XgTexture2dDesc {
                width: 387,
                height: 265,
                mip_levels: 6,
                array_size: 2,
                format,
                samples: 1,
                sample_quality: 0,
                usage: 0,
                bind_flags,
                cpu_access_flags: 0,
                misc_flags: 0,
                swizzle: swizzle.0,
                pitch: 0,
            };
            let mut raw = ptr::null_mut();
            // SAFETY: The typed constructor reads the live 2D descriptor and writes raw.
            assert_eq!(
                unsafe { xg_create_typed_computer(3, (&raw const desc).cast(), &mut raw, probe) },
                S_OK
            );
            let this: *mut ComXgTextureComputer = raw.cast();
            // SAFETY: The created object's table owns these audited Win64 function entries.
            let (layout, size, alignment, mip_offset, tile, detile, release) = unsafe {
                let vtable = (*this).vtbl;
                (
                    std::mem::transmute::<
                        usize,
                        extern "win64" fn(*mut ComXgTextureComputer, *mut u8) -> i32,
                    >(*vtable.add(2)),
                    std::mem::transmute::<usize, extern "win64" fn(*mut ComXgTextureComputer) -> u64>(
                        *vtable.add(3),
                    ),
                    std::mem::transmute::<usize, extern "win64" fn(*mut ComXgTextureComputer) -> u64>(
                        *vtable.add(4),
                    ),
                    std::mem::transmute::<
                        usize,
                        extern "win64" fn(*mut ComXgTextureComputer, u32, u32) -> u64,
                    >(*vtable.add(5)),
                    std::mem::transmute::<
                        usize,
                        extern "win64" fn(
                            *mut ComXgTextureComputer,
                            *mut c_void,
                            u32,
                            u32,
                            *const c_void,
                            u32,
                            u32,
                        ) -> i32,
                    >(*vtable.add(8)),
                    std::mem::transmute::<
                        usize,
                        extern "win64" fn(
                            *mut ComXgTextureComputer,
                            *mut c_void,
                            u32,
                            u32,
                            u32,
                            *const c_void,
                            u32,
                            u32,
                        ) -> i32,
                    >(*vtable.add(9)),
                    std::mem::transmute::<usize, extern "win64" fn(*mut ComXgTextureComputer) -> u32>(
                        *vtable.add(1),
                    ),
                )
            };
            let mut wire = [0; XG_RESOURCE_LAYOUT_SIZE];
            assert_eq!(layout(this, wire.as_mut_ptr()), S_OK);
            assert_eq!(u64_at(&wire, 0), size(this));
            assert_ne!(size(this), 0);
            assert_eq!(u64_at(&wire, 8), alignment(this));
            assert_eq!(
                alignment(this),
                if swizzle == Swizzle::LINEAR {
                    256
                } else {
                    65_536
                }
            );
            assert_eq!(u32_at(&wire, 0x10), desc.mip_levels);
            assert_eq!(u32_at(&wire, 0x14), planes);
            let mut tiled = vec![0u8; size(this) as usize];
            for plane in 0..planes {
                let p = XG_PLANE_OFFSET + plane as usize * XG_PLANE_STRIDE;
                let bpe = if plane == 0 { element_bytes } else { 1 };
                assert_eq!(u32_at(&wire, p + 0x28), bpe);
                for mip in [0, 2, 5] {
                    let m = p + XG_MIP_OFFSET_IN_PLANE + mip as usize * XG_MIP_STRIDE;
                    let width = (desc.width >> mip).div_ceil(block);
                    let height = (desc.height >> mip).div_ceil(block);
                    let pitch = width * bpe;
                    assert_eq!(u32_at(&wire, m + 0x30), width);
                    assert_eq!(u32_at(&wire, m + 0x34), height);
                    assert!(u32_at(&wire, m + 0x1c) >= pitch);
                    assert!(u64_at(&wire, m + 0x10) >= u64::from(pitch) * u64::from(height));
                    let offset = mip_offset(this, plane, mip);
                    assert_eq!(offset, u64_at(&wire, p + 0x18) + u64_at(&wire, m + 8));
                    assert!(offset < size(this));
                    let source: Vec<u8> = (0..pitch * height)
                        .map(|n| n.wrapping_mul(29) as u8)
                        .collect();
                    let mut back = vec![0; source.len()];
                    let subresource = desc.mip_levels + mip;
                    assert_eq!(
                        tile(
                            this,
                            tiled.as_mut_ptr().cast(),
                            plane,
                            subresource,
                            source.as_ptr().cast(),
                            pitch,
                            0
                        ),
                        S_OK
                    );
                    assert_eq!(
                        detile(
                            this,
                            back.as_mut_ptr().cast(),
                            plane,
                            subresource,
                            0,
                            tiled.as_ptr().cast(),
                            pitch,
                            0
                        ),
                        S_OK
                    );
                    assert_eq!(back, source);
                }
            }
            assert_eq!(release(this), 0);
        }
    }

    #[test]
    fn xg_3d_copies_touch_all_slices_with_pitched_rows() {
        let desc = XgTexture3dDesc {
            width: 13,
            height: 7,
            depth: 5,
            mip_levels: 3,
            format: 28,
            usage: 0,
            bind_flags: 8,
            cpu_access_flags: 0,
            misc_flags: 0,
            swizzle: Swizzle::SW_64KB_S_X.0,
            pitch: 0,
        };
        let mut this = ptr::null_mut();
        // SAFETY: The constructor reads a live 3D descriptor and writes the object pointer.
        assert_eq!(
            unsafe { xg_create_typed_computer(4, (&raw const desc).cast(), &mut this, probe) },
            S_OK
        );
        let this = this.cast::<ComXgTextureComputer>();
        let mut tiled = vec![0; xg_get_resource_size_bytes(this) as usize];
        let mut input = [0u8; 7 + (7 - 1) * 64 + (5 - 1) * 456 + 13 * 4];
        let input = &mut input[7..];
        let mut output = vec![0xa5; input.len()];
        for z in 0..5 {
            for y in 0..7 {
                input[z * 456 + y * 64..][..13 * 4].fill((z * 7 + y + 1) as u8);
            }
        }
        assert_eq!(
            xg_copy_into_subresource(
                this,
                tiled.as_mut_ptr().cast(),
                0,
                0,
                input.as_ptr().cast(),
                64,
                456
            ),
            S_OK
        );
        assert_eq!(
            xg_copy_from_subresource(
                this,
                output.as_mut_ptr().cast(),
                0,
                0,
                0,
                tiled.as_ptr().cast(),
                64,
                456
            ),
            S_OK
        );
        for z in 0..5 {
            for y in 0..7 {
                let from = z * 456 + y * 64;
                assert_eq!(input[from..][..13 * 4], output[from..][..13 * 4]);
                if let Some(padding) = output.get(from + 13 * 4..from + 64) {
                    assert_eq!(padding, &[0xa5; 12]);
                }
            }
        }
        assert_eq!(
            xg_copy_into_subresource(
                this,
                tiled.as_mut_ptr().cast(),
                0,
                0,
                tiled.as_ptr().cast(),
                64,
                456
            ),
            E_INVALIDARG
        );
        assert_eq!(xg_release(this), 0);
    }

    #[test]
    fn chad_decodes_zlib_and_bcpack_and_publishes_failures() {
        let audio = x2l_audio::AudioCoprocState::new();
        let dsp = audio.chad_connect(1).unwrap();
        let (stream, status) = audio.chad_activate_with_status(dsp).unwrap();
        let zlib = [
            120, 156, 203, 72, 205, 201, 201, 215, 81, 72, 206, 72, 76, 1, 0, 24, 60, 3, 241,
        ];
        // One synthetic BC1 block with a zero symbol in each of its 12 entropy lanes.
        let bc1 = [
            1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 181, 181, 22, 45, 22, 22, 45, 22, 138,
            138, 225, 130, 225, 225, 130, 225, 0, 0, 10, 4, 10, 10, 4, 10, 103, 103, 209, 193, 209,
            209, 193, 209, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 3, 3, 3, 3, 181, 181, 181, 181, 138, 138, 138, 138, 0, 0, 0, 0, 103, 103,
            103, 103,
        ];
        let mut output = [0xa5; 16];
        let mut ticket = 0;
        let mut written = 0;
        for (codec, input, length, result) in [
            (0, zlib.as_slice(), 16, S_OK),
            (1, bc1.as_slice(), 8, S_OK),
            (1, &bc1[..8], 8, E_FAIL),
        ] {
            let desc = ChadDecompressCommand {
                codec,
                source_len: input.len() as u32,
                source: input.as_ptr(),
                destination_len: length,
                destination: output.as_mut_ptr(),
                bytes_written: &mut written,
            };
            let expected_ticket = ticket + 1;
            // SAFETY: All buffers and outputs in the descriptor are live for the synchronous submit.
            assert_eq!(
                unsafe { chad_decompress(&audio, stream, &desc, &mut ticket) },
                result
            );
            assert_eq!(ticket, expected_ticket);
            assert_eq!(status.completion(), (ticket, result));
            match ticket {
                1 => assert_eq!(&output[..written as usize], b"hello, chad"),
                2 => assert_eq!(
                    &output[..written as usize],
                    &[0, 0, 0, 0, 0xaa, 0xaa, 0xaa, 0xaa]
                ),
                _ => assert_eq!(written, 0),
            }
        }
    }
}
