// Win64 COM vtable layouts and struct offsets informed by Marvisak/VIctory
// (https://github.com/Marvisak/VIctory, commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-hle/src/modules/{xg,dstorage,gameinput,lifecycle}.rs), xcl-dev/xcl (commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/{xg,dstorage,gameinput}.rs),
// and public Microsoft GDK DirectStorage / GameInput / AppNotify specifications.
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: native Linux Win64 COM objects backed by `x2l-formats`,
// `x2l-input`, and W^X spare-slot probe thunks (`x2l-gpu-vk`).

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
use x2l_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
use x2l_formats::{bcpack, decompress_zlib_bounded};
use x2l_gpu_vk::com::{D3D12ResourceDesc, E_FAIL, E_INVALIDARG, E_POINTER, S_OK, exec_native};
use x2l_gpu_vk::{VtableProbeCallback, build_probe_vtable_table};
use x2l_input::{GAMEINPUT_E_DEVICE_NOT_FOUND, GAMEINPUT_E_READING_NOT_FOUND, GameInputState};

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
    acphal: &'static [usize],
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

        let acphal = build_probe_vtable_table(
            24,
            probe_cb,
            &[
                (0, acp_qi as *const () as usize),
                (1, acp_add_ref as *const () as usize),
                (2, acp_release as *const () as usize),
            ],
        )?;

        Ok(GdkComVtables {
            xg_computer,
            dstorage_factory,
            dstorage_queue,
            dstorage_file,
            dstorage_status,
            gameinput,
            acphal,
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

#[repr(C)]
pub struct ComXgTextureComputer {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    dimension: u32,
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
    let d = unsafe { &*desc };
    let mode = match d.dimension {
        1 => 0u32,                               // BUFFER -> LINEAR
        2 => 0x100 | Swizzle::SW_4KB_S.0,        // TEXTURE1D
        3 | 4 => 0x100 | Swizzle::SW_64KB_S_X.0, // TEXTURE2D / TEXTURE3D -> standard 64 KiB RDNA2
        _ => return E_INVALIDARG,
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
    let Some(vtables) = init_gdk_vtables(probe_cb) else {
        return E_FAIL;
    };
    // SAFETY: Caller passed a readable 56-byte `D3D12ResourceDesc`.
    let d = unsafe { &*desc };
    let dim = match d.dimension {
        2 => Dimension::Tex1d,
        3 => Dimension::Tex2d,
        4 => Dimension::Tex3d,
        _ => return E_INVALIDARG,
    };
    let swizzle = if (d.layout & 0x100) != 0 {
        Swizzle(d.layout & 0xFF)
    } else if d.layout == 1 {
        Swizzle::LINEAR
    } else {
        Swizzle::SW_64KB_S_X
    };
    let tex_desc = TextureDesc {
        dimension: dim,
        width: (d.width as u32).max(1),
        height: if dim == Dimension::Tex1d {
            1
        } else {
            d.height.max(1)
        },
        depth_or_array: u32::from(d.depth_or_array_size.max(1)),
        mips: u32::from(d.mip_levels),
        format: if d.format == 0 { 28 } else { d.format },
        samples: d.sample_desc.count.max(1),
        swizzle,
    };
    let Ok(texture) = Texture::new(&tex_desc) else {
        return E_INVALIDARG;
    };
    let obj = Box::into_raw(Box::new(ComXgTextureComputer {
        vtbl: vtables.xg_computer.as_ptr(),
        ref_count: AtomicU32::new(1),
        dimension: d.dimension,
        texture,
    }));
    // SAFETY: `out_computer` is validated non-null.
    unsafe {
        *out_computer = obj.cast::<c_void>();
    }
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
            write_u32(buf, p_off, if plane.stencil { 2 } else { 0 });
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
            for (mi, mip) in lay.mips.iter().take(15).enumerate() {
                let m_off = p_off + XG_MIP_OFFSET_IN_PLANE + mi * XG_MIP_STRIDE;
                let mip_bytes = (mip.pitch as u64)
                    .saturating_mul(mip.rows as u64)
                    .saturating_mul(lay.desc.element_bytes as u64);
                write_u64(buf, m_off, mip_bytes);
                write_u64(buf, m_off + 0x08, mip.offset);
                write_u64(buf, m_off + 0x10, mip_bytes);
                write_u32(buf, m_off + 0x18, mip.pitch * lay.desc.block);
                write_u32(buf, m_off + 0x1C, mip.pitch * lay.desc.element_bytes);
                write_u32(buf, m_off + 0x20, lay.alignment as u32);
                write_u32(buf, m_off + 0x24, mip.pitch * lay.desc.block);
                write_u32(buf, m_off + 0x28, mip.rows * lay.desc.block);
                write_u32(buf, m_off + 0x2C, mip.depth_pitch);
                write_u32(buf, m_off + 0x30, mip.width);
                write_u32(buf, m_off + 0x34, mip.height);
                write_u32(buf, m_off + 0x38, mip.width * lay.desc.block);
                write_u32(buf, m_off + 0x3C, mip.height * lay.desc.block);
                write_u32(buf, m_off + 0x40, mip.depth);
                write_u32(buf, m_off + 0x44, lay.desc.samples);
                write_u32(buf, m_off + 0x48, lay.desc.swizzle.0);
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
        p.offset.saturating_add(m.offset)
    }
}

extern "win64" fn xg_get_texel_element_offset_bytes(
    this: *mut ComXgTextureComputer,
    plane: u32,
    mip: u32,
    x: u64,
    y: u32,
    z_or_slice: u32,
    _sample: u32,
    bit_offset: *mut u32,
) -> u64 {
    if !bit_offset.is_null() {
        // SAFETY: Caller supplied writable optional `bit_offset`.
        unsafe { *bit_offset = 0 };
    }
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComXgTextureComputer`.
    unsafe {
        let this_ref = &*this;
        let Some(p) = this_ref.texture.planes.get(plane as usize) else {
            return 0;
        };
        p.layout
            .offset(mip, x as u32, y, z_or_slice)
            .map(|off| p.offset.saturating_add(off))
            .unwrap_or(0)
    }
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
        if mips == 0 {
            return E_INVALIDARG;
        }
        let mip = subresource % mips;
        let slice = subresource / mips;
        let tiled_slice = std::slice::from_raw_parts_mut(
            tiled.cast::<u8>().add(p.offset as usize),
            p.layout.size as usize,
        );
        let linear_len = (slice_pitch as usize).max(
            (row_pitch as usize).saturating_mul(
                p.layout
                    .mips
                    .get(mip as usize)
                    .map(|m| m.height as usize)
                    .unwrap_or(1),
            ),
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
    _sample: u32,
    tiled: *const c_void,
    row_pitch: u32,
    slice_pitch: u32,
) -> i32 {
    if this.is_null() || tiled.is_null() || linear.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller supplies valid tiled source and linear destination buffers.
    unsafe {
        let this_ref = &*this;
        let Some(p) = this_ref.texture.planes.get(plane as usize) else {
            return E_INVALIDARG;
        };
        let mips = p.layout.mips.len() as u32;
        if mips == 0 {
            return E_INVALIDARG;
        }
        let mip = subresource % mips;
        let slice = subresource / mips;
        let tiled_slice = std::slice::from_raw_parts(
            tiled.cast::<u8>().add(p.offset as usize),
            p.layout.size as usize,
        );
        let linear_len = (slice_pitch as usize).max(
            (row_pitch as usize).saturating_mul(
                p.layout
                    .mips
                    .get(mip as usize)
                    .map(|m| m.height as usize)
                    .unwrap_or(1),
            ),
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
}

#[repr(C)]
pub struct ComDStorageQueue {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    pending: Mutex<Vec<QueuedDsOp>>,
    closed: AtomicBool,
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

extern "win64" fn ds_q_enqueue_signal(
    _this: *mut ComDStorageQueue,
    _fence: *mut c_void,
    _value: u64,
) {
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
                QueuedDsOp::Status { .. } => true,
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

extern "win64" fn gi_get_current_reading(
    _this: *mut ComGameInput,
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

#[repr(C)]
pub struct ComAcpHal {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
}

impl_iunknown!(ComAcpHal, acp_qi, acp_add_ref, acp_release);

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
        vtbl: vtables.acphal.as_ptr(),
        ref_count: AtomicU32::new(1),
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
