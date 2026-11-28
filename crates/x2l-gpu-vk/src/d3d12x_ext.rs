// Console `D3D12.X` entrypoint ABI layouts informed by Marvisak/VIctory
// (https://github.com/Marvisak/VIctory, commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-gpu/src/{device,root}.rs), xcl-dev/xcl (commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/d3d12x/mod.rs), and public
// MIT Microsoft DirectX-Headers (`d3d12.idl`, `d3dcommon.idl`).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: `ID3DBlob` COM object, `D3D12ConfigureVirtualMemoryX`,
// and DXBC `RTS0` versioned root-signature serializer.

//! Additional `d3d12_xs.dll` Xbox entrypoints: `D3D12ConfigureVirtualMemoryX`,
//! `D3D12SerializeVersionedRootSignature`, and ref-counted Win64 COM `ID3DBlob`.

use crate::com::{D3D12InputLayoutDesc, D3D12ShaderBytecode, E_INVALIDARG, E_POINTER, S_OK};
use std::ffi::c_void;
use std::mem::{MaybeUninit, offset_of, size_of};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// Win64 descriptor for console device slot 86.
#[repr(C)]
pub struct D3D12PipelineStateStreamDesc {
    pub size_in_bytes: usize,
    pub stream: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RenderTargetFormats {
    formats: [MaybeUninit<u32>; 8],
    count: u32,
}

impl RenderTargetFormats {
    /// # Safety
    /// The first `count` format entries must be initialized.
    pub(crate) unsafe fn active_formats(&self) -> Result<Vec<u32>, i32> {
        let active = self
            .formats
            .get(..self.count as usize)
            .ok_or(E_INVALIDARG)?;
        Ok(active
            .iter()
            // SAFETY: The caller guarantees initialization of the active entries.
            .map(|format| unsafe { format.assume_init() })
            .collect())
    }
}

#[derive(Default)]
pub(crate) struct PipelineStream {
    pub root_signature: Option<*const c_void>,
    pub vs: Option<D3D12ShaderBytecode>,
    pub ps: Option<D3D12ShaderBytecode>,
    pub cs: Option<D3D12ShaderBytecode>,
    pub ms: Option<D3D12ShaderBytecode>,
    pub amplification: Option<D3D12ShaderBytecode>,
    pub input_layout: Option<D3D12InputLayoutDesc>,
    pub target_formats: Option<RenderTargetFormats>,
    pub depth_format: Option<u32>,
    pub topology: Option<u32>,
}

/// Read the payload with its own ABI alignment, then advance to the next pointer-aligned tag.
///
/// # Safety
/// The stream must be readable for `remaining` bytes, and the payload representation must
/// permit all bit patterns. Padding in the guest stream may be uninitialized.
unsafe fn read_subobject<T: Copy>(stream: *const u8, remaining: &mut usize) -> Result<T, i32> {
    let payload = size_of::<u32>().next_multiple_of(std::mem::align_of::<T>());
    let stride = (payload + size_of::<T>()).next_multiple_of(size_of::<usize>());
    if *remaining < stride {
        return Err(E_INVALIDARG);
    }
    *remaining -= stride;
    // SAFETY: The stride check covers the entire payload. The caller establishes its representation.
    Ok(unsafe { stream.add(payload).cast::<T>().read_unaligned() })
}

/// # Safety
/// The descriptor must identify a readable stream of its declared size.
pub(crate) unsafe fn parse_pipeline_stream(
    desc: &D3D12PipelineStateStreamDesc,
) -> Result<PipelineStream, i32> {
    let mut remaining = desc.size_in_bytes;
    if desc.stream.is_null()
        || !(desc.stream as usize).is_multiple_of(size_of::<usize>())
        || remaining == 0
        || remaining > isize::MAX as usize
        || !remaining.is_multiple_of(size_of::<usize>())
    {
        return Err(E_INVALIDARG);
    }
    let base = desc.stream.cast::<u8>();
    let mut parsed = PipelineStream::default();
    let mut seen = 0u32;
    while remaining > 0 {
        // SAFETY: Remaining is a positive multiple of eight and never exceeds the stream size.
        let entry = unsafe { base.add(desc.size_in_bytes - remaining) };
        macro_rules! read {
            ($field:ident) => {{
                // SAFETY: Each listed field has a C ABI payload of integers and raw pointers.
                parsed.$field = Some(unsafe { read_subobject(entry, &mut remaining)? });
            }};
        }
        // SAFETY: At least eight bytes remain, including the initialized four-byte tag.
        let tag = unsafe { entry.cast::<u32>().read_unaligned() };
        let bit = 1u32.checked_shl(tag).ok_or(E_INVALIDARG)?;
        if seen & bit != 0 {
            return Err(E_INVALIDARG);
        }
        seen |= bit;
        match tag {
            0 => read!(root_signature),
            1 => read!(vs),
            2 => read!(ps),
            6 => read!(cs),
            12 => read!(input_layout),
            14 => read!(topology),
            15 => read!(target_formats),
            16 => read!(depth_format),
            24 => read!(amplification),
            25 => read!(ms),
            _ => return Err(E_INVALIDARG),
        }
    }
    Ok(parsed)
}

/// 32-byte `D3D12ConfigureVirtualMemoryX` in/out parameter block (8-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct D3D12ConfigureVirtualMemoryXParams {
    pub flags_or_mode: u32,    // @0x00
    pub field_04: u32,         // @0x04
    pub field_08: u32,         // @0x08
    pub _pad0c: u32,           // @0x0C
    pub low_address_out: u64,  // @0x10
    pub high_address_out: u64, // @0x18
}

/// 48-byte Win64 `D3D12_VERSIONED_ROOT_SIGNATURE_DESC` (v1.0 = 1, v1.1 = 2).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12VersionedRootSignatureDesc {
    pub version: u32,                     // @0x00 (1 = v1.0, 2 = v1.1)
    pub _pad04: u32,                      // @0x04
    pub num_parameters: u32,              // @0x08
    pub _pad0c: u32,                      // @0x0C
    pub p_parameters: *const c_void,      // @0x10
    pub num_static_samplers: u32,         // @0x18
    pub _pad1c: u32,                      // @0x1C
    pub p_static_samplers: *const c_void, // @0x20
    pub flags: u32,                       // @0x28
    pub _pad2c: u32,                      // @0x2C
}

const _: () = assert!(size_of::<D3D12ConfigureVirtualMemoryXParams>() == 32);
const _: () = assert!(offset_of!(D3D12ConfigureVirtualMemoryXParams, low_address_out) == 0x10);
const _: () = assert!(offset_of!(D3D12ConfigureVirtualMemoryXParams, high_address_out) == 0x18);
const _: () = assert!(size_of::<D3D12VersionedRootSignatureDesc>() == 48);
const _: () = assert!(offset_of!(D3D12VersionedRootSignatureDesc, num_parameters) == 0x08);
const _: () = assert!(offset_of!(D3D12VersionedRootSignatureDesc, p_parameters) == 0x10);
const _: () = assert!(offset_of!(D3D12VersionedRootSignatureDesc, num_static_samplers) == 0x18);
const _: () = assert!(offset_of!(D3D12VersionedRootSignatureDesc, p_static_samplers) == 0x20);
const _: () = assert!(offset_of!(D3D12VersionedRootSignatureDesc, flags) == 0x28);

/// Populate safe default GPU virtual-address range bounds for `D3D12ConfigureVirtualMemoryX`.
///
/// # Safety
/// `params` may be null (handled safely) or point to a valid writable 32-byte struct.
pub unsafe fn configure_virtual_memory_x(params: *mut D3D12ConfigureVirtualMemoryXParams) -> i32 {
    if params.is_null() {
        return E_POINTER;
    }
    // SAFETY: Caller supplied a non-null writable 32-byte `D3D12ConfigureVirtualMemoryXParams`.
    unsafe {
        (*params).low_address_out = 0x0000_0001_0000_0000;
        (*params).high_address_out = 0x0000_0FFF_0000_0000;
    }
    S_OK
}

// ============================================================================
// Ref-counted Win64 COM `ID3DBlob` (`0..2 IUnknown`, `3 GetBufferPointer`, `4 GetBufferSize`)
// ============================================================================

#[repr(C)]
pub struct ComD3DBlob {
    pub vtbl: *const usize,
    ref_count: AtomicU32,
    data: Vec<u8>,
}

static D3D_BLOB_VTBL: OnceLock<[usize; 5]> = OnceLock::new();

fn d3d_blob_vtbl() -> *const usize {
    D3D_BLOB_VTBL
        .get_or_init(|| {
            [
                blob_query_interface as *const () as usize,
                blob_add_ref as *const () as usize,
                blob_release as *const () as usize,
                blob_get_buffer_pointer as *const () as usize,
                blob_get_buffer_size as *const () as usize,
            ]
        })
        .as_ptr()
}

/// Allocate a new Win64 COM `ID3DBlob` owning `data` with refcount = 1.
pub fn create_d3d_blob(data: Vec<u8>) -> *mut ComD3DBlob {
    Box::into_raw(Box::new(ComD3DBlob {
        vtbl: d3d_blob_vtbl(),
        ref_count: AtomicU32::new(1),
        data,
    }))
}

extern "win64" fn blob_query_interface(
    this: *mut ComD3DBlob,
    riid: *const c_void,
    out_obj: *mut *mut c_void,
) -> i32 {
    // SAFETY: COM supplies a live blob, readable IID and writable output when non-null.
    unsafe {
        crate::com_guid::query_interface(&crate::com_guid::BLOB, this, riid, out_obj, blob_add_ref)
    }
}

extern "win64" fn blob_add_ref(this: *mut ComD3DBlob) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComD3DBlob`.
    unsafe { (*this).ref_count.fetch_add(1, Ordering::AcqRel) + 1 }
}

extern "win64" fn blob_release(this: *mut ComD3DBlob) -> u32 {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` was allocated via `Box::into_raw` and freed once when refcount reaches 0.
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

extern "win64" fn blob_get_buffer_pointer(this: *mut ComD3DBlob) -> *mut c_void {
    if this.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: `this` is a live `ComD3DBlob`.
    unsafe { (*this).data.as_mut_ptr().cast::<c_void>() }
}

extern "win64" fn blob_get_buffer_size(this: *mut ComD3DBlob) -> usize {
    if this.is_null() {
        return 0;
    }
    // SAFETY: `this` is a live `ComD3DBlob`.
    unsafe { (*this).data.len() }
}

/// Normalized descriptor range inside a decoded D3D12 root descriptor table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedDescriptorRange {
    pub range_type: u32, // 0=SRV, 1=UAV, 2=CBV, 3=SAMPLER
    pub num_descriptors: u32,
    pub base_shader_register: u32,
    pub register_space: u32,
    pub flags: u32,
    pub offset_in_descriptors_from_table_start: u32,
}

/// Normalized D3D12 root parameter kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedRootParamKind {
    DescriptorTable(Vec<ParsedDescriptorRange>),
    Constants {
        shader_register: u32,
        register_space: u32,
        num_32bit_values: u32,
    },
    Cbv {
        shader_register: u32,
        register_space: u32,
        flags: u32,
    },
    Srv {
        shader_register: u32,
        register_space: u32,
        flags: u32,
    },
    Uav {
        shader_register: u32,
        register_space: u32,
        flags: u32,
    },
}

/// Normalized D3D12 root parameter with computed DWORD offset inside the 64-DWORD root table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedRootParameter {
    pub shader_visibility: u32,
    pub dword_offset: u32,
    pub kind: ParsedRootParamKind,
}

/// Decoded owned D3D12 root signature produced from a DXBC `RTS0` blob.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedRootSignature {
    pub version: u32,
    pub flags: u32,
    pub parameters: Vec<ParsedRootParameter>,
    pub num_static_samplers: u32,
    pub total_dwords: u32,
}

/// Win64 `D3D12_ROOT_PARAMETER` / `D3D12_ROOT_PARAMETER1` (32 bytes, align 8).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct D3D12RootParameterRaw {
    pub parameter_type: u32,
    pub _pad04: u32,
    pub data: [u32; 4], // union @8..24 (ptr @16 for DescriptorTable)
    pub shader_visibility: u32,
    pub _pad1c: u32,
}

const _: () = assert!(size_of::<D3D12RootParameterRaw>() == 32);

/// Validate and serialize a `D3D12_VERSIONED_ROOT_SIGNATURE_DESC` (v1.0 or v1.1) into a valid
/// offset-based DXBC container carrying an `RTS0` chunk inside a ref-counted `ID3DBlob`.
///
/// # Safety
/// `desc` must point to a readable `D3D12VersionedRootSignatureDesc` and `pp_blob` must be writable.
pub unsafe fn serialize_versioned_root_signature(
    desc: *const D3D12VersionedRootSignatureDesc,
    pp_blob: *mut *mut c_void,
    pp_error_blob: *mut *mut c_void,
) -> i32 {
    if !pp_error_blob.is_null() {
        // SAFETY: Caller supplied writable optional `pp_error_blob`.
        unsafe { *pp_error_blob = std::ptr::null_mut() };
    }
    if desc.is_null() || pp_blob.is_null() {
        return E_POINTER;
    }
    // SAFETY: `desc` points to a readable 48-byte `D3D12VersionedRootSignatureDesc`.
    let d = unsafe { &*desc };
    if d.version != 1 && d.version != 2 {
        if !pp_error_blob.is_null() {
            let err_blob = create_d3d_blob(b"unsupported root signature version\0".to_vec());
            // SAFETY: Caller supplied writable `pp_error_blob`.
            unsafe { *pp_error_blob = err_blob.cast::<c_void>() };
        }
        return E_INVALIDARG;
    }
    if (d.num_parameters > 0 && d.p_parameters.is_null())
        || (d.num_static_samplers > 0 && d.p_static_samplers.is_null())
    {
        return E_INVALIDARG;
    }

    // Build authentic offset-based RTS0 payload:
    //   Header (24 B): [version, num_params, params_offset, num_samplers, samplers_offset, flags]
    //   Param table (num_params * 12 B): [param_type, visibility, data_offset]
    //   Param data blocks (with range arrays serialized by offset, never raw guest pointers)
    //   Static samplers (num_samplers * 52 B)
    let params_offset: u32 = 24;
    let param_table_bytes = (d.num_parameters as usize).saturating_mul(12);
    let mut param_headers = Vec::with_capacity(param_table_bytes);
    let mut param_data = Vec::new();
    let mut total_dwords = 0u32;

    if d.num_parameters > 0 {
        // SAFETY: Validated non-null `p_parameters` with `num_parameters` 32-byte entries.
        let params = unsafe {
            std::slice::from_raw_parts(
                d.p_parameters.cast::<D3D12RootParameterRaw>(),
                d.num_parameters as usize,
            )
        };
        let data_base = params_offset.saturating_add(param_table_bytes as u32);
        for p in params {
            let data_offset = data_base.saturating_add(param_data.len() as u32);
            for w in [p.parameter_type, p.shader_visibility, data_offset] {
                param_headers.extend_from_slice(&w.to_le_bytes());
            }
            match p.parameter_type {
                0 => {
                    // DESCRIPTOR_TABLE: data[0] = num_ranges, data[2..4] = p_ranges (u64)
                    let num_ranges = p.data[0];
                    let ptr_val = (u64::from(p.data[3]) << 32) | u64::from(p.data[2]);
                    if num_ranges > 0 && ptr_val == 0 {
                        return E_INVALIDARG;
                    }
                    total_dwords = total_dwords.saturating_add(1);
                    let ranges_offset = data_base.saturating_add((param_data.len() + 8) as u32);
                    param_data.extend_from_slice(&num_ranges.to_le_bytes());
                    param_data.extend_from_slice(&ranges_offset.to_le_bytes());
                    let range_stride = if d.version == 2 { 24usize } else { 20usize };
                    if num_ranges > 0 {
                        let byte_len = (num_ranges as usize).saturating_mul(range_stride);
                        // SAFETY: Validated non-null `pDescriptorRanges` for `num_ranges` entries.
                        let raw_ranges =
                            unsafe { std::slice::from_raw_parts(ptr_val as *const u8, byte_len) };
                        param_data.extend_from_slice(raw_ranges);
                    }
                }
                1 => {
                    // 32BIT_CONSTANTS: [shader_register, register_space, num_32bit_values]
                    total_dwords = total_dwords.saturating_add(p.data[2]);
                    for w in [p.data[0], p.data[1], p.data[2]] {
                        param_data.extend_from_slice(&w.to_le_bytes());
                    }
                }
                2..=4 => {
                    // Inline CBV / SRV / UAV descriptor: 2 DWORDs (v1.0) or 3 DWORDs (v1.1)
                    total_dwords = total_dwords.saturating_add(2);
                    param_data.extend_from_slice(&p.data[0].to_le_bytes());
                    param_data.extend_from_slice(&p.data[1].to_le_bytes());
                    if d.version == 2 {
                        param_data.extend_from_slice(&p.data[2].to_le_bytes());
                    }
                }
                _ => return E_INVALIDARG,
            }
        }
    }

    if total_dwords > 64 {
        return E_INVALIDARG;
    }

    let samplers_offset: u32 = params_offset
        .saturating_add(param_headers.len() as u32)
        .saturating_add(param_data.len() as u32);
    let mut sampler_bytes = Vec::new();
    if d.num_static_samplers > 0 {
        let samp_len = (d.num_static_samplers as usize).saturating_mul(52);
        // SAFETY: Validated `p_static_samplers` non-null; each static sampler is 52 bytes.
        let raw = unsafe { std::slice::from_raw_parts(d.p_static_samplers.cast::<u8>(), samp_len) };
        sampler_bytes.extend_from_slice(raw);
    }

    let mut payload =
        Vec::with_capacity(24 + param_headers.len() + param_data.len() + sampler_bytes.len());
    for word in [
        d.version,
        d.num_parameters,
        params_offset,
        d.num_static_samplers,
        samplers_offset,
        d.flags,
    ] {
        payload.extend_from_slice(&word.to_le_bytes());
    }
    payload.extend_from_slice(&param_headers);
    payload.extend_from_slice(&param_data);
    payload.extend_from_slice(&sampler_bytes);

    let dxbc = wrap_dxbc_single_chunk(b"RTS0", &payload);
    let blob = create_d3d_blob(dxbc);
    // SAFETY: `pp_blob` was validated non-null.
    unsafe {
        *pp_blob = blob.cast::<c_void>();
    }
    S_OK
}

fn read_u32_le(buf: &[u8], off: usize) -> Result<u32, String> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| "root signature offset overflow".to_owned())?;
    let bytes = buf
        .get(off..end)
        .ok_or_else(|| format!("root signature truncated at offset 0x{off:x}"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Extract the `RTS0` payload from either a DXBC container or a raw `RTS0` buffer and decode
/// it into a validated [`ParsedRootSignature`].
pub fn parse_root_signature_blob(blob: &[u8]) -> Result<ParsedRootSignature, String> {
    let payload = if blob.len() >= 32 && &blob[0..4] == b"DXBC" {
        let chunk_count = read_u32_le(blob, 28)? as usize;
        let mut found = None;
        for i in 0..chunk_count {
            let chunk_off = read_u32_le(blob, 32 + i * 4)? as usize;
            let hdr_end = chunk_off
                .checked_add(8)
                .ok_or_else(|| "DXBC chunk offset overflow".to_owned())?;
            if hdr_end > blob.len() {
                return Err("DXBC chunk header out of bounds".into());
            }
            if &blob[chunk_off..chunk_off + 4] == b"RTS0" {
                let sz = read_u32_le(blob, chunk_off + 4)? as usize;
                let end = hdr_end
                    .checked_add(sz)
                    .ok_or_else(|| "RTS0 size overflow".to_owned())?;
                if end > blob.len() {
                    return Err("RTS0 chunk payload truncated".into());
                }
                found = Some(&blob[hdr_end..end]);
                break;
            }
        }
        found.ok_or_else(|| "DXBC container missing RTS0 chunk".to_owned())?
    } else {
        blob
    };

    if payload.len() < 24 {
        return Err("RTS0 payload shorter than 24-byte header".into());
    }
    let version = read_u32_le(payload, 0)?;
    if version != 1 && version != 2 {
        return Err(format!("unsupported RTS0 version {version}"));
    }
    let num_params = read_u32_le(payload, 4)? as usize;
    let params_offset = read_u32_le(payload, 8)? as usize;
    let num_static_samplers = read_u32_le(payload, 12)?;
    let _samplers_offset = read_u32_le(payload, 16)? as usize;
    let flags = read_u32_le(payload, 20)?;

    let mut parameters = Vec::with_capacity(num_params);
    let mut dword_cursor = 0u32;

    for i in 0..num_params {
        let hdr_off = params_offset
            .checked_add(i * 12)
            .ok_or_else(|| "RTS0 param header overflow".to_owned())?;
        let param_type = read_u32_le(payload, hdr_off)?;
        let shader_visibility = read_u32_le(payload, hdr_off + 4)?;
        let data_off = read_u32_le(payload, hdr_off + 8)? as usize;

        let dword_offset = dword_cursor;
        let kind = match param_type {
            0 => {
                let num_ranges = read_u32_le(payload, data_off)? as usize;
                let ranges_off = read_u32_le(payload, data_off + 4)? as usize;
                let stride = if version == 2 { 24usize } else { 20usize };
                let mut ranges = Vec::with_capacity(num_ranges);
                let mut next_table_off = 0u32;
                for r in 0..num_ranges {
                    let ro = ranges_off
                        .checked_add(r * stride)
                        .ok_or_else(|| "RTS0 descriptor range overflow".to_owned())?;
                    let range_type = read_u32_le(payload, ro)?;
                    let num_descriptors = read_u32_le(payload, ro + 4)?;
                    let base_shader_register = read_u32_le(payload, ro + 8)?;
                    let register_space = read_u32_le(payload, ro + 12)?;
                    let (r_flags, raw_off) = if version == 2 {
                        (
                            read_u32_le(payload, ro + 16)?,
                            read_u32_le(payload, ro + 20)?,
                        )
                    } else {
                        (0, read_u32_le(payload, ro + 16)?)
                    };
                    let resolved_off = if raw_off == u32::MAX {
                        next_table_off
                    } else {
                        raw_off
                    };
                    next_table_off = resolved_off.saturating_add(num_descriptors.min(1024));
                    ranges.push(ParsedDescriptorRange {
                        range_type,
                        num_descriptors,
                        base_shader_register,
                        register_space,
                        flags: r_flags,
                        offset_in_descriptors_from_table_start: resolved_off,
                    });
                }
                dword_cursor = dword_cursor.saturating_add(1);
                ParsedRootParamKind::DescriptorTable(ranges)
            }
            1 => {
                let shader_register = read_u32_le(payload, data_off)?;
                let register_space = read_u32_le(payload, data_off + 4)?;
                let num_32bit_values = read_u32_le(payload, data_off + 8)?;
                dword_cursor = dword_cursor.saturating_add(num_32bit_values);
                ParsedRootParamKind::Constants {
                    shader_register,
                    register_space,
                    num_32bit_values,
                }
            }
            2..=4 => {
                let shader_register = read_u32_le(payload, data_off)?;
                let register_space = read_u32_le(payload, data_off + 4)?;
                let r_flags = if version == 2 {
                    read_u32_le(payload, data_off + 8)?
                } else {
                    0
                };
                dword_cursor = dword_cursor.saturating_add(2);
                match param_type {
                    2 => ParsedRootParamKind::Cbv {
                        shader_register,
                        register_space,
                        flags: r_flags,
                    },
                    3 => ParsedRootParamKind::Srv {
                        shader_register,
                        register_space,
                        flags: r_flags,
                    },
                    _ => ParsedRootParamKind::Uav {
                        shader_register,
                        register_space,
                        flags: r_flags,
                    },
                }
            }
            other => return Err(format!("unknown RTS0 parameter type {other}")),
        };
        parameters.push(ParsedRootParameter {
            shader_visibility,
            dword_offset,
            kind,
        });
    }

    if dword_cursor > 64 {
        return Err(format!(
            "root signature exceeds 64-DWORD budget ({dword_cursor} DWORDs)"
        ));
    }

    Ok(ParsedRootSignature {
        version,
        flags,
        parameters,
        num_static_samplers,
        total_dwords: dword_cursor,
    })
}

/// Wrap a single chunk inside a valid DXBC container with the standard DXBC MD5-variant digest.
pub fn wrap_dxbc_single_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let chunk_offset: u32 = 0x24; // 32-byte header + 1 * 4-byte chunk offset table
    let total_size: u32 = chunk_offset + 8 + payload.len() as u32;
    let mut out = Vec::with_capacity(total_size as usize);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0u8; 16]); // placeholder checksum at 0x04..0x14
    out.extend_from_slice(&1u32.to_le_bytes()); // container version 1
    out.extend_from_slice(&total_size.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // 1 chunk
    out.extend_from_slice(&chunk_offset.to_le_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);

    let digest = dxbc_md5_digest(&out[0x14..]);
    out[0x04..0x14].copy_from_slice(&digest);
    out
}

/// Compute the standard DXBC MD5-variant 16-byte checksum over `data` (bytes `[0x14..end)`).
fn dxbc_md5_digest(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cc8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut state = [0x6745_2301u32, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476];
    let bit_len = (data.len() as u32).wrapping_mul(8);
    let tail_word = ((data.len() as u32) >> 2) | 1;

    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    // DXBC MD5 variant appends bit_len at word 14 and `(byte_len >> 2) | 1` at word 15
    // (final block layout used by DXBC container validation).
    let full_blocks = (data.len() / 64) * 64;
    let rem = data.len() - full_blocks;
    if rem >= 56 {
        // Two-block tail: put standard length at end, then DXBC tail word.
        padded.extend_from_slice(&bit_len.to_le_bytes());
        padded.extend_from_slice(&tail_word.to_le_bytes());
    } else {
        // Single final block: word 0 = bit_len when padded at 56..64 -> DXBC uses bit_len & tail_word.
        padded.extend_from_slice(&bit_len.to_le_bytes());
        padded.extend_from_slice(&tail_word.to_le_bytes());
    }

    let (blocks, _) = padded.as_chunks::<64>();
    for block in blocks {
        let mut m = [0u32; 16];
        let (words, _) = block.as_chunks::<4>();
        for (i, w) in words.iter().enumerate() {
            m[i] = u32::from_le_bytes(*w);
        }
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }
        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
    }

    let mut out = [0u8; 16];
    for (i, w) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_empty_v11_root_signature_to_68_byte_dxbc_rts0_blob() {
        let desc = D3D12VersionedRootSignatureDesc {
            version: 2,
            _pad04: 0,
            num_parameters: 0,
            _pad0c: 0,
            p_parameters: std::ptr::null(),
            num_static_samplers: 0,
            _pad1c: 0,
            p_static_samplers: std::ptr::null(),
            flags: 0,
            _pad2c: 0,
        };
        let mut blob_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: Valid descriptor and output pointer.
        let hr = unsafe {
            serialize_versioned_root_signature(&desc, &mut blob_ptr, std::ptr::null_mut())
        };
        assert_eq!(hr, S_OK);
        assert!(!blob_ptr.is_null());

        // SAFETY: `blob_ptr` is a live `ComD3DBlob` with 5 vtable slots.
        unsafe {
            let blob = blob_ptr.cast::<ComD3DBlob>();
            let vt = (*blob).vtbl;
            let get_ptr: extern "win64" fn(*mut ComD3DBlob) -> *mut c_void =
                std::mem::transmute(*vt.add(3));
            let get_size: extern "win64" fn(*mut ComD3DBlob) -> usize =
                std::mem::transmute(*vt.add(4));
            let release: extern "win64" fn(*mut ComD3DBlob) -> u32 =
                std::mem::transmute(*vt.add(2));

            let sz = get_size(blob);
            assert_eq!(sz, 68);
            let bytes = std::slice::from_raw_parts(get_ptr(blob).cast::<u8>(), sz);
            assert_eq!(&bytes[0..4], b"DXBC");
            assert_eq!(&bytes[0x24..0x28], b"RTS0");
            assert_eq!(release(blob), 0);
        }
    }

    #[test]
    fn configures_virtual_memory_x_range() {
        let mut params = D3D12ConfigureVirtualMemoryXParams::default();
        // SAFETY: Stack-owned 32-byte parameter block.
        assert_eq!(unsafe { configure_virtual_memory_x(&mut params) }, S_OK);
        assert!(params.low_address_out > 0);
        assert!(params.high_address_out > params.low_address_out);
    }

    #[test]
    fn roundtrips_v11_root_signature_with_constants_cbv_and_table() {
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
            offset_from_start: u32::MAX, // APPEND
        }];
        let range_ptr = ranges.as_ptr() as usize as u64;
        let params = [
            D3D12RootParameterRaw {
                parameter_type: 1, // 32BIT_CONSTANTS
                _pad04: 0,
                data: [0, 0, 4, 0], // b0, space0, 4 dwords
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 2, // CBV
                _pad04: 0,
                data: [1, 0, 0, 0], // b1, space0
                shader_visibility: 0,
                _pad1c: 0,
            },
            D3D12RootParameterRaw {
                parameter_type: 0, // DESCRIPTOR_TABLE
                _pad04: 0,
                data: [1, 0, range_ptr as u32, (range_ptr >> 32) as u32],
                shader_visibility: 0,
                _pad1c: 0,
            },
        ];
        let desc = D3D12VersionedRootSignatureDesc {
            version: 2,
            _pad04: 0,
            num_parameters: params.len() as u32,
            _pad0c: 0,
            p_parameters: params.as_ptr().cast(),
            num_static_samplers: 0,
            _pad1c: 0,
            p_static_samplers: std::ptr::null(),
            flags: 1, // ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
            _pad2c: 0,
        };
        let mut blob_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: Valid stack descriptors.
        assert_eq!(
            unsafe {
                serialize_versioned_root_signature(&desc, &mut blob_ptr, std::ptr::null_mut())
            },
            S_OK
        );
        // SAFETY: `blob_ptr` is a live `ComD3DBlob`.
        unsafe {
            let blob = blob_ptr.cast::<ComD3DBlob>();
            let vt = (*blob).vtbl;
            let get_ptr: extern "win64" fn(*mut ComD3DBlob) -> *mut c_void =
                std::mem::transmute(*vt.add(3));
            let get_size: extern "win64" fn(*mut ComD3DBlob) -> usize =
                std::mem::transmute(*vt.add(4));
            let release: extern "win64" fn(*mut ComD3DBlob) -> u32 =
                std::mem::transmute(*vt.add(2));

            let bytes = std::slice::from_raw_parts(get_ptr(blob).cast::<u8>(), get_size(blob));
            let parsed = parse_root_signature_blob(bytes).expect("parse serialized RTS0");
            assert_eq!(parsed.version, 2);
            assert_eq!(parsed.flags, 1);
            assert_eq!(parsed.parameters.len(), 3);
            assert_eq!(parsed.total_dwords, 7); // 4 + 2 + 1
            assert_eq!(parsed.parameters[0].dword_offset, 0);
            assert_eq!(
                parsed.parameters[0].kind,
                ParsedRootParamKind::Constants {
                    shader_register: 0,
                    register_space: 0,
                    num_32bit_values: 4,
                }
            );
            assert_eq!(parsed.parameters[1].dword_offset, 4);
            assert_eq!(
                parsed.parameters[1].kind,
                ParsedRootParamKind::Cbv {
                    shader_register: 1,
                    register_space: 0,
                    flags: 0,
                }
            );
            assert_eq!(parsed.parameters[2].dword_offset, 6);
            assert_eq!(
                parsed.parameters[2].kind,
                ParsedRootParamKind::DescriptorTable(vec![ParsedDescriptorRange {
                    range_type: 1,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    flags: 0,
                    offset_in_descriptors_from_table_start: 0,
                }])
            );
            assert_eq!(release(blob), 0);
        }
    }
}
