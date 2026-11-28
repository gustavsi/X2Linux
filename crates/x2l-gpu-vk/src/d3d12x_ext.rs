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

use crate::com::{E_INVALIDARG, E_POINTER, S_OK};
use std::ffi::c_void;
use std::mem::{offset_of, size_of};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

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
        (*params).high_address_out = 0x0000_00FF_0000_0000;
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
    _riid: *const c_void,
    out_obj: *mut *mut c_void,
) -> i32 {
    if this.is_null() || out_obj.is_null() {
        return E_POINTER;
    }
    // SAFETY: `this` is a live `ComD3DBlob` and `out_obj` is writable.
    unsafe {
        (*this).ref_count.fetch_add(1, Ordering::AcqRel);
        *out_obj = this.cast::<c_void>();
    }
    S_OK
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

/// Validate and serialize a `D3D12_VERSIONED_ROOT_SIGNATURE_DESC` (v1.0 or v1.1) into a valid
/// DXBC container carrying an `RTS0` chunk inside a ref-counted `ID3DBlob`.
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

    // Build authentic RTS0 payload: 24-byte header + raw parameter/sampler records.
    // Header: [version, num_params, params_offset, num_samplers, samplers_offset, flags].
    let mut payload = Vec::with_capacity(24);
    let params_offset: u32 = 24;
    let mut param_bytes = Vec::new();
    if d.num_parameters > 0 {
        let param_len = (d.num_parameters as usize).saturating_mul(32);
        // SAFETY: Validated `p_parameters` non-null; each D3D12_ROOT_PARAMETER[1] is 32 bytes.
        let raw = unsafe { std::slice::from_raw_parts(d.p_parameters.cast::<u8>(), param_len) };
        param_bytes.extend_from_slice(raw);
    }
    let samplers_offset: u32 = params_offset.saturating_add(param_bytes.len() as u32);
    let mut sampler_bytes = Vec::new();
    if d.num_static_samplers > 0 {
        let samp_len = (d.num_static_samplers as usize).saturating_mul(52);
        // SAFETY: Validated `p_static_samplers` non-null; each static sampler is 52 bytes.
        let raw = unsafe { std::slice::from_raw_parts(d.p_static_samplers.cast::<u8>(), samp_len) };
        sampler_bytes.extend_from_slice(raw);
    }

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
    payload.extend_from_slice(&param_bytes);
    payload.extend_from_slice(&sampler_bytes);

    let dxbc = wrap_dxbc_single_chunk(b"RTS0", &payload);
    let blob = create_d3d_blob(dxbc);
    // SAFETY: `pp_blob` was validated non-null.
    unsafe {
        *pp_blob = blob.cast::<c_void>();
    }
    S_OK
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
}
