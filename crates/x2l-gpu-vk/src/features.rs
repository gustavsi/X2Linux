// SPDX-License-Identifier: GPL-3.0-or-later

//! Device feature-query ABI and guest-visible capabilities.

use crate::com::{E_INVALIDARG, S_OK};
use crate::{log_d3d12_trace, trace_d3d12_enabled};
use std::ffi::c_void;
use std::ptr;

// Values are D3D12_FEATURE enum discriminants from the public DirectX-Headers.
const OPTIONS: u32 = 0;
const ARCHITECTURE: u32 = 1;
const FEATURE_LEVELS: u32 = 2;
const FORMAT_SUPPORT: u32 = 3;
const FORMAT_INFO: u32 = 5;
const GPU_VIRTUAL_ADDRESS_SUPPORT: u32 = 6;
const SHADER_MODEL: u32 = 7;
const OPTIONS1: u32 = 8;
const ROOT_SIGNATURE: u32 = 12;
const ARCHITECTURE1: u32 = 16;
const OPTIONS2: u32 = 18;
const OPTIONS3: u32 = 21;
const OPTIONS4: u32 = 23;
const OPTIONS5: u32 = 27;
const OPTIONS6: u32 = 30;
const OPTIONS7: u32 = 32;
const OPTIONS8: u32 = 36;
const OPTIONS9: u32 = 37;
const OPTIONS10: u32 = 39;
const OPTIONS11: u32 = 40;
const OPTIONS12: u32 = 41;
const OPTIONS13: u32 = 42;
const OPTIONS14: u32 = 43;
const OPTIONS15: u32 = 44;
const OPTIONS16: u32 = 45;
const OPTIONS17: u32 = 46;
const OPTIONS18: u32 = 47;
const OPTIONS19: u32 = 48;
const OPTIONS20: u32 = 49;
const OPTIONS21: u32 = 53;

fn feature_size(feature: u32) -> Option<u32> {
    Some(match feature {
        OPTIONS => 60,
        ARCHITECTURE => 16,
        FEATURE_LEVELS => 24,
        FORMAT_SUPPORT | OPTIONS4 | OPTIONS5 | OPTIONS12 | OPTIONS14 => 12,
        FORMAT_INFO
        | GPU_VIRTUAL_ADDRESS_SUPPORT
        | OPTIONS2
        | OPTIONS7
        | OPTIONS10
        | OPTIONS15
        | OPTIONS16
        | OPTIONS17
        | OPTIONS20 => 8,
        SHADER_MODEL | ROOT_SIGNATURE | OPTIONS8 | OPTIONS11 | OPTIONS18 => 4,
        ARCHITECTURE1 | OPTIONS3 | OPTIONS6 => 20,
        OPTIONS1 | OPTIONS9 | OPTIONS13 => 24,
        OPTIONS19 => 40,
        OPTIONS21 => 16,
        _ => return None,
    })
}

/// # Safety
/// On a non-null buffer of the exact feature size, fields marked input by the public ABI must be
/// initialized, output storage must be writable, and any embedded input pointers must be readable.
pub(crate) unsafe fn check_feature_support(
    feature: u32,
    data: *mut c_void,
    size: u32,
    vulkan: bool,
) -> i32 {
    let expected = feature_size(feature);
    // SAFETY: We checked the feature's exact Win64 layout before accessing the guest data.
    let hr = match expected {
        Some(expected) if size == expected && !data.is_null() => unsafe {
            write_feature(feature, data.cast(), size as usize, vulkan)
        },
        _ => E_INVALIDARG,
    };
    if expected.is_none() || trace_d3d12_enabled() {
        let detail = if expected.is_none() {
            " unknown feature"
        } else {
            ""
        };
        log_d3d12_trace(format_args!(
            "ID3D12DeviceX::CheckFeatureSupport(feature={feature:#x}, size={size}) -> hr=0x{:08x}{detail}",
            hr as u32
        ));
    }
    hr
}

/// # Safety
/// The caller has checked the size for a known feature and supplies the required input fields.
unsafe fn write_feature(feature: u32, data: *mut u32, size: usize, vulkan: bool) -> i32 {
    // SAFETY: Each case accesses only the fields of its validated layout. Unaligned access also
    // avoids relying on guest alignment, and raw pointers permit uninitialized output fields.
    unsafe {
        match feature {
            OPTIONS => {
                // In struct order, all fields are Win64 BOOL, enum, or UINT (32 bits).
                let values: [u32; 15] = [0, 0, 0, 3, 3, 0, 0, 0, 0, 40, 0, 0, 0, 0, 2];
                ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), data.cast(), size);
            }
            ARCHITECTURE | ARCHITECTURE1 => {
                if data.read_unaligned() != 0 {
                    return E_INVALIDARG;
                }
                data.add(1).write_unaligned(0); // TileBasedRenderer
                data.add(2).write_unaligned(1); // UMA
                data.add(3).write_unaligned(1); // CacheCoherentUMA
                if feature == ARCHITECTURE1 {
                    data.add(4).write_unaligned(0); // IsolatedMMU
                }
            }
            FEATURE_LEVELS => {
                let count = data.read_unaligned() as usize;
                let requested = data.add(2).cast::<*const u32>().read_unaligned();
                if count == 0 || requested.is_null() {
                    return E_INVALIDARG;
                }
                let level = (0..count)
                    .filter_map(|i| {
                        let level = requested.add(i).read_unaligned();
                        matches!(
                            level,
                            0x9100
                                | 0x9200
                                | 0x9300
                                | 0xa000
                                | 0xa100
                                | 0xb000
                                | 0xb100
                                | 0xc000
                                | 0xc100
                        )
                        .then_some(level)
                    })
                    .max();
                let Some(level) = level else {
                    return E_INVALIDARG;
                };
                data.add(4).write_unaligned(level);
            }
            FORMAT_SUPPORT => {
                let (support1, support2) = format_support(data.read_unaligned());
                data.add(1).write_unaligned(support1);
                data.add(2).write_unaligned(support2);
            }
            FORMAT_INFO => {
                let format = data.read_unaligned();
                let planes = match format {
                    45 => 2u8, // D24_UNORM_S8_UINT
                    _ => u8::from(format_support(format).0 != 0),
                };
                ptr::write_bytes(data.add(1).cast::<u8>(), 0, 4);
                data.add(1).cast::<u8>().write(planes);
            }
            SHADER_MODEL | ROOT_SIGNATURE => {
                let maximum = if feature == SHADER_MODEL { 0x66 } else { 2 };
                data.write_unaligned(data.read_unaligned().min(maximum));
            }
            // The remaining structs are output-only, with all fields represented by
            // a 32-bit BOOL, enum, or UINT. Unadvertised optional features start at zero.
            _ => {
                ptr::write_bytes(data.cast::<u8>(), 0, size);
                match feature {
                    GPU_VIRTUAL_ADDRESS_SUPPORT => {
                        data.write_unaligned(40);
                        data.add(1).write_unaligned(44);
                    }
                    OPTIONS1 => {
                        data.write_unaligned(u32::from(vulkan)); // WaveOps
                        if vulkan {
                            data.add(1).write_unaligned(32); // WaveLaneCountMin
                            data.add(2).write_unaligned(64); // WaveLaneCountMax
                            data.add(3).write_unaligned(64); // TotalLaneCount
                            data.add(5).write_unaligned(1); // Int64ShaderOps
                        }
                    }
                    OPTIONS7 if vulkan => data.write_unaligned(10), // MeshShaderTier 1
                    OPTIONS12 => {
                        data.write_unaligned(u32::MAX); // MSPrimitivesPipelineStatisticIncludesCulledPrimitives: UNKNOWN
                        data.add(1).write_unaligned(u32::from(vulkan)); // EnhancedBarriersSupported
                    }
                    _ => {}
                }
            }
        }
    }
    S_OK
}

fn format_support(format: u32) -> (u32, u32) {
    const BUFFER: u32 = 0x1;
    const VERTEX: u32 = 0x2;
    const INDEX: u32 = 0x4;
    const TEX1D: u32 = 0x10;
    const TEX2D: u32 = 0x20;
    const TEX3D: u32 = 0x40;
    const CUBE: u32 = 0x80;
    const LOAD: u32 = 0x100;
    const SAMPLE: u32 = 0x200;
    const MIP: u32 = 0x1000;
    const RT: u32 = 0x4000;
    const BLEND: u32 = 0x8000;
    const DEPTH_STENCIL: u32 = 0x1_0000;
    const CAST: u32 = 0x10_0000;
    const GATHER: u32 = 0x80_0000;
    const UAV: u32 = 0x200_0000;
    const UAV_UINT_ATOMICS: u32 = 0x2f;
    const UAV_LOAD_STORE: u32 = 0xc0;
    const TEXTURE: u32 = TEX1D | TEX2D | TEX3D | CUBE | MIP;
    const FILTERED: u32 = LOAD | SAMPLE | GATHER;
    const COLOR: u32 = TEXTURE | FILTERED | RT | BLEND;
    const TYPED: u32 = BUFFER | VERTEX | UAV;
    const BC: u32 = TEX2D | CUBE | MIP;

    match format {
        // R32G32B32A32_FLOAT, R16G16B16A16_FLOAT, R32G32_FLOAT, R10G10B10A2_UNORM,
        // R8G8B8A8_UNORM and R32_FLOAT.
        2 | 10 | 16 | 24 | 28 | 41 => (COLOR | TYPED, UAV_LOAD_STORE),
        6 => (BUFFER | VERTEX | LOAD, 0), // R32G32B32_FLOAT
        29 | 87 | 91 => (COLOR, 0),       // R8G8B8A8_UNORM_SRGB, BGRA8 UNORM and SRGB
        42 => (
            TEXTURE | LOAD | RT | TYPED | INDEX,
            UAV_LOAD_STORE | UAV_UINT_ATOMICS,
        ),
        57 => (TEXTURE | LOAD | RT | TYPED | INDEX, UAV_LOAD_STORE), // R16_UINT
        39 => (TEX2D | MIP | CAST, 0),                               // R32_TYPELESS
        40 | 45 => (TEX2D | DEPTH_STENCIL | MIP, 0),                 // D32_FLOAT, D24_UNORM_S8_UINT
        70 | 73 | 76 | 79 | 82 | 94 | 97 => (BC | CAST, 0),          // BC1–BC7 typeless
        71 | 72 | 74 | 75 | 77 | 78 | 80 | 81 | 83 | 84 | 95 | 96 | 98 | 99 => (BC | FILTERED, 0),
        _ => (0, 0),
    }
}

pub(crate) fn descriptor_handle_increment_size(heap_type: u32) -> u32 {
    let size = descriptor_increment(heap_type);
    if trace_d3d12_enabled() {
        log_d3d12_trace(format_args!(
            "ID3D12DeviceX::GetDescriptorHandleIncrementSize(heap_type={heap_type:#x}) -> size={size}"
        ));
    }
    size
}

pub(crate) fn descriptor_increment(heap_type: u32) -> u32 {
    match heap_type {
        1 => 16,
        0 | 2..=4 => 32,
        _ => 0,
    }
}
