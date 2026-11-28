// Clean-room direct Vulkan 1.3/1.4 (`ash`) backend for X2Linux (`D3D12.X` -> Vulkan + SPIR-V 1.6).
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Direct Vulkan 1.3/1.4 backend (`gpu.backend = "vulkan"`) using:
//! - `VK_KHR_dynamic_rendering` (core 1.3)
//! - `VK_KHR_synchronization2` (core 1.3)
//! - `VK_KHR_timeline_semaphore` (core 1.2)
//! - `VK_KHR_buffer_device_address` (core 1.2)
//! - Optional `VK_EXT_external_memory_host` and `VK_EXT_descriptor_buffer` (enabled when advertised)
//! - Direct `RDNA2 -> Binary SPIR-V 1.6` compilation via `x2l-rdna2-spirv` cached in XDG via
//!   `x2l-pipeline` (`$XDG_CACHE_HOME/x2linux/<title>/spirv/`)
//! - Headless offscreen `R8G8B8A8_UNORM` rendering + `PresentX` readback + optional P6 PPM dump
//!   via `X2L_FRAME_DUMP`.

use crate::GpuError;
use crate::d3d12x_ext::{ParsedRootSignature, parse_root_signature_blob};
use crate::memory::{AllocationId, AllocationPolicy, UnifiedMemoryManager, UserfaultfdMode};
use crate::wsi::{NativeWindow, PresentSurfaceKind, SwapchainPresenter};
use ash::vk;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{debug, info, warn};
use x2l_common::PresentMode;
use x2l_pipeline::{Cache, Converter, PipelineError};
use x2l_rdna2_spirv::{
    ContainerKind, GraphicsCompileOptions, ShaderStage, SignatureElement, StageSignature,
    compile_graphics_spv, disassemble_reachable, parse_container, spirv_bytes_to_words,
    spirv_words_to_bytes,
};

/// DXGI format constants used by the M2.4/M3.3 backend.
pub const DXGI_FORMAT_R32G32B32A32_FLOAT: u32 = 2;
pub const DXGI_FORMAT_R32G32B32_FLOAT: u32 = 6;
pub const DXGI_FORMAT_R32G32_FLOAT: u32 = 16;
pub const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
pub const DXGI_FORMAT_R32_TYPELESS: u32 = 39;
pub const DXGI_FORMAT_D32_FLOAT: u32 = 40;
pub const DXGI_FORMAT_R32_FLOAT: u32 = 41;
pub const DXGI_FORMAT_R32_UINT: u32 = 42;
pub const DXGI_FORMAT_R16_UINT: u32 = 57;
pub const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;

/// D3D12 resource dimensions.
pub const D3D12_RESOURCE_DIMENSION_BUFFER: u32 = 1;
pub const D3D12_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;

/// Max root-constant/descriptor draws or dispatches per single `ExecuteCommandLists` submission.
const MAX_DRAWS_PER_SUBMIT: usize = 256;

/// Configuration for initializing [`VulkanBackend`].
#[derive(Debug, Clone, Default)]
pub struct VulkanBackendConfig {
    pub spirv_cache_dir: Option<PathBuf>,
    pub frame_dump_path: Option<PathBuf>,
    pub prefer_rebar_zero_copy: bool,
    pub enable_validation: bool,
    pub device_filter: Option<String>,
    pub present_mode: PresentMode,
}

impl VulkanBackendConfig {
    /// Build a backend configuration from `X2L_*` environment variables.
    pub fn from_env() -> Self {
        let spirv_cache_dir = std::env::var("X2L_SPIRV_CACHE_DIR")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let frame_dump_path = std::env::var("X2L_FRAME_DUMP")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        let prefer_rebar_zero_copy = std::env::var("X2L_PREFER_REBAR_ZERO_COPY")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let enable_validation = std::env::var("X2L_GPU_VALIDATION")
            .ok()
            .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
        let device_filter = std::env::var("X2L_VK_DEVICE")
            .ok()
            .filter(|v| !v.is_empty());
        let present_mode = match std::env::var("X2L_GPU_PRESENT_MODE")
            .ok()
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "window" => PresentMode::Window,
            "auto" => PresentMode::Auto,
            "wayland" => PresentMode::Wayland,
            "x11" => PresentMode::X11,
            _ => PresentMode::Headless,
        };

        Self {
            spirv_cache_dir,
            frame_dump_path,
            prefer_rebar_zero_copy,
            enable_validation,
            device_filter,
            present_mode,
        }
    }
}

/// Snapshot of Vulkan runtime telemetry persisted into `report.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanTelemetrySnapshot {
    pub backend: String,
    pub present_surface: String,
    pub device_name: String,
    pub driver_info: String,
    pub external_memory_host: bool,
    pub descriptor_buffer: bool,
    pub userfaultfd_mode: String,
    pub shaders_compiled: u64,
    pub frames_presented: u64,
}

/// Captured headless RGBA8 frame produced by `PresentX`.
#[derive(Debug, Clone)]
pub struct PresentedFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl PresentedFrame {
    /// Sample `[r, g, b, a]` at pixel coordinates `(x, y)` (top-left origin).
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        Some([
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ])
    }

    /// Write the frame as a binary P6 PPM image (`P6\n<w> <h>\n255\n` + RGB bytes).
    pub fn write_ppm(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut rgb = Vec::with_capacity((self.width as usize) * (self.height as usize) * 3);
        let (pixels, _) = self.rgba.as_chunks::<4>();
        for px in pixels {
            rgb.push(px[0]);
            rgb.push(px[1]);
            rgb.push(px[2]);
        }
        let mut file = fs::File::create(path)?;
        write!(file, "P6\n{} {}\n255\n", self.width, self.height)?;
        file.write_all(&rgb)?;
        file.flush()
    }
}

/// One vertex-input element extracted from `D3D12_INPUT_LAYOUT_DESC`.
#[derive(Debug, Clone)]
pub struct InputElementInfo {
    pub semantic_name: String,
    pub semantic_index: u32,
    pub format: u32,
    pub input_slot: u32,
    pub aligned_byte_offset: u32,
}

/// Complete graphics pipeline description passed from `ID3D12DeviceX::CreateGraphicsPipelineState`.
#[derive(Debug, Clone)]
pub struct GraphicsPipelineDescInfo {
    pub vs_blob: Vec<u8>,
    pub ps_blob: Vec<u8>,
    pub rtv_format: u32,
    pub dsv_format: u32,
    pub depth_enable: bool,
    pub depth_write_enable: bool,
    pub depth_func: u32,
    pub input_elements: Vec<InputElementInfo>,
    pub has_root_signature: bool,
}

/// Recorded D3D12 graphics/compute/copy command-list operation translated onto Vulkan 1.3.
#[derive(Clone)]
pub enum RecordedCommand {
    ClearRenderTargetView {
        resource_id: u64,
        color: [f32; 4],
    },
    ClearDepthStencilView {
        resource_id: u64,
        depth: f32,
    },
    SetViewports(Vec<vk::Viewport>),
    SetScissors(Vec<vk::Rect2D>),
    SetRenderTargets {
        rtvs: Vec<u64>,
        dsv: Option<u64>,
    },
    SetPipelineState(u64),
    SetPrimitiveTopology(u32),
    SetDescriptorHeaps(Vec<u64>),
    SetGraphicsRootSignature(u64),
    SetComputeRootSignature(u64),
    SetGraphicsRoot32BitConstants {
        root_param_index: u32,
        values: Vec<u32>,
        dest_offset: u32,
    },
    SetComputeRoot32BitConstants {
        root_param_index: u32,
        values: Vec<u32>,
        dest_offset: u32,
    },
    SetGraphicsRootConstantBufferView {
        root_param_index: u32,
        gpu_va: u64,
    },
    SetComputeRootConstantBufferView {
        root_param_index: u32,
        gpu_va: u64,
    },
    SetGraphicsRootShaderResourceView {
        root_param_index: u32,
        gpu_va: u64,
    },
    SetComputeRootShaderResourceView {
        root_param_index: u32,
        gpu_va: u64,
    },
    SetGraphicsRootUnorderedAccessView {
        root_param_index: u32,
        gpu_va: u64,
    },
    SetComputeRootUnorderedAccessView {
        root_param_index: u32,
        gpu_va: u64,
    },
    SetGraphicsRootDescriptorTable {
        root_param_index: u32,
        gpu_handle: u64,
    },
    SetComputeRootDescriptorTable {
        root_param_index: u32,
        gpu_handle: u64,
    },
    IASetVertexBuffers {
        start_slot: u32,
        views: Vec<(u64, u32, u32)>,
    },
    IASetIndexBuffer {
        gpu_va: u64,
        size_in_bytes: u32,
        format: u32,
    },
    ResourceBarrier {
        resource_id: u64,
        state_before: u32,
        state_after: u32,
    },
    DrawInstanced {
        vertex_count: u32,
        instance_count: u32,
        start_vertex: u32,
        start_instance: u32,
    },
    DrawIndexedInstanced {
        index_count: u32,
        instance_count: u32,
        start_index: u32,
        base_vertex: i32,
        start_instance: u32,
    },
    Dispatch {
        x: u32,
        y: u32,
        z: u32,
    },
    DispatchMesh {
        x: u32,
        y: u32,
        z: u32,
    },
    CopyBufferRegion {
        dst_res_id: u64,
        dst_offset: u64,
        src_res_id: u64,
        src_offset: u64,
        num_bytes: u64,
    },
    CopyTextureRegion {
        dst_res_id: u64,
        dst_is_footprint: bool,
        dst_offset: u64,
        dst_row_pitch: u32,
        dst_x: u32,
        dst_y: u32,
        src_res_id: u64,
        src_is_footprint: bool,
        src_offset: u64,
        src_row_pitch: u32,
        width: u32,
        height: u32,
    },
    CopyResource {
        dst_res_id: u64,
        src_res_id: u64,
    },
}

/// Description of a committed or placed buffer / 2D texture resource.
#[derive(Debug, Clone, Copy)]
pub struct ResourceDescInfo {
    pub dimension: u32,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub flags: u32,
}

#[derive(Debug, Clone)]
struct HeapRecord {
    alloc_id: AllocationId,
    guest_va: u64,
    size: u64,
    heap_type: u32,
    alignment: u64,
    flags: u32,
}

enum ResourceBacking {
    Buffer {
        alloc_id: AllocationId,
        guest_va: u64,
        base_offset: u64,
        size: usize,
        _is_placed: bool,
        desc: ResourceDescInfo,
    },
    Texture2D {
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        vk_format: vk::Format,
        aspect: vk::ImageAspectFlags,
        width: u32,
        height: u32,
        layout: vk::ImageLayout,
        desc: ResourceDescInfo,
    },
}

/// Description of a D3D12 sampler (`D3D12_SAMPLER_DESC` subset).
#[derive(Debug, Clone, Copy)]
pub struct SamplerDescInfo {
    pub filter: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub min_lod: f32,
    pub max_lod: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum DescriptorEntry {
    #[default]
    Empty,
    Rtv {
        resource_id: u64,
    },
    Dsv {
        resource_id: u64,
        flags: u32,
    },
    Cbv {
        gpu_va: u64,
        size_in_bytes: u32,
    },
    Srv {
        resource_id: u64,
        first_element: u64,
        num_elements: u32,
        stride: u32,
    },
    Uav {
        resource_id: u64,
        first_element: u64,
        num_elements: u32,
        stride: u32,
    },
    Sampler {
        sampler_id: u64,
    },
}

struct DescriptorHeapRecord {
    heap_type: u32,
    num_descriptors: u32,
    base_cpu_ptr: usize,
    increment: usize,
    slots: Vec<DescriptorEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineKind {
    Graphics,
    Compute,
    Mesh,
}

struct PipelineRecord {
    pipeline: vk::Pipeline,
    kind: PipelineKind,
    _rtv_format: vk::Format,
    dsv_format: Option<vk::Format>,
    has_vertex_inputs: bool,
}

struct FenceRecord {
    semaphore: vk::Semaphore,
}

struct QueueState {
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    submit_fence: vk::Fence,
    submit_descriptor_pool: vk::DescriptorPool,
    root_ubo_buf: vk::Buffer,
    root_ubo_mem: vk::DeviceMemory,
    root_ubo_ptr: *mut u8,
    root_ubo_stride: u64,
    dummy_ssbo_buf: vk::Buffer,
    dummy_ssbo_mem: vk::DeviceMemory,
    dummy_tex_img: vk::Image,
    dummy_tex_mem: vk::DeviceMemory,
    dummy_tex_view: vk::ImageView,
    dummy_tex_ready: bool,
    default_sampler: vk::Sampler,
}

/// Direct Vulkan 1.3/1.4 rendering backend backing M2.4/M3.4 `D3D12.X` COM interfaces.
pub struct VulkanBackend {
    _entry: ash::Entry,
    instance: ash::Instance,
    _physical_device: vk::PhysicalDevice,
    device: ash::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    queue_state: Mutex<QueueState>,
    memory: Mutex<Option<UnifiedMemoryManager>>,
    presenter: Mutex<Option<SwapchainPresenter>>,
    present_surface: PresentSurfaceKind,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    mesh_shader_ext: Option<ash::ext::mesh_shader::Device>,
    mesh_shader_supported: bool,
    spirv_cache: Option<Cache>,
    frame_dump_path: Option<PathBuf>,
    prefer_rebar_zero_copy: bool,
    device_name: String,
    driver_info: String,
    external_memory_host: bool,
    descriptor_buffer: bool,
    userfaultfd_mode: UserfaultfdMode,
    denorm_preserve_f32: bool,
    shaders_compiled: AtomicU64,
    frames_presented: AtomicU64,
    next_handle: AtomicU64,
    heaps: Mutex<BTreeMap<u64, HeapRecord>>,
    root_signatures: Mutex<BTreeMap<u64, ParsedRootSignature>>,
    resources: Mutex<BTreeMap<u64, ResourceBacking>>,
    samplers: Mutex<BTreeMap<u64, vk::Sampler>>,
    descriptor_heaps: Mutex<BTreeMap<u64, DescriptorHeapRecord>>,
    pipelines: Mutex<BTreeMap<u64, PipelineRecord>>,
    fences: Mutex<BTreeMap<u64, FenceRecord>>,
    last_frame: Mutex<Option<PresentedFrame>>,
}

// SAFETY: All interior mutable state inside `VulkanBackend` is protected by `Mutex` or `AtomicU64`,
// and Vulkan device handles are externally synchronized per queue/pool lock.
unsafe impl Send for VulkanBackend {}
unsafe impl Sync for VulkanBackend {}

impl VulkanBackend {
    /// Initialize a Vulkan 1.3/1.4 device, graphics queue, optional Wayland/X11 WSI swapchain,
    /// and Linux 3-tier memory manager.
    pub fn new(config: VulkanBackendConfig) -> Result<Self, GpuError> {
        // SAFETY: Loads system `libvulkan.so.1` through `libc::dlopen`.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| GpuError::Vulkan(format!("failed to load libvulkan.so.1: {e}")))?;

        let app_name = CString::new("X2Linux").expect("static ASCII");
        let engine_name = CString::new("x2l-gpu-vk").expect("static ASCII");
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // Open native Wayland/X11 window when windowed presentation is requested.
        let mut pending_window = NativeWindow::open_for_mode(config.present_mode, "X2Linux")?;
        let mut enabled_inst_exts: Vec<*const std::ffi::c_char> = Vec::new();
        let mut has_surface_maint1 = false;

        if let Some(ref win) = pending_window {
            // SAFETY: Loader is live.
            let inst_exts =
                unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
            let has_inst_ext = |target: &CStr| -> bool {
                inst_exts.iter().any(|e| {
                    // SAFETY: Vulkan extension name is NUL-terminated.
                    let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                    n == target
                })
            };
            let plat_ext = win.required_instance_extension();
            if has_inst_ext(ash::khr::surface::NAME) && has_inst_ext(plat_ext) {
                enabled_inst_exts.push(ash::khr::surface::NAME.as_ptr());
                enabled_inst_exts.push(plat_ext.as_ptr());
                if has_inst_ext(ash::khr::get_surface_capabilities2::NAME)
                    && has_inst_ext(ash::ext::surface_maintenance1::NAME)
                {
                    enabled_inst_exts.push(ash::khr::get_surface_capabilities2::NAME.as_ptr());
                    enabled_inst_exts.push(ash::ext::surface_maintenance1::NAME.as_ptr());
                    has_surface_maint1 = true;
                }
            } else if matches!(config.present_mode, PresentMode::Wayland | PresentMode::X11) {
                return Err(GpuError::Vulkan(format!(
                    "Vulkan loader missing required WSI extension {:?}",
                    plat_ext
                )));
            } else {
                warn!(
                    ext = ?plat_ext,
                    "Vulkan loader missing WSI surface extension; falling back to headless"
                );
                pending_window = None;
            }
        }

        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").expect("static ASCII");
        let mut enabled_layers = Vec::new();
        if config.enable_validation
            && let Ok(layers) = unsafe { entry.enumerate_instance_layer_properties() }
            && layers.iter().any(|l| {
                // SAFETY: Vulkan layer name is NUL-terminated.
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == validation_layer.as_c_str()
            })
        {
            enabled_layers.push(validation_layer.as_ptr());
        }

        let inst_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers)
            .enabled_extension_names(&enabled_inst_exts);
        // SAFETY: `app_info`, `enabled_layers`, and `enabled_inst_exts` stay alive across call.
        let instance = unsafe { entry.create_instance(&inst_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateInstance failed: {e:?}")))?;

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let mut pending_surface = vk::SurfaceKHR::null();

        let init_result = (|| -> Result<Self, GpuError> {
            if let Some(ref win) = pending_window {
                // SAFETY: `instance` has `VK_KHR_surface` + platform surface extension enabled.
                match unsafe { win.create_vulkan_surface(&entry, &instance) } {
                    Ok(surf) => pending_surface = surf,
                    Err(e)
                        if matches!(
                            config.present_mode,
                            PresentMode::Window | PresentMode::Auto
                        ) =>
                    {
                        warn!(error = %e, "failed to create VkSurfaceKHR; falling back to headless");
                        pending_window = None;
                    }
                    Err(e) => return Err(e),
                }
            }

            // SAFETY: `instance` is live.
            let pdevs = unsafe { instance.enumerate_physical_devices() }.map_err(|e| {
                GpuError::Vulkan(format!("vkEnumeratePhysicalDevices failed: {e:?}"))
            })?;
            if pdevs.is_empty() {
                return Err(GpuError::Vulkan("no Vulkan physical devices found".into()));
            }

            // If presenting to a surface, filter physical devices supporting `VK_KHR_swapchain`
            // and a graphics+present queue family.
            let mut eligible_pdevs: Vec<(vk::PhysicalDevice, u32)> = Vec::new();
            for &pd in &pdevs {
                // SAFETY: `pd` is live.
                let qf_props = unsafe { instance.get_physical_device_queue_family_properties(pd) };
                let exts = unsafe { instance.enumerate_device_extension_properties(pd) }
                    .unwrap_or_default();
                let has_swapchain = exts.iter().any(|e| {
                    // SAFETY: Vulkan extension name is NUL-terminated.
                    let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                    n == ash::khr::swapchain::NAME
                });
                if pending_surface != vk::SurfaceKHR::null() && !has_swapchain {
                    continue;
                }
                if let Some((qf_idx, _)) = qf_props.iter().enumerate().find(|(i, q)| {
                    if !q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        return false;
                    }
                    if pending_surface == vk::SurfaceKHR::null() {
                        return true;
                    }
                    // SAFETY: `pd` and `pending_surface` are live.
                    unsafe {
                        surface_loader
                            .get_physical_device_surface_support(pd, *i as u32, pending_surface)
                            .unwrap_or(false)
                    }
                }) {
                    eligible_pdevs.push((pd, qf_idx as u32));
                }
            }

            if eligible_pdevs.is_empty() && pending_surface != vk::SurfaceKHR::null() {
                if matches!(config.present_mode, PresentMode::Window | PresentMode::Auto) {
                    warn!("no physical device supports WSI surface; falling back to headless");
                    // SAFETY: Surface is unused; destroy before dropping window.
                    unsafe { surface_loader.destroy_surface(pending_surface, None) };
                    pending_surface = vk::SurfaceKHR::null();
                    pending_window = None;
                    for &pd in &pdevs {
                        // SAFETY: `pd` is live.
                        let qf_props =
                            unsafe { instance.get_physical_device_queue_family_properties(pd) };
                        if let Some((qf_idx, _)) = qf_props
                            .iter()
                            .enumerate()
                            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                        {
                            eligible_pdevs.push((pd, qf_idx as u32));
                        }
                    }
                } else {
                    return Err(GpuError::Vulkan(
                        "no Vulkan physical device supports the requested WSI surface".into(),
                    ));
                }
            }

            let candidate_pds: Vec<vk::PhysicalDevice> =
                eligible_pdevs.iter().map(|&(pd, _)| pd).collect();
            let physical_device =
                select_physical_device(&instance, &candidate_pds, config.device_filter.as_deref());
            let queue_family_index = eligible_pdevs
                .iter()
                .find(|&&(pd, _)| pd == physical_device)
                .map(|&(_, qf)| qf)
                .ok_or_else(|| GpuError::Vulkan("no graphics queue family found".into()))?;

            let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
            let mut float_props = vk::PhysicalDeviceFloatControlsProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut driver_props)
                .push_next(&mut float_props);
            // SAFETY: `physical_device` is live and `driver_props`/`float_props` stay on stack.
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut props2);
            }
            let props = props2.properties;
            let denorm_preserve_f32 = float_props.shader_denorm_preserve_float32 == vk::TRUE;
            // SAFETY: Vulkan device/driver strings are NUL-terminated C strings.
            let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: Vulkan driver name/info arrays are NUL-terminated.
            let drv_name = unsafe { CStr::from_ptr(driver_props.driver_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: Vulkan driver info array is NUL-terminated.
            let drv_info = unsafe { CStr::from_ptr(driver_props.driver_info.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let api_major = vk::api_version_major(props.api_version);
            let api_minor = vk::api_version_minor(props.api_version);
            let api_patch = vk::api_version_patch(props.api_version);
            let driver_info =
                format!("{drv_name} {drv_info} (Vulkan {api_major}.{api_minor}.{api_patch})")
                    .trim()
                    .to_owned();

            // SAFETY: `physical_device` is live.
            let exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .map_err(|e| {
                GpuError::Vulkan(format!(
                    "vkEnumerateDeviceExtensionProperties failed: {e:?}"
                ))
            })?;
            let has_ext = |target: &CStr| -> bool {
                exts.iter().any(|e| {
                    // SAFETY: Vulkan extension name is NUL-terminated.
                    let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                    n == target
                })
            };

            let has_ext_host = has_ext(ash::ext::external_memory_host::NAME);
            let has_desc_buf_ext = has_ext(ash::ext::descriptor_buffer::NAME);
            let has_mesh_ext = has_ext(ash::ext::mesh_shader::NAME);
            let has_swapchain_maint1_ext = pending_surface != vk::SurfaceKHR::null()
                && has_surface_maint1
                && has_ext(ash::ext::swapchain_maintenance1::NAME);

            let mut avail_v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut avail_v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut avail_desc_buf = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
            let mut avail_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut avail_sc_maint1 = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
            let mut avail_feats2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut avail_v12)
                .push_next(&mut avail_v13);
            if has_desc_buf_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_desc_buf);
            }
            if has_mesh_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_mesh);
            }
            if has_swapchain_maint1_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_sc_maint1);
            }
            // SAFETY: Feature structs live on stack for duration of query.
            unsafe {
                instance.get_physical_device_features2(physical_device, &mut avail_feats2);
            }

            let enable_bda = avail_v12.buffer_device_address == vk::TRUE;
            let enable_demote = avail_v13.shader_demote_to_helper_invocation == vk::TRUE;
            let enable_desc_buf = has_desc_buf_ext && avail_desc_buf.descriptor_buffer == vk::TRUE;
            let enable_mesh = has_mesh_ext && avail_mesh.mesh_shader == vk::TRUE;
            let enable_ext_host = has_ext_host;
            let enable_sc_maint1 =
                has_swapchain_maint1_ext && avail_sc_maint1.swapchain_maintenance1 == vk::TRUE;

            let mut enabled_exts = Vec::new();
            if enable_ext_host {
                enabled_exts.push(ash::ext::external_memory_host::NAME.as_ptr());
            }
            if enable_desc_buf {
                enabled_exts.push(ash::ext::descriptor_buffer::NAME.as_ptr());
            }
            if enable_mesh {
                enabled_exts.push(ash::ext::mesh_shader::NAME.as_ptr());
            }
            if pending_surface != vk::SurfaceKHR::null() {
                enabled_exts.push(ash::khr::swapchain::NAME.as_ptr());
            }
            if enable_sc_maint1 {
                enabled_exts.push(ash::ext::swapchain_maintenance1::NAME.as_ptr());
            }

            let mut feat12 = vk::PhysicalDeviceVulkan12Features::default()
                .timeline_semaphore(true)
                .buffer_device_address(enable_bda);
            let mut feat13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true)
                .shader_demote_to_helper_invocation(enable_demote);
            let mut feat_desc_buf =
                vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
            let mut feat_mesh =
                vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
            let mut feat_sc_maint1 = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                .swapchain_maintenance1(true);

            let priorities = [1.0f32];
            let q_ci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let mut dev_ci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&q_ci)
                .enabled_extension_names(&enabled_exts)
                .push_next(&mut feat12)
                .push_next(&mut feat13);
            if enable_desc_buf {
                dev_ci = dev_ci.push_next(&mut feat_desc_buf);
            }
            if enable_mesh {
                dev_ci = dev_ci.push_next(&mut feat_mesh);
            }
            if enable_sc_maint1 {
                dev_ci = dev_ci.push_next(&mut feat_sc_maint1);
            }

            // SAFETY: All `pNext` structs live on the stack for the duration of `create_device`.
            let device = unsafe { instance.create_device(physical_device, &dev_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateDevice failed: {e:?}")))?;
            let mesh_shader_ext = if enable_mesh {
                Some(ash::ext::mesh_shader::Device::new(&instance, &device))
            } else {
                None
            };
            // SAFETY: Queue 0 of `queue_family_index` was requested in `dev_ci`.
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            // SAFETY: `physical_device` is live.
            let mem_props =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            // SAFETY: `device` is live.
            let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateCommandPool failed: {e:?}")))?;

            let cb_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: `command_pool` is live.
            let command_buffer = unsafe { device.allocate_command_buffers(&cb_ai) }
                .map_err(|e| GpuError::Vulkan(format!("vkAllocateCommandBuffers failed: {e:?}")))?
                [0];

            let fence_ci = vk::FenceCreateInfo::default();
            // SAFETY: `device` is live.
            let submit_fence = unsafe { device.create_fence(&fence_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateFence failed: {e:?}")))?;

            // Canonical Set 0 descriptor set layout:
            //   binding 0: UniformBuffer (`RootData { uvec4[16] }`)
            //   binding 1: StorageBuffer (CBV / SRV read buffer)
            //   binding 2: StorageBuffer (UAV read/write buffer)
            //   binding 3: SampledImage (Texture2D SRV)
            //   binding 4: Sampler
            let mut all_stages = vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::COMPUTE;
            if enable_mesh {
                all_stages |= vk::ShaderStageFlags::MESH_EXT;
            }
            let dsl_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(all_stages),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(all_stages),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(all_stages),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(all_stages),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(all_stages),
            ];
            let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings);
            // SAFETY: `device` is live.
            let descriptor_set_layout = unsafe {
                device.create_descriptor_set_layout(&dsl_ci, None)
            }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateDescriptorSetLayout failed: {e:?}")))?;

            let set_layouts = [descriptor_set_layout];
            let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            // SAFETY: `device` and `descriptor_set_layout` are live.
            let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreatePipelineLayout failed: {e:?}")))?;

            let dp_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(MAX_DRAWS_PER_SUBMIT as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count((MAX_DRAWS_PER_SUBMIT * 2) as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(MAX_DRAWS_PER_SUBMIT as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(MAX_DRAWS_PER_SUBMIT as u32),
            ];
            let dp_ci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(MAX_DRAWS_PER_SUBMIT as u32)
                .pool_sizes(&dp_sizes);
            // SAFETY: `device` is live.
            let submit_descriptor_pool = unsafe { device.create_descriptor_pool(&dp_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateDescriptorPool failed: {e:?}")))?;

            // Host-visible coherent root UBO ring (256-byte-aligned slices) + 256-byte fallback SSBO.
            let root_ubo_stride = props
                .limits
                .min_uniform_buffer_offset_alignment
                .max(256)
                .next_power_of_two();
            let root_ubo_size = root_ubo_stride * (MAX_DRAWS_PER_SUBMIT as u64);
            let (root_ubo_buf, root_ubo_mem, root_ubo_ptr) = create_host_coherent_buffer(
                &device,
                &mem_props,
                root_ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let (dummy_ssbo_buf, dummy_ssbo_mem, dummy_ssbo_ptr) = create_host_coherent_buffer(
                &device,
                &mem_props,
                256,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            // SAFETY: `dummy_ssbo_ptr` is mapped for 256 bytes; zero-initialize and unmap.
            unsafe {
                ptr::write_bytes(dummy_ssbo_ptr, 0, 256);
                device.unmap_memory(dummy_ssbo_mem);
            }

            // 1x1 fallback sampled image + default nearest clamp sampler for Set 0 bindings 3 & 4.
            let dummy_img_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // SAFETY: `device` is live.
            let dummy_tex_img = unsafe { device.create_image(&dummy_img_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateImage(dummy) failed: {e:?}")))?;
            // SAFETY: `dummy_tex_img` is live.
            let d_req = unsafe { device.get_image_memory_requirements(dummy_tex_img) };
            let d_mem_type = find_memory_type(
                &mem_props,
                d_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type(
                    &mem_props,
                    d_req.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or_else(|| GpuError::Vulkan("no memory type for dummy texture".into()))?;
            // SAFETY: `device` is live.
            let dummy_tex_mem = unsafe {
                device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(d_req.size)
                        .memory_type_index(d_mem_type),
                    None,
                )
            }
            .map_err(|e| GpuError::Vulkan(format!("vkAllocateMemory(dummy) failed: {e:?}")))?;
            // SAFETY: `dummy_tex_img` and `dummy_tex_mem` are live.
            unsafe { device.bind_image_memory(dummy_tex_img, dummy_tex_mem, 0) }
                .map_err(|e| GpuError::Vulkan(format!("vkBindImageMemory(dummy) failed: {e:?}")))?;
            let dummy_view_ci = vk::ImageViewCreateInfo::default()
                .image(dummy_tex_img)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            // SAFETY: `dummy_tex_img` is bound and live.
            let dummy_tex_view = unsafe { device.create_image_view(&dummy_view_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateImageView(dummy) failed: {e:?}")))?;
            let default_samp_ci = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE);
            // SAFETY: `device` is live.
            let default_sampler = unsafe { device.create_sampler(&default_samp_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateSampler(default) failed: {e:?}")))?;

            // Create optional WSI swapchain presenter before finishing init.
            let (presenter, present_surface) = if let Some(win) = pending_window.take()
                && pending_surface != vk::SurfaceKHR::null()
            {
                let surf = pending_surface;
                pending_surface = vk::SurfaceKHR::null();
                // SAFETY: `device` has `VK_KHR_swapchain` enabled and `surf` belongs to `win`.
                match unsafe {
                    SwapchainPresenter::new(
                        &entry,
                        &instance,
                        physical_device,
                        &device,
                        queue_family_index,
                        win,
                        surf,
                        enable_sc_maint1,
                    )
                } {
                    Ok(p) => {
                        let kind = p.surface_kind();
                        (Some(p), kind)
                    }
                    Err(e)
                        if matches!(
                            config.present_mode,
                            PresentMode::Window | PresentMode::Auto
                        ) =>
                    {
                        warn!(error = %e, "WSI swapchain creation failed; falling back to headless");
                        (None, PresentSurfaceKind::Headless)
                    }
                    Err(e) => return Err(e),
                }
            } else {
                (None, PresentSurfaceKind::Headless)
            };

            // SAFETY: `instance`, `physical_device`, `device`, `queue` outlive `UnifiedMemoryManager`
            // because `Drop` explicitly drops `self.memory` before destroying `self.device`.
            let memory_mgr = unsafe {
                UnifiedMemoryManager::new(
                    instance.clone(),
                    physical_device,
                    device.clone(),
                    queue,
                    queue_family_index,
                    enable_ext_host,
                    enable_bda,
                )
            }
            .map_err(|e| GpuError::Vulkan(format!("UnifiedMemoryManager init failed: {e}")))?;

            let mem_caps = memory_mgr.capabilities();
            let spirv_cache = config.spirv_cache_dir.and_then(|dir| Cache::open(dir).ok());

            info!(
                device = %device_name,
                driver = %driver_info,
                present_surface = present_surface.as_str(),
                external_memory_host = mem_caps.external_memory_host,
                descriptor_buffer = enable_desc_buf,
                userfaultfd_mode = mem_caps.userfaultfd_mode.as_str(),
                "initialized direct Vulkan 1.3/1.4 backend"
            );

            Ok(Self {
                _entry: entry,
                instance: instance.clone(),
                _physical_device: physical_device,
                device,
                mem_props,
                queue_state: Mutex::new(QueueState {
                    queue,
                    command_pool,
                    command_buffer,
                    submit_fence,
                    submit_descriptor_pool,
                    root_ubo_buf,
                    root_ubo_mem,
                    root_ubo_ptr,
                    root_ubo_stride,
                    dummy_ssbo_buf,
                    dummy_ssbo_mem,
                    dummy_tex_img,
                    dummy_tex_mem,
                    dummy_tex_view,
                    dummy_tex_ready: false,
                    default_sampler,
                }),
                memory: Mutex::new(Some(memory_mgr)),
                presenter: Mutex::new(presenter),
                present_surface,
                descriptor_set_layout,
                pipeline_layout,
                mesh_shader_ext,
                mesh_shader_supported: enable_mesh,
                spirv_cache,
                frame_dump_path: config.frame_dump_path,
                prefer_rebar_zero_copy: config.prefer_rebar_zero_copy,
                device_name,
                driver_info,
                external_memory_host: mem_caps.external_memory_host,
                descriptor_buffer: enable_desc_buf,
                userfaultfd_mode: mem_caps.userfaultfd_mode,
                denorm_preserve_f32,
                shaders_compiled: AtomicU64::new(0),
                frames_presented: AtomicU64::new(0),
                next_handle: AtomicU64::new(1),
                heaps: Mutex::new(BTreeMap::new()),
                root_signatures: Mutex::new(BTreeMap::new()),
                resources: Mutex::new(BTreeMap::new()),
                samplers: Mutex::new(BTreeMap::new()),
                descriptor_heaps: Mutex::new(BTreeMap::new()),
                pipelines: Mutex::new(BTreeMap::new()),
                fences: Mutex::new(BTreeMap::new()),
                last_frame: Mutex::new(None),
            })
        })();

        if init_result.is_err() {
            if pending_surface != vk::SurfaceKHR::null() {
                // SAFETY: Destroy unowned surface before dropping window & instance.
                unsafe { surface_loader.destroy_surface(pending_surface, None) };
            }
            drop(pending_window);
            // SAFETY: Instance failed to finish device setup and is no longer referenced.
            unsafe { instance.destroy_instance(None) };
        }
        init_result
    }

    /// Snapshot current Vulkan device capabilities and runtime counters for `report.json`.
    pub fn telemetry_snapshot(&self) -> VulkanTelemetrySnapshot {
        VulkanTelemetrySnapshot {
            backend: "vulkan".to_owned(),
            present_surface: self.present_surface.as_str().to_owned(),
            device_name: self.device_name.clone(),
            driver_info: self.driver_info.clone(),
            external_memory_host: self.external_memory_host,
            descriptor_buffer: self.descriptor_buffer,
            userfaultfd_mode: self.userfaultfd_mode.as_str().to_owned(),
            shaders_compiled: self.shaders_compiled.load(Ordering::Relaxed),
            frames_presented: self.frames_presented.load(Ordering::Relaxed),
        }
    }

    /// Return the latest presented RGBA8 frame (if `PresentX` has been called).
    pub fn last_presented_frame(&self) -> Option<PresentedFrame> {
        self.last_frame.lock().ok().and_then(|g| g.clone())
    }

    fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed)
    }

    /// Whether `VK_EXT_mesh_shader` (`meshShader`) is enabled on this Vulkan device.
    pub fn mesh_shader_supported(&self) -> bool {
        self.mesh_shader_supported
    }

    /// Create a D3D12 root signature from a serialized DXBC `RTS0` blob.
    pub fn create_root_signature(&self, blob: &[u8]) -> Result<u64, GpuError> {
        let parsed = parse_root_signature_blob(blob).map_err(GpuError::Vulkan)?;
        let id = self.alloc_handle();
        let mut guard = self
            .root_signatures
            .lock()
            .map_err(|_| GpuError::Vulkan("root_signatures lock poisoned".into()))?;
        guard.insert(id, parsed);
        Ok(id)
    }

    /// Create a D3D12 heap backed by a single `UnifiedMemoryManager` buffer allocation.
    pub fn create_heap(
        &self,
        size: u64,
        heap_type: u32,
        alignment: u64,
        flags: u32,
    ) -> Result<u64, GpuError> {
        let alloc_len = (size.max(4096) as usize).next_multiple_of(4096);
        let policy = if self.prefer_rebar_zero_copy {
            AllocationPolicy::Auto
        } else {
            AllocationPolicy::ForceDirtyTracked
        };
        let (alloc_id, guest_va) = {
            let mut guard = self
                .memory
                .lock()
                .map_err(|_| GpuError::Vulkan("memory lock poisoned".into()))?;
            let mgr = guard
                .as_mut()
                .ok_or_else(|| GpuError::Vulkan("memory manager torn down".into()))?;
            let va = mgr
                .reserve_guest_va(alloc_len as u64)
                .map_err(|e| GpuError::Vulkan(e.to_string()))?;
            let id = mgr
                .allocate(va, alloc_len, policy)
                .map_err(|e| GpuError::Vulkan(e.to_string()))?;
            (id, va)
        };
        let id = self.alloc_handle();
        let mut heaps = self
            .heaps
            .lock()
            .map_err(|_| GpuError::Vulkan("heaps lock poisoned".into()))?;
        heaps.insert(
            id,
            HeapRecord {
                alloc_id,
                guest_va,
                size: alloc_len as u64,
                heap_type,
                alignment: alignment.max(65536),
                flags,
            },
        );
        Ok(id)
    }

    /// Create a placed buffer or 2D texture resource inside an existing D3D12 heap.
    pub fn create_placed_resource(
        &self,
        heap_id: u64,
        heap_offset: u64,
        desc: ResourceDescInfo,
    ) -> Result<u64, GpuError> {
        let heap = {
            let guard = self
                .heaps
                .lock()
                .map_err(|_| GpuError::Vulkan("heaps lock poisoned".into()))?;
            guard
                .get(&heap_id)
                .cloned()
                .ok_or_else(|| GpuError::Vulkan("unknown D3D12 heap handle".into()))?
        };
        match desc.dimension {
            D3D12_RESOURCE_DIMENSION_BUFFER => {
                let size = desc.width.max(1);
                let end = heap_offset
                    .checked_add(size)
                    .ok_or_else(|| GpuError::Vulkan("placed buffer offset overflow".into()))?;
                if end > heap.size {
                    return Err(GpuError::Vulkan(format!(
                        "placed buffer [0x{heap_offset:x}..0x{end:x}) exceeds heap size 0x{:x}",
                        heap.size
                    )));
                }
                let guest_va = heap
                    .guest_va
                    .checked_add(heap_offset)
                    .ok_or_else(|| GpuError::Vulkan("placed guest VA overflow".into()))?;
                let id = self.alloc_handle();
                let mut res_guard = self
                    .resources
                    .lock()
                    .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
                res_guard.insert(
                    id,
                    ResourceBacking::Buffer {
                        alloc_id: heap.alloc_id,
                        guest_va,
                        base_offset: heap_offset,
                        size: size as usize,
                        _is_placed: true,
                        desc,
                    },
                );
                let _ = (heap.heap_type, heap.alignment, heap.flags);
                Ok(id)
            }
            D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
                // Dedicated tracked Texture2D placement (disjoint from buffer placements).
                self.create_committed_resource(desc)
            }
            other => Err(GpuError::Vulkan(format!(
                "unsupported placed D3D12 resource dimension {other}"
            ))),
        }
    }

    /// Create a committed buffer or 2D color/depth texture resource.
    pub fn create_committed_resource(&self, desc: ResourceDescInfo) -> Result<u64, GpuError> {
        let id = self.alloc_handle();
        match desc.dimension {
            D3D12_RESOURCE_DIMENSION_BUFFER => {
                let size = desc.width.max(1) as usize;
                let policy = if self.prefer_rebar_zero_copy {
                    AllocationPolicy::Auto
                } else {
                    AllocationPolicy::ForceDirtyTracked
                };
                let mut guard = self
                    .memory
                    .lock()
                    .map_err(|_| GpuError::Vulkan("memory lock poisoned".into()))?;
                let mgr = guard
                    .as_mut()
                    .ok_or_else(|| GpuError::Vulkan("memory manager torn down".into()))?;
                let guest_va = mgr
                    .reserve_guest_va(size as u64)
                    .map_err(|e| GpuError::Vulkan(e.to_string()))?;
                let alloc_id = mgr
                    .allocate(guest_va, size, policy)
                    .map_err(|e| GpuError::Vulkan(e.to_string()))?;
                let mut res_guard = self
                    .resources
                    .lock()
                    .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
                res_guard.insert(
                    id,
                    ResourceBacking::Buffer {
                        alloc_id,
                        guest_va,
                        base_offset: 0,
                        size,
                        _is_placed: false,
                        desc,
                    },
                );
                Ok(id)
            }
            D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
                let width = (desc.width as u32).max(1);
                let height = desc.height.max(1);
                let is_depth = is_depth_dxgi_format(desc.format);
                let vk_format = dxgi_to_vk_format(desc.format);
                let aspect = if is_depth {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };
                let usage = if is_depth {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                };

                let img_ci = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk_format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(u32::from(desc.mip_levels.max(1)))
                    .array_layers(u32::from(desc.depth_or_array_size.max(1)))
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED);

                // SAFETY: `self.device` is live.
                let image = unsafe { self.device.create_image(&img_ci, None) }
                    .map_err(|e| GpuError::Vulkan(format!("vkCreateImage failed: {e:?}")))?;
                // SAFETY: `image` is live.
                let req = unsafe { self.device.get_image_memory_requirements(image) };
                let mem_type = find_memory_type(
                    &self.mem_props,
                    req.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .or_else(|| {
                    find_memory_type(
                        &self.mem_props,
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::empty(),
                    )
                })
                .ok_or_else(|| GpuError::Vulkan("no memory type for Texture2D".into()))?;

                let ai = vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type);
                // SAFETY: `self.device` is live.
                let memory = match unsafe { self.device.allocate_memory(&ai, None) } {
                    Ok(m) => m,
                    Err(e) => {
                        // SAFETY: `image` is unused.
                        unsafe { self.device.destroy_image(image, None) };
                        return Err(GpuError::Vulkan(format!(
                            "vkAllocateMemory(image) failed: {e:?}"
                        )));
                    }
                };
                // SAFETY: `image` and `memory` are live.
                if let Err(e) = unsafe { self.device.bind_image_memory(image, memory, 0) } {
                    // SAFETY: Unused handles cleaned up.
                    unsafe {
                        self.device.destroy_image(image, None);
                        self.device.free_memory(memory, None);
                    }
                    return Err(GpuError::Vulkan(format!("vkBindImageMemory failed: {e:?}")));
                }

                let view_ci = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk_format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                // SAFETY: `image` is bound and live.
                let view = match unsafe { self.device.create_image_view(&view_ci, None) } {
                    Ok(v) => v,
                    Err(e) => {
                        // SAFETY: Unused handles cleaned up.
                        unsafe {
                            self.device.destroy_image(image, None);
                            self.device.free_memory(memory, None);
                        }
                        return Err(GpuError::Vulkan(format!("vkCreateImageView failed: {e:?}")));
                    }
                };

                let mut res_guard = self
                    .resources
                    .lock()
                    .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
                res_guard.insert(
                    id,
                    ResourceBacking::Texture2D {
                        image,
                        memory,
                        view,
                        vk_format,
                        aspect,
                        width,
                        height,
                        layout: vk::ImageLayout::UNDEFINED,
                        desc,
                    },
                );
                Ok(id)
            }
            other => Err(GpuError::Vulkan(format!(
                "unsupported D3D12 resource dimension {other}"
            ))),
        }
    }

    /// Return the stored descriptor for a resource handle.
    pub fn resource_desc(&self, resource_id: u64) -> Option<ResourceDescInfo> {
        let guard = self.resources.lock().ok()?;
        guard.get(&resource_id).map(|r| match r {
            ResourceBacking::Buffer { desc, .. } | ResourceBacking::Texture2D { desc, .. } => *desc,
        })
    }

    /// Return the guest GPU virtual address of a buffer resource (`0` for textures per D3D12 spec).
    pub fn resource_gpu_va(&self, resource_id: u64) -> u64 {
        let Ok(guard) = self.resources.lock() else {
            return 0;
        };
        match guard.get(&resource_id) {
            Some(ResourceBacking::Buffer { guest_va, .. }) => *guest_va,
            _ => 0,
        }
    }

    /// Map a buffer resource (committed or placed) to a host-writable CPU pointer.
    pub fn map_resource(&self, resource_id: u64) -> Result<*mut u8, GpuError> {
        let (alloc_id, base_offset, res_size) = {
            let guard = self
                .resources
                .lock()
                .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
            match guard.get(&resource_id) {
                Some(ResourceBacking::Buffer {
                    alloc_id,
                    base_offset,
                    size,
                    ..
                }) => (*alloc_id, *base_offset, *size),
                Some(ResourceBacking::Texture2D { .. }) => {
                    return Err(GpuError::Vulkan("cannot Map an OPTIMAL Texture2D".into()));
                }
                None => return Err(GpuError::Vulkan("unknown resource handle".into())),
            }
        };
        let guard = self
            .memory
            .lock()
            .map_err(|_| GpuError::Vulkan("memory lock poisoned".into()))?;
        let mgr = guard
            .as_ref()
            .ok_or_else(|| GpuError::Vulkan("memory manager torn down".into()))?;
        // SAFETY: Resource handle remains alive while owned by COM `ID3D12Resource`.
        let (ptr, total_len) =
            unsafe { mgr.guest_ptr(alloc_id) }.map_err(|e| GpuError::Vulkan(e.to_string()))?;
        let off = base_offset as usize;
        if off.saturating_add(res_size) > total_len {
            return Err(GpuError::Vulkan(
                "placed buffer map range exceeds backing allocation".into(),
            ));
        }
        // SAFETY: `off + res_size <= total_len` validated above.
        Ok(unsafe { ptr.as_ptr().add(off) })
    }

    /// Unmap a buffer resource and flush any dirty 4 KiB pages into Vulkan.
    pub fn unmap_resource(&self, resource_id: u64) -> Result<(), GpuError> {
        let alloc_id = {
            let guard = self
                .resources
                .lock()
                .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
            match guard.get(&resource_id) {
                Some(ResourceBacking::Buffer { alloc_id, .. }) => *alloc_id,
                _ => return Ok(()),
            }
        };
        let _q = self
            .queue_state
            .lock()
            .map_err(|_| GpuError::Vulkan("queue lock poisoned".into()))?;
        let mut guard = self
            .memory
            .lock()
            .map_err(|_| GpuError::Vulkan("memory lock poisoned".into()))?;
        if let Some(mgr) = guard.as_mut() {
            // SAFETY: Guest finished writing before calling `Unmap`.
            let _ = unsafe { mgr.flush_dirty(alloc_id) }
                .map_err(|e| GpuError::Vulkan(e.to_string()))?;
        }
        Ok(())
    }

    /// Create a Vulkan `VkSampler` matching a D3D12 sampler descriptor (`ID3D12DeviceX::CreateSampler`).
    pub fn create_sampler(&self, desc: Option<SamplerDescInfo>) -> Result<u64, GpuError> {
        let map_addr = |mode: u32| -> vk::SamplerAddressMode {
            match mode {
                1 => vk::SamplerAddressMode::REPEAT,
                2 => vk::SamplerAddressMode::MIRRORED_REPEAT,
                3 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                4 => vk::SamplerAddressMode::CLAMP_TO_BORDER,
                _ => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            }
        };
        let (mag, min, mip, u, v, w, min_lod, max_lod) = if let Some(d) = desc {
            // D3D12_FILTER: bit 0 = mip linear, bit 2 = mag linear, bit 4 = min linear.
            let min_f = if (d.filter & 0x10) != 0 {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            };
            let mag_f = if (d.filter & 0x04) != 0 {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            };
            let mip_f = if (d.filter & 0x01) != 0 {
                vk::SamplerMipmapMode::LINEAR
            } else {
                vk::SamplerMipmapMode::NEAREST
            };
            (
                mag_f,
                min_f,
                mip_f,
                map_addr(d.address_u),
                map_addr(d.address_v),
                map_addr(d.address_w),
                d.min_lod,
                if d.max_lod <= 0.0 && d.min_lod <= 0.0 {
                    vk::LOD_CLAMP_NONE
                } else {
                    d.max_lod
                },
            )
        } else {
            (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                0.0,
                vk::LOD_CLAMP_NONE,
            )
        };
        let ci = vk::SamplerCreateInfo::default()
            .mag_filter(mag)
            .min_filter(min)
            .mipmap_mode(mip)
            .address_mode_u(u)
            .address_mode_v(v)
            .address_mode_w(w)
            .min_lod(min_lod)
            .max_lod(max_lod);
        // SAFETY: `self.device` is live.
        let sampler = unsafe { self.device.create_sampler(&ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateSampler failed: {e:?}")))?;
        let id = self.alloc_handle();
        let mut guard = self
            .samplers
            .lock()
            .map_err(|_| GpuError::Vulkan("samplers lock poisoned".into()))?;
        guard.insert(id, sampler);
        Ok(id)
    }

    /// Create a descriptor heap and return `(heap_id, base_cpu_handle, increment)`.
    pub fn create_descriptor_heap(
        &self,
        heap_type: u32,
        num_descriptors: u32,
    ) -> Result<(u64, usize, usize), GpuError> {
        let id = self.alloc_handle();
        let increment = 32usize;
        let base_cpu_ptr = 0x1000_0000usize
            .checked_add((id as usize).saturating_mul(0x0010_0000))
            .ok_or_else(|| GpuError::Vulkan("descriptor handle space exhausted".into()))?;
        let count = num_descriptors.max(1) as usize;
        let mut guard = self
            .descriptor_heaps
            .lock()
            .map_err(|_| GpuError::Vulkan("descriptor heap lock poisoned".into()))?;
        guard.insert(
            id,
            DescriptorHeapRecord {
                heap_type,
                num_descriptors,
                base_cpu_ptr,
                increment,
                slots: vec![DescriptorEntry::Empty; count],
            },
        );
        Ok((id, base_cpu_ptr, increment))
    }

    /// Query `(heap_type, num_descriptors, base_cpu_ptr)` for a descriptor heap.
    pub fn descriptor_heap_info(&self, heap_id: u64) -> Option<(u32, u32, usize)> {
        let guard = self.descriptor_heaps.lock().ok()?;
        let rec = guard.get(&heap_id)?;
        Some((rec.heap_type, rec.num_descriptors, rec.base_cpu_ptr))
    }

    /// Write a typed [`DescriptorEntry`] into the descriptor heap slot owning `cpu_handle`.
    pub fn write_descriptor(
        &self,
        cpu_handle: usize,
        entry: DescriptorEntry,
    ) -> Result<(), GpuError> {
        let mut guard = self
            .descriptor_heaps
            .lock()
            .map_err(|_| GpuError::Vulkan("descriptor heap lock poisoned".into()))?;
        for heap in guard.values_mut() {
            let span = heap.slots.len() * heap.increment;
            if cpu_handle >= heap.base_cpu_ptr && cpu_handle < heap.base_cpu_ptr + span {
                let offset = cpu_handle - heap.base_cpu_ptr;
                if !offset.is_multiple_of(heap.increment) {
                    return Err(GpuError::Vulkan("misaligned CPU descriptor handle".into()));
                }
                let idx = offset / heap.increment;
                heap.slots[idx] = entry;
                return Ok(());
            }
        }
        Err(GpuError::Vulkan(format!(
            "CPU descriptor handle 0x{cpu_handle:x} does not belong to any descriptor heap"
        )))
    }

    /// Read a typed [`DescriptorEntry`] from any descriptor heap by CPU/GPU handle.
    pub fn resolve_descriptor(&self, handle: usize) -> Option<DescriptorEntry> {
        let guard = self.descriptor_heaps.lock().ok()?;
        for heap in guard.values() {
            let span = heap.slots.len() * heap.increment;
            if handle >= heap.base_cpu_ptr && handle < heap.base_cpu_ptr + span {
                let offset = handle - heap.base_cpu_ptr;
                if offset.is_multiple_of(heap.increment) {
                    return Some(heap.slots[offset / heap.increment]);
                }
            }
        }
        None
    }

    /// Bind an RTV slot at `cpu_handle` to `resource_id`.
    pub fn create_render_target_view(
        &self,
        resource_id: u64,
        cpu_handle: usize,
    ) -> Result<(), GpuError> {
        self.write_descriptor(cpu_handle, DescriptorEntry::Rtv { resource_id })
    }

    /// Bind a DSV slot at `cpu_handle` to `resource_id`.
    pub fn create_depth_stencil_view(
        &self,
        resource_id: u64,
        flags: u32,
        cpu_handle: usize,
    ) -> Result<(), GpuError> {
        self.write_descriptor(cpu_handle, DescriptorEntry::Dsv { resource_id, flags })
    }

    /// Resolve a CPU RTV descriptor handle to its bound `resource_id`.
    pub fn resolve_rtv(&self, cpu_handle: usize) -> Option<u64> {
        match self.resolve_descriptor(cpu_handle)? {
            DescriptorEntry::Rtv { resource_id } => Some(resource_id),
            _ => None,
        }
    }

    /// Resolve a CPU DSV descriptor handle to its bound `resource_id`.
    pub fn resolve_dsv(&self, cpu_handle: usize) -> Option<u64> {
        match self.resolve_descriptor(cpu_handle)? {
            DescriptorEntry::Dsv { resource_id, .. } => Some(resource_id),
            _ => None,
        }
    }

    /// Backward-compatible graphics pipeline creation helper (M2.4 callers).
    pub fn create_graphics_pipeline(
        &self,
        vs_blob: &[u8],
        ps_blob: &[u8],
        rtv_format_dxgi: u32,
        has_vertex_inputs: bool,
    ) -> Result<u64, GpuError> {
        let info = GraphicsPipelineDescInfo {
            vs_blob: vs_blob.to_vec(),
            ps_blob: ps_blob.to_vec(),
            rtv_format: rtv_format_dxgi,
            dsv_format: 0,
            depth_enable: false,
            depth_write_enable: false,
            depth_func: 2, // LESS
            input_elements: if has_vertex_inputs {
                vec![InputElementInfo {
                    semantic_name: "POSITION".into(),
                    semantic_index: 0,
                    format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    input_slot: 0,
                    aligned_byte_offset: 0,
                }]
            } else {
                Vec::new()
            },
            has_root_signature: false,
        };
        self.create_graphics_pipeline_ex(&info)
    }

    /// Compile RDNA2 VS+PS bytecode directly to binary SPIR-V 1.6 (cached via `x2l-pipeline`)
    /// with full input-layout, depth-stencil, and root-binding support, and create a Vulkan 1.3
    /// Dynamic Rendering graphics pipeline.
    pub fn create_graphics_pipeline_ex(
        &self,
        info: &GraphicsPipelineDescInfo,
    ) -> Result<u64, GpuError> {
        let has_vertex_inputs = !info.input_elements.is_empty();
        let synthetic_sig = if has_vertex_inputs {
            Some(build_input_signature_from_layout(&info.input_elements))
        } else {
            None
        };
        let vs_words = self.compile_stage_spv(
            ShaderStage::Vertex,
            &info.vs_blob,
            !has_vertex_inputs,
            true,
            synthetic_sig.as_ref(),
            [0, 0, 0],
        )?;
        let ps_words = self.compile_stage_spv(
            ShaderStage::Pixel,
            &info.ps_blob,
            false,
            true,
            None,
            [0, 0, 0],
        )?;

        let vs_ci = vk::ShaderModuleCreateInfo::default().code(&vs_words);
        // SAFETY: `vs_words` is a valid binary SPIR-V 1.6 module.
        let vs_mod = unsafe { self.device.create_shader_module(&vs_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateShaderModule(VS) failed: {e:?}")))?;

        let ps_ci = vk::ShaderModuleCreateInfo::default().code(&ps_words);
        // SAFETY: `ps_words` is a valid binary SPIR-V 1.6 module.
        let ps_mod = match unsafe { self.device.create_shader_module(&ps_ci, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: `vs_mod` is unused.
                unsafe { self.device.destroy_shader_module(vs_mod, None) };
                return Err(GpuError::Vulkan(format!(
                    "vkCreateShaderModule(PS) failed: {e:?}"
                )));
            }
        };

        let entry_name = CString::new("main").expect("static ASCII");
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs_mod)
                .name(&entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(ps_mod)
                .name(&entry_name),
        ];

        // Build vertex input bindings & attributes from `D3D12_INPUT_LAYOUT_DESC`.
        let mut binding_strides: BTreeMap<u32, u32> = BTreeMap::new();
        let mut attr_descs: Vec<vk::VertexInputAttributeDescription> = Vec::new();
        for (idx, elem) in info.input_elements.iter().enumerate() {
            let slot_cursor = binding_strides.entry(elem.input_slot).or_insert(0);
            let offset = if elem.aligned_byte_offset == u32::MAX {
                *slot_cursor
            } else {
                elem.aligned_byte_offset
            };
            let elem_bytes = dxgi_format_byte_size(elem.format);
            *slot_cursor = (*slot_cursor).max(offset.saturating_add(elem_bytes));
            attr_descs.push(vk::VertexInputAttributeDescription {
                location: idx as u32,
                binding: elem.input_slot,
                format: dxgi_to_vk_format(elem.format),
                offset,
            });
        }
        let binding_descs: Vec<vk::VertexInputBindingDescription> = binding_strides
            .iter()
            .map(|(&slot, &stride)| vk::VertexInputBindingDescription {
                binding: slot,
                stride,
                input_rate: vk::VertexInputRate::VERTEX,
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descs)
            .vertex_attribute_descriptions(&attr_descs);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(info.depth_enable)
            .depth_write_enable(info.depth_enable && info.depth_write_enable)
            .depth_compare_op(d3d12_comparison_to_vk(info.depth_func))
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        if has_vertex_inputs {
            dynamic_states.push(vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE);
        }
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let rtv_vk_format = dxgi_to_vk_format(if info.rtv_format == 0 {
            DXGI_FORMAT_R8G8B8A8_UNORM
        } else {
            info.rtv_format
        });
        let dsv_vk_format = if is_depth_dxgi_format(info.dsv_format) {
            Some(vk::Format::D32_SFLOAT)
        } else {
            None
        };
        let color_formats = [rtv_vk_format];
        let mut rendering_ci =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        if let Some(dfmt) = dsv_vk_format {
            rendering_ci = rendering_ci.depth_attachment_format(dfmt);
        }

        let pso_ci = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .push_next(&mut rendering_ci)];

        // SAFETY: All stage and rendering create infos live on the stack for this call.
        let pipeline_res = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pso_ci, None)
        };

        // Always destroy temporary shader modules after pipeline creation.
        // SAFETY: `vs_mod` and `ps_mod` are no longer needed once `create_graphics_pipelines` returns.
        unsafe {
            self.device.destroy_shader_module(vs_mod, None);
            self.device.destroy_shader_module(ps_mod, None);
        }

        let pipeline = pipeline_res.map_err(|(_, e)| {
            GpuError::Vulkan(format!("vkCreateGraphicsPipelines failed: {e:?}"))
        })?[0];

        let id = self.alloc_handle();
        let mut guard = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::Vulkan("pipelines lock poisoned".into()))?;
        guard.insert(
            id,
            PipelineRecord {
                pipeline,
                kind: PipelineKind::Graphics,
                _rtv_format: rtv_vk_format,
                dsv_format: dsv_vk_format,
                has_vertex_inputs,
            },
        );
        Ok(id)
    }

    /// Compile an RDNA2 compute shader directly to binary SPIR-V 1.6 (`GLCompute`) and create
    /// a Vulkan 1.3 compute pipeline (`ID3D12DeviceX::CreateComputePipelineState`).
    pub fn create_compute_pipeline(&self, cs_blob: &[u8]) -> Result<u64, GpuError> {
        let cs_words =
            self.compile_stage_spv(ShaderStage::Compute, cs_blob, false, true, None, [64, 1, 1])?;
        let cs_ci = vk::ShaderModuleCreateInfo::default().code(&cs_words);
        // SAFETY: `cs_words` is a valid binary SPIR-V 1.6 module.
        let cs_mod = unsafe { self.device.create_shader_module(&cs_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateShaderModule(CS) failed: {e:?}")))?;

        let entry_name = CString::new("main").expect("static ASCII");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(cs_mod)
            .name(&entry_name);
        let pso_ci = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];

        // SAFETY: `stage` and `self.pipeline_layout` are valid for this call.
        let pipeline_res = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &pso_ci, None)
        };
        // SAFETY: Shader module no longer needed after pipeline creation.
        unsafe { self.device.destroy_shader_module(cs_mod, None) };

        let pipeline = pipeline_res.map_err(|(_, e)| {
            GpuError::Vulkan(format!("vkCreateComputePipelines failed: {e:?}"))
        })?[0];

        let id = self.alloc_handle();
        let mut guard = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::Vulkan("pipelines lock poisoned".into()))?;
        guard.insert(
            id,
            PipelineRecord {
                pipeline,
                kind: PipelineKind::Compute,
                _rtv_format: vk::Format::UNDEFINED,
                dsv_format: None,
                has_vertex_inputs: false,
            },
        );
        Ok(id)
    }

    fn compile_stage_spv(
        &self,
        stage: ShaderStage,
        blob: &[u8],
        synthesize_vertex_id: bool,
        d3d12_root_bindings: bool,
        override_sig: Option<&StageSignature>,
        compute_local_size: [u32; 3],
    ) -> Result<Vec<u32>, GpuError> {
        let opts = GraphicsCompileOptions {
            flip_y: stage == ShaderStage::Vertex,
            strict_float_controls: true,
            denorm_preserve: self.denorm_preserve_f32,
            d3d12_root_bindings,
            compute_local_size,
        };
        let converter = Rdna2GraphicsConverter {
            stage,
            opts,
            synthesize_vertex_id,
            override_sig: override_sig.cloned(),
        };

        let spv_bytes = if let Some(cache) = &self.spirv_cache
            && override_sig.is_none()
        {
            let mut key_input = Vec::with_capacity(blob.len() + 12);
            key_input.push(match stage {
                ShaderStage::Vertex => 1,
                ShaderStage::Pixel => 2,
                ShaderStage::Compute => 3,
                ShaderStage::Mesh => 4,
                _ => 0,
            });
            key_input.push(u8::from(opts.flip_y));
            key_input.push(u8::from(opts.strict_float_controls));
            key_input.push(u8::from(opts.denorm_preserve));
            key_input.push(u8::from(synthesize_vertex_id) | (u8::from(d3d12_root_bindings) << 1));
            key_input.extend_from_slice(blob);

            let tagged_converter = TaggedGraphicsConverter { inner: converter };
            let cached_path = cache
                .get_or_convert(&tagged_converter, &key_input)
                .map_err(|e| GpuError::Vulkan(format!("SPIR-V cache compile failed: {e}")))?;
            fs::read(&cached_path)
                .map_err(|e| GpuError::Vulkan(format!("read cached SPIR-V failed: {e}")))?
        } else {
            converter
                .convert(blob)
                .map_err(|e| GpuError::Vulkan(format!("RDNA2->SPIR-V compile failed: {e}")))?
        };

        let words = spirv_bytes_to_words(&spv_bytes)
            .map_err(|e| GpuError::Vulkan(format!("invalid cached SPIR-V bytes: {e}")))?;
        self.shaders_compiled.fetch_add(1, Ordering::Relaxed);
        Ok(words)
    }

    /// Create a Vulkan 1.2 timeline semaphore backing `ID3D12Fence`.
    pub fn create_fence(&self, initial_value: u64) -> Result<u64, GpuError> {
        let mut type_ci = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_ci);
        // SAFETY: `type_ci` stays alive across `create_semaphore`.
        let semaphore = unsafe { self.device.create_semaphore(&ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateSemaphore(timeline) failed: {e:?}")))?;
        let id = self.alloc_handle();
        let mut guard = self
            .fences
            .lock()
            .map_err(|_| GpuError::Vulkan("fences lock poisoned".into()))?;
        guard.insert(id, FenceRecord { semaphore });
        Ok(id)
    }

    /// Query `ID3D12Fence::GetCompletedValue` directly via `vkGetSemaphoreCounterValue`.
    pub fn fence_completed_value(&self, fence_id: u64) -> Result<u64, GpuError> {
        let sem = {
            let guard = self
                .fences
                .lock()
                .map_err(|_| GpuError::Vulkan("fences lock poisoned".into()))?;
            guard
                .get(&fence_id)
                .ok_or_else(|| GpuError::Vulkan("unknown fence handle".into()))?
                .semaphore
        };
        // SAFETY: `sem` is a live timeline semaphore owned by `self.device`.
        unsafe { self.device.get_semaphore_counter_value(sem) }
            .map_err(|e| GpuError::Vulkan(format!("vkGetSemaphoreCounterValue failed: {e:?}")))
    }

    /// Signal a fence from the GPU command queue (`ID3D12CommandQueue::Signal`).
    pub fn queue_signal_fence(&self, fence_id: u64, value: u64) -> Result<(), GpuError> {
        let sem = {
            let guard = self
                .fences
                .lock()
                .map_err(|_| GpuError::Vulkan("fences lock poisoned".into()))?;
            guard
                .get(&fence_id)
                .ok_or_else(|| GpuError::Vulkan("unknown fence handle".into()))?
                .semaphore
        };
        let q = self
            .queue_state
            .lock()
            .map_err(|_| GpuError::Vulkan("queue lock poisoned".into()))?;
        let sig_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(sem)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submits = [vk::SubmitInfo2::default().signal_semaphore_infos(&sig_infos)];
        // SAFETY: Queue access is serialized by `self.queue_state`; `sem` is a live timeline semaphore.
        unsafe {
            self.device
                .queue_submit2(q.queue, &submits, vk::Fence::null())
        }
        .map_err(|e| GpuError::Vulkan(format!("vkQueueSubmit2(signal) failed: {e:?}")))
    }

    /// Signal a fence from the CPU (`ID3D12Fence::Signal`).
    pub fn cpu_signal_fence(&self, fence_id: u64, value: u64) -> Result<(), GpuError> {
        let sem = {
            let guard = self
                .fences
                .lock()
                .map_err(|_| GpuError::Vulkan("fences lock poisoned".into()))?;
            guard
                .get(&fence_id)
                .ok_or_else(|| GpuError::Vulkan("unknown fence handle".into()))?
                .semaphore
        };
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(sem)
            .value(value);
        // SAFETY: `sem` is a live timeline semaphore.
        unsafe { self.device.signal_semaphore(&info) }
            .map_err(|e| GpuError::Vulkan(format!("vkSignalSemaphore failed: {e:?}")))
    }

    /// Wait until `fence_id` reaches `value` via `vkWaitSemaphores` (with queue lock released).
    pub fn wait_fence_value(&self, fence_id: u64, value: u64) -> Result<(), GpuError> {
        let sem = {
            let guard = self
                .fences
                .lock()
                .map_err(|_| GpuError::Vulkan("fences lock poisoned".into()))?;
            guard
                .get(&fence_id)
                .ok_or_else(|| GpuError::Vulkan("unknown fence handle".into()))?
                .semaphore
        };
        let sems = [sem];
        let vals = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&vals);
        // SAFETY: `sem` is a live timeline semaphore; no queue lock is held during wait.
        unsafe { self.device.wait_semaphores(&wait_info, u64::MAX) }
            .map_err(|e| GpuError::Vulkan(format!("vkWaitSemaphores failed: {e:?}")))
    }

    /// Execute one or more closed command lists on the Vulkan 1.3 graphics/compute queue.
    pub fn execute_command_lists(&self, lists: &[Vec<RecordedCommand>]) -> Result<(), GpuError> {
        let mut q = self
            .queue_state
            .lock()
            .map_err(|_| GpuError::Vulkan("queue lock poisoned".into()))?;

        // Flush any dirty guest buffer allocations before dependent GPU draws/dispatches/copies.
        if let Ok(mut mem_guard) = self.memory.lock()
            && let Some(mgr) = mem_guard.as_mut()
        {
            let mut flushed = BTreeSet::new();
            let buf_allocs: Vec<AllocationId> = self
                .resources
                .lock()
                .map(|r| {
                    r.values()
                        .filter_map(|b| match b {
                            ResourceBacking::Buffer { alloc_id, .. } => Some(*alloc_id),
                            ResourceBacking::Texture2D { .. } => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            for alloc_id in buf_allocs {
                if flushed.insert(alloc_id) {
                    // SAFETY: Command list submission guarantees guest writers finished updating buffers.
                    let _ = unsafe { mgr.flush_dirty(alloc_id) };
                }
            }
        }

        let cmd = q.command_buffer;
        // SAFETY: `q` exclusively owns `cmd`, `submit_fence`, and `submit_descriptor_pool` (all idle).
        unsafe {
            self.device
                .reset_fences(&[q.submit_fence])
                .map_err(|e| GpuError::Vulkan(format!("vkResetFences failed: {e:?}")))?;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| GpuError::Vulkan(format!("vkResetCommandBuffer failed: {e:?}")))?;
            self.device
                .reset_descriptor_pool(
                    q.submit_descriptor_pool,
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .map_err(|e| GpuError::Vulkan(format!("vkResetDescriptorPool failed: {e:?}")))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| GpuError::Vulkan(format!("vkBeginCommandBuffer failed: {e:?}")))?;
        }

        if !q.dummy_tex_ready {
            transition_image_layout_aspect(
                &self.device,
                cmd,
                q.dummy_tex_img,
                vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            q.dummy_tex_ready = true;
        }

        let mut resources = self
            .resources
            .lock()
            .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
        let samplers = self
            .samplers
            .lock()
            .map_err(|_| GpuError::Vulkan("samplers lock poisoned".into()))?;
        let pipelines = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::Vulkan("pipelines lock poisoned".into()))?;
        let root_sigs = self
            .root_signatures
            .lock()
            .map_err(|_| GpuError::Vulkan("root_signatures lock poisoned".into()))?;
        let mem_guard = self
            .memory
            .lock()
            .map_err(|_| GpuError::Vulkan("memory lock poisoned".into()))?;

        #[derive(Clone)]
        struct BindPointState {
            root_sig_id: Option<u64>,
            root_dwords: [u32; 64],
            cbv_srv: Option<(vk::Buffer, u64, u64)>,
            uav: Option<(vk::Buffer, u64, u64, AllocationId)>,
            sampled_tex_id: Option<u64>,
            sampler_id: Option<u64>,
        }
        impl Default for BindPointState {
            fn default() -> Self {
                Self {
                    root_sig_id: None,
                    root_dwords: [0u32; 64],
                    cbv_srv: None,
                    uav: None,
                    sampled_tex_id: None,
                    sampler_id: None,
                }
            }
        }

        let resolve_va_range =
            |va: u64, min_size: u64| -> Option<(vk::Buffer, u64, u64, AllocationId)> {
                let mgr = mem_guard.as_ref()?;
                let res = mgr.registry().lookup(va)?;
                let total_len = mgr.allocation_len(res.allocation_id).ok()? as u64;
                let rem = total_len.saturating_sub(res.offset);
                let range = if min_size == 0 {
                    rem.max(4)
                } else {
                    min_size.min(rem).max(4)
                };
                Some((res.buffer, res.offset, range, res.allocation_id))
            };

        let resolve_res_buf = |res_map: &BTreeMap<u64, ResourceBacking>,
                               res_id: u64|
         -> Option<(vk::Buffer, u64, u64, AllocationId)> {
            let mgr = mem_guard.as_ref()?;
            let ResourceBacking::Buffer {
                alloc_id,
                base_offset,
                size,
                ..
            } = res_map.get(&res_id)?
            else {
                return None;
            };
            let (buf, _) = mgr.allocation_vulkan(*alloc_id).ok()?;
            Some((buf, *base_offset, (*size as u64).max(4), *alloc_id))
        };

        let mut ubo_slice_idx = 0usize;
        let mut gpu_written_allocs: BTreeSet<AllocationId> = BTreeSet::new();

        let alloc_bind_set = |slice_idx: &mut usize,
                              bp: &BindPointState,
                              bind_point: vk::PipelineBindPoint,
                              sampled_view: Option<vk::ImageView>|
         -> Result<(), GpuError> {
            let idx = *slice_idx;
            if idx >= MAX_DRAWS_PER_SUBMIT {
                return Err(GpuError::Vulkan(
                    "exceeded MAX_DRAWS_PER_SUBMIT descriptor slices".into(),
                ));
            }
            *slice_idx += 1;
            let ubo_off = (idx as u64) * q.root_ubo_stride;
            // SAFETY: `q.root_ubo_ptr` is mapped for `MAX_DRAWS_PER_SUBMIT * root_ubo_stride` bytes
            // and `bp.root_dwords` is 256 bytes (`<= root_ubo_stride`).
            unsafe {
                ptr::copy_nonoverlapping(
                    bp.root_dwords.as_ptr().cast::<u8>(),
                    q.root_ubo_ptr.add(ubo_off as usize),
                    256,
                );
            }
            let set_layouts = [self.descriptor_set_layout];
            let ds_ai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(q.submit_descriptor_pool)
                .set_layouts(&set_layouts);
            // SAFETY: Pool has capacity for `MAX_DRAWS_PER_SUBMIT` sets.
            let set = unsafe { self.device.allocate_descriptor_sets(&ds_ai) }
                .map_err(|e| GpuError::Vulkan(format!("vkAllocateDescriptorSets failed: {e:?}")))?
                [0];

            let (cbv_buf, cbv_off, cbv_rng) = bp.cbv_srv.unwrap_or((q.dummy_ssbo_buf, 0, 256));
            let (uav_buf, uav_off, uav_rng) =
                bp.uav
                    .map(|(b, o, r, _)| (b, o, r))
                    .unwrap_or((q.dummy_ssbo_buf, 0, 256));
            let tex_view = sampled_view.unwrap_or(q.dummy_tex_view);
            let sampler = bp
                .sampler_id
                .and_then(|id| samplers.get(&id).copied())
                .unwrap_or(q.default_sampler);

            let b0 = [vk::DescriptorBufferInfo::default()
                .buffer(q.root_ubo_buf)
                .offset(ubo_off)
                .range(256)];
            let b1 = [vk::DescriptorBufferInfo::default()
                .buffer(cbv_buf)
                .offset(cbv_off)
                .range(cbv_rng)];
            let b2 = [vk::DescriptorBufferInfo::default()
                .buffer(uav_buf)
                .offset(uav_off)
                .range(uav_rng)];
            let b3 = [vk::DescriptorImageInfo::default()
                .image_view(tex_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let b4 = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&b0),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&b1),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&b2),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&b3),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&b4),
            ];
            // SAFETY: Descriptor set, buffers, image view, and sampler are live for this submission.
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    bind_point,
                    self.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
            }
            Ok(())
        };

        let root_param_dword_base = |sig_id: Option<u64>, param_idx: u32| -> u32 {
            sig_id
                .and_then(|id| root_sigs.get(&id))
                .and_then(|sig| sig.parameters.get(param_idx as usize))
                .map(|p| p.dword_offset)
                .unwrap_or(param_idx.saturating_mul(4))
        };

        let mut current_rt: Option<u64> = None;
        let mut current_dsv: Option<u64> = None;
        let mut current_pso: Option<u64> = None;
        let mut current_viewport: Option<vk::Viewport> = None;
        let mut current_scissor: Option<vk::Rect2D> = None;
        let mut gfx_bp = BindPointState::default();
        let mut cmp_bp = BindPointState::default();
        let mut vertex_buffers: BTreeMap<u32, (u64, u32, u32)> = BTreeMap::new();
        let mut index_buffer: Option<(u64, u32, u32)> = None;

        for list in lists {
            for op in list {
                match op {
                    RecordedCommand::ClearRenderTargetView { resource_id, color } => {
                        if let Some(ResourceBacking::Texture2D {
                            image,
                            layout,
                            aspect,
                            ..
                        }) = resources.get_mut(resource_id)
                        {
                            transition_image_layout_aspect(
                                &self.device,
                                cmd,
                                *image,
                                *aspect,
                                *layout,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

                            let clear_val = vk::ClearColorValue { float32: *color };
                            let range = [vk::ImageSubresourceRange {
                                aspect_mask: *aspect,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }];
                            // SAFETY: `image` is in `TRANSFER_DST_OPTIMAL` and created with `TRANSFER_DST`.
                            unsafe {
                                self.device.cmd_clear_color_image(
                                    cmd,
                                    *image,
                                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    &clear_val,
                                    &range,
                                );
                            }
                        }
                    }
                    RecordedCommand::ClearDepthStencilView { resource_id, depth } => {
                        if let Some(ResourceBacking::Texture2D {
                            image,
                            layout,
                            aspect,
                            ..
                        }) = resources.get_mut(resource_id)
                        {
                            transition_image_layout_aspect(
                                &self.device,
                                cmd,
                                *image,
                                *aspect,
                                *layout,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

                            let clear_val = vk::ClearDepthStencilValue {
                                depth: *depth,
                                stencil: 0,
                            };
                            let range = [vk::ImageSubresourceRange {
                                aspect_mask: *aspect,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }];
                            // SAFETY: Depth image is in `TRANSFER_DST_OPTIMAL` with `TRANSFER_DST`.
                            unsafe {
                                self.device.cmd_clear_depth_stencil_image(
                                    cmd,
                                    *image,
                                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    &clear_val,
                                    &range,
                                );
                            }
                        }
                    }
                    RecordedCommand::SetViewports(vps) => {
                        current_viewport = vps.first().copied();
                    }
                    RecordedCommand::SetScissors(scs) => {
                        current_scissor = scs.first().copied();
                    }
                    RecordedCommand::SetRenderTargets { rtvs, dsv } => {
                        current_rt = rtvs.first().copied();
                        current_dsv = *dsv;
                    }
                    RecordedCommand::SetPipelineState(pso_id) => {
                        current_pso = Some(*pso_id);
                    }
                    RecordedCommand::SetPrimitiveTopology(_)
                    | RecordedCommand::SetDescriptorHeaps(_)
                    | RecordedCommand::ResourceBarrier { .. } => {}
                    RecordedCommand::SetGraphicsRootSignature(sig_id) => {
                        gfx_bp = BindPointState {
                            root_sig_id: Some(*sig_id),
                            ..Default::default()
                        };
                    }
                    RecordedCommand::SetComputeRootSignature(sig_id) => {
                        cmp_bp = BindPointState {
                            root_sig_id: Some(*sig_id),
                            ..Default::default()
                        };
                    }
                    RecordedCommand::SetGraphicsRoot32BitConstants {
                        root_param_index,
                        values,
                        dest_offset,
                    } => {
                        let base = root_param_dword_base(gfx_bp.root_sig_id, *root_param_index)
                            as usize
                            + (*dest_offset as usize);
                        for (i, &val) in values.iter().enumerate() {
                            if base + i < gfx_bp.root_dwords.len() {
                                gfx_bp.root_dwords[base + i] = val;
                            }
                        }
                    }
                    RecordedCommand::SetComputeRoot32BitConstants {
                        root_param_index,
                        values,
                        dest_offset,
                    } => {
                        let base = root_param_dword_base(cmp_bp.root_sig_id, *root_param_index)
                            as usize
                            + (*dest_offset as usize);
                        for (i, &val) in values.iter().enumerate() {
                            if base + i < cmp_bp.root_dwords.len() {
                                cmp_bp.root_dwords[base + i] = val;
                            }
                        }
                    }
                    RecordedCommand::SetGraphicsRootConstantBufferView {
                        root_param_index,
                        gpu_va,
                    }
                    | RecordedCommand::SetGraphicsRootShaderResourceView {
                        root_param_index,
                        gpu_va,
                    } => {
                        if let Some((buf, off, rng, _)) = resolve_va_range(*gpu_va, 0) {
                            gfx_bp.cbv_srv = Some((buf, off, rng));
                            let base = root_param_dword_base(gfx_bp.root_sig_id, *root_param_index)
                                as usize;
                            if base + 1 < gfx_bp.root_dwords.len() {
                                gfx_bp.root_dwords[base] = 0;
                                gfx_bp.root_dwords[base + 1] = 0;
                            }
                        }
                    }
                    RecordedCommand::SetComputeRootConstantBufferView {
                        root_param_index,
                        gpu_va,
                    }
                    | RecordedCommand::SetComputeRootShaderResourceView {
                        root_param_index,
                        gpu_va,
                    } => {
                        if let Some((buf, off, rng, _)) = resolve_va_range(*gpu_va, 0) {
                            cmp_bp.cbv_srv = Some((buf, off, rng));
                            let base = root_param_dword_base(cmp_bp.root_sig_id, *root_param_index)
                                as usize;
                            if base + 1 < cmp_bp.root_dwords.len() {
                                cmp_bp.root_dwords[base] = 0;
                                cmp_bp.root_dwords[base + 1] = 0;
                            }
                        }
                    }
                    RecordedCommand::SetGraphicsRootUnorderedAccessView {
                        root_param_index,
                        gpu_va,
                    } => {
                        if let Some((buf, off, rng, alloc_id)) = resolve_va_range(*gpu_va, 0) {
                            gfx_bp.uav = Some((buf, off, rng, alloc_id));
                            let base = root_param_dword_base(gfx_bp.root_sig_id, *root_param_index)
                                as usize;
                            if base + 1 < gfx_bp.root_dwords.len() {
                                gfx_bp.root_dwords[base] = 0;
                                gfx_bp.root_dwords[base + 1] = 0;
                            }
                        }
                    }
                    RecordedCommand::SetComputeRootUnorderedAccessView {
                        root_param_index,
                        gpu_va,
                    } => {
                        if let Some((buf, off, rng, alloc_id)) = resolve_va_range(*gpu_va, 0) {
                            cmp_bp.uav = Some((buf, off, rng, alloc_id));
                            let base = root_param_dword_base(cmp_bp.root_sig_id, *root_param_index)
                                as usize;
                            if base + 1 < cmp_bp.root_dwords.len() {
                                cmp_bp.root_dwords[base] = 0;
                                cmp_bp.root_dwords[base + 1] = 0;
                            }
                        }
                    }
                    RecordedCommand::SetGraphicsRootDescriptorTable { gpu_handle, .. } => {
                        if let Some(entry) = self.resolve_descriptor(*gpu_handle as usize) {
                            match entry {
                                DescriptorEntry::Cbv {
                                    gpu_va,
                                    size_in_bytes,
                                } => {
                                    if let Some((b, o, r, _)) =
                                        resolve_va_range(gpu_va, u64::from(size_in_bytes))
                                    {
                                        gfx_bp.cbv_srv = Some((b, o, r));
                                    }
                                }
                                DescriptorEntry::Srv { resource_id, .. } => {
                                    match resources.get(&resource_id) {
                                        Some(ResourceBacking::Texture2D { .. }) => {
                                            gfx_bp.sampled_tex_id = Some(resource_id);
                                        }
                                        _ => {
                                            if let Some((b, o, r, _)) =
                                                resolve_res_buf(&resources, resource_id)
                                            {
                                                gfx_bp.cbv_srv = Some((b, o, r));
                                            }
                                        }
                                    }
                                }
                                DescriptorEntry::Uav { resource_id, .. } => {
                                    if let Some(u) = resolve_res_buf(&resources, resource_id) {
                                        gfx_bp.uav = Some(u);
                                    }
                                }
                                DescriptorEntry::Sampler { sampler_id } => {
                                    gfx_bp.sampler_id = Some(sampler_id);
                                }
                                _ => {}
                            }
                        }
                    }
                    RecordedCommand::SetComputeRootDescriptorTable { gpu_handle, .. } => {
                        if let Some(entry) = self.resolve_descriptor(*gpu_handle as usize) {
                            match entry {
                                DescriptorEntry::Cbv {
                                    gpu_va,
                                    size_in_bytes,
                                } => {
                                    if let Some((b, o, r, _)) =
                                        resolve_va_range(gpu_va, u64::from(size_in_bytes))
                                    {
                                        cmp_bp.cbv_srv = Some((b, o, r));
                                    }
                                }
                                DescriptorEntry::Srv { resource_id, .. } => {
                                    match resources.get(&resource_id) {
                                        Some(ResourceBacking::Texture2D { .. }) => {
                                            cmp_bp.sampled_tex_id = Some(resource_id);
                                        }
                                        _ => {
                                            if let Some((b, o, r, _)) =
                                                resolve_res_buf(&resources, resource_id)
                                            {
                                                cmp_bp.cbv_srv = Some((b, o, r));
                                            }
                                        }
                                    }
                                }
                                DescriptorEntry::Uav { resource_id, .. } => {
                                    if let Some(u) = resolve_res_buf(&resources, resource_id) {
                                        cmp_bp.uav = Some(u);
                                    }
                                }
                                DescriptorEntry::Sampler { sampler_id } => {
                                    cmp_bp.sampler_id = Some(sampler_id);
                                }
                                _ => {}
                            }
                        }
                    }
                    RecordedCommand::IASetVertexBuffers { start_slot, views } => {
                        for (i, &view) in views.iter().enumerate() {
                            vertex_buffers.insert(start_slot + i as u32, view);
                        }
                    }
                    RecordedCommand::IASetIndexBuffer {
                        gpu_va,
                        size_in_bytes,
                        format,
                    } => {
                        index_buffer = Some((*gpu_va, *size_in_bytes, *format));
                    }
                    RecordedCommand::DrawInstanced {
                        vertex_count,
                        instance_count,
                        start_vertex,
                        start_instance,
                    }
                    | RecordedCommand::DrawIndexedInstanced {
                        index_count: vertex_count,
                        instance_count,
                        start_index: start_vertex,
                        start_instance,
                        ..
                    } => {
                        let is_indexed = matches!(op, RecordedCommand::DrawIndexedInstanced { .. });
                        let base_vertex = match op {
                            RecordedCommand::DrawIndexedInstanced { base_vertex, .. } => {
                                *base_vertex
                            }
                            _ => 0,
                        };
                        let Some(rt_id) = current_rt else {
                            warn!("Draw called without an active render target");
                            continue;
                        };
                        let Some(pso_id) = current_pso else {
                            warn!("Draw called without a bound pipeline state");
                            continue;
                        };
                        let Some(pso) = pipelines.get(&pso_id) else {
                            warn!(pso_id, "Draw references unknown PSO");
                            continue;
                        };
                        if pso.kind != PipelineKind::Graphics {
                            warn!(pso_id, "Draw called with non-graphics PSO");
                            continue;
                        }

                        // Transition RTV to COLOR_ATTACHMENT_OPTIMAL.
                        let (rt_view, rt_w, rt_h) = match resources.get_mut(&rt_id) {
                            Some(ResourceBacking::Texture2D {
                                image,
                                view,
                                width,
                                height,
                                layout,
                                aspect,
                                ..
                            }) => {
                                transition_image_layout_aspect(
                                    &self.device,
                                    cmd,
                                    *image,
                                    *aspect,
                                    *layout,
                                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                );
                                *layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                                (*view, *width, *height)
                            }
                            _ => {
                                warn!(rt_id, "Draw render target is not a Texture2D");
                                continue;
                            }
                        };

                        // Transition optional DSV to DEPTH_ATTACHMENT_OPTIMAL if PSO expects depth.
                        let depth_view = if pso.dsv_format.is_some()
                            && let Some(dsv_id) = current_dsv
                            && let Some(ResourceBacking::Texture2D {
                                image,
                                view,
                                layout,
                                aspect,
                                ..
                            }) = resources.get_mut(&dsv_id)
                        {
                            transition_image_layout_aspect(
                                &self.device,
                                cmd,
                                *image,
                                *aspect,
                                *layout,
                                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
                            Some(*view)
                        } else {
                            None
                        };

                        // Transition optional sampled Texture2D SRV to SHADER_READ_ONLY_OPTIMAL
                        // before beginning Vulkan 1.3 Dynamic Rendering.
                        let sampled_view = if let Some(tex_id) = gfx_bp.sampled_tex_id
                            && tex_id != rt_id
                            && let Some(ResourceBacking::Texture2D {
                                image,
                                view,
                                layout,
                                aspect,
                                ..
                            }) = resources.get_mut(&tex_id)
                        {
                            transition_image_layout_aspect(
                                &self.device,
                                cmd,
                                *image,
                                *aspect,
                                *layout,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                            Some(*view)
                        } else {
                            None
                        };

                        let color_attachment = [vk::RenderingAttachmentInfo::default()
                            .image_view(rt_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)];
                        let depth_attachment = depth_view.map(|dv| {
                            vk::RenderingAttachmentInfo::default()
                                .image_view(dv)
                                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)
                        });
                        let render_area = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: rt_w,
                                height: rt_h,
                            },
                        };
                        let mut rendering_info = vk::RenderingInfo::default()
                            .render_area(render_area)
                            .layer_count(1)
                            .color_attachments(&color_attachment);
                        if let Some(ref da) = depth_attachment {
                            rendering_info = rendering_info.depth_attachment(da);
                        }

                        let vp = current_viewport.unwrap_or(vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: rt_w as f32,
                            height: rt_h as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        });
                        let sc = current_scissor.unwrap_or(render_area);

                        // SAFETY: Images are transitioned and PSO matches attachment formats.
                        unsafe {
                            self.device.cmd_begin_rendering(cmd, &rendering_info);
                            self.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pso.pipeline,
                            );
                            self.device.cmd_set_viewport(cmd, 0, &[vp]);
                            self.device.cmd_set_scissor(cmd, 0, &[sc]);
                        }
                        alloc_bind_set(
                            &mut ubo_slice_idx,
                            &gfx_bp,
                            vk::PipelineBindPoint::GRAPHICS,
                            sampled_view,
                        )?;

                        if pso.has_vertex_inputs {
                            for (&slot, &(vb_va, vb_sz, vb_stride)) in &vertex_buffers {
                                if let Some((buf, off, rng, _)) =
                                    resolve_va_range(vb_va, u64::from(vb_sz))
                                {
                                    // SAFETY: `buf` was created with `VERTEX_BUFFER` usage.
                                    unsafe {
                                        self.device.cmd_bind_vertex_buffers2(
                                            cmd,
                                            slot,
                                            &[buf],
                                            &[off],
                                            Some(&[rng]),
                                            Some(&[u64::from(vb_stride)]),
                                        );
                                    }
                                }
                            }
                        }

                        if is_indexed {
                            if let Some((ib_va, ib_sz, ib_fmt)) = index_buffer
                                && let Some((buf, off, _, _)) =
                                    resolve_va_range(ib_va, u64::from(ib_sz))
                            {
                                let idx_ty = if ib_fmt == DXGI_FORMAT_R16_UINT {
                                    vk::IndexType::UINT16
                                } else {
                                    vk::IndexType::UINT32
                                };
                                // SAFETY: `buf` was created with `INDEX_BUFFER` usage.
                                unsafe {
                                    self.device.cmd_bind_index_buffer(cmd, buf, off, idx_ty);
                                    self.device.cmd_draw_indexed(
                                        cmd,
                                        *vertex_count,
                                        *instance_count,
                                        *start_vertex,
                                        base_vertex,
                                        *start_instance,
                                    );
                                }
                            } else {
                                warn!("DrawIndexedInstanced called without a valid index buffer");
                            }
                        } else {
                            // SAFETY: Graphics pipeline, descriptor set, viewport, and scissor bound.
                            unsafe {
                                self.device.cmd_draw(
                                    cmd,
                                    *vertex_count,
                                    *instance_count,
                                    *start_vertex,
                                    *start_instance,
                                );
                            }
                        }

                        // SAFETY: Balances `cmd_begin_rendering`.
                        unsafe {
                            self.device.cmd_end_rendering(cmd);
                        }
                    }
                    RecordedCommand::Dispatch { x, y, z } => {
                        let Some(pso_id) = current_pso else {
                            warn!("Dispatch called without a bound compute PSO");
                            continue;
                        };
                        let Some(pso) = pipelines.get(&pso_id) else {
                            warn!(pso_id, "Dispatch references unknown PSO");
                            continue;
                        };
                        if pso.kind != PipelineKind::Compute {
                            warn!(pso_id, "Dispatch called with non-compute PSO");
                            continue;
                        }
                        let cmp_sampled_view = if let Some(tex_id) = cmp_bp.sampled_tex_id
                            && let Some(ResourceBacking::Texture2D {
                                image,
                                view,
                                layout,
                                aspect,
                                ..
                            }) = resources.get_mut(&tex_id)
                        {
                            transition_image_layout_aspect(
                                &self.device,
                                cmd,
                                *image,
                                *aspect,
                                *layout,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                            Some(*view)
                        } else {
                            None
                        };
                        // SAFETY: Compute PSO is valid.
                        unsafe {
                            self.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                pso.pipeline,
                            );
                        }
                        alloc_bind_set(
                            &mut ubo_slice_idx,
                            &cmp_bp,
                            vk::PipelineBindPoint::COMPUTE,
                            cmp_sampled_view,
                        )?;
                        if let Some((_, _, _, uav_alloc)) = cmp_bp.uav {
                            gpu_written_allocs.insert(uav_alloc);
                        }
                        // SAFETY: Compute pipeline & Set 0 descriptors are bound; follow with
                        // Synchronization2 barrier for downstream shader/copy/host visibility.
                        unsafe {
                            self.device.cmd_dispatch(cmd, *x, *y, *z);
                            let mb = [vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                                .dst_stage_mask(
                                    vk::PipelineStageFlags2::ALL_COMMANDS
                                        | vk::PipelineStageFlags2::HOST,
                                )
                                .dst_access_mask(
                                    vk::AccessFlags2::MEMORY_READ
                                        | vk::AccessFlags2::MEMORY_WRITE
                                        | vk::AccessFlags2::HOST_READ,
                                )];
                            let dep = vk::DependencyInfo::default().memory_barriers(&mb);
                            self.device.cmd_pipeline_barrier2(cmd, &dep);
                        }
                    }
                    RecordedCommand::DispatchMesh { x, y, z } => {
                        let Some(ref mesh_ext) = self.mesh_shader_ext else {
                            warn!("DispatchMesh ignored: VK_EXT_mesh_shader unavailable");
                            continue;
                        };
                        let Some(rt_id) = current_rt else {
                            continue;
                        };
                        let Some(pso_id) = current_pso else {
                            continue;
                        };
                        let Some(pso) = pipelines.get(&pso_id) else {
                            continue;
                        };
                        if pso.kind != PipelineKind::Mesh {
                            continue;
                        }
                        let (rt_view, rt_w, rt_h) = match resources.get_mut(&rt_id) {
                            Some(ResourceBacking::Texture2D {
                                image,
                                view,
                                width,
                                height,
                                layout,
                                aspect,
                                ..
                            }) => {
                                transition_image_layout_aspect(
                                    &self.device,
                                    cmd,
                                    *image,
                                    *aspect,
                                    *layout,
                                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                                );
                                *layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
                                (*view, *width, *height)
                            }
                            _ => continue,
                        };
                        // Transition optional sampled Texture2D SRV to SHADER_READ_ONLY_OPTIMAL
                        // before beginning Vulkan 1.3 Dynamic Rendering.
                        let sampled_view = if let Some(tex_id) = gfx_bp.sampled_tex_id
                            && tex_id != rt_id
                            && let Some(ResourceBacking::Texture2D {
                                image,
                                view,
                                layout,
                                aspect,
                                ..
                            }) = resources.get_mut(&tex_id)
                        {
                            transition_image_layout_aspect(
                                &self.device,
                                cmd,
                                *image,
                                *aspect,
                                *layout,
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                            Some(*view)
                        } else {
                            None
                        };

                        let color_attachment = [vk::RenderingAttachmentInfo::default()
                            .image_view(rt_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)];
                        let render_area = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: rt_w,
                                height: rt_h,
                            },
                        };
                        let rendering_info = vk::RenderingInfo::default()
                            .render_area(render_area)
                            .layer_count(1)
                            .color_attachments(&color_attachment);
                        let vp = current_viewport.unwrap_or(vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: rt_w as f32,
                            height: rt_h as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        });
                        let sc = current_scissor.unwrap_or(render_area);
                        // SAFETY: Mesh pipeline & VK_EXT_mesh_shader enabled.
                        unsafe {
                            self.device.cmd_begin_rendering(cmd, &rendering_info);
                            self.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pso.pipeline,
                            );
                            self.device.cmd_set_viewport(cmd, 0, &[vp]);
                            self.device.cmd_set_scissor(cmd, 0, &[sc]);
                        }
                        alloc_bind_set(
                            &mut ubo_slice_idx,
                            &gfx_bp,
                            vk::PipelineBindPoint::GRAPHICS,
                            sampled_view,
                        )?;
                        // SAFETY: Inside dynamic rendering with valid mesh pipeline bound.
                        unsafe {
                            mesh_ext.cmd_draw_mesh_tasks(cmd, *x, *y, *z);
                            self.device.cmd_end_rendering(cmd);
                        }
                    }
                    RecordedCommand::CopyBufferRegion {
                        dst_res_id,
                        dst_offset,
                        src_res_id,
                        src_offset,
                        num_bytes,
                    } => {
                        if *num_bytes == 0 {
                            continue;
                        }
                        let Some((dst_buf, dst_base, dst_sz, dst_alloc)) =
                            resolve_res_buf(&resources, *dst_res_id)
                        else {
                            warn!(dst_res_id, "CopyBufferRegion unknown dst buffer");
                            continue;
                        };
                        let Some((src_buf, src_base, src_sz, _)) =
                            resolve_res_buf(&resources, *src_res_id)
                        else {
                            warn!(src_res_id, "CopyBufferRegion unknown src buffer");
                            continue;
                        };
                        if dst_offset.saturating_add(*num_bytes) > dst_sz
                            || src_offset.saturating_add(*num_bytes) > src_sz
                        {
                            warn!("CopyBufferRegion out of bounds");
                            continue;
                        }
                        let real_src_off = src_base + *src_offset;
                        let real_dst_off = dst_base + *dst_offset;
                        if src_buf == dst_buf
                            && real_src_off < real_dst_off + *num_bytes
                            && real_dst_off < real_src_off + *num_bytes
                        {
                            warn!("CopyBufferRegion overlapping physical range rejected");
                            continue;
                        }
                        let region = [vk::BufferCopy2::default()
                            .src_offset(real_src_off)
                            .dst_offset(real_dst_off)
                            .size(*num_bytes)];
                        let copy_info = vk::CopyBufferInfo2::default()
                            .src_buffer(src_buf)
                            .dst_buffer(dst_buf)
                            .regions(&region);
                        gpu_written_allocs.insert(dst_alloc);
                        // SAFETY: Non-overlapping validated buffer ranges with TRANSFER_SRC/DST.
                        unsafe {
                            self.device.cmd_copy_buffer2(cmd, &copy_info);
                            let mb = [vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_stage_mask(
                                    vk::PipelineStageFlags2::ALL_COMMANDS
                                        | vk::PipelineStageFlags2::HOST,
                                )
                                .dst_access_mask(
                                    vk::AccessFlags2::MEMORY_READ
                                        | vk::AccessFlags2::MEMORY_WRITE
                                        | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                                        | vk::AccessFlags2::INDEX_READ
                                        | vk::AccessFlags2::HOST_READ,
                                )];
                            let dep = vk::DependencyInfo::default().memory_barriers(&mb);
                            self.device.cmd_pipeline_barrier2(cmd, &dep);
                        }
                    }
                    RecordedCommand::CopyTextureRegion {
                        dst_res_id,
                        dst_is_footprint,
                        dst_offset,
                        dst_row_pitch,
                        dst_x,
                        dst_y,
                        src_res_id,
                        src_is_footprint,
                        src_offset,
                        src_row_pitch,
                        width,
                        height,
                    } => {
                        match (*src_is_footprint, *dst_is_footprint) {
                            (true, false) => {
                                // Buffer -> Texture2D
                                let Some((src_buf, src_base, _, _)) =
                                    resolve_res_buf(&resources, *src_res_id)
                                else {
                                    continue;
                                };
                                if let Some(ResourceBacking::Texture2D {
                                    image,
                                    layout,
                                    aspect,
                                    width: tw,
                                    height: th,
                                    vk_format,
                                    ..
                                }) = resources.get_mut(dst_res_id)
                                {
                                    transition_image_layout_aspect(
                                        &self.device,
                                        cmd,
                                        *image,
                                        *aspect,
                                        *layout,
                                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    );
                                    *layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                                    let bpp = vk_format_bytes_per_pixel(*vk_format).max(1);
                                    let row_texels = (*src_row_pitch) / bpp;
                                    let cw = if *width == 0 { *tw } else { *width };
                                    let ch = if *height == 0 { *th } else { *height };
                                    let reg = [vk::BufferImageCopy2::default()
                                        .buffer_offset(src_base + *src_offset)
                                        .buffer_row_length(row_texels)
                                        .buffer_image_height(0)
                                        .image_subresource(vk::ImageSubresourceLayers {
                                            aspect_mask: *aspect,
                                            mip_level: 0,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                        .image_offset(vk::Offset3D {
                                            x: *dst_x as i32,
                                            y: *dst_y as i32,
                                            z: 0,
                                        })
                                        .image_extent(vk::Extent3D {
                                            width: cw,
                                            height: ch,
                                            depth: 1,
                                        })];
                                    let info = vk::CopyBufferToImageInfo2::default()
                                        .src_buffer(src_buf)
                                        .dst_image(*image)
                                        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                        .regions(&reg);
                                    // SAFETY: Image is in TRANSFER_DST_OPTIMAL; buffer has TRANSFER_SRC.
                                    unsafe {
                                        self.device.cmd_copy_buffer_to_image2(cmd, &info);
                                    }
                                }
                            }
                            (false, true) => {
                                // Texture2D -> Buffer
                                let Some((dst_buf, dst_base, _, dst_alloc)) =
                                    resolve_res_buf(&resources, *dst_res_id)
                                else {
                                    continue;
                                };
                                if let Some(ResourceBacking::Texture2D {
                                    image,
                                    layout,
                                    aspect,
                                    width: tw,
                                    height: th,
                                    vk_format,
                                    ..
                                }) = resources.get_mut(src_res_id)
                                {
                                    transition_image_layout_aspect(
                                        &self.device,
                                        cmd,
                                        *image,
                                        *aspect,
                                        *layout,
                                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                    );
                                    *layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                                    let bpp = vk_format_bytes_per_pixel(*vk_format).max(1);
                                    let row_texels = (*dst_row_pitch) / bpp;
                                    let cw = if *width == 0 { *tw } else { *width };
                                    let ch = if *height == 0 { *th } else { *height };
                                    let reg = [vk::BufferImageCopy2::default()
                                        .buffer_offset(dst_base + *dst_offset)
                                        .buffer_row_length(row_texels)
                                        .buffer_image_height(0)
                                        .image_subresource(vk::ImageSubresourceLayers {
                                            aspect_mask: *aspect,
                                            mip_level: 0,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                                        .image_extent(vk::Extent3D {
                                            width: cw,
                                            height: ch,
                                            depth: 1,
                                        })];
                                    let info = vk::CopyImageToBufferInfo2::default()
                                        .src_image(*image)
                                        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                                        .dst_buffer(dst_buf)
                                        .regions(&reg);
                                    gpu_written_allocs.insert(dst_alloc);
                                    // SAFETY: Image is in TRANSFER_SRC_OPTIMAL; buffer has TRANSFER_DST.
                                    unsafe {
                                        self.device.cmd_copy_image_to_buffer2(cmd, &info);
                                        let mb = [vk::MemoryBarrier2::default()
                                            .src_stage_mask(vk::PipelineStageFlags2::COPY)
                                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                                            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                                        let dep =
                                            vk::DependencyInfo::default().memory_barriers(&mb);
                                        self.device.cmd_pipeline_barrier2(cmd, &dep);
                                    }
                                }
                            }
                            (false, false) => {
                                // Texture2D -> Texture2D
                                if src_res_id == dst_res_id {
                                    continue;
                                }
                                let src_info = match resources.get_mut(src_res_id) {
                                    Some(ResourceBacking::Texture2D {
                                        image,
                                        layout,
                                        aspect,
                                        width: tw,
                                        height: th,
                                        ..
                                    }) => {
                                        transition_image_layout_aspect(
                                            &self.device,
                                            cmd,
                                            *image,
                                            *aspect,
                                            *layout,
                                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                        );
                                        *layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                                        Some((*image, *aspect, *tw, *th))
                                    }
                                    _ => None,
                                };
                                if let Some((s_img, s_aspect, sw, sh)) = src_info
                                    && let Some(ResourceBacking::Texture2D {
                                        image: d_img,
                                        layout: d_layout,
                                        aspect: d_aspect,
                                        ..
                                    }) = resources.get_mut(dst_res_id)
                                {
                                    transition_image_layout_aspect(
                                        &self.device,
                                        cmd,
                                        *d_img,
                                        *d_aspect,
                                        *d_layout,
                                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    );
                                    *d_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                                    let cw = if *width == 0 { sw } else { *width };
                                    let ch = if *height == 0 { sh } else { *height };
                                    let reg = [vk::ImageCopy2::default()
                                        .src_subresource(vk::ImageSubresourceLayers {
                                            aspect_mask: s_aspect,
                                            mip_level: 0,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                        .dst_subresource(vk::ImageSubresourceLayers {
                                            aspect_mask: *d_aspect,
                                            mip_level: 0,
                                            base_array_layer: 0,
                                            layer_count: 1,
                                        })
                                        .dst_offset(vk::Offset3D {
                                            x: *dst_x as i32,
                                            y: *dst_y as i32,
                                            z: 0,
                                        })
                                        .extent(vk::Extent3D {
                                            width: cw,
                                            height: ch,
                                            depth: 1,
                                        })];
                                    let info = vk::CopyImageInfo2::default()
                                        .src_image(s_img)
                                        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                                        .dst_image(*d_img)
                                        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                        .regions(&reg);
                                    // SAFETY: Both images transitioned to TRANSFER_SRC/DST_OPTIMAL.
                                    unsafe {
                                        self.device.cmd_copy_image2(cmd, &info);
                                    }
                                }
                            }
                            (true, true) => {}
                        }
                    }
                    RecordedCommand::CopyResource {
                        dst_res_id,
                        src_res_id,
                    } => {
                        if dst_res_id == src_res_id {
                            continue;
                        }
                        if let (
                            Some((dst_buf, dst_base, dst_sz, dst_alloc)),
                            Some((src_buf, src_base, src_sz, _)),
                        ) = (
                            resolve_res_buf(&resources, *dst_res_id),
                            resolve_res_buf(&resources, *src_res_id),
                        ) {
                            let sz = dst_sz.min(src_sz);
                            let region = [vk::BufferCopy2::default()
                                .src_offset(src_base)
                                .dst_offset(dst_base)
                                .size(sz)];
                            let copy_info = vk::CopyBufferInfo2::default()
                                .src_buffer(src_buf)
                                .dst_buffer(dst_buf)
                                .regions(&region);
                            gpu_written_allocs.insert(dst_alloc);
                            // SAFETY: Distinct buffers with TRANSFER_SRC/DST usage.
                            unsafe {
                                self.device.cmd_copy_buffer2(cmd, &copy_info);
                            }
                        }
                    }
                }
            }
        }

        drop(mem_guard);
        drop(root_sigs);
        drop(pipelines);
        drop(resources);

        // SAFETY: All recorded Vulkan commands are balanced; submit and wait on fence.
        unsafe {
            self.device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Vulkan(format!("vkEndCommandBuffer failed: {e:?}")))?;
            let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submits = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
            self.device
                .queue_submit2(q.queue, &submits, q.submit_fence)
                .map_err(|e| GpuError::Vulkan(format!("vkQueueSubmit2(draw) failed: {e:?}")))?;
            self.device
                .wait_for_fences(&[q.submit_fence], true, u64::MAX)
                .map_err(|e| GpuError::Vulkan(format!("vkWaitForFences(draw) failed: {e:?}")))?;
        }

        // Sync any staged GPU-written buffers back to guest CPU view so subsequent `Map()` sees them.
        if !gpu_written_allocs.is_empty()
            && let Ok(mut mem_guard) = self.memory.lock()
            && let Some(mgr) = mem_guard.as_mut()
        {
            for alloc_id in gpu_written_allocs {
                let _ = mgr.sync_gpu_to_guest(alloc_id);
            }
        }
        Ok(())
    }

    /// `PresentX`: blits to the live Wayland/X11 swapchain when windowed presentation is active,
    /// performs RGBA8 readback when headless or when `X2L_FRAME_DUMP` is configured, and
    /// increments `frames_presented`.
    pub fn present_resource(&self, resource_id: u64) -> Result<PresentedFrame, GpuError> {
        let q = self
            .queue_state
            .lock()
            .map_err(|_| GpuError::Vulkan("queue lock poisoned".into()))?;
        let mut resources = self
            .resources
            .lock()
            .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;

        let Some(ResourceBacking::Texture2D {
            image,
            width,
            height,
            layout,
            ..
        }) = resources.get_mut(&resource_id)
        else {
            return Err(GpuError::Vulkan(format!(
                "PresentX resource {resource_id} is not a 2D texture"
            )));
        };

        let w = *width;
        let h = *height;
        let prior_layout = *layout;

        let mut presented_windowed = false;
        if let Ok(mut pres_guard) = self.presenter.lock()
            && let Some(ref mut presenter) = *pres_guard
        {
            presented_windowed =
                presenter.present_image(&self.device, q.queue, *image, w, h, prior_layout)?;
            // If windowed without a requested PPM frame dump, skip host readback allocation.
            if self.frame_dump_path.is_none() {
                if presented_windowed {
                    self.frames_presented.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(PresentedFrame {
                    width: w,
                    height: h,
                    rgba: Vec::new(),
                });
            }
        }

        let byte_len = (w as usize) * (h as usize) * 4;
        let _ = presented_windowed;

        let buf_ci = vk::BufferCreateInfo::default()
            .size(byte_len as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `self.device` is live.
        let rb_buf = unsafe { self.device.create_buffer(&buf_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateBuffer(readback) failed: {e:?}")))?;
        // SAFETY: `rb_buf` is live.
        let req = unsafe { self.device.get_buffer_memory_requirements(rb_buf) };
        let (mem_type, coherent) = if let Some(idx) = find_memory_type(
            &self.mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            (idx, true)
        } else if let Some(idx) = find_memory_type(
            &self.mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        ) {
            (idx, false)
        } else {
            // SAFETY: `rb_buf` is unused.
            unsafe { self.device.destroy_buffer(rb_buf, None) };
            return Err(GpuError::Vulkan(
                "no host-visible memory for readback".into(),
            ));
        };

        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        // SAFETY: `self.device` is live.
        let rb_mem = match unsafe { self.device.allocate_memory(&ai, None) } {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: `rb_buf` is unused.
                unsafe { self.device.destroy_buffer(rb_buf, None) };
                return Err(GpuError::Vulkan(format!(
                    "vkAllocateMemory(readback) failed: {e:?}"
                )));
            }
        };
        // SAFETY: `rb_buf` and `rb_mem` are live.
        if let Err(e) = unsafe { self.device.bind_buffer_memory(rb_buf, rb_mem, 0) } {
            // SAFETY: Unused handles cleaned up.
            unsafe {
                self.device.destroy_buffer(rb_buf, None);
                self.device.free_memory(rb_mem, None);
            }
            return Err(GpuError::Vulkan(format!(
                "vkBindBufferMemory(readback) failed: {e:?}"
            )));
        }

        let cmd = q.command_buffer;
        let copy_res = (|| -> Result<Vec<u8>, GpuError> {
            // SAFETY: `q` owns `cmd` and `submit_fence`; both are idle.
            unsafe {
                self.device.reset_fences(&[q.submit_fence]).map_err(|e| {
                    GpuError::Vulkan(format!("vkResetFences(present) failed: {e:?}"))
                })?;
                self.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|e| {
                        GpuError::Vulkan(format!("vkResetCommandBuffer(present) failed: {e:?}"))
                    })?;
                let begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device.begin_command_buffer(cmd, &begin).map_err(|e| {
                    GpuError::Vulkan(format!("vkBeginCommandBuffer(present) failed: {e:?}"))
                })?;

                transition_image_layout(
                    &self.device,
                    cmd,
                    *image,
                    prior_layout,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );

                let region = [vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })];
                self.device.cmd_copy_image_to_buffer(
                    cmd,
                    *image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    rb_buf,
                    &region,
                );

                let buf_barrier = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(rb_buf)
                    .offset(0)
                    .size(byte_len as u64)];
                let dep = vk::DependencyInfo::default().buffer_memory_barriers(&buf_barrier);
                self.device.cmd_pipeline_barrier2(cmd, &dep);

                transition_image_layout(
                    &self.device,
                    cmd,
                    *image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                );
                *layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;

                self.device.end_command_buffer(cmd).map_err(|e| {
                    GpuError::Vulkan(format!("vkEndCommandBuffer(present) failed: {e:?}"))
                })?;

                let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                let submits = [vk::SubmitInfo2::default().command_buffer_infos(&cb_infos)];
                self.device
                    .queue_submit2(q.queue, &submits, q.submit_fence)
                    .map_err(|e| {
                        GpuError::Vulkan(format!("vkQueueSubmit2(present) failed: {e:?}"))
                    })?;
                self.device
                    .wait_for_fences(&[q.submit_fence], true, u64::MAX)
                    .map_err(|e| {
                        GpuError::Vulkan(format!("vkWaitForFences(present) failed: {e:?}"))
                    })?;

                let mapped = self
                    .device
                    .map_memory(rb_mem, 0, byte_len as u64, vk::MemoryMapFlags::empty())
                    .map_err(|e| {
                        GpuError::Vulkan(format!("vkMapMemory(readback) failed: {e:?}"))
                    })?;
                if !coherent {
                    let range = [vk::MappedMemoryRange::default()
                        .memory(rb_mem)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)];
                    let _ = self.device.invalidate_mapped_memory_ranges(&range);
                }
                let mut rgba = vec![0u8; byte_len];
                ptr::copy_nonoverlapping(mapped.cast::<u8>(), rgba.as_mut_ptr(), byte_len);
                self.device.unmap_memory(rb_mem);
                Ok(rgba)
            }
        })();

        // SAFETY: Readback fence completed (or failed cleanly); destroy temporary readback handles.
        unsafe {
            self.device.destroy_buffer(rb_buf, None);
            self.device.free_memory(rb_mem, None);
        }

        let rgba = copy_res?;
        let frame = PresentedFrame {
            width: w,
            height: h,
            rgba,
        };
        if let Some(ref dump_path) = self.frame_dump_path
            && let Err(e) = frame.write_ppm(dump_path)
        {
            warn!(path = %dump_path.display(), error = %e, "failed to dump PPM frame");
        } else if let Some(ref dump_path) = self.frame_dump_path {
            debug!(path = %dump_path.display(), width = w, height = h, "dumped rendered PPM frame");
        }

        if let Ok(mut last) = self.last_frame.lock() {
            *last = Some(frame.clone());
        }
        self.frames_presented.fetch_add(1, Ordering::Relaxed);
        Ok(frame)
    }
    /// Explicitly tear down any active Wayland/X11 WSI swapchain & surface (idempotent).
    ///
    /// Called during orderly process shutdown (e.g. `libx2l_core.so` `.fini_array`) so static
    /// `OnceLock`-owned backends still release Vulkan WSI objects cleanly with 0 validation errors.
    pub fn shutdown_wsi(&self) {
        if let Ok(mut pres) = self.presenter.lock()
            && let Some(mut presenter) = pres.take()
        {
            presenter.shutdown(&self.device);
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        self.shutdown_wsi();
        // SAFETY: Wait for device idle before tearing down all Vulkan handles in dependency order.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        if let Ok(mut pipelines) = self.pipelines.lock() {
            for (_, rec) in std::mem::take(&mut *pipelines) {
                // SAFETY: Device is idle.
                unsafe { self.device.destroy_pipeline(rec.pipeline, None) };
            }
        }
        if let Ok(mut resources) = self.resources.lock() {
            for (_, res) in std::mem::take(&mut *resources) {
                if let ResourceBacking::Texture2D {
                    image,
                    memory,
                    view,
                    ..
                } = res
                {
                    // SAFETY: Device is idle; destroy view -> image -> memory.
                    unsafe {
                        self.device.destroy_image_view(view, None);
                        self.device.destroy_image(image, None);
                        self.device.free_memory(memory, None);
                    }
                }
            }
        }
        if let Ok(mut samplers) = self.samplers.lock() {
            for (_, s) in std::mem::take(&mut *samplers) {
                // SAFETY: Device is idle.
                unsafe { self.device.destroy_sampler(s, None) };
            }
        }
        if let Ok(mut fences) = self.fences.lock() {
            for (_, f) in std::mem::take(&mut *fences) {
                // SAFETY: Device is idle.
                unsafe { self.device.destroy_semaphore(f.semaphore, None) };
            }
        }
        // Drop UnifiedMemoryManager BEFORE destroying `self.device`.
        if let Ok(mut mem) = self.memory.lock() {
            let _ = mem.take();
        }
        if let Ok(q) = self.queue_state.lock() {
            // SAFETY: Device is idle.
            unsafe {
                self.device.destroy_sampler(q.default_sampler, None);
                self.device.destroy_image_view(q.dummy_tex_view, None);
                self.device.destroy_image(q.dummy_tex_img, None);
                self.device.free_memory(q.dummy_tex_mem, None);
                self.device.unmap_memory(q.root_ubo_mem);
                self.device.destroy_buffer(q.root_ubo_buf, None);
                self.device.free_memory(q.root_ubo_mem, None);
                self.device.destroy_buffer(q.dummy_ssbo_buf, None);
                self.device.free_memory(q.dummy_ssbo_mem, None);
                self.device
                    .destroy_descriptor_pool(q.submit_descriptor_pool, None);
                self.device.destroy_fence(q.submit_fence, None);
                self.device.destroy_command_pool(q.command_pool, None);
            }
        }
        // SAFETY: All child handles have been destroyed and device is idle.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Rdna2GraphicsConverter {
    stage: ShaderStage,
    opts: GraphicsCompileOptions,
    synthesize_vertex_id: bool,
    override_sig: Option<StageSignature>,
}

impl Converter for Rdna2GraphicsConverter {
    fn id(&self) -> &'static str {
        match self.stage {
            ShaderStage::Vertex => "shader.rdna2_spirv_vs",
            ShaderStage::Pixel => "shader.rdna2_spirv_ps",
            ShaderStage::Compute => "shader.rdna2_spirv_cs",
            ShaderStage::Mesh => "shader.rdna2_spirv_ms",
            _ => "shader.rdna2_spirv_gfx",
        }
    }

    fn version(&self) -> u32 {
        2
    }

    fn convert(&self, input: &[u8]) -> Result<Vec<u8>, PipelineError> {
        let mk_err = |message: String| PipelineError::Convert {
            id: self.id(),
            version: self.version(),
            message,
        };
        let parsed = parse_container(input).map_err(|e| mk_err(format!("container parse: {e}")))?;
        let words = if let Some(variant) = parsed.variants.first() {
            variant.code_words()
        } else if parsed.kind == ContainerKind::RawRdna2 {
            let (chunks, _) = input.as_chunks::<4>();
            chunks.iter().map(|c| u32::from_le_bytes(*c)).collect()
        } else {
            return Err(mk_err(format!(
                "container kind '{}' has no native RDNA2 code words",
                parsed.kind
            )));
        };

        let insts =
            disassemble_reachable(&words).map_err(|e| mk_err(format!("RDNA2 decode: {e}")))?;

        let mut sig: StageSignature = parsed.signature;
        if sig.inputs.is_empty()
            && let Some(ref ov) = self.override_sig
        {
            sig = ov.clone();
        }
        if self.stage == ShaderStage::Vertex && sig.inputs.is_empty() && self.synthesize_vertex_id {
            sig.inputs.push(SignatureElement {
                stream: 0,
                semantic_name: "SV_VertexID".to_owned(),
                semantic_index: 0,
                system_value_type: 6,
                component_type: 1,
                register: 0,
                mask: 0x1,
                rw_mask: 0x1,
                min_precision: 0,
            });
        }

        let spv_words = compile_graphics_spv(self.stage, &insts, &sig, &self.opts)
            .map_err(|e| mk_err(format!("SPIR-V emit: {e}")))?;
        Ok(spirv_words_to_bytes(&spv_words))
    }
}

struct TaggedGraphicsConverter {
    inner: Rdna2GraphicsConverter,
}

impl Converter for TaggedGraphicsConverter {
    fn id(&self) -> &'static str {
        self.inner.id()
    }

    fn version(&self) -> u32 {
        self.inner.version()
    }

    fn convert(&self, tagged_input: &[u8]) -> Result<Vec<u8>, PipelineError> {
        if tagged_input.len() <= 5 {
            return Err(PipelineError::Convert {
                id: self.id(),
                version: self.version(),
                message: "empty tagged shader payload".into(),
            });
        }
        self.inner.convert(&tagged_input[5..])
    }
}

fn select_physical_device(
    instance: &ash::Instance,
    pdevs: &[vk::PhysicalDevice],
    filter: Option<&str>,
) -> vk::PhysicalDevice {
    if let Some(want) = filter {
        let want_lc = want.to_lowercase();
        if let Some(&pd) = pdevs.iter().find(|&&pd| {
            // SAFETY: `pd` is a live physical device from `instance`.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            // SAFETY: `device_name` is a NUL-terminated Vulkan C string.
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .to_lowercase();
            name.contains(&want_lc)
        }) {
            return pd;
        }
    }
    pdevs
        .iter()
        .copied()
        .find(|&pd| {
            // SAFETY: `pd` is a live physical device from `instance`.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
        })
        .unwrap_or(pdevs[0])
}

fn is_depth_dxgi_format(dxgi: u32) -> bool {
    matches!(dxgi, DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_TYPELESS)
}

fn dxgi_to_vk_format(dxgi: u32) -> vk::Format {
    match dxgi {
        DXGI_FORMAT_R32G32B32A32_FLOAT => vk::Format::R32G32B32A32_SFLOAT,
        DXGI_FORMAT_R32G32B32_FLOAT => vk::Format::R32G32B32_SFLOAT,
        DXGI_FORMAT_R32G32_FLOAT => vk::Format::R32G32_SFLOAT,
        DXGI_FORMAT_R32_FLOAT => vk::Format::R32_SFLOAT,
        DXGI_FORMAT_D32_FLOAT | DXGI_FORMAT_R32_TYPELESS => vk::Format::D32_SFLOAT,
        DXGI_FORMAT_R32_UINT => vk::Format::R32_UINT,
        DXGI_FORMAT_R16_UINT => vk::Format::R16_UINT,
        DXGI_FORMAT_B8G8R8A8_UNORM => vk::Format::B8G8R8A8_UNORM,
        _ => vk::Format::R8G8B8A8_UNORM,
    }
}

fn dxgi_format_byte_size(dxgi: u32) -> u32 {
    match dxgi {
        DXGI_FORMAT_R32G32B32A32_FLOAT => 16,
        DXGI_FORMAT_R32G32B32_FLOAT => 12,
        DXGI_FORMAT_R32G32_FLOAT => 8,
        DXGI_FORMAT_R16_UINT => 2,
        _ => 4,
    }
}

fn vk_format_bytes_per_pixel(fmt: vk::Format) -> u32 {
    match fmt {
        vk::Format::R32G32B32A32_SFLOAT => 16,
        vk::Format::R32G32B32_SFLOAT => 12,
        vk::Format::R32G32_SFLOAT => 8,
        vk::Format::R16_UINT => 2,
        _ => 4,
    }
}

fn d3d12_comparison_to_vk(func: u32) -> vk::CompareOp {
    match func {
        1 => vk::CompareOp::NEVER,
        2 => vk::CompareOp::LESS,
        3 => vk::CompareOp::EQUAL,
        4 => vk::CompareOp::LESS_OR_EQUAL,
        5 => vk::CompareOp::GREATER,
        6 => vk::CompareOp::NOT_EQUAL,
        7 => vk::CompareOp::GREATER_OR_EQUAL,
        8 => vk::CompareOp::ALWAYS,
        _ => vk::CompareOp::LESS,
    }
}

fn build_input_signature_from_layout(elems: &[InputElementInfo]) -> StageSignature {
    let inputs = elems
        .iter()
        .enumerate()
        .map(|(idx, e)| {
            let mask = match e.format {
                DXGI_FORMAT_R32G32B32A32_FLOAT => 0xF,
                DXGI_FORMAT_R32G32B32_FLOAT => 0x7,
                DXGI_FORMAT_R32G32_FLOAT => 0x3,
                DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32_UINT => 0x1,
                _ => 0xF,
            };
            SignatureElement {
                stream: 0,
                semantic_name: e.semantic_name.clone(),
                semantic_index: e.semantic_index,
                system_value_type: 0,
                component_type: 3, // float32
                register: idx as u32,
                mask,
                rw_mask: mask,
                min_precision: 0,
            }
        })
        .collect();
    StageSignature {
        inputs,
        outputs: Vec::new(),
    }
}

fn create_host_coherent_buffer(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), GpuError> {
    let ci = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: `device` is live.
    let buf = unsafe { device.create_buffer(&ci, None) }
        .map_err(|e| GpuError::Vulkan(format!("vkCreateBuffer(host) failed: {e:?}")))?;
    // SAFETY: `buf` is live.
    let req = unsafe { device.get_buffer_memory_requirements(buf) };
    let mem_type = find_memory_type(
        mem_props,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .or_else(|| {
        find_memory_type(
            mem_props,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
    })
    .ok_or_else(|| GpuError::Vulkan("no host-visible memory type for internal UBO/SSBO".into()))?;

    let ai = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    // SAFETY: `device` is live.
    let mem = match unsafe { device.allocate_memory(&ai, None) } {
        Ok(m) => m,
        Err(e) => {
            // SAFETY: `buf` is unused.
            unsafe { device.destroy_buffer(buf, None) };
            return Err(GpuError::Vulkan(format!(
                "vkAllocateMemory(host) failed: {e:?}"
            )));
        }
    };
    // SAFETY: `buf` and `mem` are live.
    if let Err(e) = unsafe { device.bind_buffer_memory(buf, mem, 0) } {
        // SAFETY: Unused handles cleaned up.
        unsafe {
            device.destroy_buffer(buf, None);
            device.free_memory(mem, None);
        }
        return Err(GpuError::Vulkan(format!(
            "vkBindBufferMemory(host) failed: {e:?}"
        )));
    }
    // SAFETY: `mem` is host-visible and bound.
    let mapped = match unsafe { device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) } {
        Ok(p) => p.cast::<u8>(),
        Err(e) => {
            // SAFETY: Unused handles cleaned up.
            unsafe {
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            return Err(GpuError::Vulkan(format!("vkMapMemory(host) failed: {e:?}")));
        }
    };
    Ok((buf, mem, mapped))
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1u32 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(required)
    })
}

fn layout_stage_access(layout: vk::ImageLayout) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    match layout {
        vk::ImageLayout::UNDEFINED => {
            (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::NONE)
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags2::CLEAR | vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::VERTEX_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        ),
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        ),
    }
}

fn transition_image_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    transition_image_layout_aspect(
        device,
        cmd,
        image,
        vk::ImageAspectFlags::COLOR,
        old_layout,
        new_layout,
    );
}

fn transition_image_layout_aspect(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    if old_layout == new_layout && old_layout != vk::ImageLayout::UNDEFINED {
        return;
    }
    let (src_stage, src_access) = layout_stage_access(old_layout);
    let (dst_stage, dst_access) = layout_stage_access(new_layout);

    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&barrier);
    // SAFETY: `cmd` is recording and `image` is a live image owned by `device`.
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep);
    }
}
