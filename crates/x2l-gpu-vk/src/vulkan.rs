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
use crate::memory::{AllocationId, AllocationPolicy, UnifiedMemoryManager, UserfaultfdMode};
use crate::wsi::{NativeWindow, PresentSurfaceKind, SwapchainPresenter};
use ash::vk;
use std::collections::BTreeMap;
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

/// DXGI format constants used by the M2.4 backend.
pub const DXGI_FORMAT_R8G8B8A8_UNORM: u32 = 28;
pub const DXGI_FORMAT_B8G8R8A8_UNORM: u32 = 87;

/// D3D12 resource dimensions.
pub const D3D12_RESOURCE_DIMENSION_BUFFER: u32 = 1;
pub const D3D12_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;

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

/// Recorded D3D12 graphics command-list operation translated onto Vulkan 1.3 dynamic rendering.
#[derive(Clone)]
pub enum RecordedCommand {
    ClearRenderTargetView {
        resource_id: u64,
        color: [f32; 4],
    },
    SetViewports(Vec<vk::Viewport>),
    SetScissors(Vec<vk::Rect2D>),
    SetRenderTargets(Vec<u64>),
    SetPipelineState(u64),
    SetPrimitiveTopology(u32),
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
}

/// Description of a committed buffer or 2D texture resource.
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

enum ResourceBacking {
    Buffer {
        alloc_id: AllocationId,
        guest_va: u64,
        _size: usize,
        desc: ResourceDescInfo,
    },
    Texture2D {
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        _vk_format: vk::Format,
        width: u32,
        height: u32,
        layout: vk::ImageLayout,
        desc: ResourceDescInfo,
    },
}

struct DescriptorHeapRecord {
    heap_type: u32,
    num_descriptors: u32,
    base_cpu_ptr: usize,
    increment: usize,
    slots: Vec<Option<u64>>,
}

struct PipelineRecord {
    pipeline: vk::Pipeline,
    _rtv_format: vk::Format,
}

struct FenceRecord {
    semaphore: vk::Semaphore,
}

struct QueueState {
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    submit_fence: vk::Fence,
}

/// Direct Vulkan 1.3/1.4 rendering backend backing M2.4 `D3D12.X` COM interfaces.
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
    pipeline_layout: vk::PipelineLayout,
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
    resources: Mutex<BTreeMap<u64, ResourceBacking>>,
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
            let has_swapchain_maint1_ext = pending_surface != vk::SurfaceKHR::null()
                && has_surface_maint1
                && has_ext(ash::ext::swapchain_maintenance1::NAME);

            let mut avail_v12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut avail_v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut avail_desc_buf = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
            let mut avail_sc_maint1 = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
            let mut avail_feats2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut avail_v12)
                .push_next(&mut avail_v13);
            if has_desc_buf_ext {
                avail_feats2 = avail_feats2.push_next(&mut avail_desc_buf);
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
            if enable_sc_maint1 {
                dev_ci = dev_ci.push_next(&mut feat_sc_maint1);
            }

            // SAFETY: All `pNext` structs live on the stack for the duration of `create_device`.
            let device = unsafe { instance.create_device(physical_device, &dev_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreateDevice failed: {e:?}")))?;
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

            let pl_ci = vk::PipelineLayoutCreateInfo::default();
            // SAFETY: `device` is live.
            let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_ci, None) }
                .map_err(|e| GpuError::Vulkan(format!("vkCreatePipelineLayout failed: {e:?}")))?;

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
                }),
                memory: Mutex::new(Some(memory_mgr)),
                presenter: Mutex::new(presenter),
                present_surface,
                pipeline_layout,
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
                resources: Mutex::new(BTreeMap::new()),
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

    /// Create a committed buffer or 2D render-target texture resource.
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
                        _size: size,
                        desc,
                    },
                );
                Ok(id)
            }
            D3D12_RESOURCE_DIMENSION_TEXTURE2D => {
                let width = (desc.width as u32).max(1);
                let height = desc.height.max(1);
                let vk_format = dxgi_to_vk_format(desc.format);

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
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST,
                    )
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
                        aspect_mask: vk::ImageAspectFlags::COLOR,
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
                        _vk_format: vk_format,
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

    /// Map a buffer resource to a host-writable CPU pointer.
    pub fn map_resource(&self, resource_id: u64) -> Result<*mut u8, GpuError> {
        let alloc_id = {
            let guard = self
                .resources
                .lock()
                .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
            match guard.get(&resource_id) {
                Some(ResourceBacking::Buffer { alloc_id, .. }) => *alloc_id,
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
        let (ptr, _) =
            unsafe { mgr.guest_ptr(alloc_id) }.map_err(|e| GpuError::Vulkan(e.to_string()))?;
        Ok(ptr.as_ptr())
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
                slots: vec![None; count],
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

    /// Bind an RTV slot at `cpu_handle` to `resource_id`.
    pub fn create_render_target_view(
        &self,
        resource_id: u64,
        cpu_handle: usize,
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
                    return Err(GpuError::Vulkan(
                        "misaligned RTV CPU descriptor handle".into(),
                    ));
                }
                let idx = offset / heap.increment;
                heap.slots[idx] = Some(resource_id);
                return Ok(());
            }
        }
        Err(GpuError::Vulkan(format!(
            "RTV CPU descriptor handle 0x{cpu_handle:x} does not belong to any descriptor heap"
        )))
    }

    /// Resolve a CPU RTV descriptor handle to its bound `resource_id`.
    pub fn resolve_rtv(&self, cpu_handle: usize) -> Option<u64> {
        let guard = self.descriptor_heaps.lock().ok()?;
        for heap in guard.values() {
            let span = heap.slots.len() * heap.increment;
            if cpu_handle >= heap.base_cpu_ptr && cpu_handle < heap.base_cpu_ptr + span {
                let offset = cpu_handle - heap.base_cpu_ptr;
                if offset.is_multiple_of(heap.increment) {
                    return heap.slots[offset / heap.increment];
                }
            }
        }
        None
    }

    /// Compile RDNA2 VS+PS bytecode directly to binary SPIR-V 1.6 (cached via `x2l-pipeline`)
    /// and create a Vulkan 1.3 Dynamic Rendering graphics pipeline.
    pub fn create_graphics_pipeline(
        &self,
        vs_blob: &[u8],
        ps_blob: &[u8],
        rtv_format_dxgi: u32,
        has_vertex_inputs: bool,
    ) -> Result<u64, GpuError> {
        let vs_words = self.compile_stage_spv(ShaderStage::Vertex, vs_blob, !has_vertex_inputs)?;
        let ps_words = self.compile_stage_spv(ShaderStage::Pixel, ps_blob, false)?;

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

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
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
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let rtv_vk_format = dxgi_to_vk_format(if rtv_format_dxgi == 0 {
            DXGI_FORMAT_R8G8B8A8_UNORM
        } else {
            rtv_format_dxgi
        });
        let color_formats = [rtv_vk_format];
        let mut rendering_ci =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let pso_ci = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
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
                _rtv_format: rtv_vk_format,
            },
        );
        Ok(id)
    }

    fn compile_stage_spv(
        &self,
        stage: ShaderStage,
        blob: &[u8],
        synthesize_vertex_id: bool,
    ) -> Result<Vec<u32>, GpuError> {
        let opts = GraphicsCompileOptions {
            flip_y: stage == ShaderStage::Vertex,
            strict_float_controls: true,
            denorm_preserve: self.denorm_preserve_f32,
        };
        let converter = Rdna2GraphicsConverter {
            stage,
            opts,
            synthesize_vertex_id,
        };

        let spv_bytes = if let Some(cache) = &self.spirv_cache {
            let mut key_input = Vec::with_capacity(blob.len() + 8);
            key_input.push(match stage {
                ShaderStage::Vertex => 1,
                ShaderStage::Pixel => 2,
                _ => 0,
            });
            key_input.push(u8::from(opts.flip_y));
            key_input.push(u8::from(opts.strict_float_controls));
            key_input.push(u8::from(opts.denorm_preserve));
            key_input.push(u8::from(synthesize_vertex_id));
            key_input.extend_from_slice(blob);

            // Use the cache converter with the actual shader payload encoded right after the 5-byte tag.
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

    /// Execute one or more closed command lists on the Vulkan 1.3 graphics queue.
    pub fn execute_command_lists(&self, lists: &[Vec<RecordedCommand>]) -> Result<(), GpuError> {
        let q = self
            .queue_state
            .lock()
            .map_err(|_| GpuError::Vulkan("queue lock poisoned".into()))?;

        // Flush any dirty guest buffer allocations before dependent GPU draws.
        if let Ok(mut mem_guard) = self.memory.lock()
            && let Some(mgr) = mem_guard.as_mut()
        {
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
                // SAFETY: Command list submission point guarantees guest writers finished updating buffers.
                let _ = unsafe { mgr.flush_dirty(alloc_id) };
            }
        }

        let cmd = q.command_buffer;
        // SAFETY: `q` exclusively owns `cmd` and `submit_fence`, both idle before recording.
        unsafe {
            self.device
                .reset_fences(&[q.submit_fence])
                .map_err(|e| GpuError::Vulkan(format!("vkResetFences failed: {e:?}")))?;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| GpuError::Vulkan(format!("vkResetCommandBuffer failed: {e:?}")))?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|e| GpuError::Vulkan(format!("vkBeginCommandBuffer failed: {e:?}")))?;
        }

        let mut resources = self
            .resources
            .lock()
            .map_err(|_| GpuError::Vulkan("resources lock poisoned".into()))?;
        let pipelines = self
            .pipelines
            .lock()
            .map_err(|_| GpuError::Vulkan("pipelines lock poisoned".into()))?;

        let mut current_rt: Option<u64> = None;
        let mut current_pso: Option<u64> = None;
        let mut current_viewport: Option<vk::Viewport> = None;
        let mut current_scissor: Option<vk::Rect2D> = None;

        for list in lists {
            for op in list {
                match op {
                    RecordedCommand::ClearRenderTargetView { resource_id, color } => {
                        if let Some(ResourceBacking::Texture2D { image, layout, .. }) =
                            resources.get_mut(resource_id)
                        {
                            transition_image_layout(
                                &self.device,
                                cmd,
                                *image,
                                *layout,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            );
                            *layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

                            let clear_val = vk::ClearColorValue { float32: *color };
                            let range = [vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
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
                    RecordedCommand::SetViewports(vps) => {
                        current_viewport = vps.first().copied();
                    }
                    RecordedCommand::SetScissors(scs) => {
                        current_scissor = scs.first().copied();
                    }
                    RecordedCommand::SetRenderTargets(rts) => {
                        current_rt = rts.first().copied();
                    }
                    RecordedCommand::SetPipelineState(pso_id) => {
                        current_pso = Some(*pso_id);
                    }
                    RecordedCommand::SetPrimitiveTopology(_)
                    | RecordedCommand::ResourceBarrier { .. } => {
                        // Tracked Vulkan image layouts handle synchronization2 transitions automatically.
                    }
                    RecordedCommand::DrawInstanced {
                        vertex_count,
                        instance_count,
                        start_vertex,
                        start_instance,
                    } => {
                        let Some(rt_id) = current_rt else {
                            warn!("DrawInstanced called without an active render target");
                            continue;
                        };
                        let Some(pso_id) = current_pso else {
                            warn!("DrawInstanced called without a bound pipeline state");
                            continue;
                        };
                        let Some(pso) = pipelines.get(&pso_id) else {
                            warn!(pso_id, "DrawInstanced references unknown PSO");
                            continue;
                        };
                        let Some(ResourceBacking::Texture2D {
                            image,
                            view,
                            width,
                            height,
                            layout,
                            ..
                        }) = resources.get_mut(&rt_id)
                        else {
                            warn!(rt_id, "DrawInstanced render target is not a Texture2D");
                            continue;
                        };

                        transition_image_layout(
                            &self.device,
                            cmd,
                            *image,
                            *layout,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        );
                        *layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;

                        let color_attachment = [vk::RenderingAttachmentInfo::default()
                            .image_view(*view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)];
                        let render_area = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: *width,
                                height: *height,
                            },
                        };
                        let rendering_info = vk::RenderingInfo::default()
                            .render_area(render_area)
                            .layer_count(1)
                            .color_attachments(&color_attachment);

                        let vp = current_viewport.unwrap_or(vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: *width as f32,
                            height: *height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        });
                        let sc = current_scissor.unwrap_or(render_area);

                        // SAFETY: Image is in `COLOR_ATTACHMENT_OPTIMAL`, PSO matches RTV format
                        // and dynamic viewport/scissor states.
                        unsafe {
                            self.device.cmd_begin_rendering(cmd, &rendering_info);
                            self.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                pso.pipeline,
                            );
                            self.device.cmd_set_viewport(cmd, 0, &[vp]);
                            self.device.cmd_set_scissor(cmd, 0, &[sc]);
                            self.device.cmd_draw(
                                cmd,
                                *vertex_count,
                                *instance_count,
                                *start_vertex,
                                *start_instance,
                            );
                            self.device.cmd_end_rendering(cmd);
                        }
                    }
                }
            }
        }

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
                self.device.destroy_fence(q.submit_fence, None);
                self.device.destroy_command_pool(q.command_pool, None);
            }
        }
        // SAFETY: All child handles have been destroyed and device is idle.
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct Rdna2GraphicsConverter {
    stage: ShaderStage,
    opts: GraphicsCompileOptions,
    synthesize_vertex_id: bool,
}

impl Converter for Rdna2GraphicsConverter {
    fn id(&self) -> &'static str {
        match self.stage {
            ShaderStage::Vertex => "shader.rdna2_spirv_vs",
            ShaderStage::Pixel => "shader.rdna2_spirv_ps",
            _ => "shader.rdna2_spirv_gfx",
        }
    }

    fn version(&self) -> u32 {
        1
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

fn dxgi_to_vk_format(dxgi: u32) -> vk::Format {
    match dxgi {
        DXGI_FORMAT_B8G8R8A8_UNORM => vk::Format::B8G8R8A8_UNORM,
        _ => vk::Format::R8G8B8A8_UNORM,
    }
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
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })];
    let dep = vk::DependencyInfo::default().image_memory_barriers(&barrier);
    // SAFETY: `cmd` is recording and `image` is a live color image owned by `device`.
    unsafe {
        device.cmd_pipeline_barrier2(cmd, &dep);
    }
}
