// Clean-room Vulkan 1.3 compute harness for bit-exact RDNA2 SPIR-V instruction verification.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Real Vulkan 1.3 (`ash`) compute shader runner for executing compiled RDNA2 SPIR-V 1.6
//! programs against the canonical `WaveState` + `MemoryBuffer` storage-buffer ABI.

use crate::container::{ShaderStage, StageSignature};
use crate::decode::disassemble_reachable;
use crate::graphics::{GraphicsCompileOptions, MeshLayout, compile_graphics_spv, compile_mesh_spv};
#[cfg(not(feature = "spirv-val"))]
use crate::spirv::is_spirv_val_available;
use crate::spirv::{
    CompileOptions, CrossLaneMode, SpirvError, WAVE_STATE_STRIDE, WaveSize,
    compile_compute_spv_with_options, validate_spirv_vulkan13,
};
use ash::vk;
use std::ffi::CStr;
use std::fmt;
use std::sync::Mutex;
use thiserror::Error;

/// Exact host-side mirror of the GPU `Wave` storage-buffer ABI (`66_072` bytes).
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveState {
    /// Scalar general-purpose registers `s0..s127` (`s124` holds `M0`).
    pub sgpr: [u32; 128],
    /// Per-lane vector general-purpose registers `vgpr[lane][reg]` (lane-first, 64x256).
    pub vgpr: [[u32; 256]; 64],
    /// Wave execution mask `[exec_lo, exec_hi]`.
    pub exec: [u32; 2],
    /// Vector condition code `[vcc_lo, vcc_hi]`.
    pub vcc: [u32; 2],
    /// Scalar condition code (canonical 0 or 1).
    pub scc: u32,
    /// Trailing alignment padding for 8-byte `Wave` alignment.
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<WaveState>() == WAVE_STATE_STRIDE as usize);

impl Default for WaveState {
    fn default() -> Self {
        Self::new_wave32()
    }
}

impl WaveState {
    /// Construct a zeroed Wave32 state with all 32 lanes active (`EXEC = 0xFFFF_FFFF`).
    pub fn new_wave32() -> Self {
        Self {
            sgpr: [0; 128],
            vgpr: [[0; 256]; 64],
            exec: [u32::MAX, 0],
            vcc: [0, 0],
            scc: 0,
            _pad: 0,
        }
    }

    /// Construct a zeroed Wave64 state with all 64 lanes active (`EXEC = 0xFFFF_FFFF_FFFF_FFFF`).
    pub fn new_wave64() -> Self {
        Self {
            sgpr: [0; 128],
            vgpr: [[0; 256]; 64],
            exec: [u32::MAX, u32::MAX],
            vcc: [0, 0],
            scc: 0,
            _pad: 0,
        }
    }

    /// Load a 4-DWORD V# descriptor into `s[base..base+4]`.
    pub fn set_vsharp(&mut self, base_sgpr: usize, vsharp: [u32; 4]) {
        self.sgpr[base_sgpr..base_sgpr + 4].copy_from_slice(&vsharp);
    }

    /// Convenience helper: set `v[reg]` across lanes `0..lanes` from a closure.
    pub fn fill_vgpr(&mut self, reg: usize, lanes: usize, mut f: impl FnMut(usize) -> u32) {
        for (lane, bank) in self.vgpr.iter_mut().enumerate().take(lanes) {
            bank[reg] = f(lane);
        }
    }

    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `WaveState` is `#[repr(C)]` containing exclusively `u32` fields with no padding holes.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), std::mem::size_of::<Self>());
        let mut out = Self::new_wave32();
        // SAFETY: `WaveState` is POD `#[repr(C)]` of exact matching byte size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (&mut out as *mut Self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            );
        }
        out
    }
}

/// Vertex input for offscreen Vulkan 1.3 dynamic rendering tests (Location 0 = pos, Location 1 = attr).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex4x2 {
    pub pos: [f32; 4],
    pub attr: [f32; 4],
}

/// 2D RGBA8 texture payload for offscreen sampled-image rendering tests.
#[derive(Debug, Clone)]
pub struct Texture2D {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// One draw call inside an offscreen dynamic-rendering pass.
pub struct DrawCall<'a> {
    pub vs_words: &'a [u32],
    pub ps_words: &'a [u32],
    pub vs_sig: &'a StageSignature,
    pub vertices: &'a [Vertex4x2],
    pub texture: Option<&'a Texture2D>,
    pub flip_y: bool,
}

/// Parameters for headless Vulkan 1.3 Dynamic Rendering offscreen verification.
pub struct RenderRequest<'a> {
    pub width: u32,
    pub height: u32,
    pub clear_color: [f32; 4],
    pub clear_depth: f32,
    pub depth_test: bool,
    pub draws: &'a [DrawCall<'a>],
}

/// Parameters for headless `VK_EXT_mesh_shader` (`MS` + `PS`) offscreen rendering verification.
pub struct MeshRenderRequest<'a> {
    pub width: u32,
    pub height: u32,
    pub clear_color: [f32; 4],
    pub ms_words: &'a [u32],
    pub ps_words: &'a [u32],
    pub mesh_layout: MeshLayout,
    pub group_count: [u32; 3],
    pub flip_y: bool,
}

/// Capabilities and identity reported by the selected Vulkan 1.3 physical device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub device_name: String,
    pub driver_info: String,
    pub device_type: &'static str,
    pub api_version: u32,
    pub subgroup_size: u32,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub compute_required_subgroup_size: bool,
    pub subgroup_size_control_enabled: bool,
    pub compute_full_subgroups_enabled: bool,
    pub dynamic_rendering_enabled: bool,
    pub synchronization2_enabled: bool,
    pub demote_to_helper_enabled: bool,
    pub mesh_shader_enabled: bool,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub shader_denorm_preserve_f32: bool,
    pub shader_rounding_mode_rte_f32: bool,
    pub shader_signed_zero_inf_nan_preserve_f32: bool,
}

impl GpuDeviceInfo {
    /// Returns `true` if this device can pin a full Vulkan compute subgroup of `width` lanes.
    pub fn supports_required_subgroup_size(&self, width: u32) -> bool {
        self.subgroup_size_control_enabled
            && self.compute_required_subgroup_size
            && width >= self.min_subgroup_size
            && width <= self.max_subgroup_size
    }
}

impl fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}, Vulkan {}.{}.{}, driver: '{}', subgroup={} [{}..{}], req_compute={}, denorm_f32={})",
            self.device_name,
            self.device_type,
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version),
            self.driver_info,
            self.subgroup_size,
            self.min_subgroup_size,
            self.max_subgroup_size,
            self.compute_required_subgroup_size,
            self.shader_denorm_preserve_f32,
        )
    }
}

/// Errors raised by `GpuRunner`.
#[derive(Debug, Error)]
pub enum GpuRunnerError {
    #[error("failed to load Vulkan loader (libvulkan.so.1): {0}")]
    Load(String),
    #[error("no suitable Vulkan 1.3 compute physical device found")]
    NoDevice,
    #[error("Vulkan API call '{call}' failed: {code:?}")]
    Vulkan {
        call: &'static str,
        code: vk::Result,
    },
    #[error("no host-visible coherent memory type available")]
    NoHostMemory,
    #[error("RDNA2 decode error: {0}")]
    Decode(#[from] crate::decode::DecodeError),
    #[error("SPIR-V compile/validation error: {0}")]
    Spirv(#[from] SpirvError),
    #[error("VK_EXT_mesh_shader is not supported on the active Vulkan device")]
    MeshShaderUnsupported,
}

fn vk_err(call: &'static str) -> impl FnOnce(vk::Result) -> GpuRunnerError {
    move |code| GpuRunnerError::Vulkan { call, code }
}

struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

/// Self-contained Vulkan 1.3 compute harness for executing compiled RDNA2 SPIR-V 1.6 shaders.
pub struct GpuRunner {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    mesh_shader_ext: Option<ash::ext::mesh_shader::Device>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    dispatch_lock: Mutex<()>,
    info: GpuDeviceInfo,
}

impl GpuRunner {
    /// Initialize Vulkan 1.3, pick the best compute GPU (respecting `X2L_VK_DEVICE`), and build
    /// reusable compute dispatch objects.
    pub fn new() -> Result<Self, GpuRunnerError> {
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| GpuRunnerError::Load(e.to_string()))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"x2linux-gpu-tests")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"x2l-rdna2-spirv")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(vk_err("vkCreateInstance"))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(vk_err("vkEnumeratePhysicalDevices"))?;

        let device_filter = std::env::var("X2L_VK_DEVICE")
            .ok()
            .map(|s| s.to_ascii_lowercase());

        let mut best: Option<(i32, vk::PhysicalDevice, u32, GpuDeviceInfo)> = None;

        for pd in physical_devices {
            let mut props_11 = vk::PhysicalDeviceVulkan11Properties::default();
            let mut props_12 = vk::PhysicalDeviceVulkan12Properties::default();
            let mut props_13 = vk::PhysicalDeviceVulkan13Properties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut props_11)
                .push_next(&mut props_12)
                .push_next(&mut props_13);
            unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

            let base = props2.properties;
            if base.api_version < vk::API_VERSION_1_3 {
                continue;
            }

            let device_name = unsafe { CStr::from_ptr(base.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let driver_info = unsafe { CStr::from_ptr(props_12.driver_info.as_ptr()) }
                .to_string_lossy()
                .into_owned();

            if let Some(ref filter) = device_filter
                && !device_name.to_ascii_lowercase().contains(filter)
                && !driver_info.to_ascii_lowercase().contains(filter)
            {
                continue;
            }

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let desired_flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
            let Some(compute_family) = queue_families
                .iter()
                .position(|qf| qf.queue_flags.contains(desired_flags))
                .or_else(|| {
                    queue_families
                        .iter()
                        .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
                })
            else {
                continue;
            };

            // SAFETY: `pd` is a valid physical device enumerated from `instance`.
            let exts =
                unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
            let has_mesh_ext = exts.iter().any(|e| {
                // SAFETY: `extension_name` is a NUL-terminated fixed-size C string from Vulkan.
                let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                n == ash::ext::mesh_shader::NAME
            });

            let mut mesh_props = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
            if has_mesh_ext {
                let mut mprops2 =
                    vk::PhysicalDeviceProperties2::default().push_next(&mut mesh_props);
                // SAFETY: `mprops2` chains `mesh_props` for a physical device advertising `VK_EXT_mesh_shader`.
                unsafe { instance.get_physical_device_properties2(pd, &mut mprops2) };
            }

            let mut feat_13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut feat_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut feat_13);
            if has_mesh_ext {
                feat2 = feat2.push_next(&mut feat_mesh);
            }
            // SAFETY: `feat2` chains stack-local feature structs for the duration of the query.
            unsafe { instance.get_physical_device_features2(pd, &mut feat2) };
            let mesh_shader_enabled = has_mesh_ext && feat_mesh.mesh_shader == vk::TRUE;

            let (score, device_type) = match base.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => (400, "DISCRETE_GPU"),
                vk::PhysicalDeviceType::INTEGRATED_GPU => (300, "INTEGRATED_GPU"),
                vk::PhysicalDeviceType::VIRTUAL_GPU => (200, "VIRTUAL_GPU"),
                vk::PhysicalDeviceType::CPU => (100, "CPU"),
                _ => (50, "OTHER"),
            };

            let info = GpuDeviceInfo {
                device_name,
                driver_info,
                device_type,
                api_version: base.api_version,
                subgroup_size: props_11.subgroup_size,
                min_subgroup_size: props_13.min_subgroup_size,
                max_subgroup_size: props_13.max_subgroup_size,
                compute_required_subgroup_size: props_13
                    .required_subgroup_size_stages
                    .contains(vk::ShaderStageFlags::COMPUTE),
                subgroup_size_control_enabled: feat_13.subgroup_size_control == vk::TRUE,
                compute_full_subgroups_enabled: feat_13.compute_full_subgroups == vk::TRUE,
                dynamic_rendering_enabled: feat_13.dynamic_rendering == vk::TRUE,
                synchronization2_enabled: feat_13.synchronization2 == vk::TRUE,
                demote_to_helper_enabled: feat_13.shader_demote_to_helper_invocation == vk::TRUE,
                mesh_shader_enabled,
                max_mesh_output_vertices: if mesh_shader_enabled {
                    mesh_props.max_mesh_output_vertices
                } else {
                    0
                },
                max_mesh_output_primitives: if mesh_shader_enabled {
                    mesh_props.max_mesh_output_primitives
                } else {
                    0
                },
                shader_denorm_preserve_f32: props_12.shader_denorm_preserve_float32 == vk::TRUE,
                shader_rounding_mode_rte_f32: props_12.shader_rounding_mode_rte_float32 == vk::TRUE,
                shader_signed_zero_inf_nan_preserve_f32: props_12
                    .shader_signed_zero_inf_nan_preserve_float32
                    == vk::TRUE,
            };

            if best.as_ref().is_none_or(|(s, _, _, _)| score > *s) {
                best = Some((score, pd, compute_family as u32, info));
            }
        }

        let Some((_, physical_device, queue_family_idx, info)) = best else {
            unsafe { instance.destroy_instance(None) };
            return Err(GpuRunnerError::NoDevice);
        };

        let priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_idx)
            .queue_priorities(&priorities)];

        let mut enable_exts: Vec<*const std::ffi::c_char> = Vec::new();
        if info.mesh_shader_enabled {
            enable_exts.push(ash::ext::mesh_shader::NAME.as_ptr());
        }
        let mut enable_13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(info.subgroup_size_control_enabled)
            .compute_full_subgroups(info.compute_full_subgroups_enabled)
            .dynamic_rendering(info.dynamic_rendering_enabled)
            .synchronization2(info.synchronization2_enabled)
            .shader_demote_to_helper_invocation(info.demote_to_helper_enabled);
        let mut enable_mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
        let mut dev_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enable_exts)
            .push_next(&mut enable_13);
        if info.mesh_shader_enabled {
            dev_info = dev_info.push_next(&mut enable_mesh);
        }

        let device = unsafe { instance.create_device(physical_device, &dev_info, None) }
            .map_err(vk_err("vkCreateDevice"))?;
        let mesh_shader_ext = if info.mesh_shader_enabled {
            Some(ash::ext::mesh_shader::Device::new(&instance, &device))
        } else {
            None
        };
        let queue = unsafe { device.get_device_queue(queue_family_idx, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_idx)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("vkCreateCommandPool"))?;

        // Canonical 3-binding StorageBuffer layout:
        //   binding 0: state_in  (NonWritable Wave[])
        //   binding 1: state_out (Wave[])
        //   binding 2: memory_buf (read/write u32[])
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&dsl_info, None) }
            .map_err(vk_err("vkCreateDescriptorSetLayout"))?;

        let set_layouts = [descriptor_set_layout];
        let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_info, None) }
            .map_err(vk_err("vkCreatePipelineLayout"))?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(64),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(16),
        ];
        let dp_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(16)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&dp_info, None) }
            .map_err(vk_err("vkCreateDescriptorPool"))?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            mesh_shader_ext,
            queue,
            command_pool,
            descriptor_set_layout,
            pipeline_layout,
            descriptor_pool,
            dispatch_lock: Mutex::new(()),
            info,
        })
    }

    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    /// Build `CompileOptions` tuned to the active Vulkan device's subgroup & float capabilities.
    pub fn compile_options(&self, wave_size: WaveSize) -> CompileOptions {
        let cross_lane_mode = if self.info.supports_required_subgroup_size(wave_size.width()) {
            CrossLaneMode::Subgroup
        } else {
            CrossLaneMode::WorkgroupShared
        };
        CompileOptions {
            wave_size,
            cross_lane_mode,
            strict_float_controls: self.info.shader_rounding_mode_rte_f32
                && self.info.shader_signed_zero_inf_nan_preserve_f32,
            denorm_preserve: self.info.shader_denorm_preserve_f32,
        }
    }

    /// Decode, compile, validate, and execute an RDNA2 instruction stream on the Vulkan GPU.
    pub fn run_rdna2(
        &self,
        code_words: &[u32],
        wave_size: WaveSize,
        initial_state: &WaveState,
        initial_memory: &[u8],
    ) -> Result<(WaveState, Vec<u8>), GpuRunnerError> {
        let opts = self.compile_options(wave_size);
        self.run_rdna2_with_options(code_words, opts, initial_state, initial_memory)
    }

    /// Decode, compile with explicit options, validate, and execute on the Vulkan GPU.
    pub fn run_rdna2_with_options(
        &self,
        code_words: &[u32],
        opts: CompileOptions,
        initial_state: &WaveState,
        initial_memory: &[u8],
    ) -> Result<(WaveState, Vec<u8>), GpuRunnerError> {
        let insts = disassemble_reachable(code_words)?;
        let spv = compile_compute_spv_with_options(&insts, opts)?;
        self.run_spirv(&spv, opts, initial_state, initial_memory)
    }

    /// Validate and execute a pre-assembled SPIR-V 1.6 compute module on the Vulkan GPU.
    pub fn run_spirv(
        &self,
        spv_words: &[u32],
        opts: CompileOptions,
        initial_state: &WaveState,
        initial_memory: &[u8],
    ) -> Result<(WaveState, Vec<u8>), GpuRunnerError> {
        #[cfg(feature = "spirv-val")]
        validate_spirv_vulkan13(spv_words)?;
        #[cfg(not(feature = "spirv-val"))]
        if is_spirv_val_available() {
            validate_spirv_vulkan13(spv_words)?;
        }

        let state_bytes = std::mem::size_of::<WaveState>() as u64;
        let mem_alloc_bytes = initial_memory.len().max(4) as u64;
        let mem_bind_bytes = if initial_memory.is_empty() {
            4u64
        } else {
            initial_memory.len() as u64
        };

        let buf_in = self.create_host_buffer(state_bytes, initial_state.as_bytes())?;
        let zero_out = vec![0u8; state_bytes as usize];
        let buf_out = self.create_host_buffer(state_bytes, &zero_out)?;
        let mut mem_init = vec![0u8; mem_alloc_bytes as usize];
        mem_init[..initial_memory.len()].copy_from_slice(initial_memory);
        let buf_mem = self.create_host_buffer(mem_alloc_bytes, &mem_init)?;

        let result = self.dispatch_and_readback(
            spv_words,
            opts,
            &buf_in,
            &buf_out,
            &buf_mem,
            mem_bind_bytes,
            initial_memory.len(),
        );

        self.destroy_host_buffer(buf_in);
        self.destroy_host_buffer(buf_out);
        self.destroy_host_buffer(buf_mem);

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_and_readback(
        &self,
        spv_words: &[u32],
        opts: CompileOptions,
        buf_in: &HostBuffer,
        buf_out: &HostBuffer,
        buf_mem: &HostBuffer,
        mem_bind_bytes: u64,
        mem_return_len: usize,
    ) -> Result<(WaveState, Vec<u8>), GpuRunnerError> {
        let _guard = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());
        let sm_info = vk::ShaderModuleCreateInfo::default().code(spv_words);
        let shader_module = unsafe { self.device.create_shader_module(&sm_info, None) }
            .map_err(vk_err("vkCreateShaderModule"))?;

        let mut req_sg = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(opts.wave_size.width());

        let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        if opts.cross_lane_mode == CrossLaneMode::Subgroup
            && self
                .info
                .supports_required_subgroup_size(opts.wave_size.width())
        {
            if self.info.compute_full_subgroups_enabled {
                stage_info =
                    stage_info.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
            }
            stage_info = stage_info.push_next(&mut req_sg);
        }

        let pipe_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.pipeline_layout);

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipe_info], None)
        };
        let pipeline = match pipelines {
            Ok(v) => v[0],
            Err((_, code)) => {
                unsafe { self.device.destroy_shader_module(shader_module, None) };
                return Err(GpuRunnerError::Vulkan {
                    call: "vkCreateComputePipelines",
                    code,
                });
            }
        };

        let set_layouts = [self.descriptor_set_layout];
        let ds_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&ds_alloc) }
            .map_err(vk_err("vkAllocateDescriptorSets"))?[0];

        let info_in = [vk::DescriptorBufferInfo::default()
            .buffer(buf_in.buffer)
            .offset(0)
            .range(buf_in.size)];
        let info_out = [vk::DescriptorBufferInfo::default()
            .buffer(buf_out.buffer)
            .offset(0)
            .range(buf_out.size)];
        let info_mem = [vk::DescriptorBufferInfo::default()
            .buffer(buf_mem.buffer)
            .offset(0)
            .range(mem_bind_bytes)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info_in),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info_out),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info_mem),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.device.allocate_command_buffers(&cb_alloc) }
            .map_err(vk_err("vkAllocateCommandBuffers"))?[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(vk_err("vkBeginCommandBuffer"))?;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(cmd, 1, 1, 1);

            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            self.device
                .end_command_buffer(cmd)
                .map_err(vk_err("vkEndCommandBuffer"))?;
        }

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(vk_err("vkCreateFence"))?;

        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        let wait_res = unsafe {
            self.device
                .queue_submit(self.queue, &submit, fence)
                .and_then(|()| self.device.wait_for_fences(&[fence], true, u64::MAX))
        };

        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set]);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_shader_module(shader_module, None);
        }

        wait_res.map_err(vk_err("vkQueueSubmit/vkWaitForFences"))?;

        let out_state_bytes = self.read_host_buffer(buf_out, buf_out.size as usize)?;
        let out_mem_bytes = self.read_host_buffer(buf_mem, mem_return_len)?;

        Ok((WaveState::from_bytes(&out_state_bytes), out_mem_bytes))
    }

    fn find_host_memory_type(&self, type_bits: u32) -> Result<u32, GpuRunnerError> {
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let desired =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        for i in 0..mem_props.memory_type_count {
            let matches_type = (type_bits & (1 << i)) != 0;
            let matches_flags = mem_props.memory_types[i as usize]
                .property_flags
                .contains(desired);
            if matches_type && matches_flags {
                return Ok(i);
            }
        }
        Err(GpuRunnerError::NoHostMemory)
    }

    fn create_host_buffer(
        &self,
        size: u64,
        init_data: &[u8],
    ) -> Result<HostBuffer, GpuRunnerError> {
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buf_info, None) }
            .map_err(vk_err("vkCreateBuffer"))?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.find_host_memory_type(req.memory_type_bits)?;

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }
            .map_err(vk_err("vkAllocateMemory"))?;
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_err("vkBindBufferMemory"))?;
            let ptr = self
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(vk_err("vkMapMemory"))?;
            std::ptr::copy_nonoverlapping(init_data.as_ptr(), ptr.cast::<u8>(), init_data.len());
            self.device.unmap_memory(memory);
        }

        Ok(HostBuffer {
            buffer,
            memory,
            size,
        })
    }

    fn read_host_buffer(&self, hb: &HostBuffer, len: usize) -> Result<Vec<u8>, GpuRunnerError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; len];
        unsafe {
            let ptr = self
                .device
                .map_memory(hb.memory, 0, len as u64, vk::MemoryMapFlags::empty())
                .map_err(vk_err("vkMapMemory"))?;
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), out.as_mut_ptr(), len);
            self.device.unmap_memory(hb.memory);
        }
        Ok(out)
    }

    fn destroy_host_buffer(&self, hb: HostBuffer) {
        unsafe {
            self.device.destroy_buffer(hb.buffer, None);
            self.device.free_memory(hb.memory, None);
        }
    }

    /// Execute one or more RDNA2 VS + PS draw calls into a headless Vulkan 1.3 Dynamic Rendering
    /// `R8G8B8A8_UNORM` offscreen color target (with optional `D32_SFLOAT` depth target and
    /// sampled texture), returning exact RGBA8 pixels row-major from top-left `(0, 0)`.
    pub fn render_offscreen(
        &self,
        req: &RenderRequest<'_>,
    ) -> Result<Vec<[u8; 4]>, GpuRunnerError> {
        let _guard = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());
        let width = req.width;
        let height = req.height;
        let pixel_bytes = (width as u64) * (height as u64) * 4;

        // 1. Compile & validate all VS/PS pairs first
        let mut compiled_draws = Vec::with_capacity(req.draws.len());
        for draw in req.draws {
            let gfx_opts = GraphicsCompileOptions {
                flip_y: draw.flip_y,
                strict_float_controls: false,
                denorm_preserve: false,
                ..Default::default()
            };
            let vs_insts = disassemble_reachable(draw.vs_words)?;
            let ps_insts = disassemble_reachable(draw.ps_words)?;
            let vs_spv =
                compile_graphics_spv(ShaderStage::Vertex, &vs_insts, draw.vs_sig, &gfx_opts)?;
            let ps_spv = compile_graphics_spv(
                ShaderStage::Pixel,
                &ps_insts,
                &StageSignature::default(),
                &gfx_opts,
            )?;
            #[cfg(feature = "spirv-val")]
            {
                validate_spirv_vulkan13(&vs_spv)?;
                validate_spirv_vulkan13(&ps_spv)?;
            }
            #[cfg(not(feature = "spirv-val"))]
            if is_spirv_val_available() {
                validate_spirv_vulkan13(&vs_spv)?;
                validate_spirv_vulkan13(&ps_spv)?;
            }
            compiled_draws.push((vs_spv, ps_spv, draw));
        }

        // 2. Create offscreen color image + view (+ optional D32_SFLOAT depth image + view)
        let color_fmt = vk::Format::R8G8B8A8_UNORM;
        let (color_img, color_mem, color_view) = self.create_image_2d(
            width,
            height,
            color_fmt,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let depth_fmt = vk::Format::D32_SFLOAT;
        let depth_res = if req.depth_test {
            Some(self.create_image_2d(
                width,
                height,
                depth_fmt,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
            )?)
        } else {
            None
        };

        let readback_buf =
            self.create_buffer_with_usage(pixel_bytes, vk::BufferUsageFlags::TRANSFER_DST, None)?;

        // 3. Prepare per-draw GPU resources (vertex buffers, optional textures, pipelines)
        struct PreparedDraw {
            vb: HostBuffer,
            vert_count: u32,
            vs_mod: vk::ShaderModule,
            ps_mod: vk::ShaderModule,
            dsl: Option<vk::DescriptorSetLayout>,
            ds: Option<vk::DescriptorSet>,
            pl: vk::PipelineLayout,
            pipeline: vk::Pipeline,
            tex: Option<(
                vk::Image,
                vk::DeviceMemory,
                vk::ImageView,
                vk::Sampler,
                HostBuffer,
                u32,
                u32,
            )>,
        }

        let mut prepared = Vec::with_capacity(compiled_draws.len());
        for (vs_spv, ps_spv, draw) in &compiled_draws {
            let vb_bytes = unsafe {
                std::slice::from_raw_parts(
                    draw.vertices.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(draw.vertices),
                )
            };
            let vb = self.create_buffer_with_usage(
                vb_bytes.len().max(4) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                Some(vb_bytes),
            )?;

            let vs_mod = unsafe {
                self.device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(vs_spv), None)
            }
            .map_err(vk_err("vkCreateShaderModule(VS)"))?;
            let ps_mod = unsafe {
                self.device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(ps_spv), None)
            }
            .map_err(vk_err("vkCreateShaderModule(PS)"))?;

            let (dsl, ds, tex) = if let Some(t) = draw.texture {
                let (t_img, t_mem, t_view) = self.create_image_2d(
                    t.width,
                    t.height,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                )?;
                let t_staging = self.create_buffer_with_usage(
                    t.rgba.len().max(4) as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    Some(&t.rgba),
                )?;
                let samp_info = vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
                let sampler = unsafe { self.device.create_sampler(&samp_info, None) }
                    .map_err(vk_err("vkCreateSampler"))?;

                let bindings = [
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(2)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                ];
                let dsl = unsafe {
                    self.device.create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                        None,
                    )
                }
                .map_err(vk_err("vkCreateDescriptorSetLayout(gfx)"))?;
                let dsls = [dsl];
                let ds = unsafe {
                    self.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
                            .set_layouts(&dsls),
                    )
                }
                .map_err(vk_err("vkAllocateDescriptorSets(gfx)"))?[0];

                let img_info = [vk::DescriptorImageInfo::default()
                    .image_view(t_view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let samp_desc = [vk::DescriptorImageInfo::default().sampler(sampler)];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&img_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&samp_desc),
                ];
                unsafe { self.device.update_descriptor_sets(&writes, &[]) };

                (
                    Some(dsl),
                    Some(ds),
                    Some((t_img, t_mem, t_view, sampler, t_staging, t.width, t.height)),
                )
            } else {
                (None, None, None)
            };

            let set_layouts: Vec<vk::DescriptorSetLayout> = dsl.into_iter().collect();
            let pl = unsafe {
                self.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
            }
            .map_err(vk_err("vkCreatePipelineLayout(gfx)"))?;

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vs_mod)
                    .name(c"main"),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(ps_mod)
                    .name(c"main"),
            ];
            let v_binding = [vk::VertexInputBindingDescription::default()
                .binding(0)
                .stride(std::mem::size_of::<Vertex4x2>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)];
            let v_attrs = [
                vk::VertexInputAttributeDescription::default()
                    .location(0)
                    .binding(0)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(0),
                vk::VertexInputAttributeDescription::default()
                    .location(1)
                    .binding(0)
                    .format(vk::Format::R32G32B32A32_SFLOAT)
                    .offset(16),
            ];
            let vi = vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&v_binding)
                .vertex_attribute_descriptions(&v_attrs);
            let ia = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewport = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            }];
            let vp = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewport)
                .scissors(&scissor);
            let rast = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let msaa = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let ds_state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_test_enable(req.depth_test)
                .depth_write_enable(req.depth_test)
                .depth_compare_op(vk::CompareOp::LESS);
            let blend_att = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)];
            let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);
            let color_fmts = [color_fmt];
            let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&color_fmts)
                .depth_attachment_format(if req.depth_test {
                    depth_fmt
                } else {
                    vk::Format::UNDEFINED
                });

            let gp_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vi)
                .input_assembly_state(&ia)
                .viewport_state(&vp)
                .rasterization_state(&rast)
                .multisample_state(&msaa)
                .depth_stencil_state(&ds_state)
                .color_blend_state(&blend)
                .layout(pl)
                .push_next(&mut rendering_info);

            let pipeline = unsafe {
                self.device
                    .create_graphics_pipelines(vk::PipelineCache::null(), &[gp_info], None)
            }
            .map_err(|(_, code)| GpuRunnerError::Vulkan {
                call: "vkCreateGraphicsPipelines",
                code,
            })?[0];

            prepared.push(PreparedDraw {
                vb,
                vert_count: draw.vertices.len() as u32,
                vs_mod,
                ps_mod,
                dsl,
                ds,
                pl,
                pipeline,
                tex,
            });
        }

        // 4. Record command buffer with Synchronization2 barriers & Vulkan 1.3 Dynamic Rendering
        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(vk_err("vkAllocateCommandBuffers(gfx)"))?[0];

        unsafe {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_err("vkBeginCommandBuffer(gfx)"))?;

            // Upload any sampled textures before BeginRendering
            for p in &prepared {
                if let Some((t_img, _, _, _, ref t_stg, tw, th)) = p.tex {
                    let b_to_dst = [vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(t_img)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })];
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&b_to_dst),
                    );
                    let region = [vk::BufferImageCopy::default()
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D {
                            width: tw,
                            height: th,
                            depth: 1,
                        })];
                    self.device.cmd_copy_buffer_to_image(
                        cmd,
                        t_stg.buffer,
                        t_img,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &region,
                    );
                    let b_to_read = [vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(t_img)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })];
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&b_to_read),
                    );
                }
            }

            // Transition color (+ optional depth) attachment to optimal attachment layout
            let mut init_barriers = vec![
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(
                        vk::AccessFlags2::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(color_img)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            ];
            if let Some((d_img, _, _)) = depth_res {
                init_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(d_img)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                );
            }
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&init_barriers),
            );

            let color_att = [vk::RenderingAttachmentInfo::default()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: req.clear_color,
                    },
                })];
            let depth_att = depth_res.map(|(_, _, dv)| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(dv)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: req.clear_depth,
                            stencil: 0,
                        },
                    })
            });
            let mut render_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .layer_count(1)
                .color_attachments(&color_att);
            if let Some(ref d_att) = depth_att {
                render_info = render_info.depth_attachment(d_att);
            }

            self.device.cmd_begin_rendering(cmd, &render_info);
            for p in &prepared {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, p.pipeline);
                if let Some(ds) = p.ds {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        p.pl,
                        0,
                        &[ds],
                        &[],
                    );
                }
                self.device
                    .cmd_bind_vertex_buffers(cmd, 0, &[p.vb.buffer], &[0]);
                self.device.cmd_draw(cmd, p.vert_count, 1, 0, 0);
            }
            self.device.cmd_end_rendering(cmd);

            // Transition color -> TRANSFER_SRC_OPTIMAL, copy to staging buffer, sync to HOST_READ
            let to_src = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_src),
            );
            let copy_reg = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })];
            self.device.cmd_copy_image_to_buffer(
                cmd,
                color_img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buf.buffer,
                &copy_reg,
            );
            let to_host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            self.device
                .end_command_buffer(cmd)
                .map_err(vk_err("vkEndCommandBuffer(gfx)"))?;
        }

        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(vk_err("vkCreateFence(gfx)"))?;
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        let wait_res = unsafe {
            self.device
                .queue_submit(self.queue, &submit, fence)
                .and_then(|()| self.device.wait_for_fences(&[fence], true, u64::MAX))
        };

        let raw_pixels = if wait_res.is_ok() {
            Some(self.read_host_buffer(&readback_buf, pixel_bytes as usize)?)
        } else {
            None
        };

        // Cleanup all per-render resources cleanly
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
            for p in prepared {
                if let Some((ti, tm, tv, samp, tstg, _, _)) = p.tex {
                    self.device.destroy_sampler(samp, None);
                    self.device.destroy_image_view(tv, None);
                    self.device.destroy_image(ti, None);
                    self.device.free_memory(tm, None);
                    self.destroy_host_buffer(tstg);
                }
                if let Some(ds) = p.ds {
                    let _ = self
                        .device
                        .free_descriptor_sets(self.descriptor_pool, &[ds]);
                }
                if let Some(dsl) = p.dsl {
                    self.device.destroy_descriptor_set_layout(dsl, None);
                }
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.pl, None);
                self.device.destroy_shader_module(p.vs_mod, None);
                self.device.destroy_shader_module(p.ps_mod, None);
                self.destroy_host_buffer(p.vb);
            }
            self.destroy_host_buffer(readback_buf);
            if let Some((di, dm, dv)) = depth_res {
                self.device.destroy_image_view(dv, None);
                self.device.destroy_image(di, None);
                self.device.free_memory(dm, None);
            }
            self.device.destroy_image_view(color_view, None);
            self.device.destroy_image(color_img, None);
            self.device.free_memory(color_mem, None);
        }

        wait_res.map_err(vk_err("vkQueueSubmit/vkWaitForFences(gfx)"))?;
        let raw = raw_pixels.expect("pixels read");
        let (chunks, _) = raw.as_chunks::<4>();
        Ok(chunks.to_vec())
    }

    /// Compile and render an RDNA2 Mesh Shader (`MS` -> SPIR-V 1.6 `MeshEXT`) + Pixel Shader (`PS`)
    /// via `VK_EXT_mesh_shader` and Vulkan 1.3 Dynamic Rendering, returning `width * height` RGBA8 pixels.
    pub fn render_mesh_offscreen(
        &self,
        req: &MeshRenderRequest<'_>,
    ) -> Result<Vec<[u8; 4]>, GpuRunnerError> {
        let Some(ref mesh_ext) = self.mesh_shader_ext else {
            return Err(GpuRunnerError::MeshShaderUnsupported);
        };
        let _guard = self.dispatch_lock.lock().expect("dispatch lock poisoned");
        let width = req.width.max(1);
        let height = req.height.max(1);
        let pixel_bytes = (width as u64) * (height as u64) * 4;

        // 1. Disassemble, compile to SPIR-V 1.6 (`MeshEXT` + `Fragment`), and validate.
        let ms_insts = disassemble_reachable(req.ms_words)?;
        let ps_insts = disassemble_reachable(req.ps_words)?;
        let gfx_opts = GraphicsCompileOptions {
            flip_y: req.flip_y,
            ..Default::default()
        };
        let ms_spv = compile_mesh_spv(
            &ms_insts,
            &StageSignature::default(),
            &gfx_opts,
            req.mesh_layout,
        )?;
        let ps_spv = compile_graphics_spv(
            ShaderStage::Pixel,
            &ps_insts,
            &StageSignature::default(),
            &gfx_opts,
        )?;
        #[cfg(feature = "spirv-val")]
        {
            validate_spirv_vulkan13(&ms_spv)?;
            validate_spirv_vulkan13(&ps_spv)?;
        }
        #[cfg(not(feature = "spirv-val"))]
        if is_spirv_val_available() {
            validate_spirv_vulkan13(&ms_spv)?;
            validate_spirv_vulkan13(&ps_spv)?;
        }

        // 2. Create offscreen RGBA8 color target + host-visible readback buffer.
        let color_fmt = vk::Format::R8G8B8A8_UNORM;
        let (color_img, color_mem, color_view) = self.create_image_2d(
            width,
            height,
            color_fmt,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        let readback_buf =
            self.create_buffer_with_usage(pixel_bytes, vk::BufferUsageFlags::TRANSFER_DST, None)?;

        // 3. Create shader modules, empty pipeline layout, and MeshEXT + Fragment pipeline.
        // SAFETY: `ms_spv` and `ps_spv` are validated SPIR-V 1.6 word slices on a live `ash::Device`.
        let ms_mod = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&ms_spv), None)
        }
        .map_err(vk_err("vkCreateShaderModule(MS)"))?;
        // SAFETY: `ps_spv` is a validated SPIR-V 1.6 word slice on a live `ash::Device`.
        let ps_mod = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&ps_spv), None)
        }
        .map_err(vk_err("vkCreateShaderModule(PS)"))?;
        // SAFETY: Creating an empty pipeline layout on a live `ash::Device`.
        let pl = unsafe {
            self.device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        }
        .map_err(vk_err("vkCreatePipelineLayout(mesh)"))?;

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_EXT)
                .module(ms_mod)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(ps_mod)
                .name(c"main"),
        ];
        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        }];
        let vp = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewport)
            .scissors(&scissor);
        let rast = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let msaa = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_att = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)];
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_att);
        let color_fmts = [color_fmt];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_fmts);

        // Mesh shader pipelines omit vertex input and input assembly states per VK_EXT_mesh_shader.
        let gp_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .viewport_state(&vp)
            .rasterization_state(&rast)
            .multisample_state(&msaa)
            .color_blend_state(&blend)
            .layout(pl)
            .push_next(&mut rendering_info);

        // SAFETY: All referenced shader modules, pipeline layout, and chained structs remain valid for the call.
        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[gp_info], None)
        }
        .map_err(|(_, code)| GpuRunnerError::Vulkan {
            call: "vkCreateGraphicsPipelines(mesh)",
            code,
        })?[0];

        // 4. Record Synchronization2 + Dynamic Rendering + vkCmdDrawMeshTasksEXT command buffer.
        // SAFETY: Allocating one primary command buffer from a live command pool.
        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(vk_err("vkAllocateCommandBuffers(mesh)"))?[0];

        // SAFETY: Recording valid Vulkan 1.3 + VK_EXT_mesh_shader commands into `cmd`.
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(vk_err("vkBeginCommandBuffer(mesh)"))?;

            let init_barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&init_barrier),
            );

            let color_att = [vk::RenderingAttachmentInfo::default()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: req.clear_color,
                    },
                })];
            let render_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .layer_count(1)
                .color_attachments(&color_att);

            self.device.cmd_begin_rendering(cmd, &render_info);
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            mesh_ext.cmd_draw_mesh_tasks(
                cmd,
                req.group_count[0],
                req.group_count[1],
                req.group_count[2],
            );
            self.device.cmd_end_rendering(cmd);

            let to_src = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(color_img)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_src),
            );
            let copy_reg = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })];
            self.device.cmd_copy_image_to_buffer(
                cmd,
                color_img,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buf.buffer,
                &copy_reg,
            );
            let to_host = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            self.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            self.device
                .end_command_buffer(cmd)
                .map_err(vk_err("vkEndCommandBuffer(mesh)"))?;
        }

        // SAFETY: Creating a fence, submitting `cmd`, and waiting for completion before teardown.
        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        }
        .map_err(vk_err("vkCreateFence(mesh)"))?;
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        // SAFETY: `cmd` and `fence` are valid and not in use by any other submission.
        let wait_res = unsafe {
            self.device
                .queue_submit(self.queue, &submit, fence)
                .and_then(|()| self.device.wait_for_fences(&[fence], true, u64::MAX))
        };

        let raw_pixels = if wait_res.is_ok() {
            Some(self.read_host_buffer(&readback_buf, pixel_bytes as usize)?)
        } else {
            None
        };

        // SAFETY: Queue work signaled `fence` (or failed submission) and all resources are owned by this call.
        unsafe {
            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(self.command_pool, &[cmd]);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_pipeline_layout(pl, None);
            self.device.destroy_shader_module(ms_mod, None);
            self.device.destroy_shader_module(ps_mod, None);
            self.destroy_host_buffer(readback_buf);
            self.device.destroy_image_view(color_view, None);
            self.device.destroy_image(color_img, None);
            self.device.free_memory(color_mem, None);
        }

        wait_res.map_err(vk_err("vkQueueSubmit/vkWaitForFences(mesh)"))?;
        let raw = raw_pixels.expect("pixels read");
        let (chunks, _) = raw.as_chunks::<4>();
        Ok(chunks.to_vec())
    }

    fn create_buffer_with_usage(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        init_data: Option<&[u8]>,
    ) -> Result<HostBuffer, GpuRunnerError> {
        let buf_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buf_info, None) }
            .map_err(vk_err("vkCreateBuffer"))?;
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_type = self.find_host_memory_type(req.memory_type_bits)?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .map_err(vk_err("vkAllocateMemory"))?;
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_err("vkBindBufferMemory"))?;
            if let Some(data) = init_data {
                let ptr = self
                    .device
                    .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                    .map_err(vk_err("vkMapMemory"))?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len());
                self.device.unmap_memory(memory);
            }
        }
        Ok(HostBuffer {
            buffer,
            memory,
            size,
        })
    }

    fn create_image_2d(
        &self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), GpuRunnerError> {
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&img_info, None) }
            .map_err(vk_err("vkCreateImage"))?;
        let req = unsafe { self.device.get_image_memory_requirements(image) };
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let mut mem_type = None;
        for i in 0..mem_props.memory_type_count {
            if (req.memory_type_bits & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            {
                mem_type = Some(i);
                break;
            }
        }
        let mem_type = mem_type.unwrap_or_else(|| req.memory_type_bits.trailing_zeros());
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(mem_type),
                None,
            )
        }
        .map_err(vk_err("vkAllocateMemory(img)"))?;
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(vk_err("vkBindImageMemory"))?;
        }
        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .map_err(vk_err("vkCreateImageView"))?;
        Ok((image, memory, view))
    }
}

impl Drop for GpuRunner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
