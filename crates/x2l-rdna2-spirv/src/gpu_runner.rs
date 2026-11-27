// Clean-room Vulkan 1.3 compute harness for bit-exact RDNA2 SPIR-V instruction verification.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Real Vulkan 1.3 (`ash`) compute shader runner for executing compiled RDNA2 SPIR-V 1.6
//! programs against the canonical `WaveState` + `MemoryBuffer` storage-buffer ABI.

use crate::decode::disassemble_reachable;
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
            let Some(compute_family) = queue_families
                .iter()
                .position(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };

            let mut feat_13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut feat_13);
            unsafe { instance.get_physical_device_features2(pd, &mut feat2) };

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

        let mut enable_13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(info.subgroup_size_control_enabled)
            .compute_full_subgroups(info.compute_full_subgroups_enabled);
        let dev_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut enable_13);

        let device = unsafe { instance.create_device(physical_device, &dev_info, None) }
            .map_err(vk_err("vkCreateDevice"))?;
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

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(64)];
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
