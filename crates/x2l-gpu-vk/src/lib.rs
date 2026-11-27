//! Direct `D3D12.X` → Vulkan 1.3/1.4 translation backend (`ash`).
//!
//! Targets `VK_KHR_buffer_device_address`, `VK_EXT_descriptor_buffer`, `VK_EXT_mesh_shader`,
//! `VK_EXT_graphics_pipeline_library`, `VK_EXT_external_memory_host`, `VK_KHR_timeline_semaphore`,
//! and Linux `userfaultfd` write-protect dirty-page tracking without `vkd3d-proton`.

use thiserror::Error;
use x2l_common::GpuBackend;

/// Errors raised while initializing or driving the Vulkan 1.3/1.4 backend.
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("Vulkan backend error: {0}")]
    Vulkan(String),
}

/// Minimum target Vulkan API version (`Vulkan 1.3`).
pub const TARGET_VULKAN_API_VERSION: u32 = ash::vk::API_VERSION_1_3;

/// Required Vulkan device extensions for full M2 `D3D12.X` translation.
pub const REQUIRED_VULKAN_EXTENSIONS: &[&str] = &[
    "VK_KHR_buffer_device_address",
    "VK_EXT_descriptor_buffer",
    "VK_EXT_mesh_shader",
    "VK_EXT_graphics_pipeline_library",
    "VK_KHR_timeline_semaphore",
];

/// Summary of active GPU backend configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRuntimeState {
    pub backend: GpuBackend,
    pub validation: bool,
}
