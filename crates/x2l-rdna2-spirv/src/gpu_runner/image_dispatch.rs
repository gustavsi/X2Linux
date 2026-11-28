// Clean-room Vulkan image fixtures for executing synthetic RDNA2 compute shaders.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

use super::*;

type ImagePixels = Vec<[f32; 4]>;

const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

/// A synthetic RGBA32F image and the descriptor slot used by a compute fixture.
pub struct ImageInput<'a> {
    pub binding: u32,
    pub view_type: vk::ImageViewType,
    pub extent: [u32; 3],
    pub layers: u32,
    pub texels: &'a [[f32; 4]],
    pub storage: bool,
}

struct PreparedImage {
    binding: u32,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    staging: HostBuffer,
    copy: vk::BufferImageCopy,
    storage: bool,
}

impl PreparedImage {
    fn descriptor_type(&self) -> vk::DescriptorType {
        if self.storage {
            vk::DescriptorType::STORAGE_IMAGE
        } else {
            vk::DescriptorType::SAMPLED_IMAGE
        }
    }

    fn layout(&self) -> vk::ImageLayout {
        if self.storage {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        }
    }

    fn transition(
        &self,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src: vk::AccessFlags,
        dst: vk::AccessFlags,
    ) -> vk::ImageMemoryBarrier<'static> {
        vk::ImageMemoryBarrier::default()
            .image(self.image)
            .old_layout(old)
            .new_layout(new)
            .src_access_mask(src)
            .dst_access_mask(dst)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(self.copy.image_subresource.layer_count),
            )
    }
}

pub(super) struct ImageDispatch<'a> {
    runner: &'a GpuRunner,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    images: Vec<PreparedImage>,
    sampler: Option<(u32, vk::Sampler)>,
}

impl GpuRunner {
    /// Run a synthetic wave with image descriptors and return storage image texels in input order.
    pub fn run_rdna2_with_images(
        &self,
        code: &[u32],
        init: &WaveState,
        inputs: &[ImageInput<'_>],
        sampler_binding: Option<u32>,
    ) -> Result<(WaveState, Vec<ImagePixels>), GpuRunnerError> {
        let opts = self.compile_options(WaveSize::Wave32);
        let spv = compile_compute_spv_with_options(&disassemble_reachable(code)?, opts)?;
        let images = ImageDispatch::new(self, inputs, sampler_binding)?;
        let (state, _) = self.run_spirv_with_images(&spv, opts, init, &[], Some(&images))?;
        let outputs = images
            .images
            .iter()
            .filter(|i| i.storage)
            .map(|i| {
                let bytes = self.read_host_buffer(&i.staging, i.staging.size as usize)?;
                Ok(bytes
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|texel| {
                        let [r, g, b, a] = texel.as_chunks::<4>().0 else {
                            unreachable!("RGBA32F texel must contain four components")
                        };
                        [*r, *g, *b, *a].map(f32::from_ne_bytes)
                    })
                    .collect())
            })
            .collect::<Result<_, GpuRunnerError>>()?;
        Ok((state, outputs))
    }
}

impl<'a> ImageDispatch<'a> {
    fn new(
        runner: &'a GpuRunner,
        inputs: &[ImageInput<'_>],
        sampler_binding: Option<u32>,
    ) -> Result<Self, GpuRunnerError> {
        let mut dispatch = Self {
            runner,
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            images: Vec::with_capacity(inputs.len()),
            sampler: None,
        };
        for input in inputs {
            dispatch.images.push(dispatch.prepare(input)?);
        }
        let mut bindings = Vec::new();
        let binding = |slot, ty| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(slot)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        bindings.extend((0..3).map(|slot| binding(slot, vk::DescriptorType::STORAGE_BUFFER)));
        bindings.extend(
            dispatch
                .images
                .iter()
                .map(|i| binding(i.binding, i.descriptor_type())),
        );
        let device = &runner.device;
        if let Some(slot) = sampler_binding {
            bindings.push(binding(slot, vk::DescriptorType::SAMPLER));
            let info = vk::SamplerCreateInfo::default()
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            // SAFETY: Sampler create info is initialized and the device is live.
            let sampler = unsafe { device.create_sampler(&info, None) }
                .map_err(vk_err("vkCreateSampler(test image)"))?;
            dispatch.sampler = Some((slot, sampler));
        }
        // SAFETY: The binding list is live and describes each test fixture resource.
        dispatch.descriptor_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(vk_err("vkCreateDescriptorSetLayout(test image)"))?;
        // SAFETY: The descriptor layout is owned by this dispatch.
        dispatch.pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[dispatch.descriptor_layout]),
                None,
            )
        }
        .map_err(vk_err("vkCreatePipelineLayout(test image)"))?;
        Ok(dispatch)
    }

    fn prepare(&self, input: &ImageInput<'_>) -> Result<PreparedImage, GpuRunnerError> {
        let [width, height, depth] = input.extent;
        assert_eq!(
            input.texels.len(),
            input.extent.into_iter().product::<u32>() as usize * input.layers as usize
        );
        let extent = vk::Extent3D {
            width,
            height,
            depth,
        };
        let flags = if input.view_type == vk::ImageViewType::CUBE {
            vk::ImageCreateFlags::CUBE_COMPATIBLE
        } else {
            vk::ImageCreateFlags::empty()
        };
        let image_type = if input.view_type == vk::ImageViewType::TYPE_3D {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };
        let usage = if input.storage {
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::SAMPLED
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(image_type)
            .extent(extent)
            .mip_levels(1)
            .array_layers(input.layers)
            .flags(flags)
            .format(FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage | vk::ImageUsageFlags::TRANSFER_DST);
        let upload: Vec<u8> = input
            .texels
            .iter()
            .flatten()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let staging = self.runner.create_buffer_with_usage(
            upload.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            Some(&upload),
        )?;
        let (image, memory, view) = match self.runner.create_image_with_view(
            &info,
            input.view_type,
            vk::ImageAspectFlags::COLOR,
        ) {
            Ok(image) => image,
            Err(err) => {
                self.runner.destroy_host_buffer(staging);
                return Err(err);
            }
        };
        let copy = vk::BufferImageCopy::default()
            .image_extent(extent)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(input.layers),
            );
        Ok(PreparedImage {
            binding: input.binding,
            image,
            memory,
            view,
            staging,
            copy,
            storage: input.storage,
        })
    }

    pub fn update_descriptors(&self, set: vk::DescriptorSet) {
        for image in &self.images {
            let info = [vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .image_layout(image.layout())];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(image.binding)
                .descriptor_type(image.descriptor_type())
                .image_info(&info)];
            // SAFETY: The set was allocated from this image dispatch's layout and the infos are live.
            unsafe { self.runner.device.update_descriptor_sets(&writes, &[]) };
        }
        if let Some((binding, sampler)) = self.sampler {
            let info = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&info)];
            // SAFETY: The set contains this sampler binding and retains a live sampler.
            unsafe { self.runner.device.update_descriptor_sets(&writes, &[]) };
        }
    }

    pub fn record_uploads(&self, cmd: vk::CommandBuffer) {
        let device = &self.runner.device;
        let to_transfer: Vec<_> = self
            .images
            .iter()
            .map(|i| {
                i.transition(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                )
            })
            .collect();
        let to_shader: Vec<_> = self
            .images
            .iter()
            .map(|i| {
                i.transition(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    i.layout(),
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                )
            })
            .collect();
        // SAFETY: The recording command buffer and all resources remain live through submission.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_transfer,
            );
            for i in &self.images {
                device.cmd_copy_buffer_to_image(
                    cmd,
                    i.staging.buffer,
                    i.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[i.copy],
                );
            }
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_shader,
            );
        }
    }

    pub fn record_readbacks(&self, cmd: vk::CommandBuffer) {
        let device = &self.runner.device;
        let storage = self.images.iter().filter(|i| i.storage);
        let to_transfer: Vec<_> = storage
            .clone()
            .map(|i| {
                i.transition(
                    i.layout(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )
            })
            .collect();
        // SAFETY: The compute dispatch precedes these copies and all resources remain live.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_transfer,
            );
            for i in storage {
                device.cmd_copy_image_to_buffer(
                    cmd,
                    i.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    i.staging.buffer,
                    &[i.copy],
                );
            }
        }
    }
}

impl Drop for ImageDispatch<'_> {
    fn drop(&mut self) {
        let device = &self.runner.device;
        // SAFETY: The fixture owns these handles and any submitted work has been awaited.
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            if let Some((_, sampler)) = self.sampler {
                device.destroy_sampler(sampler, None);
            }
            for i in self.images.drain(..) {
                device.destroy_image_view(i.view, None);
                device.destroy_image(i.image, None);
                device.free_memory(i.memory, None);
                self.runner.destroy_host_buffer(i.staging);
            }
        }
    }
}
