// Clean-room RDNA2 image instructions and SPIR-V resource declarations.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

use crate::decode::{DecodedInst, InstExtra};
use crate::spirv::{Alu, SpirvError};
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    Capability, Decoration, Dim, GLOp, ImageFormat, ImageOperands, SelectionControl, StorageClass,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy)]
pub(crate) struct Mimg {
    pub op: u16,
    pub dim: u8,
    pub dmask: u8,
    pub vdata: u8,
    srsrc: u8,
    ssamp: u8,
    is_nsa: bool,
    addrs: [u8; 13],
}

impl Mimg {
    pub fn parse(inst: &DecodedInst, fragment: bool) -> Result<Self, SpirvError> {
        let InstExtra::Mimg {
            op,
            dim,
            dmask,
            unrm,
            tfe,
            lwe,
            a16,
            d16,
            vdata,
            srsrc,
            ssamp,
            addr_count,
            is_nsa,
            addrs,
            ..
        } = inst.extra
        else {
            unreachable!("expected MIMG instruction");
        };
        if dim > 7
            || a16
            || d16
            || tfe
            || lwe
            || dmask == 0
            || (is_nsa && addr_count < Self::address_count(op, dim))
            || !match op {
                0x00 | 0x0e => true,
                0x01 | 0x08 => dim < 6,
                0x20 | 0x24 | 0x27 => dim < 6 && !unrm,
                0x25 => fragment && dim < 6 && !unrm,
                0x2f => dim != 2 && dim < 6 && !unrm,
                0x40 | 0x47 => matches!(dim, 1 | 3 | 5) && !unrm && dmask.count_ones() == 1,
                _ => false,
            }
        {
            return Err(SpirvError::UnsupportedInstruction {
                pc: inst.pc,
                mnemonic: format!("{} (unsupported MIMG operands, DIM={dim})", inst.mnemonic),
            });
        }
        Ok(Self {
            op,
            dim,
            dmask,
            vdata,
            srsrc,
            ssamp,
            is_nsa,
            addrs,
        })
    }

    fn coord_width(dim: u8) -> u8 {
        match dim {
            0 => 1,
            1 | 4 | 6 => 2,
            _ => 3,
        }
    }

    pub fn address_count(op: u16, dim: u8) -> u8 {
        let width = Self::coord_width(dim);
        match op {
            0x0e => 1,
            0x01 | 0x09 | 0x24 | 0x25 | 0x2f => width + 1,
            0x00 if dim >= 6 => width + 1,
            _ => width,
        }
    }

    pub fn address_registers(self) -> impl Iterator<Item = u8> {
        (0..Self::address_count(self.op, self.dim)).map(move |index| {
            if self.is_nsa {
                self.addrs[index as usize]
            } else {
                self.addrs[0].wrapping_add(index)
            }
        })
    }

    pub fn is_store(self) -> bool {
        self.op == 0x08
    }

    fn samples(self) -> bool {
        self.op >= 0x20
    }

    fn image_key(self, storage: bool) -> ImageKey {
        // Vulkan texel fetch requires a 2D array view of a cube's faces.
        let cube_faces = self.dim == 3 && (storage || matches!(self.op, 0x00 | 0x01));
        ImageKey {
            resource: self.srsrc,
            dim: self.dim,
            storage,
            cube_faces,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ImageKey {
    resource: u8,
    dim: u8,
    storage: bool,
    cube_faces: bool,
}

#[derive(Default)]
struct ImageUse {
    depth: bool,
    read: bool,
    sampled: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct ImageBinding {
    ty: u32,
    var: u32,
    sampled_ty: Option<u32>,
    storage: bool,
}

#[derive(Clone, Copy)]
struct SamplerBinding {
    ty: u32,
    var: u32,
}

pub(crate) struct ImageBindings {
    images: BTreeMap<ImageKey, ImageBinding>,
    samplers: BTreeMap<u8, SamplerBinding>,
}

impl ImageBindings {
    /// Keep the first SRV, sampler, and UAV in their reserved ABI slots.
    /// Assign additional resources in first-use order starting at `extra`.
    pub fn declare(
        b: &mut Builder,
        f32_ty: u32,
        insts: &[DecodedInst],
        fragment: bool,
        [srv, sampler, uav, extra]: [u32; 4],
        interface: &mut Vec<u32>,
    ) -> Result<Self, SpirvError> {
        let mimgs = insts
            .iter()
            .filter(|inst| matches!(inst.extra, InstExtra::Mimg { .. }))
            .map(|inst| Mimg::parse(inst, fragment))
            .collect::<Result<Vec<_>, _>>()?;
        let mut uses = BTreeMap::<ImageKey, ImageUse>::new();
        let mut layout = Self {
            images: BTreeMap::new(),
            samplers: BTreeMap::new(),
        };
        let storage_keys = mimgs
            .iter()
            .filter(|i| i.is_store())
            .map(|i| i.image_key(true))
            .collect::<BTreeSet<_>>();
        let sampled_resources = mimgs
            .iter()
            .filter(|i| i.samples())
            .map(|i| (i.srsrc, i.dim))
            .collect::<BTreeSet<_>>();
        let key_for = |i: Mimg| {
            let storage = i.is_store()
                || (!sampled_resources.contains(&(i.srsrc, i.dim))
                    && storage_keys.contains(&i.image_key(true)));
            i.image_key(storage)
        };
        for &i in &mimgs {
            let usage = uses.entry(key_for(i)).or_default();
            usage.depth |= i.op == 0x2f;
            usage.read |= !i.is_store() && i.op != 0x0e;
            usage.sampled |= i.samples();
        }

        let mut first_srv = Some(srv);
        let mut first_sampler = Some(sampler);
        let mut first_uav = Some(uav);
        let mut extras = extra..;
        for i in mimgs {
            let key = key_for(i);
            layout.images.entry(key).or_insert_with(|| {
                let usage = &uses[&key];
                let (dim, arrayed, ms) = match (key.dim, key.cube_faces) {
                    (0, _) => (Dim::Dim1D, 0, 0),
                    (1, _) => (Dim::Dim2D, 0, 0),
                    (2, _) => (Dim::Dim3D, 0, 0),
                    (3, false) => (Dim::DimCube, 0, 0),
                    (3, true) | (5, _) => (Dim::Dim2D, 1, 0),
                    (4, _) => (Dim::Dim1D, 1, 0),
                    (6, _) => (Dim::Dim2D, 0, 1),
                    (7, _) => (Dim::Dim2D, 1, 1),
                    _ => unreachable!("three-bit image dimension"),
                };
                if dim == Dim::Dim1D {
                    b.capability(if key.storage {
                        Capability::Image1D
                    } else {
                        Capability::Sampled1D
                    });
                }
                if ms == 1 && arrayed == 1 {
                    b.capability(Capability::ImageMSArray);
                }
                let depth = if usage.depth { 2 } else { 0 };
                let sampled = if key.storage { 2 } else { 1 };
                let ty = b.type_image(
                    f32_ty,
                    dim,
                    depth,
                    arrayed,
                    ms,
                    sampled,
                    ImageFormat::Unknown,
                    None,
                );
                let slot = if key.storage {
                    &mut first_uav
                } else {
                    &mut first_srv
                };
                let binding = slot.take().unwrap_or_else(|| extras.next().unwrap());
                let var = declare_descriptor(b, ty, binding, interface);
                if key.storage {
                    b.capability(Capability::StorageImageWriteWithoutFormat);
                    if usage.read {
                        b.capability(Capability::StorageImageReadWithoutFormat);
                    } else {
                        b.decorate(var, Decoration::NonReadable, []);
                    }
                }
                ImageBinding {
                    ty,
                    var,
                    sampled_ty: usage.sampled.then(|| b.type_sampled_image(ty)),
                    storage: key.storage,
                }
            });
            if i.samples() {
                layout.samplers.entry(i.ssamp).or_insert_with(|| {
                    let ty = b.type_sampler();
                    let binding = first_sampler
                        .take()
                        .unwrap_or_else(|| extras.next().unwrap());
                    SamplerBinding {
                        ty,
                        var: declare_descriptor(b, ty, binding, interface),
                    }
                });
            }
        }
        Ok(layout)
    }

    pub fn binding(&self, i: Mimg) -> ImageBinding {
        let key = i.image_key(false);
        if i.is_store() {
            self.images[&i.image_key(true)]
        } else if let Some(&binding) = self.images.get(&key) {
            binding
        } else {
            self.images[&i.image_key(true)]
        }
    }

    pub fn sampler(&self, i: Mimg) -> Option<(u32, u32)> {
        self.samplers.get(&i.ssamp).map(|s| (s.ty, s.var))
    }
}

fn declare_descriptor(b: &mut Builder, ty: u32, binding: u32, interface: &mut Vec<u32>) -> u32 {
    let ptr = b.type_pointer(None, StorageClass::UniformConstant, ty);
    let var = b.variable(ptr, None, StorageClass::UniformConstant, None);
    b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
    b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    interface.push(var);
    var
}

impl Alu<'_> {
    pub(crate) fn emit_cube_alu(
        &mut self,
        op: u16,
        [x, y, z]: [u32; 3],
    ) -> Result<u32, SpirvError> {
        let [ax, ay, az] = [
            self.float_ext(GLOp::FAbs, [x])?,
            self.float_ext(GLOp::FAbs, [y])?,
            self.float_ext(GLOp::FAbs, [z])?,
        ];
        let z_ge_x = self
            .b
            .f_ord_greater_than_equal(self.bool_ty, None, az, ax)?;
        let z_ge_y = self
            .b
            .f_ord_greater_than_equal(self.bool_ty, None, az, ay)?;
        let z_major = self.b.logical_and(self.bool_ty, None, z_ge_x, z_ge_y)?;
        let y_ge_x = self
            .b
            .f_ord_greater_than_equal(self.bool_ty, None, ay, ax)?;
        let zero = self.const_f32(0.0);
        let result = match op {
            0x144 | 0x147 => {
                let xy = self.select_float(y_ge_x, y, x)?;
                let major = self.select_float(z_major, z, xy)?;
                if op == 0x147 {
                    // The instruction returns twice the signed component. Cube sampling
                    // sequences take its absolute value before the reciprocal.
                    let two = self.const_f32(2.0);
                    self.b.f_mul(self.f32_ty, None, two, major)?
                } else {
                    let two = self.const_f32(2.0);
                    let four = self.const_f32(4.0);
                    let one = self.const_f32(1.0);
                    let xy_face = self.select_float(y_ge_x, two, zero)?;
                    let base_face = self.select_float(z_major, four, xy_face)?;
                    let negative = self.b.f_ord_less_than(self.bool_ty, None, major, zero)?;
                    let offset = self.select_float(negative, one, zero)?;
                    self.b.f_add(self.f32_ty, None, base_face, offset)?
                }
            }
            0x145 => {
                let negative_z = self.b.f_ord_less_than(self.bool_ty, None, z, zero)?;
                let negative_x = self.b.f_ord_less_than(self.bool_ty, None, x, zero)?;
                let neg_x = self.b.f_negate(self.f32_ty, None, x)?;
                let neg_z = self.b.f_negate(self.f32_ty, None, z)?;
                let z_face = self.select_float(negative_z, neg_x, x)?;
                let x_face = self.select_float(negative_x, z, neg_z)?;
                let xy_face = self.select_float(y_ge_x, x, x_face)?;
                self.select_float(z_major, z_face, xy_face)?
            }
            0x146 => {
                let negative_y = self.b.f_ord_less_than(self.bool_ty, None, y, zero)?;
                let neg_y = self.b.f_negate(self.f32_ty, None, y)?;
                let neg_z = self.b.f_negate(self.f32_ty, None, z)?;
                let y_face = self.select_float(negative_y, neg_z, z)?;
                let xy_face = self.select_float(y_ge_x, y_face, neg_y)?;
                self.select_float(z_major, neg_y, xy_face)?
            }
            _ => unreachable!("expected cube ALU opcode"),
        };
        Ok(result)
    }

    fn float_ext<const N: usize>(&mut self, op: GLOp, args: [u32; N]) -> Result<u32, SpirvError> {
        Ok(self.b.ext_inst(
            self.f32_ty,
            None,
            self.glsl,
            op as u32,
            args.map(Operand::IdRef),
        )?)
    }

    fn select_float(&mut self, cond: u32, yes: u32, no: u32) -> Result<u32, SpirvError> {
        Ok(self.b.select(self.f32_ty, None, cond, yes, no)?)
    }

    fn image_coord(&mut self, ty: u32, coord: &[u32]) -> Result<u32, SpirvError> {
        if let &[scalar] = coord {
            Ok(scalar)
        } else {
            let vector = self.b.type_vector(ty, coord.len() as u32);
            Ok(self
                .b
                .composite_construct(vector, None, coord.iter().copied())?)
        }
    }

    fn cube_direction(&mut self, [s, t, face]: [u32; 3]) -> Result<[u32; 3], SpirvError> {
        // Cube MIMG takes face-local S/T with a 1.5 bias. Undo that projection
        // so Vulkan can apply cube face selection and seamless filtering itself.
        let center = self.const_f32(1.5);
        let two = self.const_f32(2.0);
        let one = self.const_f32(1.0);
        let neg_one = self.const_f32(-1.0);
        let four = self.const_f32(4.0);
        let limit = self.const_f32(f32::from_bits(1.0f32.to_bits() - 1));
        let neg_limit = self.const_f32(-f32::from_bits(1.0f32.to_bits() - 1));
        let unproject = |alu: &mut Self, coord| {
            let centered = alu.b.f_sub(alu.f32_ty, None, coord, center)?;
            let slope = alu.b.f_mul(alu.f32_ty, None, centered, two)?;
            // Preserve the guest-selected face on exact edge and corner ties.
            alu.float_ext(GLOp::FClamp, [slope, neg_limit, limit])
        };
        let s = unproject(self, s)?;
        let t = unproject(self, t)?;
        let face = self.float_ext(GLOp::RoundEven, [face])?;
        let x_face = self.b.f_ord_less_than(self.bool_ty, None, face, two)?;
        let z_face = self
            .b
            .f_ord_greater_than_equal(self.bool_ty, None, face, four)?;
        let parity = self.b.f_mod(self.f32_ty, None, face, two)?;
        let negative = self.b.f_ord_equal(self.bool_ty, None, parity, one)?;
        let sign = self.select_float(negative, neg_one, one)?;
        let signed_s = self.b.f_mul(self.f32_ty, None, sign, s)?;
        let signed_t = self.b.f_mul(self.f32_ty, None, sign, t)?;
        let neg_s = self.b.f_negate(self.f32_ty, None, signed_s)?;
        let neg_t = self.b.f_negate(self.f32_ty, None, t)?;
        let x_rest = self.select_float(z_face, signed_s, s)?;
        let x = self.select_float(x_face, sign, x_rest)?;
        let x_or_z = self.b.logical_or(self.bool_ty, None, x_face, z_face)?;
        let y = self.select_float(x_or_z, neg_t, sign)?;
        let z_rest = self.select_float(z_face, sign, signed_t)?;
        let z = self.select_float(x_face, neg_s, z_rest)?;
        Ok([x, y, z])
    }

    pub(crate) fn emit_image_read(
        &mut self,
        i: Mimg,
        binding: ImageBinding,
        sampler: Option<(u32, u32)>,
        addr: &[u32],
        fragment: bool,
    ) -> Result<Vec<u32>, SpirvError> {
        let img = self.b.load(binding.ty, None, binding.var, None, [])?;
        let width = usize::from(Mimg::coord_width(i.dim));
        let zero = self.const_u32(0);
        if i.op == 0x0e {
            return self.image_resinfo(i, binding.storage, img, addr[0]);
        }
        let vec4 = self.b.type_vector(self.f32_ty, 4);
        let value = if i.samples() {
            let (sampler_ty, sampler_var) = sampler.expect("sampler declared for MIMG sample");
            let sampler = self.b.load(sampler_ty, None, sampler_var, None, [])?;
            let si = self
                .b
                .sampled_image(binding.sampled_ty.unwrap(), None, img, sampler)?;
            let start = usize::from(matches!(i.op, 0x25 | 0x2f));
            let mut coords = [0; 3];
            for (dst, &bits) in coords.iter_mut().zip(&addr[start..start + width]) {
                *dst = self.b.bitcast(self.f32_ty, None, bits)?;
            }
            if i.dim == 3 {
                coords = self.cube_direction(coords)?;
            } else if matches!(i.dim, 4 | 5) {
                coords[width - 1] = self.float_ext(GLOp::RoundEven, [coords[width - 1]])?;
            }
            let coord = self.image_coord(self.f32_ty, &coords[..width])?;
            let fzero = self.const_f32(0.0);
            match i.op {
                0x2f => {
                    let reference = self.b.bitcast(self.f32_ty, None, addr[0])?;
                    let depth = self.b.image_sample_dref_explicit_lod(
                        self.f32_ty,
                        None,
                        si,
                        coord,
                        reference,
                        ImageOperands::LOD,
                        [Operand::IdRef(fzero)],
                    )?;
                    let bits = self.b.bitcast(self.uint, None, depth)?;
                    return Ok(vec![bits; i.dmask.count_ones() as usize]);
                }
                0x40 | 0x47 => {
                    let component = self.const_u32(i.dmask.trailing_zeros());
                    self.b
                        .image_gather(vec4, None, si, coord, component, None, [])?
                }
                0x25 => {
                    let bias = self.b.bitcast(self.f32_ty, None, addr[0])?;
                    self.b.image_sample_implicit_lod(
                        vec4,
                        None,
                        si,
                        coord,
                        Some(ImageOperands::BIAS),
                        [Operand::IdRef(bias)],
                    )?
                }
                0x20 if fragment => {
                    self.b
                        .image_sample_implicit_lod(vec4, None, si, coord, None, [])?
                }
                _ => {
                    let lod = if i.op == 0x24 {
                        self.b.bitcast(self.f32_ty, None, addr[width])?
                    } else {
                        fzero
                    };
                    self.b.image_sample_explicit_lod(
                        vec4,
                        None,
                        si,
                        coord,
                        ImageOperands::LOD,
                        [Operand::IdRef(lod)],
                    )?
                }
            }
        } else {
            let coord = self.image_coord(self.uint, &addr[..width])?;
            if binding.storage {
                let texel = self.b.image_read(vec4, None, img, coord, None, [])?;
                if i.op == 0x01 {
                    // A storage image view exposes one mip level.
                    let base_level = self.b.i_equal(self.bool_ty, None, addr[width], zero)?;
                    let empty = self.b.constant_null(vec4);
                    self.b.select(vec4, None, base_level, texel, empty)?
                } else {
                    texel
                }
            } else {
                let (kind, level) = match (i.op, i.dim) {
                    (_, 6 | 7) => (ImageOperands::SAMPLE, addr[width]),
                    (0x01, _) => (ImageOperands::LOD, addr[width]),
                    _ => (ImageOperands::LOD, zero),
                };
                self.b
                    .image_fetch(vec4, None, img, coord, Some(kind), [Operand::IdRef(level)])?
            }
        };
        let mask = if matches!(i.op, 0x40 | 0x47) {
            0xf
        } else {
            i.dmask
        };
        (0..4)
            .filter(|c| mask & (1 << c) != 0)
            .map(|c| {
                let comp = self.b.composite_extract(self.f32_ty, None, value, [c])?;
                Ok(self.b.bitcast(self.uint, None, comp)?)
            })
            .collect()
    }

    fn image_resinfo(
        &mut self,
        i: Mimg,
        storage: bool,
        img: u32,
        mip: u32,
    ) -> Result<Vec<u32>, SpirvError> {
        self.b.capability(Capability::ImageQuery);
        let (width, component_map) = match i.dim {
            0 => (1, [Some(0), None, None]),
            4 => (2, [Some(0), None, Some(1)]),
            2 | 5 | 7 => (3, [Some(0), Some(1), Some(2)]),
            3 if storage => (3, [Some(0), Some(1), None]),
            _ => (2, [Some(0), Some(1), None]),
        };
        let one = self.const_u32(1);
        let zero = self.const_u32(0);
        let size_ty = if width == 1 {
            self.uint
        } else {
            self.b.type_vector(self.uint, width)
        };
        let levels = if storage || i.dim >= 6 {
            one
        } else {
            self.b.image_query_levels(self.uint, None, img)?
        };
        let in_range = self.b.u_less_than(self.bool_ty, None, mip, levels)?;
        let size = if storage || i.dim >= 6 {
            self.b.image_query_size(size_ty, None, img)?
        } else {
            let lod = self.b.select(self.uint, None, in_range, mip, zero)?;
            self.b.image_query_size_lod(size_ty, None, img, lod)?
        };
        let mut result = Vec::with_capacity(i.dmask.count_ones() as usize);
        for (component, index) in component_map.into_iter().enumerate() {
            if i.dmask & (1 << component) != 0 {
                let extent = match index {
                    Some(_) if width == 1 => size,
                    Some(index) => self.b.composite_extract(self.uint, None, size, [index])?,
                    None => one,
                };
                result.push(self.b.select(self.uint, None, in_range, extent, zero)?);
            }
        }
        if i.dmask & 8 != 0 {
            result.push(levels);
        }
        Ok(result)
    }

    pub(crate) fn emit_image_store(
        &mut self,
        i: Mimg,
        binding: ImageBinding,
        addr: &[u32],
        data: &[u32],
        active: u32,
    ) -> Result<(), SpirvError> {
        let store = self.b.id();
        let merge = self.b.id();
        self.b.selection_merge(merge, SelectionControl::NONE)?;
        self.b.branch_conditional(active, store, merge, [])?;
        self.b.begin_block(Some(store))?;
        let coord = self.image_coord(self.uint, addr)?;
        let zero = self.const_f32(0.0);
        let mut comps = [zero; 4];
        let mut packed = data.iter().copied();
        for (index, value) in comps.iter_mut().enumerate() {
            if i.dmask & (1 << index) != 0 {
                *value = self.b.bitcast(self.f32_ty, None, packed.next().unwrap())?;
            }
        }
        let vec4 = self.b.type_vector(self.f32_ty, 4);
        let texel = self.b.composite_construct(vec4, None, comps)?;
        let img = self.b.load(binding.ty, None, binding.var, None, [])?;
        self.b.image_write(img, coord, texel, None, [])?;
        self.b.branch(merge)?;
        self.b.begin_block(Some(merge))?;
        Ok(())
    }
}
