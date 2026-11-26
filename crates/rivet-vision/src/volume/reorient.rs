use super::VolumeSample;
use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use rivet_core::{CpuStorageRef, Tensor};

impl VolumeSample {
    /// Reorient output DHW axes by permuting input axes and then reflecting
    /// selected output axes. The channel axis and label are preserved.
    pub fn reorient(mut self, order: [usize; 3], reflected_axes: &[usize]) -> RivetResult<Self> {
        self.validate()?;
        if volume_axis_mask(&order)? != 0b111 {
            return Err(invalid_argument("3D axis order must be a permutation"));
        }
        let reflected = volume_axis_mask(reflected_axes)?;
        let shape = self.spatial_shape()?;
        let reorder = |t: Tensor| -> rivet_core::Result<Tensor> {
            let t = if order == [0, 1, 2] {
                t
            } else if t.rank() == 4 {
                t.permute(&[order[0], order[1], order[2], 3])?
            } else {
                t.permute(&order)?
            };
            if reflected_axes.is_empty() {
                Ok(t)
            } else {
                t.flip(reflected_axes)
            }
        };
        self.volume = reorder(self.volume)?;
        self.mask = self.mask.map(reorder).transpose()?;
        if let Some(points) = &self.keypoints
            && (order != [0, 1, 2] || reflected != 0)
        {
            let coords = points.to_vec::<f32>()?;
            self.keypoints = Some(Tensor::from_exact_iter(
                (0..coords.len()).map(|i| {
                    let axis = 2 - i % 3;
                    let original = order[axis];
                    let value = coords[i / 3 * 3 + 2 - original];
                    if reflected & (1 << axis) == 0 {
                        value
                    } else {
                        ((shape[original] - 1) as f64 - f64::from(value)) as f32
                    }
                }),
                points.dims(),
                points.device(),
            )?);
        }
        Ok(self)
    }

    pub fn rotate90(self, [a, b]: [usize; 2], turns: u8) -> RivetResult<Self> {
        if a >= b || b >= 3 || turns >= 4 {
            return Err(invalid_argument(
                "expected ascending DHW axes and 0 to 3 quarter turns",
            ));
        }
        let mut order = [0, 1, 2];
        if turns % 2 == 1 {
            order.swap(a, b);
        }
        match turns {
            0 => self.reorient(order, &[]),
            1 => self.reorient(order, &[a]),
            2 => self.reorient(order, &[a, b]),
            _ => self.reorient(order, &[b]),
        }
    }

    /// Apply one of the 48 signed spatial permutations, with identity at 0.
    pub fn cubic_symmetry(self, element: usize) -> RivetResult<Self> {
        const PERMUTATIONS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let order = PERMUTATIONS
            .get(element / 8)
            .copied()
            .ok_or_else(|| invalid_argument("CubicSymmetry element must be in [0, 47]"))?;
        let reflected = (0..3)
            .filter(|axis| element & (1 << axis) != 0)
            .collect::<Vec<_>>();
        self.reorient(order, &reflected)
    }
}

pub fn volume_axis_mask(axes: &[usize]) -> RivetResult<u8> {
    let mut mask = 0;
    for &axis in axes {
        if axis >= 3 || mask & (1 << axis) != 0 {
            return Err(invalid_argument("expected distinct DHW spatial axes"));
        }
        mask |= 1 << axis;
    }
    Ok(mask)
}

impl VolumeSample {
    pub fn grid_shuffle(
        mut self,
        grid: [usize; 3],
        rng: &mut rivet_data::random::RandomStream,
    ) -> RivetResult<Self> {
        self.validate()?;
        let shape = self.spatial_shape()?;
        if (0..3).any(|i| grid[i] == 0 || grid[i] > shape[i]) {
            return Err(invalid_shape("GridShuffle3D grid must fit the DHW volume"));
        }
        let bounds: [Vec<usize>; 3] = std::array::from_fn(|i| {
            let count = grid[i];
            let (width, extra) = (shape[i] / count, shape[i] % count);
            (0..=count).map(|j| j * width + j.min(extra)).collect()
        });
        let mut origins = Vec::new();
        let mut groups = [const { Vec::new() }; 8];
        for z in 0..grid[0] {
            for y in 0..grid[1] {
                for x in 0..grid[2] {
                    let cell = [z, y, x];
                    let class = (0..3).fold(0, |bits, i| {
                        bits | (usize::from(cell[i] < shape[i] % grid[i]) << i)
                    });
                    groups[class].push(origins.len());
                    origins.push(std::array::from_fn(|i| bounds[i][cell[i]]));
                }
            }
        }
        let mut sources: Vec<_> = (0..origins.len()).collect();
        for group in &groups {
            for i in (1..group.len()).rev() {
                let j = rng.gen_range_usize(0..i + 1)?;
                sources.swap(group[i], group[j]);
            }
        }
        if sources.iter().copied().eq(0..origins.len()) {
            return Ok(self);
        }
        let shuffle = |t: &Tensor| shuffle_volume(t, &bounds, grid, &origins, &sources);
        self.volume = shuffle(&self.volume)?;
        self.mask = self.mask.as_ref().map(shuffle).transpose()?;
        if let Some(points) = &self.keypoints {
            let mut destination = vec![0; sources.len()];
            for (dest, &source) in sources.iter().enumerate() {
                destination[source] = dest;
            }
            let mut coords = points.to_vec::<f32>()?;
            for point in coords.chunks_exact_mut(3) {
                let zyx = [point[2], point[1], point[0]];
                if (0..3).all(|i| zyx[i] >= 0.0 && f64::from(zyx[i]) < shape[i] as f64) {
                    let [z, y, x]: [usize; 3] = std::array::from_fn(|i| {
                        bounds[i].partition_point(|&bound| bound <= zyx[i] as usize) - 1
                    });
                    let source = (z * grid[1] + y) * grid[2] + x;
                    let dest = destination[source];
                    for i in 0..3 {
                        point[2 - i] = (f64::from(zyx[i]) + origins[dest][i] as f64
                            - origins[source][i] as f64)
                            as f32;
                    }
                }
            }
            self.keypoints = Some(Tensor::from_vec(coords, points.dims(), points.device())?);
        }
        Ok(self)
    }
}

fn shuffle_volume(
    tensor: &Tensor,
    bounds: &[Vec<usize>; 3],
    grid: [usize; 3],
    origins: &[[usize; 3]],
    sources: &[usize],
) -> RivetResult<Tensor> {
    let input = tensor.contiguous()?;
    let channels = input.dims().get(3).copied().unwrap_or(1);
    let height = input.dims()[1];
    let width = input.dims()[2];
    Ok(input.with_cpu_storage(|storage, layout| {
        macro_rules! copy_rows {
            ($values:expr) => {
                Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                    for z in 0..grid[0] {
                        for dz in 0..bounds[0][z + 1] - bounds[0][z] {
                            for y in 0..grid[1] {
                                for dy in 0..bounds[1][y + 1] - bounds[1][y] {
                                    let first_cell = (z * grid[1] + y) * grid[2];
                                    for x in 0..grid[2] {
                                        let [sz, sy, sx] = origins[sources[first_cell + x]];
                                        let first = layout.start_offset()
                                            + (((sz + dz) * height + sy + dy) * width + sx)
                                                * channels;
                                        let len = (bounds[2][x + 1] - bounds[2][x]) * channels;
                                        out.extend_from_slice(&$values[first..first + len])?;
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                })
            };
        }
        match storage {
            CpuStorageRef::U8(v) => copy_rows!(v),
            CpuStorageRef::F32(v) => copy_rows!(v),
            CpuStorageRef::I64(v) => copy_rows!(v),
            _ => unreachable!("validated volume grid dtype"),
        }
    })?)
}
