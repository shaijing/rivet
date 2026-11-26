use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use rivet_core::{CpuStorageRef, Tensor, WithDType};
use rivet_data::random::RandomStream;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomGridShuffleConfig {
    /// Grid rows and columns.
    pub grid: (u32, u32),
}

impl RandomGridShuffleConfig {
    pub(in crate::transforms) fn valid(&self) -> bool {
        [self.grid.0, self.grid.1]
            .into_iter()
            .all(|d| (1..=256).contains(&d))
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<GridShuffleMap> {
        if h == 0 || w == 0 {
            return Err(invalid_argument(
                "RandomGridShuffle requires nonempty spatial axes",
            ));
        }
        let (rows, cols) = ((self.grid.0 as usize).min(h), (self.grid.1 as usize).min(w));
        let edges = |n: usize, k: usize| {
            (0..=k)
                .map(|i| i * (n / k) + i.min(n % k))
                .collect::<Vec<_>>()
        };
        let (ys, xs) = (edges(h, rows), edges(w, cols));
        let mut blocks = Vec::with_capacity(rows * cols);
        let mut groups = BTreeMap::<(usize, usize), Vec<usize>>::new();
        for y in ys.windows(2) {
            for x in xs.windows(2) {
                groups
                    .entry((y[1] - y[0], x[1] - x[0]))
                    .or_default()
                    .push(blocks.len());
                blocks.push([x[0], y[0], x[1], y[1]]);
            }
        }
        let mut source: Vec<_> = (0..blocks.len()).collect();
        for group in groups.values() {
            for i in (1..group.len()).rev() {
                let j = rng.gen_range_usize(0..i + 1)?;
                source.swap(group[i], group[j]);
            }
        }
        let mut destination = vec![0; blocks.len()];
        for (output, &input) in source.iter().enumerate() {
            destination[input] = output;
        }
        let identity = source.iter().enumerate().all(|(dest, &src)| dest == src);
        let indices = |edges: Vec<usize>| {
            edges
                .windows(2)
                .enumerate()
                .flat_map(|(i, v)| std::iter::repeat_n(i, v[1] - v[0]))
                .collect()
        };
        Ok(GridShuffleMap {
            blocks,
            source,
            destination,
            cols,
            y_cells: indices(ys),
            x_cells: indices(xs),
            identity,
        })
    }
}

pub(crate) struct GridShuffleMap {
    blocks: Vec<[usize; 4]>,
    source: Vec<usize>,
    destination: Vec<usize>,
    cols: usize,
    y_cells: Vec<usize>,
    x_cells: Vec<usize>,
    pub(crate) identity: bool,
}

impl GridShuffleMap {
    fn translate(&self, p: [usize; 2], mapping: &[usize]) -> [usize; 2] {
        let tile = self.y_cells[p[1]] * self.cols + self.x_cells[p[0]];
        let a = self.blocks[tile];
        let b = self.blocks[mapping[tile]];
        [p[0] - a[0] + b[0], p[1] - a[1] + b[1]]
    }

    pub(crate) fn keypoint(&self, [x, y]: [f32; 2]) -> [f32; 2] {
        if x < 0.0
            || y < 0.0
            || x as usize >= self.x_cells.len()
            || y as usize >= self.y_cells.len()
        {
            return [x, y];
        }
        let (ix, iy) = (x.floor() as usize, y.floor() as usize);
        let [px, py] = self.translate([ix, iy], &self.destination);
        [px as f32 + (x - ix as f32), py as f32 + (y - iy as f32)]
    }

    pub(crate) fn box_bounds(&self, b: &[f32]) -> Option<[f32; 4]> {
        let mut out = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (i, a) in self.blocks.iter().enumerate() {
            let p = self.blocks[self.destination[i]];
            let (dx, dy) = (p[0] as f64 - a[0] as f64, p[1] as f64 - a[1] as f64);
            let [l, t, r, b] = [
                f64::from(b[0]).max(a[0] as f64),
                f64::from(b[1]).max(a[1] as f64),
                f64::from(b[2]).min(a[2] as f64),
                f64::from(b[3]).min(a[3] as f64),
            ];
            if l < r && t < b {
                out = [
                    out[0].min(l + dx),
                    out[1].min(t + dy),
                    out[2].max(r + dx),
                    out[3].max(b + dy),
                ];
            }
        }
        (out[0] < out[2] && out[1] < out[3]).then(|| out.map(|x| x as f32))
    }

    pub(crate) fn apply(&self, input: &Tensor, axis: ImageAxisOrder) -> RivetResult<Tensor> {
        if self.identity {
            return Ok(input.clone());
        }
        let (c, ay, ax, ac) = match (input.dims(), axis) {
            (&[_, _], _) => (1, 0, 1, 2),
            (&[_, _, c], ImageAxisOrder::Hwc) => (c, 0, 1, 2),
            (&[c, _, _], ImageAxisOrder::Chw) => (c, 1, 2, 0),
            _ => return Err(invalid_argument("RandomGridShuffle expects HW, HWC or CHW")),
        };
        let (h, w) = (self.y_cells.len(), self.x_cells.len());
        if input.dims()[ay] != h || input.dims()[ax] != w {
            return Err(invalid_argument(
                "RandomGridShuffle spatial axes do not match the grid",
            ));
        }
        fn write<T: WithDType>(
            input: &Tensor,
            pixels: &[T],
            map: &GridShuffleMap,
            offset: usize,
            [sy, sx, sc]: [usize; 3],
            [h, w, c]: [usize; 3],
            planar: bool,
        ) -> rivet_core::Result<Tensor> {
            let read = |y, x| {
                let [x, y] = map.translate([x, y], &map.source);
                offset + y * sy + x * sx
            };
            Tensor::from_exact_writer(input.dims(), input.device(), |out| {
                if planar {
                    for c in 0..c {
                        for y in 0..h {
                            for x in 0..w {
                                out.write_next(pixels[read(y, x) + c * sc])?;
                            }
                        }
                    }
                } else {
                    for y in 0..h {
                        for x in 0..w {
                            let base = read(y, x);
                            for c in 0..c {
                                out.write_next(pixels[base + c * sc])?;
                            }
                        }
                    }
                }
                Ok(())
            })
        }
        Ok(input.with_cpu_storage(|storage, layout| {
            let strides = [
                layout.stride()[ay],
                layout.stride()[ax],
                layout.stride().get(ac).copied().unwrap_or(0),
            ];
            let planar = axis == ImageAxisOrder::Chw && input.rank() == 3;
            macro_rules! run {
                ($v:expr) => {
                    write(
                        input,
                        $v,
                        self,
                        layout.start_offset(),
                        strides,
                        [h, w, c],
                        planar,
                    )
                };
            }
            match storage {
                CpuStorageRef::U8(v) => run!(v),
                CpuStorageRef::F32(v) => run!(v),
                CpuStorageRef::I64(v) => run!(v),
                other => Err(rivet_core::Error::UnexpectedDType {
                    expected: rivet_core::DType::U8,
                    actual: other.dtype(),
                }),
            }
        })?)
    }
}
