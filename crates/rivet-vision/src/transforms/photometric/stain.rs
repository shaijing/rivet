use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::{contiguous_values, photometric::pca::diagonalize};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

type Vector = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StainBasis {
    Preset(usize),
    RandomPreset,
    Macenko,
    Vahadane,
    Custom { rows: [Vector; 3], count: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StainResidual {
    Project,
    Preserve,
    Augment,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HEStainConfig {
    pub basis: StainBasis,
    pub scale: [f64; 2],
    pub shift: [f64; 2],
    pub augment_background: bool,
    pub residual: StainResidual,
}

// Reference H&E directions and fixed variants spanning common color balances.
const PRESETS: [[Vector; 2]; 8] = [
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]],
    [[0.56, 0.72, 0.41], [0.22, 0.80, 0.56]],
    [[0.64, 0.71, 0.29], [0.14, 0.95, 0.27]],
    [[0.68, 0.70, 0.21], [0.04, 0.99, 0.12]],
    [[0.71, 0.63, 0.31], [0.18, 0.94, 0.30]],
    [[0.58, 0.76, 0.29], [0.14, 0.91, 0.39]],
    [[0.64, 0.65, 0.40], [0.16, 0.85, 0.50]],
    [[0.56, 0.77, 0.30], [0.24, 0.95, 0.20]],
];

fn dot(a: Vector, b: Vector) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}

fn cross(a: Vector, b: Vector) -> Vector {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(v: Vector) -> Option<Vector> {
    let magnitude = dot(v, v).sqrt();
    (magnitude.is_finite() && magnitude > 0.0).then(|| v.map(|x| x / magnitude))
}

fn dual(rows: [Vector; 3]) -> Option<[Vector; 3]> {
    let mut cofactors = [
        cross(rows[1], rows[2]),
        cross(rows[2], rows[0]),
        cross(rows[0], rows[1]),
    ];
    let determinant = dot(rows[0], cofactors[0]);
    let norm_product = rows.iter().map(|&r| dot(r, r).sqrt()).product::<f64>();
    if !determinant.is_finite()
        || !norm_product.is_finite()
        || determinant.abs() <= 1e-10 * norm_product
    {
        return None;
    }
    for row in &mut cofactors {
        *row = row.map(|v| v / determinant);
    }
    Some(cofactors)
}

fn complete(mut rows: [Vector; 3], count: usize) -> Option<([Vector; 3], [Vector; 3])> {
    if count == 2 {
        rows[2] = unit(cross(rows[0], rows[1]))?;
    }
    Some((rows, dual(rows)?))
}

fn optical_density<T: Into<f64> + Copy>(pixel: &[T], limit: f64) -> Vector {
    std::array::from_fn(|i| -(pixel[i].into() / limit).clamp(1.0 / 255.0, 1.0).ln())
}

fn macenko(samples: &[Vector]) -> Option<[Vector; 2]> {
    if samples.len() < 3 {
        return None;
    }
    let count = samples.len() as f64;
    let mean: Vector = std::array::from_fn(|i| samples.iter().map(|s| s[i]).sum::<f64>() / count);
    let covariance = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            samples
                .iter()
                .map(|s| (s[i] - mean[i]) * (s[j] - mean[j]))
                .sum::<f64>()
                / count
        })
    });
    let (values, axes) = diagonalize(covariance);
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    if values[order[1]] <= 1e-8 {
        return None;
    }
    let mut u = axes.map(|row| row[order[0]]);
    if dot(u, mean) < 0.0 {
        u = u.map(|v| -v);
    }
    let v = axes.map(|row| row[order[1]]);
    let mut angles: Vec<_> = samples
        .iter()
        .map(|&s| dot(v, s).atan2(dot(u, s)))
        .collect();
    angles.sort_by(f64::total_cmp);
    let p = (angles.len() - 1) / 100;
    Some([angles[p], angles[angles.len() - 1 - p]].map(|angle| {
        let (sin, cos) = angle.sin_cos();
        std::array::from_fn(|i| (u[i] * cos + v[i] * sin).max(0.0))
    }))
}

// Sparse nonnegative dictionary learning with two unit-length stain atoms.
fn vahadane(samples: &[Vector], initial: [Vector; 2]) -> [Vector; 2] {
    let mut basis = initial;
    for _ in 0..32 {
        let correlation = dot(basis[0], basis[1]);
        let mut gradient = [[0.0; 3]; 2];
        let mut energy = [0.0; 2];
        for &sample in samples {
            let target = basis.map(|axis| (dot(axis, sample) - 0.03).max(0.0));
            let mut code = [0.0; 2];
            for _ in 0..8 {
                code[0] = (target[0] - correlation * code[1]).max(0.0);
                code[1] = (target[1] - correlation * code[0]).max(0.0);
            }
            for i in 0..3 {
                let residual = sample[i] - code[0] * basis[0][i] - code[1] * basis[1][i];
                for j in 0..2 {
                    gradient[j][i] += residual * code[j];
                }
            }
            for j in 0..2 {
                energy[j] += code[j] * code[j];
            }
        }
        for j in 0..2 {
            let step = 0.5 / energy[j].max(1e-8);
            if let Some(v) = unit(std::array::from_fn(|i| {
                (basis[j][i] + step * gradient[j][i]).max(0.0)
            })) {
                basis[j] = v;
            }
        }
    }
    basis
}

impl HEStainConfig {
    pub fn validate(self) -> RivetResult<()> {
        let [a, b] = self.scale;
        let [c, d] = self.shift;
        if !a.is_finite()
            || !b.is_finite()
            || a < 0.0
            || b < a
            || !(-1.0..=1.0).contains(&c)
            || !(c..=1.0).contains(&d)
        {
            return Err(invalid_argument("invalid HEStain scale or shift"));
        }
        match self.basis {
            StainBasis::Preset(i) if i >= PRESETS.len() => {
                Err(invalid_argument("invalid HEStain preset"))
            }
            StainBasis::Custom { rows, count }
                if !(2..=3).contains(&count)
                    || (count == 3 && self.residual == StainResidual::Project)
                    || complete(rows, count).is_none() =>
            {
                Err(invalid_argument(
                    "HEStain custom matrix must have independent finite rows and a compatible residual mode",
                ))
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn apply(self, input: &Tensor, rng: &mut RandomStream) -> RivetResult<Tensor> {
        self.validate()?;
        if input.dims().last() != Some(&3) || !matches!(input.dtype(), DType::U8 | DType::F32) {
            return Err(invalid_argument(
                "HEStain requires channel-last uint8 or float32 RGB",
            ));
        }
        let input = input.contiguous()?;
        Ok(input.with_cpu_storage(|storage, layout| match storage {
            CpuStorageRef::U8(v) => {
                self.write(&input, contiguous_values(v, layout)?, 255.0, rng, |v| {
                    (255.0 * v).round() as u8
                })
            }
            CpuStorageRef::F32(v) => {
                self.write(&input, contiguous_values(v, layout)?, 1.0, rng, |v| {
                    v as f32
                })
            }
            _ => unreachable!("validated HEStain dtype"),
        })?)
    }

    fn write<T: WithDType + Into<f64>>(
        self,
        input: &Tensor,
        pixels: &[T],
        limit: f64,
        rng: &mut RandomStream,
        cast: impl Fn(f64) -> T,
    ) -> rivet_core::Result<Tensor> {
        let (pair, count) = match self.basis {
            StainBasis::Custom { rows, count } => (rows, count),
            basis => {
                let stains = match basis {
                    StainBasis::Preset(i) => PRESETS[i],
                    StainBasis::RandomPreset => PRESETS[(rng.next_u64() & 7) as usize],
                    StainBasis::Macenko | StainBasis::Vahadane => {
                        let samples: Vec<_> = pixels
                            .chunks_exact(3)
                            .step_by((pixels.len() / 3).div_ceil(4096).max(1))
                            .map(|p| optical_density(p, limit))
                            .filter(|&p| dot(p, p) > 0.15_f64.powi(2))
                            .collect();
                        let mut initial = macenko(&samples)
                            .unwrap_or(PRESETS[2])
                            .map(|row| unit(row).unwrap_or(PRESETS[2][0]));
                        if basis == StainBasis::Vahadane {
                            initial = vahadane(&samples, initial);
                        }
                        if initial[0][0] < initial[1][0] {
                            initial.swap(0, 1);
                        }
                        initial
                    }
                    StainBasis::Custom { .. } => unreachable!(),
                };
                ([stains[0], stains[1], [0.0; 3]], 2)
            }
        };
        let (rows, inverse) = complete(pair, count)
            .or_else(|| complete([PRESETS[2][0], PRESETS[2][1], [0.0; 3]], 2))
            .expect("reference stains are independent");
        let mut draw = |[a, b]: [f64; 2]| a + (b - a) * rng.next_f64();
        let adjustments = (0..3)
            .map(|_| (draw(self.scale), draw(self.shift)))
            .collect::<Vec<_>>();
        let transform = |od: Vector| {
            let mut output = [0.0; 3];
            for i in 0..3 {
                let concentration = dot(od, inverse[i]);
                let (scale, shift) = adjustments[i];
                let value = match (i, self.residual) {
                    (2, StainResidual::Project) => 0.0,
                    (2, StainResidual::Preserve) => concentration,
                    _ => concentration * scale + shift,
                };
                for (out, stain) in output.iter_mut().zip(rows[i]) {
                    *out += value * stain;
                }
            }
            output
        };
        Tensor::from_exact_writer(input.dims(), input.device(), |out| {
            for pixel in pixels.chunks_exact(3) {
                let od = optical_density(pixel, limit);
                if !self.augment_background && dot(od, od) < 0.15_f64.powi(2) {
                    out.extend_from_slice(pixel)?;
                } else {
                    for value in transform(od) {
                        out.write_next(cast((-value).exp().clamp(0.0, 1.0)))?;
                    }
                }
            }
            Ok(())
        })
    }
}
