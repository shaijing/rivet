use super::{RivetResult, contiguous_values, invalid_argument};
use rivet_core::{CpuStorageRef, DType, Tensor, WithDType};
use rivet_data::random::RandomStream;

pub(super) fn fancy_pca(input: &Tensor, alpha: f64, rng: &mut RandomStream) -> RivetResult<Tensor> {
    let [height, width, channels] = *input.dims() else {
        return Err(invalid_argument("FancyPCA expects rank-3 HWC input"));
    };
    if height == 0
        || width == 0
        || !matches!(channels, 1 | 3)
        || !matches!(input.dtype(), DType::U8 | DType::F32)
    {
        return Err(invalid_argument(
            "FancyPCA requires nonempty uint8 or float32 HWC grayscale or RGB input",
        ));
    }
    if alpha == 0.0 || height * width == 1 {
        return Ok(input.clone());
    }
    let input = input.contiguous()?;
    Ok(input.with_cpu_storage(|storage, layout| match storage {
        CpuStorageRef::U8(values) => apply(
            &input,
            contiguous_values(values, layout)?,
            channels,
            1.0 / 255.0,
            alpha,
            rng,
            |value| (value * 255.0).round() as u8,
        ),
        CpuStorageRef::F32(values) => apply(
            &input,
            contiguous_values(values, layout)?,
            channels,
            1.0,
            alpha,
            rng,
            |value| value as f32,
        ),
        _ => unreachable!(),
    })?)
}

fn apply<T: WithDType + Into<f64>>(
    input: &Tensor,
    pixels: &[T],
    channels: usize,
    scale: f64,
    alpha: f64,
    rng: &mut RandomStream,
    cast: impl Fn(f64) -> T,
) -> rivet_core::Result<Tensor> {
    let count = (pixels.len() / channels) as f64;
    let mut mean = [0.0; 3];
    for pixel in pixels.chunks_exact(channels) {
        for (sum, &value) in mean.iter_mut().zip(pixel) {
            *sum += value.into() * scale;
        }
    }
    mean = mean.map(|sum| sum / count);
    let mut covariance = [[0.0; 3]; 3];
    for pixel in pixels.chunks_exact(channels) {
        let mut delta = [0.0; 3];
        for (i, &value) in pixel.iter().enumerate() {
            delta[i] = value.into() * scale - mean[i];
        }
        for i in 0..channels {
            for j in i..channels {
                covariance[i][j] += delta[i] * delta[j];
            }
        }
    }
    for i in 0..channels {
        for j in i..channels {
            covariance[i][j] /= count - 1.0;
            covariance[j][i] = covariance[i][j];
        }
    }
    let (values, axes) = diagonalize(covariance);
    let (a, b) = super::noise::normal_pair(rng);
    let (c, _) = super::noise::normal_pair(rng);
    let weights = [a, b, c];
    let offsets = axes.map(|axis| {
        (axis
            .into_iter()
            .zip(values)
            .zip(weights)
            .map(|((direction, variance), noise)| direction * variance.max(0.0) * noise)
            .sum::<f64>())
            * alpha
    });
    Tensor::from_exact_writer(input.dims(), input.device(), |out| {
        for pixel in pixels.chunks_exact(channels) {
            for (&v, offset) in pixel.iter().zip(offsets) {
                out.write_next(cast((v.into() * scale + offset).clamp(0.0, 1.0)))?;
            }
        }
        Ok(())
    })
}

/// Jacobi rotations for a real symmetric 3×3 matrix. Columns of the
/// accumulated rotations are the principal axes.
fn diagonalize(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        let (p, q) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|&(p, q), &(r, s)| a[p][q].abs().total_cmp(&a[r][s].abs()))
            .expect("three coordinate pairs");
        let off = a[p][q];
        if off.abs() <= f64::EPSILON {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * off);
        let t = tau.signum() / (tau.abs() + tau.hypot(1.0));
        let cos = t.hypot(1.0).recip();
        let sin = t * cos;
        a[p][p] -= t * off;
        a[q][q] += t * off;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for i in 0..3 {
            if i != p && i != q {
                let (ap, aq) = (a[i][p], a[i][q]);
                a[i][p] = cos * ap - sin * aq;
                a[i][q] = sin * ap + cos * aq;
                a[p][i] = a[i][p];
                a[q][i] = a[i][q];
            }
            let (vp, vq) = (axes[i][p], axes[i][q]);
            axes[i][p] = cos * vp - sin * vq;
            axes[i][q] = sin * vp + cos * vq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], axes)
}

#[cfg(test)]
mod tests {
    use super::diagonalize;

    #[test]
    fn jacobi_axes_reconstruct_correlated_covariance() {
        let matrix = [[0.14, -0.04, 0.02], [-0.04, 0.09, 0.03], [0.02, 0.03, 0.05]];
        let (values, axes) = diagonalize(matrix);
        for (i, row) in matrix.iter().enumerate() {
            for (j, &expected) in row.iter().enumerate() {
                let reconstructed: f64 = (0..3).map(|k| axes[i][k] * values[k] * axes[j][k]).sum();
                assert!((expected - reconstructed).abs() < 1e-13);
            }
        }
    }
}
