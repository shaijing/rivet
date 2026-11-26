use rustfft::{Fft, FftPlanner, num_complex::Complex64};
use std::sync::Arc;

/// Reusable 2D FFT plans and scratch storage for row-major planes.
pub(crate) struct Fft2 {
    width: usize,
    height: usize,
    forward: [Arc<dyn Fft<f64>>; 2],
    inverse: [Arc<dyn Fft<f64>>; 2],
    column: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl Fft2 {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let mut planner = FftPlanner::new();
        let forward = [
            planner.plan_fft_forward(width),
            planner.plan_fft_forward(height),
        ];
        let inverse = [
            planner.plan_fft_inverse(width),
            planner.plan_fft_inverse(height),
        ];
        let scratch_len = forward
            .iter()
            .chain(&inverse)
            .map(|plan| plan.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        Self {
            width,
            height,
            forward,
            inverse,
            column: vec![Complex64::default(); height],
            scratch: vec![Complex64::default(); scratch_len],
        }
    }

    pub(crate) fn transform(&mut self, plane: &mut [Complex64], inverse: bool) {
        let [row, col] = if inverse {
            &self.inverse
        } else {
            &self.forward
        };
        for values in plane.chunks_exact_mut(self.width) {
            row.process_with_scratch(values, &mut self.scratch);
        }
        for x in 0..self.width {
            for (v, r) in self.column.iter_mut().zip(plane.chunks_exact(self.width)) {
                *v = r[x];
            }
            col.process_with_scratch(&mut self.column, &mut self.scratch);
            for (r, &v) in plane.chunks_exact_mut(self.width).zip(&self.column) {
                r[x] = v;
            }
        }
        if inverse {
            let count = (self.width * self.height) as f64;
            for v in plane {
                *v /= count;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_inverse_roundtrips_rectangular_odd_planes() {
        let mut fft = Fft2::new(7, 5);
        let expected: Vec<_> = (0..35)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect();
        let mut plane = expected.clone();
        fft.transform(&mut plane, false);
        fft.transform(&mut plane, true);
        for (a, b) in plane.into_iter().zip(expected) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}

/// Separable DHW FFT. Inversion normalizes each spatial axis exactly once.
pub(crate) struct Fft3 {
    plane: Fft2,
    depth: usize,
    depth_forward: Arc<dyn Fft<f64>>,
    depth_inverse: Arc<dyn Fft<f64>>,
    line: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl Fft3 {
    pub(crate) fn new([d, h, w]: [usize; 3]) -> Self {
        let mut planner = FftPlanner::new();
        let depth_forward = planner.plan_fft_forward(d);
        let depth_inverse = planner.plan_fft_inverse(d);
        let scratch = depth_forward
            .get_inplace_scratch_len()
            .max(depth_inverse.get_inplace_scratch_len());
        Self {
            plane: Fft2::new(w, h),
            depth: d,
            depth_forward,
            depth_inverse,
            line: vec![Complex64::default(); d],
            scratch: vec![Complex64::default(); scratch],
        }
    }

    pub(crate) fn transform(&mut self, values: &mut [Complex64], inverse: bool) {
        let plane_len = self.plane.width * self.plane.height;
        for plane in values.chunks_exact_mut(plane_len) {
            self.plane.transform(plane, inverse);
        }
        if self.depth == 1 {
            return;
        }
        let plan = if inverse {
            &self.depth_inverse
        } else {
            &self.depth_forward
        };
        let scale = if inverse {
            1.0 / self.depth as f64
        } else {
            1.0
        };
        for yx in 0..plane_len {
            for (line, plane) in self.line.iter_mut().zip(values.chunks_exact(plane_len)) {
                *line = plane[yx];
            }
            plan.process_with_scratch(&mut self.line, &mut self.scratch);
            for (plane, &value) in values.chunks_exact_mut(plane_len).zip(&self.line) {
                plane[yx] = value * scale;
            }
        }
    }
}
