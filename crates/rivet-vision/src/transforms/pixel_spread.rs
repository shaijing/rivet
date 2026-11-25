use super::{RivetResult, remap::SpatialRemap};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelSpreadConfig {
    pub radius: u32,
    pub padding_mode: super::PaddingMode,
    pub fill: u8,
    pub fill_mask: i64,
}

impl PixelSpreadConfig {
    pub(super) fn valid(&self) -> bool {
        self.radius <= 1024
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<PixelSpreadMap> {
        let radius = self.radius as usize;
        let mut points = Vec::new();
        if radius > 0 {
            points.reserve(h * w);
            let border = |i: i64, len: usize| {
                use super::PaddingMode::*;
                match self.padding_mode {
                    Constant => i,
                    Edge => i.clamp(0, len as i64 - 1),
                    Reflect => super::blur::reflect101(i, len) as i64,
                    Symmetric => {
                        let n = len as i64;
                        let v = i.rem_euclid(2 * n);
                        v.min(2 * n - 1 - v)
                    }
                }
            };
            for y in 0..h {
                for x in 0..w {
                    let dx = rng.gen_range_usize(0..2 * radius + 1)? as i64 - radius as i64;
                    let dy = rng.gen_range_usize(0..2 * radius + 1)? as i64 - radius as i64;
                    points.push([border(x as i64 + dx, w), border(y as i64 + dy, h)]);
                }
            }
        }
        Ok(PixelSpreadMap {
            width: w,
            height: h,
            radius,
            points,
        })
    }
}

pub(crate) struct PixelSpreadMap {
    width: usize,
    height: usize,
    radius: usize,
    points: Vec<[i64; 2]>,
}

impl SpatialRemap for PixelSpreadMap {
    fn source(&self, p: [f64; 2]) -> [f64; 2] {
        if self.is_identity() {
            return p;
        }
        self.points[p[1] as usize * self.width + p[0] as usize].map(|v| v as f64)
    }
    fn destination(&self, p: [f64; 2]) -> [f64; 2] {
        if self.is_identity() {
            return p;
        }
        let q = p.map(|v| (v + 0.5).floor() as i64);
        let [x, y] = q;
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return p;
        }
        let (x, y) = (x as usize, y as usize);
        let mut best = None;
        let mut distance = usize::MAX;
        for yy in y.saturating_sub(self.radius)..=(y + self.radius).min(self.height - 1) {
            for xx in x.saturating_sub(self.radius)..=(x + self.radius).min(self.width - 1) {
                if self.points[yy * self.width + xx] == q {
                    let delta = xx.abs_diff(x).pow(2) + yy.abs_diff(y).pow(2);
                    if delta < distance {
                        distance = delta;
                        best = Some([xx as f64 + p[0] - x as f64, yy as f64 + p[1] - y as f64]);
                    }
                }
            }
        }
        best.unwrap_or(p)
    }
    fn is_identity(&self) -> bool {
        self.points.is_empty()
    }
}
