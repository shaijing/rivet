use super::{InterpolationMode, RivetResult, remap::SpatialRemap};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterRefractionConfig {
    pub amplitude_range: (f64, f64),
    pub wavelength_range: (f64, f64),
    pub num_waves_range: (u32, u32),
    pub interpolation: InterpolationMode,
    pub fill: u8,
    pub fill_mask: i64,
}

impl WaterRefractionConfig {
    pub(super) fn valid(&self) -> bool {
        let (a, b) = self.amplitude_range;
        let (lo, hi) = self.wavelength_range;
        let (min, max) = self.num_waves_range;
        a >= 0.0
            && a <= b
            && b <= 1.0
            && lo >= 0.001
            && lo <= hi
            && hi <= 1.0
            && min > 0
            && min <= max
            && max <= 64
            && matches!(
                self.interpolation,
                InterpolationMode::Nearest | InterpolationMode::Bilinear
            )
    }

    pub(crate) fn sample(
        &self,
        h: usize,
        w: usize,
        rng: &mut RandomStream,
    ) -> RivetResult<WaterMap> {
        let (min, max) = self.num_waves_range;
        let count = rng.gen_range_usize(min as usize..max as usize + 1)?;
        let tau = std::f64::consts::TAU;
        let side = h.min(w) as f64;
        let mut draw = |(lo, hi): (f64, f64)| lo + rng.next_f64() * (hi - lo);
        let amplitude = draw(self.amplitude_range) * side / count as f64;
        let mut waves = Vec::new();
        if amplitude == 0.0 {
            return Ok(WaterMap { waves });
        }
        let mut derivative_bound = 0.0;
        for _ in 0..count {
            let (sin, cos) = draw((0.0, tau)).sin_cos();
            let frequency = tau / (side * draw(self.wavelength_range)).max(1.0);
            let wave = Wave {
                phase: draw((0.0, tau)),
                k: [cos * frequency, sin * frequency],
                amplitude: [amplitude * cos, amplitude * sin],
            };
            derivative_bound += wave.amplitude[0].abs().max(wave.amplitude[1].abs())
                * (wave.k[0].abs() + wave.k[1].abs());
            waves.push(wave);
        }
        // A contraction of the displacement field makes the inverse unique.
        if derivative_bound > 0.65 {
            for wave in &mut waves {
                wave.amplitude = wave.amplitude.map(|v| v * 0.65 / derivative_bound);
            }
        }
        Ok(WaterMap { waves })
    }
}

struct Wave {
    phase: f64,
    k: [f64; 2],
    amplitude: [f64; 2],
}

pub(crate) struct WaterMap {
    waves: Vec<Wave>,
}

impl WaterMap {
    fn displacement(&self, [x, y]: [f64; 2]) -> [f64; 2] {
        let mut d = [0.0; 2];
        for wave in &self.waves {
            let s = (wave.k[0] * x + wave.k[1] * y + wave.phase).sin();
            d[0] += s * wave.amplitude[0];
            d[1] += s * wave.amplitude[1];
        }
        d
    }
}

impl SpatialRemap for WaterMap {
    fn source(&self, p: [f64; 2]) -> [f64; 2] {
        let [dx, dy] = self.displacement(p);
        [p[0] + dx, p[1] + dy]
    }

    fn destination(&self, p: [f64; 2]) -> [f64; 2] {
        let mut output = p;
        for _ in 0..64 {
            let [dx, dy] = self.displacement(output);
            let next = [p[0] - dx, p[1] - dy];
            let err = (next[0] - output[0]).abs().max((next[1] - output[1]).abs());
            output = next;
            if err < 1e-10 {
                break;
            }
        }
        output
    }

    fn is_identity(&self) -> bool {
        self.waves.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_roundtrips_even_for_high_frequency_waves() {
        let config = WaterRefractionConfig {
            amplitude_range: (1.0, 1.0),
            wavelength_range: (0.001, 0.01),
            num_waves_range: (5, 5),
            interpolation: InterpolationMode::Bilinear,
            fill: 0,
            fill_mask: 0,
        };
        let map = config
            .sample(17, 23, &mut RandomStream::from_seed(51))
            .unwrap();
        for point in [[0., 0.], [22., 16.], [-2., 4.], [12.5, 8.5]] {
            let inverse = map.destination(map.source(point));
            assert!((inverse[0] - point[0]).abs() < 1e-8);
            assert!((inverse[1] - point[1]).abs() < 1e-8);
        }
    }
}
