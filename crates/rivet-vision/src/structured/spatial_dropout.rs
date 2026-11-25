use super::StructuredSample;
use crate::errors::{RivetResult, invalid_argument};
use crate::transforms::{
    V2Config, replace_spatial_pixels, sample_coarse_dropout, sample_grid_dropout,
};
use rivet_data::random::RandomStream;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpatialDropout {
    Coarse {
        holes: (u32, u32),
        hole_height: (u32, u32),
        hole_width: (u32, u32),
    },
    Grid {
        unit_size: usize,
        ratio: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructuredDropoutConfig {
    pub pattern: SpatialDropout,
    pub fill: f32,
    pub fill_mask: Option<i64>,
    pub min_visibility: f64,
}

impl StructuredDropoutConfig {
    pub fn name(&self) -> &'static str {
        match self.pattern {
            SpatialDropout::Coarse { .. } => "CoarseDropout",
            SpatialDropout::Grid { .. } => "GridDropout",
        }
    }
    pub fn validate(&self) -> RivetResult<()> {
        if !(0.0..=1.0).contains(&self.min_visibility) {
            return Err(invalid_argument("dropout min_visibility must be in [0, 1]"));
        }
        let image = match self.pattern {
            SpatialDropout::Coarse {
                holes,
                hole_height,
                hole_width,
            } => V2Config::CoarseDropout {
                holes,
                hole_height,
                hole_width,
                fill: self.fill,
            },
            SpatialDropout::Grid { unit_size, ratio } => V2Config::GridDropout {
                unit_size,
                ratio,
                fill: self.fill,
            },
        };
        image.validate()
    }
}

impl StructuredSample {
    pub fn spatial_dropout(
        self,
        config: &StructuredDropoutConfig,
        rng: &mut RandomStream,
    ) -> RivetResult<Self> {
        config.validate()?;
        let (h, w) = self.size()?;
        match config.pattern {
            SpatialDropout::Coarse {
                holes,
                hole_height,
                hole_width,
            } => {
                let rects = sample_coarse_dropout((h, w), holes, hole_height, hole_width, rng)?;
                if rects.is_empty() {
                    return Ok(self);
                }
                apply(self, config, h, w, |y, x| {
                    rects
                        .iter()
                        .any(|(rows, cols)| rows.contains(&y) && cols.contains(&x))
                })
            }
            SpatialDropout::Grid { unit_size, ratio } => {
                let (rows, cols) = sample_grid_dropout((h, w), unit_size, ratio, rng)?;
                apply(self, config, h, w, |y, x| rows[y] && cols[x])
            }
        }
    }
}

fn apply(
    mut sample: StructuredSample,
    config: &StructuredDropoutConfig,
    h: usize,
    w: usize,
    covers: impl Fn(usize, usize) -> bool,
) -> RivetResult<StructuredSample> {
    sample.image = replace_spatial_pixels(
        &sample.image,
        sample.axis_order,
        h,
        w,
        [config.fill; 2],
        |y, x| covers(y, x).then_some(false),
    )?;
    super::geometry::apply_occlusion_with_visibility(
        sample,
        h,
        w,
        config.fill_mask,
        config.min_visibility,
        covers,
    )
}
