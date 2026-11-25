use crate::error::to_py_err;
use crate::loader::PyDataLoader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rivet_vision::api::{
    DType, ElasticTransformConfig, ImagePipeline, InterpolationMode, LetterBoxConfig,
    PerspectiveConfig, Point2, RandomAffineConfig, RandomPerspectiveConfig, RotationAngle,
    TransformSequence, V2Config,
};

fn parse_pair<T: Clone + std::fmt::Debug>(
    values: Option<Vec<T>>,
    name: &str,
    default: [T; 2],
) -> PyResult<[T; 2]> {
    let values = values.unwrap_or_else(|| default.to_vec());
    values.try_into().map_err(|values: Vec<T>| {
        PyValueError::new_err(format!(
            "{name} must contain exactly two values, got {values:?}"
        ))
    })
}

fn parse_f32_scalar_or_pair(value: &Bound<'_, PyAny>, name: &str) -> PyResult<[f32; 2]> {
    if let Ok(value) = value.extract::<f32>() {
        return Ok([value.clone(), value]);
    }
    let values = value.extract::<Vec<f32>>()?;
    values.try_into().map_err(|values: Vec<f32>| {
        PyValueError::new_err(format!(
            "{name} must be a scalar or contain exactly two values, got {values:?}"
        ))
    })
}

fn parse_f32_sigma_axes(value: &Bound<'_, PyAny>) -> PyResult<[f32; 2]> {
    if let Ok(value) = value.extract::<f32>() {
        return Ok([value; 2]);
    }
    let values = value.extract::<Vec<f32>>()?;
    match values.as_slice() {
        [sigma] => Ok([*sigma; 2]),
        [horizontal, vertical] => Ok([*horizontal, *vertical]),
        _ => Err(PyValueError::new_err(format!(
            "sigma must be a scalar or contain one or two values, got {values:?}"
        ))),
    }
}

fn parse_u32_scalar_or_pair(value: &Bound<'_, PyAny>, name: &str) -> PyResult<[u32; 2]> {
    if let Ok(value) = value.extract::<u32>() {
        return Ok([value; 2]);
    }
    let values = value.extract::<Vec<u32>>()?;
    values.try_into().map_err(|values: Vec<u32>| {
        PyValueError::new_err(format!(
            "{name} must be a scalar or contain exactly two values, got {values:?}"
        ))
    })
}

fn parse_optional_kernel(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<[u32; 2]>> {
    value
        .map(|value| parse_u32_scalar_or_pair(value, "kernel_size"))
        .transpose()
}

fn parse_fill_u8(value: Option<&Bound<'_, PyAny>>) -> PyResult<(u8, Option<[u8; 3]>)> {
    let Some(value) = value else {
        return Ok((0, None));
    };
    if let Ok(value) = value.extract::<u8>() {
        return Ok((value, None));
    }
    let values = value.extract::<Vec<u8>>()?;
    match values.as_slice() {
        [value] => Ok((*value, None)),
        [red, green, blue] => Ok((0, Some([*red, *green, *blue]))),
        _ => Err(PyValueError::new_err(
            "fill must be a scalar or a 3-value channel sequence",
        )),
    }
}

fn parse_fill_f32(value: Option<&Bound<'_, PyAny>>) -> PyResult<(f32, Option<[f32; 3]>)> {
    let Some(value) = value else {
        return Ok((0.0, None));
    };
    if let Ok(value) = value.extract::<f32>() {
        return Ok((value, None));
    }
    let values = value.extract::<Vec<f32>>()?;
    match values.as_slice() {
        [value] => Ok((*value, None)),
        [red, green, blue] => Ok((0.0, Some([*red, *green, *blue]))),
        _ => Err(PyValueError::new_err(
            "fill must be a scalar or a 3-value channel sequence",
        )),
    }
}

fn parse_points(values: Vec<Vec<f32>>, name: &str) -> PyResult<[Point2; 4]> {
    if values.len() != 4 || values.iter().any(|point| point.len() != 2) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain four (x, y) points"
        )));
    }
    Ok([
        Point2::new(values[0][0], values[0][1]),
        Point2::new(values[1][0], values[1][1]),
        Point2::new(values[2][0], values[2][1]),
        Point2::new(values[3][0], values[3][1]),
    ])
}

pub(crate) fn parse_interpolation(value: &str) -> PyResult<InterpolationMode> {
    value.parse::<InterpolationMode>().map_err(to_py_err)
}

fn parse_padding_mode(value: &str) -> PyResult<rivet_vision::api::PaddingMode> {
    use rivet_vision::api::PaddingMode;
    match value {
        "constant" => Ok(PaddingMode::Constant),
        "edge" => Ok(PaddingMode::Edge),
        "reflect" => Ok(PaddingMode::Reflect),
        "symmetric" => Ok(PaddingMode::Symmetric),
        _ => Err(PyValueError::new_err(
            "padding_mode must be constant, edge, reflect or symmetric",
        )),
    }
}

fn parse_crop_padding(value: Option<&Bound<'_, PyAny>>) -> PyResult<(u32, Option<[u32; 4]>)> {
    let Some(value) = value else {
        return Ok((0, None));
    };
    if let Ok(padding) = value.extract::<u32>() {
        return Ok((padding, None));
    }
    let values = value.extract::<Vec<u32>>()?;
    match values.as_slice() {
        [all] => Ok((*all, None)),
        [horizontal, vertical] => Ok((0, Some([*horizontal, *vertical, *horizontal, *vertical]))),
        [left, top, right, bottom] => Ok((0, Some([*left, *top, *right, *bottom]))),
        _ => Err(PyValueError::new_err(
            "padding must be an integer or a sequence of 1, 2, or 4 integers",
        )),
    }
}

fn parse_strip_length(
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<rivet_vision::transforms::StripLength> {
    use rivet_vision::transforms::StripLength;
    let Some(value) = value else {
        return Ok(StripLength::Pixels(0, 0));
    };
    if let Ok((lo, hi)) = value.extract::<(u32, u32)>() {
        return Ok(StripLength::Pixels(lo, hi));
    }
    let (lo, hi) = value.extract::<(f64, f64)>()?;
    Ok(StripLength::Fraction(lo, hi))
}

fn parse_crop_fill(value: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f32>> {
    let Some(value) = value else {
        return Ok(vec![0.0]);
    };
    if let Ok(fill) = value.extract::<f32>() {
        return Ok(vec![fill]);
    }
    value.extract::<Vec<f32>>()
}

#[pyclass(name = "_Transform")]
pub(crate) struct PyTransform {
    pub(crate) inner: TransformSequence,
}

#[pymethods]
impl PyTransform {
    #[new]
    fn new() -> Self {
        Self {
            inner: TransformSequence::new(),
        }
    }

    fn compose(&self, other: PyRef<'_, Self>) -> Self {
        Self {
            inner: self.inner.clone().compose(other.inner.clone()),
        }
    }

    fn decode_image(&self) -> Self {
        Self {
            inner: self.inner.clone().decode_image(),
        }
    }

    #[pyo3(signature = (width, height=None, interpolation="bilinear", max_size=None))]
    fn resize(
        &self,
        width: u32,
        height: Option<u32>,
        interpolation: &str,
        max_size: Option<u32>,
    ) -> PyResult<Self> {
        let interpolation = interpolation
            .parse::<InterpolationMode>()
            .map_err(to_py_err)?;
        let inner = if let Some(height) = height {
            self.inner
                .clone()
                .resize_with_interpolation(width, height, interpolation)
        } else {
            self.inner
                .clone()
                .resize_short_edge_with_interpolation(width, max_size, interpolation)
        };
        Ok(Self { inner })
    }

    fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().crop(x, y, width, height),
        }
    }

    fn five_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().five_crop(width, height),
        }
    }

    #[pyo3(signature = (width, height, vertical_flip=false))]
    fn ten_crop(&self, width: u32, height: u32, vertical_flip: bool) -> Self {
        Self {
            inner: self.inner.clone().ten_crop(width, height, vertical_flip),
        }
    }

    #[pyo3(signature = (range, y=None, interpolation="bilinear"))]
    fn random_scale(
        &self,
        range: (f64, f64),
        y: Option<(f64, f64)>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::RandomScale {
            range,
            y,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (max_size=None, *, max_size_hw=None, interpolation="bilinear"))]
    fn longest_max_size(
        &self,
        max_size: Option<Vec<u32>>,
        max_size_hw: Option<(Option<u32>, Option<u32>)>,
        interpolation: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{LongestMaxSizeConfig, LongestSizeLimit};
        let limit = match (max_size, max_size_hw) {
            (Some(sizes), None) => LongestSizeLimit::Longest(sizes),
            (None, Some((height, width))) => LongestSizeLimit::Axes { height, width },
            _ => {
                return Err(PyValueError::new_err(
                    "set exactly one of max_size or max_size_hw",
                ));
            }
        };
        let config = V2Config::LongestMaxSize(LongestMaxSizeConfig {
            limit,
            interpolation: parse_interpolation(interpolation)?,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (min_height=None, min_width=None, *, pad_height_divisor=None, pad_width_divisor=None, position="center", padding_mode="constant", fill=None, fill_mask=0))]
    fn pad_if_needed(
        &self,
        min_height: Option<u32>,
        min_width: Option<u32>,
        pad_height_divisor: Option<u32>,
        pad_width_divisor: Option<u32>,
        position: &str,
        padding_mode: &str,
        fill: Option<&Bound<'_, PyAny>>,
        fill_mask: i64,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{PadIfNeededConfig, PadPosition};
        let position = match position {
            "center" => PadPosition::Center,
            "top_left" => PadPosition::TopLeft,
            "top_right" => PadPosition::TopRight,
            "bottom_left" => PadPosition::BottomLeft,
            "bottom_right" => PadPosition::BottomRight,
            "random" => PadPosition::Random,
            _ => return Err(PyValueError::new_err("unknown padding position")),
        };
        let (fill, fill_channels) = parse_fill_f32(fill)?;
        let config = V2Config::PadIfNeeded(std::sync::Arc::new(PadIfNeededConfig {
            min_height,
            min_width,
            pad_height_divisor,
            pad_width_divisor,
            position,
            padding_mode: parse_padding_mode(padding_mode)?,
            fill,
            fill_channels,
            fill_mask,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn center_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().center_crop(width, height),
        }
    }

    #[pyo3(signature = (padding, fill=None, padding_mode="constant"))]
    fn pad(
        &self,
        padding: u32,
        fill: Option<&Bound<'_, PyAny>>,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let (fill, channels) = parse_fill_f32(fill)?;
        let mut config = rivet_vision::api::PadConfig::new(padding).with_fill(fill);
        if let Some(channels) = channels {
            config = config.with_fill_channels(channels);
        }
        let config = config.with_mode(parse_padding_mode(padding_mode)?);
        Ok(Self {
            inner: self.inner.clone().pad_with_config(config),
        })
    }

    #[pyo3(signature = (left, top, right, bottom, fill=None, padding_mode="constant"))]
    fn pad_with_sides(
        &self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        fill: Option<&Bound<'_, PyAny>>,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let (fill, channels) = parse_fill_f32(fill)?;
        let mut config =
            rivet_vision::api::PadConfig::with_sides(left, top, right, bottom).with_fill(fill);
        if let Some(channels) = channels {
            config = config.with_fill_channels(channels);
        }
        let config = config.with_mode(parse_padding_mode(padding_mode)?);
        Ok(Self {
            inner: self.inner.clone().pad_with_config(config),
        })
    }

    #[pyo3(signature=(policy="imagenet",interpolation="nearest",fill=0))]
    fn auto_augment(&self, policy: &str, interpolation: &str, fill: u8) -> PyResult<Self> {
        let mut config =
            rivet_vision::api::AutoAugmentConfig::new(policy.parse().map_err(to_py_err)?);
        config.interpolation = parse_interpolation(interpolation)?;
        config.fill = fill;
        Ok(Self {
            inner: self.inner.clone().auto_augment(config),
        })
    }
    #[pyo3(signature=(num_ops=2,magnitude=9,num_magnitude_bins=31,interpolation="nearest",fill=0))]
    fn rand_augment(
        &self,
        num_ops: usize,
        magnitude: usize,
        num_magnitude_bins: usize,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::RandAugmentConfig {
            num_ops,
            magnitude,
            num_magnitude_bins,
            interpolation: parse_interpolation(interpolation)?,
            fill,
        };
        Ok(Self {
            inner: self.inner.clone().rand_augment(config),
        })
    }
    #[pyo3(signature=(num_magnitude_bins=31,interpolation="nearest",fill=0))]
    fn trivial_augment_wide(
        &self,
        num_magnitude_bins: usize,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::TrivialAugmentWideConfig {
            num_magnitude_bins,
            interpolation: parse_interpolation(interpolation)?,
            fill,
        };
        Ok(Self {
            inner: self.inner.clone().trivial_augment_wide(config),
        })
    }
    #[pyo3(signature=(severity=3,mixture_width=3,chain_depth=-1,alpha=1.0,all_ops=true,interpolation="bilinear",fill=0))]
    fn augmix(
        &self,
        severity: usize,
        mixture_width: usize,
        chain_depth: i32,
        alpha: f32,
        all_ops: bool,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::AugMixConfig {
            severity,
            mixture_width,
            chain_depth,
            alpha,
            all_ops,
            interpolation: parse_interpolation(interpolation)?,
            fill,
        };
        Ok(Self {
            inner: self.inner.clone().augmix(config),
        })
    }

    fn to_tensor(&self) -> Self {
        Self {
            inner: self.inner.clone().to_tensor(),
        }
    }
    fn pil_to_tensor(&self) -> Self {
        Self {
            inner: self.inner.clone().pil_to_tensor(),
        }
    }
    fn linear_transform(&self, matrix: Vec<Vec<f32>>, mean: Vec<f32>) -> PyResult<Self> {
        let n = matrix.len();
        if matrix.iter().any(|row| row.len() != n) {
            return Err(PyValueError::new_err(
                "linear transformation matrix must be square",
            ));
        }
        let matrix = rivet_core::Tensor::from_vec(
            matrix.into_iter().flatten().collect::<Vec<_>>(),
            [n, n],
            &rivet_core::Device::Cpu,
        )
        .map_err(to_py_err)?;
        let n_mean = mean.len();
        let mean = rivet_core::Tensor::from_vec(mean, [n_mean], &rivet_core::Device::Cpu)
            .map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().linear_transform(
                rivet_vision::api::LinearTransformationConfig::new(matrix, mean),
            ),
        })
    }

    fn saturation(&self, factor: f32) -> Self {
        Self {
            inner: self.inner.clone().saturation(factor),
        }
    }
    #[pyo3(signature=(gamma,gain=1.0))]
    fn gamma(&self, gamma: f64, gain: f64) -> Self {
        Self {
            inner: self.inner.clone().gamma(gamma, gain),
        }
    }
    #[pyo3(signature=(angle,translate=None,scale=1.0,shear=None,interpolation="nearest",fill=None,center=None))]
    fn affine(
        &self,
        angle: f64,
        translate: Option<Vec<f64>>,
        scale: f64,
        shear: Option<Vec<f64>>,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
        center: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let mut config = rivet_vision::api::AffineConfig::new(
            angle,
            parse_pair(translate, "translate", [0.0, 0.0])?,
            scale,
            parse_pair(shear, "shear", [0.0, 0.0])?,
        )
        .with_options(parse_interpolation(interpolation)?, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        if let Some(center) = center {
            config = config.with_center(parse_pair(Some(center), "center", [0.0, 0.0])?);
        }
        Ok(Self {
            inner: self.inner.clone().affine(config),
        })
    }
    #[pyo3(signature=(x,y,width,height,value=None))]
    fn erase(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        value: Option<Vec<f32>>,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .erase(rivet_vision::api::EraseConfig::new(
                    x,
                    y,
                    width,
                    height,
                    value.unwrap_or_else(|| vec![0.0]),
                )),
        }
    }
    #[pyo3(signature=(degrees,interpolation="nearest",expand=false,fill=None,center=None))]
    fn random_rotation(
        &self,
        degrees: Vec<f32>,
        interpolation: &str,
        expand: bool,
        fill: Option<&Bound<'_, PyAny>>,
        center: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let mut config = rivet_vision::api::RandomRotationConfig::new(parse_pair(
            Some(degrees),
            "degrees",
            [0.0, 0.0],
        )?)
        .with_options(parse_interpolation(interpolation)?, expand, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        let config = if let Some(center) = center {
            config.with_center(parse_pair(Some(center), "center", [0.0, 0.0])?)
        } else {
            config
        };
        Ok(Self {
            inner: self.inner.clone().random_rotation(config),
        })
    }
    #[pyo3(signature=(kernel_size,sigma=None))]
    fn random_gaussian_blur(
        &self,
        kernel_size: &Bound<'_, PyAny>,
        sigma: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dimensions = parse_u32_scalar_or_pair(kernel_size, "kernel_size")?;
        let sigma = match sigma {
            Some(value) => parse_f32_scalar_or_pair(value, "sigma")?,
            None => [0.1, 2.0],
        };
        let config = rivet_vision::api::RandomGaussianBlurConfig::new(dimensions[0], sigma)
            .with_kernel_dimensions(dimensions);
        Ok(Self {
            inner: self.inner.clone().random_gaussian_blur(config),
        })
    }
    #[pyo3(signature=(x,y,width,height,out_width,out_height,interpolation="bilinear"))]
    fn resized_crop(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        out_width: u32,
        out_height: u32,
        interpolation: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().resized_crop(
                x,
                y,
                width,
                height,
                out_width,
                out_height,
                parse_interpolation(interpolation)?,
            ),
        })
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_vertical_flip(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_vertical_flip(probability),
        }
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_invert(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_invert(probability),
        }
    }
    #[pyo3(signature=(bits,probability=0.5))]
    fn random_posterize(&self, bits: u8, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_posterize(bits, probability),
        }
    }
    #[pyo3(signature=(threshold,probability=0.5))]
    fn random_solarize(&self, threshold: u8, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_solarize(threshold, probability),
        }
    }
    #[pyo3(signature=(factor,probability=0.5))]
    fn random_adjust_sharpness(&self, factor: f32, probability: f64) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .random_adjust_sharpness(factor, probability),
        }
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_autocontrast(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_autocontrast(probability),
        }
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_equalize(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_equalize(probability),
        }
    }

    fn horizontal_flip(&self) -> Self {
        Self {
            inner: self.inner.clone().horizontal_flip(),
        }
    }

    fn vertical_flip(&self) -> Self {
        Self {
            inner: self.inner.clone().vertical_flip(),
        }
    }

    #[pyo3(signature = (width, height, padding=None, pad_if_needed=false, fill=None, padding_mode="constant"))]
    fn random_crop(
        &self,
        width: u32,
        height: u32,
        padding: Option<&Bound<'_, PyAny>>,
        pad_if_needed: bool,
        fill: Option<&Bound<'_, PyAny>>,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let (uniform, sides) = parse_crop_padding(padding)?;
        let config = rivet_vision::api::RandomCropConfig::new(width, height, uniform).with_options(
            sides,
            pad_if_needed,
            parse_crop_fill(fill)?,
            parse_padding_mode(padding_mode)?,
        );
        Ok(Self {
            inner: self.inner.clone().random_crop_with_config(config),
        })
    }

    fn random_resized_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().random_resized_crop(width, height),
        }
    }

    #[pyo3(signature = (width, height, scale=None, ratio=None, interpolation="bilinear"))]
    fn random_resized_crop_with_options(
        &self,
        width: u32,
        height: u32,
        scale: Option<Vec<f32>>,
        ratio: Option<Vec<f32>>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let scale = parse_pair(scale, "scale", [0.08, 1.0])?;
        let ratio = parse_pair(ratio, "ratio", [0.75, 1.3333334])?;
        let interpolation = parse_interpolation(interpolation)?;
        let config = rivet_vision::api::RandomResizedCropConfig::new(width, height)
            .with_params(scale[0], scale[1], ratio[0], ratio[1])
            .with_interpolation(interpolation);
        Ok(Self {
            inner: self.inner.clone().random_resized_crop_with_config(config),
        })
    }

    #[pyo3(signature = (probability=0.5))]
    fn random_horizontal_flip(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_horizontal_flip(probability),
        }
    }

    fn brightness(&self, value: f32) -> Self {
        Self {
            inner: self.inner.clone().brightness(value),
        }
    }

    fn contrast(&self, value: f32) -> Self {
        Self {
            inner: self.inner.clone().contrast(value),
        }
    }

    #[pyo3(signature = (brightness=0.0, contrast=0.0, hue=0.0, saturation=0.0))]
    fn color_jitter(&self, brightness: f32, contrast: f32, hue: f32, saturation: f32) -> Self {
        Self {
            inner: self.inner.clone().color_jitter_with_config(
                rivet_vision::api::ColorJitterConfig::new(brightness, contrast, hue)
                    .with_saturation(saturation),
            ),
        }
    }

    fn invert(&self) -> Self {
        Self {
            inner: self.inner.clone().invert(),
        }
    }

    fn posterize(&self, bits: u8) -> Self {
        Self {
            inner: self.inner.clone().posterize(bits),
        }
    }

    fn solarize(&self, threshold: u8) -> Self {
        Self {
            inner: self.inner.clone().solarize(threshold),
        }
    }

    fn autocontrast(&self) -> Self {
        Self {
            inner: self.inner.clone().autocontrast(),
        }
    }

    fn equalize(&self) -> Self {
        Self {
            inner: self.inner.clone().equalize(),
        }
    }

    fn sharpness(&self, amount: f32) -> Self {
        Self {
            inner: self.inner.clone().sharpness(amount),
        }
    }

    #[pyo3(signature = (sigma, kernel_size=None))]
    fn gaussian_blur(
        &self,
        sigma: &Bound<'_, PyAny>,
        kernel_size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let sigmas = parse_f32_sigma_axes(sigma)?;
        let mut config =
            rivet_vision::api::GaussianBlurConfig::new(sigmas[0]).with_axis_sigmas(sigmas);
        if let Some(dimensions) = parse_optional_kernel(kernel_size)? {
            config = config.with_kernel_dimensions(dimensions);
        }
        Ok(Self {
            inner: self.inner.clone().gaussian_blur_with_config(config),
        })
    }

    fn hue(&self, factor: f32) -> Self {
        Self {
            inner: self.inner.clone().hue(factor),
        }
    }

    #[pyo3(signature = (num_output_channels=1))]
    fn grayscale(&self, num_output_channels: u8) -> Self {
        Self {
            inner: self.inner.clone().grayscale(num_output_channels),
        }
    }

    #[pyo3(signature = (probability=0.1, num_output_channels=3))]
    fn random_grayscale(&self, probability: f64, num_output_channels: u8) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .random_grayscale(probability, num_output_channels),
        }
    }

    #[pyo3(signature = (probability=0.5))]
    fn random_erasing(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_erasing(probability),
        }
    }

    #[pyo3(signature = (probability=0.5, scale=None, ratio=None, value=None))]
    fn random_erasing_with_options(
        &self,
        probability: f64,
        scale: Option<Vec<f32>>,
        ratio: Option<Vec<f32>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let scale = parse_pair(scale, "scale", [0.02, 0.33])?;
        let ratio = parse_pair(ratio, "ratio", [0.3, 3.3333333])?;
        let config = rivet_vision::api::RandomErasingConfig::new(probability)
            .with_params(scale[0], scale[1], ratio[0], ratio[1]);
        let config = if let Some(value) = value {
            if let Ok(value) = value.extract::<String>() {
                if value != "random" {
                    return Err(PyValueError::new_err("value string must be 'random'"));
                }
                config.with_random_value()
            } else if let Ok(value) = value.extract::<f32>() {
                config.with_value(value)
            } else {
                config.with_values(value.extract::<Vec<f32>>()?)
            }
        } else {
            config
        };
        Ok(Self {
            inner: self.inner.clone().random_erasing_with_config(config),
        })
    }

    fn permute_channels(&self, permutation: Vec<usize>) -> PyResult<Self> {
        let config = V2Config::PermuteChannels(permutation);
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn random_channel_permutation(&self) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .v2(V2Config::RandomChannelPermutation { probability: 1.0 }),
        }
    }

    #[pyo3(signature = (brightness, contrast, saturation, hue, probability=0.5))]
    fn random_photometric_distort(
        &self,
        brightness: Vec<f32>,
        contrast: Vec<f32>,
        saturation: Vec<f32>,
        hue: Vec<f32>,
        probability: f64,
    ) -> PyResult<Self> {
        let brightness = parse_pair(Some(brightness), "brightness", [0.875, 1.125])?;
        let contrast = parse_pair(Some(contrast), "contrast", [0.5, 1.5])?;
        let saturation = parse_pair(Some(saturation), "saturation", [0.5, 1.5])?;
        let hue = parse_pair(Some(hue), "hue", [-0.05, 0.05])?;
        let config = V2Config::RandomPhotometricDistort {
            brightness: (brightness[0], brightness[1]),
            contrast: (contrast[0], contrast[1]),
            saturation: (saturation[0], saturation[1]),
            hue: (hue[0], hue[1]),
            probability,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (min_size, max_size, interpolation="bilinear"))]
    fn random_resize(&self, min_size: u32, max_size: u32, interpolation: &str) -> PyResult<Self> {
        let config = V2Config::RandomResize {
            min_size,
            max_size,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (min_sizes, max_size=None, interpolation="bilinear"))]
    fn random_shortest_size(
        &self,
        min_sizes: Vec<u32>,
        max_size: Option<u32>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::RandomShortestSize {
            min_sizes,
            max_size,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (width, height, scale_range, interpolation="bilinear"))]
    fn scale_jitter(
        &self,
        width: u32,
        height: u32,
        scale_range: Vec<f64>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let range = parse_pair(Some(scale_range), "scale_range", [0.1, 2.0])?;
        let config = V2Config::ScaleJitter {
            target_width: width,
            target_height: height,
            scale_range: (range[0], range[1]),
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (fill, side_range, probability=0.5))]
    fn random_zoom_out(
        &self,
        fill: &Bound<'_, PyAny>,
        side_range: Vec<f64>,
        probability: f64,
    ) -> PyResult<Self> {
        let fill = if let Ok(value) = fill.extract::<f32>() {
            vec![value]
        } else {
            fill.extract::<Vec<f32>>()?
        };
        let range = parse_pair(Some(side_range), "side_range", [1.0, 4.0])?;
        let config = V2Config::RandomZoomOut {
            fill,
            side_range: (range[0], range[1]),
            probability,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (mean=0.0, sigma=0.1, clip=true))]
    fn gaussian_noise(&self, mean: f64, sigma: f64, clip: bool) -> PyResult<Self> {
        let config = V2Config::GaussianNoise { mean, sigma, clip };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (width, height, *, interpolation="bilinear", fill=0.0, mask_fill=0))]
    fn letterbox(
        &self,
        width: u32,
        height: u32,
        interpolation: &str,
        fill: f32,
        mask_fill: i64,
    ) -> PyResult<Self> {
        let config = LetterBoxConfig {
            width,
            height,
            interpolation: parse_interpolation(interpolation)?,
            fill,
            mask_fill,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().letterbox(config),
        })
    }

    #[pyo3(signature = (noise_type="uniform", spatial_mode="constant", *, ranges=None, mean_range=(0.0, 0.0), scale_range=(0.1, 0.1), alpha_range=(0.5, 1.5), beta_range=(0.5, 1.5), patch_count_range=(1, 1), patch_height_range=(0.1, 1.0), patch_width_range=(0.1, 1.0), per_channel=false))]
    #[allow(clippy::too_many_arguments)]
    fn additive_noise(
        &self,
        noise_type: &str,
        spatial_mode: &str,
        ranges: Option<Vec<(f64, f64)>>,
        mean_range: (f64, f64),
        scale_range: (f64, f64),
        alpha_range: (f64, f64),
        beta_range: (f64, f64),
        patch_count_range: (u32, u32),
        patch_height_range: (f64, f64),
        patch_width_range: (f64, f64),
        per_channel: bool,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{AdditiveDistribution, AdditiveNoiseConfig, NoiseSharing};
        let distribution = match noise_type {
            "uniform" => AdditiveDistribution::Uniform(ranges.unwrap_or_else(|| vec![(-0.1, 0.1)])),
            "gaussian" => AdditiveDistribution::Gaussian {
                mean: mean_range,
                std: scale_range,
            },
            "laplace" => AdditiveDistribution::Laplace {
                mean: mean_range,
                scale: scale_range,
            },
            "beta" => AdditiveDistribution::Beta {
                alpha: alpha_range,
                beta: beta_range,
                scale: scale_range,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "noise_type must be uniform, gaussian, laplace or beta",
                ));
            }
        };
        let sharing = match spatial_mode {
            "constant" => NoiseSharing::Constant,
            "per_pixel" => NoiseSharing::PerPixel,
            "shared" => NoiseSharing::Shared,
            "patch" => NoiseSharing::Patch {
                count: patch_count_range,
                height: patch_height_range,
                width: patch_width_range,
                per_channel,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "spatial_mode must be constant, per_pixel, shared or patch",
                ));
            }
        };
        let config = V2Config::AdditiveNoise(std::sync::Arc::new(AdditiveNoiseConfig {
            distribution,
            sharing,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (multiplier=(0.9, 1.1), *, per_channel=false, elementwise=false))]
    fn multiplicative_noise(
        &self,
        multiplier: (f64, f64),
        per_channel: bool,
        elementwise: bool,
    ) -> PyResult<Self> {
        let config = V2Config::MultiplicativeNoise {
            multiplier,
            per_channel,
            elementwise,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (flare_roi=(0.0, 0.0, 1.0, 0.5), angle_range=(0.0, 1.0), num_flare_circles_range=(6, 10), src_radius=400, src_color=(255, 255, 255), physics_based=false))]
    fn random_sun_flare(
        &self,
        flare_roi: (f64, f64, f64, f64),
        angle_range: (f64, f64),
        num_flare_circles_range: (u32, u32),
        src_radius: u32,
        src_color: (u8, u8, u8),
        physics_based: bool,
    ) -> PyResult<Self> {
        let config = V2Config::RandomSunFlare(std::sync::Arc::new(
            rivet_vision::transforms::RandomSunFlareConfig {
                flare_roi,
                angle_range,
                num_flare_circles_range,
                src_radius,
                src_color,
                physics_based,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (flare_roi=(0.0, 0.0, 1.0, 0.5), num_ghosts_range=(3, 7), intensity_range=(0.3, 0.7), num_rays_range=(4, 8), bloom_range=(0.01, 0.05)))]
    fn lens_flare(
        &self,
        flare_roi: (f64, f64, f64, f64),
        num_ghosts_range: (u32, u32),
        intensity_range: (f64, f64),
        num_rays_range: (u32, u32),
        bloom_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::LensFlare(std::sync::Arc::new(
            rivet_vision::transforms::LensFlareConfig {
                flare_roi,
                num_ghosts_range,
                intensity_range,
                num_rays_range,
                bloom_range,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (brightness_range=(-0.2, 0.2), contrast_range=(-0.2, 0.2), *, brightness_by_max=false, ensure_safe_output=false))]
    fn random_brightness_contrast(
        &self,
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        brightness_by_max: bool,
        ensure_safe_output: bool,
    ) -> PyResult<Self> {
        let config = V2Config::RandomBrightnessContrast {
            brightness_range,
            contrast_range,
            brightness_by_max,
            ensure_safe_output,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (hue_shift_range=(-20.0, 20.0), sat_shift_range=(-30.0, 30.0), val_shift_range=(-20.0, 20.0)))]
    fn hue_saturation_value(
        &self,
        hue_shift_range: (f64, f64),
        sat_shift_range: (f64, f64),
        val_shift_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::HueSaturationValue(std::sync::Arc::new([
            hue_shift_range,
            sat_shift_range,
            val_shift_range,
        ]));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (dot_size_range=(4,10), blend_range=(0.0,0.5)))]
    fn halftone(&self, dot_size_range: (u32, u32), blend_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::Halftone {
            dot_size_range,
            blend_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (method="error_diffusion", n_colors=2, color_mode="grayscale", error_diffusion_algorithm="floyd_steinberg", bayer_matrix_size=4, serpentine=false, noise_range=(-0.5,0.5)))]
    fn dithering(
        &self,
        method: &str,
        n_colors: u16,
        color_mode: &str,
        error_diffusion_algorithm: &str,
        bayer_matrix_size: u8,
        serpentine: bool,
        noise_range: (f64, f64),
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{DiffusionAlgorithm, DitherMethod, DitheringConfig};
        let grayscale = match color_mode {
            "grayscale" => true,
            "per_channel" => false,
            _ => {
                return Err(PyValueError::new_err(
                    "color_mode must be grayscale or per_channel",
                ));
            }
        };
        let method = match method {
            "random" => DitherMethod::Random { noise_range },
            "ordered" => DitherMethod::Ordered {
                size: bayer_matrix_size,
            },
            "error_diffusion" => {
                let algorithm = match error_diffusion_algorithm {
                    "floyd_steinberg" => DiffusionAlgorithm::FloydSteinberg,
                    "jarvis" => DiffusionAlgorithm::Jarvis,
                    "stucki" => DiffusionAlgorithm::Stucki,
                    "atkinson" => DiffusionAlgorithm::Atkinson,
                    "burkes" => DiffusionAlgorithm::Burkes,
                    "sierra" => DiffusionAlgorithm::Sierra,
                    "sierra_2row" => DiffusionAlgorithm::Sierra2Row,
                    "sierra_lite" => DiffusionAlgorithm::SierraLite,
                    _ => return Err(PyValueError::new_err("unknown error diffusion algorithm")),
                };
                DitherMethod::ErrorDiffusion {
                    algorithm,
                    serpentine,
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "method must be random, ordered or error_diffusion",
                ));
            }
        };
        let config = V2Config::Dithering(DitheringConfig {
            method,
            n_colors,
            grayscale,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (blur_range=(7,15), cutoff_range=(std::f64::consts::FRAC_PI_4,std::f64::consts::FRAC_PI_2)))]
    fn ringing_overshoot(
        &self,
        blur_range: (u32, u32),
        cutoff_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::RingingOvershoot {
            blur_range,
            cutoff_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (mode="edge", alpha_range=(0.5, 1.0)))]
    fn enhance(&self, mode: &str, alpha_range: (f64, f64)) -> PyResult<Self> {
        use rivet_vision::transforms::EnhanceMode;
        let mode = match mode {
            "edge" => EnhanceMode::Edge,
            "detail" => EnhanceMode::Detail,
            _ => return Err(PyValueError::new_err("mode must be edge or detail")),
        };
        let config = V2Config::Enhance { mode, alpha_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (alpha_range=(0.2, 0.5), strength_range=(0.2, 0.7)))]
    fn emboss(&self, alpha_range: (f64, f64), strength_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::Emboss {
            alpha_range,
            strength_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (target_mean_range=(0.3, 0.5), gain_range=None))]
    fn exposure_matching(
        &self,
        target_mean_range: (f64, f64),
        gain_range: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let config = V2Config::ExposureMatching {
            target_mean_range,
            gain_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (alpha=0.1))]
    fn fancy_pca(&self, alpha: f64) -> PyResult<Self> {
        let config = V2Config::FancyPca { alpha };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (black_range=((0,0,0),(0,0,0)), white_range=((255,255,255),(255,255,255)), mid_range=None, mid_value_range=(127,127)))]
    fn colorize(
        &self,
        black_range: ((u8, u8, u8), (u8, u8, u8)),
        white_range: ((u8, u8, u8), (u8, u8, u8)),
        mid_range: Option<((u8, u8, u8), (u8, u8, u8))>,
        mid_value_range: (u8, u8),
    ) -> PyResult<Self> {
        let anchors = |((r0, g0, b0), (r1, g1, b1))| ([r0, g0, b0], [r1, g1, b1]);
        let config = V2Config::Colorize(rivet_vision::transforms::ColorizeConfig {
            black_range: anchors(black_range),
            white_range: anchors(white_range),
            mid_range: mid_range.map(anchors),
            mid_value_range,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (alpha=1.0))]
    fn to_sepia(&self, alpha: f64) -> PyResult<Self> {
        let config = V2Config::ToSepia { alpha };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (r_shift_range=(-20.0, 20.0), g_shift_range=(-20.0, 20.0), b_shift_range=(-20.0, 20.0)))]
    fn rgb_shift(
        &self,
        r_shift_range: (f64, f64),
        g_shift_range: (f64, f64),
        b_shift_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::RgbShift(std::sync::Arc::new([
            r_shift_range,
            g_shift_range,
            b_shift_range,
        ]));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (gamma_range=(80.0, 120.0)))]
    fn random_gamma(&self, gamma_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::RandomGamma { gamma_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (fog_coef_range=(0.3, 1.0), alpha_coef=0.08))]
    fn random_fog(&self, fog_coef_range: (f64, f64), alpha_coef: f64) -> PyResult<Self> {
        let config = V2Config::RandomFog(rivet_vision::transforms::RandomFogConfig {
            fog_coef_range,
            alpha_coef,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (density_range=(1.0, 3.0), fog_color=(200, 200, 200), depth_mode="linear"))]
    fn atmospheric_fog(
        &self,
        density_range: (f64, f64),
        fog_color: (u8, u8, u8),
        depth_mode: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{AtmosphericFogConfig, FogDepthMode};
        let depth_mode = match depth_mode {
            "linear" => FogDepthMode::Linear,
            "diagonal" => FogDepthMode::Diagonal,
            "radial" => FogDepthMode::Radial,
            _ => {
                return Err(PyValueError::new_err(
                    "depth_mode must be linear, diagonal or radial",
                ));
            }
        };
        let (r, g, b) = fog_color;
        let config = V2Config::AtmosphericFog(AtmosphericFogConfig {
            density_range,
            fog_color: [r, g, b],
            depth_mode,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (mean_range=(0.65, 0.65), std_range=(0.3, 0.3), gauss_sigma_range=(2.0, 3.0), cutout_threshold_range=(0.68, 0.68), intensity_range=(0.6, 0.6), mode="rain", color=None))]
    fn spatter(
        &self,
        mean_range: (f64, f64),
        std_range: (f64, f64),
        gauss_sigma_range: (f64, f64),
        cutout_threshold_range: (f64, f64),
        intensity_range: (f64, f64),
        mode: &str,
        color: Option<(u8, u8, u8)>,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{SpatterConfig, SpatterMode};
        let mode = match mode {
            "rain" => SpatterMode::Rain,
            "mud" => SpatterMode::Mud,
            _ => return Err(PyValueError::new_err("mode must be rain or mud")),
        };
        let config = V2Config::Spatter(std::sync::Arc::new(SpatterConfig {
            mean_range,
            std_range,
            gauss_sigma_range,
            cutout_threshold_range,
            intensity_range,
            mode,
            color: color.map(|(r, g, b)| [r, g, b]),
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (gravel_roi=(0.1, 0.4, 0.9, 0.9), number_of_patches=2))]
    fn random_gravel(
        &self,
        gravel_roi: (f64, f64, f64, f64),
        number_of_patches: u32,
    ) -> PyResult<Self> {
        let config = V2Config::RandomGravel(rivet_vision::transforms::RandomGravelConfig {
            gravel_roi,
            number_of_patches,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (snow_point_range=(0.1, 0.3), brightness_coeff=2.5, method="bleach"))]
    fn random_snow(
        &self,
        snow_point_range: (f64, f64),
        brightness_coeff: f64,
        method: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{RandomSnowConfig, SnowMethod};
        let method = match method {
            "bleach" => SnowMethod::Bleach,
            "texture" => SnowMethod::Texture,
            _ => return Err(PyValueError::new_err("method must be bleach or texture")),
        };
        let config = V2Config::RandomSnow(RandomSnowConfig {
            snow_point_range,
            brightness_coeff,
            method,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (slant_range=(-10.0, 10.0), drop_length=None, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type="default"))]
    fn random_rain(
        &self,
        slant_range: (f64, f64),
        drop_length: Option<u32>,
        drop_width: u32,
        drop_color: (u8, u8, u8),
        blur_value: u32,
        brightness_coefficient: f64,
        rain_type: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{RainType, RandomRainConfig};
        let style = match rain_type {
            "default" => RainType::Default,
            "drizzle" => RainType::Drizzle,
            "heavy" => RainType::Heavy,
            "torrential" => RainType::Torrential,
            _ => {
                return Err(PyValueError::new_err(
                    "rain_type must be default, drizzle, heavy or torrential",
                ));
            }
        };
        let (r, g, b) = drop_color;
        let config = V2Config::RandomRain(std::sync::Arc::new(RandomRainConfig {
            slant_range,
            drop_length,
            drop_width,
            drop_color: [r, g, b],
            blur_value,
            brightness_coefficient,
            style,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (shadow_roi=(0.0, 0.5, 1.0, 1.0), num_shadows_range=(1, 2), shadow_dimension=5, shadow_intensity_range=(0.5, 0.5)))]
    fn random_shadow(
        &self,
        shadow_roi: (f64, f64, f64, f64),
        num_shadows_range: (u32, u32),
        shadow_dimension: u32,
        shadow_intensity_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::RandomShadow(std::sync::Arc::new(
            rivet_vision::transforms::RandomShadowConfig {
                shadow_roi,
                num_shadows_range,
                shadow_dimension,
                shadow_intensity_range,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (shadow_intensity_range=(0.3, 0.7), plasma_size=256, roughness=3.0))]
    fn plasma_shadow(
        &self,
        shadow_intensity_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    ) -> PyResult<Self> {
        let config = V2Config::PlasmaShadow {
            shadow_intensity_range,
            plasma_size,
            roughness,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (brightness_range=(-0.3, 0.3), contrast_range=(-0.3, 0.3), plasma_size=256, roughness=3.0))]
    fn plasma_brightness_contrast(
        &self,
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    ) -> PyResult<Self> {
        let config = V2Config::PlasmaBrightnessContrast {
            brightness_range,
            contrast_range,
            plasma_size,
            roughness,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (mode="linear", intensity_range=(0.01, 0.2), effect_type="both"))]
    fn illumination(
        &self,
        mode: &str,
        intensity_range: (f64, f64),
        effect_type: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{IlluminationEffect, IlluminationMode};
        let mode = match mode {
            "linear" => IlluminationMode::Linear,
            "corner" => IlluminationMode::Corner,
            "gaussian" => IlluminationMode::Gaussian,
            _ => {
                return Err(PyValueError::new_err(
                    "mode must be linear, corner or gaussian",
                ));
            }
        };
        let effect = match effect_type {
            "brighten" => IlluminationEffect::Brighten,
            "darken" => IlluminationEffect::Darken,
            "both" => IlluminationEffect::Both,
            _ => {
                return Err(PyValueError::new_err(
                    "effect_type must be brighten, darken or both",
                ));
            }
        };
        let config = V2Config::Illumination {
            intensity_range,
            mode,
            effect,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (intensity_range=(0.1, 0.3), grain_size_range=(1, 3), *, per_channel=false))]
    fn film_grain(
        &self,
        intensity_range: (f64, f64),
        grain_size_range: (u32, u32),
        per_channel: bool,
    ) -> PyResult<Self> {
        let config = V2Config::FilmGrain {
            intensity_range,
            grain_size_range,
            per_channel,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (color_shift_range=(0.01, 0.05), intensity_range=(0.1, 0.5)))]
    fn iso_noise(
        &self,
        color_shift_range: (f64, f64),
        intensity_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::IsoNoise {
            color_shift_range,
            intensity_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (sigma=0.7, max_delta=4, iterations=2))]
    fn glass_blur(&self, sigma: f64, max_delta: u32, iterations: u32) -> PyResult<Self> {
        let config = V2Config::GlassBlur {
            sigma,
            max_delta,
            iterations,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (max_factor_range=(1.0, 1.31), step_factor_range=(0.01, 0.03)))]
    fn zoom_blur(
        &self,
        max_factor_range: (f64, f64),
        step_factor_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::ZoomBlur {
            max_factor_range,
            step_factor_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (kernel_range=(3, 7), strength_range=(0.0, 1.0), *, per_channel=false, padding_mode="reflect"))]
    fn stochastic_convolution(
        &self,
        kernel_range: (u32, u32),
        strength_range: (f32, f32),
        per_channel: bool,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let config = V2Config::StochasticConvolution {
            kernel_range,
            strength_range,
            per_channel,
            padding_mode: parse_padding_mode(padding_mode)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (kernel_range=(3, 7), sigma_x_range=(0.2, 1.0), sigma_y_range=(0.2, 1.0), rotate_range=(-90.0, 90.0), beta_range=(0.5, 8.0), noise_range=(0.75, 1.25)))]
    fn advanced_blur(
        &self,
        kernel_range: (u32, u32),
        sigma_x_range: (f64, f64),
        sigma_y_range: (f64, f64),
        rotate_range: (f64, f64),
        beta_range: (f64, f64),
        noise_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::AdvancedBlur(std::sync::Arc::new(
            rivet_vision::transforms::AdvancedBlurConfig {
                kernel_range,
                sigma_x_range,
                sigma_y_range,
                rotate_range,
                beta_range,
                noise_range,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (kernel_range=(3, 7)))]
    fn blur(&self, kernel_range: (u32, u32)) -> PyResult<Self> {
        let config = V2Config::Blur { kernel_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn median_blur(&self) -> Self {
        Self {
            inner: self.inner.clone().v2(V2Config::MedianBlur),
        }
    }

    #[pyo3(signature = (temperature_range=(3000.0, 15000.0)))]
    fn planckian_jitter(&self, temperature_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::PlanckianJitter { temperature_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (intensity_range=(0.2, 0.5), center_range=(0.3, 0.7)))]
    fn vignetting(&self, intensity_range: (f64, f64), center_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::Vignetting {
            intensity_range,
            center_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (scale=0.1))]
    fn random_tone_curve(&self, scale: f64) -> PyResult<Self> {
        let config = V2Config::RandomToneCurve { scale };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (clip_limit=2.0, *, grid=(8, 8)))]
    fn clahe(&self, clip_limit: f64, grid: (u32, u32)) -> PyResult<Self> {
        let config = V2Config::Clahe { grid, clip_limit };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (red_range=(0.0, 0.05), blue_range=(-0.05, 0.0), *, interpolation="bilinear"))]
    fn chromatic_aberration(
        &self,
        red_range: (f64, f64),
        blue_range: (f64, f64),
        interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::ChromaticAberration {
            red_range,
            blue_range,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (radius_range=(1, 3)))]
    fn defocus(&self, radius_range: (u32, u32)) -> PyResult<Self> {
        let config = V2Config::Defocus { radius_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (holes=(1, 4), *, hole_height=(4, 8), hole_width=(4, 8), fill=0.0))]
    fn coarse_dropout(
        &self,
        holes: (u32, u32),
        hole_height: (u32, u32),
        hole_width: (u32, u32),
        fill: f32,
    ) -> PyResult<Self> {
        let config = V2Config::CoarseDropout {
            holes,
            hole_height,
            hole_width,
            fill,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (length_range=(3, 7), *, angle_range=(0.0, 180.0)))]
    fn motion_blur(&self, length_range: (u32, u32), angle_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::MotionBlur {
            length_range,
            angle_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (sigma=1.0, *, amount=1.0, threshold=0))]
    fn unsharp_mask(&self, sigma: f32, amount: f32, threshold: u8) -> PyResult<Self> {
        let config = V2Config::UnsharpMask {
            sigma,
            amount,
            threshold,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (scale_range=(0.25, 0.5), *, down_interpolation="nearest", up_interpolation="nearest"))]
    fn downscale(
        &self,
        scale_range: (f64, f64),
        down_interpolation: &str,
        up_interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::Downscale {
            scale_range,
            down_interpolation: parse_interpolation(down_interpolation)?,
            up_interpolation: parse_interpolation(up_interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (amount=0.05, *, salt_ratio=0.5))]
    fn salt_and_pepper(&self, amount: f64, salt_ratio: f64) -> PyResult<Self> {
        let config = V2Config::SaltAndPepper { amount, salt_ratio };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (channels_to_drop=None, dropout_ratio=0.5, fill=None))]
    fn channel_dropout(
        &self,
        channels_to_drop: Option<Vec<usize>>,
        dropout_ratio: f64,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{ChannelDropoutConfig, ChannelDropoutFill};
        let fill = match fill {
            None => ChannelDropoutFill::Constant(0.0),
            Some(value) if value.extract::<&str>().is_ok_and(|s| s == "mean") => {
                ChannelDropoutFill::Mean
            }
            Some(value) => ChannelDropoutFill::Constant(value.extract::<f32>()?),
        };
        let config = V2Config::ChannelDropout(std::sync::Arc::new(ChannelDropoutConfig {
            channels_to_drop,
            dropout_ratio,
            fill,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (num_masks_x_range=(0,0), num_masks_y_range=(0,0), mask_x_length_range=None, mask_y_length_range=None, *, fill=0.0, fill_mask=None))]
    fn xy_masking(
        &self,
        num_masks_x_range: (u32, u32),
        num_masks_y_range: (u32, u32),
        mask_x_length_range: Option<&Bound<'_, PyAny>>,
        mask_y_length_range: Option<&Bound<'_, PyAny>>,
        fill: f32,
        fill_mask: Option<i64>,
    ) -> PyResult<Self> {
        let config = V2Config::XyMasking(std::sync::Arc::new(
            rivet_vision::transforms::XyMaskingConfig {
                num_masks_x_range,
                num_masks_y_range,
                mask_x_length: parse_strip_length(mask_x_length_range)?,
                mask_y_length: parse_strip_length(mask_y_length_range)?,
                fill,
                fill_mask,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (dropout_prob=0.01, *, per_channel=false, drop_value=None, mask_drop_value=None))]
    fn pixel_dropout(
        &self,
        dropout_prob: f64,
        per_channel: bool,
        drop_value: Option<Bound<'_, PyAny>>,
        mask_drop_value: Option<i64>,
    ) -> PyResult<Self> {
        let drop_value = drop_value
            .as_ref()
            .map(|v| parse_crop_fill(Some(v)))
            .transpose()?;
        let config = V2Config::PixelDropout(std::sync::Arc::new(
            rivet_vision::transforms::PixelDropoutConfig {
                dropout_prob,
                per_channel,
                drop_value,
                mask_drop_value,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (unit_size, *, ratio=0.5, fill=0.0))]
    fn grid_dropout(&self, unit_size: usize, ratio: f64, fill: f32) -> PyResult<Self> {
        let config = V2Config::GridDropout {
            unit_size,
            ratio,
            fill,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (intensity=0.2))]
    fn shot_noise(&self, intensity: f32) -> PyResult<Self> {
        V2Config::ShotNoise { intensity }
            .validate()
            .map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().shot_noise(intensity),
        })
    }

    fn jpeg(&self, quality: Vec<u8>) -> PyResult<Self> {
        let range = parse_pair(Some(quality), "quality", [1, 100])?;
        let config = V2Config::Jpeg {
            quality: (range[0], range[1]),
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn rgb(&self) -> Self {
        Self {
            inner: self.inner.clone().v2(V2Config::Rgb),
        }
    }

    #[pyo3(signature = (dtype, scale=false))]
    fn to_dtype(&self, dtype: &str, scale: bool) -> PyResult<Self> {
        let dtype = match dtype.trim().to_ascii_lowercase().as_str() {
            "uint8" | "u8" => DType::U8,
            "float32" | "f32" => DType::F32,
            _ => {
                return Err(PyValueError::new_err(
                    "dtype must be uint8/u8 or float32/f32",
                ));
            }
        };
        let config = V2Config::ToDtype { dtype, scale };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn convert_image_dtype(&self, dtype: &str) -> PyResult<Self> {
        let dtype = match dtype.trim().to_ascii_lowercase().as_str() {
            "uint8" | "u8" => DType::U8,
            "float32" | "f32" => DType::F32,
            _ => {
                return Err(to_py_err(rivet_vision::api::invalid_argument(
                    "dtype must be uint8/u8 or float32/f32",
                )));
            }
        };
        Ok(Self {
            inner: self.inner.clone().convert_image_dtype(dtype),
        })
    }

    fn rotate(&self, angle: i32) -> PyResult<Self> {
        let angle = RotationAngle::try_from(angle).map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().rotate(angle),
        })
    }

    #[pyo3(signature = (angle, interpolation="nearest", expand=false, fill=None, center=None))]
    fn arbitrary_rotate(
        &self,
        angle: f32,
        interpolation: &str,
        expand: bool,
        fill: Option<&Bound<'_, PyAny>>,
        center: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config = rivet_vision::api::ArbitraryRotateConfig::new(angle).with_options(
            expand,
            interpolation,
            fill,
        );
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        if let Some(center) = center {
            let center = parse_pair(Some(center), "center", [0.0, 0.0])?;
            config = config.with_center(center[0], center[1]);
        }
        Ok(Self {
            inner: self.inner.clone().arbitrary_rotate_with_config(config),
        })
    }

    #[pyo3(signature = (degrees, translate=None, scale=None, shear=None, interpolation="nearest", fill=None, shear_ranges=None))]
    fn random_affine(
        &self,
        degrees: &Bound<'_, PyAny>,
        translate: Option<Vec<f32>>,
        scale: Option<Vec<f32>>,
        shear: Option<Vec<f32>>,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
        shear_ranges: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let (degrees, degree_range) = if let Ok(value) = degrees.extract::<f32>() {
            (value, None)
        } else {
            let range = parse_pair(Some(degrees.extract::<Vec<f32>>()?), "degrees", [0.0, 0.0])?;
            (0.0, Some(range))
        };
        let translate = parse_pair(translate, "translate", [0.0, 0.0])?;
        let scale = parse_pair(scale, "scale", [1.0, 1.0])?;
        if shear.is_some() && shear_ranges.is_some() {
            return Err(PyValueError::new_err(
                "pass either shear or shear_ranges, not both",
            ));
        }
        let shear = shear.unwrap_or_else(|| vec![0.0, 0.0]);
        let interpolation = parse_interpolation(interpolation)?;
        let mut config = RandomAffineConfig::new(degrees)
            .with_translate(translate[0], translate[1])
            .with_scale(scale[0], scale[1]);
        match shear.as_slice() {
            [max_x, max_y] => config = config.with_shear(*max_x, *max_y),
            _ => {
                return Err(PyValueError::new_err(
                    "shear must contain two symmetric maxima",
                ));
            }
        }
        if let Some(shear_ranges) = shear_ranges {
            match shear_ranges.as_slice() {
                [min_x, max_x] => config = config.with_shear_ranges([*min_x, *max_x], [0.0, 0.0]),
                [min_x, max_x, min_y, max_y] => {
                    config = config.with_shear_ranges([*min_x, *max_x], [*min_y, *max_y])
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "shear_ranges must have two x bounds or four x/y bounds",
                    ));
                }
            }
        }
        if let Some(range) = degree_range {
            config = config.with_degrees_range(range[0], range[1]);
        }
        config = config.with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().random_affine_with_config(config),
        })
    }

    #[pyo3(signature = (start_points, end_points, interpolation="bilinear", fill=None))]
    fn perspective(
        &self,
        start_points: Vec<Vec<f32>>,
        end_points: Vec<Vec<f32>>,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let start_points = parse_points(start_points, "start_points")?;
        let end_points = parse_points(end_points, "end_points")?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config =
            PerspectiveConfig::new(start_points, end_points).with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().perspective_with_config(config),
        })
    }

    #[pyo3(signature = (distortion_scale=0.5, probability=0.5, interpolation="bilinear", fill=None))]
    fn random_perspective(
        &self,
        distortion_scale: f32,
        probability: f64,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config = RandomPerspectiveConfig::new(distortion_scale, probability)
            .with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().random_perspective_with_config(config),
        })
    }

    #[pyo3(signature = (alpha, sigma, interpolation="bilinear", fill=None))]
    fn elastic_transform(
        &self,
        alpha: f32,
        sigma: f32,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config =
            ElasticTransformConfig::new(alpha, sigma).with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().elastic_transform_with_config(config),
        })
    }

    fn normalize(&self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self {
            inner: self.inner.clone().normalize(mean, std),
        }
    }

    fn hwc_to_chw(&self) -> Self {
        Self {
            inner: self.inner.clone().hwc_to_chw(),
        }
    }

    fn chw_to_hwc(&self) -> Self {
        Self {
            inner: self.inner.clone().chw_to_hwc(),
        }
    }

    fn random_apply(&self, probability: f64, nested: PyRef<'_, Self>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .random_apply(probability, nested.inner.clone()),
        }
    }

    #[pyo3(signature = (choices, p=None))]
    fn random_choice(&self, choices: Vec<PyRef<'_, Self>>, p: Option<Vec<f64>>) -> PyResult<Self> {
        validate_choice_weights(choices.len(), p.as_deref())?;
        let choices = choices
            .into_iter()
            .map(|choice| choice.inner.clone())
            .collect();
        Ok(Self {
            inner: match p {
                Some(weights) => self
                    .inner
                    .clone()
                    .random_choice_with_weights(choices, weights),
                None => self.inner.clone().random_choice(choices),
            },
        })
    }

    fn random_order(&self, nested: PyRef<'_, Self>) -> Self {
        Self {
            inner: self.inner.clone().random_order(nested.inner.clone()),
        }
    }
}

#[pyclass(name = "_ImagePipeline")]
pub(crate) struct PyImagePipeline {
    pub(crate) inner: ImagePipeline,
}

#[pymethods]
impl PyImagePipeline {
    fn decode_image(&self) -> Self {
        Self {
            inner: self.inner.clone().decode_image(),
        }
    }

    #[pyo3(signature = (width, height=None, interpolation="bilinear", max_size=None))]
    fn resize(
        &self,
        width: u32,
        height: Option<u32>,
        interpolation: &str,
        max_size: Option<u32>,
    ) -> PyResult<Self> {
        let interpolation = interpolation
            .parse::<InterpolationMode>()
            .map_err(to_py_err)?;
        let inner = if let Some(height) = height {
            self.inner
                .clone()
                .resize_with_interpolation(width, height, interpolation)
        } else {
            self.inner
                .clone()
                .resize_short_edge_with_interpolation(width, max_size, interpolation)
        };
        Ok(Self { inner })
    }

    fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().crop(x, y, width, height),
        }
    }

    fn five_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().five_crop(width, height),
        }
    }

    #[pyo3(signature = (width, height, vertical_flip=false))]
    fn ten_crop(&self, width: u32, height: u32, vertical_flip: bool) -> Self {
        Self {
            inner: self.inner.clone().ten_crop(width, height, vertical_flip),
        }
    }

    #[pyo3(signature = (range, y=None, interpolation="bilinear"))]
    fn random_scale(
        &self,
        range: (f64, f64),
        y: Option<(f64, f64)>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::RandomScale {
            range,
            y,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (max_size=None, *, max_size_hw=None, interpolation="bilinear"))]
    fn longest_max_size(
        &self,
        max_size: Option<Vec<u32>>,
        max_size_hw: Option<(Option<u32>, Option<u32>)>,
        interpolation: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{LongestMaxSizeConfig, LongestSizeLimit};
        let limit = match (max_size, max_size_hw) {
            (Some(sizes), None) => LongestSizeLimit::Longest(sizes),
            (None, Some((height, width))) => LongestSizeLimit::Axes { height, width },
            _ => {
                return Err(PyValueError::new_err(
                    "set exactly one of max_size or max_size_hw",
                ));
            }
        };
        let config = V2Config::LongestMaxSize(LongestMaxSizeConfig {
            limit,
            interpolation: parse_interpolation(interpolation)?,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (min_height=None, min_width=None, *, pad_height_divisor=None, pad_width_divisor=None, position="center", padding_mode="constant", fill=None, fill_mask=0))]
    fn pad_if_needed(
        &self,
        min_height: Option<u32>,
        min_width: Option<u32>,
        pad_height_divisor: Option<u32>,
        pad_width_divisor: Option<u32>,
        position: &str,
        padding_mode: &str,
        fill: Option<&Bound<'_, PyAny>>,
        fill_mask: i64,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{PadIfNeededConfig, PadPosition};
        let position = match position {
            "center" => PadPosition::Center,
            "top_left" => PadPosition::TopLeft,
            "top_right" => PadPosition::TopRight,
            "bottom_left" => PadPosition::BottomLeft,
            "bottom_right" => PadPosition::BottomRight,
            "random" => PadPosition::Random,
            _ => return Err(PyValueError::new_err("unknown padding position")),
        };
        let (fill, fill_channels) = parse_fill_f32(fill)?;
        let config = V2Config::PadIfNeeded(std::sync::Arc::new(PadIfNeededConfig {
            min_height,
            min_width,
            pad_height_divisor,
            pad_width_divisor,
            position,
            padding_mode: parse_padding_mode(padding_mode)?,
            fill,
            fill_channels,
            fill_mask,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn center_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().center_crop(width, height),
        }
    }

    #[pyo3(signature = (padding, fill=None, padding_mode="constant"))]
    fn pad(
        &self,
        padding: u32,
        fill: Option<&Bound<'_, PyAny>>,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let (fill, channels) = parse_fill_f32(fill)?;
        let mut config = rivet_vision::api::PadConfig::new(padding).with_fill(fill);
        if let Some(channels) = channels {
            config = config.with_fill_channels(channels);
        }
        let config = config.with_mode(parse_padding_mode(padding_mode)?);
        Ok(Self {
            inner: self.inner.clone().pad_with_config(config),
        })
    }

    #[pyo3(signature = (left, top, right, bottom, fill=None, padding_mode="constant"))]
    fn pad_with_sides(
        &self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        fill: Option<&Bound<'_, PyAny>>,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let (fill, channels) = parse_fill_f32(fill)?;
        let mut config =
            rivet_vision::api::PadConfig::with_sides(left, top, right, bottom).with_fill(fill);
        if let Some(channels) = channels {
            config = config.with_fill_channels(channels);
        }
        let config = config.with_mode(parse_padding_mode(padding_mode)?);
        Ok(Self {
            inner: self.inner.clone().pad_with_config(config),
        })
    }

    #[pyo3(signature=(policy="imagenet",interpolation="nearest",fill=0))]
    fn auto_augment(&self, policy: &str, interpolation: &str, fill: u8) -> PyResult<Self> {
        let mut config =
            rivet_vision::api::AutoAugmentConfig::new(policy.parse().map_err(to_py_err)?);
        config.interpolation = parse_interpolation(interpolation)?;
        config.fill = fill;
        Ok(Self {
            inner: self.inner.clone().auto_augment(config),
        })
    }
    #[pyo3(signature=(num_ops=2,magnitude=9,num_magnitude_bins=31,interpolation="nearest",fill=0))]
    fn rand_augment(
        &self,
        num_ops: usize,
        magnitude: usize,
        num_magnitude_bins: usize,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::RandAugmentConfig {
            num_ops,
            magnitude,
            num_magnitude_bins,
            interpolation: parse_interpolation(interpolation)?,
            fill,
        };
        Ok(Self {
            inner: self.inner.clone().rand_augment(config),
        })
    }
    #[pyo3(signature=(num_magnitude_bins=31,interpolation="nearest",fill=0))]
    fn trivial_augment_wide(
        &self,
        num_magnitude_bins: usize,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::TrivialAugmentWideConfig {
            num_magnitude_bins,
            interpolation: parse_interpolation(interpolation)?,
            fill,
        };
        Ok(Self {
            inner: self.inner.clone().trivial_augment_wide(config),
        })
    }
    #[pyo3(signature=(severity=3,mixture_width=3,chain_depth=-1,alpha=1.0,all_ops=true,interpolation="bilinear",fill=0))]
    fn augmix(
        &self,
        severity: usize,
        mixture_width: usize,
        chain_depth: i32,
        alpha: f32,
        all_ops: bool,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::AugMixConfig {
            severity,
            mixture_width,
            chain_depth,
            alpha,
            all_ops,
            interpolation: parse_interpolation(interpolation)?,
            fill,
        };
        Ok(Self {
            inner: self.inner.clone().augmix(config),
        })
    }

    fn to_tensor(&self) -> Self {
        Self {
            inner: self.inner.clone().to_tensor(),
        }
    }
    fn pil_to_tensor(&self) -> Self {
        Self {
            inner: self.inner.clone().pil_to_tensor(),
        }
    }
    fn linear_transform(&self, matrix: Vec<Vec<f32>>, mean: Vec<f32>) -> PyResult<Self> {
        let n = matrix.len();
        if matrix.iter().any(|row| row.len() != n) {
            return Err(PyValueError::new_err(
                "linear transformation matrix must be square",
            ));
        }
        let matrix = rivet_core::Tensor::from_vec(
            matrix.into_iter().flatten().collect::<Vec<_>>(),
            [n, n],
            &rivet_core::Device::Cpu,
        )
        .map_err(to_py_err)?;
        let n_mean = mean.len();
        let mean = rivet_core::Tensor::from_vec(mean, [n_mean], &rivet_core::Device::Cpu)
            .map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().linear_transform(
                rivet_vision::api::LinearTransformationConfig::new(matrix, mean),
            ),
        })
    }

    fn saturation(&self, factor: f32) -> Self {
        Self {
            inner: self.inner.clone().saturation(factor),
        }
    }
    #[pyo3(signature=(gamma,gain=1.0))]
    fn gamma(&self, gamma: f64, gain: f64) -> Self {
        Self {
            inner: self.inner.clone().gamma(gamma, gain),
        }
    }
    #[pyo3(signature=(angle,translate=None,scale=1.0,shear=None,interpolation="nearest",fill=None,center=None))]
    fn affine(
        &self,
        angle: f64,
        translate: Option<Vec<f64>>,
        scale: f64,
        shear: Option<Vec<f64>>,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
        center: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let mut config = rivet_vision::api::AffineConfig::new(
            angle,
            parse_pair(translate, "translate", [0.0, 0.0])?,
            scale,
            parse_pair(shear, "shear", [0.0, 0.0])?,
        )
        .with_options(parse_interpolation(interpolation)?, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        if let Some(center) = center {
            config = config.with_center(parse_pair(Some(center), "center", [0.0, 0.0])?);
        }
        Ok(Self {
            inner: self.inner.clone().affine(config),
        })
    }
    #[pyo3(signature=(x,y,width,height,value=None))]
    fn erase(
        &self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        value: Option<Vec<f32>>,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .erase(rivet_vision::api::EraseConfig::new(
                    x,
                    y,
                    width,
                    height,
                    value.unwrap_or_else(|| vec![0.0]),
                )),
        }
    }
    #[pyo3(signature=(degrees,interpolation="nearest",expand=false,fill=None,center=None))]
    fn random_rotation(
        &self,
        degrees: Vec<f32>,
        interpolation: &str,
        expand: bool,
        fill: Option<&Bound<'_, PyAny>>,
        center: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let mut config = rivet_vision::api::RandomRotationConfig::new(parse_pair(
            Some(degrees),
            "degrees",
            [0.0, 0.0],
        )?)
        .with_options(parse_interpolation(interpolation)?, expand, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        let config = if let Some(center) = center {
            config.with_center(parse_pair(Some(center), "center", [0.0, 0.0])?)
        } else {
            config
        };
        Ok(Self {
            inner: self.inner.clone().random_rotation(config),
        })
    }
    #[pyo3(signature=(kernel_size,sigma=None))]
    fn random_gaussian_blur(
        &self,
        kernel_size: &Bound<'_, PyAny>,
        sigma: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dimensions = parse_u32_scalar_or_pair(kernel_size, "kernel_size")?;
        let sigma = match sigma {
            Some(value) => parse_f32_scalar_or_pair(value, "sigma")?,
            None => [0.1, 2.0],
        };
        let config = rivet_vision::api::RandomGaussianBlurConfig::new(dimensions[0], sigma)
            .with_kernel_dimensions(dimensions);
        Ok(Self {
            inner: self.inner.clone().random_gaussian_blur(config),
        })
    }
    #[pyo3(signature=(x,y,width,height,out_width,out_height,interpolation="bilinear"))]
    fn resized_crop(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        out_width: u32,
        out_height: u32,
        interpolation: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().resized_crop(
                x,
                y,
                width,
                height,
                out_width,
                out_height,
                parse_interpolation(interpolation)?,
            ),
        })
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_vertical_flip(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_vertical_flip(probability),
        }
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_invert(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_invert(probability),
        }
    }
    #[pyo3(signature=(bits,probability=0.5))]
    fn random_posterize(&self, bits: u8, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_posterize(bits, probability),
        }
    }
    #[pyo3(signature=(threshold,probability=0.5))]
    fn random_solarize(&self, threshold: u8, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_solarize(threshold, probability),
        }
    }
    #[pyo3(signature=(factor,probability=0.5))]
    fn random_adjust_sharpness(&self, factor: f32, probability: f64) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .random_adjust_sharpness(factor, probability),
        }
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_autocontrast(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_autocontrast(probability),
        }
    }
    #[pyo3(signature=(probability=0.5))]
    fn random_equalize(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_equalize(probability),
        }
    }

    fn horizontal_flip(&self) -> Self {
        Self {
            inner: self.inner.clone().horizontal_flip(),
        }
    }

    fn vertical_flip(&self) -> Self {
        Self {
            inner: self.inner.clone().vertical_flip(),
        }
    }

    #[pyo3(signature = (width, height, padding=None, pad_if_needed=false, fill=None, padding_mode="constant"))]
    fn random_crop(
        &self,
        width: u32,
        height: u32,
        padding: Option<&Bound<'_, PyAny>>,
        pad_if_needed: bool,
        fill: Option<&Bound<'_, PyAny>>,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let (uniform, sides) = parse_crop_padding(padding)?;
        let config = rivet_vision::api::RandomCropConfig::new(width, height, uniform).with_options(
            sides,
            pad_if_needed,
            parse_crop_fill(fill)?,
            parse_padding_mode(padding_mode)?,
        );
        Ok(Self {
            inner: self.inner.clone().random_crop_with_config(config),
        })
    }

    fn random_resized_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().random_resized_crop(width, height),
        }
    }

    #[pyo3(signature = (width, height, scale=None, ratio=None, interpolation="bilinear"))]
    fn random_resized_crop_with_options(
        &self,
        width: u32,
        height: u32,
        scale: Option<Vec<f32>>,
        ratio: Option<Vec<f32>>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let scale = parse_pair(scale, "scale", [0.08, 1.0])?;
        let ratio = parse_pair(ratio, "ratio", [0.75, 1.3333334])?;
        let interpolation = parse_interpolation(interpolation)?;
        let config = rivet_vision::api::RandomResizedCropConfig::new(width, height)
            .with_params(scale[0], scale[1], ratio[0], ratio[1])
            .with_interpolation(interpolation);
        Ok(Self {
            inner: self.inner.clone().random_resized_crop_with_config(config),
        })
    }

    #[pyo3(signature = (probability=0.5))]
    fn random_horizontal_flip(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_horizontal_flip(probability),
        }
    }

    fn brightness(&self, value: f32) -> Self {
        Self {
            inner: self.inner.clone().brightness(value),
        }
    }

    fn contrast(&self, value: f32) -> Self {
        Self {
            inner: self.inner.clone().contrast(value),
        }
    }

    #[pyo3(signature = (brightness=0.0, contrast=0.0, hue=0.0, saturation=0.0))]
    fn color_jitter(&self, brightness: f32, contrast: f32, hue: f32, saturation: f32) -> Self {
        Self {
            inner: self.inner.clone().color_jitter_with_config(
                rivet_vision::api::ColorJitterConfig::new(brightness, contrast, hue)
                    .with_saturation(saturation),
            ),
        }
    }

    fn compose(&self, sequence: PyRef<'_, PyTransform>) -> Self {
        Self {
            inner: self.inner.clone().compose(sequence.inner.clone()),
        }
    }

    fn random_apply(&self, probability: f64, sequence: PyRef<'_, PyTransform>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .random_apply(probability, sequence.inner.clone()),
        }
    }

    #[pyo3(signature = (choices, p=None))]
    fn random_choice(
        &self,
        choices: Vec<PyRef<'_, PyTransform>>,
        p: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        validate_choice_weights(choices.len(), p.as_deref())?;
        let choices = choices
            .into_iter()
            .map(|choice| choice.inner.clone())
            .collect();
        Ok(Self {
            inner: match p {
                Some(weights) => self
                    .inner
                    .clone()
                    .random_choice_with_weights(choices, weights),
                None => self.inner.clone().random_choice(choices),
            },
        })
    }

    fn random_order(&self, sequence: PyRef<'_, PyTransform>) -> Self {
        Self {
            inner: self.inner.clone().random_order(sequence.inner.clone()),
        }
    }

    fn invert(&self) -> Self {
        Self {
            inner: self.inner.clone().invert(),
        }
    }

    fn posterize(&self, bits: u8) -> Self {
        Self {
            inner: self.inner.clone().posterize(bits),
        }
    }

    fn solarize(&self, threshold: u8) -> Self {
        Self {
            inner: self.inner.clone().solarize(threshold),
        }
    }

    fn autocontrast(&self) -> Self {
        Self {
            inner: self.inner.clone().autocontrast(),
        }
    }

    fn equalize(&self) -> Self {
        Self {
            inner: self.inner.clone().equalize(),
        }
    }

    fn sharpness(&self, amount: f32) -> Self {
        Self {
            inner: self.inner.clone().sharpness(amount),
        }
    }

    #[pyo3(signature = (sigma, kernel_size=None))]
    fn gaussian_blur(
        &self,
        sigma: &Bound<'_, PyAny>,
        kernel_size: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let sigmas = parse_f32_sigma_axes(sigma)?;
        let mut config =
            rivet_vision::api::GaussianBlurConfig::new(sigmas[0]).with_axis_sigmas(sigmas);
        if let Some(dimensions) = parse_optional_kernel(kernel_size)? {
            config = config.with_kernel_dimensions(dimensions);
        }
        Ok(Self {
            inner: self.inner.clone().gaussian_blur_with_config(config),
        })
    }

    fn hue(&self, factor: f32) -> Self {
        Self {
            inner: self.inner.clone().hue(factor),
        }
    }

    #[pyo3(signature = (num_output_channels=1))]
    fn grayscale(&self, num_output_channels: u8) -> Self {
        Self {
            inner: self.inner.clone().grayscale(num_output_channels),
        }
    }

    #[pyo3(signature = (probability=0.1, num_output_channels=3))]
    fn random_grayscale(&self, probability: f64, num_output_channels: u8) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .random_grayscale(probability, num_output_channels),
        }
    }

    #[pyo3(signature = (probability=0.5))]
    fn random_erasing(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_erasing(probability),
        }
    }

    #[pyo3(signature = (probability=0.5, scale=None, ratio=None, value=None))]
    fn random_erasing_with_options(
        &self,
        probability: f64,
        scale: Option<Vec<f32>>,
        ratio: Option<Vec<f32>>,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let scale = parse_pair(scale, "scale", [0.02, 0.33])?;
        let ratio = parse_pair(ratio, "ratio", [0.3, 3.3333333])?;
        let config = rivet_vision::api::RandomErasingConfig::new(probability)
            .with_params(scale[0], scale[1], ratio[0], ratio[1]);
        let config = if let Some(value) = value {
            if let Ok(value) = value.extract::<String>() {
                if value != "random" {
                    return Err(PyValueError::new_err("value string must be 'random'"));
                }
                config.with_random_value()
            } else if let Ok(value) = value.extract::<f32>() {
                config.with_value(value)
            } else {
                config.with_values(value.extract::<Vec<f32>>()?)
            }
        } else {
            config
        };
        Ok(Self {
            inner: self.inner.clone().random_erasing_with_config(config),
        })
    }

    fn permute_channels(&self, permutation: Vec<usize>) -> PyResult<Self> {
        let config = V2Config::PermuteChannels(permutation);
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn random_channel_permutation(&self) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .v2(V2Config::RandomChannelPermutation { probability: 1.0 }),
        }
    }

    #[pyo3(signature = (brightness, contrast, saturation, hue, probability=0.5))]
    fn random_photometric_distort(
        &self,
        brightness: Vec<f32>,
        contrast: Vec<f32>,
        saturation: Vec<f32>,
        hue: Vec<f32>,
        probability: f64,
    ) -> PyResult<Self> {
        let brightness = parse_pair(Some(brightness), "brightness", [0.875, 1.125])?;
        let contrast = parse_pair(Some(contrast), "contrast", [0.5, 1.5])?;
        let saturation = parse_pair(Some(saturation), "saturation", [0.5, 1.5])?;
        let hue = parse_pair(Some(hue), "hue", [-0.05, 0.05])?;
        let config = V2Config::RandomPhotometricDistort {
            brightness: (brightness[0], brightness[1]),
            contrast: (contrast[0], contrast[1]),
            saturation: (saturation[0], saturation[1]),
            hue: (hue[0], hue[1]),
            probability,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (min_size, max_size, interpolation="bilinear"))]
    fn random_resize(&self, min_size: u32, max_size: u32, interpolation: &str) -> PyResult<Self> {
        let config = V2Config::RandomResize {
            min_size,
            max_size,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (min_sizes, max_size=None, interpolation="bilinear"))]
    fn random_shortest_size(
        &self,
        min_sizes: Vec<u32>,
        max_size: Option<u32>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::RandomShortestSize {
            min_sizes,
            max_size,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (width, height, scale_range, interpolation="bilinear"))]
    fn scale_jitter(
        &self,
        width: u32,
        height: u32,
        scale_range: Vec<f64>,
        interpolation: &str,
    ) -> PyResult<Self> {
        let range = parse_pair(Some(scale_range), "scale_range", [0.1, 2.0])?;
        let config = V2Config::ScaleJitter {
            target_width: width,
            target_height: height,
            scale_range: (range[0], range[1]),
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (fill, side_range, probability=0.5))]
    fn random_zoom_out(
        &self,
        fill: &Bound<'_, PyAny>,
        side_range: Vec<f64>,
        probability: f64,
    ) -> PyResult<Self> {
        let fill = if let Ok(value) = fill.extract::<f32>() {
            vec![value]
        } else {
            fill.extract::<Vec<f32>>()?
        };
        let range = parse_pair(Some(side_range), "side_range", [1.0, 4.0])?;
        let config = V2Config::RandomZoomOut {
            fill,
            side_range: (range[0], range[1]),
            probability,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (mean=0.0, sigma=0.1, clip=true))]
    fn gaussian_noise(&self, mean: f64, sigma: f64, clip: bool) -> PyResult<Self> {
        let config = V2Config::GaussianNoise { mean, sigma, clip };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (width, height, *, interpolation="bilinear", fill=0.0, mask_fill=0))]
    fn letterbox(
        &self,
        width: u32,
        height: u32,
        interpolation: &str,
        fill: f32,
        mask_fill: i64,
    ) -> PyResult<Self> {
        let config = LetterBoxConfig {
            width,
            height,
            interpolation: parse_interpolation(interpolation)?,
            fill,
            mask_fill,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().letterbox(config),
        })
    }

    #[pyo3(signature = (noise_type="uniform", spatial_mode="constant", *, ranges=None, mean_range=(0.0, 0.0), scale_range=(0.1, 0.1), alpha_range=(0.5, 1.5), beta_range=(0.5, 1.5), patch_count_range=(1, 1), patch_height_range=(0.1, 1.0), patch_width_range=(0.1, 1.0), per_channel=false))]
    #[allow(clippy::too_many_arguments)]
    fn additive_noise(
        &self,
        noise_type: &str,
        spatial_mode: &str,
        ranges: Option<Vec<(f64, f64)>>,
        mean_range: (f64, f64),
        scale_range: (f64, f64),
        alpha_range: (f64, f64),
        beta_range: (f64, f64),
        patch_count_range: (u32, u32),
        patch_height_range: (f64, f64),
        patch_width_range: (f64, f64),
        per_channel: bool,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{AdditiveDistribution, AdditiveNoiseConfig, NoiseSharing};
        let distribution = match noise_type {
            "uniform" => AdditiveDistribution::Uniform(ranges.unwrap_or_else(|| vec![(-0.1, 0.1)])),
            "gaussian" => AdditiveDistribution::Gaussian {
                mean: mean_range,
                std: scale_range,
            },
            "laplace" => AdditiveDistribution::Laplace {
                mean: mean_range,
                scale: scale_range,
            },
            "beta" => AdditiveDistribution::Beta {
                alpha: alpha_range,
                beta: beta_range,
                scale: scale_range,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "noise_type must be uniform, gaussian, laplace or beta",
                ));
            }
        };
        let sharing = match spatial_mode {
            "constant" => NoiseSharing::Constant,
            "per_pixel" => NoiseSharing::PerPixel,
            "shared" => NoiseSharing::Shared,
            "patch" => NoiseSharing::Patch {
                count: patch_count_range,
                height: patch_height_range,
                width: patch_width_range,
                per_channel,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "spatial_mode must be constant, per_pixel, shared or patch",
                ));
            }
        };
        let config = V2Config::AdditiveNoise(std::sync::Arc::new(AdditiveNoiseConfig {
            distribution,
            sharing,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (multiplier=(0.9, 1.1), *, per_channel=false, elementwise=false))]
    fn multiplicative_noise(
        &self,
        multiplier: (f64, f64),
        per_channel: bool,
        elementwise: bool,
    ) -> PyResult<Self> {
        let config = V2Config::MultiplicativeNoise {
            multiplier,
            per_channel,
            elementwise,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (flare_roi=(0.0, 0.0, 1.0, 0.5), angle_range=(0.0, 1.0), num_flare_circles_range=(6, 10), src_radius=400, src_color=(255, 255, 255), physics_based=false))]
    fn random_sun_flare(
        &self,
        flare_roi: (f64, f64, f64, f64),
        angle_range: (f64, f64),
        num_flare_circles_range: (u32, u32),
        src_radius: u32,
        src_color: (u8, u8, u8),
        physics_based: bool,
    ) -> PyResult<Self> {
        let config = V2Config::RandomSunFlare(std::sync::Arc::new(
            rivet_vision::transforms::RandomSunFlareConfig {
                flare_roi,
                angle_range,
                num_flare_circles_range,
                src_radius,
                src_color,
                physics_based,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (flare_roi=(0.0, 0.0, 1.0, 0.5), num_ghosts_range=(3, 7), intensity_range=(0.3, 0.7), num_rays_range=(4, 8), bloom_range=(0.01, 0.05)))]
    fn lens_flare(
        &self,
        flare_roi: (f64, f64, f64, f64),
        num_ghosts_range: (u32, u32),
        intensity_range: (f64, f64),
        num_rays_range: (u32, u32),
        bloom_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::LensFlare(std::sync::Arc::new(
            rivet_vision::transforms::LensFlareConfig {
                flare_roi,
                num_ghosts_range,
                intensity_range,
                num_rays_range,
                bloom_range,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (brightness_range=(-0.2, 0.2), contrast_range=(-0.2, 0.2), *, brightness_by_max=false, ensure_safe_output=false))]
    fn random_brightness_contrast(
        &self,
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        brightness_by_max: bool,
        ensure_safe_output: bool,
    ) -> PyResult<Self> {
        let config = V2Config::RandomBrightnessContrast {
            brightness_range,
            contrast_range,
            brightness_by_max,
            ensure_safe_output,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (hue_shift_range=(-20.0, 20.0), sat_shift_range=(-30.0, 30.0), val_shift_range=(-20.0, 20.0)))]
    fn hue_saturation_value(
        &self,
        hue_shift_range: (f64, f64),
        sat_shift_range: (f64, f64),
        val_shift_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::HueSaturationValue(std::sync::Arc::new([
            hue_shift_range,
            sat_shift_range,
            val_shift_range,
        ]));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (dot_size_range=(4,10), blend_range=(0.0,0.5)))]
    fn halftone(&self, dot_size_range: (u32, u32), blend_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::Halftone {
            dot_size_range,
            blend_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (method="error_diffusion", n_colors=2, color_mode="grayscale", error_diffusion_algorithm="floyd_steinberg", bayer_matrix_size=4, serpentine=false, noise_range=(-0.5,0.5)))]
    fn dithering(
        &self,
        method: &str,
        n_colors: u16,
        color_mode: &str,
        error_diffusion_algorithm: &str,
        bayer_matrix_size: u8,
        serpentine: bool,
        noise_range: (f64, f64),
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{DiffusionAlgorithm, DitherMethod, DitheringConfig};
        let grayscale = match color_mode {
            "grayscale" => true,
            "per_channel" => false,
            _ => {
                return Err(PyValueError::new_err(
                    "color_mode must be grayscale or per_channel",
                ));
            }
        };
        let method = match method {
            "random" => DitherMethod::Random { noise_range },
            "ordered" => DitherMethod::Ordered {
                size: bayer_matrix_size,
            },
            "error_diffusion" => {
                let algorithm = match error_diffusion_algorithm {
                    "floyd_steinberg" => DiffusionAlgorithm::FloydSteinberg,
                    "jarvis" => DiffusionAlgorithm::Jarvis,
                    "stucki" => DiffusionAlgorithm::Stucki,
                    "atkinson" => DiffusionAlgorithm::Atkinson,
                    "burkes" => DiffusionAlgorithm::Burkes,
                    "sierra" => DiffusionAlgorithm::Sierra,
                    "sierra_2row" => DiffusionAlgorithm::Sierra2Row,
                    "sierra_lite" => DiffusionAlgorithm::SierraLite,
                    _ => return Err(PyValueError::new_err("unknown error diffusion algorithm")),
                };
                DitherMethod::ErrorDiffusion {
                    algorithm,
                    serpentine,
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "method must be random, ordered or error_diffusion",
                ));
            }
        };
        let config = V2Config::Dithering(DitheringConfig {
            method,
            n_colors,
            grayscale,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (blur_range=(7,15), cutoff_range=(std::f64::consts::FRAC_PI_4,std::f64::consts::FRAC_PI_2)))]
    fn ringing_overshoot(
        &self,
        blur_range: (u32, u32),
        cutoff_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::RingingOvershoot {
            blur_range,
            cutoff_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (mode="edge", alpha_range=(0.5, 1.0)))]
    fn enhance(&self, mode: &str, alpha_range: (f64, f64)) -> PyResult<Self> {
        use rivet_vision::transforms::EnhanceMode;
        let mode = match mode {
            "edge" => EnhanceMode::Edge,
            "detail" => EnhanceMode::Detail,
            _ => return Err(PyValueError::new_err("mode must be edge or detail")),
        };
        let config = V2Config::Enhance { mode, alpha_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (alpha_range=(0.2, 0.5), strength_range=(0.2, 0.7)))]
    fn emboss(&self, alpha_range: (f64, f64), strength_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::Emboss {
            alpha_range,
            strength_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (target_mean_range=(0.3, 0.5), gain_range=None))]
    fn exposure_matching(
        &self,
        target_mean_range: (f64, f64),
        gain_range: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let config = V2Config::ExposureMatching {
            target_mean_range,
            gain_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (alpha=0.1))]
    fn fancy_pca(&self, alpha: f64) -> PyResult<Self> {
        let config = V2Config::FancyPca { alpha };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (black_range=((0,0,0),(0,0,0)), white_range=((255,255,255),(255,255,255)), mid_range=None, mid_value_range=(127,127)))]
    fn colorize(
        &self,
        black_range: ((u8, u8, u8), (u8, u8, u8)),
        white_range: ((u8, u8, u8), (u8, u8, u8)),
        mid_range: Option<((u8, u8, u8), (u8, u8, u8))>,
        mid_value_range: (u8, u8),
    ) -> PyResult<Self> {
        let anchors = |((r0, g0, b0), (r1, g1, b1))| ([r0, g0, b0], [r1, g1, b1]);
        let config = V2Config::Colorize(rivet_vision::transforms::ColorizeConfig {
            black_range: anchors(black_range),
            white_range: anchors(white_range),
            mid_range: mid_range.map(anchors),
            mid_value_range,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (alpha=1.0))]
    fn to_sepia(&self, alpha: f64) -> PyResult<Self> {
        let config = V2Config::ToSepia { alpha };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (r_shift_range=(-20.0, 20.0), g_shift_range=(-20.0, 20.0), b_shift_range=(-20.0, 20.0)))]
    fn rgb_shift(
        &self,
        r_shift_range: (f64, f64),
        g_shift_range: (f64, f64),
        b_shift_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::RgbShift(std::sync::Arc::new([
            r_shift_range,
            g_shift_range,
            b_shift_range,
        ]));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (gamma_range=(80.0, 120.0)))]
    fn random_gamma(&self, gamma_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::RandomGamma { gamma_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (fog_coef_range=(0.3, 1.0), alpha_coef=0.08))]
    fn random_fog(&self, fog_coef_range: (f64, f64), alpha_coef: f64) -> PyResult<Self> {
        let config = V2Config::RandomFog(rivet_vision::transforms::RandomFogConfig {
            fog_coef_range,
            alpha_coef,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (density_range=(1.0, 3.0), fog_color=(200, 200, 200), depth_mode="linear"))]
    fn atmospheric_fog(
        &self,
        density_range: (f64, f64),
        fog_color: (u8, u8, u8),
        depth_mode: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{AtmosphericFogConfig, FogDepthMode};
        let depth_mode = match depth_mode {
            "linear" => FogDepthMode::Linear,
            "diagonal" => FogDepthMode::Diagonal,
            "radial" => FogDepthMode::Radial,
            _ => {
                return Err(PyValueError::new_err(
                    "depth_mode must be linear, diagonal or radial",
                ));
            }
        };
        let (r, g, b) = fog_color;
        let config = V2Config::AtmosphericFog(AtmosphericFogConfig {
            density_range,
            fog_color: [r, g, b],
            depth_mode,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (mean_range=(0.65, 0.65), std_range=(0.3, 0.3), gauss_sigma_range=(2.0, 3.0), cutout_threshold_range=(0.68, 0.68), intensity_range=(0.6, 0.6), mode="rain", color=None))]
    fn spatter(
        &self,
        mean_range: (f64, f64),
        std_range: (f64, f64),
        gauss_sigma_range: (f64, f64),
        cutout_threshold_range: (f64, f64),
        intensity_range: (f64, f64),
        mode: &str,
        color: Option<(u8, u8, u8)>,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{SpatterConfig, SpatterMode};
        let mode = match mode {
            "rain" => SpatterMode::Rain,
            "mud" => SpatterMode::Mud,
            _ => return Err(PyValueError::new_err("mode must be rain or mud")),
        };
        let config = V2Config::Spatter(std::sync::Arc::new(SpatterConfig {
            mean_range,
            std_range,
            gauss_sigma_range,
            cutout_threshold_range,
            intensity_range,
            mode,
            color: color.map(|(r, g, b)| [r, g, b]),
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (gravel_roi=(0.1, 0.4, 0.9, 0.9), number_of_patches=2))]
    fn random_gravel(
        &self,
        gravel_roi: (f64, f64, f64, f64),
        number_of_patches: u32,
    ) -> PyResult<Self> {
        let config = V2Config::RandomGravel(rivet_vision::transforms::RandomGravelConfig {
            gravel_roi,
            number_of_patches,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (snow_point_range=(0.1, 0.3), brightness_coeff=2.5, method="bleach"))]
    fn random_snow(
        &self,
        snow_point_range: (f64, f64),
        brightness_coeff: f64,
        method: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{RandomSnowConfig, SnowMethod};
        let method = match method {
            "bleach" => SnowMethod::Bleach,
            "texture" => SnowMethod::Texture,
            _ => return Err(PyValueError::new_err("method must be bleach or texture")),
        };
        let config = V2Config::RandomSnow(RandomSnowConfig {
            snow_point_range,
            brightness_coeff,
            method,
        });
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (slant_range=(-10.0, 10.0), drop_length=None, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type="default"))]
    fn random_rain(
        &self,
        slant_range: (f64, f64),
        drop_length: Option<u32>,
        drop_width: u32,
        drop_color: (u8, u8, u8),
        blur_value: u32,
        brightness_coefficient: f64,
        rain_type: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{RainType, RandomRainConfig};
        let style = match rain_type {
            "default" => RainType::Default,
            "drizzle" => RainType::Drizzle,
            "heavy" => RainType::Heavy,
            "torrential" => RainType::Torrential,
            _ => {
                return Err(PyValueError::new_err(
                    "rain_type must be default, drizzle, heavy or torrential",
                ));
            }
        };
        let (r, g, b) = drop_color;
        let config = V2Config::RandomRain(std::sync::Arc::new(RandomRainConfig {
            slant_range,
            drop_length,
            drop_width,
            drop_color: [r, g, b],
            blur_value,
            brightness_coefficient,
            style,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (shadow_roi=(0.0, 0.5, 1.0, 1.0), num_shadows_range=(1, 2), shadow_dimension=5, shadow_intensity_range=(0.5, 0.5)))]
    fn random_shadow(
        &self,
        shadow_roi: (f64, f64, f64, f64),
        num_shadows_range: (u32, u32),
        shadow_dimension: u32,
        shadow_intensity_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::RandomShadow(std::sync::Arc::new(
            rivet_vision::transforms::RandomShadowConfig {
                shadow_roi,
                num_shadows_range,
                shadow_dimension,
                shadow_intensity_range,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (shadow_intensity_range=(0.3, 0.7), plasma_size=256, roughness=3.0))]
    fn plasma_shadow(
        &self,
        shadow_intensity_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    ) -> PyResult<Self> {
        let config = V2Config::PlasmaShadow {
            shadow_intensity_range,
            plasma_size,
            roughness,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (brightness_range=(-0.3, 0.3), contrast_range=(-0.3, 0.3), plasma_size=256, roughness=3.0))]
    fn plasma_brightness_contrast(
        &self,
        brightness_range: (f64, f64),
        contrast_range: (f64, f64),
        plasma_size: u32,
        roughness: f64,
    ) -> PyResult<Self> {
        let config = V2Config::PlasmaBrightnessContrast {
            brightness_range,
            contrast_range,
            plasma_size,
            roughness,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (mode="linear", intensity_range=(0.01, 0.2), effect_type="both"))]
    fn illumination(
        &self,
        mode: &str,
        intensity_range: (f64, f64),
        effect_type: &str,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{IlluminationEffect, IlluminationMode};
        let mode = match mode {
            "linear" => IlluminationMode::Linear,
            "corner" => IlluminationMode::Corner,
            "gaussian" => IlluminationMode::Gaussian,
            _ => {
                return Err(PyValueError::new_err(
                    "mode must be linear, corner or gaussian",
                ));
            }
        };
        let effect = match effect_type {
            "brighten" => IlluminationEffect::Brighten,
            "darken" => IlluminationEffect::Darken,
            "both" => IlluminationEffect::Both,
            _ => {
                return Err(PyValueError::new_err(
                    "effect_type must be brighten, darken or both",
                ));
            }
        };
        let config = V2Config::Illumination {
            intensity_range,
            mode,
            effect,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (intensity_range=(0.1, 0.3), grain_size_range=(1, 3), *, per_channel=false))]
    fn film_grain(
        &self,
        intensity_range: (f64, f64),
        grain_size_range: (u32, u32),
        per_channel: bool,
    ) -> PyResult<Self> {
        let config = V2Config::FilmGrain {
            intensity_range,
            grain_size_range,
            per_channel,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (color_shift_range=(0.01, 0.05), intensity_range=(0.1, 0.5)))]
    fn iso_noise(
        &self,
        color_shift_range: (f64, f64),
        intensity_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::IsoNoise {
            color_shift_range,
            intensity_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (sigma=0.7, max_delta=4, iterations=2))]
    fn glass_blur(&self, sigma: f64, max_delta: u32, iterations: u32) -> PyResult<Self> {
        let config = V2Config::GlassBlur {
            sigma,
            max_delta,
            iterations,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (max_factor_range=(1.0, 1.31), step_factor_range=(0.01, 0.03)))]
    fn zoom_blur(
        &self,
        max_factor_range: (f64, f64),
        step_factor_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::ZoomBlur {
            max_factor_range,
            step_factor_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (kernel_range=(3, 7), strength_range=(0.0, 1.0), *, per_channel=false, padding_mode="reflect"))]
    fn stochastic_convolution(
        &self,
        kernel_range: (u32, u32),
        strength_range: (f32, f32),
        per_channel: bool,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let config = V2Config::StochasticConvolution {
            kernel_range,
            strength_range,
            per_channel,
            padding_mode: parse_padding_mode(padding_mode)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (kernel_range=(3, 7), sigma_x_range=(0.2, 1.0), sigma_y_range=(0.2, 1.0), rotate_range=(-90.0, 90.0), beta_range=(0.5, 8.0), noise_range=(0.75, 1.25)))]
    fn advanced_blur(
        &self,
        kernel_range: (u32, u32),
        sigma_x_range: (f64, f64),
        sigma_y_range: (f64, f64),
        rotate_range: (f64, f64),
        beta_range: (f64, f64),
        noise_range: (f64, f64),
    ) -> PyResult<Self> {
        let config = V2Config::AdvancedBlur(std::sync::Arc::new(
            rivet_vision::transforms::AdvancedBlurConfig {
                kernel_range,
                sigma_x_range,
                sigma_y_range,
                rotate_range,
                beta_range,
                noise_range,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (kernel_range=(3, 7)))]
    fn blur(&self, kernel_range: (u32, u32)) -> PyResult<Self> {
        let config = V2Config::Blur { kernel_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn median_blur(&self) -> Self {
        Self {
            inner: self.inner.clone().v2(V2Config::MedianBlur),
        }
    }

    #[pyo3(signature = (temperature_range=(3000.0, 15000.0)))]
    fn planckian_jitter(&self, temperature_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::PlanckianJitter { temperature_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (intensity_range=(0.2, 0.5), center_range=(0.3, 0.7)))]
    fn vignetting(&self, intensity_range: (f64, f64), center_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::Vignetting {
            intensity_range,
            center_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (scale=0.1))]
    fn random_tone_curve(&self, scale: f64) -> PyResult<Self> {
        let config = V2Config::RandomToneCurve { scale };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (clip_limit=2.0, *, grid=(8, 8)))]
    fn clahe(&self, clip_limit: f64, grid: (u32, u32)) -> PyResult<Self> {
        let config = V2Config::Clahe { grid, clip_limit };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (red_range=(0.0, 0.05), blue_range=(-0.05, 0.0), *, interpolation="bilinear"))]
    fn chromatic_aberration(
        &self,
        red_range: (f64, f64),
        blue_range: (f64, f64),
        interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::ChromaticAberration {
            red_range,
            blue_range,
            interpolation: parse_interpolation(interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (radius_range=(1, 3)))]
    fn defocus(&self, radius_range: (u32, u32)) -> PyResult<Self> {
        let config = V2Config::Defocus { radius_range };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (holes=(1, 4), *, hole_height=(4, 8), hole_width=(4, 8), fill=0.0))]
    fn coarse_dropout(
        &self,
        holes: (u32, u32),
        hole_height: (u32, u32),
        hole_width: (u32, u32),
        fill: f32,
    ) -> PyResult<Self> {
        let config = V2Config::CoarseDropout {
            holes,
            hole_height,
            hole_width,
            fill,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (length_range=(3, 7), *, angle_range=(0.0, 180.0)))]
    fn motion_blur(&self, length_range: (u32, u32), angle_range: (f64, f64)) -> PyResult<Self> {
        let config = V2Config::MotionBlur {
            length_range,
            angle_range,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (sigma=1.0, *, amount=1.0, threshold=0))]
    fn unsharp_mask(&self, sigma: f32, amount: f32, threshold: u8) -> PyResult<Self> {
        let config = V2Config::UnsharpMask {
            sigma,
            amount,
            threshold,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (scale_range=(0.25, 0.5), *, down_interpolation="nearest", up_interpolation="nearest"))]
    fn downscale(
        &self,
        scale_range: (f64, f64),
        down_interpolation: &str,
        up_interpolation: &str,
    ) -> PyResult<Self> {
        let config = V2Config::Downscale {
            scale_range,
            down_interpolation: parse_interpolation(down_interpolation)?,
            up_interpolation: parse_interpolation(up_interpolation)?,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (amount=0.05, *, salt_ratio=0.5))]
    fn salt_and_pepper(&self, amount: f64, salt_ratio: f64) -> PyResult<Self> {
        let config = V2Config::SaltAndPepper { amount, salt_ratio };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (channels_to_drop=None, dropout_ratio=0.5, fill=None))]
    fn channel_dropout(
        &self,
        channels_to_drop: Option<Vec<usize>>,
        dropout_ratio: f64,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        use rivet_vision::transforms::{ChannelDropoutConfig, ChannelDropoutFill};
        let fill = match fill {
            None => ChannelDropoutFill::Constant(0.0),
            Some(value) if value.extract::<&str>().is_ok_and(|s| s == "mean") => {
                ChannelDropoutFill::Mean
            }
            Some(value) => ChannelDropoutFill::Constant(value.extract::<f32>()?),
        };
        let config = V2Config::ChannelDropout(std::sync::Arc::new(ChannelDropoutConfig {
            channels_to_drop,
            dropout_ratio,
            fill,
        }));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (num_masks_x_range=(0,0), num_masks_y_range=(0,0), mask_x_length_range=None, mask_y_length_range=None, *, fill=0.0, fill_mask=None))]
    fn xy_masking(
        &self,
        num_masks_x_range: (u32, u32),
        num_masks_y_range: (u32, u32),
        mask_x_length_range: Option<&Bound<'_, PyAny>>,
        mask_y_length_range: Option<&Bound<'_, PyAny>>,
        fill: f32,
        fill_mask: Option<i64>,
    ) -> PyResult<Self> {
        let config = V2Config::XyMasking(std::sync::Arc::new(
            rivet_vision::transforms::XyMaskingConfig {
                num_masks_x_range,
                num_masks_y_range,
                mask_x_length: parse_strip_length(mask_x_length_range)?,
                mask_y_length: parse_strip_length(mask_y_length_range)?,
                fill,
                fill_mask,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (dropout_prob=0.01, *, per_channel=false, drop_value=None, mask_drop_value=None))]
    fn pixel_dropout(
        &self,
        dropout_prob: f64,
        per_channel: bool,
        drop_value: Option<Bound<'_, PyAny>>,
        mask_drop_value: Option<i64>,
    ) -> PyResult<Self> {
        let drop_value = drop_value
            .as_ref()
            .map(|v| parse_crop_fill(Some(v)))
            .transpose()?;
        let config = V2Config::PixelDropout(std::sync::Arc::new(
            rivet_vision::transforms::PixelDropoutConfig {
                dropout_prob,
                per_channel,
                drop_value,
                mask_drop_value,
            },
        ));
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (unit_size, *, ratio=0.5, fill=0.0))]
    fn grid_dropout(&self, unit_size: usize, ratio: f64, fill: f32) -> PyResult<Self> {
        let config = V2Config::GridDropout {
            unit_size,
            ratio,
            fill,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    #[pyo3(signature = (intensity=0.2))]
    fn shot_noise(&self, intensity: f32) -> PyResult<Self> {
        V2Config::ShotNoise { intensity }
            .validate()
            .map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().shot_noise(intensity),
        })
    }

    fn jpeg(&self, quality: Vec<u8>) -> PyResult<Self> {
        let range = parse_pair(Some(quality), "quality", [1, 100])?;
        let config = V2Config::Jpeg {
            quality: (range[0], range[1]),
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn rgb(&self) -> Self {
        Self {
            inner: self.inner.clone().v2(V2Config::Rgb),
        }
    }

    #[pyo3(signature = (dtype, scale=false))]
    fn to_dtype(&self, dtype: &str, scale: bool) -> PyResult<Self> {
        let dtype = match dtype.trim().to_ascii_lowercase().as_str() {
            "uint8" | "u8" => DType::U8,
            "float32" | "f32" => DType::F32,
            _ => {
                return Err(PyValueError::new_err(
                    "dtype must be uint8/u8 or float32/f32",
                ));
            }
        };
        let config = V2Config::ToDtype { dtype, scale };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().v2(config),
        })
    }

    fn convert_image_dtype(&self, dtype: &str) -> PyResult<Self> {
        let dtype = match dtype.trim().to_ascii_lowercase().as_str() {
            "uint8" | "u8" => DType::U8,
            "float32" | "f32" => DType::F32,
            _ => {
                return Err(to_py_err(rivet_vision::api::invalid_argument(
                    "dtype must be uint8/u8 or float32/f32",
                )));
            }
        };
        Ok(Self {
            inner: self.inner.clone().convert_image_dtype(dtype),
        })
    }

    fn rotate(&self, angle: i32) -> PyResult<Self> {
        let angle = RotationAngle::try_from(angle).map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().rotate(angle),
        })
    }

    #[pyo3(signature = (angle, interpolation="nearest", expand=false, fill=None, center=None))]
    fn arbitrary_rotate(
        &self,
        angle: f32,
        interpolation: &str,
        expand: bool,
        fill: Option<&Bound<'_, PyAny>>,
        center: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config = rivet_vision::api::ArbitraryRotateConfig::new(angle).with_options(
            expand,
            interpolation,
            fill,
        );
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        if let Some(center) = center {
            let center = parse_pair(Some(center), "center", [0.0, 0.0])?;
            config = config.with_center(center[0], center[1]);
        }
        Ok(Self {
            inner: self.inner.clone().arbitrary_rotate_with_config(config),
        })
    }

    #[pyo3(signature = (degrees, translate=None, scale=None, shear=None, interpolation="nearest", fill=None, shear_ranges=None))]
    fn random_affine(
        &self,
        degrees: &Bound<'_, PyAny>,
        translate: Option<Vec<f32>>,
        scale: Option<Vec<f32>>,
        shear: Option<Vec<f32>>,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
        shear_ranges: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let (degrees, degree_range) = if let Ok(value) = degrees.extract::<f32>() {
            (value, None)
        } else {
            let range = parse_pair(Some(degrees.extract::<Vec<f32>>()?), "degrees", [0.0, 0.0])?;
            (0.0, Some(range))
        };
        let translate = parse_pair(translate, "translate", [0.0, 0.0])?;
        let scale = parse_pair(scale, "scale", [1.0, 1.0])?;
        if shear.is_some() && shear_ranges.is_some() {
            return Err(PyValueError::new_err(
                "pass either shear or shear_ranges, not both",
            ));
        }
        let shear = shear.unwrap_or_else(|| vec![0.0, 0.0]);
        let interpolation = parse_interpolation(interpolation)?;
        let mut config = RandomAffineConfig::new(degrees)
            .with_translate(translate[0], translate[1])
            .with_scale(scale[0], scale[1]);
        match shear.as_slice() {
            [max_x, max_y] => config = config.with_shear(*max_x, *max_y),
            _ => {
                return Err(PyValueError::new_err(
                    "shear must contain two symmetric maxima",
                ));
            }
        }
        if let Some(shear_ranges) = shear_ranges {
            match shear_ranges.as_slice() {
                [min_x, max_x] => config = config.with_shear_ranges([*min_x, *max_x], [0.0, 0.0]),
                [min_x, max_x, min_y, max_y] => {
                    config = config.with_shear_ranges([*min_x, *max_x], [*min_y, *max_y])
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "shear_ranges must have two x bounds or four x/y bounds",
                    ));
                }
            }
        }
        if let Some(range) = degree_range {
            config = config.with_degrees_range(range[0], range[1]);
        }
        config = config.with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().random_affine_with_config(config),
        })
    }

    #[pyo3(signature = (start_points, end_points, interpolation="bilinear", fill=None))]
    fn perspective(
        &self,
        start_points: Vec<Vec<f32>>,
        end_points: Vec<Vec<f32>>,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let start_points = parse_points(start_points, "start_points")?;
        let end_points = parse_points(end_points, "end_points")?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config =
            PerspectiveConfig::new(start_points, end_points).with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().perspective_with_config(config),
        })
    }

    #[pyo3(signature = (distortion_scale=0.5, probability=0.5, interpolation="bilinear", fill=None))]
    fn random_perspective(
        &self,
        distortion_scale: f32,
        probability: f64,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config = RandomPerspectiveConfig::new(distortion_scale, probability)
            .with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().random_perspective_with_config(config),
        })
    }

    #[pyo3(signature = (alpha, sigma, interpolation="bilinear", fill=None))]
    fn elastic_transform(
        &self,
        alpha: f32,
        sigma: f32,
        interpolation: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let (fill, fill_channels) = parse_fill_u8(fill)?;
        let interpolation = parse_interpolation(interpolation)?;
        let mut config =
            ElasticTransformConfig::new(alpha, sigma).with_options(interpolation, fill);
        if let Some(channels) = fill_channels {
            config = config.with_fill_channels(channels);
        }
        Ok(Self {
            inner: self.inner.clone().elastic_transform_with_config(config),
        })
    }

    fn normalize(&self, mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self {
            inner: self.inner.clone().normalize(mean, std),
        }
    }

    fn hwc_to_chw(&self) -> Self {
        Self {
            inner: self.inner.clone().hwc_to_chw(),
        }
    }

    fn chw_to_hwc(&self) -> Self {
        Self {
            inner: self.inner.clone().chw_to_hwc(),
        }
    }

    fn skip(&self, count: usize) -> Self {
        Self {
            inner: self.inner.clone().skip(count),
        }
    }

    fn take(&self, count: usize) -> Self {
        Self {
            inner: self.inner.clone().take(count),
        }
    }

    #[pyo3(signature = (size, drop_last=false))]
    fn batch(&self, size: usize, drop_last: bool) -> Self {
        Self {
            inner: self.inner.clone().batch(size, drop_last),
        }
    }

    #[pyo3(signature = (alpha=1.0, num_classes=None, cutmix=false))]
    fn batch_mix(&self, alpha: f64, num_classes: Option<usize>, cutmix: bool) -> PyResult<Self> {
        let config = rivet_vision::api::BatchMixConfig {
            alpha,
            num_classes,
            cutmix,
        };
        config.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().batch_mix(config),
        })
    }

    fn seed(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().seed(seed),
        }
    }

    #[pyo3(signature = (seed=None))]
    fn shuffle(&self, seed: Option<u64>) -> Self {
        Self {
            inner: self.inner.clone().shuffle(seed.unwrap_or(0)),
        }
    }

    fn epoch(&self, epoch: u64) -> Self {
        Self {
            inner: self.inner.clone().epoch(epoch),
        }
    }

    fn workers(&self, num_workers: usize) -> Self {
        Self {
            inner: self.inner.clone().workers(num_workers),
        }
    }

    fn prefetch_batches(&self, prefetch: usize) -> Self {
        Self {
            inner: self.inner.clone().prefetch_batches(prefetch),
        }
    }

    fn profiling(&self, enabled: bool) -> Self {
        Self {
            inner: self.inner.clone().profiling(enabled),
        }
    }

    fn execute(&self) -> PyResult<PyDataLoader> {
        Ok(PyDataLoader {
            inner: self.inner.clone().compile().map_err(to_py_err)?,
        })
    }

    fn explain(&self) -> PyResult<String> {
        self.inner
            .optimization_explain(Default::default())
            .map_err(to_py_err)
    }
}

fn validate_choice_weights(count: usize, weights: Option<&[f64]>) -> PyResult<()> {
    if count == 0 {
        return Err(PyValueError::new_err(
            "random_choice requires at least one choice",
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != count {
            return Err(PyValueError::new_err(
                "random_choice weights must match the number of choices",
            ));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(PyValueError::new_err(
                "random_choice weights must be finite and non-negative",
            ));
        }
        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(PyValueError::new_err(
                "random_choice weights must have a finite positive sum",
            ));
        }
    }
    Ok(())
}
