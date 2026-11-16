use crate::error::to_py_err;
use crate::loader::PyDataLoader;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rivet_vision::api::{
    DType, ElasticTransformConfig, ImagePipeline, InterpolationMode, PerspectiveConfig, Point2,
    RandomAffineConfig, RandomPerspectiveConfig, RotationAngle, TransformSequence,
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

fn parse_interpolation(value: &str) -> PyResult<InterpolationMode> {
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

    #[pyo3(signature = (width, height, interpolation="bilinear"))]
    fn resize(&self, width: u32, height: u32, interpolation: &str) -> PyResult<Self> {
        let interpolation = interpolation
            .parse::<InterpolationMode>()
            .map_err(to_py_err)?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .resize_with_interpolation(width, height, interpolation),
        })
    }

    fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().crop(x, y, width, height),
        }
    }

    fn center_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().center_crop(width, height),
        }
    }

    #[pyo3(signature = (padding, fill=0.0, padding_mode="constant"))]
    fn pad(&self, padding: u32, fill: f32, padding_mode: &str) -> PyResult<Self> {
        let config = rivet_vision::api::PadConfig::new(padding)
            .with_fill(fill)
            .with_mode(parse_padding_mode(padding_mode)?);
        Ok(Self {
            inner: self.inner.clone().pad_with_config(config),
        })
    }

    #[pyo3(signature = (left, top, right, bottom, fill=0.0, padding_mode="constant"))]
    fn pad_with_sides(
        &self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        fill: f32,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::PadConfig::with_sides(left, top, right, bottom)
            .with_fill(fill)
            .with_mode(parse_padding_mode(padding_mode)?);
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
    #[pyo3(signature=(angle,translate=None,scale=1.0,shear=None,interpolation="nearest",fill=0,center=None))]
    fn affine(
        &self,
        angle: f64,
        translate: Option<Vec<f64>>,
        scale: f64,
        shear: Option<Vec<f64>>,
        interpolation: &str,
        fill: u8,
        center: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut config = rivet_vision::api::AffineConfig::new(
            angle,
            parse_pair(translate, "translate", [0.0, 0.0])?,
            scale,
            parse_pair(shear, "shear", [0.0, 0.0])?,
        )
        .with_options(parse_interpolation(interpolation)?, fill);
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
    #[pyo3(signature=(degrees,interpolation="nearest",expand=false,fill=0))]
    fn random_rotation(
        &self,
        degrees: Vec<f32>,
        interpolation: &str,
        expand: bool,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::RandomRotationConfig::new(parse_pair(
            Some(degrees),
            "degrees",
            [0.0, 0.0],
        )?)
        .with_options(parse_interpolation(interpolation)?, expand, fill);
        Ok(Self {
            inner: self.inner.clone().random_rotation(config),
        })
    }
    #[pyo3(signature=(kernel_size,sigma=None))]
    fn random_gaussian_blur(&self, kernel_size: u32, sigma: Option<Vec<f32>>) -> PyResult<Self> {
        let config = rivet_vision::api::RandomGaussianBlurConfig::new(
            kernel_size,
            parse_pair(sigma, "sigma", [0.1, 2.0])?,
        );
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

    #[pyo3(signature = (width, height, padding=0))]
    fn random_crop(&self, width: u32, height: u32, padding: u32) -> Self {
        Self {
            inner: self.inner.clone().random_crop(width, height, padding),
        }
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
    fn gaussian_blur(&self, sigma: f32, kernel_size: Option<u32>) -> Self {
        let mut config = rivet_vision::api::GaussianBlurConfig::new(sigma);
        if let Some(size) = kernel_size {
            config = config.with_kernel_size(size);
        }
        Self {
            inner: self.inner.clone().gaussian_blur_with_config(config),
        }
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

    #[pyo3(signature = (probability=0.5, scale=None, ratio=None, value=0.0))]
    fn random_erasing_with_options(
        &self,
        probability: f64,
        scale: Option<Vec<f32>>,
        ratio: Option<Vec<f32>>,
        value: f32,
    ) -> PyResult<Self> {
        let scale = parse_pair(scale, "scale", [0.02, 0.33])?;
        let ratio = parse_pair(ratio, "ratio", [0.3, 3.3333333])?;
        let config = rivet_vision::api::RandomErasingConfig::new(probability)
            .with_params(scale[0], scale[1], ratio[0], ratio[1])
            .with_value(value);
        Ok(Self {
            inner: self.inner.clone().random_erasing_with_config(config),
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

    #[pyo3(signature = (angle, interpolation="nearest", expand=false, fill=0))]
    fn arbitrary_rotate(
        &self,
        angle: f32,
        interpolation: &str,
        expand: bool,
        fill: u8,
    ) -> PyResult<Self> {
        let interpolation = parse_interpolation(interpolation)?;
        Ok(Self {
            inner: self.inner.clone().arbitrary_rotate_with_options(
                angle,
                expand,
                interpolation,
                fill,
            ),
        })
    }

    #[pyo3(signature = (degrees, translate=None, scale=None, shear=None, interpolation="nearest", fill=0))]
    fn random_affine(
        &self,
        degrees: f32,
        translate: Option<Vec<f32>>,
        scale: Option<Vec<f32>>,
        shear: Option<Vec<f32>>,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let translate = parse_pair(translate, "translate", [0.0, 0.0])?;
        let scale = parse_pair(scale, "scale", [1.0, 1.0])?;
        let shear = parse_pair(shear, "shear", [0.0, 0.0])?;
        let interpolation = parse_interpolation(interpolation)?;
        let config = RandomAffineConfig::new(degrees)
            .with_translate(translate[0], translate[1])
            .with_scale(scale[0], scale[1])
            .with_shear(shear[0], shear[1])
            .with_options(interpolation, fill);
        Ok(Self {
            inner: self.inner.clone().random_affine_with_config(config),
        })
    }

    #[pyo3(signature = (start_points, end_points, interpolation="bilinear", fill=0))]
    fn perspective(
        &self,
        start_points: Vec<Vec<f32>>,
        end_points: Vec<Vec<f32>>,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let start_points = parse_points(start_points, "start_points")?;
        let end_points = parse_points(end_points, "end_points")?;
        let interpolation = parse_interpolation(interpolation)?;
        let config =
            PerspectiveConfig::new(start_points, end_points).with_options(interpolation, fill);
        Ok(Self {
            inner: self.inner.clone().perspective_with_config(config),
        })
    }

    #[pyo3(signature = (distortion_scale=0.5, probability=0.5, interpolation="bilinear", fill=0))]
    fn random_perspective(
        &self,
        distortion_scale: f32,
        probability: f64,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let interpolation = parse_interpolation(interpolation)?;
        let config = RandomPerspectiveConfig::new(distortion_scale, probability)
            .with_options(interpolation, fill);
        Ok(Self {
            inner: self.inner.clone().random_perspective_with_config(config),
        })
    }

    #[pyo3(signature = (alpha, sigma, interpolation="bilinear", fill=0))]
    fn elastic_transform(
        &self,
        alpha: f32,
        sigma: f32,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let interpolation = parse_interpolation(interpolation)?;
        let config = ElasticTransformConfig::new(alpha, sigma).with_options(interpolation, fill);
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

    fn random_choice(&self, choices: Vec<PyRef<'_, Self>>) -> Self {
        Self {
            inner: self.inner.clone().random_choice(
                choices
                    .into_iter()
                    .map(|choice| choice.inner.clone())
                    .collect(),
            ),
        }
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

    #[pyo3(signature = (width, height, interpolation="bilinear"))]
    fn resize(&self, width: u32, height: u32, interpolation: &str) -> PyResult<Self> {
        let interpolation = interpolation
            .parse::<InterpolationMode>()
            .map_err(to_py_err)?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .resize_with_interpolation(width, height, interpolation),
        })
    }

    fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().crop(x, y, width, height),
        }
    }

    fn center_crop(&self, width: u32, height: u32) -> Self {
        Self {
            inner: self.inner.clone().center_crop(width, height),
        }
    }

    #[pyo3(signature = (padding, fill=0.0, padding_mode="constant"))]
    fn pad(&self, padding: u32, fill: f32, padding_mode: &str) -> PyResult<Self> {
        let config = rivet_vision::api::PadConfig::new(padding)
            .with_fill(fill)
            .with_mode(parse_padding_mode(padding_mode)?);
        Ok(Self {
            inner: self.inner.clone().pad_with_config(config),
        })
    }

    #[pyo3(signature = (left, top, right, bottom, fill=0.0, padding_mode="constant"))]
    fn pad_with_sides(
        &self,
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
        fill: f32,
        padding_mode: &str,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::PadConfig::with_sides(left, top, right, bottom)
            .with_fill(fill)
            .with_mode(parse_padding_mode(padding_mode)?);
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
    #[pyo3(signature=(angle,translate=None,scale=1.0,shear=None,interpolation="nearest",fill=0,center=None))]
    fn affine(
        &self,
        angle: f64,
        translate: Option<Vec<f64>>,
        scale: f64,
        shear: Option<Vec<f64>>,
        interpolation: &str,
        fill: u8,
        center: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let mut config = rivet_vision::api::AffineConfig::new(
            angle,
            parse_pair(translate, "translate", [0.0, 0.0])?,
            scale,
            parse_pair(shear, "shear", [0.0, 0.0])?,
        )
        .with_options(parse_interpolation(interpolation)?, fill);
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
    #[pyo3(signature=(degrees,interpolation="nearest",expand=false,fill=0))]
    fn random_rotation(
        &self,
        degrees: Vec<f32>,
        interpolation: &str,
        expand: bool,
        fill: u8,
    ) -> PyResult<Self> {
        let config = rivet_vision::api::RandomRotationConfig::new(parse_pair(
            Some(degrees),
            "degrees",
            [0.0, 0.0],
        )?)
        .with_options(parse_interpolation(interpolation)?, expand, fill);
        Ok(Self {
            inner: self.inner.clone().random_rotation(config),
        })
    }
    #[pyo3(signature=(kernel_size,sigma=None))]
    fn random_gaussian_blur(&self, kernel_size: u32, sigma: Option<Vec<f32>>) -> PyResult<Self> {
        let config = rivet_vision::api::RandomGaussianBlurConfig::new(
            kernel_size,
            parse_pair(sigma, "sigma", [0.1, 2.0])?,
        );
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

    #[pyo3(signature = (width, height, padding=0))]
    fn random_crop(&self, width: u32, height: u32, padding: u32) -> Self {
        Self {
            inner: self.inner.clone().random_crop(width, height, padding),
        }
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

    fn random_choice(&self, choices: Vec<PyRef<'_, PyTransform>>) -> Self {
        Self {
            inner: self.inner.clone().random_choice(
                choices
                    .into_iter()
                    .map(|choice| choice.inner.clone())
                    .collect(),
            ),
        }
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
    fn gaussian_blur(&self, sigma: f32, kernel_size: Option<u32>) -> Self {
        let mut config = rivet_vision::api::GaussianBlurConfig::new(sigma);
        if let Some(size) = kernel_size {
            config = config.with_kernel_size(size);
        }
        Self {
            inner: self.inner.clone().gaussian_blur_with_config(config),
        }
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

    #[pyo3(signature = (probability=0.5, scale=None, ratio=None, value=0.0))]
    fn random_erasing_with_options(
        &self,
        probability: f64,
        scale: Option<Vec<f32>>,
        ratio: Option<Vec<f32>>,
        value: f32,
    ) -> PyResult<Self> {
        let scale = parse_pair(scale, "scale", [0.02, 0.33])?;
        let ratio = parse_pair(ratio, "ratio", [0.3, 3.3333333])?;
        let config = rivet_vision::api::RandomErasingConfig::new(probability)
            .with_params(scale[0], scale[1], ratio[0], ratio[1])
            .with_value(value);
        Ok(Self {
            inner: self.inner.clone().random_erasing_with_config(config),
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

    #[pyo3(signature = (angle, interpolation="nearest", expand=false, fill=0))]
    fn arbitrary_rotate(
        &self,
        angle: f32,
        interpolation: &str,
        expand: bool,
        fill: u8,
    ) -> PyResult<Self> {
        let interpolation = parse_interpolation(interpolation)?;
        Ok(Self {
            inner: self.inner.clone().arbitrary_rotate_with_options(
                angle,
                expand,
                interpolation,
                fill,
            ),
        })
    }

    #[pyo3(signature = (degrees, translate=None, scale=None, shear=None, interpolation="nearest", fill=0))]
    fn random_affine(
        &self,
        degrees: f32,
        translate: Option<Vec<f32>>,
        scale: Option<Vec<f32>>,
        shear: Option<Vec<f32>>,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let translate = parse_pair(translate, "translate", [0.0, 0.0])?;
        let scale = parse_pair(scale, "scale", [1.0, 1.0])?;
        let shear = parse_pair(shear, "shear", [0.0, 0.0])?;
        let interpolation = parse_interpolation(interpolation)?;
        let config = RandomAffineConfig::new(degrees)
            .with_translate(translate[0], translate[1])
            .with_scale(scale[0], scale[1])
            .with_shear(shear[0], shear[1])
            .with_options(interpolation, fill);
        Ok(Self {
            inner: self.inner.clone().random_affine_with_config(config),
        })
    }

    #[pyo3(signature = (start_points, end_points, interpolation="bilinear", fill=0))]
    fn perspective(
        &self,
        start_points: Vec<Vec<f32>>,
        end_points: Vec<Vec<f32>>,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let start_points = parse_points(start_points, "start_points")?;
        let end_points = parse_points(end_points, "end_points")?;
        let interpolation = parse_interpolation(interpolation)?;
        let config =
            PerspectiveConfig::new(start_points, end_points).with_options(interpolation, fill);
        Ok(Self {
            inner: self.inner.clone().perspective_with_config(config),
        })
    }

    #[pyo3(signature = (distortion_scale=0.5, probability=0.5, interpolation="bilinear", fill=0))]
    fn random_perspective(
        &self,
        distortion_scale: f32,
        probability: f64,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let interpolation = parse_interpolation(interpolation)?;
        let config = RandomPerspectiveConfig::new(distortion_scale, probability)
            .with_options(interpolation, fill);
        Ok(Self {
            inner: self.inner.clone().random_perspective_with_config(config),
        })
    }

    #[pyo3(signature = (alpha, sigma, interpolation="bilinear", fill=0))]
    fn elastic_transform(
        &self,
        alpha: f32,
        sigma: f32,
        interpolation: &str,
        fill: u8,
    ) -> PyResult<Self> {
        let interpolation = parse_interpolation(interpolation)?;
        let config = ElasticTransformConfig::new(alpha, sigma).with_options(interpolation, fill);
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
