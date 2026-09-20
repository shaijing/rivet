use crate::error::to_py_err;
use crate::loader::PyDataLoader;
use pyo3::prelude::*;
use rivet_vision::api::{
    DType, ImagePipeline, InterpolationMode, RotationAngle, TransformSequence,
};

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

    #[pyo3(signature = (padding, fill=0.0))]
    fn pad(&self, padding: u32, fill: f32) -> Self {
        Self {
            inner: self.inner.clone().pad_with_fill(padding, fill),
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

    #[pyo3(signature = (probability=0.5))]
    fn random_horizontal_flip(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_horizontal_flip(probability),
        }
    }

    fn brightness(&self, value: i32) -> Self {
        Self {
            inner: self.inner.clone().brightness(value),
        }
    }

    fn contrast(&self, value: f32) -> Self {
        Self {
            inner: self.inner.clone().contrast(value),
        }
    }

    #[pyo3(signature = (brightness=0, contrast=0.0, hue=0))]
    fn color_jitter(&self, brightness: i32, contrast: f32, hue: i32) -> Self {
        Self {
            inner: self.inner.clone().color_jitter(brightness, contrast, hue),
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

    fn gaussian_blur(&self, sigma: f32) -> Self {
        Self {
            inner: self.inner.clone().gaussian_blur(sigma),
        }
    }

    fn hue(&self, degrees: i32) -> Self {
        Self {
            inner: self.inner.clone().hue(degrees),
        }
    }

    #[pyo3(signature = (num_output_channels=1))]
    fn grayscale(&self, num_output_channels: u8) -> Self {
        Self {
            inner: self.inner.clone().grayscale(num_output_channels),
        }
    }

    #[pyo3(signature = (probability=0.1, num_output_channels=1))]
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

    #[pyo3(signature = (padding, fill=0.0))]
    fn pad(&self, padding: u32, fill: f32) -> Self {
        Self {
            inner: self.inner.clone().pad_with_fill(padding, fill),
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

    #[pyo3(signature = (probability=0.5))]
    fn random_horizontal_flip(&self, probability: f64) -> Self {
        Self {
            inner: self.inner.clone().random_horizontal_flip(probability),
        }
    }

    fn brightness(&self, value: i32) -> Self {
        Self {
            inner: self.inner.clone().brightness(value),
        }
    }

    fn contrast(&self, value: f32) -> Self {
        Self {
            inner: self.inner.clone().contrast(value),
        }
    }

    #[pyo3(signature = (brightness=0, contrast=0.0, hue=0))]
    fn color_jitter(&self, brightness: i32, contrast: f32, hue: i32) -> Self {
        Self {
            inner: self.inner.clone().color_jitter(brightness, contrast, hue),
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

    fn gaussian_blur(&self, sigma: f32) -> Self {
        Self {
            inner: self.inner.clone().gaussian_blur(sigma),
        }
    }

    fn hue(&self, degrees: i32) -> Self {
        Self {
            inner: self.inner.clone().hue(degrees),
        }
    }

    #[pyo3(signature = (num_output_channels=1))]
    fn grayscale(&self, num_output_channels: u8) -> Self {
        Self {
            inner: self.inner.clone().grayscale(num_output_channels),
        }
    }

    #[pyo3(signature = (probability=0.1, num_output_channels=1))]
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

    fn shuffle(&self, seed: u64) -> Self {
        Self {
            inner: self.inner.clone().shuffle(seed),
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

    fn execute(&self) -> PyResult<PyDataLoader> {
        Ok(PyDataLoader {
            inner: self.inner.clone().compile().map_err(to_py_err)?,
        })
    }
}
