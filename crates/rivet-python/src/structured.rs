use crate::dlpack::PyDLPackTensor;
use crate::error::to_py_err;
use crate::numpy::tensor_to_numpy;
use crate::pipeline::PyTransform;
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rivet_core::{Device, Tensor};
use rivet_vision::api::{
    AtLeastOneBBoxRandomCropConfig, BBoxSafeRandomCropConfig, BBoxSubsetSafeRandomCropConfig,
    BatchMixConfig, BoxFormat, ImageAxisOrder, ImageBatch, RandomIoUCropConfig, StructuredBatch,
    StructuredLoader, StructuredOp, StructuredPipeline, StructuredSample, VideoLayout, VideoSample,
};
use rivet_vision::structured::CropNonEmptyMaskConfig;

pub(crate) fn tensor_from_numpy(value: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    macro_rules! read {
        ($ty:ty) => {
            if let Ok(array) = value.cast::<PyArrayDyn<$ty>>() {
                let read = array.readonly();
                let view = read.as_array();
                return Tensor::from_exact_iter(view.iter().copied(), view.shape(), &Device::Cpu)
                    .map_err(to_py_err);
            }
        };
    }
    read!(u8);
    read!(f32);
    read!(i64);
    read!(f64);
    Err(PyValueError::new_err(
        "expected a NumPy array of dtype uint8, float32, int64, or float64",
    ))
}
fn axis(layout: &str) -> PyResult<ImageAxisOrder> {
    match layout {
        "HWC" => Ok(ImageAxisOrder::Hwc),
        "CHW" => Ok(ImageAxisOrder::Chw),
        _ => Err(PyValueError::new_err("layout must be HWC or CHW")),
    }
}
fn export(py: Python<'_>, tensor: Tensor, dlpack: bool) -> PyResult<Py<PyAny>> {
    if dlpack {
        Ok(Py::new(py, PyDLPackTensor::new(tensor))?.into_any())
    } else {
        tensor_to_numpy(py, tensor)
    }
}
fn target(py: Python<'_>, sample: &StructuredSample, dlpack: bool) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    let (h, w) = sample.size().map_err(to_py_err)?;
    out.set_item("canvas_size", [h, w])?;
    out.set_item("box_format", sample.box_format.as_str())?;
    let shapes = PyDict::new(py);
    let dtypes = PyDict::new(py);
    for (name, value) in [
        ("boxes", &sample.boxes),
        ("labels", &sample.labels),
        ("masks", &sample.masks),
        ("keypoints", &sample.keypoints),
    ] {
        if let Some(tensor) = value {
            shapes.set_item(name, tensor.dims().to_vec())?;
            dtypes.set_item(name, crate::dtype::name(tensor.dtype()))?;
            out.set_item(name, export(py, tensor.clone(), dlpack)?)?;
        }
    }
    out.set_item("shapes", shapes)?;
    out.set_item("dtypes", dtypes)?;
    Ok(out.unbind())
}
#[pyclass(name = "_AnnotatedSample")]
#[derive(Clone)]
pub(crate) struct PyAnnotatedSample {
    pub(crate) inner: StructuredSample,
}
#[pymethods]
impl PyAnnotatedSample {
    #[new]
    #[pyo3(signature=(image,boxes=None,labels=None,masks=None,keypoints=None,box_format="xyxy",layout="HWC"))]
    fn new(
        image: &Bound<'_, PyAny>,
        boxes: Option<&Bound<'_, PyAny>>,
        labels: Option<&Bound<'_, PyAny>>,
        masks: Option<&Bound<'_, PyAny>>,
        keypoints: Option<&Bound<'_, PyAny>>,
        box_format: &str,
        layout: &str,
    ) -> PyResult<Self> {
        let inner = StructuredSample {
            image: tensor_from_numpy(image)?,
            axis_order: axis(layout)?,
            boxes: boxes.map(tensor_from_numpy).transpose()?,
            labels: labels.map(tensor_from_numpy).transpose()?,
            masks: masks.map(tensor_from_numpy).transpose()?,
            keypoints: keypoints.map(tensor_from_numpy).transpose()?,
            box_format: box_format.parse::<BoxFormat>().map_err(to_py_err)?,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }
    #[pyo3(signature=(dlpack=false))]
    fn to_dict(&self, py: Python<'_>, dlpack: bool) -> PyResult<Py<PyDict>> {
        let out = target(py, &self.inner, dlpack)?;
        out.bind(py)
            .set_item("image", export(py, self.inner.image.clone(), dlpack)?)?;
        out.bind(py)
            .set_item("layout", self.inner.axis_order.as_str())?;
        Ok(out)
    }
    #[pyo3(signature=(transform,seed=0,epoch=0,sample_index=0))]
    fn apply(
        &self,
        py: Python<'_>,
        transform: &PyTransform,
        seed: u64,
        epoch: u64,
        sample_index: usize,
    ) -> PyResult<Self> {
        let mut p = StructuredPipeline::new(Vec::new())
            .map_err(to_py_err)?
            .compose(transform.inner.clone());
        p.seed = seed;
        p.epoch = epoch;
        for op in &p.ops {
            op.validate().map_err(to_py_err)?;
        }
        let sample = self.inner.clone();
        Ok(Self {
            inner: py
                .detach(move || p.apply(sample, sample_index))
                .map_err(to_py_err)?,
        })
    }
}
#[pyclass(name = "_StructuredPipeline")]
#[derive(Clone)]
pub(crate) struct PyStructuredPipeline {
    inner: StructuredPipeline,
}
#[pymethods]
impl PyStructuredPipeline {
    #[new]
    fn new(samples: Vec<PyRef<'_, PyAnnotatedSample>>) -> PyResult<Self> {
        Ok(Self {
            inner: StructuredPipeline::new(samples.into_iter().map(|s| s.inner.clone()).collect())
                .map_err(to_py_err)?,
        })
    }
    fn compose(&self, transform: &PyTransform) -> Self {
        Self {
            inner: self.inner.clone().compose(transform.inner.clone()),
        }
    }
    fn convert_bounding_box_format(&self, format: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().push(StructuredOp::ConvertBoxes(
                format.parse().map_err(to_py_err)?,
            )),
        })
    }
    fn clamp_bounding_boxes(&self) -> Self {
        Self {
            inner: self.inner.clone().push(StructuredOp::ClampBoxes),
        }
    }
    fn clamp_keypoints(&self) -> Self {
        Self {
            inner: self.inner.clone().push(StructuredOp::ClampKeypoints),
        }
    }
    fn sanitize_keypoints(&self) -> Self {
        Self {
            inner: self.inner.clone().push(StructuredOp::SanitizeKeypoints),
        }
    }
    #[pyo3(signature=(min_size=1.0,min_area=1.0))]
    fn sanitize_bounding_boxes(&self, min_size: f32, min_area: f32) -> PyResult<Self> {
        let op = StructuredOp::Sanitize { min_size, min_area };
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }
    #[pyo3(signature = (subset_fraction_range=(0.5,1.0), erosion_rate=0.0, aspect_ratio_range=(0.5,2.0)))]
    fn bbox_subset_safe_random_crop(
        &self,
        subset_fraction_range: (f64, f64),
        erosion_rate: f64,
        aspect_ratio_range: (f64, f64),
    ) -> PyResult<Self> {
        let op = StructuredOp::BBoxSubsetSafeRandomCrop(BBoxSubsetSafeRandomCropConfig {
            subset_fraction_range,
            erosion_rate,
            aspect_ratio_range,
        });
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }

    #[pyo3(signature = (width, height, *, ignore_values=Vec::new(), ignore_channels=Vec::new()))]
    fn crop_non_empty_mask_if_exists(
        &self,
        width: u32,
        height: u32,
        ignore_values: Vec<i64>,
        ignore_channels: Vec<usize>,
    ) -> PyResult<Self> {
        let op = StructuredOp::CropNonEmptyMask(CropNonEmptyMaskConfig {
            width,
            height,
            ignore_values,
            ignore_channels,
        });
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }

    #[pyo3(signature = (max_part_shift=(0.0, 0.3), *, box_index=0, reference_box=None))]
    fn random_crop_near_bbox(
        &self,
        max_part_shift: (f64, f64),
        box_index: usize,
        reference_box: Option<(f64, f64, f64, f64)>,
    ) -> PyResult<Self> {
        let op =
            StructuredOp::RandomCropNearBBox(rivet_vision::structured::RandomCropNearBBoxConfig {
                max_part_shift,
                box_index,
                reference_box: reference_box.map(|(l, t, r, b)| [l, t, r, b]),
            });
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }

    #[pyo3(signature = (erosion_rate=0.0))]
    fn bbox_safe_random_crop(&self, erosion_rate: f64) -> PyResult<Self> {
        let op = StructuredOp::BBoxSafeRandomCrop(BBoxSafeRandomCropConfig {
            erosion_rate,
            output: None,
        });
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }

    #[pyo3(signature = (width, height, *, erosion_rate=0.0, interpolation="bilinear"))]
    fn random_sized_bbox_safe_crop(
        &self,
        width: u32,
        height: u32,
        erosion_rate: f64,
        interpolation: &str,
    ) -> PyResult<Self> {
        let interpolation = crate::pipeline::parse_interpolation(interpolation)?;
        let op = StructuredOp::BBoxSafeRandomCrop(BBoxSafeRandomCropConfig {
            erosion_rate,
            output: Some((width, height, interpolation)),
        });
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }

    #[pyo3(signature = (width, height, *, erosion_factor=0.0))]
    fn at_least_one_bbox_random_crop(
        &self,
        width: u32,
        height: u32,
        erosion_factor: f64,
    ) -> PyResult<Self> {
        let op = StructuredOp::AtLeastOneBBoxRandomCrop(AtLeastOneBBoxRandomCropConfig {
            width,
            height,
            erosion_factor,
        });
        op.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(op),
        })
    }

    #[pyo3(signature=(min_scale=0.3,max_scale=1.0,min_aspect_ratio=0.5,max_aspect_ratio=2.0,sampler_options=None,trials=40,max_attempts=100))]
    fn random_iou_crop(
        &self,
        min_scale: f64,
        max_scale: f64,
        min_aspect_ratio: f64,
        max_aspect_ratio: f64,
        sampler_options: Option<Vec<f64>>,
        trials: usize,
        max_attempts: usize,
    ) -> PyResult<Self> {
        let mut c = RandomIoUCropConfig {
            min_scale,
            max_scale,
            min_aspect: min_aspect_ratio,
            max_aspect: max_aspect_ratio,
            trials,
            max_attempts,
            ..Default::default()
        };
        if let Some(options) = sampler_options {
            c.options = options;
        }
        c.validate().map_err(to_py_err)?;
        Ok(Self {
            inner: self.inner.clone().push(StructuredOp::RandomIoUCrop(c)),
        })
    }
    fn seed(&self, seed: u64) -> Self {
        let mut out = self.clone();
        out.inner.seed = seed;
        out
    }
    fn epoch(&self, epoch: u64) -> Self {
        let mut out = self.clone();
        out.inner.epoch = epoch;
        out
    }
    fn workers(&self, workers: usize) -> PyResult<Self> {
        if workers > 256 {
            return Err(PyValueError::new_err("workers must be <=256"));
        }
        let mut out = self.clone();
        out.inner.workers = workers;
        Ok(out)
    }
    fn shuffle(&self) -> Self {
        let mut out = self.clone();
        out.inner.shuffle = true;
        out
    }
    #[pyo3(signature=(size,drop_last=false))]
    fn batch(&self, size: usize, drop_last: bool) -> PyResult<Self> {
        if size == 0 {
            return Err(PyValueError::new_err("batch size must be positive"));
        }
        let mut out = self.clone();
        out.inner.batch_size = size;
        out.inner.drop_last = drop_last;
        Ok(out)
    }
    fn execute(&self) -> PyResult<PyStructuredLoader> {
        Ok(PyStructuredLoader {
            inner: self.inner.clone().compile().map_err(to_py_err)?,
        })
    }
}
#[pyclass(name = "_StructuredLoader")]
pub(crate) struct PyStructuredLoader {
    inner: StructuredLoader,
}
fn batch_dict(py: Python<'_>, batch: StructuredBatch, dlpack: bool) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("shape", batch.images.dims().to_vec())?;
    out.set_item("dtype", crate::dtype::name(batch.images.dtype()))?;
    out.set_item("layout", batch.samples[0].axis_order.batch_as_str())?;
    out.set_item("images", export(py, batch.images, dlpack)?)?;
    let targets = batch
        .samples
        .iter()
        .map(|s| target(py, s, dlpack))
        .collect::<PyResult<Vec<_>>>()?;
    out.set_item("targets", targets)?;
    Ok(out.unbind())
}
#[pymethods]
impl PyStructuredLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        py.detach(|| self.inner.next_batch())
            .map_err(to_py_err)?
            .map(|b| batch_dict(py, b, false))
            .transpose()
    }
    fn next_dlpack(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        py.detach(|| self.inner.next_batch())
            .map_err(to_py_err)?
            .map(|b| batch_dict(py, b, true))
            .transpose()
    }
}
#[pyfunction]
#[pyo3(signature=(images,labels,num_classes=None,alpha=1.0,cutmix=false,layout="NHWC",seed=0,epoch=0,batch_index=0,dlpack=false))]
pub(crate) fn mix_batch(
    py: Python<'_>,
    images: &Bound<'_, PyAny>,
    labels: &Bound<'_, PyAny>,
    num_classes: Option<usize>,
    alpha: f64,
    cutmix: bool,
    layout: &str,
    seed: u64,
    epoch: u64,
    batch_index: usize,
    dlpack: bool,
) -> PyResult<Py<PyDict>> {
    let axis_order = match layout {
        "NHWC" => ImageAxisOrder::Hwc,
        "NCHW" => ImageAxisOrder::Chw,
        _ => return Err(PyValueError::new_err("layout must be NHWC or NCHW")),
    };
    let batch = ImageBatch {
        images: tensor_from_numpy(images)?,
        labels: tensor_from_numpy(labels)?,
        axis_order,
    };
    let config = BatchMixConfig {
        alpha,
        num_classes,
        cutmix,
    };
    config.validate().map_err(to_py_err)?;
    let batch = py
        .detach(move || {
            let mut rng = rivet_vision::api::RandomContext::new(seed)
                .with_epoch(epoch)
                .stream(
                    rivet_vision::api::SampleKey::from_index(batch_index),
                    rivet_vision::api::OpKey::from_parts(config.name(), 0),
                );
            config.apply(batch, &mut rng)
        })
        .map_err(to_py_err)?;
    if dlpack {
        crate::loader::image_batch_to_dlpack(py, batch)
    } else {
        crate::loader::image_batch_to_py(py, batch)
    }
}
#[pyclass(name = "_VideoSample")]
#[derive(Clone)]
pub(crate) struct PyVideoSample {
    inner: VideoSample,
}
#[pymethods]
impl PyVideoSample {
    #[new]
    #[pyo3(signature=(frames,timestamps=None,label=0,layout="THWC"))]
    fn new(
        frames: &Bound<'_, PyAny>,
        timestamps: Option<&Bound<'_, PyAny>>,
        label: i64,
        layout: &str,
    ) -> PyResult<Self> {
        let layout = match layout {
            "THWC" => VideoLayout::Thwc,
            "TCHW" => VideoLayout::Tchw,
            _ => return Err(PyValueError::new_err("video layout must be THWC or TCHW")),
        };
        let inner = VideoSample {
            frames: tensor_from_numpy(frames)?,
            timestamps: timestamps.map(tensor_from_numpy).transpose()?,
            label,
            layout,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }
    fn uniform_temporal_subsample(&self, py: Python<'_>, num_samples: usize) -> PyResult<Self> {
        let video = self.inner.clone();
        Ok(Self {
            inner: py
                .detach(move || video.uniform_temporal_subsample(num_samples))
                .map_err(to_py_err)?,
        })
    }
    #[pyo3(signature=(dlpack=false))]
    fn to_dict(&self, py: Python<'_>, dlpack: bool) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        out.set_item("frames", export(py, self.inner.frames.clone(), dlpack)?)?;
        out.set_item("label", self.inner.label)?;
        out.set_item("layout", self.inner.layout.as_str())?;
        if let Some(t) = &self.inner.timestamps {
            out.set_item("timestamps", export(py, t.clone(), dlpack)?)?;
        }
        Ok(out.unbind())
    }
}
#[pyfunction]
#[pyo3(signature=(video,num_samples,temporal_dim=-4,dlpack=false))]
pub(crate) fn temporal_subsample(
    py: Python<'_>,
    video: &Bound<'_, PyAny>,
    num_samples: usize,
    temporal_dim: isize,
    dlpack: bool,
) -> PyResult<Py<PyAny>> {
    let tensor = tensor_from_numpy(video)?;
    let rank = tensor.rank() as isize;
    let axis = if temporal_dim < 0 {
        rank + temporal_dim
    } else {
        temporal_dim
    };
    if axis < 0 || axis >= rank {
        return Err(PyValueError::new_err("temporal_dim outside video rank"));
    }
    let out = py
        .detach(move || {
            rivet_vision::api::uniform_temporal_subsample(&tensor, num_samples, axis as usize)
        })
        .map_err(to_py_err)?;
    export(py, out, dlpack)
}
