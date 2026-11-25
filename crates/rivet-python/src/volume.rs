use crate::{error::to_py_err, numpy::tensor_to_numpy, structured::tensor_from_numpy};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rivet_vision::volume::{
    Crop3DConfig, Pad3DConfig, PadIfNeeded3DConfig, Resize3DConfig, VolumeInterpolation,
    VolumeSample,
};

#[pyclass(name = "_VolumeSample")]
pub(crate) struct PyVolumeSample {
    inner: VolumeSample,
}

#[pymethods]
impl PyVolumeSample {
    #[new]
    #[pyo3(signature=(volume,*,mask=None,keypoints=None,label=0))]
    fn new(
        volume: &Bound<'_, PyAny>,
        mask: Option<&Bound<'_, PyAny>>,
        keypoints: Option<&Bound<'_, PyAny>>,
        label: i64,
    ) -> PyResult<Self> {
        let inner = VolumeSample {
            volume: tensor_from_numpy(volume)?,
            mask: mask.map(tensor_from_numpy).transpose()?,
            keypoints: keypoints.map(tensor_from_numpy).transpose()?,
            label,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(size,interpolation="linear"))]
    fn resize3d(&self, size: [usize; 3], interpolation: &str) -> PyResult<Self> {
        let interpolation = match interpolation {
            "linear" | "trilinear" => VolumeInterpolation::Linear,
            "nearest" => VolumeInterpolation::Nearest,
            _ => {
                return Err(PyValueError::new_err(
                    "Resize3D interpolation must be linear or nearest",
                ));
            }
        };
        let inner = self
            .inner
            .clone()
            .resize(Resize3DConfig {
                size,
                interpolation,
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(padding,fill,fill_mask=0))]
    fn pad3d(&self, padding: [usize; 6], fill: Vec<f32>, fill_mask: i64) -> PyResult<Self> {
        let inner = self
            .inner
            .clone()
            .pad(&Pad3DConfig {
                padding,
                fill,
                fill_mask,
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(min_zyx,pad_divisor_zyx,position,fill,fill_mask=0,seed=0))]
    fn pad_if_needed3d(
        &self,
        min_zyx: Option<[usize; 3]>,
        pad_divisor_zyx: Option<[usize; 3]>,
        position: &str,
        fill: Vec<f32>,
        fill_mask: i64,
        seed: u64,
    ) -> PyResult<Self> {
        let random_position = match position {
            "center" => false,
            "random" => true,
            _ => {
                return Err(PyValueError::new_err(
                    "PadIfNeeded3D position must be center or random",
                ));
            }
        };
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let inner = self
            .inner
            .clone()
            .pad_if_needed(
                &PadIfNeeded3DConfig {
                    min_zyx,
                    pad_divisor_zyx,
                    random_position,
                    fill,
                    fill_mask,
                },
                &mut rng,
            )
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(size,pad_if_needed,fill,fill_mask,seed=None))]
    fn crop3d(
        &self,
        size: [usize; 3],
        pad_if_needed: bool,
        fill: Vec<f32>,
        fill_mask: i64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut rng = seed.map(rivet_vision::RandomStream::from_seed);
        let inner = self
            .inner
            .clone()
            .crop(
                &Crop3DConfig {
                    size,
                    pad_if_needed,
                    fill,
                    fill_mask,
                },
                rng.as_mut(),
            )
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("volume", tensor_to_numpy(py, self.inner.volume.clone())?)?;
        if let Some(mask) = &self.inner.mask {
            out.set_item("mask", tensor_to_numpy(py, mask.clone())?)?;
        }
        if let Some(points) = &self.inner.keypoints {
            out.set_item("keypoints", tensor_to_numpy(py, points.clone())?)?;
        }
        out.set_item("label", self.inner.label)?;
        Ok(out)
    }
}
