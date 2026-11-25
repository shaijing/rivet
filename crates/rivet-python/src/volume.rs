use crate::{error::to_py_err, numpy::tensor_to_numpy, structured::tensor_from_numpy};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rivet_vision::volume::{Resize3DConfig, VolumeInterpolation, VolumeSample};

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
