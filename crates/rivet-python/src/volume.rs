use crate::{error::to_py_err, numpy::tensor_to_numpy, structured::tensor_from_numpy};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rivet_vision::volume::{
    Affine3DConfig, Anisotropy3DConfig, CoarseDropout3DConfig, Crop3DConfig,
    ElasticTransform3DConfig, Pad3DConfig, PadIfNeeded3DConfig, Resize3DConfig, VolumeBorder,
    VolumeInterpolation, VolumeSample, volume_axis_mask,
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

    #[pyo3(signature=(axes, flip_axes, seed))]
    fn flip3d(&self, axes: Vec<usize>, flip_axes: Option<Vec<usize>>, seed: u64) -> PyResult<Self> {
        let allowed = volume_axis_mask(&axes).map_err(to_py_err)?;
        if allowed == 0 {
            return Err(PyValueError::new_err(
                "Flip3D requires at least one allowed axis",
            ));
        }
        let reflected = if let Some(fixed) = flip_axes {
            if volume_axis_mask(&fixed).map_err(to_py_err)? & !allowed != 0 {
                return Err(PyValueError::new_err("flip_axes must be a subset of axes"));
            }
            fixed
        } else {
            let bits = rivet_vision::RandomStream::from_seed(seed).next_u64() as u8 & allowed;
            (0..3).filter(|axis| bits & (1 << axis) != 0).collect()
        };
        let inner = self
            .inner
            .clone()
            .reorient([0, 1, 2], &reflected)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(axis_pairs, axis_pair, group_element, seed))]
    fn random_rotate90_3d(
        &self,
        axis_pairs: Vec<[usize; 2]>,
        axis_pair: Option<[usize; 2]>,
        group_element: Option<&str>,
        seed: u64,
    ) -> PyResult<Self> {
        let valid_pair = |[a, b]: [usize; 2]| a < b && b < 3;
        if axis_pairs.is_empty()
            || axis_pairs
                .iter()
                .enumerate()
                .any(|(i, &pair)| !valid_pair(pair) || axis_pairs[..i].contains(&pair))
            || axis_pair.is_some_and(|pair| !valid_pair(pair))
        {
            return Err(PyValueError::new_err(
                "expected distinct ascending DHW axis pairs",
            ));
        }
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let pair = match axis_pair {
            Some(pair) => pair,
            None => axis_pairs[rng.gen_range_usize(0..axis_pairs.len()).unwrap()],
        };
        let turns = match group_element {
            None => (rng.next_u64() & 3) as u8,
            Some("e") => 0,
            Some("r90") => 1,
            Some("r180") => 2,
            Some("r270") => 3,
            _ => {
                return Err(PyValueError::new_err(
                    "group_element must be e, r90, r180 or r270",
                ));
            }
        };
        let inner = self
            .inner
            .clone()
            .rotate90(pair, turns)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(element, seed))]
    fn cubic_symmetry(&self, element: Option<usize>, seed: u64) -> PyResult<Self> {
        let element = element.unwrap_or_else(|| {
            rivet_vision::RandomStream::from_seed(seed)
                .gen_range_usize(0..48)
                .unwrap()
        });
        let inner = self
            .inner
            .clone()
            .cubic_symmetry(element)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn grid_shuffle3d(&self, grid_zyx: [usize; 3], seed: u64) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let inner = self
            .inner
            .clone()
            .grid_shuffle(grid_zyx, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[pyo3(signature=(num_holes_range, hole_ranges, fill, fill_mask, remove_invisible, seed))]
    fn coarse_dropout3d(
        &self,
        num_holes_range: [usize; 2],
        hole_ranges: [[f64; 2]; 3],
        fill: Vec<f32>,
        fill_mask: Option<i64>,
        remove_invisible: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let config = CoarseDropout3DConfig {
            num_holes_range,
            hole_ranges,
            fill,
            fill_mask,
            remove_invisible,
        };
        let inner = self
            .inner
            .clone()
            .coarse_dropout(&config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn anisotropy3d(
        &self,
        axes: Vec<usize>,
        num_axes_range: [usize; 2],
        downscale_factor_range: [f64; 2],
        antialias: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let config = Anisotropy3DConfig {
            axes,
            num_axes_range,
            downscale_factor_range,
            antialias,
        };
        let inner = self
            .inner
            .clone()
            .anisotropy(&config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn affine3d(
        &self,
        rotate_range: [[f64; 2]; 3],
        scale_range: [[f64; 2]; 3],
        translate_percent_range: [[f64; 2]; 3],
        interpolation: &str,
        border_mode: &str,
        fill: Vec<f32>,
        fill_mask: i64,
        seed: u64,
    ) -> PyResult<Self> {
        let (interpolation, border) = warp_modes(interpolation, border_mode)?;
        let config = Affine3DConfig {
            rotate_range,
            scale_range,
            translate_percent_range,
            interpolation,
            border,
            fill,
            fill_mask,
        };
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let inner = self
            .inner
            .clone()
            .affine(&config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn elastic_transform3d(
        &self,
        displacement_range: [f64; 2],
        control_grid_shape: [usize; 2],
        interpolation: &str,
        border_mode: &str,
        fill: Vec<f32>,
        fill_mask: i64,
        seed: u64,
    ) -> PyResult<Self> {
        let (interpolation, border) = warp_modes(interpolation, border_mode)?;
        let config = ElasticTransform3DConfig {
            displacement_range,
            control_grid_shape,
            interpolation,
            border,
            fill,
            fill_mask,
        };
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let inner = self
            .inner
            .clone()
            .elastic_transform(&config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn ghosting_artifact(
        &self,
        num_ghosts_range: [usize; 2],
        intensity_range: [f64; 2],
        axis: usize,
        restore_range: [f64; 2],
        seed: u64,
    ) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let config = rivet_vision::volume::GhostingArtifactConfig {
            num_ghosts_range,
            intensity_range,
            axis,
            restore_range,
        };
        let inner = self
            .inner
            .clone()
            .ghosting_artifact(config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn gibbs_ringing(&self, retained_fraction_range: [f64; 2], seed: u64) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let config = rivet_vision::volume::GibbsRingingConfig {
            retained_fraction_range,
        };
        let inner = self
            .inner
            .clone()
            .gibbs_ringing(config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn bias_field(
        &self,
        std_range: [f64; 2],
        scale_range: [f64; 2],
        per_channel: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let config = rivet_vision::volume::BiasFieldConfig {
            std_range,
            scale_range,
            per_channel,
        };
        let inner = self
            .inner
            .clone()
            .bias_field(config, &mut rng)
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn rician_noise(&self, std_range: [f64; 2], per_channel: bool, seed: u64) -> PyResult<Self> {
        let mut rng = rivet_vision::RandomStream::from_seed(seed);
        let inner = self
            .inner
            .clone()
            .rician_noise(std_range, per_channel, &mut rng)
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

fn warp_modes(
    interpolation: &str,
    border_mode: &str,
) -> PyResult<(VolumeInterpolation, VolumeBorder)> {
    let interpolation = match interpolation {
        "nearest" => VolumeInterpolation::Nearest,
        "linear" | "trilinear" => VolumeInterpolation::Linear,
        _ => {
            return Err(PyValueError::new_err(
                "3D warp interpolation must be linear or nearest",
            ));
        }
    };
    let border = match border_mode {
        "constant" => VolumeBorder::Constant,
        "replicate" => VolumeBorder::Replicate,
        _ => {
            return Err(PyValueError::new_err(
                "3D warp border mode must be constant or replicate",
            ));
        }
    };
    Ok((interpolation, border))
}
