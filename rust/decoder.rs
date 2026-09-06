use crate::errors::runtime_err;
use crate::sample::DecodedSample;
use pyo3::prelude::*;

pub(crate) fn decode_rgb(encoded: &[u8], label: i64) -> PyResult<DecodedSample> {
    let img = image::load_from_memory(encoded)
        .map_err(runtime_err)?
        .to_rgb8();
    let (width, height) = img.dimensions();

    Ok(DecodedSample {
        image: img.into_raw(),
        width,
        height,
        channels: 3,
        label,
    })
}
