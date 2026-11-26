//! libjpeg-v6-compatible baseline JPEG, with RGB 4:2:0 chroma sampling.
use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::ImageAxisOrder;
use crate::transforms::contiguous_values;
use mozjpeg::{ColorSpace, Compress, Decompress};
use rivet_core::{CpuStorageRef, DType, Tensor};

pub(in crate::transforms) fn jpeg(
    input: &Tensor,
    axis: ImageAxisOrder,
    quality: u8,
) -> RivetResult<Tensor> {
    if input.dtype() != DType::U8 || input.rank() != 3 || !(1..=100).contains(&quality) {
        return Err(invalid_argument(
            "JPEG requires rank-3 uint8 pixels and quality 1..100",
        ));
    }
    let dims = input.dims();
    let (h, w, c) = match axis {
        ImageAxisOrder::Hwc => (dims[0], dims[1], dims[2]),
        ImageAxisOrder::Chw => (dims[1], dims[2], dims[0]),
    };
    if !matches!(c, 1 | 3) || h == 0 || w == 0 || h > 65500 || w > 65500 {
        return Err(invalid_argument(
            "JPEG requires nonempty grayscale or RGB images, axes <=65500",
        ));
    }
    // mozjpeg's error manager unwinds on codec errors. Convert these into a
    // regular pipeline error at this boundary, including allocation failures.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        jpeg_inner(input, axis, quality, h, w, c)
    }))
    .map_err(|_| invalid_argument("JPEG codec failed"))?
}

fn jpeg_inner(
    input: &Tensor,
    axis: ImageAxisOrder,
    quality: u8,
    h: usize,
    w: usize,
    c: usize,
) -> RivetResult<Tensor> {
    let mut encoder = Compress::new(if c == 1 {
        ColorSpace::JCS_GRAYSCALE
    } else {
        ColorSpace::JCS_RGB
    });
    // Reset MozJPEG's compression optimizations to conventional libjpeg-v6
    // settings, as used by torchvision/Pillow. This must precede quality.
    encoder.set_fastest_defaults();
    encoder.set_size(w, h);
    encoder.set_quality(f32::from(quality));
    // The wrapper's set_quality allows 16-bit quantization values. Pillow and
    // torchvision force baseline (8-bit) tables. For low qualities install
    // the same standard tables with the integer IJG quality scale, clamped
    // to 255 by QTable::scaled / jpeg_add_quant_table(force_baseline=true).
    if quality < 25 {
        let scale = 5000 / u32::from(quality);
        // Bias by less than 0.002 of a table unit so f32 evaluation
        // rounds exact x.5 ties upward, as IJG's integer formula does.
        let equivalent_quality = 5000.0 / (scale as f32 + 0.001);
        encoder.set_luma_qtable(
            &mozjpeg::qtable::AnnexK_Luma.scaled(equivalent_quality, equivalent_quality),
        );
        encoder.set_chroma_qtable(
            &mozjpeg::qtable::AnnexK_Chroma.scaled(equivalent_quality, equivalent_quality),
        );
    }
    encoder.set_optimize_coding(false);
    if c == 3 {
        encoder.set_chroma_sampling_pixel_sizes((2, 2), (2, 2));
    }
    let mut encoder = encoder
        .start_compress(Vec::new())
        .map_err(invalid_argument)?;
    let mut io_error = None;
    input.with_cpu_storage(|storage, layout| {
        let CpuStorageRef::U8(values) = storage else {
            unreachable!()
        };
        if axis == ImageAxisOrder::Hwc && layout.is_contiguous() {
            if let Err(error) = encoder.write_scanlines(contiguous_values(values, layout)?) {
                io_error = Some(error);
            }
        } else {
            let mut row = vec![0u8; w * c];
            let stride = layout.stride();
            for y in 0..h {
                for x in 0..w {
                    for channel in 0..c {
                        let offset = layout.start_offset()
                            + match axis {
                                ImageAxisOrder::Hwc => {
                                    y * stride[0] + x * stride[1] + channel * stride[2]
                                }
                                ImageAxisOrder::Chw => {
                                    channel * stride[0] + y * stride[1] + x * stride[2]
                                }
                            };
                        row[x * c + channel] = *values
                            .get(offset)
                            .ok_or(rivet_core::Error::StorageOutOfBounds)?;
                    }
                }
                if let Err(error) = encoder.write_scanlines(&row) {
                    io_error = Some(error);
                    break;
                }
            }
        }
        Ok(())
    })?;
    if let Some(error) = io_error {
        return Err(invalid_argument(error));
    }
    let bytes = encoder.finish().map_err(invalid_argument)?;
    let decoder = Decompress::new_mem(&bytes).map_err(invalid_argument)?;
    let mut decoder = if c == 1 {
        decoder.grayscale()
    } else {
        decoder.rgb()
    }
    .map_err(invalid_argument)?;
    // Decode one scanline at a time into the final aligned tensor allocation;
    // only the codec row buffer and compressed bytes are temporary.
    let mut row = vec![0u8; w * c];
    let result = Tensor::from_exact_writer::<u8, _, _>([h, w, c], input.device(), |output| {
        for _ in 0..h {
            if let Err(error) = decoder.read_scanlines_into(&mut row) {
                io_error = Some(error);
                return Err(rivet_core::Error::StorageOutOfBounds);
            }
            output.extend_from_slice(&row)?;
        }
        Ok(())
    });
    if let Some(error) = io_error {
        return Err(invalid_argument(error));
    }
    let result = result?;
    decoder.finish().map_err(invalid_argument)?;
    Ok(match axis {
        ImageAxisOrder::Hwc => result,
        ImageAxisOrder::Chw => result.permute(&[2, 0, 1])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rivet_core::Device;
    #[test]
    fn jpeg_preserves_gray_and_chw_strided_contract() {
        let base = Tensor::from_vec(
            (0..120).map(|i| (i * 17) as u8).collect(),
            [5, 8, 3],
            &Device::Cpu,
        )
        .unwrap();
        let image = base.narrow(1, 1, 6).unwrap();
        let hwc = jpeg(&image, ImageAxisOrder::Hwc, 80).unwrap();
        let chw = jpeg(&image.permute(&[2, 0, 1]).unwrap(), ImageAxisOrder::Chw, 80).unwrap();
        assert_eq!(
            hwc.to_vec::<u8>().unwrap(),
            chw.permute(&[1, 2, 0]).unwrap().to_vec::<u8>().unwrap()
        );
        let gray = Tensor::from_vec(vec![127u8; 12], [3, 4, 1], &Device::Cpu).unwrap();
        assert_eq!(
            jpeg(&gray, ImageAxisOrder::Hwc, 100)
                .unwrap()
                .to_vec::<u8>()
                .unwrap(),
            vec![127; 12]
        );
    }
}
