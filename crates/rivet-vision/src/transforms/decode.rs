use crate::errors::{RivetResult, invalid_argument};
use crate::sample::image::{DecodedSample, ImageSample};
use crate::transforms::from_rgb_image;

#[derive(Clone, Copy)]
pub struct DecodeImageConfig;

impl DecodeImageConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        match sample {
            ImageSample::Encoded(sample) => {
                let image = image::load_from_memory(sample.image.as_slice())?.into_rgb8();
                Ok(ImageSample::Decoded(from_rgb_image(image, sample.label)?))
            }
            ImageSample::Decoded(_) => {
                Err(invalid_argument("decode_image received a decoded sample"))
            }
        }
    }
}

pub fn decode_rgb(encoded: &[u8], label: i64) -> RivetResult<DecodedSample> {
    let image = image::load_from_memory(encoded)?.into_rgb8();
    Ok(from_rgb_image(image, label)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::image::EncodedImageSample;
    use arrow_buffer::Buffer;
    use rivet_core::DType;
    use std::io::Cursor;

    #[test]
    fn decode_reads_buffer_backed_sample() {
        let pixels = vec![10u8, 20, 30, 200, 210, 220];
        let source = image::RgbImage::from_raw(2, 1, pixels.clone()).unwrap();
        let mut cursor = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(source)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();

        let sample = ImageSample::Encoded(EncodedImageSample {
            image: Buffer::from(cursor.into_inner()),
            label: 42,
        });

        let ImageSample::Decoded(decoded) = DecodeImageConfig.apply(sample).unwrap() else {
            panic!("expected decoded sample");
        };
        assert_eq!(decoded.label, 42);
        assert_eq!(decoded.image.dtype(), DType::U8);
        assert_eq!(decoded.image.dims(), [1, 2, 3]);
        assert!(decoded.image.is_contiguous());
        assert_eq!(decoded.image.to_vec::<u8>().unwrap(), pixels);
    }
}
