use crate::errors::{RivetResult, invalid_argument};
use crate::image::from_rgb_image;
use crate::sample::{DecodedSample, ImageSample};

#[derive(Clone, Copy)]
pub struct DecodeImageConfig;

impl DecodeImageConfig {
    pub fn apply(&self, sample: ImageSample) -> RivetResult<ImageSample> {
        match sample {
            ImageSample::Encoded(sample) => {
                let image = image::load_from_memory(sample.image.as_slice())?.to_rgb8();
                Ok(ImageSample::Decoded(from_rgb_image(image, sample.label)))
            }
            ImageSample::Decoded(_) => {
                Err(invalid_argument("decode_image received a decoded sample"))
            }
        }
    }
}

pub fn decode_rgb(encoded: &[u8], label: i64) -> RivetResult<DecodedSample> {
    let image = image::load_from_memory(encoded)?.to_rgb8();
    Ok(from_rgb_image(image, label))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{EncodedImageSample, ImageBuffer, ImageDType, ImageLayout};
    use arrow_buffer::Buffer;
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
        assert_eq!((decoded.width, decoded.height, decoded.channels), (2, 1, 3));
        assert_eq!(decoded.label, 42);
        assert_eq!(decoded.layout, ImageLayout::Hwc);
        assert_eq!(decoded.image.dtype(), ImageDType::U8);
        let ImageBuffer::U8(out) = decoded.image else {
            panic!("expected u8 image");
        };
        assert_eq!(out, pixels);
    }
}
