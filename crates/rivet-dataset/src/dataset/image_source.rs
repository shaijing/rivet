//! State-aware image sources used by the image pipeline.

use super::cache::{CachePolicy, materialize_decoded_to_memory};
use super::source::{Dataset, Source};
use crate::errors::{RivetResult, invalid_argument};
use crate::pipeline::op::PipelineImageState;
use crate::sample::image::ImageLayout;
use crate::sample::image::{DecodedSample, EncodedImageSample, ImageSample};
use rivet_core::DType;
use std::sync::Arc;

/// A typed image source that keeps the pipeline's initial representation
/// alongside its backend. Persistent and encoded-memory sources stay encoded;
/// decoded caches expose decoded U8 HWC samples directly.
#[derive(Clone)]
pub enum ImageSource {
    Encoded(Source<EncodedImageSample>),
    Decoded(Source<DecodedSample>),
}

impl ImageSource {
    pub fn from_encoded<D>(dataset: Arc<D>) -> Self
    where
        D: Dataset<Item = EncodedImageSample> + 'static,
    {
        Self::Encoded(Source::new(dataset))
    }

    pub fn from_encoded_source(source: Source<EncodedImageSample>) -> Self {
        Self::Encoded(source)
    }

    pub fn from_decoded<D>(dataset: Arc<D>) -> Self
    where
        D: Dataset<Item = DecodedSample> + 'static,
    {
        Self::Decoded(Source::new(dataset))
    }

    pub fn from_decoded_source(source: Source<DecodedSample>) -> Self {
        Self::Decoded(source)
    }

    pub fn state(&self) -> PipelineImageState {
        match self {
            Self::Encoded(_) => PipelineImageState::Encoded,
            Self::Decoded(_) => PipelineImageState::Decoded {
                dtype: DType::U8,
                layout: ImageLayout::Hwc,
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Encoded(source) => source.len(),
            Self::Decoded(source) => source.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> RivetResult<ImageSample> {
        let mut samples = self.get_many(&[index])?;
        if samples.len() != 1 {
            return Err(invalid_argument(format!(
                "image source returned {} rows for one requested index",
                samples.len()
            )));
        }
        Ok(samples.remove(0))
    }

    pub fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<ImageSample>> {
        match self {
            Self::Encoded(source) => source
                .get_many(indices)
                .map(|samples| samples.into_iter().map(ImageSample::Encoded).collect()),
            Self::Decoded(source) => source
                .get_many(indices)
                .map(|samples| samples.into_iter().map(ImageSample::Decoded).collect()),
        }
    }

    /// Materialize this source at the requested representation level.
    ///
    /// Decoded caching is only valid for encoded sources. Re-caching an
    /// already decoded source is a cheap clone of the source handle.
    pub fn cache(&self, policy: CachePolicy) -> RivetResult<Self> {
        match policy {
            CachePolicy::None => Ok(self.clone()),
            CachePolicy::Encoded { chunk_size } => match self {
                Self::Encoded(source) => Ok(Self::Encoded(source.cache_encoded(chunk_size)?)),
                Self::Decoded(_) => Err(invalid_argument(
                    "encoded cache requires an encoded image source",
                )),
            },
            CachePolicy::Decoded {
                chunk_size,
                max_bytes,
            } => match self {
                Self::Encoded(source) => Ok(Self::Decoded(Source::new(Arc::new(
                    materialize_decoded_to_memory(source.as_dataset(), chunk_size, max_bytes)?,
                )))),
                Self::Decoded(_) => Ok(self.clone()),
            },
        }
    }

    pub fn cache_encoded(&self, chunk_size: usize) -> RivetResult<Self> {
        self.cache(CachePolicy::Encoded { chunk_size })
    }

    pub fn cache_decoded(&self, chunk_size: usize, max_bytes: Option<usize>) -> RivetResult<Self> {
        self.cache(CachePolicy::Decoded {
            chunk_size,
            max_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::image::{EncodedImageSample, ImageSample};
    use arrow_buffer::Buffer;
    use rivet_core::DType;
    use std::sync::Arc;

    const PNG_1X1: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90\x77\x53\xde\x00\x00\x00\x0c\x49\x44\x41\x54\x78\x9c\x63\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82";

    struct EncodedStub;

    impl Dataset for EncodedStub {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            1
        }

        fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
            indices
                .iter()
                .map(|&index| {
                    if index != 0 {
                        return Err(crate::errors::RivetError::IndexOutOfRange { index, len: 1 });
                    }
                    Ok(EncodedImageSample {
                        image: Buffer::from(PNG_1X1.to_vec()),
                        label: 4,
                    })
                })
                .collect()
        }
    }

    #[test]
    fn decoded_cache_exposes_decoded_state_and_shared_pixels() {
        let source = ImageSource::from_encoded(Arc::new(EncodedStub));
        let cached = source.cache_decoded(1, None).unwrap();

        assert_eq!(
            cached.state(),
            PipelineImageState::Decoded {
                dtype: DType::U8,
                layout: crate::sample::image::ImageLayout::Hwc,
            }
        );
        let samples = cached.get_many(&[0, 0]).unwrap();
        let [ImageSample::Decoded(first), ImageSample::Decoded(second)] = samples.as_slice() else {
            panic!("expected decoded samples");
        };
        assert!(first.image.same_storage(&second.image));
        assert_eq!(first.image.to_vec::<u8>().unwrap(), [255, 0, 0]);
        assert_eq!(first.image.dims(), [1, 1, 3]);
        assert_eq!(first.label, 4);
    }
}
