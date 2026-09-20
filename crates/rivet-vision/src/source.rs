//! State-aware image sources used by the image pipeline.

use crate::cache::{
    CachePolicy, DecodedImageMemoryDataset, DenseImageMemoryDataset, materialize_decoded_to_memory,
};
use crate::errors::{RivetResult, VisionResult, invalid_argument};
use crate::pipeline::op::PipelineImageState;
use crate::sample::image::ImageAxisOrder;
use crate::sample::image::{DecodedSample, EncodedImageSample, ImageBatch, ImageSample};
use rivet_core::DType;
use rivet_data::dataset::{Dataset, Source};
use rivet_data::materialize_to_memory;
use std::sync::Arc;

/// A typed image source that keeps the pipeline's initial representation
/// alongside its backend. Persistent and encoded-memory sources stay encoded;
/// decoded caches expose decoded U8 HWC samples directly.
#[derive(Clone)]
pub enum ImageSource {
    Encoded(Source<EncodedImageSample>),
    Decoded(Source<DecodedSample>),
    DenseDecoded(Source<DecodedSample>, Arc<DenseImageMemoryDataset>),
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

    /// Construct a decoded source with the optional dense batch-read capability.
    ///
    /// Generic decoded datasets continue to use [`Self::Decoded`] and the
    /// sample-oriented fallback path.
    pub fn from_dense_decoded(dataset: Arc<DenseImageMemoryDataset>) -> Self {
        Self::DenseDecoded(Source::new(Arc::clone(&dataset)), dataset)
    }

    pub fn state(&self) -> PipelineImageState {
        match self {
            Self::Encoded(_) => PipelineImageState::Encoded,
            Self::Decoded(_) | Self::DenseDecoded(..) => PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: ImageAxisOrder::Hwc,
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Encoded(source) => source.len(),
            Self::Decoded(source) | Self::DenseDecoded(source, _) => source.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> VisionResult<ImageSample> {
        let mut samples = self.get_many(&[index])?;
        if samples.len() != 1 {
            return Err(invalid_argument(format!(
                "image source returned {} rows for one requested index",
                samples.len()
            )));
        }
        Ok(samples.remove(0))
    }

    /// Return an encoded sample when this source still exposes encoded data.
    ///
    /// This keeps integrations from matching on the internal source enum just
    /// to implement a point read.
    pub fn get_encoded(&self, index: usize) -> VisionResult<EncodedImageSample> {
        match self.get(index)? {
            ImageSample::Encoded(sample) => Ok(sample),
            ImageSample::Decoded(_) => Err(invalid_argument(
                "encoded samples are unavailable after decoded caching",
            )),
        }
    }

    /// Return one decoded sample as the standard one-item image batch.
    pub fn get_decoded_batch(&self, index: usize) -> VisionResult<ImageBatch> {
        match self.get(index)? {
            ImageSample::Encoded(sample) => {
                crate::api::decode_image_batch(sample.image.as_slice(), sample.label)
            }
            ImageSample::Decoded(sample) => crate::api::single_sample_batch(sample),
        }
    }

    pub fn get_many(&self, indices: &[usize]) -> VisionResult<Vec<ImageSample>> {
        match self {
            Self::Encoded(source) => Ok(source
                .get_many(indices)?
                .into_iter()
                .map(ImageSample::Encoded)
                .collect()),
            Self::Decoded(source) => Ok(source
                .get_many(indices)?
                .into_iter()
                .map(ImageSample::Decoded)
                .collect()),
            Self::DenseDecoded(source, _) => Ok(source
                .get_many(indices)?
                .into_iter()
                .map(ImageSample::Decoded)
                .collect()),
        }
    }

    pub fn supports_batch_read(&self) -> bool {
        matches!(self, Self::DenseDecoded(..))
    }

    pub fn get_batch(&self, indices: &[usize]) -> Option<RivetResult<ImageBatch>> {
        match self {
            Self::DenseDecoded(_, dataset) => Some(dataset.get_batch(indices)),
            Self::Encoded(_) | Self::Decoded(_) => None,
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
                Self::Encoded(source) => Ok(Self::Encoded(Source::new(Arc::new(
                    materialize_to_memory(source.as_dataset(), chunk_size)?,
                )))),
                Self::Decoded(_) | Self::DenseDecoded(..) => Err(invalid_argument(
                    "encoded cache requires an encoded image source",
                )),
            },
            CachePolicy::Decoded {
                chunk_size,
                max_bytes,
            } => match self {
                Self::Encoded(source) => {
                    let materialized =
                        materialize_decoded_to_memory(source.as_dataset(), chunk_size, max_bytes)?;
                    match materialized {
                        DecodedImageMemoryDataset::Dense(dataset) => {
                            Ok(Self::from_dense_decoded(Arc::new(dataset)))
                        }
                        DecodedImageMemoryDataset::Variable(dataset) => Ok(Self::Decoded(
                            Source::new(Arc::new(DecodedImageMemoryDataset::Variable(dataset))),
                        )),
                    }
                }
                Self::Decoded(_) | Self::DenseDecoded(..) => Ok(self.clone()),
            },
        }
    }

    /// Cache this source using the public vision cache configuration.
    pub fn cache_config(&self, config: crate::cache::CacheConfig) -> RivetResult<Self> {
        self.cache(config.policy())
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

        fn get_many(&self, indices: &[usize]) -> rivet_data::DataResult<Vec<Self::Item>> {
            indices
                .iter()
                .map(|&index| {
                    if index != 0 {
                        return Err(rivet_data::DataError::IndexOutOfRange { index, len: 1 });
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
        assert!(cached.supports_batch_read());

        assert_eq!(
            cached.state(),
            PipelineImageState::Decoded {
                dtype: DType::U8,
                axis_order: crate::sample::image::ImageAxisOrder::Hwc,
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

        let batch = cached.get_batch(&[0, 0]).unwrap().unwrap();
        assert_eq!(batch.images.dims(), [2, 1, 1, 3]);
        assert_eq!(batch.labels.to_vec::<i64>().unwrap(), [4, 4]);
    }
}
