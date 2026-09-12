//! Logical collections of named dataset splits.

use super::image_source::ImageSource;
use std::collections::BTreeMap;

/// A logical dataset made up of independently opened, physical splits.
///
/// The bundle owns split handles, but never concatenates their rows. A
/// bundle's length is the number of named splits.
pub struct DatasetBundle {
    splits: BTreeMap<String, ImageSource>,
}

pub enum DatasetLoadResult {
    Single(ImageSource),
    Bundle(DatasetBundle),
}

impl DatasetBundle {
    pub(crate) fn from_splits(splits: BTreeMap<String, ImageSource>) -> Self {
        Self { splits }
    }

    pub fn split(&self, name: &str) -> Option<ImageSource> {
        self.splits.get(name).cloned()
    }

    pub fn split_names(&self) -> impl Iterator<Item = &str> {
        self.splits.keys().map(String::as_str)
    }

    pub fn splits(&self) -> impl Iterator<Item = (&str, &ImageSource)> {
        self.splits
            .iter()
            .map(|(name, source)| (name.as_str(), source))
    }

    pub fn len(&self) -> usize {
        self.splits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splits.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::memory::MemoryDataset;
    use crate::sample::image::{
        DecodedSample, EncodedImageSample, ImageBuffer, ImageLayout, ImageSample,
    };
    use arrow_buffer::Buffer;
    use std::sync::Arc;

    struct LazyTestDataset {
        samples: Vec<EncodedImageSample>,
    }

    impl crate::dataset::source::Dataset for LazyTestDataset {
        type Item = EncodedImageSample;

        fn len(&self) -> usize {
            self.samples.len()
        }

        fn get_many(&self, indices: &[usize]) -> crate::errors::RivetResult<Vec<Self::Item>> {
            indices
                .iter()
                .map(|&index| {
                    self.samples.get(index).cloned().ok_or_else(|| {
                        crate::errors::RivetError::IndexOutOfRange {
                            index,
                            len: self.samples.len(),
                        }
                    })
                })
                .collect()
        }
    }

    fn sample(label: i64) -> EncodedImageSample {
        EncodedImageSample {
            image: Buffer::from(label.to_string().into_bytes()),
            label,
        }
    }

    #[test]
    fn splits_can_mix_backends_for_the_same_item_type() {
        let mut splits = BTreeMap::new();
        splits.insert(
            "train".to_owned(),
            ImageSource::from_encoded(Arc::new(LazyTestDataset {
                samples: vec![sample(1)],
            })),
        );
        splits.insert(
            "validation".to_owned(),
            ImageSource::from_decoded(Arc::new(MemoryDataset::new(vec![DecodedSample {
                image: ImageBuffer::SharedU8(Arc::from(vec![2u8].into_boxed_slice())),
                width: 1,
                height: 1,
                channels: 1,
                label: 2,
                layout: ImageLayout::Hwc,
            }]))),
        );

        let bundle = DatasetBundle::from_splits(splits);
        let train = bundle.split("train").unwrap();
        let validation = bundle.split("validation").unwrap();

        assert_eq!(
            bundle.split_names().collect::<Vec<_>>(),
            ["train", "validation"]
        );
        let ImageSample::Encoded(train) = train.get(0).unwrap() else {
            panic!("expected an encoded train sample");
        };
        assert_eq!(train.label, 1);
        let ImageSample::Decoded(validation) = validation.get(0).unwrap() else {
            panic!("expected a decoded validation sample");
        };
        assert_eq!(validation.label, 2);
    }
}
