//! Logical collections of named dataset splits.

use super::source::Source;
use std::collections::BTreeMap;

/// A logical dataset made up of independently opened, physical splits.
///
/// The bundle owns split handles, but never concatenates their rows. A
/// bundle's length is the number of named splits.
pub struct DatasetBundle<T> {
    splits: BTreeMap<String, Source<T>>,
}

pub enum DatasetLoadResult<T> {
    Single(Source<T>),
    Bundle(DatasetBundle<T>),
}

impl<T> DatasetBundle<T>
where
    T: Send,
{
    pub(crate) fn from_splits(splits: BTreeMap<String, Source<T>>) -> Self {
        Self { splits }
    }

    pub fn split(&self, name: &str) -> Option<Source<T>> {
        self.splits.get(name).cloned()
    }

    pub fn split_names(&self) -> impl Iterator<Item = &str> {
        self.splits.keys().map(String::as_str)
    }

    pub fn splits(&self) -> impl Iterator<Item = (&str, &Source<T>)> {
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
    use crate::errors::RivetResult;
    use crate::sample::image::EncodedImageSample;
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

        fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<Self::Item>> {
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
            Source::new(Arc::new(LazyTestDataset {
                samples: vec![sample(1)],
            })),
        );
        splits.insert(
            "validation".to_owned(),
            Source::new(Arc::new(MemoryDataset::new(vec![sample(2)]))),
        );

        let bundle = DatasetBundle::from_splits(splits);
        let train = bundle.split("train").unwrap();
        let validation = bundle.split("validation").unwrap();

        assert_eq!(
            bundle.split_names().collect::<Vec<_>>(),
            ["train", "validation"]
        );
        assert_eq!(train.get(0).unwrap().label, 1);
        assert_eq!(validation.get(0).unwrap().label, 2);
    }
}
