//! Logical collections of named dataset splits.

use std::collections::BTreeMap;

/// A logical dataset made up of independently opened, physical splits.
///
/// The bundle owns split handles, but never concatenates their rows. A
/// bundle's length is the number of named splits.
pub struct DatasetBundle<T> {
    splits: BTreeMap<String, T>,
}

pub enum DatasetLoadResult<T> {
    Single(T),
    Bundle(DatasetBundle<T>),
}

impl<T> DatasetBundle<T> {
    pub fn from_splits(splits: BTreeMap<String, T>) -> Self {
        Self { splits }
    }

    pub fn split(&self, name: &str) -> Option<T>
    where
        T: Clone,
    {
        self.splits.get(name).cloned()
    }

    pub fn split_names(&self) -> impl Iterator<Item = &str> {
        self.splits.keys().map(String::as_str)
    }

    pub fn splits(&self) -> impl Iterator<Item = (&str, &T)> {
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
