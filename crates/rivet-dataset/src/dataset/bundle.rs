//! Logical collections of named dataset splits.

use std::collections::BTreeMap;
use std::sync::Arc;

/// A logical dataset made up of independently opened, physical splits.
///
/// The bundle owns split handles, but never concatenates their rows. A
/// bundle's length is the number of named splits.
pub struct DatasetBundle<D> {
    splits: BTreeMap<String, Arc<D>>,
}

pub enum DatasetLoadResult<D> {
    Single(Arc<D>),
    Bundle(DatasetBundle<D>),
}

impl<D> DatasetBundle<D> {
    pub(crate) fn from_splits(splits: BTreeMap<String, Arc<D>>) -> Self {
        Self { splits }
    }

    pub fn split(&self, name: &str) -> Option<Arc<D>> {
        self.splits.get(name).cloned()
    }

    pub fn split_names(&self) -> impl Iterator<Item = &str> {
        self.splits.keys().map(String::as_str)
    }

    pub fn splits(&self) -> impl Iterator<Item = (&str, &Arc<D>)> {
        self.splits
            .iter()
            .map(|(name, dataset)| (name.as_str(), dataset))
    }

    pub fn len(&self) -> usize {
        self.splits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splits.is_empty()
    }
}
