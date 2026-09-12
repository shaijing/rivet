//! In-memory dataset backends.

mod dataset;
mod decoded;

pub use dataset::MemoryDataset;
pub use decoded::DecodedImageMemoryDataset;
