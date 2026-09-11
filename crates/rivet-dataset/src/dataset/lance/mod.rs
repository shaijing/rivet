//! Lance-backed datasets.
//!
//! The storage layer exposes a synchronous `take` bridge because Rivet's
//! public loader is synchronous. Image adaptation stays separate from the
//! generic table reader so future modalities can reuse the same projection
//! and runtime code.

mod image;
mod table;

pub use image::LanceImageDataset;
pub use table::LanceTable;
