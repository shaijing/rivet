//! Deprecated compatibility namespace for image transforms.
//!
//! Use [`crate::transforms`] (or [`crate::api`] for integration-facing
//! helpers) in new code. This module remains temporarily so downstream users
//! can migrate from `rivet_vision::image` without an immediate break.

pub use crate::transforms::{color, crop, decode, flip, layout, normalize, resize};
pub use crate::transforms::{from_rgb_image, into_rgb_image, require_u8_hwc};
