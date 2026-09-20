//! Representation-boundary and tensor representation transform facade.

pub use super::decode::{DecodeImageConfig, decode_rgb};
pub use super::normalize::{
    NormalizeConfig, normalize_u8_batch_to_f32, normalize_u8_batch_to_nchw_f32, normalize_u8_to_f32,
};
