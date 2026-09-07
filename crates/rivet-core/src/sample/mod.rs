//! Typed sample definitions, split by modality. Storage backends and
//! pipelines stay generic over `Dataset<Item = ...>`; each modality module
//! owns its sample types and state.

pub mod image;
pub mod text;
