#![doc = include_str!("../docs/text-pipeline.md")]

//! Text preprocessing backed by tokenizers 1.0: immutable tokenizer loading,
//! ordered encoding, truncation, padding and CPU tensor collation.

mod batch;
pub mod errors;
mod noise;
mod records;
mod sample;
mod text_file;
mod tokenizer;

pub use batch::{LabelMode, TextBatch};
pub use errors::{TextError, TextResult};
pub use noise::{TextNoise, TextNoiseKind};
pub use records::{TextRecordsDataset, TextRecordsOptions};
pub use sample::{TextEncoding, TextSample};
pub use text_file::{
    TextEncodingErrors, TextFileDataset, TextFileEncoding, TextFileOptions, TextSampleBy,
};
pub use tokenizer::{
    EncodeOptions, Override, PaddingDirection, PaddingParams, PaddingStrategy, TextTokenizer,
    TruncationDirection, TruncationParams, TruncationStrategy,
};

mod dataset;
mod packing;
mod pipeline;
mod planning;

pub use dataset::{ArrowTextDataset, TextColumns};
pub use packing::{PackingOptions, pack_encodings};
pub use pipeline::{TextDataLoader, TextPipeline};
pub use planning::{TEXT_DOMAIN_ID, TextPlanOp, TextPropertyInference};
