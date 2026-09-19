//! Compatibility facade for the pre-split dataset crate.
//!
//! New code should depend on `rivet-data` for modality-neutral storage and on
//! `rivet-vision` for image datasets and pipelines. This crate keeps the old
//! module paths available while the ecosystem migrates.

pub mod batch {
    pub use rivet_vision::batch::*;
}

pub mod dataset {
    pub mod arrow {
        pub use rivet_data::dataset::arrow::*;
        pub use rivet_vision::datasets::arrow::*;
    }

    pub mod bundle {
        pub use rivet_data::dataset::bundle::*;
    }

    pub mod cache {
        pub use rivet_data::cache::materialize_to_memory;
        pub use rivet_vision::cache::materialize_decoded_to_memory;
        pub use rivet_vision::cache::{
            CacheConfig, CacheLevel, CachePolicy, DEFAULT_DECODED_CHUNK_SIZE,
            DEFAULT_ENCODED_CHUNK_SIZE,
        };
    }

    pub mod filesystem {
        pub use rivet_vision::datasets::filesystem::*;
    }

    pub mod image_source {
        pub use rivet_vision::source::*;
    }

    #[cfg(feature = "lance")]
    pub mod lance {
        pub use rivet_data::dataset::lance::*;
        pub use rivet_vision::datasets::lance::*;
    }

    pub mod manifest {
        pub use rivet_data::dataset::manifest::*;
    }

    pub mod memory {
        pub use rivet_data::dataset::memory::*;
        pub use rivet_vision::cache::{
            DecodedImageMemoryDataset, DenseImageMemoryDataset, VariableImageMemoryDataset,
        };
    }

    pub mod source {
        pub use rivet_data::dataset::source::*;
    }

    pub use rivet_data::cache::materialize_to_memory;
    pub use rivet_data::dataset::{
        Dataset, DatasetBundle, DatasetLoadResult, DatasetManifest, MANIFEST_FILE_NAME,
        MemoryDataset, Source, SplitManifest,
    };
    pub use rivet_vision::cache::{
        CacheConfig, CacheLevel, CachePolicy, DEFAULT_DECODED_CHUNK_SIZE,
        DEFAULT_ENCODED_CHUNK_SIZE, DecodedImageMemoryDataset, DenseImageMemoryDataset,
        VariableImageMemoryDataset, materialize_decoded_to_memory,
    };
    pub use rivet_vision::datasets::{
        ArrowImageDataset, ImageFolderDatasetCore, ImageFolderSample,
    };
    pub use rivet_vision::source::ImageSource;

    #[cfg(feature = "lance")]
    pub use rivet_data::dataset::lance::LanceTable;
    #[cfg(feature = "lance")]
    pub use rivet_vision::datasets::{LanceImageDataset, load_lance_image_dataset};
}

pub mod errors {
    pub use rivet_vision::errors::{
        RivetError, RivetResult, VisionError, VisionResult, invalid_argument, invalid_pipeline,
        invalid_shape,
    };
}

pub mod image {
    pub use rivet_vision::image::*;
    pub use rivet_vision::transforms::{color, crop, decode, flip, layout, normalize, resize};
}

pub mod pipeline {
    pub use rivet_vision::pipeline::*;
}

pub mod runtime {
    pub use rivet_vision::runtime::*;
}

pub mod sample {
    pub mod image {
        pub use rivet_vision::sample::image::*;
    }

    pub mod text {
        include!("sample/text.rs");
    }
}

pub mod sampler {
    pub use rivet_data::sampler::*;
}
