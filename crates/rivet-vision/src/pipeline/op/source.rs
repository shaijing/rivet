use crate::errors::RivetResult;
use crate::sample::image::{ImageBatch, ImageSample};
use crate::source::ImageSource;

#[derive(Clone)]
pub struct SourceOp {
    source: ImageSource,
}

impl SourceOp {
    pub fn new(source: ImageSource) -> Self {
        Self { source }
    }

    pub fn len(&self) -> usize {
        self.source.len()
    }

    pub fn state(&self) -> super::PipelineImageState {
        self.source.state()
    }

    pub fn get(&self, index: usize) -> RivetResult<ImageSample> {
        self.source.get(index)
    }

    pub fn get_many(&self, indices: &[usize]) -> RivetResult<Vec<ImageSample>> {
        self.source.get_many(indices)
    }

    pub fn supports_batch_read(&self) -> bool {
        self.source.supports_batch_read()
    }

    pub fn get_batch(&self, indices: &[usize]) -> Option<RivetResult<ImageBatch>> {
        self.source.get_batch(indices)
    }
}
