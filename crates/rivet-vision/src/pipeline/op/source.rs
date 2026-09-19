use crate::dataset::ImageSource;
use crate::errors::RivetResult;
use crate::sample::image::ImageSample;

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
}
