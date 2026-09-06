pub(crate) struct IndexSampler {
    indices: Vec<usize>,
    position: usize,
}

impl IndexSampler {
    pub(crate) fn new(indices: Vec<usize>, position: usize) -> Self {
        let len = indices.len();
        Self {
            indices,
            position: position.min(len),
        }
    }

    pub(crate) fn next_indices(&mut self, batch_size: usize) -> Option<Vec<usize>> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + batch_size).min(self.indices.len());
        let indices = self.indices[self.position..end].to_vec();
        self.position = end;
        Some(indices)
    }
}
