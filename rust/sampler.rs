pub(crate) struct SequentialSampler {
    position: usize,
    len: usize,
}

impl SequentialSampler {
    pub(crate) fn with_position(len: usize, position: usize) -> Self {
        Self {
            position: position.min(len),
            len,
        }
    }

    pub(crate) fn next_indices(&mut self, batch_size: usize) -> Option<Vec<usize>> {
        if self.position >= self.len {
            return None;
        }

        let end = (self.position + batch_size).min(self.len);
        let indices = (self.position..end).collect();
        self.position = end;
        Some(indices)
    }
}
