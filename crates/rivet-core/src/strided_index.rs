use crate::Layout;

/// Iterates logical row-major positions as offsets into the backing storage.
#[derive(Debug)]
pub struct StridedIndex<'a> {
    dims: &'a [usize],
    stride: &'a [usize],
    index: Vec<usize>,
    next_storage_index: usize,
    remaining: usize,
}

impl<'a> StridedIndex<'a> {
    pub fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        debug_assert_eq!(dims.len(), stride.len());
        Self {
            dims,
            stride,
            index: vec![0; dims.len()],
            next_storage_index: start_offset,
            remaining: dims.iter().product(),
        }
    }

    pub fn from_layout(layout: &'a Layout) -> Self {
        Self::new(layout.dims(), layout.stride(), layout.start_offset())
    }
}

impl Iterator for StridedIndex<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let current = self.next_storage_index;
        let mut advanced = false;
        let mut next = current;
        for ((index, &dim), &stride) in self.index.iter_mut().zip(self.dims).zip(self.stride).rev()
        {
            let next_index = *index + 1;
            if next_index < dim {
                *index = next_index;
                next += stride;
                advanced = true;
                break;
            }
            next -= *index * stride;
            *index = 0;
        }
        self.remaining -= 1;
        if advanced {
            self.next_storage_index = next;
        } else {
            self.next_storage_index = 0;
        }
        Some(current)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedIndex<'_> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl std::iter::FusedIterator for StridedIndex<'_> {}
