/// Conservative default for encoded cache materialization.
pub const DEFAULT_ENCODED_CHUNK_SIZE: usize = 4096;

/// Controls whether a source remains backed by its original dataset or is
/// materialized into an eager cache.
///
/// `Encoded` retains compressed image payloads. It does not decode images;
/// decoding remains a pipeline operation and therefore still happens on each
/// requested sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    None,
    Encoded { chunk_size: usize },
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::None
    }
}

impl CachePolicy {
    pub const fn encoded() -> Self {
        Self::Encoded {
            chunk_size: DEFAULT_ENCODED_CHUNK_SIZE,
        }
    }
}
