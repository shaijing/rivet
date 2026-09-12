/// Conservative default chunk size for cache materialization.
pub const DEFAULT_ENCODED_CHUNK_SIZE: usize = 4096;
pub const DEFAULT_DECODED_CHUNK_SIZE: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLevel {
    None,
    Encoded,
    Decoded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub level: CacheLevel,
    pub chunk_size: usize,
    pub max_bytes: Option<usize>,
}

impl CacheConfig {
    pub const fn encoded() -> Self {
        Self {
            level: CacheLevel::Encoded,
            chunk_size: DEFAULT_ENCODED_CHUNK_SIZE,
            max_bytes: None,
        }
    }

    pub const fn decoded() -> Self {
        Self {
            level: CacheLevel::Decoded,
            chunk_size: DEFAULT_DECODED_CHUNK_SIZE,
            max_bytes: None,
        }
    }

    pub const fn policy(self) -> CachePolicy {
        match self.level {
            CacheLevel::None => CachePolicy::None,
            CacheLevel::Encoded => CachePolicy::Encoded {
                chunk_size: self.chunk_size,
            },
            CacheLevel::Decoded => CachePolicy::Decoded {
                chunk_size: self.chunk_size,
                max_bytes: self.max_bytes,
            },
        }
    }
}

/// Controls whether a source remains backed by its original dataset or is
/// materialized into an eager cache.
///
/// `Encoded` retains compressed image payloads. It does not decode images;
/// decoding remains a pipeline operation and therefore still happens on each
/// requested sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    None,
    Encoded {
        chunk_size: usize,
    },
    Decoded {
        chunk_size: usize,
        max_bytes: Option<usize>,
    },
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

    pub const fn decoded() -> Self {
        Self::Decoded {
            chunk_size: DEFAULT_DECODED_CHUNK_SIZE,
            max_bytes: None,
        }
    }
}
