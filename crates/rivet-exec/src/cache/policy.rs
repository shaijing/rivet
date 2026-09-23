/// Conservative default chunk size for cache materialization.
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLevel {
    None,
    Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub level: CacheLevel,
    pub chunk_size: usize,
}

impl CacheConfig {
    pub const fn memory() -> Self {
        Self {
            level: CacheLevel::Memory,
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }

    pub const fn policy(self) -> CachePolicy {
        match self.level {
            CacheLevel::None => CachePolicy::None,
            CacheLevel::Memory => CachePolicy::Memory {
                chunk_size: self.chunk_size,
            },
        }
    }
}

/// Controls whether a dataset is materialized into generic memory storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    None,
    Memory { chunk_size: usize },
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::None
    }
}

impl CachePolicy {
    pub const fn memory() -> Self {
        Self::Memory {
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }
}
