use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use cudarc::driver::CudaModule;

/// Per-device module storage. Loading functions is intentionally deferred to
/// the static-PTX/kernel phase, but the cache belongs to the CUDA device from
/// the beginning so later kernel dispatch does not add another owner.
#[derive(Clone, Debug, Default)]
pub(crate) struct ModuleCache {
    #[allow(dead_code)]
    modules: Arc<RwLock<HashMap<String, Arc<CudaModule>>>>,
}

impl ModuleCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}
