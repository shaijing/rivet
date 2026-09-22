use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::Ptx;

use crate::{Error, Result};

type FunctionKey = (usize, String);

/// Per-device cache for statically compiled PTX modules and their functions.
/// The write-side recheck prevents concurrent callers from loading the same
/// module or function twice.
#[derive(Clone, Debug, Default)]
pub(crate) struct ModuleCache {
    modules: Arc<RwLock<HashMap<usize, Arc<CudaModule>>>>,
    functions: Arc<RwLock<HashMap<FunctionKey, CudaFunction>>>,
}

impl ModuleCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn get_or_load_func(
        &self,
        context: &Arc<CudaContext>,
        module: rivet_kernels::Module,
        function: &str,
    ) -> Result<CudaFunction> {
        let key = (module.index(), function.to_owned());
        if let Some(cached) = self.functions.read().unwrap().get(&key).cloned() {
            return Ok(cached);
        }

        let mut functions = self.functions.write().unwrap();
        if let Some(cached) = functions.get(&key).cloned() {
            return Ok(cached);
        }

        let module = self.get_or_load_module(context, module)?;
        let loaded = module
            .load_function(function)
            .map_err(|error| cuda_error("load_kernel_function", function, error))?;
        functions.insert(key, loaded.clone());
        Ok(loaded)
    }

    pub(crate) fn counts(&self) -> (usize, usize) {
        (
            self.modules.read().unwrap().len(),
            self.functions.read().unwrap().len(),
        )
    }

    fn get_or_load_module(
        &self,
        context: &Arc<CudaContext>,
        module: rivet_kernels::Module,
    ) -> Result<Arc<CudaModule>> {
        let key = module.index();
        if let Some(cached) = self.modules.read().unwrap().get(&key).cloned() {
            return Ok(cached);
        }

        let mut modules = self.modules.write().unwrap();
        if let Some(cached) = modules.get(&key).cloned() {
            return Ok(cached);
        }

        let loaded = context
            .load_module(Ptx::from_src(module.ptx()))
            .map_err(|error| cuda_error("load_kernel_module", &key.to_string(), error))?;
        modules.insert(key, Arc::clone(&loaded));
        Ok(loaded)
    }
}

fn cuda_error(op: &'static str, name: &str, error: impl std::fmt::Display) -> Error {
    Error::CudaOperationFailed {
        op,
        message: format!("{name}: {error}"),
    }
}
