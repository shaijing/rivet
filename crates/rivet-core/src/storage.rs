use crate::cpu_backend::CpuStorage;

#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
}
