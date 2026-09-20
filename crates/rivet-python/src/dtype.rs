use rivet_core::DType;

pub(crate) const fn name(dtype: DType) -> &'static str {
    dtype.name()
}

/// DLPack's type code and bit width for the dtypes exposed by Rivet.
pub(crate) const fn dlpack_code_bits(dtype: DType) -> (u8, u8) {
    match dtype {
        DType::U8 => (1, 8),
        DType::U32 => (1, 32),
        DType::I16 => (0, 16),
        DType::I32 => (0, 32),
        DType::I64 => (0, 64),
        DType::BF16 => (4, 16),
        DType::F16 => (2, 16),
        DType::F32 => (2, 32),
        DType::F64 => (2, 64),
    }
}
