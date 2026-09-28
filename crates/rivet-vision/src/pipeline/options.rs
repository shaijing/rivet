/// Optional logical optimizations. Safety validation and source-index ordering
/// remain enabled regardless of these settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageOptimizationOptions {
    pub common_subplan_elimination: bool,
    pub canonicalize_selections: bool,
    pub canonicalize_layouts: bool,
    /// Allow U8->F32 conversion followed by normalization to use a reassociated
    /// affine expression. This can change low floating-point bits.
    pub allow_float_reassociation: bool,
    /// Retain a full plan snapshot after every pass for debugging.
    pub record_snapshots: bool,
}

impl Default for ImageOptimizationOptions {
    fn default() -> Self {
        Self {
            common_subplan_elimination: true,
            canonicalize_selections: true,
            canonicalize_layouts: true,
            allow_float_reassociation: false,
            record_snapshots: false,
        }
    }
}
