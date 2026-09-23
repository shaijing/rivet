//! Backend-independent logical planning for Rivet pipelines.
//!
//! This crate owns logical IR and planning decisions. It must not execute
//! tensor operations or hold backend runtime handles. Domain crates provide
//! semantic operations and may register planning hooks here.
