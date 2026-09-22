use std::fmt::Display;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, CudaView};

use super::device::CudaDevice;
use super::storage::{CudaStorageSlice, CudaStorageView};
use crate::{Error, Layout, Result};

/// The physical matrix layout needed by the cuBLAS planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatrixLayout {
    /// Row-major storage. The leading dimension is the source row stride.
    RowMajor { leading_dim: usize },
    /// Column-major storage. The leading dimension is the source column stride.
    ColMajor { leading_dim: usize },
    /// Arbitrary non-overlapping or overlapping strides. This is materialized
    /// to a contiguous row-major allocation before the cuBLAS call.
    GeneralStrided {
        row_stride: usize,
        col_stride: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transpose {
    NoTrans,
    Transpose,
}

/// Device-independent rank-2 matmul plan.
///
/// Rivet tensors are row-major logically, while cuBLAS consumes column-major
/// matrices. The `transpose` and `leading_dim` fields describe how each
/// source allocation is presented to the cuBLAS call computing `C^T = B^T A^T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatrixOperand {
    pub layout: MatrixLayout,
    pub transpose: Transpose,
    pub leading_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatmulPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub lhs: MatrixOperand,
    pub rhs: MatrixOperand,
}

impl MatmulPlan {
    pub(crate) fn new(lhs: &Layout, rhs: &Layout) -> Result<Self> {
        if lhs.dims().len() != 2 || rhs.dims().len() != 2 {
            return Err(Error::MatmulShapeMismatch {
                lhs: lhs.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }

        let [m, k] = [lhs.dims()[0], lhs.dims()[1]];
        let [rhs_k, n] = [rhs.dims()[0], rhs.dims()[1]];
        if k != rhs_k {
            return Err(Error::MatmulShapeMismatch {
                lhs: lhs.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }

        Ok(Self {
            m,
            n,
            k,
            batch: 1,
            lhs: MatrixOperand::from_layout(lhs),
            rhs: MatrixOperand::from_layout(rhs),
        })
    }

    pub(crate) fn output_len(&self) -> Result<usize> {
        self.m
            .checked_mul(self.n)
            .ok_or(Error::ShapeElementCountOverflow)
    }

    pub(crate) fn has_empty_dimension(&self) -> bool {
        self.m == 0 || self.n == 0 || self.k == 0
    }

    pub(crate) fn needs_lhs_materialization(&self) -> bool {
        matches!(self.lhs.layout, MatrixLayout::GeneralStrided { .. })
    }

    pub(crate) fn needs_rhs_materialization(&self) -> bool {
        matches!(self.rhs.layout, MatrixLayout::GeneralStrided { .. })
    }

    fn config<T: Copy>(&self, alpha: T, beta: T) -> Result<GemmConfig<T>> {
        let m = cublas_dim(self.n)?;
        let n = cublas_dim(self.m)?;
        let k = cublas_dim(self.k)?;
        let lda = cublas_dim(self.rhs.leading_dim)?;
        let ldb = cublas_dim(self.lhs.leading_dim)?;
        let ldc = cublas_dim(self.n)?;

        Ok(GemmConfig {
            // C is allocated row-major. Its column-major view is C^T, so the
            // cuBLAS operands are rhs^T and lhs^T in that order.
            transa: cublas_transpose(self.rhs.transpose),
            transb: cublas_transpose(self.lhs.transpose),
            m,
            n,
            k,
            alpha,
            lda,
            ldb,
            beta,
            ldc,
        })
    }
}

impl MatrixOperand {
    fn from_layout(layout: &Layout) -> Self {
        let rows = layout.dims().first().copied().unwrap_or(0);
        let cols = layout.dims().get(1).copied().unwrap_or(0);
        let row_stride = layout.stride().first().copied().unwrap_or(0);
        let col_stride = layout.stride().get(1).copied().unwrap_or(0);

        // Singleton dimensions do not contribute addresses. Treat them as a
        // regular layout when the remaining dimension has a valid leading
        // dimension; broadcast/overlapping cases still go through the generic
        // materialization path.
        if (col_stride == 1 || cols <= 1) && row_stride >= cols.max(1) {
            return Self {
                layout: MatrixLayout::RowMajor {
                    leading_dim: row_stride,
                },
                // A row-major source is already the column-major view of its
                // transpose, so no cuBLAS transpose flag is needed.
                transpose: Transpose::NoTrans,
                leading_dim: row_stride,
            };
        }

        if (row_stride == 1 || rows <= 1) && col_stride >= rows.max(1) {
            return Self {
                layout: MatrixLayout::ColMajor {
                    leading_dim: col_stride,
                },
                // A column-major source can be transposed in place by cuBLAS.
                transpose: Transpose::Transpose,
                leading_dim: col_stride,
            };
        }

        Self {
            layout: MatrixLayout::GeneralStrided {
                row_stride,
                col_stride,
            },
            // These values are replaced after materialization.
            transpose: Transpose::NoTrans,
            leading_dim: 0,
        }
    }
}

fn cublas_dim(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::ShapeElementCountOverflow)
}

fn cublas_transpose(transpose: Transpose) -> cudarc::cublas::sys::cublasOperation_t {
    match transpose {
        Transpose::NoTrans => cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        Transpose::Transpose => cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
    }
}

fn matmul_view<'a>(data: &'a CudaStorageSlice, layout: &Layout) -> Result<CudaStorageView<'a>> {
    let start = layout.start_offset();
    let len = data
        .len()
        .checked_sub(start)
        .ok_or(Error::StorageOutOfBounds)?;
    data.view(start, len)
}

fn cublas_error(error: impl Display) -> Error {
    Error::CudaOperationFailed {
        op: "cublas_gemm",
        message: error.to_string(),
    }
}

fn launch_gemm<T: Copy>(
    device: &CudaDevice,
    config: GemmConfig<T>,
    rhs: &CudaView<'_, T>,
    lhs: &CudaView<'_, T>,
    output: &mut CudaSlice<T>,
) -> Result<()>
where
    CudaBlas: Gemm<T>,
{
    unsafe { device.cublas_handle().gemm(config, rhs, lhs, output) }.map_err(cublas_error)?;
    device.record_cublas_call();
    Ok(())
}

macro_rules! launch_typed {
    ($device:expr, $config:expr, $rhs:expr, $lhs:expr, $output:expr) => {{ launch_gemm($device, $config, $rhs, $lhs, $output) }};
}

pub(crate) fn matmul(
    device: &CudaDevice,
    lhs: &CudaStorageSlice,
    lhs_layout: &Layout,
    rhs: &CudaStorageSlice,
    rhs_layout: &Layout,
    plan: &MatmulPlan,
    output: &mut CudaStorageSlice,
) -> Result<()> {
    let dtype = lhs.dtype();
    let lhs_view = matmul_view(lhs, lhs_layout)?;
    let rhs_view = matmul_view(rhs, rhs_layout)?;

    match (lhs_view, rhs_view, output) {
        (CudaStorageView::F32(lhs), CudaStorageView::F32(rhs), CudaStorageSlice::F32(output)) => {
            launch_typed!(device, plan.config(1.0f32, 0.0f32)?, &rhs, &lhs, output)
        }
        (CudaStorageView::F16(lhs), CudaStorageView::F16(rhs), CudaStorageSlice::F16(output)) => {
            launch_typed!(
                device,
                plan.config(half::f16::from_f32(1.0), half::f16::from_f32(0.0))?,
                &rhs,
                &lhs,
                output
            )
        }
        (
            CudaStorageView::BF16(lhs),
            CudaStorageView::BF16(rhs),
            CudaStorageSlice::BF16(output),
        ) => launch_typed!(
            device,
            plan.config(half::bf16::from_f32(1.0), half::bf16::from_f32(0.0))?,
            &rhs,
            &lhs,
            output
        ),
        _ => Err(Error::UnsupportedMatmulDType { dtype }),
    }
}
