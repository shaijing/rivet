use super::{ExclusiveTensor, Tensor, Tensor_, TensorId};
use crate::cpu_backend::buffer::AlignedBuffer;
use crate::cpu_backend::linalg::{self, Input, LinearFloat};
use crate::storage::{Storage, validate_layout_for_storage};
use crate::{CpuStorage, DType, Error, Result, Shape, WithDType};
use std::sync::Arc;

/// Select which side of a matrix product contains the structured matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixSide {
    #[default]
    Left,
    Right,
}

/// Triangle is selected before applying the optional transpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangularOptions {
    pub upper: bool,
    pub unit_diagonal: bool,
    pub transpose: bool,
    pub side: MatrixSide,
}

impl Default for TriangularOptions {
    fn default() -> Self {
        Self {
            upper: false,
            unit_diagonal: false,
            transpose: false,
            side: MatrixSide::Left,
        }
    }
}

pub use extensions::givens_rotation;

macro_rules! dispatch {
    ($tensor:expr, $op:literal, $fun:ident $(, $arg:expr)*) => {
        match $tensor.dtype() {
            DType::F32 => $fun::<f32>($($arg),*),
            DType::F64 => $fun::<f64>($($arg),*),
            dtype => Err(Error::UnsupportedDTypeForOp { op: $op, dtype }),
        }
    };
}

mod extensions;

fn values<'a, T: WithDType>(tensor: &'a Tensor, _op: &'static str) -> Result<&'a [T]> {
    match tensor.storage() {
        Storage::Cpu(storage) => T::cpu_storage_as_slice(storage),
        #[cfg(feature = "cuda")]
        Storage::Cuda(_) => Err(Error::UnsupportedCudaOp { op: _op }),
    }
}

fn input<'a, T: WithDType>(tensor: &'a Tensor, _op: &'static str) -> Result<Input<'a, T>> {
    Input::new(values(tensor, _op)?, tensor.layout())
}

fn finish<T: LinearFloat>(output: AlignedBuffer<T>, shape: impl Into<Shape>) -> Result<Tensor> {
    Tensor::from_exact_owned_storage(
        Storage::Cpu(CpuStorage::from_aligned_buffer(output)),
        shape.into(),
    )
}

impl Tensor {
    fn check_linear_peer(&self, rhs: &Self) -> Result<()> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        Ok(())
    }

    /// Computes `beta * self + alpha * lhs @ rhs` for CPU F32/F64 tensors.
    /// The addend must have exactly the output shape; no broadcasting occurs.
    /// A zero beta ignores addend values, and a zero alpha ignores products.
    pub fn addmm(&self, lhs: &Self, rhs: &Self, alpha: f64, beta: f64) -> Result<Self> {
        self.check_linear_peer(lhs)?;
        self.check_linear_peer(rhs)?;
        dispatch!(self, "addmm", addmm, self, lhs, rhs, alpha, beta)
    }

    /// Computes `beta * self + alpha * matrix @ vector` with strict shapes.
    pub fn addmv(&self, matrix: &Self, vector: &Self, alpha: f64, beta: f64) -> Result<Self> {
        if self.rank() != 1 || matrix.rank() != 2 || vector.rank() != 1 {
            return Err(Error::MatmulShapeMismatch {
                lhs: matrix.dims().to_vec(),
                rhs: vector.dims().to_vec(),
            });
        }
        self.check_linear_peer(matrix)?;
        self.check_linear_peer(vector)?;
        dispatch!(self, "addmv", addmv, self, matrix, vector, alpha, beta)
    }

    /// Computes the outer product of two CPU F32/F64 rank-1 tensors.
    pub fn outer(&self, rhs: &Self) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        dispatch!(self, "outer", outer, self, rhs, None, 1.0, 0.0)
    }

    /// Computes `beta * self + alpha * x * y^T` with an exact matrix addend.
    pub fn addr(&self, x: &Self, y: &Self, alpha: f64, beta: f64) -> Result<Self> {
        self.check_linear_peer(x)?;
        self.check_linear_peer(y)?;
        dispatch!(self, "addr", outer, x, y, Some(self), alpha, beta)
    }

    /// Sum of absolute values of all elements of a CPU F32/F64 tensor.
    pub fn norm_l1(&self) -> Result<Self> {
        dispatch!(self, "norm_l1", norm_l1, self)
    }

    /// Computes `A^T A` when `transpose=true`, otherwise `A A^T`.
    /// The result is a full symmetric matrix, not a triangular representation.
    pub fn gram(&self, transpose: bool) -> Result<Self> {
        dispatch!(self, "gram", gram, self, transpose)
    }

    /// Multiplies the symmetric matrix described by one triangle of `self`
    /// with a rank-2 RHS. Values in the other triangle are ignored.
    pub fn symmetric_matmul(&self, rhs: &Self, upper: bool) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        dispatch!(self, "symmetric_matmul", symmetric, self, rhs, upper)
    }

    /// Symmetric matrix-vector multiplication, reading only the chosen triangle.
    pub fn symmetric_mv(&self, rhs: &Self, upper: bool) -> Result<Self> {
        if rhs.rank() != 1 {
            return Err(Error::InvalidRank {
                expected: 1,
                actual: rhs.rank(),
            });
        }
        self.symmetric_matmul(&rhs.unsqueeze(1)?, upper)?.squeeze(1)
    }

    /// Solves `self @ X = rhs` using the specified triangle. The RHS may be
    /// a vector or matrix. Unit diagonal ignores stored diagonal values;
    /// otherwise an exactly zero diagonal is reported before computation.
    pub fn triangular_solve(&self, rhs: &Self, upper: bool, unit_diagonal: bool) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        if rhs.rank() == 1 {
            return self
                .triangular_solve(&rhs.unsqueeze(1)?, upper, unit_diagonal)?
                .squeeze(1);
        }
        dispatch!(
            self,
            "triangular_solve",
            triangular,
            self,
            rhs,
            upper,
            unit_diagonal
        )
    }

    pub(super) fn cpu_broadcast_matmul(&self, rhs: &Self, shape: Shape) -> Result<Option<Self>> {
        match (self.storage(), rhs.storage()) {
            (Storage::Cpu(_), Storage::Cpu(_)) => {
                dispatch!(self, "broadcast_matmul", batched, self, rhs, shape).map(Some)
            }
            #[cfg(feature = "cuda")]
            _ => Ok(None),
        }
    }
}

fn addmm<T: LinearFloat>(
    add: &Tensor,
    lhs: &Tensor,
    rhs: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<Tensor> {
    let output = linalg::addmm::<T>(
        input(lhs, "addmm")?,
        input(rhs, "addmm")?,
        Some(input(add, "addmm")?),
        alpha,
        beta,
    )?;
    finish(output, (lhs.dims()[0], rhs.dims()[1]))
}

fn addmv<T: LinearFloat>(
    add: &Tensor,
    matrix: &Tensor,
    vector: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<Tensor> {
    let output = linalg::addmv::<T>(
        input(matrix, "addmv")?,
        input(vector, "addmv")?,
        input(add, "addmv")?,
        alpha,
        beta,
    )?;
    finish(output, add.dims()[0])
}

fn outer<T: LinearFloat>(
    x: &Tensor,
    y: &Tensor,
    add: Option<&Tensor>,
    alpha: f64,
    beta: f64,
) -> Result<Tensor> {
    let output = linalg::outer::<T>(
        input(x, "outer")?,
        input(y, "outer")?,
        add.map(|tensor| input(tensor, "addr")).transpose()?,
        alpha,
        beta,
    )?;
    finish(output, (x.dims()[0], y.dims()[0]))
}

fn norm_l1<T: LinearFloat>(tensor: &Tensor) -> Result<Tensor> {
    finish(linalg::norm_l1::<T>(input(tensor, "norm_l1")?)?, ())
}

fn gram<T: LinearFloat>(tensor: &Tensor, transpose: bool) -> Result<Tensor> {
    let output = linalg::gram::<T>(input(tensor, "gram")?, transpose)?;
    let size = tensor.dims()[usize::from(transpose)];
    finish(output, (size, size))
}

fn symmetric<T: LinearFloat>(matrix: &Tensor, rhs: &Tensor, upper: bool) -> Result<Tensor> {
    let output = linalg::symmetric::<T>(
        input(matrix, "symmetric_matmul")?,
        input(rhs, "symmetric_matmul")?,
        upper,
    )?;
    finish(output, (matrix.dims()[0], rhs.dims()[1]))
}

fn triangular<T: LinearFloat>(
    matrix: &Tensor,
    rhs: &Tensor,
    upper: bool,
    unit: bool,
) -> Result<Tensor> {
    let output = linalg::triangular::<T>(
        input(matrix, "triangular_solve")?,
        input(rhs, "triangular_solve")?,
        upper,
        unit,
    )?;
    finish(output, (matrix.dims()[0], rhs.dims()[1]))
}

fn batched<T: LinearFloat>(lhs: &Tensor, rhs: &Tensor, shape: Shape) -> Result<Tensor> {
    finish(
        linalg::batched::<T>(
            input(lhs, "broadcast_matmul")?,
            input(rhs, "broadcast_matmul")?,
        )?,
        shape,
    )
}

impl ExclusiveTensor {
    /// Restores an ordinary immutable tensor without copying its allocation.
    pub fn into_tensor(self) -> Tensor {
        Tensor(Arc::new(Tensor_ {
            id: TensorId::new(),
            storage: Arc::new(self.storage),
            layout: self.layout,
            dtype: self.dtype,
            device: self.device,
        }))
    }

    /// Updates the exclusive contiguous view: `self <- self + alpha * x`.
    /// CPU F32/F64 only, with exact shape and dtype matching.
    /// A coefficient that converts to zero leaves the destination unchanged.
    pub fn axpy(&mut self, alpha: f64, x: &Tensor) -> Result<()> {
        if self.dtype != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype,
                rhs: x.dtype(),
            });
        }
        if !self.device.same_device(x.device()) {
            return Err(Error::DeviceMismatch);
        }
        if self.layout.shape() != x.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.layout.dims().to_vec(),
                rhs: x.dims().to_vec(),
            });
        }
        match self.dtype {
            DType::F32 => update_axpy::<f32>(self, alpha, x),
            DType::F64 => update_axpy::<f64>(self, alpha, x),
            dtype => Err(Error::UnsupportedDTypeForOp { op: "axpy", dtype }),
        }
    }

    /// Scales the exclusive contiguous CPU F32/F64 view in place.
    /// A coefficient that converts to zero clears the view, including NaNs.
    pub fn scale(&mut self, alpha: f64) -> Result<()> {
        match self.dtype {
            DType::F32 => update_scale::<f32>(self, alpha),
            DType::F64 => update_scale::<f64>(self, alpha),
            dtype => Err(Error::UnsupportedDTypeForOp { op: "scale", dtype }),
        }
    }
}

fn mutable_values<'a, T: WithDType>(
    tensor: &'a mut ExclusiveTensor,
    op: &'static str,
) -> Result<&'a mut [T]> {
    if !tensor.layout.is_contiguous() {
        return Err(Error::UnsupportedLayoutForOp { op });
    }
    #[cfg(not(feature = "cuda"))]
    let Storage::Cpu(storage) = &mut tensor.storage;
    #[cfg(feature = "cuda")]
    let storage = match &mut tensor.storage {
        Storage::Cpu(storage) => storage,
        #[cfg(feature = "cuda")]
        Storage::Cuda(_) => return Err(Error::UnsupportedCudaOp { op }),
    };
    let values = T::cpu_storage_as_mut_slice(storage)?;
    validate_layout_for_storage(&tensor.layout, values.len())?;
    let count = tensor.layout.checked_elem_count()?;
    if count == 0 {
        return Ok(&mut values[..0]);
    }
    let start = tensor.layout.start_offset();
    Ok(&mut values[start..start + count])
}

fn update_axpy<T: LinearFloat>(tensor: &mut ExclusiveTensor, alpha: f64, x: &Tensor) -> Result<()> {
    let x = input::<T>(x, "axpy")?;
    linalg::axpy(mutable_values::<T>(tensor, "axpy")?, x, alpha);
    Ok(())
}

fn update_scale<T: LinearFloat>(tensor: &mut ExclusiveTensor, alpha: f64) -> Result<()> {
    linalg::scale(mutable_values::<T>(tensor, "scale")?, alpha);
    Ok(())
}
