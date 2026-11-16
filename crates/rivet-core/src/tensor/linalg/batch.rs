//! Explicit vector/matrix RHS APIs over the existing typed batch boundaries.
use super::extensions::broadcast_matrix_inputs;
use super::*;
use crate::cpu_backend::linalg::extensions::ExtendedFloat;

impl Tensor {
    /// Computes op(A)*x for op(A)=[...,M,K] and x=[...,K], broadcasting leading
    /// dimensions. transpose swaps A's last two axes before multiplication.
    /// CPU F32/F64 only; the result has shape [...,M].
    pub fn batch_mv(&self, x: &Self, transpose: bool) -> Result<Self> {
        self.check_linear_peer(x)?;
        dispatch!(self, "batch_mv", mv, self, x, None, transpose, 1.0, 0.0)
    }

    /// Computes beta*self + alpha*op(A)*x per broadcast batch. The addend must
    /// have the exact [...,M] result shape. Zero coefficients ignore NaNs.
    pub fn batch_addmv(
        &self,
        a: &Self,
        x: &Self,
        alpha: f64,
        beta: f64,
        transpose: bool,
    ) -> Result<Self> {
        self.check_linear_peer(a)?;
        self.check_linear_peer(x)?;
        dispatch!(
            self,
            "batch_addmv",
            mv,
            a,
            x,
            Some(self),
            transpose,
            alpha,
            beta
        )
    }

    /// Solves op(A)*X=B (Left) or X*op(A)=B (Right) with broadcast batch axes.
    /// A=[...,N,N]; matrix B=[...,N,R] on the left or [...,R,N] on the right.
    /// Triangle selection precedes transpose; unit diagonal values are ignored.
    /// A zero diagonal reports BatchedSingularMatrix with a flattened batch index.
    pub fn batch_triangular_solve(&self, rhs: &Self, options: TriangularOptions) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        dispatch!(
            self,
            "batch_triangular_solve",
            solve_result,
            self,
            rhs,
            options,
            false
        )
    }

    /// Vector RHS version of batch_triangular_solve: rhs=[...,N], result=[...,N].
    /// Right-side solves interpret these as row vectors. This explicit method
    /// avoids ambiguity between a matrix RHS and a batch of vectors.
    pub fn batch_triangular_solve_vector(
        &self,
        rhs: &Self,
        options: TriangularOptions,
    ) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        dispatch!(
            self,
            "batch_triangular_solve_vector",
            solve_result,
            self,
            rhs,
            options,
            true
        )
    }
}

impl ExclusiveTensor {
    /// Updates an existing contiguous [...,M] CPU F32/F64 destination with
    /// beta*self + alpha*op(A)*x. Inputs broadcast; the output shape is exact.
    pub fn batch_addmv(
        &mut self,
        a: &Tensor,
        x: &Tensor,
        alpha: f64,
        beta: f64,
        transpose: bool,
    ) -> Result<()> {
        self.check_peer(a)?;
        self.check_peer(x)?;
        dispatch_exclusive!(
            self,
            "batch_addmv",
            mv_into,
            self,
            a,
            x,
            alpha,
            beta,
            transpose
        )
    }

    /// Solves a matrix RHS in place, treating self as B. A may broadcast to
    /// the existing batch shape; the destination cannot expand. All diagonals
    /// are checked before writing, including those in later logical batches.
    pub fn batch_triangular_solve(&mut self, a: &Tensor, options: TriangularOptions) -> Result<()> {
        self.check_peer(a)?;
        dispatch_exclusive!(
            self,
            "batch_triangular_solve",
            solve_owned,
            self,
            a,
            options,
            false
        )
    }

    /// In-place vector RHS solve. self=[...,N]; A broadcasts to its batch axes.
    /// Supports offset views, but requires contiguous exclusive output storage.
    pub fn batch_triangular_solve_vector(
        &mut self,
        a: &Tensor,
        options: TriangularOptions,
    ) -> Result<()> {
        self.check_peer(a)?;
        dispatch_exclusive!(
            self,
            "batch_triangular_solve_vector",
            solve_owned,
            self,
            a,
            options,
            true
        )
    }
}

fn vector_shape(matrix: &[usize], side: MatrixSide) -> Shape {
    let mut dims = matrix.to_vec();
    let axis = dims.len() - if side == MatrixSide::Left { 1 } else { 2 };
    dims.remove(axis);
    dims.into()
}

fn prepare_mv(a: &Tensor, x: &Tensor, transpose: bool) -> Result<(Tensor, Tensor, Shape)> {
    if a.rank() < 2 || x.rank() < 1 {
        return Err(Error::MatmulShapeMismatch {
            lhs: a.dims().to_vec(),
            rhs: x.dims().to_vec(),
        });
    }
    let a = if transpose {
        a.transpose(a.rank() - 2, a.rank() - 1)?
    } else {
        a.clone()
    };
    let x = x.unsqueeze(x.rank())?;
    let (a, x) = broadcast_matrix_inputs(&a, &x)?;
    let rank = a.rank();
    let mut dims = a.dims()[..rank - 2].to_vec();
    dims.extend_from_slice(&[a.dims()[rank - 2], 1]);
    Ok((a, x, dims.into()))
}

fn mv<T: ExtendedFloat>(
    a: &Tensor,
    x: &Tensor,
    seed: Option<&Tensor>,
    transpose: bool,
    alpha: f64,
    beta: f64,
) -> Result<Tensor> {
    let (a, x, matrix_shape) = prepare_mv(a, x, transpose)?;
    let shape = vector_shape(matrix_shape.dims(), MatrixSide::Left);
    if let Some(seed) = seed
        && seed.shape() != &shape
    {
        return Err(Error::ShapeMismatchBinary {
            lhs: seed.dims().to_vec(),
            rhs: shape.dims().to_vec(),
        });
    }
    let seed = seed.map(|seed| seed.unsqueeze(seed.rank())).transpose()?;
    let add = seed
        .as_ref()
        .map(|seed| input::<T>(seed, "batch_addmv"))
        .transpose()?;
    finish(
        linalg::batched_fused(
            input::<T>(&a, "batch_mv")?,
            input::<T>(&x, "batch_mv")?,
            add,
            alpha,
            beta,
            false,
        )?,
        shape,
    )
}

fn mv_into<T: ExtendedFloat>(
    t: &mut ExclusiveTensor,
    a: &Tensor,
    x: &Tensor,
    alpha: f64,
    beta: f64,
    transpose: bool,
) -> Result<()> {
    let (a, x, matrix_shape) = prepare_mv(a, x, transpose)?;
    let shape = vector_shape(matrix_shape.dims(), MatrixSide::Left);
    if t.layout.shape() != &shape {
        return Err(Error::ShapeMismatchBinary {
            lhs: t.layout.dims().to_vec(),
            rhs: shape.dims().to_vec(),
        });
    }
    let a = input::<T>(&a, "batch_addmv")?;
    let x = input::<T>(&x, "batch_addmv")?;
    linalg::batched_into(
        mutable_values::<T>(t, "batch_addmv")?,
        matrix_shape.dims(),
        a,
        x,
        alpha,
        beta,
        false,
    )
}

fn matrix_rhs(rhs: &Tensor, side: MatrixSide, vector: bool) -> Result<Tensor> {
    let minimum_rank = if vector { 1 } else { 2 };
    if rhs.rank() < minimum_rank {
        return Err(Error::InvalidRank {
            expected: minimum_rank,
            actual: rhs.rank(),
        });
    }
    if vector {
        rhs.unsqueeze(rhs.rank() - usize::from(side == MatrixSide::Right))
    } else {
        Ok(rhs.clone())
    }
}

fn solve_result<T: ExtendedFloat>(
    a: &Tensor,
    rhs: &Tensor,
    options: TriangularOptions,
    vector: bool,
) -> Result<Tensor> {
    let rhs = matrix_rhs(rhs, options.side, vector)?;
    let (a, rhs) = broadcast_matrix_inputs(a, &rhs)?;
    let shape = if vector {
        vector_shape(rhs.dims(), options.side)
    } else {
        rhs.shape().clone()
    };
    finish(
        linalg::triangular_batch(
            input::<T>(&a, "batch_triangular_solve")?,
            input::<T>(&rhs, "batch_triangular_solve")?,
            options,
        )?,
        shape,
    )
}

fn solve_owned<T: ExtendedFloat>(
    t: &mut ExclusiveTensor,
    a: &Tensor,
    options: TriangularOptions,
    vector: bool,
) -> Result<()> {
    let minimum_rank = if vector { 1 } else { 2 };
    if t.layout.dims().len() < minimum_rank || a.rank() < 2 {
        return Err(Error::MatmulShapeMismatch {
            lhs: a.dims().to_vec(),
            rhs: t.layout.dims().to_vec(),
        });
    }
    let mut dims = t.layout.dims().to_vec();
    if vector {
        let axis = dims.len() - usize::from(options.side == MatrixSide::Right);
        dims.insert(axis, 1);
    }
    let rank = dims.len();
    let batch = Shape::from(a.dims()[..a.rank() - 2].to_vec())
        .broadcast_shape_binary_op(&Shape::from(dims[..rank - 2].to_vec()))?;
    if batch.dims() != &dims[..rank - 2] {
        let mut expanded = batch.dims().to_vec();
        expanded.extend_from_slice(&dims[rank - 2..]);
        let shape = if vector {
            vector_shape(&expanded, options.side)
        } else {
            expanded.into()
        };
        return Err(Error::ShapeMismatchBinary {
            lhs: t.layout.dims().to_vec(),
            rhs: shape.dims().to_vec(),
        });
    }
    let mut ashape = batch.dims().to_vec();
    ashape.extend_from_slice(&a.dims()[a.rank() - 2..]);
    let a = a.broadcast_as(ashape)?;
    let a = input::<T>(&a, "batch_triangular_solve")?;
    linalg::triangular_batch_into(
        mutable_values::<T>(t, "batch_triangular_solve")?,
        &dims,
        a,
        options,
    )
}
