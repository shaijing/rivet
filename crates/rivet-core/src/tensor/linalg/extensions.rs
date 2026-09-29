//! Public CPU F32/F64 extensions over the typed linear algebra kernels.
use super::*;
use crate::Layout;
use crate::cpu_backend::linalg::extensions::{
    self as kernels, ExtendedFloat, RankOptions, RankUpdate,
};

macro_rules! dispatch_exclusive {
    ($tensor:expr,$op:literal,$fun:ident $(,$arg:expr)*)=> {
        match $tensor.dtype {
            DType::F32=>$fun::<f32>($($arg),*),
            DType::F64=>$fun::<f64>($($arg),*),
            dtype=>Err(Error::UnsupportedDTypeForOp {op:$op,dtype}),
        }
    }
}

impl Tensor {
    /// Computes beta*C + alpha*x*x^T. Reads only C's selected triangle
    /// and returns a complete symmetric matrix.
    pub fn symmetric_rank1_update(
        &self,
        x: &Self,
        alpha: f64,
        beta: f64,
        upper: bool,
    ) -> Result<Self> {
        self.check_linear_peer(x)?;
        dispatch!(
            self,
            "symmetric_rank1_update",
            rank,
            self,
            x,
            None,
            RankOptions {
                kind: RankUpdate::One,
                transpose: false,
                upper,
                alpha,
                beta
            }
        )
    }
    /// Computes beta*C + alpha*(x*y^T + y*x^T), returning full symmetry.
    pub fn symmetric_rank2_update(
        &self,
        x: &Self,
        y: &Self,
        alpha: f64,
        beta: f64,
        upper: bool,
    ) -> Result<Self> {
        self.check_linear_peer(x)?;
        self.check_linear_peer(y)?;
        dispatch!(
            self,
            "symmetric_rank2_update",
            rank,
            self,
            x,
            Some(y),
            RankOptions {
                kind: RankUpdate::Two,
                transpose: false,
                upper,
                alpha,
                beta
            }
        )
    }
    /// Computes beta*C + alpha*A*A^T, or alpha*A^T*A when transpose=true.
    pub fn gram_update(
        &self,
        a: &Self,
        transpose: bool,
        alpha: f64,
        beta: f64,
        upper: bool,
    ) -> Result<Self> {
        self.check_linear_peer(a)?;
        dispatch!(
            self,
            "gram_update",
            rank,
            self,
            a,
            None,
            RankOptions {
                kind: RankUpdate::Gram,
                transpose,
                upper,
                alpha,
                beta
            }
        )
    }
    /// Computes beta*C + alpha*(A*B^T+B*A^T); transpose=true instead
    /// uses A^T*B+B^T*A. A and B must have the same shape.
    pub fn symmetric_rank2k_update(
        &self,
        a: &Self,
        b: &Self,
        transpose: bool,
        alpha: f64,
        beta: f64,
        upper: bool,
    ) -> Result<Self> {
        self.check_linear_peer(a)?;
        self.check_linear_peer(b)?;
        dispatch!(
            self,
            "symmetric_rank2k_update",
            rank,
            self,
            a,
            Some(b),
            RankOptions {
                kind: RankUpdate::TwoK,
                transpose,
                upper,
                alpha,
                beta
            }
        )
    }
    /// Adds a matrix product per broadcast batch. The addend must exactly
    /// match the resulting batch shape; no addend broadcasting occurs.
    pub fn baddbmm(&self, a: &Self, b: &Self, alpha: f64, beta: f64) -> Result<Self> {
        self.check_linear_peer(a)?;
        self.check_linear_peer(b)?;
        let (a, b, shape) = broadcast_inputs(a, b, false)?;
        dispatch!(
            self, "baddbmm", batch, self, &a, &b, shape, alpha, beta, false
        )
    }
    /// Adds the sum of all products across the broadcast batch dimensions.
    /// Writes every product directly into one matrix allocation.
    pub fn addbmm(&self, a: &Self, b: &Self, alpha: f64, beta: f64) -> Result<Self> {
        self.check_linear_peer(a)?;
        self.check_linear_peer(b)?;
        let (a, b, shape) = broadcast_inputs(a, b, true)?;
        dispatch!(
            self, "addbmm", batch, self, &a, &b, shape, alpha, beta, true
        )
    }
    /// Computes A*B or B*A using the chosen triangle of symmetric A=self.
    pub fn symmetric_matmul_with_side(
        &self,
        rhs: &Self,
        upper: bool,
        side: MatrixSide,
    ) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        let options = TriangularOptions {
            upper,
            side,
            ..Default::default()
        };
        dispatch!(
            self,
            "symmetric_matmul_with_side",
            structured,
            self,
            rhs,
            options,
            false,
            true
        )
    }
    /// Left-side triangular multiplication; unused triangle is ignored.
    pub fn triangular_matmul(&self, rhs: &Self, upper: bool, unit_diagonal: bool) -> Result<Self> {
        self.triangular_matmul_with_options(
            rhs,
            TriangularOptions {
                upper,
                unit_diagonal,
                ..Default::default()
            },
        )
    }
    pub fn triangular_matmul_with_options(
        &self,
        rhs: &Self,
        options: TriangularOptions,
    ) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        dispatch!(
            self,
            "triangular_matmul",
            structured,
            self,
            rhs,
            options,
            false,
            false
        )
    }
    /// Left-side triangular matrix-vector product, using TRMV when possible.
    pub fn triangular_mv(&self, rhs: &Self, upper: bool, unit_diagonal: bool) -> Result<Self> {
        if rhs.rank() != 1 {
            return Err(Error::InvalidRank {
                expected: 1,
                actual: rhs.rank(),
            });
        }
        self.triangular_matmul(&rhs.unsqueeze(1)?, upper, unit_diagonal)?
            .squeeze(1)
    }
    /// Solves op(A)*X=B or X*op(A)=B. Triangle refers to A before
    /// transposition. A vector RHS is supported for left-side solves.
    pub fn triangular_solve_with_options(
        &self,
        rhs: &Self,
        options: TriangularOptions,
    ) -> Result<Self> {
        self.check_linear_peer(rhs)?;
        if rhs.rank() == 1 && options.side == MatrixSide::Left {
            return self
                .triangular_solve_with_options(&rhs.unsqueeze(1)?, options)?
                .squeeze(1);
        }
        dispatch!(
            self,
            "triangular_solve",
            structured,
            self,
            rhs,
            options,
            true,
            false
        )
    }
    /// Makes an independent contiguous copy, optionally transposing and
    /// scaling. A zero coefficient clears the output without reading values.
    pub fn matrix_copy(&self, transpose: bool, alpha: f64) -> Result<Self> {
        dispatch!(self, "matrix_copy", matrix_copy, self, transpose, alpha)
    }
    /// Returns a flattened logical index. Ties choose the first index;
    /// any NaN chooses the first NaN, consistently across BLAS providers.
    /// Empty tensors are rejected. CPU F32/F64 only.
    pub fn argmax_abs(&self) -> Result<usize> {
        dispatch!(self, "argmax_abs", argmax_abs, self)
    }
}

fn rank<T: ExtendedFloat>(
    c: &Tensor,
    a: &Tensor,
    b: Option<&Tensor>,
    options: RankOptions,
) -> Result<Tensor> {
    finish(
        kernels::rank_update(
            input::<T>(c, "rank_update")?,
            input::<T>(a, "rank_update")?,
            b.map(|b| input::<T>(b, "rank_update")).transpose()?,
            options,
        )?,
        c.shape().clone(),
    )
}

fn broadcast_inputs(a: &Tensor, b: &Tensor, reduce: bool) -> Result<(Tensor, Tensor, Shape)> {
    if a.rank() < 2 || b.rank() < 2 {
        return Err(Error::MatmulShapeMismatch {
            lhs: a.dims().to_vec(),
            rhs: b.dims().to_vec(),
        });
    }
    let ad = a.dims();
    let bd = b.dims();
    let batch = Shape::from(ad[..ad.len() - 2].to_vec())
        .broadcast_shape_binary_op(&Shape::from(bd[..bd.len() - 2].to_vec()))?;
    let mut ashape = batch.dims().to_vec();
    ashape.extend_from_slice(&ad[ad.len() - 2..]);
    let mut bshape = batch.dims().to_vec();
    bshape.extend_from_slice(&bd[bd.len() - 2..]);
    let mut shape = if reduce {
        Vec::new()
    } else {
        batch.dims().to_vec()
    };
    shape.extend_from_slice(&[ad[ad.len() - 2], bd[bd.len() - 1]]);
    Ok((
        a.broadcast_as(ashape)?,
        b.broadcast_as(bshape)?,
        shape.into(),
    ))
}
fn batch<T: ExtendedFloat>(
    c: &Tensor,
    a: &Tensor,
    b: &Tensor,
    shape: Shape,
    alpha: f64,
    beta: f64,
    reduce: bool,
) -> Result<Tensor> {
    finish(
        linalg::batched_fused(
            input::<T>(a, "batch")?,
            input::<T>(b, "batch")?,
            Some(input::<T>(c, "batch")?),
            alpha,
            beta,
            reduce,
        )?,
        shape,
    )
}
fn structured<T: ExtendedFloat>(
    a: &Tensor,
    b: &Tensor,
    o: TriangularOptions,
    solve: bool,
    symmetric: bool,
) -> Result<Tensor> {
    finish(
        kernels::structured(
            input::<T>(a, "structured")?,
            input::<T>(b, "structured")?,
            o,
            solve,
            symmetric,
        )?,
        b.shape().clone(),
    )
}
fn matrix_copy<T: ExtendedFloat>(x: &Tensor, transpose: bool, alpha: f64) -> Result<Tensor> {
    let out = kernels::matrix_copy(input::<T>(x, "matrix_copy")?, transpose, alpha)?;
    let dims = x.dims();
    let shape = if transpose {
        (dims[1], dims[0])
    } else {
        (dims[0], dims[1])
    };
    finish(out, shape)
}
fn argmax_abs<T: ExtendedFloat>(x: &Tensor) -> Result<usize> {
    kernels::argmax_abs(input::<T>(x, "argmax_abs")?)
}

impl ExclusiveTensor {
    fn check_peer(&self, x: &Tensor) -> Result<()> {
        if self.dtype != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype,
                rhs: x.dtype(),
            });
        }
        if !self.device.same_device(x.device()) {
            return Err(Error::DeviceMismatch);
        }
        Ok(())
    }
    fn check_shape_peer(&self, x: &Tensor) -> Result<()> {
        self.check_peer(x)?;
        if self.layout.shape() != x.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.layout.dims().to_vec(),
                rhs: x.dims().to_vec(),
            });
        }
        Ok(())
    }
    /// Updates an exclusive contiguous matrix without allocating an output.
    pub fn addmm(&mut self, a: &Tensor, b: &Tensor, alpha: f64, beta: f64) -> Result<()> {
        self.check_peer(a)?;
        self.check_peer(b)?;
        dispatch_exclusive!(self, "addmm", update_mm, self, a, b, alpha, beta)
    }
    /// Updates an exclusive contiguous vector without allocating an output.
    pub fn addmv(&mut self, a: &Tensor, x: &Tensor, alpha: f64, beta: f64) -> Result<()> {
        self.check_peer(a)?;
        self.check_peer(x)?;
        dispatch_exclusive!(self, "addmv", update_mv, self, a, x, alpha, beta)
    }
    /// Computes beta*self + alpha*x*y^T in the existing allocation.
    pub fn addr(&mut self, x: &Tensor, y: &Tensor, alpha: f64, beta: f64) -> Result<()> {
        self.check_peer(x)?;
        self.check_peer(y)?;
        dispatch_exclusive!(self, "addr", update_outer, self, x, y, alpha, beta)
    }
    /// Fused self <- alpha*x + beta*self. Zero coefficients ignore their
    /// operand values, including NaNs. Exact peer shapes are required.
    pub fn axpby(&mut self, alpha: f64, x: &Tensor, beta: f64) -> Result<()> {
        self.check_shape_peer(x)?;
        dispatch_exclusive!(self, "axpby", update_axpby, self, alpha, x, beta)
    }
    /// Copies a matching tensor's logical values into the existing view.
    pub fn copy_from(&mut self, x: &Tensor) -> Result<()> {
        self.check_shape_peer(x)?;
        dispatch_exclusive!(self, "copy_from", update_copy, self, x)
    }
    /// Applies x'=c*x+s*y and y'=c*y-s*x to two separate exclusive views.
    pub fn rotate(&mut self, y: &mut Self, c: f64, s: f64) -> Result<()> {
        self.check_exclusive_peer(y)?;
        dispatch_exclusive!(self, "rotate", update_rotate, self, y, c, s)
    }
    pub fn swap(&mut self, y: &mut Self) -> Result<()> {
        self.check_exclusive_peer(y)?;
        dispatch_exclusive!(self, "swap", update_swap, self, y)
    }
    fn check_exclusive_peer(&self, y: &Self) -> Result<()> {
        if self.dtype != y.dtype {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype,
                rhs: y.dtype,
            });
        }
        if !self.device.same_device(&y.device) {
            return Err(Error::DeviceMismatch);
        }
        if self.layout.shape() != y.layout.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.layout.dims().to_vec(),
                rhs: y.layout.dims().to_vec(),
            });
        }
        Ok(())
    }
}

fn matrix_dims(t: &ExclusiveTensor) -> Result<[usize; 2]> {
    let [m, n] = *t.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 2,
            actual: t.layout.dims().len(),
        });
    };
    Ok([m, n])
}
fn update_mm<T: ExtendedFloat>(
    t: &mut ExclusiveTensor,
    a: &Tensor,
    b: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    let dims = matrix_dims(t)?;
    let a = input::<T>(a, "addmm")?;
    let b = input::<T>(b, "addmm")?;
    kernels::matmul_into(mutable_values::<T>(t, "addmm")?, &dims, a, b, alpha, beta)
}
fn update_mv<T: ExtendedFloat>(
    t: &mut ExclusiveTensor,
    a: &Tensor,
    x: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    let [m] = *t.layout.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: t.layout.dims().len(),
        });
    };
    let [k] = *x.dims() else {
        return Err(Error::InvalidRank {
            expected: 1,
            actual: x.rank(),
        });
    };
    let layout = Layout::new(
        (k, 1).into(),
        vec![x.stride()[0], 1],
        x.layout().start_offset(),
    )?;
    let x = Input::new(values::<T>(x, "addmv")?, &layout)?;
    let a = input::<T>(a, "addmv")?;
    kernels::matmul_into(mutable_values::<T>(t, "addmv")?, &[m, 1], a, x, alpha, beta)
}
fn update_outer<T: ExtendedFloat>(
    t: &mut ExclusiveTensor,
    x: &Tensor,
    y: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    let dims = matrix_dims(t)?;
    let x = input::<T>(x, "addr")?;
    let y = input::<T>(y, "addr")?;
    kernels::outer_into(mutable_values::<T>(t, "addr")?, &dims, x, y, alpha, beta)
}
fn update_axpby<T: ExtendedFloat>(
    t: &mut ExclusiveTensor,
    alpha: f64,
    x: &Tensor,
    beta: f64,
) -> Result<()> {
    let x = input::<T>(x, "axpby")?;
    kernels::axpby(mutable_values::<T>(t, "axpby")?, x, alpha, beta);
    Ok(())
}
fn update_copy<T: ExtendedFloat>(t: &mut ExclusiveTensor, x: &Tensor) -> Result<()> {
    let x = input::<T>(x, "copy_from")?;
    kernels::copy_into(mutable_values::<T>(t, "copy_from")?, x, false, 1.0);
    Ok(())
}
fn update_rotate<T: ExtendedFloat>(
    x: &mut ExclusiveTensor,
    y: &mut ExclusiveTensor,
    c: f64,
    s: f64,
) -> Result<()> {
    kernels::rotate(
        mutable_values::<T>(x, "rotate")?,
        mutable_values::<T>(y, "rotate")?,
        c,
        s,
    );
    Ok(())
}
fn update_swap<T: ExtendedFloat>(x: &mut ExclusiveTensor, y: &mut ExclusiveTensor) -> Result<()> {
    kernels::swap(
        mutable_values::<T>(x, "swap")?,
        mutable_values::<T>(y, "swap")?,
    );
    Ok(())
}

/// Constructs a Givens rotation: returns (c,s,r) with c*a+s*b=r and
/// c*b-s*a=0. Rejects nonfinite inputs; intermediate scaling avoids overflow.
pub fn givens_rotation(a: f64, b: f64) -> Option<(f64, f64, f64)> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    Some(kernels::givens(a, b))
}
