use super::Tensor;
use crate::ops::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::storage::Storage;
use crate::{DType, Error, Result, Shape, WithDType};
use std::sync::Arc;

impl Tensor {
    fn with_two_storage<R>(
        lhs: &Self,
        rhs: &Self,
        f: impl FnOnce(&Storage, &Storage) -> Result<R>,
    ) -> Result<R> {
        let lhs_address = Arc::as_ptr(&lhs.0.storage) as usize;
        let rhs_address = Arc::as_ptr(&rhs.0.storage) as usize;
        if lhs_address == rhs_address {
            let storage = lhs.storage();
            return f(&storage, &storage);
        }

        if lhs_address < rhs_address {
            let lhs_storage = lhs.storage();
            let rhs_storage = rhs.storage();
            f(&lhs_storage, &rhs_storage)
        } else {
            let rhs_storage = rhs.storage();
            let lhs_storage = lhs.storage();
            f(&lhs_storage, &rhs_storage)
        }
    }

    fn binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self> {
        if self.shape() != rhs.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }

        let lhs_storage = self.storage();
        let rhs_storage = rhs.storage();
        let storage = Storage::binary(&lhs_storage, self.layout(), &rhs_storage, rhs.layout(), op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    fn broadcast_binary(&self, rhs: &Self, op: BinaryOp) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        let shape = self.shape().broadcast_shape_binary_op(rhs.shape())?;
        let lhs = self.broadcast_as(shape.clone())?;
        let rhs = rhs.broadcast_as(shape)?;
        lhs.binary(&rhs, op)
    }

    fn unary(&self, op: UnaryOp) -> Result<Self> {
        let storage = self.storage();
        let storage = Storage::unary(&storage, self.layout(), op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    fn binary_scalar<T: WithDType>(&self, scalar: T, op: BinaryOp) -> Result<Self> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: T::DTYPE,
            });
        }
        let storage = self.storage();
        let storage = Storage::binary_scalar(&storage, self.layout(), scalar, op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Add)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Sub)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Mul)
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, BinaryOp::Div)
    }

    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Add)
    }

    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Sub)
    }

    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Mul)
    }

    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        self.broadcast_binary(rhs, BinaryOp::Div)
    }

    pub fn add_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Add)
    }

    pub fn sub_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Sub)
    }

    pub fn mul_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Mul)
    }

    pub fn div_scalar<T: WithDType>(&self, rhs: T) -> Result<Self> {
        self.binary_scalar(rhs, BinaryOp::Div)
    }

    pub fn neg(&self) -> Result<Self> {
        self.unary(UnaryOp::Neg)
    }

    pub fn abs(&self) -> Result<Self> {
        self.unary(UnaryOp::Abs)
    }

    /// Applies `value * mul + add` element-wise in the tensor's dtype.
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage();
        let storage = Storage::affine(&storage, self.layout(), mul, add)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    /// Applies the Exponential Linear Unit function element-wise.
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage();
        let storage = Storage::elu(&storage, self.layout(), alpha)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    /// Raises every element to a scalar floating-point exponent.
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        if self.elem_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.storage();
        let storage = Storage::powf(&storage, self.layout(), exponent)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    /// Raises each element of `self` to the matching element of `rhs`.
    pub fn pow(&self, rhs: &Self) -> Result<Self> {
        if self.shape() != rhs.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        let storage = Self::with_two_storage(self, rhs, |lhs_storage, rhs_storage| {
            Storage::pow(lhs_storage, self.layout(), rhs_storage, rhs.layout())
        })?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    /// Broadcasting version of [`Tensor::pow`].
    pub fn broadcast_pow(&self, rhs: &Self) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        let shape = self.shape().broadcast_shape_binary_op(rhs.shape())?;
        let lhs = self.broadcast_as(shape.clone())?;
        let rhs = rhs.broadcast_as(shape.clone())?;
        lhs.pow(&rhs)
    }

    /// Computes the dot product of two same-shaped floating-point vectors.
    pub fn dot(&self, rhs: &Self) -> Result<Self> {
        if self.rank() != 1 || rhs.rank() != 1 || self.shape() != rhs.shape() {
            return Err(Error::ShapeMismatchBinary {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        let storage = Self::with_two_storage(self, rhs, |lhs_storage, rhs_storage| {
            Storage::dot(lhs_storage, self.layout(), rhs_storage, rhs.layout())
        })?;
        Self::from_storage(storage, Shape::from(()), self.device())
    }

    /// Computes the Frobenius/L2 norm of all elements.
    pub fn norm(&self) -> Result<Self> {
        let storage = self.storage();
        let storage = Storage::norm(&storage, self.layout())?;
        Self::from_storage(storage, Shape::from(()), self.device())
    }

    /// Computes a rank-2 F32 matrix product using the CPU GEMM backend.
    ///
    /// Phase 1 deliberately accepts only contiguous rank-2 inputs. The
    /// output is always a fresh row-major contiguous tensor.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        if self.rank() != 2 || rhs.rank() != 2 || self.dims()[1] != rhs.dims()[0] {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }

        let output_shape = Shape::from([self.dims()[0], rhs.dims()[1]]);
        let storage = Self::with_two_storage(self, rhs, |lhs_storage, rhs_storage| {
            Storage::matmul(lhs_storage, self.layout(), rhs_storage, rhs.layout())
        })?;
        Self::from_storage(storage, output_shape, self.device())
    }

    /// Performs strict matrix-vector multiplication: `[m, n] * [n] = [m]`.
    pub fn mv(&self, rhs: &Self) -> Result<Self> {
        if self.rank() != 2 || rhs.rank() != 1 || self.dims()[1] != rhs.dims()[0] {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }
        self.matmul(&rhs.unsqueeze(1)?)?.squeeze(1)
    }

    /// Performs matrix multiplication after broadcasting leading batch
    /// dimensions. Matrix dimensions remain strict: `[... , m, k]` times
    /// `[... , k, n]` produces `[broadcast(...), m, n]`.
    pub fn broadcast_matmul(&self, rhs: &Self) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        if self.rank() < 2 || rhs.rank() < 2 {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }

        let lhs_dims = self.dims();
        let rhs_dims = rhs.dims();
        let lhs_matrix = [lhs_dims[lhs_dims.len() - 2], lhs_dims[lhs_dims.len() - 1]];
        let rhs_matrix = [rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1]];
        if lhs_matrix[1] != rhs_matrix[0] {
            return Err(Error::MatmulShapeMismatch {
                lhs: self.dims().to_vec(),
                rhs: rhs.dims().to_vec(),
            });
        }

        let lhs_batch = Shape::from(lhs_dims[..lhs_dims.len() - 2].to_vec());
        let rhs_batch = Shape::from(rhs_dims[..rhs_dims.len() - 2].to_vec());
        let batch = lhs_batch.broadcast_shape_binary_op(&rhs_batch)?;
        let mut lhs_broadcast_dims = batch.dims().to_vec();
        lhs_broadcast_dims.extend_from_slice(&lhs_matrix);
        let mut rhs_broadcast_dims = batch.dims().to_vec();
        rhs_broadcast_dims.extend_from_slice(&rhs_matrix);

        let lhs = self
            .broadcast_as(lhs_broadcast_dims.clone())?
            .contiguous()?;
        let rhs = rhs.broadcast_as(rhs_broadcast_dims.clone())?.contiguous()?;
        let batch_count = batch.elem_count();
        let mut output_dims = batch.dims().to_vec();
        output_dims.extend_from_slice(&[lhs_matrix[0], rhs_matrix[1]]);
        if batch_count == 0 {
            return Self::zeros(output_dims, self.dtype(), self.device());
        }

        let lhs = lhs.reshape((batch_count, lhs_matrix[0], lhs_matrix[1]))?;
        let rhs = rhs.reshape((batch_count, rhs_matrix[0], rhs_matrix[1]))?;
        let mut products = Vec::with_capacity(batch_count);
        for batch_index in 0..batch_count {
            products.push(lhs.get(batch_index)?.matmul(&rhs.get(batch_index)?)?);
        }
        let product_refs = products.iter().collect::<Vec<_>>();
        let result = Self::stack(&product_refs, 0)?;
        result.reshape(output_dims)
    }

    fn reduce_dim(&self, dim: usize, keepdim: bool, op: ReduceOp) -> Result<Self> {
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::reduce_dim(&storage, self.layout(), dim, keepdim, op)?;
        let mut dims = self.dims().to_vec();
        if keepdim {
            dims[dim] = 1;
        } else {
            dims.remove(dim);
        }
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    fn reduce_all(&self, op: ReduceOp) -> Result<Self> {
        let storage = self.storage();
        let storage = Storage::reduce_all(&storage, self.layout(), op)?;
        Self::from_storage(storage, Shape::from(()), self.device())
    }

    pub fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, true, ReduceOp::Sum)
    }

    pub fn sum(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, false, ReduceOp::Sum)
    }

    pub fn sum_all(&self) -> Result<Self> {
        self.reduce_all(ReduceOp::Sum)
    }

    pub fn mean_keepdim(&self, dim: usize) -> Result<Self> {
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::mean_dim(&storage, self.layout(), dim, true)?;
        let mut dims = self.dims().to_vec();
        dims[dim] = 1;
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    pub fn mean(&self, dim: usize) -> Result<Self> {
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::mean_dim(&storage, self.layout(), dim, false)?;
        let mut dims = self.dims().to_vec();
        dims.remove(dim);
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    pub fn mean_all(&self) -> Result<Self> {
        let storage = self.storage();
        let storage = Storage::mean_all(&storage, self.layout())?;
        Self::from_storage(storage, Shape::from(()), self.device())
    }

    /// Computes cumulative sums along one dimension.
    pub fn cumsum(&self, dim: usize) -> Result<Self> {
        if self.rank() == 0 {
            return Ok(self.clone());
        }
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::cumsum(&storage, self.layout(), dim)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    fn log_sum_exp_dim(&self, dim: usize) -> Result<Self> {
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::log_sum_exp(&storage, self.layout(), dim)?;
        let mut dims = self.dims().to_vec();
        dims.remove(dim);
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    /// Computes a numerically stable log-sum-exp over the listed dimensions.
    /// Dimensions are reduced in descending order and the result removes them.
    pub fn log_sum_exp(&self, dims: &[usize]) -> Result<Self> {
        if dims.is_empty() {
            return Ok(self.clone());
        }
        let mut dims = dims.to_vec();
        dims.sort_unstable();
        dims.dedup();
        for &dim in &dims {
            self.dim(dim)?;
        }
        let mut result = self.clone();
        for dim in dims.into_iter().rev() {
            result = result.log_sum_exp_dim(dim)?;
        }
        Ok(result)
    }

    pub fn min_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, true, ReduceOp::Min)
    }

    pub fn min(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, false, ReduceOp::Min)
    }

    pub fn min_all(&self) -> Result<Self> {
        self.reduce_all(ReduceOp::Min)
    }

    pub fn max_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, true, ReduceOp::Max)
    }

    pub fn max(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, false, ReduceOp::Max)
    }

    pub fn max_all(&self) -> Result<Self> {
        self.reduce_all(ReduceOp::Max)
    }

    pub fn argmin_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, true, ReduceOp::ArgMin)
    }

    pub fn argmin(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, false, ReduceOp::ArgMin)
    }

    pub fn argmax_keepdim(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, true, ReduceOp::ArgMax)
    }

    pub fn argmax(&self, dim: usize) -> Result<Self> {
        self.reduce_dim(dim, false, ReduceOp::ArgMax)
    }

    pub fn var_keepdim(&self, dim: usize) -> Result<Self> {
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::var_dim(&storage, self.layout(), dim, true)?;
        let mut dims = self.dims().to_vec();
        dims[dim] = 1;
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    pub fn var(&self, dim: usize) -> Result<Self> {
        self.dim(dim)?;
        let storage = self.storage();
        let storage = Storage::var_dim(&storage, self.layout(), dim, false)?;
        let mut dims = self.dims().to_vec();
        dims.remove(dim);
        Self::from_storage(storage, Shape::from(dims), self.device())
    }

    fn cmp_tensor(&self, rhs: &Self, op: CmpOp) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: rhs.dtype(),
            });
        }
        if !self.device().same_device(rhs.device()) {
            return Err(Error::DeviceMismatch);
        }
        let shape = self.shape().broadcast_shape_binary_op(rhs.shape())?;
        let lhs = self.broadcast_as(shape.clone())?;
        let rhs = rhs.broadcast_as(shape.clone())?;
        let lhs_storage = lhs.storage();
        let rhs_storage = rhs.storage();
        let storage = Storage::cmp(&lhs_storage, lhs.layout(), &rhs_storage, rhs.layout(), op)?;
        Self::from_storage(storage, shape, self.device())
    }

    fn cmp_scalar_tensor<T: WithDType>(&self, scalar: T, op: CmpOp) -> Result<Self> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: T::DTYPE,
            });
        }
        let storage = self.storage();
        let storage = Storage::cmp_scalar(&storage, self.layout(), scalar, op)?;
        Self::from_storage(storage, self.shape().clone(), self.device())
    }

    pub fn cmp(&self, rhs: &Self, op: CmpOp) -> Result<Self> {
        self.cmp_tensor(rhs, op)
    }

    pub fn cmp_scalar<T: WithDType>(&self, scalar: T, op: CmpOp) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, op)
    }

    pub fn eq(&self, rhs: &Self) -> Result<Self> {
        self.cmp_tensor(rhs, CmpOp::Eq)
    }

    pub fn eq_scalar<T: WithDType>(&self, scalar: T) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, CmpOp::Eq)
    }

    pub fn ne(&self, rhs: &Self) -> Result<Self> {
        self.cmp_tensor(rhs, CmpOp::Ne)
    }

    pub fn ne_scalar<T: WithDType>(&self, scalar: T) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, CmpOp::Ne)
    }

    pub fn lt(&self, rhs: &Self) -> Result<Self> {
        self.cmp_tensor(rhs, CmpOp::Lt)
    }

    pub fn lt_scalar<T: WithDType>(&self, scalar: T) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, CmpOp::Lt)
    }

    pub fn le(&self, rhs: &Self) -> Result<Self> {
        self.cmp_tensor(rhs, CmpOp::Le)
    }

    pub fn le_scalar<T: WithDType>(&self, scalar: T) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, CmpOp::Le)
    }

    pub fn gt(&self, rhs: &Self) -> Result<Self> {
        self.cmp_tensor(rhs, CmpOp::Gt)
    }

    pub fn gt_scalar<T: WithDType>(&self, scalar: T) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, CmpOp::Gt)
    }

    pub fn ge(&self, rhs: &Self) -> Result<Self> {
        self.cmp_tensor(rhs, CmpOp::Ge)
    }

    pub fn ge_scalar<T: WithDType>(&self, scalar: T) -> Result<Self> {
        self.cmp_scalar_tensor(scalar, CmpOp::Ge)
    }

    pub fn clamp<T: WithDType>(&self, min: T, max: T) -> Result<Self> {
        self.binary_scalar(min, BinaryOp::Maximum)?
            .binary_scalar(max, BinaryOp::Minimum)
    }

    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        if self.dtype() != DType::U8 {
            return Err(Error::UnexpectedDType {
                expected: DType::U8,
                actual: self.dtype(),
            });
        }
        if on_true.dtype() != on_false.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: on_true.dtype(),
                rhs: on_false.dtype(),
            });
        }
        if !self.device().same_device(on_true.device())
            || !self.device().same_device(on_false.device())
        {
            return Err(Error::DeviceMismatch);
        }
        let shape = self
            .shape()
            .broadcast_shape_binary_op(on_true.shape())?
            .broadcast_shape_binary_op(on_false.shape())?;
        let condition = self.broadcast_as(shape.clone())?;
        let on_true = on_true.broadcast_as(shape.clone())?;
        let on_false = on_false.broadcast_as(shape.clone())?;
        let condition_storage = condition.storage();
        let true_storage = on_true.storage();
        let false_storage = on_false.storage();
        let storage = Storage::where_cond(
            &condition_storage,
            condition.layout(),
            &true_storage,
            on_true.layout(),
            &false_storage,
            on_false.layout(),
        )?;
        Self::from_storage(storage, shape, self.device())
    }

    pub fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        let first = tensors.first().ok_or(Error::EmptyTensorList)?;
        let rank = first.rank();
        if dim >= rank {
            return Err(Error::InvalidConcatDim { dim, rank });
        }

        let mut output_dims = first.dims().to_vec();
        output_dims[dim] = 0;
        for tensor in tensors {
            if tensor.rank() != rank {
                return Err(Error::ShapeMismatchBinary {
                    lhs: first.dims().to_vec(),
                    rhs: tensor.dims().to_vec(),
                });
            }
            if tensor.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch {
                    lhs: first.dtype(),
                    rhs: tensor.dtype(),
                });
            }
            if !tensor.device().same_device(first.device()) {
                return Err(Error::DeviceMismatch);
            }
            for axis in 0..rank {
                if axis != dim && tensor.dims()[axis] != first.dims()[axis] {
                    return Err(Error::ShapeMismatchBinary {
                        lhs: first.dims().to_vec(),
                        rhs: tensor.dims().to_vec(),
                    });
                }
            }
            output_dims[dim] = output_dims[dim]
                .checked_add(tensor.dims()[dim])
                .ok_or(Error::StorageOutOfBounds)?;
        }

        let guards: Vec<_> = tensors.iter().map(|tensor| tensor.storage()).collect();
        let inputs: Vec<_> = guards
            .iter()
            .zip(tensors.iter())
            .map(|(storage, tensor)| (&**storage, tensor.layout()))
            .collect();
        let output_shape = Shape::from(output_dims);
        let storage = Storage::cat(&inputs, &output_shape, dim)?;
        Self::from_storage(storage, output_shape, first.device())
    }

    pub fn stack(tensors: &[&Self], dim: usize) -> Result<Self> {
        let first = tensors.first().ok_or(Error::EmptyTensorList)?;
        if dim > first.rank() {
            return Err(Error::InvalidConcatDim {
                dim,
                rank: first.rank() + 1,
            });
        }
        let expanded: Vec<Self> = tensors
            .iter()
            .map(|tensor| tensor.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        let expanded_refs: Vec<&Self> = expanded.iter().collect();
        Self::cat(&expanded_refs, dim)
    }
}
