use super::Tensor;
use crate::ops::{BinaryOp, UnaryOp};
use crate::storage::Storage;
use crate::{Error, Result, Shape, WithDType};

impl Tensor {
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
