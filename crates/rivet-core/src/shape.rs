/// The dimensions of a tensor. An empty shape represents a scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Row-major element strides for this shape.
    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.rank());
        let mut stride = 1usize;
        for &dim in self.dims.iter().rev() {
            strides.push(stride);
            stride = stride.saturating_mul(dim);
        }
        strides.reverse();
        strides
    }

    pub fn is_contiguous(&self, stride: &[usize]) -> bool {
        if self.rank() != stride.len() {
            return false;
        }
        let mut expected = 1usize;
        for (&dim, &actual) in self.dims.iter().zip(stride).rev() {
            if dim > 1 && actual != expected {
                return false;
            }
            expected = expected.saturating_mul(dim);
        }
        true
    }

    pub fn is_fortran_contiguous(&self, stride: &[usize]) -> bool {
        if self.rank() != stride.len() {
            return false;
        }
        let mut expected = 1usize;
        for (&dim, &actual) in self.dims.iter().zip(stride) {
            if dim > 1 && actual != expected {
                return false;
            }
            expected = expected.saturating_mul(dim);
        }
        true
    }

    pub fn into_dims(self) -> Vec<usize> {
        self.dims
    }

    /// Computes the right-aligned broadcast shape for two tensors.
    pub fn broadcast_shape_binary_op(&self, rhs_shape: &Self) -> crate::Result<Self> {
        let rank = self.rank().max(rhs_shape.rank());
        let lhs_leading = rank - self.rank();
        let rhs_leading = rank - rhs_shape.rank();
        let mut dims = Vec::with_capacity(rank);
        for index in 0..rank {
            let lhs = if index < lhs_leading {
                1
            } else {
                self.dims[index - lhs_leading]
            };
            let rhs = if index < rhs_leading {
                1
            } else {
                rhs_shape.dims[index - rhs_leading]
            };
            if lhs == rhs {
                dims.push(lhs);
            } else if lhs == 1 {
                dims.push(rhs);
            } else if rhs == 1 {
                dims.push(lhs);
            } else {
                return Err(crate::Error::InvalidBroadcast {
                    lhs: self.dims.clone(),
                    rhs: rhs_shape.dims.clone(),
                });
            }
        }
        Ok(Self::new(dims))
    }
}

impl From<()> for Shape {
    fn from((): ()) -> Self {
        Self::new(Vec::new())
    }
}

impl From<usize> for Shape {
    fn from(dim: usize) -> Self {
        Self::new(vec![dim])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(dims: &[usize; N]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl From<&Shape> for Shape {
    fn from(shape: &Shape) -> Self {
        shape.clone()
    }
}

macro_rules! impl_shape_tuple {
    (($($ty:ty),+), $($name:tt),+) => {
        impl From<($($ty,)+)> for Shape {
            fn from(dims: ($($ty,)+)) -> Self {
                Self::new(vec![$(dims.$name,)+])
            }
        }
    };
}

impl_shape_tuple!((usize, usize), 0, 1);
impl From<(usize,)> for Shape {
    fn from(dims: (usize,)) -> Self {
        Self::new(vec![dims.0])
    }
}
impl_shape_tuple!((usize, usize, usize), 0, 1, 2);
impl_shape_tuple!((usize, usize, usize, usize), 0, 1, 2, 3);
impl_shape_tuple!((usize, usize, usize, usize, usize), 0, 1, 2, 3, 4);
impl_shape_tuple!((usize, usize, usize, usize, usize, usize), 0, 1, 2, 3, 4, 5);
