use crate::{Error, Result};
use std::alloc::{self, Layout};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ops::Deref;
use std::ptr::{self, NonNull};

/// Alignment guaranteed for every non-empty Rivet-owned CPU allocation.
pub(crate) const CPU_STORAGE_ALIGNMENT: usize = 256;

/// A fixed-length, fully initialized, immutable-after-publication allocation.
///
/// The allocation address is stable for the lifetime of the buffer. Mutable
/// construction is intentionally exposed through [`AlignedBufferBuilder`]
/// instead of this type.
#[derive(Debug)]
pub(crate) struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T> AlignedBuffer<T> {
    pub(crate) fn from_slice(values: &[T]) -> Result<Self>
    where
        T: Copy,
    {
        let mut builder = AlignedBufferBuilder::new(values.len())?;
        builder.extend_from_slice(values)?;
        builder.finish()
    }

    pub(crate) fn from_vec(values: Vec<T>) -> Result<Self>
    where
        T: Copy,
    {
        Self::from_slice(&values)
    }

    pub(crate) fn try_clone(&self) -> Result<Self>
    where
        T: Copy,
    {
        Self::from_slice(self.as_slice())
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: a finished buffer contains exactly `len` initialized T
        // values, and `ptr` remains valid until this buffer is dropped.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn alignment(&self) -> usize {
        allocation_alignment::<T>()
    }

    pub(crate) fn is_aligned_to(&self, alignment: usize) -> bool {
        self.is_empty() || (alignment != 0 && (self.as_ptr() as usize) % alignment == 0)
    }
}

impl<T> AsRef<[T]> for AlignedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> Deref for AlignedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy> Clone for AlignedBuffer<T> {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("AlignedBuffer clone allocation failed")
    }
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        // SAFETY: all `len` elements are initialized after `finish`, and the
        // allocation layout is reconstructed from the immutable type/length
        // pair used to allocate it.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
            alloc::dealloc(
                self.ptr.as_ptr().cast::<u8>(),
                allocation_layout_unchecked::<T>(self.len),
            );
        }
    }
}

/// Construction-only owner for an aligned allocation containing possibly
/// uninitialized elements.
pub(crate) struct AlignedBufferBuilder<T> {
    ptr: NonNull<MaybeUninit<T>>,
    len: usize,
    initialized: usize,
}

impl<T> AlignedBufferBuilder<T> {
    pub(crate) fn new(len: usize) -> Result<Self> {
        let layout = allocation_layout::<T>(len)?;
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                initialized: 0,
            });
        }

        let ptr = NonNull::new(unsafe { alloc::alloc(layout) })
            .ok_or(Error::AllocationFailed {
                bytes: layout.size(),
                alignment: layout.align(),
            })?
            .cast::<MaybeUninit<T>>();

        Ok(Self {
            ptr,
            len,
            initialized: 0,
        })
    }

    pub(crate) fn remaining(&self) -> usize {
        self.len - self.initialized
    }

    pub(crate) fn write_next(&mut self, value: T) -> Result<()> {
        if self.initialized >= self.len {
            return Err(Error::StorageOutOfBounds);
        }

        // SAFETY: the bounds check above selects the next uninitialized slot
        // inside the allocation owned by this builder.
        unsafe {
            self.ptr
                .as_ptr()
                .add(self.initialized)
                .write(MaybeUninit::new(value));
        }
        self.initialized += 1;
        Ok(())
    }

    pub(crate) fn extend_from_slice(&mut self, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        let end = self
            .initialized
            .checked_add(values.len())
            .ok_or(Error::StorageOutOfBounds)?;
        if end > self.len {
            return Err(Error::StorageOutOfBounds);
        }
        if values.is_empty() {
            return Ok(());
        }

        // SAFETY: `end <= len` was checked above, the destination is the
        // builder's still-uninitialized tail, and Copy values cannot overlap
        // the destination through this separate allocation.
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.ptr.as_ptr().cast::<T>().add(self.initialized),
                values.len(),
            );
        }
        self.initialized = end;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<AlignedBuffer<T>> {
        if self.initialized != self.len {
            return Err(Error::UninitializedStorage {
                initialized: self.initialized,
                len: self.len,
            });
        }

        let this = ManuallyDrop::new(self);
        Ok(AlignedBuffer {
            ptr: this.ptr.cast::<T>(),
            len: this.len,
        })
    }
}

impl<T> Drop for AlignedBufferBuilder<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        // SAFETY: only the initialized prefix contains T values. The
        // allocation layout is the same one used by `new`.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr().cast::<T>(),
                self.initialized,
            ));
            alloc::dealloc(
                self.ptr.as_ptr().cast::<u8>(),
                allocation_layout_unchecked::<T>(self.len),
            );
        }
    }
}

fn allocation_alignment<T>() -> usize {
    CPU_STORAGE_ALIGNMENT.max(mem::align_of::<T>())
}

fn allocation_layout<T>(len: usize) -> Result<Layout> {
    if mem::size_of::<T>() == 0 {
        return Err(Error::UnsupportedZeroSizedType);
    }
    let bytes = len
        .checked_mul(mem::size_of::<T>())
        .ok_or(Error::StorageOutOfBounds)?;
    Layout::from_size_align(bytes, allocation_alignment::<T>()).map_err(|_| {
        Error::AllocationFailed {
            bytes,
            alignment: allocation_alignment::<T>(),
        }
    })
}

unsafe fn allocation_layout_unchecked<T>(len: usize) -> Layout {
    let bytes = len * mem::size_of::<T>();
    // SAFETY: this helper is only called after `allocation_layout` succeeded;
    // T is non-zero-sized and the alignment is a valid power of two.
    unsafe { Layout::from_size_align_unchecked(bytes, allocation_alignment::<T>()) }
}

#[cfg(test)]
mod tests {
    use super::{AlignedBuffer, AlignedBufferBuilder, CPU_STORAGE_ALIGNMENT};
    use crate::Error;

    #[test]
    fn aligned_buffer_preserves_values_and_alignment() {
        let buffer = AlignedBuffer::from_slice(&[1u8, 2, 3, 4]).unwrap();
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(buffer.alignment(), CPU_STORAGE_ALIGNMENT);
        assert!(buffer.is_aligned_to(CPU_STORAGE_ALIGNMENT));
    }

    #[test]
    fn builder_supports_sequential_and_bulk_writes() {
        let mut builder = AlignedBufferBuilder::<u32>::new(4).unwrap();
        builder.write_next(1).unwrap();
        builder.extend_from_slice(&[2, 3, 4]).unwrap();
        assert_eq!(builder.remaining(), 0);
        assert_eq!(builder.finish().unwrap().as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn empty_buffer_does_not_allocate() {
        let buffer = AlignedBuffer::<u64>::from_slice(&[]).unwrap();
        assert!(buffer.is_empty());
        assert!(buffer.as_slice().is_empty());
        assert!(buffer.is_aligned_to(CPU_STORAGE_ALIGNMENT));
    }

    #[test]
    fn builder_rejects_incomplete_and_overfilled_buffers() {
        let mut incomplete = AlignedBufferBuilder::<u8>::new(2).unwrap();
        incomplete.write_next(1).unwrap();
        assert!(matches!(
            incomplete.finish(),
            Err(Error::UninitializedStorage {
                initialized: 1,
                len: 2
            })
        ));

        let mut full = AlignedBufferBuilder::<u8>::new(1).unwrap();
        full.write_next(1).unwrap();
        assert!(matches!(full.write_next(2), Err(Error::StorageOutOfBounds)));
    }

    #[test]
    fn builder_rejects_zero_sized_types_and_overflow() {
        assert!(matches!(
            AlignedBufferBuilder::<()>::new(1),
            Err(Error::UnsupportedZeroSizedType)
        ));
        assert!(matches!(
            AlignedBufferBuilder::<u64>::new(usize::MAX),
            Err(Error::StorageOutOfBounds | Error::AllocationFailed { .. })
        ));
    }
}
