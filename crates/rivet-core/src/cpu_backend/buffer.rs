use crate::{Error, Result};
use std::alloc::{self, Layout};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ops::Deref;
use std::ptr::{self, NonNull};

/// Alignment guaranteed for every non-empty Rivet-owned CPU allocation.
pub const CPU_STORAGE_ALIGNMENT: usize = 256;

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

    pub(crate) fn base_ptr(&self) -> *const u8 {
        self.as_ptr().cast()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: a finished buffer contains exactly `len` initialized T
        // values, and `ptr` remains valid until this buffer is dropped.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: a finished buffer contains exactly `len` initialized T
        // values, and the unique `&mut self` excludes other accesses.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
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

    #[inline]
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

pub(crate) fn alignment_after_offset(base_alignment: usize, byte_offset: usize) -> usize {
    if byte_offset == 0 {
        return base_alignment;
    }
    base_alignment.min(byte_offset & byte_offset.wrapping_neg())
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
    use half::{bf16, f16};
    use std::fmt::Debug;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::{Arc, atomic::AtomicUsize, atomic::Ordering};

    fn assert_dtype<T>(sample: T)
    where
        T: Copy + Debug + PartialEq,
    {
        let lengths = [0, 1, 2, 3, 31, 32, 63, 64, 65, 1024];
        for len in lengths {
            let values = vec![sample; len];
            let from_slice = AlignedBuffer::from_slice(&values).unwrap();
            assert_eq!(from_slice.len(), len);
            assert_eq!(from_slice.as_slice(), values.as_slice());
            assert_eq!(from_slice.alignment(), CPU_STORAGE_ALIGNMENT);
            assert!(from_slice.is_aligned_to(CPU_STORAGE_ALIGNMENT));
            if len > 0 {
                assert!(!from_slice.is_aligned_to(0));
            }

            let from_vec = AlignedBuffer::from_vec(values.clone()).unwrap();
            assert_eq!(from_vec.as_slice(), values.as_slice());

            let cloned = from_slice.try_clone().unwrap();
            assert_eq!(cloned.as_slice(), values.as_slice());
        }
    }

    #[test]
    fn every_supported_dtype_preserves_values_and_alignment() {
        assert_dtype(7u8);
        assert_dtype(7u32);
        assert_dtype(7i16);
        assert_dtype(7i32);
        assert_dtype(7i64);
        assert_dtype(bf16::from_f32(7.0));
        assert_dtype(f16::from_f32(7.0));
        assert_dtype(7.0f32);
        assert_dtype(7.0f64);
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
        assert_eq!(buffer.len(), 0);
        assert!(buffer.as_slice().is_empty());
        assert!(buffer.is_aligned_to(CPU_STORAGE_ALIGNMENT));

        let mut builder = AlignedBufferBuilder::<u64>::new(0).unwrap();
        builder.extend_from_slice(&[]).unwrap();
        assert_eq!(builder.remaining(), 0);
        assert!(builder.finish().unwrap().is_empty());
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

        let mut bulk = AlignedBufferBuilder::<u8>::new(2).unwrap();
        assert!(matches!(
            bulk.extend_from_slice(&[1, 2, 3]),
            Err(Error::StorageOutOfBounds)
        ));
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

    #[derive(Debug)]
    struct DropProbe(Arc<AtomicUsize>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn builder_and_finished_buffer_drop_initialized_values() {
        let drops = Arc::new(AtomicUsize::new(0));
        {
            let mut builder = AlignedBufferBuilder::<DropProbe>::new(3).unwrap();
            for _ in 0..2 {
                builder.write_next(DropProbe(Arc::clone(&drops))).unwrap();
            }
        }
        assert_eq!(drops.load(Ordering::Relaxed), 2);

        {
            let mut builder = AlignedBufferBuilder::<DropProbe>::new(2).unwrap();
            for _ in 0..2 {
                builder.write_next(DropProbe(Arc::clone(&drops))).unwrap();
            }
            let buffer = builder.finish().unwrap();
            assert_eq!(drops.load(Ordering::Relaxed), 2);
            drop(buffer);
        }
        assert_eq!(drops.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn builder_drop_is_panic_safe_for_initialized_prefix() {
        let drops = Arc::new(AtomicUsize::new(0));
        let panic_result = catch_unwind(AssertUnwindSafe({
            let drops = Arc::clone(&drops);
            move || {
                let mut builder = AlignedBufferBuilder::<DropProbe>::new(2).unwrap();
                builder
                    .write_next(DropProbe(drops))
                    .expect("first write should fit");
                panic!("abort construction");
            }
        }));

        assert!(panic_result.is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }
}
