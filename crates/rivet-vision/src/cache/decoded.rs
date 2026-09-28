use crate::errors::{RivetResult, invalid_argument, invalid_shape};
use crate::sample::image::{DecodedSample, ImageAxisOrder, ImageBatch};
use rivet_core::{CpuStorageRef, DType, Device, Error, Tensor};
use rivet_data::dataset::MemoryDataset;
use rivet_data::dataset::source::{Dataset, validate_indices};
use rivet_data::errors::{DataError, DataResult};

/// A fixed-shape decoded image dataset backed by one image Tensor and one
/// label Tensor. Per-sample access only creates views into `images`.
#[derive(Debug)]
pub struct DenseImageMemoryDataset {
    images: Tensor,
    labels: Tensor,
}

impl DenseImageMemoryDataset {
    pub fn new(images: Tensor, labels: Tensor) -> RivetResult<Self> {
        if images.rank() == 0 {
            return Err(invalid_shape(
                "dense image dataset images must have a leading sample dimension",
            ));
        }
        if labels.rank() != 1 {
            return Err(invalid_shape(format!(
                "dense image dataset labels must be rank 1, got shape {:?}",
                labels.dims()
            )));
        }
        if labels.dtype() != DType::I64 {
            return Err(invalid_argument(format!(
                "dense image dataset labels must be int64, got {:?}",
                labels.dtype()
            )));
        }
        if images.dims()[0] != labels.dims()[0] {
            return Err(invalid_shape(format!(
                "dense image dataset length mismatch: images {}, labels {}",
                images.dims()[0],
                labels.dims()[0]
            )));
        }

        Ok(Self { images, labels })
    }

    pub fn images(&self) -> &Tensor {
        &self.images
    }

    pub fn labels(&self) -> &Tensor {
        &self.labels
    }

    /// Read a logical batch directly from the dense backing tensors.
    ///
    /// Consecutive indices stay as views into the backing image/label tensors;
    /// arbitrary order and duplicate indices are gathered into one output
    /// tensor while preserving the request order.
    pub fn get_batch(&self, indices: &[usize]) -> RivetResult<ImageBatch> {
        validate_indices(indices, self.len())?;

        let images = gather_rows(&self.images, indices)?;
        let labels = gather_rows(&self.labels, indices)?;
        Ok(ImageBatch {
            images,
            labels,
            axis_order: ImageAxisOrder::Hwc,
        })
    }

    fn get_one(&self, index: usize) -> RivetResult<DecodedSample> {
        let image = self.images.get(index)?;
        let label = self.labels.read_scalar_at::<i64>(index)?;
        Ok(DecodedSample { image, label })
    }
}

fn gather_rows(input: &Tensor, indices: &[usize]) -> RivetResult<Tensor> {
    if indices.is_empty() {
        let mut shape = input.dims().to_vec();
        shape[0] = 0;
        return Ok(Tensor::zeros(shape, input.dtype(), input.device())?);
    }

    if let Some((start, len)) = consecutive_range(indices) {
        return Ok(input.narrow(0, start, len)?);
    }

    let rows = indices
        .iter()
        .map(|&index| input.narrow(0, index, 1))
        .collect::<Result<Vec<_>, _>>()?;
    let refs = rows.iter().collect::<Vec<_>>();
    let stacked = Tensor::stack(&refs, 0)?;
    Ok(stacked.squeeze(1)?)
}

fn consecutive_range(indices: &[usize]) -> Option<(usize, usize)> {
    let start = *indices.first()?;
    let consecutive = indices
        .iter()
        .enumerate()
        .all(|(offset, &index)| start.checked_add(offset) == Some(index));
    consecutive.then_some((start, indices.len()))
}

impl Dataset for DenseImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        self.images.dims()[0]
    }

    fn capabilities(&self) -> rivet_data::dataset::SourceCapabilities {
        rivet_data::dataset::SourceCapabilities {
            access_pattern: rivet_data::dataset::AccessPattern::RandomAccess,
            batched_reads: true,
            preferred_batch_size: None,
            zero_copy: true,
            parallel_reads: true,
            async_reads: false,
            read_device: Some("cpu"),
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        validate_indices(indices, self.len())?;
        indices
            .iter()
            .map(|&index| {
                self.get_one(index)
                    .map_err(|error| DataError::InvalidArgument(error.to_string()))
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct PackedImageEntry {
    slab: usize,
    offset: usize,
    len: usize,
    shape: [usize; 3],
    label: i64,
}

/// Decoded RGB images packed into bounded aligned slabs.
///
/// Unlike the sample-oriented fallback, this representation needs one pixel
/// allocation per materialization chunk instead of one allocation per image.
/// Samples returned by `get_many` are zero-copy tensor views into those slabs.
#[derive(Debug)]
pub struct PackedImageMemoryDataset {
    slabs: Vec<Tensor>,
    entries: Vec<PackedImageEntry>,
    fixed_shape: Option<[usize; 3]>,
    payload_bytes: usize,
}

impl PackedImageMemoryDataset {
    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }

    pub fn is_fixed_shape(&self) -> bool {
        self.fixed_shape.is_some()
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    /// Approximate owned resident bytes, including packed pixels and entry
    /// metadata. Tensor and Vec allocator bookkeeping is intentionally not
    /// included because it is allocator-specific.
    pub fn resident_bytes(&self) -> usize {
        self.slabs
            .iter()
            .map(Tensor::storage_bytes)
            .sum::<usize>()
            .saturating_add(
                self.entries
                    .len()
                    .saturating_mul(std::mem::size_of::<PackedImageEntry>()),
            )
    }

    fn get_one(&self, index: usize) -> RivetResult<DecodedSample> {
        let entry = self.entries.get(index).ok_or_else(|| {
            invalid_argument(format!("decoded cache index {index} is out of range"))
        })?;
        let image = self.slabs[entry.slab]
            .narrow(0, entry.offset, entry.len)?
            .reshape(entry.shape)?;
        Ok(DecodedSample {
            image,
            label: entry.label,
        })
    }

    pub fn get_batch(&self, indices: &[usize]) -> RivetResult<ImageBatch> {
        validate_indices(indices, self.len())?;
        let shape = self.fixed_shape.ok_or_else(|| {
            invalid_shape("packed variable-shape cache cannot produce a direct image batch")
        })?;
        let labels = Tensor::from_exact_iter(
            indices.iter().map(|&index| self.entries[index].label),
            [indices.len()],
            &Device::Cpu,
        )?;
        let images = if indices.is_empty() {
            Tensor::zeros([0, shape[0], shape[1], shape[2]], DType::U8, &Device::Cpu)?
        } else if let Some(image) = self.consecutive_batch_view(indices, shape)? {
            image
        } else {
            let samples = indices
                .iter()
                .map(|&index| self.get_one(index).map(|sample| sample.image))
                .collect::<RivetResult<Vec<_>>>()?;
            let refs = samples.iter().collect::<Vec<_>>();
            Tensor::stack(&refs, 0)?
        };
        Ok(ImageBatch {
            images,
            labels,
            axis_order: ImageAxisOrder::Hwc,
        })
    }

    fn consecutive_batch_view(
        &self,
        indices: &[usize],
        shape: [usize; 3],
    ) -> RivetResult<Option<Tensor>> {
        let Some((&first_index, rest)) = indices.split_first() else {
            return Ok(None);
        };
        let first = self.entries[first_index];
        let contiguous = rest.iter().enumerate().all(|(offset, &index)| {
            let entry = self.entries[index];
            index == first_index + offset + 1
                && entry.slab == first.slab
                && entry.offset == first.offset + (offset + 1) * first.len
                && entry.len == first.len
        });
        if !contiguous {
            return Ok(None);
        }
        let len = first
            .len
            .checked_mul(indices.len())
            .ok_or_else(|| invalid_shape("packed decoded batch size overflow"))?;
        Ok(Some(
            self.slabs[first.slab]
                .narrow(0, first.offset, len)?
                .reshape([indices.len(), shape[0], shape[1], shape[2]])?,
        ))
    }
}

impl Dataset for PackedImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn capabilities(&self) -> rivet_data::dataset::SourceCapabilities {
        rivet_data::dataset::SourceCapabilities {
            access_pattern: rivet_data::dataset::AccessPattern::RandomAccess,
            batched_reads: self.is_fixed_shape(),
            preferred_batch_size: None,
            zero_copy: true,
            parallel_reads: true,
            async_reads: false,
            read_device: Some("cpu"),
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        validate_indices(indices, self.len())?;
        indices
            .iter()
            .map(|&index| {
                self.get_one(index)
                    .map_err(|error| DataError::InvalidArgument(error.to_string()))
            })
            .collect()
    }
}

pub(crate) struct PackedImageMemoryBuilder {
    slabs: Vec<Tensor>,
    entries: Vec<PackedImageEntry>,
    first_shape: Option<[usize; 3]>,
    fixed_shape: bool,
    payload_bytes: usize,
}

impl PackedImageMemoryBuilder {
    pub(crate) fn with_capacity(len: usize) -> Self {
        Self {
            slabs: Vec::new(),
            entries: Vec::with_capacity(len),
            first_shape: None,
            fixed_shape: true,
            payload_bytes: 0,
        }
    }

    pub(crate) fn push_chunk(&mut self, samples: Vec<DecodedSample>) -> RivetResult<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut shapes = Vec::with_capacity(samples.len());
        let mut slab_bytes = 0usize;
        for sample in &samples {
            let [height, width, channels] = sample.image.dims() else {
                return Err(invalid_shape(format!(
                    "decoded cache expects rank-3 HWC images, got {:?}",
                    sample.image.dims()
                )));
            };
            if sample.image.dtype() != DType::U8 || !sample.image.is_contiguous() {
                return Err(invalid_argument(
                    "decoded cache expects contiguous uint8 images",
                ));
            }
            let shape = [*height, *width, *channels];
            if let Some(first) = self.first_shape {
                self.fixed_shape &= first == shape;
            } else {
                self.first_shape = Some(shape);
            }
            let len = sample.image.logical_bytes();
            slab_bytes = slab_bytes
                .checked_add(len)
                .ok_or_else(|| invalid_argument("decoded cache slab byte count overflow"))?;
            shapes.push((shape, len));
        }

        let pixels = Tensor::from_exact_writer::<u8, _, _>([slab_bytes], &Device::Cpu, |output| {
            for sample in &samples {
                sample.image.with_cpu_storage(|storage, layout| {
                    let CpuStorageRef::U8(values) = storage else {
                        return Err(Error::UnexpectedDType {
                            expected: DType::U8,
                            actual: storage.dtype(),
                        });
                    };
                    let (start, end) = layout
                        .contiguous_offsets()
                        .ok_or(Error::StorageOutOfBounds)?;
                    let values = values.get(start..end).ok_or(Error::StorageOutOfBounds)?;
                    output.extend_from_slice(values)
                })?;
            }
            Ok(())
        })?;

        let slab = self.slabs.len();
        let mut offset = 0usize;
        for (sample, (shape, len)) in samples.into_iter().zip(shapes) {
            self.entries.push(PackedImageEntry {
                slab,
                offset,
                len,
                shape,
                label: sample.label,
            });
            offset += len;
        }
        self.payload_bytes = self
            .payload_bytes
            .checked_add(slab_bytes)
            .ok_or_else(|| invalid_argument("decoded cache byte count overflow"))?;
        self.slabs.push(pixels);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> PackedImageMemoryDataset {
        self.slabs.shrink_to_fit();
        self.entries.shrink_to_fit();
        PackedImageMemoryDataset {
            slabs: self.slabs,
            entries: self.entries,
            fixed_shape: self.fixed_shape.then_some(self.first_shape).flatten(),
            payload_bytes: self.payload_bytes,
        }
    }
}

/// Variable-shape decoded image storage. It remains Tensor-backed, but keeps
/// one Tensor handle per sample because a single dense image shape is not
/// available.
#[derive(Debug)]
pub struct VariableImageMemoryDataset {
    inner: MemoryDataset<DecodedSample>,
}

impl VariableImageMemoryDataset {
    pub fn new(items: Vec<DecodedSample>) -> Self {
        Self {
            inner: MemoryDataset::new(items),
        }
    }

    pub fn as_slice(&self) -> &[DecodedSample] {
        self.inner.as_slice()
    }
}

impl Dataset for VariableImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn capabilities(&self) -> rivet_data::dataset::SourceCapabilities {
        rivet_data::dataset::SourceCapabilities {
            access_pattern: rivet_data::dataset::AccessPattern::RandomAccess,
            batched_reads: true,
            preferred_batch_size: None,
            zero_copy: true,
            parallel_reads: true,
            async_reads: false,
            read_device: Some("cpu"),
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        self.inner.get_many(indices)
    }
}

/// Decoded image cache storage.
///
/// Cache materialization uses packed slabs. Dense and variable variants remain
/// available for callers that explicitly construct an in-memory dataset.
#[derive(Debug)]
pub enum DecodedImageMemoryDataset {
    Dense(DenseImageMemoryDataset),
    Packed(PackedImageMemoryDataset),
    Variable(VariableImageMemoryDataset),
}

impl DecodedImageMemoryDataset {
    /// Preserve the explicit sample-oriented constructor for callers that do
    /// not want shape detection. Cache materialization uses `from_samples`.
    pub fn new(items: Vec<DecodedSample>) -> Self {
        Self::Variable(VariableImageMemoryDataset::new(items))
    }

    pub fn from_samples(items: Vec<DecodedSample>) -> RivetResult<Self> {
        let Some(first) = items.first() else {
            return Ok(Self::Variable(VariableImageMemoryDataset::new(items)));
        };

        let shape = first.image.dims();
        let dtype = first.image.dtype();
        let fixed_shape = items
            .iter()
            .all(|sample| sample.image.dims() == shape && sample.image.dtype() == dtype);
        if !fixed_shape {
            return Ok(Self::Variable(VariableImageMemoryDataset::new(items)));
        }

        let labels = items.iter().map(|sample| sample.label).collect::<Vec<_>>();
        let image_refs = items.iter().map(|sample| &sample.image).collect::<Vec<_>>();
        let images = Tensor::stack(&image_refs, 0)?;
        let labels = Tensor::from_vec(labels, [items.len()], &Device::Cpu)?;
        Ok(Self::Dense(DenseImageMemoryDataset::new(images, labels)?))
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    pub fn as_dense(&self) -> Option<&DenseImageMemoryDataset> {
        match self {
            Self::Dense(dataset) => Some(dataset),
            Self::Packed(_) | Self::Variable(_) => None,
        }
    }

    pub fn as_packed(&self) -> Option<&PackedImageMemoryDataset> {
        match self {
            Self::Packed(dataset) => Some(dataset),
            Self::Dense(_) | Self::Variable(_) => None,
        }
    }

    pub fn payload_bytes(&self) -> usize {
        match self {
            Self::Dense(dataset) => dataset.images.logical_bytes(),
            Self::Packed(dataset) => dataset.payload_bytes(),
            Self::Variable(dataset) => dataset
                .as_slice()
                .iter()
                .map(|sample| sample.image.logical_bytes())
                .sum(),
        }
    }

    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Dense(dataset) => dataset
                .images
                .storage_bytes()
                .saturating_add(dataset.labels.storage_bytes()),
            Self::Packed(dataset) => dataset.resident_bytes(),
            Self::Variable(dataset) => {
                let mut allocations = std::collections::HashSet::new();
                dataset
                    .as_slice()
                    .iter()
                    .filter_map(|sample| {
                        let allocation = (
                            sample.image.storage_base_ptr() as usize,
                            sample.image.storage_bytes(),
                        );
                        allocations.insert(allocation).then_some(allocation.1)
                    })
                    .sum()
            }
        }
    }
}

impl Dataset for DecodedImageMemoryDataset {
    type Item = DecodedSample;

    fn len(&self) -> usize {
        match self {
            Self::Dense(dataset) => dataset.len(),
            Self::Packed(dataset) => dataset.len(),
            Self::Variable(dataset) => dataset.len(),
        }
    }

    fn get_many(&self, indices: &[usize]) -> DataResult<Vec<Self::Item>> {
        match self {
            Self::Dense(dataset) => dataset.get_many(indices),
            Self::Packed(dataset) => dataset.get_many(indices),
            Self::Variable(dataset) => dataset.get_many(indices),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(value: u8, label: i64, shape: [usize; 3]) -> DecodedSample {
        DecodedSample {
            image: Tensor::from_vec(vec![value; shape.iter().product()], shape, &Device::Cpu)
                .unwrap(),
            label,
        }
    }

    #[test]
    fn fixed_shape_samples_use_dense_zero_copy_views() {
        let dataset = DecodedImageMemoryDataset::from_samples(vec![
            sample(10, 10, [1, 2, 1]),
            sample(20, 20, [1, 2, 1]),
            sample(30, 30, [1, 2, 1]),
        ])
        .unwrap();

        assert!(dataset.is_dense());
        let dense = dataset.as_dense().unwrap();
        let samples = dataset.get_many(&[2, 0, 2]).unwrap();
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.label)
                .collect::<Vec<_>>(),
            [30, 10, 30]
        );
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.image.to_vec::<u8>().unwrap()[0])
                .collect::<Vec<_>>(),
            [30, 10, 30]
        );
        assert!(samples[0].image.same_storage(dense.images()));
        assert!(samples[1].image.same_storage(dense.images()));
    }

    #[test]
    fn dense_batch_read_preserves_order_duplicates_and_views() {
        let dataset = DenseImageMemoryDataset::new(
            Tensor::from_vec(vec![10u8, 11, 20, 21, 30, 31], [3, 1, 2, 1], &Device::Cpu).unwrap(),
            Tensor::from_vec(vec![10i64, 20, 30], [3], &Device::Cpu).unwrap(),
        )
        .unwrap();

        let range = dataset.get_batch(&[1, 2]).unwrap();
        assert_eq!(range.images.dims(), [2, 1, 2, 1]);
        assert_eq!(range.images.to_vec::<u8>().unwrap(), [20, 21, 30, 31]);
        assert_eq!(range.labels.to_vec::<i64>().unwrap(), [20, 30]);
        assert!(range.images.same_storage(dataset.images()));

        let gathered = dataset.get_batch(&[2, 0, 2]).unwrap();
        assert_eq!(gathered.images.dims(), [3, 1, 2, 1]);
        assert_eq!(
            gathered.images.to_vec::<u8>().unwrap(),
            [30, 31, 10, 11, 30, 31]
        );
        assert_eq!(gathered.labels.to_vec::<i64>().unwrap(), [30, 10, 30]);
        assert!(!gathered.images.same_storage(dataset.images()));
    }

    #[test]
    fn dense_batch_read_preserves_empty_shape() {
        let dataset = DenseImageMemoryDataset::new(
            Tensor::from_vec(Vec::<u8>::new(), [0, 2, 2, 3], &Device::Cpu).unwrap(),
            Tensor::from_vec(Vec::<i64>::new(), [0], &Device::Cpu).unwrap(),
        )
        .unwrap();
        let batch = dataset.get_batch(&[]).unwrap();

        assert_eq!(batch.images.dims(), [0, 2, 2, 3]);
        assert_eq!(batch.labels.dims(), [0]);
    }

    #[test]
    fn variable_shapes_keep_tensor_sample_fallback() {
        let dataset = DecodedImageMemoryDataset::from_samples(vec![
            sample(10, 10, [1, 2, 1]),
            sample(20, 20, [2, 2, 1]),
        ])
        .unwrap();

        assert!(!dataset.is_dense());
        let samples = dataset.get_many(&[1, 0, 1]).unwrap();
        assert_eq!(samples[0].image.dims(), [2, 2, 1]);
        assert_eq!(samples[1].image.dims(), [1, 2, 1]);
        assert_eq!(samples[2].label, 20);
    }

    #[test]
    fn packed_chunks_share_slab_storage_and_preserve_batch_order() {
        let mut builder = PackedImageMemoryBuilder::with_capacity(4);
        builder
            .push_chunk(vec![sample(10, 10, [1, 2, 1]), sample(20, 20, [1, 2, 1])])
            .unwrap();
        builder
            .push_chunk(vec![sample(30, 30, [1, 2, 1]), sample(40, 40, [1, 2, 1])])
            .unwrap();
        let dataset = builder.finish();

        assert!(dataset.is_fixed_shape());
        assert_eq!(dataset.payload_bytes(), 8);
        assert!(dataset.resident_bytes() >= 8);
        let first_slab = dataset.get_many(&[0, 1]).unwrap();
        assert!(first_slab[0].image.same_storage(&first_slab[1].image));
        let second_slab = dataset.get_many(&[2]).unwrap();
        assert!(!first_slab[0].image.same_storage(&second_slab[0].image));

        let view = dataset.get_batch(&[0, 1]).unwrap();
        assert_eq!(view.images.dims(), [2, 1, 2, 1]);
        assert_eq!(view.images.to_vec::<u8>().unwrap(), [10, 10, 20, 20]);
        assert!(view.images.same_storage(&first_slab[0].image));

        let gathered = dataset.get_batch(&[3, 0, 3]).unwrap();
        assert_eq!(gathered.images.dims(), [3, 1, 2, 1]);
        assert_eq!(
            gathered.images.to_vec::<u8>().unwrap(),
            [40, 40, 10, 10, 40, 40]
        );
        assert_eq!(gathered.labels.to_vec::<i64>().unwrap(), [40, 10, 40]);
    }

    #[test]
    fn packed_variable_shapes_return_zero_copy_sample_views() {
        let mut builder = PackedImageMemoryBuilder::with_capacity(2);
        builder
            .push_chunk(vec![sample(10, 10, [1, 2, 1]), sample(20, 20, [2, 2, 1])])
            .unwrap();
        let dataset = builder.finish();

        assert!(!dataset.is_fixed_shape());
        let samples = dataset.get_many(&[1, 0, 1]).unwrap();
        assert!(samples[0].image.same_storage(&samples[1].image));
        assert_eq!(samples[0].image.dims(), [2, 2, 1]);
        assert_eq!(samples[1].image.dims(), [1, 2, 1]);
        assert_eq!(samples[2].label, 20);
        assert!(dataset.get_batch(&[0, 1]).is_err());
    }
}
