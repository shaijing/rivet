use rivet_core::{CpuStorageRef, DType, Device, Error, Layout, Shape, Tensor};

#[test]
fn shape_and_layout_metadata_match_candle_semantics() {
    assert_eq!(Shape::from(()).elem_count(), 1);
    assert_eq!(Shape::from((2, 3, 4)).stride_contiguous(), vec![12, 4, 1]);
    assert_eq!(Shape::from([0, 3]).elem_count(), 0);
    assert!(Shape::from([1, 3]).is_contiguous(&[100, 1]));
    assert!(!Shape::from([2, 3]).is_contiguous(&[1, 2]));

    let offset = Layout::new(Shape::from((2, 3)), vec![3, 1], 7).unwrap();
    assert!(offset.is_contiguous());
    assert_eq!(offset.contiguous_offsets(), Some((7, 13)));

    let narrow = Layout::contiguous(Shape::from((2, 3, 4)))
        .narrow(1, 1, 1)
        .unwrap();
    assert_eq!(narrow.dims(), &[2, 1, 4]);
    assert_eq!(narrow.stride(), &[12, 4, 1]);
    assert_eq!(narrow.start_offset(), 4);

    let broadcast = Layout::contiguous(Shape::from((1, 3)))
        .broadcast_as(Shape::from((4, 3)))
        .unwrap();
    assert_eq!(broadcast.stride(), &[0, 1]);
}

#[test]
fn strided_index_handles_transpose_and_broadcast() {
    let layout = Layout::new(Shape::from((2, 3)), vec![4, 1], 10).unwrap();
    assert_eq!(
        layout.strided_index().collect::<Vec<_>>(),
        vec![10, 11, 12, 14, 15, 16]
    );

    let empty = Layout::contiguous(Shape::from((0, 3)));
    assert_eq!(
        empty.strided_index().collect::<Vec<_>>(),
        Vec::<usize>::new()
    );

    let broadcast = Layout::new(Shape::from((4, 3)), vec![0, 1], 2).unwrap();
    assert_eq!(
        broadcast.strided_index().collect::<Vec<_>>(),
        vec![2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
    );
}

#[test]
fn tensor_views_preserve_logical_order() {
    let tensor = Tensor::from_vec(vec![0u8, 1, 2, 3, 4, 5], (2, 3), &Device::Cpu).unwrap();
    let transpose = tensor.transpose(0, 1).unwrap();
    assert_ne!(tensor.id(), transpose.id());
    assert_eq!(transpose.dims(), &[3, 2]);
    assert_eq!(transpose.to_vec::<u8>().unwrap(), vec![0, 3, 1, 4, 2, 5]);

    let chw = Tensor::from_vec((0u8..24).collect(), (2, 3, 4), &Device::Cpu)
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap();
    assert_eq!(chw.dims(), &[4, 2, 3]);
    assert_eq!(chw.stride(), &[1, 12, 4]);

    let narrow = tensor.narrow(1, 1, 1).unwrap();
    assert_eq!(narrow.to_vec::<u8>().unwrap(), vec![1, 4]);
}

#[test]
fn borrowed_cpu_storage_exposes_view_layout_without_materializing() {
    let tensor = Tensor::from_vec(vec![0u8, 1, 2, 3, 4, 5], (2, 3), &Device::Cpu).unwrap();
    let transpose = tensor.transpose(0, 1).unwrap();

    let values = transpose
        .with_cpu_storage(|storage, layout| {
            let CpuStorageRef::U8(storage) = storage else {
                panic!("expected uint8 storage");
            };
            Ok(layout
                .strided_index()
                .map(|index| storage[index])
                .collect::<Vec<_>>())
        })
        .unwrap();

    assert_eq!(values, vec![0, 3, 1, 4, 2, 5]);
    assert_eq!(values, transpose.to_vec::<u8>().unwrap());
}

#[test]
fn stack_reads_non_contiguous_views_without_materializing_each_input() {
    let first = Tensor::from_vec((0u8..12).collect(), (2, 2, 3), &Device::Cpu)
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap();
    let second = Tensor::from_vec((12u8..24).collect(), (2, 2, 3), &Device::Cpu)
        .unwrap()
        .permute(&[2, 0, 1])
        .unwrap();

    let batch = Tensor::stack(&[&first, &second], 0).unwrap();

    assert_eq!(batch.dims(), &[2, 3, 2, 2]);
    assert!(batch.is_contiguous());
    assert_eq!(
        batch.to_vec::<u8>().unwrap(),
        [
            0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 15, 18, 21, 13, 16, 19, 22, 14, 17, 20, 23,
        ]
    );
}

#[test]
fn reshape_copy_contiguous_and_dtype_follow_their_distinct_semantics() {
    let tensor = Tensor::from_vec(vec![0u8, 1, 2, 3, 4, 5], (2, 3), &Device::Cpu).unwrap();
    let clone = tensor.clone();
    assert_eq!(clone.id(), tensor.id());

    let reshaped = tensor.reshape((3, 2)).unwrap();
    assert_ne!(reshaped.id(), tensor.id());
    assert_eq!(reshaped.to_vec::<u8>().unwrap(), vec![0, 1, 2, 3, 4, 5]);

    let transposed = tensor.transpose(0, 1).unwrap();
    let materialized = transposed.contiguous().unwrap();
    assert_eq!(materialized.dims(), &[3, 2]);
    assert!(materialized.is_contiguous());
    assert_eq!(materialized.to_vec::<u8>().unwrap(), vec![0, 3, 1, 4, 2, 5]);

    let cast = transposed.to_dtype(DType::F32).unwrap();
    assert_eq!(
        cast.to_vec::<f32>().unwrap(),
        vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
    );
}

#[test]
fn scalar_empty_and_broadcast_tensors_work() {
    let scalar = Tensor::from_vec(vec![7i32], (), &Device::Cpu).unwrap();
    assert_eq!(scalar.elem_count(), 1);
    assert_eq!(scalar.to_vec0::<i32>().unwrap(), 7);

    let empty = Tensor::zeros((2, 0, 3), DType::F32, &Device::Cpu).unwrap();
    assert_eq!(empty.elem_count(), 0);
    assert!(empty.to_vec::<f32>().unwrap().is_empty());

    let empty_broadcast = Tensor::zeros((0, 3), DType::F32, &Device::Cpu).unwrap();
    let row = Tensor::ones((1, 3), DType::F32, &Device::Cpu).unwrap();
    let empty_result = empty_broadcast.broadcast_add(&row).unwrap();
    assert_eq!(empty_result.dims(), &[0, 3]);
    assert!(empty_result.to_vec::<f32>().unwrap().is_empty());

    let source = Tensor::from_vec(vec![1u8, 2, 3], (1, 3), &Device::Cpu).unwrap();
    let broadcast = source.broadcast_as((4, 3)).unwrap();
    assert_eq!(
        broadcast.to_vec::<u8>().unwrap(),
        vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    );
}

#[test]
fn strict_binary_ops_and_scalar_ops_are_layout_aware() {
    let lhs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
    let rhs = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], 3, &Device::Cpu).unwrap();
    assert_eq!(
        lhs.add(&rhs).unwrap().to_vec::<f32>().unwrap(),
        vec![11.0, 22.0, 33.0]
    );
    assert_eq!(
        lhs.sub(&rhs).unwrap().to_vec::<f32>().unwrap(),
        vec![-9.0, -18.0, -27.0]
    );
    assert_eq!(
        lhs.mul(&rhs).unwrap().to_vec::<f32>().unwrap(),
        vec![10.0, 40.0, 90.0]
    );
    assert_eq!(
        rhs.div(&lhs).unwrap().to_vec::<f32>().unwrap(),
        vec![10.0, 10.0, 10.0]
    );
    assert_eq!(
        lhs.add_scalar(2.0f32).unwrap().to_vec::<f32>().unwrap(),
        vec![3.0, 4.0, 5.0]
    );
    assert_eq!(
        lhs.mul_scalar(2.0f32).unwrap().to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0]
    );

    let base =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
    let transposed = base.transpose(0, 1).unwrap();
    let result = transposed.add(&transposed).unwrap();
    assert!(result.is_contiguous());
    assert_eq!(result.layout().start_offset(), 0);
    assert_eq!(
        result.to_vec::<f32>().unwrap(),
        vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
    );
}

#[test]
fn broadcast_unary_and_dtype_errors_are_explicit() {
    let matrix =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
    let row = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], (1, 3), &Device::Cpu).unwrap();
    assert_eq!(
        matrix.broadcast_add(&row).unwrap().to_vec::<f32>().unwrap(),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );

    let scalar = Tensor::from_vec(vec![2.0f32], (), &Device::Cpu).unwrap();
    let broadcast_scalar = scalar.broadcast_mul(&matrix).unwrap();
    assert_eq!(
        broadcast_scalar.to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
    assert_eq!(
        matrix.neg().unwrap().to_vec::<f32>().unwrap(),
        vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
    );
    assert_eq!(
        matrix
            .sub_scalar(10.0f32)
            .unwrap()
            .abs()
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0]
    );

    let ints = Tensor::from_vec(vec![1i32, 2, 3], 3, &Device::Cpu).unwrap();
    assert!(matrix.add(&ints).is_err());
    assert!(matrix.add(&row.reshape((3, 1)).unwrap()).is_err());
    assert!(matrix.add_scalar(1i32).is_err());
}

#[test]
fn cat_and_stack_support_non_contiguous_inputs() {
    let base = Tensor::from_vec(vec![1i32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
    let transposed = base.transpose(0, 1).unwrap();
    let cat_rows = Tensor::cat(&[&transposed, &transposed], 0).unwrap();
    assert_eq!(cat_rows.dims(), &[6, 2]);
    assert_eq!(
        cat_rows.to_vec::<i32>().unwrap(),
        vec![1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6]
    );

    let cat_cols = Tensor::cat(&[&transposed, &transposed], 1).unwrap();
    assert_eq!(cat_cols.dims(), &[3, 4]);
    assert_eq!(
        cat_cols.to_vec::<i32>().unwrap(),
        vec![1, 4, 1, 4, 2, 5, 2, 5, 3, 6, 3, 6]
    );

    let stacked = Tensor::stack(&[&transposed, &transposed], 0).unwrap();
    assert_eq!(stacked.dims(), &[2, 3, 2]);
    assert_eq!(
        stacked.to_vec::<i32>().unwrap(),
        vec![1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6]
    );
}

#[test]
fn phase1_construction_and_access_apis_follow_tensor_contracts() {
    let full = Tensor::full(7u8, [2, 2], &Device::Cpu).unwrap();
    assert_eq!(full.to_vec::<u8>().unwrap(), [7, 7, 7, 7]);

    let zeros = full.zeros_like().unwrap();
    let ones = full.ones_like().unwrap();
    assert_eq!(zeros.dtype(), DType::U8);
    assert_eq!(zeros.dims(), &[2, 2]);
    assert_eq!(zeros.to_vec::<u8>().unwrap(), [0, 0, 0, 0]);
    assert_eq!(ones.to_vec::<u8>().unwrap(), [1, 1, 1, 1]);
    assert!(!zeros.same_storage(&full));

    let from_iter = Tensor::from_iter(0u32..4, &Device::Cpu).unwrap();
    assert_eq!(from_iter.dims(), &[4]);
    assert_eq!(from_iter.to_vec::<u32>().unwrap(), [0, 1, 2, 3]);

    assert_eq!(
        Tensor::arange(0u8, 5u8, &Device::Cpu)
            .unwrap()
            .to_vec::<u8>()
            .unwrap(),
        [0, 1, 2, 3, 4]
    );
    assert_eq!(
        Tensor::arange_step(5i32, 0i32, -2i32, &Device::Cpu)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [5, 3, 1]
    );
    assert!(matches!(
        Tensor::arange_step(0i32, 1i32, 0i32, &Device::Cpu),
        Err(Error::InvalidRangeStep)
    ));

    let scalar = Tensor::from_vec(vec![42i64], (), &Device::Cpu).unwrap();
    assert_eq!(scalar.to_scalar::<i64>().unwrap(), 42);
    assert_eq!(scalar.to_vec0::<i64>().unwrap(), 42);

    let tensor = Tensor::from_vec((0u8..24).collect(), [2, 3, 4], &Device::Cpu).unwrap();
    assert_eq!(tensor.dim(0).unwrap(), 2);
    assert_eq!(tensor.dim(2).unwrap(), 4);
    assert!(matches!(
        tensor.dim(3),
        Err(Error::InvalidDim { dim: 3, .. })
    ));
    assert_eq!(
        tensor.to_vec3::<u8>().unwrap(),
        vec![
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]],
            vec![
                vec![12, 13, 14, 15],
                vec![16, 17, 18, 19],
                vec![20, 21, 22, 23]
            ],
        ]
    );

    let transposed = tensor.permute(&[2, 0, 1]).unwrap();
    assert_eq!(
        transposed.to_vec3::<u8>().unwrap(),
        vec![
            vec![vec![0, 4, 8], vec![12, 16, 20]],
            vec![vec![1, 5, 9], vec![13, 17, 21]],
            vec![vec![2, 6, 10], vec![14, 18, 22]],
            vec![vec![3, 7, 11], vec![15, 19, 23]],
        ]
    );
    let empty = Tensor::zeros([2, 0, 3], DType::F32, &Device::Cpu).unwrap();
    assert_eq!(
        empty.to_vec3::<f32>().unwrap(),
        vec![Vec::<Vec<f32>>::new(), Vec::<Vec<f32>>::new()]
    );
}

#[test]
fn phase1_view_composition_preserves_logical_order_and_storage_contracts() {
    let tensor = Tensor::from_vec((0u8..24).collect(), [2, 3, 4], &Device::Cpu).unwrap();

    let first = tensor.get(1).unwrap();
    assert_eq!(first.dims(), &[3, 4]);
    assert_eq!(
        first.to_vec::<u8>().unwrap(),
        (12u8..24).collect::<Vec<_>>()
    );
    assert!(first.same_storage(&tensor));

    let middle = tensor.get_on_dim(1, 2).unwrap();
    assert_eq!(middle.dims(), &[2, 4]);
    assert_eq!(
        middle.to_vec::<u8>().unwrap(),
        [8, 9, 10, 11, 20, 21, 22, 23]
    );
    assert!(middle.same_storage(&tensor));

    let matrix = Tensor::from_vec((0u8..6).collect(), [2, 3], &Device::Cpu).unwrap();
    assert_eq!(matrix.t().unwrap().dims(), &[3, 2]);
    assert_eq!(
        matrix.t().unwrap().to_vec::<u8>().unwrap(),
        [0, 3, 1, 4, 2, 5]
    );
    assert!(matches!(tensor.t(), Ok(_)));
    assert!(
        Tensor::from_vec(vec![1u8], 1, &Device::Cpu)
            .unwrap()
            .t()
            .is_err()
    );

    let flattened = tensor.flatten(1, 2).unwrap();
    assert_eq!(flattened.dims(), &[2, 12]);
    assert!(flattened.same_storage(&tensor));
    assert_eq!(tensor.flatten_to(1).unwrap().dims(), &[6, 4]);
    assert_eq!(tensor.flatten_from(1).unwrap().dims(), &[2, 12]);

    let non_contiguous = tensor.permute(&[1, 0, 2]).unwrap();
    let flattened_view = non_contiguous.flatten(0, 1).unwrap();
    assert_eq!(flattened_view.dims(), &[6, 4]);
    assert!(!flattened_view.same_storage(&non_contiguous));
    assert_eq!(
        flattened_view.to_vec::<u8>().unwrap(),
        vec![
            0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23,
        ]
    );

    let row = Tensor::from_vec(vec![1u8, 2, 3], [1, 3], &Device::Cpu).unwrap();
    let left = row.broadcast_left([2]).unwrap();
    assert_eq!(left.dims(), &[2, 1, 3]);
    assert_eq!(left.to_vec::<u8>().unwrap(), [1, 2, 3, 1, 2, 3]);
    assert!(left.same_storage(&row));
    assert_eq!(row.expand([4, 3]).unwrap().dims(), &[4, 3]);

    let one_dim = Tensor::from_vec((0u8..10).collect(), 10, &Device::Cpu).unwrap();
    let chunks = one_dim.chunk(3, 0).unwrap();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.dims().to_vec())
            .collect::<Vec<_>>(),
        vec![vec![4], vec![3], vec![3]]
    );
    assert_eq!(chunks[0].to_vec::<u8>().unwrap(), [0, 1, 2, 3]);
    assert_eq!(chunks[2].to_vec::<u8>().unwrap(), [7, 8, 9]);
    assert_eq!(one_dim.chunk(20, 0).unwrap().len(), 10);
    assert!(matches!(
        one_dim.chunk(0, 0),
        Err(Error::InvalidChunkCount { chunks: 0 })
    ));

    let repeated = Tensor::from_vec(vec![1u8, 2], [2, 1], &Device::Cpu)
        .unwrap()
        .repeat([2, 3])
        .unwrap();
    assert_eq!(repeated.dims(), &[4, 3]);
    assert_eq!(
        repeated.to_vec::<u8>().unwrap(),
        [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
    );

    assert_eq!(
        one_dim.roll(1, 0).unwrap().to_vec::<u8>().unwrap(),
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(
        one_dim.roll(-2, 0).unwrap().to_vec::<u8>().unwrap(),
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    );
    let empty = Tensor::zeros([0], DType::U8, &Device::Cpu).unwrap();
    assert!(empty.roll(1, 0).unwrap().same_storage(&empty));

    let forced = tensor.force_contiguous().unwrap();
    assert!(forced.is_contiguous());
    assert!(!forced.same_storage(&tensor));
    assert_eq!(
        forced.to_vec::<u8>().unwrap(),
        tensor.to_vec::<u8>().unwrap()
    );

    let offset = tensor.narrow(0, 1, 1).unwrap();
    let forced_offset = offset.force_contiguous().unwrap();
    assert_eq!(forced_offset.layout().start_offset(), 0);
    assert!(!forced_offset.same_storage(&offset));
    assert_eq!(
        forced_offset.to_vec::<u8>().unwrap(),
        offset.to_vec::<u8>().unwrap()
    );
}

#[test]
fn phase2_reductions_are_layout_aware_and_keep_the_contract_explicit() {
    let tensor =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
    assert_eq!(tensor.sum_keepdim(0).unwrap().dims(), &[1, 3]);
    assert_eq!(
        tensor.sum_keepdim(0).unwrap().to_vec::<f32>().unwrap(),
        [5.0, 7.0, 9.0]
    );
    assert_eq!(tensor.sum(1).unwrap().to_vec::<f32>().unwrap(), [6.0, 15.0]);
    assert_eq!(tensor.sum_all().unwrap().to_scalar::<f32>().unwrap(), 21.0);
    assert_eq!(
        tensor.mean_keepdim(1).unwrap().to_vec::<f32>().unwrap(),
        [2.0, 5.0]
    );
    assert_eq!(tensor.min(1).unwrap().to_vec::<f32>().unwrap(), [1.0, 4.0]);
    assert_eq!(tensor.min_all().unwrap().to_scalar::<f32>().unwrap(), 1.0);
    assert_eq!(
        tensor.max_keepdim(0).unwrap().to_vec::<f32>().unwrap(),
        [4.0, 5.0, 6.0]
    );
    assert_eq!(tensor.max_all().unwrap().to_scalar::<f32>().unwrap(), 6.0);

    let transposed = tensor.transpose(0, 1).unwrap();
    assert_eq!(
        transposed.sum(1).unwrap().to_vec::<f32>().unwrap(),
        [5.0, 7.0, 9.0]
    );
    assert_eq!(
        transposed.argmin(1).unwrap().to_vec::<i64>().unwrap(),
        [0, 0, 0]
    );
    assert_eq!(
        transposed.argmax(1).unwrap().to_vec::<i64>().unwrap(),
        [1, 1, 1]
    );
    assert_eq!(
        transposed
            .argmin_keepdim(1)
            .unwrap()
            .to_vec::<i64>()
            .unwrap(),
        [0, 0, 0]
    );
    assert_eq!(
        transposed
            .argmax_keepdim(1)
            .unwrap()
            .to_vec::<i64>()
            .unwrap(),
        [1, 1, 1]
    );

    let values = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
    assert_eq!(values.mean_all().unwrap().to_scalar::<f32>().unwrap(), 2.0);
    assert_eq!(values.mean(0).unwrap().to_scalar::<f32>().unwrap(), 2.0);
    assert_eq!(values.var(0).unwrap().to_scalar::<f32>().unwrap(), 1.0);
    assert_eq!(
        values.var_keepdim(0).unwrap().to_vec::<f32>().unwrap(),
        [1.0]
    );

    let integer_mean = Tensor::from_vec(vec![1u8, 2, 4], 3, &Device::Cpu).unwrap();
    assert_eq!(
        integer_mean.mean_all().unwrap().to_scalar::<u8>().unwrap(),
        2
    );
    assert!(matches!(
        Tensor::zeros([2, 0], DType::F32, &Device::Cpu)
            .unwrap()
            .min(1),
        Err(Error::EmptyReduction { .. })
    ));
    assert_eq!(
        Tensor::zeros([2, 0], DType::F32, &Device::Cpu)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap(),
        0.0
    );
    assert!(matches!(
        Tensor::zeros([2, 0], DType::F32, &Device::Cpu)
            .unwrap()
            .mean(1),
        Err(Error::EmptyReduction { .. })
    ));
    assert!(matches!(
        Tensor::ones([2, 1], DType::F32, &Device::Cpu)
            .unwrap()
            .var(1),
        Err(Error::InvalidReduction { .. })
    ));

    let nan_values = Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0], 3, &Device::Cpu).unwrap();
    assert!(
        nan_values
            .min_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .is_nan()
    );
    assert_eq!(nan_values.argmax(0).unwrap().to_scalar::<i64>().unwrap(), 1);
}

#[test]
fn phase2_comparisons_masks_clamp_and_where_support_broadcasting() {
    let matrix = Tensor::from_vec(vec![1i32, 2, 3, 4, 5, 6], (2, 3), &Device::Cpu).unwrap();
    let row = Tensor::from_vec(vec![2i32, 4, 6], 3, &Device::Cpu).unwrap();
    let mask = matrix.lt(&row).unwrap();
    assert_eq!(mask.dtype(), DType::U8);
    assert_eq!(mask.dims(), &[2, 3]);
    assert_eq!(mask.to_vec::<u8>().unwrap(), [1, 1, 1, 0, 0, 0]);
    assert_eq!(
        matrix.ge_scalar(4).unwrap().to_vec::<u8>().unwrap(),
        [0, 0, 0, 1, 1, 1]
    );
    assert_eq!(matrix.eq(&matrix).unwrap().to_vec::<u8>().unwrap(), [1; 6]);
    assert_eq!(
        matrix.ne_scalar(3).unwrap().to_vec::<u8>().unwrap(),
        [1, 1, 0, 1, 1, 1]
    );

    let clamped = matrix.clamp(2i32, 5i32).unwrap();
    assert_eq!(clamped.to_vec::<i32>().unwrap(), [2, 2, 3, 4, 5, 5]);

    let condition = Tensor::from_vec(vec![0u8, 1], (2, 1), &Device::Cpu).unwrap();
    let fallback = Tensor::full(99i32, (), &Device::Cpu).unwrap();
    let selected = condition.where_cond(&matrix, &fallback).unwrap();
    assert_eq!(selected.to_vec::<i32>().unwrap(), [99, 99, 99, 4, 5, 6]);
    assert!(matches!(
        matrix.where_cond(&matrix, &matrix),
        Err(Error::UnexpectedDType {
            expected: DType::U8,
            actual: DType::I32
        })
    ));
}

#[test]
fn phase_gemm_rank2_f32_produces_fresh_contiguous_output() {
    let lhs =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &Device::Cpu).unwrap();
    let rhs = Tensor::from_vec(
        vec![
            7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
        ],
        (3, 4),
        &Device::Cpu,
    )
    .unwrap();

    let result = lhs.matmul(&rhs).unwrap();
    assert_eq!(result.dims(), &[2, 4]);
    assert!(result.is_contiguous());
    assert!(!result.same_storage(&lhs));
    assert_eq!(
        result.to_vec::<f32>().unwrap(),
        [74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]
    );

    let shared_rhs = lhs.reshape((3, 2)).unwrap();
    let shared_result = lhs.matmul(&shared_rhs).unwrap();
    assert_eq!(
        shared_result.to_vec::<f32>().unwrap(),
        [22.0, 28.0, 49.0, 64.0]
    );
}

#[test]
fn phase_gemm_reports_phase1_shape_dtype_layout_and_empty_contracts() {
    let lhs = Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap();
    let rhs = Tensor::ones((4, 2), DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        lhs.matmul(&rhs),
        Err(Error::MatmulShapeMismatch { .. })
    ));

    let f64_lhs = Tensor::ones((2, 2), DType::F64, &Device::Cpu).unwrap();
    let f64_rhs = Tensor::ones((2, 2), DType::F64, &Device::Cpu).unwrap();
    assert!(matches!(
        f64_lhs.matmul(&f64_rhs),
        Err(Error::UnsupportedMatmulDType { dtype: DType::F64 })
    ));

    let transposed = lhs.transpose(0, 1).unwrap();
    let compatible_rhs = Tensor::ones((2, 2), DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        transposed.matmul(&compatible_rhs),
        Err(Error::UnsupportedMatmulLayout)
    ));

    let empty_lhs = Tensor::zeros((0, 3), DType::F32, &Device::Cpu).unwrap();
    let empty_rhs = Tensor::ones((3, 2), DType::F32, &Device::Cpu).unwrap();
    let empty_result = empty_lhs.matmul(&empty_rhs).unwrap();
    assert_eq!(empty_result.dims(), &[0, 2]);
    assert!(empty_result.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn phase3_unfold_flip_and_padding_preserve_index_contracts() {
    let tensor = Tensor::from_vec((0u8..6).collect(), (2, 3), &Device::Cpu).unwrap();
    let windows = tensor.unfold(1, 2, 1).unwrap();
    assert_eq!(windows.dims(), &[2, 2, 2]);
    assert_eq!(windows.stride(), &[3, 1, 1]);
    assert!(windows.same_storage(&tensor));
    assert_eq!(windows.to_vec::<u8>().unwrap(), [0, 1, 1, 2, 3, 4, 4, 5]);
    assert!(matches!(
        tensor.unfold(1, 2, 0),
        Err(Error::InvalidUnfold { .. })
    ));
    assert!(matches!(
        tensor.unfold(1, 4, 1),
        Err(Error::InvalidUnfold { .. })
    ));

    let flipped = tensor.flip(&[0, 1]).unwrap();
    assert!(!flipped.same_storage(&tensor));
    assert_eq!(flipped.to_vec::<u8>().unwrap(), [5, 4, 3, 2, 1, 0]);
    assert_eq!(
        tensor.flip(&[0]).unwrap().to_vec::<u8>().unwrap(),
        [3, 4, 5, 0, 1, 2]
    );

    let padded = tensor.pad_with_zeros(1, 1, 2).unwrap();
    assert_eq!(padded.dims(), &[2, 6]);
    assert_eq!(
        padded.to_vec::<u8>().unwrap(),
        [0, 0, 1, 2, 0, 0, 0, 3, 4, 5, 0, 0]
    );
    let same = tensor.pad_with_same(0, 1, 1).unwrap();
    assert_eq!(
        same.to_vec::<u8>().unwrap(),
        [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]
    );
    assert!(matches!(
        Tensor::zeros([2, 0], DType::U8, &Device::Cpu)
            .unwrap()
            .pad_with_same(1, 1, 0),
        Err(Error::EmptyTensorForOp {
            op: "pad_with_same"
        })
    ));
}

#[test]
fn phase3_gather_index_select_and_embedding_are_layout_aware() {
    let values = Tensor::from_vec((0i32..6).collect(), (2, 3), &Device::Cpu).unwrap();
    let indexes = Tensor::from_vec(vec![2i64, 0, 1, 2], (2, 2), &Device::Cpu).unwrap();
    assert_eq!(
        values.gather(&indexes, 1).unwrap().to_vec::<i32>().unwrap(),
        [2, 0, 4, 5]
    );

    let selected_rows = Tensor::from_vec(vec![1i64, 0], 2, &Device::Cpu).unwrap();
    assert_eq!(
        values
            .index_select(&selected_rows, 0)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [3, 4, 5, 0, 1, 2]
    );
    let selected_columns = Tensor::from_vec(vec![2u32, 0], 2, &Device::Cpu).unwrap();
    assert_eq!(
        values
            .index_select(&selected_columns, 1)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [2, 0, 5, 3]
    );

    let embedding =
        Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], (3, 2), &Device::Cpu).unwrap();
    let ids = Tensor::from_vec(vec![2i64, 1], 2, &Device::Cpu).unwrap();
    assert_eq!(
        embedding.embedding(&ids).unwrap().to_vec::<f32>().unwrap(),
        [4.0, 5.0, 2.0, 3.0]
    );

    let bad = Tensor::from_vec(vec![3i64], 1, &Device::Cpu).unwrap();
    assert!(matches!(
        values.index_select(&bad, 0),
        Err(Error::InvalidIndex {
            op: "index_select",
            index: 3,
            size: 2
        })
    ));
    let negative = Tensor::from_vec(vec![-1i32], 1, &Device::Cpu).unwrap();
    assert!(matches!(
        values.index_select(&negative, 0),
        Err(Error::NegativeIndex {
            op: "index_select",
            value: -1
        })
    ));
}

#[test]
fn phase3_scatter_index_add_and_slice_scatter_define_duplicate_and_aliasing_behavior() {
    let base = Tensor::zeros((1, 3), DType::I32, &Device::Cpu).unwrap();
    let indexes = Tensor::from_vec(vec![1i64, 1], (1, 2), &Device::Cpu).unwrap();
    let source = Tensor::from_vec(vec![2i32, 3], (1, 2), &Device::Cpu).unwrap();
    assert_eq!(
        base.scatter(&indexes, &source, 1)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [0, 3, 0]
    );
    assert_eq!(
        base.scatter_add(&indexes, &source, 1)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [0, 5, 0]
    );

    let inplace = Tensor::zeros((1, 3), DType::I32, &Device::Cpu).unwrap();
    inplace.scatter_set(&indexes, &source, 1).unwrap();
    assert_eq!(inplace.to_vec::<i32>().unwrap(), [0, 3, 0]);
    inplace.scatter_add_set(&indexes, &source, 1).unwrap();
    assert_eq!(inplace.to_vec::<i32>().unwrap(), [0, 8, 0]);
    assert!(matches!(
        inplace.scatter_set(
            &Tensor::from_vec(vec![0i64, 1, 2], (1, 3), &Device::Cpu).unwrap(),
            &inplace,
            1
        ),
        Err(Error::StorageAliasConflict { op: "scatter_set" })
    ));

    let target = Tensor::from_vec(vec![1i32, 1, 1, 1], (2, 2), &Device::Cpu).unwrap();
    let add_indexes = Tensor::from_vec(vec![1i64, 0, 1], 3, &Device::Cpu).unwrap();
    let add_source = Tensor::from_vec(vec![2i32, 3, 4, 5, 6, 7], (3, 2), &Device::Cpu).unwrap();
    assert_eq!(
        target
            .index_add(&add_indexes, &add_source, 0)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [5, 6, 9, 11]
    );

    let destination = Tensor::zeros((2, 4), DType::I32, &Device::Cpu).unwrap();
    let slice = Tensor::from_vec(vec![1i32, 2, 3, 4], (2, 2), &Device::Cpu).unwrap();
    assert_eq!(
        destination
            .slice_scatter(&slice, 1, 1)
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [0, 1, 2, 0, 0, 3, 4, 0]
    );
    assert_eq!(
        destination
            .slice_scatter0(
                &Tensor::from_vec(vec![7i32, 8, 9, 10], (1, 4), &Device::Cpu).unwrap(),
                1
            )
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        [0, 0, 0, 0, 7, 8, 9, 10]
    );
}
