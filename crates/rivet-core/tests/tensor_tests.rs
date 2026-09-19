use rivet_core::{DType, Device, Layout, Shape, Tensor};

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
