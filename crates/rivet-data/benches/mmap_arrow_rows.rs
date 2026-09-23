use arrow::array::{Array, BinaryArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_ipc::writer::StreamWriter;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rivet_data::dataset::arrow::MmapArrowTable;
use std::fs::File;
use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;

const BATCH_ROWS: usize = 512;
const BATCH_COUNT: usize = 128;
const REQUEST_SIZES: [usize; 3] = [128, 1024, 8192];

fn bench_row_access(criterion: &mut Criterion) {
    let (path, table) = build_table();
    let total_rows = BATCH_ROWS * BATCH_COUNT;
    let mut group = criterion.benchmark_group("mmap_arrow_row_access");
    group.sample_size(20);

    for size in REQUEST_SIZES {
        for (pattern, indices) in [
            ("sequential", sequential_indices(total_rows, size)),
            ("random", random_indices(total_rows, size)),
            ("repeated", repeated_indices(total_rows, size)),
        ] {
            assert_eq!(
                read_individually(&table, &indices),
                read_grouped(&table, &indices),
                "grouped row lookup must preserve requested order and duplicates"
            );
            assert_eq!(
                read_individually(&table, &indices),
                read_rows(&table, &indices),
                "rows() must preserve requested order and duplicates"
            );
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("row/{pattern}"), size),
                &indices,
                |bencher, indices| {
                    bencher.iter(|| black_box(read_individually(&table, black_box(indices))));
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("rows_grouped/{pattern}"), size),
                &indices,
                |bencher, indices| {
                    bencher.iter(|| black_box(read_grouped(&table, black_box(indices))));
                },
            );
        }
    }

    group.finish();
    drop(table);
    let _ = std::fs::remove_file(path);
}

fn read_individually(table: &MmapArrowTable, indices: &[usize]) -> usize {
    indices
        .iter()
        .enumerate()
        .map(|(request_index, &index)| {
            let row = table.row(index).expect("benchmark indices are valid");
            (request_index + 1).wrapping_mul(
                black_box(row.binary("image").expect("image column").len())
                    + black_box(row.i64("label").expect("label column") as usize),
            )
        })
        .fold(0usize, usize::wrapping_add)
}

fn read_grouped(table: &MmapArrowTable, indices: &[usize]) -> usize {
    let groups = table
        .group_rows(indices)
        .expect("benchmark indices are valid");
    let mut output = vec![0usize; indices.len()];
    for group in groups {
        let image = group
            .batch
            .column_by_name("image")
            .expect("image column")
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("binary image column");
        let label = group
            .batch
            .column_by_name("label")
            .expect("label column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 label column");
        let image_offsets = image.value_offsets();

        for requested in group.rows {
            let start = image_offsets[requested.batch_row] as usize;
            let end = image_offsets[requested.batch_row + 1] as usize;
            let image = image.values().slice_with_length(start, end - start);
            let value =
                black_box(image.len()) + black_box(label.value(requested.batch_row) as usize);
            output[requested.request_index] = (requested.request_index + 1).wrapping_mul(value);
        }
    }
    output.into_iter().fold(0usize, usize::wrapping_add)
}

fn read_rows(table: &MmapArrowTable, indices: &[usize]) -> usize {
    table
        .rows(indices)
        .expect("benchmark indices are valid")
        .iter()
        .enumerate()
        .map(|(request_index, row)| {
            (request_index + 1).wrapping_mul(
                row.binary("image").expect("image column").len()
                    + row.i64("label").expect("label column") as usize,
            )
        })
        .fold(0usize, usize::wrapping_add)
}

fn sequential_indices(total_rows: usize, count: usize) -> Vec<usize> {
    (0..count).map(|index| index % total_rows).collect()
}

fn random_indices(total_rows: usize, count: usize) -> Vec<usize> {
    let mut state = 0x8a5c_9d37_4b1e_620f_u64;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as usize) % total_rows
        })
        .collect()
}

fn repeated_indices(total_rows: usize, count: usize) -> Vec<usize> {
    let unique = (0..16.min(total_rows))
        .map(|index| index * (total_rows / 16).max(1))
        .collect::<Vec<_>>();
    (0..count)
        .map(|index| unique[index % unique.len()])
        .collect()
}

fn build_table() -> (PathBuf, MmapArrowTable) {
    let path = std::env::temp_dir().join(format!(
        "rivet-mmap-arrow-rows-{}.arrow",
        std::process::id()
    ));
    let file = File::create(&path).expect("create benchmark IPC file");
    let schema = Arc::new(Schema::new(vec![
        Field::new("image", DataType::Binary, false),
        Field::new("label", DataType::Int64, false),
    ]));
    let mut writer = StreamWriter::try_new(file, &schema).expect("create IPC stream writer");

    for batch_index in 0..BATCH_COUNT {
        let start = batch_index * BATCH_ROWS;
        let images = (start..start + BATCH_ROWS)
            .map(|row| vec![(row % 251) as u8; 96 + row % 64])
            .collect::<Vec<_>>();
        let labels = (start..start + BATCH_ROWS)
            .map(|row| (row % 5) as i64)
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(BinaryArray::from_iter_values(
                    images.iter().map(Vec::as_slice),
                )),
                Arc::new(Int64Array::from(labels)),
            ],
        )
        .expect("construct benchmark record batch");
        writer.write(&batch).expect("write benchmark record batch");
    }
    writer.finish().expect("finish benchmark IPC stream");
    drop(writer);
    let table = MmapArrowTable::from_files([&path]).expect("open mmap benchmark table");
    (path, table)
}

criterion_group!(benches, bench_row_access);
criterion_main!(benches);
