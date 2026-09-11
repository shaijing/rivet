import lance
import pyarrow.dataset as ds

input_path = "/tmp/data.parquet"
output_path = "/tmp/data.lance"

parquet_ds = ds.dataset(input_path, format="parquet")

lance_ds = lance.write_dataset(
    parquet_ds,
    output_path,
    mode="overwrite",
)

print(lance_ds.count_rows())
print(lance_ds.schema)
