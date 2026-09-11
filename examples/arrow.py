# import pyarrow as pa
# import pyarrow.parquet as pq
# from pyarrow import ipc

# input_path = "/home/ling/.cache/huggingface/datasets/uoft-cs___cifar10/plain_text/0.0.0/0b2714987fa478483af9968de7c934580d0bb9a2/cifar10-train.arrow"
# output_path = "/tmp/data.parquet"

# # 读取 Arrow IPC file
# with pa.memory_map(input_path, "r") as source:
#     reader = ipc.open_stream(source)
#     table = reader.read_all()

# # 写成 Parquet
# pq.write_table(table, output_path)
from pathlib import Path

import lance
import pyarrow as pa
from pyarrow import ipc

train_arrow = "/home/ling/.cache/huggingface/datasets/uoft-cs___cifar100/cifar100/0.0.0/aadb3af77e9048adbea6b47c21a81e47dd092ae5/cifar100-train.arrow"

test_arrow = "/home/ling/.cache/huggingface/datasets/uoft-cs___cifar100/cifar100/0.0.0/aadb3af77e9048adbea6b47c21a81e47dd092ae5/cifar100-test.arrow"

output_root = Path("/data/datasets/rivet/cifar100")
output_root.mkdir(parents=True, exist_ok=True)


def arrow_stream_to_lance(input_path: str, output_path: str):
    with pa.memory_map(input_path, "r") as source:
        reader = ipc.open_stream(source)

        # 直接把 RecordBatchReader 交给 Lance
        dataset = lance.write_dataset(
            reader,
            output_path,
            schema=reader.schema,
            mode="overwrite",
        )

    print(f"written: {output_path}")
    print("rows:", dataset.count_rows())
    print("schema:")
    print(dataset.schema)


arrow_stream_to_lance(
    train_arrow,
    str(output_root / "train.lance"),
)

arrow_stream_to_lance(
    test_arrow,
    str(output_root / "test.lance"),
)