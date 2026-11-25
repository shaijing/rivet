from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import rivet

TOKENIZER = (
    Path(__file__).resolve().parents[1]
    / "crates/rivet-text/tests/fixtures/wordlevel.json"
)


def test_local_text_pipeline(tmp_path):
    text = rivet.text
    first = tmp_path / "first.txt"
    first.write_bytes("hello world\r\ngood\r\n".encode("utf-16"))
    second = tmp_path / "second.txt"
    second.write_bytes("bad".encode("utf-16"))
    dataset = text.TextDataset.from_files([first, second], encoding="utf-16")
    assert list(dataset) == [
        ("hello world", None, None),
        ("good", None, None),
        ("bad", None, None),
    ]
    assert dataset[-1] == ("bad", None, None)
    with pytest.raises(IndexError):
        dataset[-4]

    tokenizer = text.TextTokenizer.from_file(TOKENIZER)
    ids = tokenizer.encode("hello world")["input_ids"]
    assert tokenizer.decode(ids) == "hello world"
    pipeline = (
        text.TextPipeline(dataset, tokenizer)
        .indices([1, 0, 1])
        .truncate(8)
        .padding(tokenizer.token_to_id("[PAD]"))
        .batch(2)
    )
    first_batch, tail = pipeline
    assert first_batch["source_indices"] == [1, 0]
    assert tail["source_indices"] == [1]
    assert first_batch["labels"] is None
    assert first_batch["lengths"] == [3, 4]
    assert next(iter(pipeline))["source_indices"] == [1, 0]
    for field in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
        assert first_batch[field].shape == (2, 4)
        assert first_batch[field].dtype == np.int64
        assert not first_batch[field].flags.writeable
    np.testing.assert_array_equal(
        first_batch["attention_mask"], [[1, 1, 1, 0], [1, 1, 1, 1]]
    )


def test_structured_and_arrow_sources(tmp_path):
    text = rivet.text
    rows = tmp_path / "reviews.jsonl"
    rows.write_text(
        '{"body":"hello","pair":"world","class":2}\n'
        '{"body":"world","pair":"hello","class":4}\n',
        encoding="utf-8",
    )
    arrow = tmp_path / "reviews.arrow"
    batch = pa.record_batch(
        {"body": ["hello", "world"], "pair": ["world", "hello"], "class": [2, 4]}
    )
    with pa.ipc.new_stream(arrow, batch.schema) as writer:
        writer.write_batch(batch)
    columns = {
        "text_column": "body",
        "text_pair_column": "pair",
        "label_column": "class",
    }
    tokenizer = text.TextTokenizer.from_json(TOKENIZER.read_text(encoding="utf-8"))
    for dataset in (
        text.TextDataset.from_records(str(rows), **columns),
        text.TextDataset.from_arrow(arrow, **columns),
    ):
        assert dataset[0] == ("hello", "world", 2)
        loader = text.TextPipeline(dataset, tokenizer).indices([1, 0]).compile()
        batch = next(loader)
        assert batch["source_indices"] == [1, 0]
        assert batch["labels"].tolist() == [4, 2]
        assert (
            batch["input_ids"][0].tolist()
            == tokenizer.encode("world", "hello")["input_ids"]
        )
        assert next(loader, None) is None
        with pytest.raises(ValueError, match="batch size"):
            text.TextPipeline(dataset, tokenizer).batch(0).compile()
