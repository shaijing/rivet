"""Load local text datasets, tokenize rows, and iterate NumPy batches.

Use :meth:`TextDataset.from_files` for plain text,
:meth:`TextDataset.from_records` for structured files and archives, or
:meth:`TextDataset.from_arrow` for an Arrow IPC stream. Pass the dataset and a
:class:`TextTokenizer` to :class:`TextPipeline` to build a reusable pipeline.

Example::

    from rivet import text

    dataset = text.TextDataset.from_records("reviews.jsonl", label_column="label")
    tokenizer = text.TextTokenizer.from_file("tokenizer.json")
    pipeline = (
        text.TextPipeline(dataset, tokenizer)
        .truncate(128)
        .padding(tokenizer.token_to_id("[PAD]"))
        .batch(32)
    )
    for batch in pipeline:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

Batch dictionaries contain read-only NumPy int64 arrays for model inputs
and labels, plus lists for lengths and source indices. Unlabeled batches
have a None value for labels.
"""

from ._rivet import TextDataLoader, TextDataset, TextPipeline, TextTokenizer

__all__ = ["TextDataLoader", "TextDataset", "TextPipeline", "TextTokenizer"]
