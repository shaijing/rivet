# SST-5 ArrowText reference

The `inspect_sst5_arrowtext` example reads the Hugging Face cached Arrow IPC
streams under:

```text
/home/ling/.cache/huggingface/datasets/setfit___sst5/
```

It searches below the supplied directory for `sst5-{train,validation,test}.arrow`,
prints each schema and label histogram, and samples three rows. Run it with:

```bash
cargo run -p rivet-text --example inspect_sst5_arrowtext
# Or pass the dataset cache directory as the first argument.
cargo run -p rivet-text --example inspect_sst5_arrowtext -- /path/to/setfit___sst5
```

## Observed layout

The cache contains three Arrow IPC stream files. The split metadata reports
8,544 training rows, 1,101 validation rows, and 2,210 test rows (11,855 total).
Each file has the same schema:

| Column | Arrow type | Meaning |
| --- | --- | --- |
| `text` | nullable `Utf8` | Review sentence or phrase |
| `label` | nullable `Int64` | Sentiment class id, 0 through 4 |
| `label_text` | nullable `Utf8` | Human-readable sentiment name |

The cached batches inspected here contain no null values. Both label columns
encode the same target: `0 = very negative`, `1 = negative`, `2 = neutral`,
`3 = positive`, and `4 = very positive`.

| Split | Rows | Very negative | Negative | Neutral | Positive | Very positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 8,544 | 1,092 | 2,218 | 1,624 | 2,322 | 1,288 |
| validation | 1,101 | 139 | 289 | 229 | 279 | 165 |
| test | 2,210 | 279 | 633 | 389 | 510 | 399 |

Approximate sequence lengths from `text.split_whitespace()` are:

| Split | Mean | p50 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 19.1 | 18 | 35 | 43 | 52 |
| validation | 19.3 | 19 | 35 | 42 | 49 |
| test | 19.2 | 18 | 35 | 42 | 56 |

These are whitespace counts, not counts from a model tokenizer. Sample strings
show punctuation and some contractions separated by spaces, so preprocessing
should preserve the input string and let an explicitly selected tokenizer
decide how to split it.

## Design notes

- A first text sample can be represented as raw text plus a compact integer
  target; `label_text` can be supplied by a shared label vocabulary instead
  of copied into every in-memory sample.
- Keep split identity (`train`, `validation`, `test`) at the source/dataset
  level. It is not a per-row feature in these files.
- Variable-length input is the normal case. A batch operation should own
  truncation and padding policy; the observed whitespace lengths are useful
  estimates, but must not become tokenizer-independent limits.
- Arrow `Utf8` stores variable-length strings, which suits source reads and
  projection of only `text` and `label`. Whether to keep Arrow buffers alive,
  return owned strings, or tokenize during reads should be decided by the
  later source and execution API.
- The three splits show some label imbalance. Sampling or stratification
  policy should remain an explicit data/planning choice rather than being
  hidden in tokenization.
