# Pipeline DAG validation

Both implementations were built as release Python extensions with maturin.
The baseline was built from commit `1de12fa` in a temporary detached worktree.
The final extension was rebuilt and its new graph compiler was verified before
collecting the final results. The environment now uses the final branch's
editable extension; the temporary worktree has been removed.

```bash
VIRTUAL_ENV="$PWD/.venv" uvx --from 'maturin>=1.12,<2.0' maturin develop --release -j 12
.venv/bin/python bench/compare_cifar10_torchvision_rivet_decoded.py \
  --torch-root /home/lingyu/.data/pytorch \
  --rivet-root /home/lingyu/.cache/rivet/datasets/cifar10 \
  --batch 128 --workers 4 --epochs 3
```

| Implementation | Steady runs (img/s) | Mean | Range | Population stddev |
| --- | --- | --- | --- | --- |
| Base commit 1de12fa | 750794, 751890, 755614 | 752766 | 750794–755614 | 2063 |
| Executable graph with typed CPU traversal | 757132, 743646, 758249 | 753009 | 743646–758249 | 6636 |

The final mean differs by +0.032%. These runs show comparable steady
throughput, with overlapping ranges; they do not establish a speedup. An earlier
wrapper-based graph traversal averaged 743032 img/s. Investigation removed the
extra enum dispatch and singleton program vectors from the linear path: it now
caches typed operator handles shared with physical nodes. The specialized image
kernels and aligned output allocations remain unchanged.

For worktree comparisons, avoid reusing stale shared Cargo outputs: this run
required a scoped release cleanup of `rivet-plan`, `rivet-exec`, `rivet-vision`
and `rivet-python` when restoring the final extension.

Validation passed:

- `cargo test -j 12 -p rivet-plan --lib`: 12 tests.
- `cargo test -j 12 -p rivet-exec --lib --no-default-features`: 26 tests.
- `cargo test -j 12 -p rivet-vision --lib`: 169 tests, including Lance.
- `.venv/bin/pytest -q tests/test_arrow_pipeline.py`: 33 tests on the final release extension.
- The updated `pipeline_regression_bench` example passed its scoped check.
- `git diff --check` passed.

Regular Clippy completed. Strict `-D warnings` remains blocked by existing
repository warnings, including `rivet-core` needless borrows and manually
implemented defaults. The new graph compiler, DAG scheduler, typed vision
operators and loader report no Clippy warnings.

See [pipeline IR](../pipeline-ir.md) for the Rust graph API and current DAG
constraints.
