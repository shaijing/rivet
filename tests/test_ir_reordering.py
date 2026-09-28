"""Standalone IR reordering checks using generated PNG/Arrow data.

Run: .venv/bin/python tests/test_ir_reordering.py --show-plans
Also collected by pytest; no downloaded dataset or pytest fixture is needed.
"""

from __future__ import annotations

import argparse
import io
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image

import rivet


def write_arrow(path: Path, encoded: list[bytes]) -> None:
    table = pa.table(
        {"image": encoded, "label": list(range(len(encoded)))},
        schema=pa.schema([("image", pa.binary()), ("label", pa.int64())]),
    )
    with (
        pa.OSFile(str(path), "wb") as output,
        pa.ipc.new_stream(output, table.schema) as writer,
    ):
        writer.write_table(table, max_chunksize=6)


def graph_chain(graph: str) -> list[str]:
    """Follow input edges, since arena/display order is not execution order."""
    root = re.search(r"^LogicalPlan\(root=%(\d+)\)$", graph, re.MULTILINE)
    if root is None:
        raise AssertionError(f"missing logical graph root:\n{graph}")
    nodes: dict[int, tuple[str, list[int]]] = {}
    for match in re.finditer(
        r"^\s+%(\d+) (\w+)(?: ([\w:]+))? <- \[(.*?)\]$", graph, re.MULTILINE
    ):
        node, kind, payload, parents = match.groups()
        label = payload.rsplit("::", 1)[-1] if payload else kind
        nodes[int(node)] = (
            label,
            [int(parent) for parent in re.findall(r"%(\d+)", parents)],
        )
    chain = []
    seen = set()
    current = int(root.group(1))
    while True:
        if current in seen:
            raise AssertionError("cycle in explained graph")
        seen.add(current)
        label, parents = nodes[current]
        chain.append(label)
        if not parents:
            break
        if len(parents) != 1:
            raise AssertionError("this test expects a unary pipeline")
        current = parents[0]
    return chain[::-1]


def plans(explanation: str) -> tuple[list[str], list[str]]:
    declared, optimized = explanation.split("Optimized logical graph:\n", 1)
    declared = declared.split("Declared logical graph:\n", 1)[1]
    optimized = optimized.split("PlacementPlan", 1)[0]
    return graph_chain(declared), graph_chain(optimized)


def collect(pipeline: rivet.Pipeline) -> tuple[np.ndarray, np.ndarray]:
    batches = list(pipeline.execute())
    if not batches:
        raise AssertionError("expected at least one batch")
    return (
        np.concatenate([batch["images"] for batch in batches]),
        np.concatenate([batch["labels"] for batch in batches]),
    )


class TestIRReordering(unittest.TestCase):
    worker_counts = (0, 3)
    epoch_values = (0, 7)
    show_plans = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="rivet-ir-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.path = Path(cls.temp.name) / "images.arrow"
        cls.bad_path = Path(cls.temp.name) / "corrupt.arrow"
        cls.pixels = np.random.default_rng(123).integers(
            0, 256, (24, 32, 32, 3), dtype=np.uint8
        )
        encoded = []
        for pixels in cls.pixels:
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            encoded.append(buffer.getvalue())
        write_arrow(cls.path, encoded)
        corrupt = encoded.copy()
        corrupt[4] = b"not an encoded image"
        write_arrow(cls.bad_path, corrupt)

    def scan(
        self, workers: int, epoch: int = 0, *, corrupt: bool = False
    ) -> rivet.Pipeline:
        return (
            rivet.scan_arrow(
                [self.bad_path if corrupt else self.path], image_column="image"
            )
            .seed(123)
            .epoch(epoch)
            .workers(workers)
        )

    def assert_plan(
        self, pipeline: rivet.Pipeline, declared: list[str], optimized: list[str]
    ) -> str:
        explanation = pipeline.explain()
        before, after = plans(explanation)
        self.assertEqual(before, ["Source", *declared, "BatchConfig", "Sink"])
        self.assertEqual(after, ["Source", *optimized, "BatchConfig", "Sink"])
        if self.show_plans:
            print(
                f"\n{self._testMethodName}:\n  declared: {' -> '.join(before)}\n  optimized: {' -> '.join(after)}"
            )
        return explanation

    def test_take_crosses_decode_and_random_crop(self) -> None:
        for workers in self.worker_counts:
            for epoch in self.epoch_values:
                with self.subTest(workers=workers, epoch=epoch):
                    base = self.scan(workers, epoch)
                    original = (
                        base.decode_image()
                        .random_crop(32, 32, padding=4)
                        .take(4)
                        .shuffle(11)
                        .batch(4)
                    )
                    explanation = self.assert_plan(
                        original,
                        ["Decode", "RandomCrop", "IndexOp", "IndexOp"],
                        ["IndexOp", "IndexOp", "Decode", "RandomCrop"],
                    )
                    self.assertIn("rewrite.index-source-pushdown", explanation)
                    pushed = (
                        base.take(4)
                        .decode_image()
                        .random_crop(32, 32, padding=4)
                        .shuffle(11)
                        .batch(4)
                    )
                    actual, labels = collect(original)
                    expected, expected_labels = collect(pushed)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(labels, expected_labels)
                    self.assertEqual(sorted(labels.tolist()), [0, 1, 2, 3])

    def test_selected_random_samples_match_full_execution(self) -> None:
        for workers in self.worker_counts:
            for epoch in self.epoch_values:
                with self.subTest(workers=workers, epoch=epoch):
                    transformed = (
                        self.scan(workers, epoch)
                        .decode_image()
                        .random_crop(32, 32, padding=4)
                        .random_horizontal_flip(0.5)
                    )
                    full, full_labels = collect(transformed.batch(5))
                    selected = transformed.skip(3).take(7).shuffle(11).batch(3)
                    self.assert_plan(
                        selected,
                        [
                            "Decode",
                            "RandomCrop",
                            "RandomHorizontalFlip",
                            "IndexOp",
                            "IndexOp",
                            "IndexOp",
                        ],
                        [
                            "IndexOp",
                            "IndexOp",
                            "IndexOp",
                            "Decode",
                            "RandomCrop",
                            "RandomHorizontalFlip",
                        ],
                    )
                    actual, labels = collect(selected)
                    self.assertEqual(sorted(labels.tolist()), list(range(3, 10)))
                    by_label = dict(zip(full_labels.tolist(), full, strict=True))
                    np.testing.assert_array_equal(
                        actual, np.stack([by_label[int(label)] for label in labels])
                    )

    def test_take_and_shuffle_retain_different_semantics(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                decoded = self.scan(workers).decode_image()
                full, full_labels = collect(decoded.shuffle(11).batch(5))
                taken, taken_labels = collect(decoded.take(4).shuffle(11).batch(4))
                shuffled, shuffled_labels = collect(
                    decoded.shuffle(11).take(4).batch(4)
                )
                np.testing.assert_array_equal(shuffled, full[:4])
                np.testing.assert_array_equal(shuffled_labels, full_labels[:4])
                self.assertEqual(sorted(taken_labels.tolist()), [0, 1, 2, 3])
                np.testing.assert_array_equal(taken, self.pixels[taken_labels])
                self.assertFalse(np.array_equal(taken_labels, shuffled_labels))

    def test_slices_do_not_cross_a_shuffle(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                decoded = self.scan(workers).decode_image()
                full, labels = collect(
                    decoded.take(15).shuffle(11).shuffle(19).batch(6)
                )
                sliced = (
                    decoded.take(15)
                    .shuffle(11)
                    .skip(2)
                    .take(9)
                    .shuffle(19)
                    .take(5)
                    .batch(3)
                )
                self.assert_plan(
                    sliced,
                    ["Decode", *["IndexOp"] * 6],
                    [*["IndexOp"] * 6, "Decode"],
                )
                actual, actual_labels = collect(sliced)
                # Apply the second permutation to the already sliced sequence.
                first_order = collect(decoded.take(15).shuffle(11).batch(6))[1]
                small = decoded.take(9)
                first_small = collect(small.shuffle(11).batch(6))[1]
                composed_small = collect(small.shuffle(11).shuffle(19).batch(6))[1]
                # Recover the second permutation's positions from a complete
                # sequence, independently of the interleaved slice pipeline.
                second_positions = np.argsort(first_small)[composed_small]
                expected_labels = first_order[2:11][second_positions[:5]]
                np.testing.assert_array_equal(actual_labels, expected_labels)
                np.testing.assert_array_equal(actual, self.pixels[actual_labels])
                # Repeated shuffle must not collapse to the first permutation.
                np.testing.assert_array_equal(full, self.pixels[labels])
                self.assertFalse(
                    np.array_equal(
                        labels, collect(decoded.take(15).shuffle(11).batch(6))[1]
                    )
                )

    def test_random_streams_match_across_workers_and_change_with_epoch(self) -> None:
        def run(workers: int, epoch: int) -> tuple[np.ndarray, np.ndarray]:
            return collect(
                self.scan(workers, epoch)
                .decode_image()
                .random_crop(32, 32, padding=4)
                .random_horizontal_flip(0.5)
                .take(8)
                .shuffle(11)
                .batch(3)
            )

        for epoch in self.epoch_values:
            expected, expected_labels = run(0, epoch)
            for workers in self.worker_counts:
                with self.subTest(workers=workers, epoch=epoch):
                    actual, labels = run(workers, epoch)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(labels, expected_labels)
        # Sort by original source identity to isolate augmentation changes
        # from the sampler's different epoch permutation.
        before, before_labels = run(0, 0)
        after, after_labels = run(0, 1)
        self.assertFalse(
            np.array_equal(
                before[np.argsort(before_labels)], after[np.argsort(after_labels)]
            )
        )

    def test_explain_is_repeatable_and_keeps_declared_pipeline(self) -> None:
        pipeline = (
            self.scan(0, corrupt=True)
            .decode_image()
            .random_crop(32, 32, padding=4)
            .take(4)
            .batch(4)
        )
        explanation = pipeline.explain()
        self.assertEqual(pipeline.explain(), explanation)
        before, after = plans(explanation)
        self.assertNotEqual(before, after)
        first, labels = collect(pipeline)
        repeated, repeated_labels = collect(pipeline)
        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(labels, repeated_labels)
        self.assertEqual(pipeline.explain(), explanation)

    def test_adjacent_slices_are_canonicalized(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                selected = (
                    self.scan(workers)
                    .decode_image()
                    .skip(2)
                    .take(12)
                    .skip(3)
                    .take(7)
                    .take(20)
                    .batch(4)
                )
                explanation = self.assert_plan(
                    selected,
                    ["Decode", *["IndexOp"] * 5],
                    ["IndexOp", "IndexOp", "Decode"],
                )
                self.assertIn("rewrite.canonical-selection", explanation)
                images, labels = collect(selected)
                np.testing.assert_array_equal(labels, np.arange(5, 12))
                np.testing.assert_array_equal(images, self.pixels[5:12])

    def test_inverse_layouts_are_removed(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                pipeline = (
                    self.scan(workers)
                    .decode_image()
                    .hwc_to_chw()
                    .chw_to_hwc()
                    .take(4)
                    .batch(4)
                )
                explanation = self.assert_plan(
                    pipeline,
                    ["Decode", "Layout", "Layout", "IndexOp"],
                    ["IndexOp", "Decode"],
                )
                self.assertIn("rewrite.inverse-layout", explanation)
                images, labels = collect(pipeline)
                np.testing.assert_array_equal(images, self.pixels[:4])
                np.testing.assert_array_equal(labels, np.arange(4))

    def test_padded_crop_stays_before_normalize(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                pipeline = (
                    self.scan(workers)
                    .decode_image()
                    .random_crop(40, 40, padding=4)
                    .normalize([0.5], [0.5])
                    .take(4)
                    .batch(4)
                )
                self.assert_plan(
                    pipeline,
                    ["Decode", "RandomCrop", "Normalize", "IndexOp"],
                    ["IndexOp", "Decode", "RandomCrop", "Normalize"],
                )
                images, labels = collect(pipeline)
                padded = np.pad(self.pixels[labels], ((0, 0), (4, 4), (4, 4), (0, 0)))
                expected = padded.astype(np.float32) * np.float32(2 / 255) - np.float32(
                    1
                )
                np.testing.assert_allclose(images, expected, rtol=1e-6, atol=1e-7)
                self.assertTrue(np.all(images[:, :4] == -1))
                self.assertTrue(np.all(images[:, -4:] == -1))
                self.assertTrue(np.all(images[:, :, :4] == -1))
                self.assertTrue(np.all(images[:, :, -4:] == -1))

    def test_normalize_then_crop_is_rejected(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                invalid = (
                    self.scan(workers)
                    .decode_image()
                    .normalize([0.5], [0.5])
                    .crop(0, 0, 16, 16)
                    .batch(4)
                )
                for action in (invalid.explain, invalid.execute):
                    with self.assertRaisesRegex(ValueError, "Crop"):
                        action()

    def test_selection_avoids_decoding_excluded_corrupt_row(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                base = self.scan(workers, corrupt=True).decode_image()
                selected = base.take(4).shuffle(11).batch(4)
                self.assert_plan(
                    selected,
                    ["Decode", "IndexOp", "IndexOp"],
                    ["IndexOp", "IndexOp", "Decode"],
                )
                images, labels = collect(selected)
                np.testing.assert_array_equal(images, self.pixels[labels])
                self.assertEqual(sorted(labels.tolist()), [0, 1, 2, 3])
                with self.assertRaisesRegex(RuntimeError, "(?i)(image|format|decode)"):
                    list(base.take(5).batch(5).execute())
                self.assertEqual(list(base.take(0).batch(4).execute()), [])

    def test_index_cannot_cross_an_explicit_batch(self) -> None:
        for workers in self.worker_counts:
            with self.subTest(workers=workers):
                invalid = self.scan(workers).decode_image().batch(4).take(2)
                for action in (invalid.explain, invalid.execute):
                    with self.assertRaisesRegex(
                        ValueError, "outside the shared source prefix"
                    ):
                        action()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 7])
    parser.add_argument("--show-plans", action="store_true")
    args = parser.parse_args()
    if any(value < 0 for value in [*args.workers, *args.epochs]):
        parser.error("workers and epochs must be nonnegative")
    TestIRReordering.worker_counts = tuple(args.workers)
    TestIRReordering.epoch_values = tuple(args.epochs)
    TestIRReordering.show_plans = args.show_plans
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestIRReordering)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
