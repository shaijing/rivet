"""Native multi-view expansion and batching, compared with torchvision."""

import numpy as np
import pytest
from PIL import Image

from rivet import vision as rv

F = pytest.importorskip("torchvision.transforms.functional")


@pytest.fixture
def source(tmp_path):
    images = []
    for index, label in enumerate([0, 0, 1]):
        folder = tmp_path / str(label)
        folder.mkdir(exist_ok=True)
        pixels = np.random.default_rng(index).integers(
            0, 256, (19, 23, 3), dtype=np.uint8
        )
        image = Image.fromarray(pixels)
        image.save(folder / f"{index}.png")
        images.append(image)
    return tmp_path, images


def crop(target, count, vertical=False, width=9, height=7):
    if count == 5:
        return target.five_crop(width, height)
    return target.ten_crop(width, height, vertical_flip=vertical)


@pytest.mark.parametrize("count,vertical", [(5, False), (10, False), (10, True)])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("representation", ["hwc", "chw", "float_chw"])
def test_views_match_torchvision_order_and_repeat_labels(
    source, count, vertical, nested, workers, representation
):
    root, images = source
    pipeline = rv.scan_image_folder(root).decode_image()
    if representation == "chw":
        pipeline = pipeline.pil_to_tensor()
    elif representation == "float_chw":
        pipeline = pipeline.to_tensor()
    if nested:
        pipeline = pipeline.compose(crop(rv.Transform(), count, vertical))
    else:
        pipeline = crop(pipeline, count, vertical)
    loader = pipeline.workers(workers).batch(2).execute()
    batches = list(loader)
    assert [batch["source_batch_size"] for batch in batches] == [2, 1]
    assert [len(batch["images"]) for batch in batches] == [2 * count, count]
    assert all(batch["views_per_sample"] == count for batch in batches)
    assert all(batch["shape"] == list(batch["images"].shape) for batch in batches)
    expected = []
    for image in images:
        if representation == "chw":
            image = F.pil_to_tensor(image)
        elif representation == "float_chw":
            image = F.to_tensor(image)
        views = (
            F.five_crop(image, [7, 9])
            if count == 5
            else F.ten_crop(image, [7, 9], vertical_flip=vertical)
        )
        expected.extend(np.asarray(view) for view in views)
    actual = np.concatenate([batch["images"] for batch in batches])
    np.testing.assert_array_equal(actual, np.stack(expected))
    np.testing.assert_array_equal(
        np.concatenate([batch["labels"] for batch in batches]),
        np.repeat([0, 0, 1], count),
    )
    assert batches[0]["layout"] == ("NHWC" if representation == "hwc" else "NCHW")
    assert not batches[0]["images"].flags.writeable
    with pytest.raises(StopIteration):
        next(loader)


@pytest.mark.parametrize("count", [5, 10])
def test_post_crop_tensor_conversion_and_normalization(source, count):
    root, images = source
    batches = list(
        crop(rv.scan_image_folder(root).decode_image(), count)
        .to_tensor()
        .normalize([0.1, 0.2, 0.3], [0.5, 0.6, 0.7])
        .batch(2)
        .execute()
    )
    expected = []
    for image in images:
        views = F.five_crop(image, [7, 9]) if count == 5 else F.ten_crop(image, [7, 9])
        expected.extend(
            F.normalize(F.to_tensor(view), [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]).numpy()
            for view in views
        )
    np.testing.assert_allclose(
        np.concatenate([batch["images"] for batch in batches]),
        np.stack(expected),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("workers", [0, 2])
def test_drop_last_counts_source_samples_and_keeps_view_groups(source, workers):
    batches = list(
        rv.scan_image_folder(source[0])
        .decode_image()
        .five_crop(9, 7)
        .workers(workers)
        .batch(2, drop_last=True)
        .execute()
    )
    assert len(batches) == 1
    assert batches[0]["images"].shape == (10, 7, 9, 3)
    assert batches[0]["source_batch_size"] == 2


def test_repeated_crops_multiply_views_in_parent_then_child_order(source):
    batch = next(
        rv.scan_image_folder(source[0])
        .take(1)
        .decode_image()
        .five_crop(11, 9)
        .five_crop(5, 3)
        .batch(1)
        .execute()
    )
    expected = [
        np.asarray(child)
        for parent in F.five_crop(source[1][0], [9, 11])
        for child in F.five_crop(parent, [3, 5])
    ]
    assert batch["views_per_sample"] == 25
    assert batch["source_batch_size"] == 1
    np.testing.assert_array_equal(batch["images"], np.stack(expected))
    np.testing.assert_array_equal(batch["labels"], np.zeros(25, dtype=np.int64))


def test_random_transforms_have_stable_independent_view_streams(source):
    def run(workers, batch_size=2, epoch=0):
        pipeline = (
            rv.scan_image_folder(source[0])
            .decode_image()
            .five_crop(23, 19)
            .random_apply(0.5, rv.Transform().invert())
            .seed(37)
            .epoch(epoch)
            .workers(workers)
            .batch(batch_size)
        )
        return np.concatenate([batch["images"] for batch in pipeline.execute()])

    original = run(0)
    np.testing.assert_array_equal(original, run(2))
    np.testing.assert_array_equal(original, run(4, batch_size=1))
    assert not np.array_equal(original, run(0, epoch=1))
    # Equal full-size crops should still receive different random draws.
    assert any(not np.array_equal(original[i], original[0]) for i in range(1, 5))


@pytest.mark.parametrize("mode", ["bytes", "dlpack", "torch"])
def test_multiview_framework_handoff_and_metadata(source, mode):
    loader = (
        rv.scan_image_folder(source[0])
        .take(1)
        .decode_image()
        .five_crop(9, 7)
        .batch(2)
        .execute(as_numpy=mode != "bytes")
    )
    if mode == "bytes":
        batch = next(loader)
        assert len(batch["images"]) == 5 * 7 * 9 * 3
        assert batch["labels"] == [0] * 5
    else:
        torch = pytest.importorskip("torch")
        batch = loader.next_dlpack() if mode == "dlpack" else loader.next_torch()
        images = (
            torch.from_dlpack(batch["images"]) if mode == "dlpack" else batch["images"]
        )
        assert tuple(images.shape) == (5, 7, 9, 3)
    assert batch["views_per_sample"] == 5
    assert batch["source_batch_size"] == 1


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p.five_crop(0, 7),
        lambda p: p.ten_crop(9, 0),
        lambda p: p.five_crop(24, 7),
        lambda p: p.ten_crop(9, 20),
        lambda p: p.batch(1).five_crop(9, 7),
        lambda p: p.random_apply(1, rv.Transform().five_crop(9, 7)),
        lambda p: p.random_choice([rv.Transform().ten_crop(9, 7)]),
        lambda p: p.random_order(rv.Transform().five_crop(9, 7)),
        lambda p: p.five_crop(9, 7).take(1),
    ],
)
def test_unsupported_or_invalid_multiview_plans_are_rejected(source, build):
    with pytest.raises((ValueError, RuntimeError)):
        list(build(rv.scan_image_folder(source[0]).decode_image()).batch(1).execute())


def test_view_count_overflow_is_rejected_before_loading(source):
    pipeline = rv.scan_image_folder(source[0]).decode_image()
    for _ in range(30):
        pipeline = pipeline.five_crop(1, 1)
    with pytest.raises((ValueError, RuntimeError), match="overflow"):
        pipeline.batch(1).execute()


def test_loader_exhaustion_is_sticky_after_bad_crop(source):
    loader = (
        rv.scan_image_folder(source[0])
        .decode_image()
        .five_crop(24, 7)
        .workers(2)
        .batch(1)
        .execute()
    )
    with pytest.raises((ValueError, RuntimeError)):
        next(loader)
    with pytest.raises(StopIteration):
        next(loader)


def test_explain_is_lazy_and_records_crop(tmp_path):
    folder = tmp_path / "class"
    folder.mkdir()
    (folder / "bad.png").write_bytes(b"not an image")
    explanation = (
        rv.scan_image_folder(tmp_path).decode_image().ten_crop(9, 7).batch(2).explain()
    )
    assert "TenCrop" in explanation


def test_source_selection_and_shuffle_preserve_view_groups(source):
    def base():
        return (
            rv.scan_image_folder(source[0]).shuffle(11).skip(1).take(2).decode_image()
        )

    expected = next(base().batch(2).execute())
    batch = next(base().five_crop(23, 19).batch(2).execute())
    np.testing.assert_array_equal(
        batch["images"], np.repeat(expected["images"], 5, axis=0)
    )
    np.testing.assert_array_equal(batch["labels"], np.repeat(expected["labels"], 5))


def test_augmentation_before_cropping_is_shared_by_all_views(source):
    batch = next(
        rv.scan_image_folder(source[0])
        .decode_image()
        .random_apply(0.5, rv.Transform().invert())
        .seed(37)
        .five_crop(23, 19)
        .batch(3)
        .execute()
    )
    for views in batch["images"].reshape(3, 5, 19, 23, 3):
        np.testing.assert_array_equal(views, np.repeat(views[:1], 5, axis=0))


@pytest.mark.parametrize("workers", [0, 2])
def test_empty_source_finishes_without_partial_view_groups(tmp_path, workers):
    (tmp_path / "class").mkdir()
    loader = (
        rv.scan_image_folder(tmp_path)
        .decode_image()
        .ten_crop(9, 7)
        .workers(workers)
        .batch(2)
        .execute()
    )
    assert list(loader) == []
    with pytest.raises(StopIteration):
        next(loader)


def test_managed_dag_budget_on_view_batches(source):
    def pipeline():
        return (
            rv.scan_image_folder(source[0])
            .decode_image()
            .five_crop(9, 7)
            .workers(2)
            .batch(2)
        )

    expected = list(pipeline().execute())
    # Five views share each decoded allocation. During stacking the two
    # decoded backings and the new image and label storage coexist.
    peak = 2 * 19 * 23 * 3 + 10 * (7 * 9 * 3 + 8)
    actual = list(pipeline().dag_managed_budget_bytes(peak).execute())
    assert len(actual) == len(expected) == 2
    for managed, baseline in zip(actual, expected):
        np.testing.assert_array_equal(managed["images"], baseline["images"])
        np.testing.assert_array_equal(managed["labels"], baseline["labels"])

    loader = pipeline().dag_managed_budget_bytes(peak - 1).execute()
    with pytest.raises(RuntimeError, match="ImageStack.*managed output bytes before allocation"):
        next(loader)


def test_dag_stack_workspace_preserves_exported_labels(source):
    torch = pytest.importorskip("torch")
    loader = (
        rv.scan_image_folder(source[0])
        .decode_image()
        .five_crop(9, 7)
        .workers(0)
        .batch(1)
        .dag_managed_budget_bytes(4096)
        .execute()
    )
    first = loader.next_dlpack()
    labels = torch.utils.dlpack.from_dlpack(first["labels"].into_dlpack())
    labels.fill_(23)
    second = loader.next_dlpack()
    next_labels = torch.utils.dlpack.from_dlpack(second["labels"])
    assert next_labels.tolist() == [0] * 5
    assert labels.tolist() == [23] * 5
    assert next_labels.data_ptr() != labels.data_ptr()
