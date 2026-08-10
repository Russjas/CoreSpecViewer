"""Regression tests for ProcessedObject API use in app.interface.tools.

Set PO_PATH to any dataset file belonging to a processed box, activate the
application environment, and run from the project root:

    python test_tools_api_consumers.py

The tests make no source-data writes. Each test loads a fresh ProcessedObject,
and all changes remain in its in-memory temp registry.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np

from app.interface import tools
from app.models.processed_object import ProcessedObject


PO_PATH = "c:/Users/Hyperspectral/Desktop/DemoData/UCD_Course/UCD_Demo_data/ProcessedOutput/00_13-18-088-11w5_00_103_1590m80_1593m60_2022-02-11_03-12-25_DholeAverage.npy"


class SkipTest(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def clone(data):
    if np.ma.isMaskedArray(data):
        return np.ma.array(data, copy=True)
    if isinstance(data, np.ndarray):
        return np.array(data, copy=True)
    return deepcopy(data)


def equal(left, right) -> bool:
    if np.ma.isMaskedArray(left) or np.ma.isMaskedArray(right):
        return (
            np.ma.isMaskedArray(left)
            and np.ma.isMaskedArray(right)
            and np.array_equal(left.data, right.data, equal_nan=True)
            and np.array_equal(
                np.ma.getmaskarray(left), np.ma.getmaskarray(right)
            )
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(left, right, equal_nan=True)
        except TypeError:
            return np.array_equal(left, right)
    return left == right


def load_po() -> ProcessedObject:
    return ProcessedObject.from_path(Path(PO_PATH))


def spatial_bounds(po: ProcessedObject):
    shape = po.savgol.shape[:2]
    if min(shape) < 3:
        raise SkipTest(f"savgol is too small to crop: {shape}")
    # Remove one row/column where possible, while remaining valid for tiny data.
    y1 = max(1, shape[0] - 1)
    x1 = max(1, shape[1] - 1)
    return (slice(1 if shape[0] > 2 else 0, y1),
            slice(1 if shape[1] > 2 else 0, x1))


def distinctive_mask(mask):
    result = np.array(mask, copy=True)
    if result.size == 0:
        raise SkipTest("mask is empty")
    result.flat[0] = 0 if bool(result.flat[0]) else 1
    return result


def test_crop_temp_first() -> None:
    """Manual crop uses active data and preserves spatial/non-spatial rules."""
    po = load_po()
    check(po.has("savgol") and po.has("mask"), "PO needs savgol and mask")

    base_shape = po.savgol.shape[:2]
    ys, xs = spatial_bounds(po)
    staged_mask = distinctive_mask(po.mask)
    permanent_mask = clone(po.datasets["mask"].data)
    mask_ext = po["mask"].ext
    bands_before = clone(po.bands)

    extra = np.arange(base_shape[0] * base_shape[1], dtype=np.int32).reshape(
        base_shape
    )

    with patch.object(po, "build_thumb", return_value=None):
        po.add_temp_dataset("mask", staged_mask, ext=mask_ext)
        po.add_temp_dataset("TOOLS_SPATIAL", extra, ext=".npz")
        po.add_temp_dataset("TOOLS_VECTOR", np.arange(7), ext=".json")
        tools.crop(po, ys.start, ys.stop, xs.start, xs.stop)

    expected_shape = staged_mask[ys, xs].shape
    check(po.has_temp("mask"), "crop did not stage mask")
    check(equal(po.mask, staged_mask[ys, xs]), "crop read permanent mask")
    check(equal(po.datasets["mask"].data, permanent_mask),
          "crop mutated permanent mask")
    check(po["mask"].ext == mask_ext, "crop changed mask extension")
    check(equal(po.get_data("TOOLS_SPATIAL"), extra[ys, xs]),
          "crop did not use temp-only spatial input")
    check(po["TOOLS_SPATIAL"].ext == ".npz",
          "crop did not preserve temp-only extension")
    check(equal(po.get_data("TOOLS_VECTOR"), np.arange(7)),
          "crop altered a non-spatial vector")
    check(po["TOOLS_VECTOR"].ext == ".json",
          "non-spatial extension changed")
    check(equal(po.bands, bands_before), "crop altered bands")
    check(po.has_temp("display"), "crop did not regenerate display")
    check(po.display.shape[:2] == expected_shape,
          "regenerated display has the wrong dimensions")


def test_crop_auto_temp_first() -> None:
    """Automatic crop uses active display/data and applies its known slicer."""
    po = load_po()
    check(po.has("savgol") and po.has("mask"), "PO needs savgol and mask")
    ys, xs = spatial_bounds(po)
    slicer = (ys, xs)
    staged_mask = distinctive_mask(po.mask)
    permanent_mask = clone(po.datasets["mask"].data)

    if po.has("display"):
        active_display = np.array(po.display, copy=True)
    else:
        h, w = po.savgol.shape[:2]
        active_display = np.zeros((h, w, 3), dtype=np.uint8)
    active_display.flat[0] = (int(active_display.flat[0]) + 1) % 255
    seen = {}

    def fake_auto_crop(image, mode="references"):
        seen["image"] = image
        seen["mode"] = mode
        return image[slicer], slicer

    with (
        patch.object(po, "build_thumb", return_value=None),
        patch.object(tools.sm, "auto_crop", side_effect=fake_auto_crop),
    ):
        po.add_temp_dataset("display", active_display, ext=".npy")
        po.add_temp_dataset("mask", staged_mask, ext=po["mask"].ext)
        staged_display = po.display
        tools.crop_auto(po, mode="references")

    check(seen.get("image") is staged_display,
          "crop_auto did not pass the active temp display to auto_crop")
    check(seen.get("mode") == "references", "crop_auto changed its mode")
    check(equal(po.mask, staged_mask[slicer]),
          "crop_auto read the permanent mask")
    check(equal(po.datasets["mask"].data, permanent_mask),
          "crop_auto mutated permanent mask")
    check(po.has_temp("display"), "crop_auto did not regenerate display")
    check(po.display.shape[:2] == staged_mask[slicer].shape,
          "crop_auto display has the wrong dimensions")


def test_crop_auto_invalid_slicer() -> None:
    """A failed auto-crop returns without staging or changing anything."""
    po = load_po()
    before_keys = set(po.keys())
    before_mask = clone(po.mask)

    with patch.object(tools.sm, "auto_crop", return_value=(None, None)):
        returned = tools.crop_auto(po)

    check(returned is po, "crop_auto did not return the original PO")
    check(not po.has_temps, "invalid crop staged datasets")
    check(set(po.keys()) == before_keys, "invalid crop changed keys")
    check(equal(po.mask, before_mask), "invalid crop changed active data")


def test_calc_unwrap_stats_temp_first() -> None:
    """Stats consume active mask and metadata changes remain staged."""
    po = load_po()
    check(po.has("mask") and po.has("metadata"), "PO needs mask and metadata")
    staged_mask = distinctive_mask(po.mask)
    permanent_metadata = deepcopy(po.datasets["metadata"].data)
    label_image = np.full(staged_mask.shape, 7, dtype=np.int32)
    stats = np.array([[0, 0, 2, 2, 4]], dtype=np.int32)
    seen = {}

    def fake_stats(mask):
        seen["mask"] = mask
        return label_image, stats

    with (
        patch.object(po, "build_thumb", return_value=None),
        patch.object(tools.sm, "get_stats_from_mask", side_effect=fake_stats),
    ):
        po.add_temp_dataset("mask", staged_mask, ext=po["mask"].ext)
        active_mask = po.mask
        tools.calc_unwrap_stats(po)

    check(seen.get("mask") is active_mask,
          "calc_unwrap_stats did not consume the active temp mask")
    check(equal(po.stats, stats), "wrong staged stats")
    check(equal(po.segments, label_image), "wrong staged segments")
    check(po["stats"].ext == ".npy" and po["segments"].ext == ".npy",
          "stats/segments extensions are wrong")
    check(po.has_temp("metadata"), "metadata was not staged")
    check(po.metadata["box_convention"] == tools.config.box_convention,
          "box convention was not embedded")
    check(po["metadata"].ext == ".json", "metadata extension is wrong")
    check(equal(po.datasets["metadata"].data, permanent_metadata),
          "permanent metadata was mutated")


def test_feature_extraction_temp_cache() -> None:
    """Feature extraction consumes complete temp cache replacements."""
    po = load_po()
    shape = po.savgol.shape[:2]
    cached_indices = np.full((*shape, 2), 13, dtype=np.int32)
    cached_heights = np.full((*shape, 2), 0.37, dtype=np.float32)
    pos = np.full(shape, 2205.0, dtype=np.float32)
    dep = np.full(shape, 0.18, dtype=np.float32)
    feat_mask = np.asarray(po.mask, dtype=bool)
    seen = {}

    def fake_combined(savgol, savgol_cr, mask, bands, key, technique,
                      cached_arrays):
        seen["cached"] = cached_arrays
        seen["key"] = key
        seen["technique"] = technique
        return pos, dep, feat_mask

    with (
        patch.object(po, "build_thumb", return_value=None),
        patch.object(tools.sa, "Combined_MWL", side_effect=fake_combined),
    ):
        po.add_temp_dataset("feature-indices", cached_indices, ext=".npy")
        po.add_temp_dataset("feature-heights", cached_heights, ext=".npy")
        active_indices = po.get_data("feature-indices")
        active_heights = po.get_data("feature-heights")
        tools.run_feature_extraction(po, "2200W")

    check(seen["cached"][0] is active_indices,
          "feature extraction did not use active temp feature-indices")
    check(seen["cached"][1] is active_heights,
          "feature extraction did not use active temp feature-heights")
    check(seen["key"] == "2200W" and seen["technique"] == "POLY",
          "feature extraction forwarded the wrong arguments")
    check(po.has_temp("2200WPOS") and po.has_temp("2200WDEP"),
          "feature outputs were not staged")
    check(po["2200WPOS"].ext == ".npz" and po["2200WDEP"].ext == ".npz",
          "feature output extensions are wrong")
    check(np.ma.isMaskedArray(po.get_data("2200WPOS")),
          "POS output is not a masked array")
    check(equal(po.get_data("2200WPOS").data, pos), "wrong POS data")
    check(equal(po.get_data("2200WDEP").data, dep), "wrong DEP data")
    check(equal(np.ma.getmaskarray(po.get_data("2200WPOS")), feat_mask),
          "wrong feature mask")


def test_feature_extraction_partial_cache() -> None:
    """One missing cache member causes recomputation, not partial cache use."""
    po = load_po()
    shape = po.savgol.shape[:2]
    pos = np.zeros(shape, dtype=np.float32)
    dep = np.zeros(shape, dtype=np.float32)
    feat_mask = np.asarray(po.mask, dtype=bool)
    seen = {}

    def fake_combined(*args, cached_arrays=None, **kwargs):
        seen["cached"] = cached_arrays
        return pos, dep, feat_mask

    # Remove any permanent cache pair from logical availability in this fresh
    # in-memory fixture, then add only one member as temp-only.
    po.datasets.pop("feature-indices", None)
    po.datasets.pop("feature-heights", None)
    with (
        patch.object(po, "build_thumb", return_value=None),
        patch.object(tools.sa, "Combined_MWL", side_effect=fake_combined),
    ):
        po.add_temp_dataset(
            "feature-indices", np.zeros((*shape, 1), dtype=np.int32)
        )
        tools.run_feature_extraction(po, "2200W")

    check(seen.get("cached", "missing") is None,
          "partial cache was passed to Combined_MWL")


TESTS = [
    ("manual crop from active temp inputs", test_crop_temp_first),
    ("automatic crop from active temp inputs", test_crop_auto_temp_first),
    ("automatic crop invalid-slicer no-op", test_crop_auto_invalid_slicer),
    ("unwrap stats and metadata staging", test_calc_unwrap_stats_temp_first),
    ("feature extraction from temp cache", test_feature_extraction_temp_cache),
    ("feature extraction partial-cache fallback", test_feature_extraction_partial_cache),
]


def validate_path() -> None:
    if not PO_PATH:
        raise ValueError("Set PO_PATH at the top of this script")
    path = Path(PO_PATH)
    if not path.is_file():
        raise ValueError(f"PO_PATH is not a file: {path}")


def main() -> int:
    try:
        validate_path()
    except Exception as exc:
        print(f"SETUP ERROR: {exc}")
        return 2

    print(f"PO_PATH: {PO_PATH}")
    print("Source-data writes: disabled by test design")

    passed = skipped = failed = 0
    for name, test in TESTS:
        try:
            test()
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP  {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS  {name}")

    print(f"Result: {passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
