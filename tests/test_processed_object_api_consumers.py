"""Command-line regression checks for the ProcessedObject registry API.

Fill in PO_PATH and HOLE_PATH, activate the application's environment, and run:

    python test_processed_object_api.py

PO_PATH must point to any one dataset file belonging to a processed box.
HOLE_PATH must point to a directory containing the processed boxes for one hole.

The script does not save datasets or intentionally alter files on disk.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import inspect
from pathlib import Path
import sys
import tempfile
import traceback
from unittest.mock import patch

import numpy as np

from app.models.hole_object import HoleObject
from app.models.dataset import Dataset
from app.models.processed_object import ProcessedObject
from app.create_report.pdf_booklet import (
    create_hole_pdf_booklet,
    create_po_pdf_booklet,
)


HOLE_PATH = "c:/Users/Hyperspectral/Desktop/DemoData/UCD_Course/UCD_Demo_data/Hole"
PO_PATH = "c:/Users/Hyperspectral/Desktop/DemoData/UCD_Course/UCD_Demo_data/ProcessedOutput/00_13-18-088-11w5_00_103_1590m80_1593m60_2022-02-11_03-12-25_DholeAverage.npy"
TEST_KEY = "__API_TEST__"


class SkipTest(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def same_data(left, right) -> bool:
    """Compare arrays, masked arrays, mappings, sequences, and scalars."""
    if np.ma.isMaskedArray(left) or np.ma.isMaskedArray(right):
        if not (np.ma.isMaskedArray(left) and np.ma.isMaskedArray(right)):
            return False
        return np.array_equal(left.data, right.data) and np.array_equal(
            np.ma.getmaskarray(left), np.ma.getmaskarray(right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(left, right, equal_nan=True)
        except TypeError:
            return np.array_equal(left, right)
    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def clone_data(data):
    if np.ma.isMaskedArray(data):
        return np.ma.array(data, copy=True)
    if isinstance(data, np.ndarray):
        return np.array(data, copy=True)
    return deepcopy(data)


def altered_copy(data):
    """Return a distinguishable, type-compatible replacement."""
    result = clone_data(data)

    if isinstance(result, dict):
        result[TEST_KEY] = "temporary replacement"
        return result

    if np.ma.isMaskedArray(result):
        if result.size == 0:
            raise SkipTest("candidate masked array is empty")
        flat = result.data.reshape(-1)
        flat[0] = flat[0] + 1
        return result

    if isinstance(result, np.ndarray):
        if result.size == 0:
            raise SkipTest("candidate array is empty")
        flat = result.reshape(-1)
        if np.issubdtype(result.dtype, np.bool_):
            flat[0] = ~flat[0]
        elif np.issubdtype(result.dtype, np.number):
            flat[0] = flat[0] + 1
        else:
            raise SkipTest(f"cannot safely alter dtype {result.dtype}")
        return result

    if isinstance(result, list):
        result.append(TEST_KEY)
        return result

    raise SkipTest(f"cannot make a safe replacement for {type(data).__name__}")


def load_po() -> ProcessedObject:
    return ProcessedObject.from_path(Path(PO_PATH))


def load_hole() -> HoleObject:
    return HoleObject.build_from_parent_dir(Path(HOLE_PATH))


def move_permanent_to_temp(po: ProcessedObject, keys=None) -> list[str]:
    """Make datasets temp-only without copying data or building thumbnails."""
    selected = list(po.datasets) if keys is None else list(keys)
    moved = []
    for key in selected:
        if key in po.datasets:
            po.temp_datasets[key] = po.datasets.pop(key)
            moved.append(key)
    return moved


@contextmanager
def suppress_integration_writes():
    """Keep HoleObject generation in memory and preserve temp-only fixtures."""
    with (
        patch.object(ProcessedObject, "save_all", autospec=True, return_value=None),
        patch.object(ProcessedObject, "reload_all", autospec=True, return_value=None),
        patch.object(Dataset, "save_dataset", autospec=True, return_value=None),
    ):
        yield


def common_keys(hole: HoleObject) -> set[str]:
    boxes = list(hole)
    if not boxes:
        return set()
    result = set(boxes[0].keys())
    for box in boxes[1:]:
        result &= set(box.keys())
    return result


def require_common(hole: HoleObject, keys) -> None:
    missing = [key for key in keys if not hole.check_for_all_keys(key)]
    if missing:
        raise SkipTest(f"hole is missing required keys: {', '.join(missing)}")


def assert_pdf(path: Path) -> None:
    check(path.is_file(), f"PDF was not created: {path}")
    check(path.stat().st_size > 1_000, f"PDF is unexpectedly small: {path.stat().st_size} bytes")
    with path.open("rb") as stream:
        check(stream.read(5) == b"%PDF-", "output does not have a PDF header")


def choose_small_key(po: ProcessedObject) -> str:
    """Prefer metadata so a registry test does not copy a spectral cube."""
    if po.has("metadata"):
        return "metadata"

    for key in po.keys():
        data = po.get_data(key)
        if isinstance(data, (dict, list)):
            return key
        if isinstance(data, np.ndarray) and data.size <= 100_000:
            return key

    raise SkipTest("no suitably small existing dataset was found")


def assert_absent(po: ProcessedObject, key: str) -> None:
    check(not po.has(key), f"has({key!r}) should be False")
    check(key not in po.keys(), f"{key!r} should not appear in keys()")

    try:
        po[key]
    except KeyError:
        pass
    else:
        raise AssertionError("__getitem__ did not raise KeyError for a missing key")

    try:
        po.get_data(key)
    except KeyError:
        pass
    else:
        raise AssertionError("get_data() did not raise KeyError for a missing key")

    try:
        getattr(po, key)
    except AttributeError:
        pass
    else:
        raise AssertionError("attribute access did not raise AttributeError for a missing key")


def test_permanent_contract() -> None:
    po = load_po()
    check(bool(po.datasets), "loaded PO has no permanent datasets")
    check(not po.has_temps, "freshly loaded PO unexpectedly has temps")

    for key, dataset in po.datasets.items():
        check(po.has(key), f"permanent key {key!r} is not reported by has()")
        check(key in po.keys(), f"permanent key {key!r} is absent from keys()")
        check(po[key] is dataset, f"__getitem__ did not return permanent {key!r}")
        check(po.get_data(key) is dataset.data, f"get_data did not return {key!r} data")
        check(getattr(po, key) is dataset.data, f"attribute access disagrees for {key!r}")

    check(po.keys() == sorted(po.keys()), "keys() is not sorted")


def test_missing_contract() -> None:
    po = load_po()
    assert_absent(po, TEST_KEY)


def test_new_temp_and_clear() -> None:
    po = load_po()
    data = {"purpose": "API test", "values": [1, 2, 3]}
    po.add_temp_dataset(TEST_KEY, data, ext=".json")

    check(TEST_KEY not in po.datasets, "new temp leaked into permanent registry")
    check(po.has_temp(TEST_KEY), "has_temp() missed a new temp")
    check(po.has(TEST_KEY), "has() missed a new temp")
    check(TEST_KEY in po.keys(), "keys() missed a new temp")
    check(po[TEST_KEY] is po.temp_datasets[TEST_KEY], "temp-first lookup failed")
    check(po[TEST_KEY].ext == ".json", "new temp did not retain its extension")
    check(same_data(po.get_data(TEST_KEY), data), "new temp data changed")
    check(getattr(po, TEST_KEY) is po.get_data(TEST_KEY), "temp attribute access disagrees")
    check(po.has_temps, "has_temps should be True")

    po.clear_temps()
    check(not po.has_temps, "clear_temps() left temporary datasets")
    assert_absent(po, TEST_KEY)


def test_replacement_and_clear() -> None:
    po = load_po()
    key = choose_small_key(po)
    permanent = po.datasets[key]
    original = clone_data(permanent.data)
    replacement = altered_copy(permanent.data)

    po.add_temp_dataset(key, replacement, ext=permanent.ext)

    check(po.has_temp(key), "replacement was not staged as temp")
    check(po[key] is po.temp_datasets[key], "replacement did not win lookup")
    check(po[key] is not permanent, "replacement reused the permanent Dataset object")
    check(po[key].ext == permanent.ext, "replacement extension changed")
    check(same_data(po.get_data(key), replacement), "active replacement data is wrong")
    check(same_data(po.datasets[key].data, original), "permanent data was mutated")

    po.clear_temps()
    check(not po.has_temp(key), "clear_temps() retained the replacement")
    check(po[key] is po.datasets[key], "permanent lookup was not restored")
    check(same_data(po.get_data(key), original), "original data was not restored")


def test_commit_new_temp_in_memory() -> None:
    po = load_po()
    data = np.arange(12, dtype=np.int16).reshape(3, 4)
    po.add_temp_dataset(TEST_KEY, data, ext=".npy")
    temp = po.temp_datasets[TEST_KEY]

    po.commit_temps()

    check(not po.has_temps, "commit_temps() did not clear temp registry")
    check(TEST_KEY in po.datasets, "committed new temp is not permanent in memory")
    check(po[TEST_KEY] is temp, "commit_temps() replaced the temp Dataset object")
    check(same_data(po.get_data(TEST_KEY), data), "committed temp data changed")
    check(not temp.path.exists(), "test unexpectedly wrote its synthetic dataset")


def test_commit_replacement_in_memory() -> None:
    po = load_po()
    key = choose_small_key(po)
    old_dataset = po.datasets[key]
    replacement = altered_copy(old_dataset.data)
    source_path = old_dataset.path

    po.add_temp_dataset(key, replacement, ext=old_dataset.ext)
    temp = po.temp_datasets[key]
    po.commit_temps()

    check(not po.has_temps, "commit_temps() did not clear replacement temp")
    check(po.datasets[key] is temp, "replacement temp was not promoted")
    check(po.datasets[key] is not old_dataset, "old Dataset object remains active")
    check(po.datasets[key].path == source_path, "replacement path changed")
    check(same_data(po.get_data(key), replacement), "promoted replacement data is wrong")

    # Prove that an in-memory commit did not overwrite the source file.
    reopened = load_po()
    check(not same_data(reopened.get_data(key), replacement),
          "source file appears to have been overwritten without save_all()")


def test_hole_contract() -> None:
    hole = load_hole()
    boxes = list(hole)
    check(boxes, "hole contains no boxes")
    check(len(boxes) == len(hole), "HoleObject iteration and length disagree")

    for box in boxes:
        check(isinstance(box, ProcessedObject), "hole yielded a non-ProcessedObject")

    common = set(boxes[0].keys())
    union = set(boxes[0].keys())
    for box in boxes[1:]:
        common &= set(box.keys())
        union |= set(box.keys())

    for key in common:
        check(hole.check_for_all_keys(key), f"common key {key!r} was rejected")

    check(not hole.check_for_all_keys(TEST_KEY), "missing test key was reported in every box")

    # Temp in one box only must not satisfy an all-box requirement.
    boxes[0].add_temp_dataset(TEST_KEY, {"box": 0}, ext=".json")
    expected = len(boxes) == 1
    check(hole.check_for_all_keys(TEST_KEY) is expected,
          "check_for_all_keys() mishandled a temp present in only one box")
    boxes[0].clear_temps()

    # Temp-only in all boxes must count as logically present everywhere.
    for number, box in enumerate(boxes):
        box.add_temp_dataset(TEST_KEY, {"box": number}, ext=".json")
    check(hole.check_for_all_keys(TEST_KEY), "temp-only key in every box was rejected")
    for box in boxes:
        check(box.has(TEST_KEY), "box public API missed its temp-only key")
        box.clear_temps()

    # A temp replacement of a common permanent key must remain available.
    if common:
        preferred = "metadata" if "metadata" in common else sorted(common)[0]
        target = boxes[0]
        original_ds = target[preferred]
        replacement = clone_data(original_ds.data)
        target.add_temp_dataset(preferred, replacement, ext=original_ds.ext)
        check(hole.check_for_all_keys(preferred),
              "temp-over-permanent key failed the all-box check")
        check(target[preferred] is target.temp_datasets[preferred],
              "hole box did not resolve replacement temp first")
        target.clear_temps()

    check(all(not box.has_temps for box in boxes), "hole test left temporary datasets")


def test_hole_create_base_temp_only() -> None:
    """Run base generation when every PO dataset is available only as a temp."""
    hole = load_hole()
    require_common(hole, ("metadata", "mask", "savgol", "stats", "segments"))

    for po in hole:
        move_permanent_to_temp(po)

    with suppress_integration_writes():
        result = hole.create_base_datasets()

    check(result is hole, "create_base_datasets() did not return the hole")
    check("depths" in hole.base_datasets, "base generation did not create depths")
    check("AvSpectra" in hole.base_datasets, "base generation did not create AvSpectra")
    depths = hole.base_datasets["depths"].data
    spectra = hole.base_datasets["AvSpectra"].data
    check(len(depths) > 0, "generated depths are empty")
    check(spectra.shape[0] == len(depths), "depth and averaged-spectrum row counts disagree")


def choose_index_family(hole: HoleObject) -> tuple[str, str]:
    common = common_keys(hole)
    for index_key in sorted(key for key in common if key.endswith("INDEX")):
        legend_key = index_key.removesuffix("INDEX") + "LEGEND"
        if legend_key in common:
            return index_key, legend_key
    raise SkipTest("no common INDEX/LEGEND family is present in every box")


def test_hole_create_minmap_temp_only() -> None:
    """Run mineral-map generation with all box inputs temp-only."""
    hole = load_hole()
    require_common(hole, ("metadata", "mask", "stats", "segments"))
    index_key, legend_key = choose_index_family(hole)

    for po in hole:
        move_permanent_to_temp(po)

    with suppress_integration_writes():
        hole.create_dhole_minmap(index_key)

    fractions_key = index_key.removesuffix("INDEX") + "FRACTIONS"
    dominant_key = index_key.removesuffix("INDEX") + "DOM-MIN"
    for key in (fractions_key, dominant_key, legend_key):
        check(key in hole.product_datasets, f"mineral-map generation did not create {key!r}")
    check(
        hole.product_datasets[fractions_key].data.shape[0]
        == hole.product_datasets[dominant_key].data.shape[0],
        "mineral fractions and dominant-mineral row counts disagree",
    )


def choose_feature_key(hole: HoleObject) -> str:
    """Find a common spatial masked-array product accepted by create_dhole_features."""
    for key in sorted(common_keys(hole)):
        if key in {"stats", "segments"} or key.endswith(("LEGEND", "CLUSTERS")):
            continue
        valid = True
        for po in hole:
            ds = po[key]
            data = ds.data
            mask = po.mask
            if (
                ds.ext != ".npz"
                or not np.ma.isMaskedArray(data)
                or data.ndim != 2
                or data.shape != mask.shape
            ):
                valid = False
                break
        if valid:
            return key
    raise SkipTest("no common 2-D masked-array feature product is present in every box")


def test_hole_create_feature_temp_only() -> None:
    """Run feature aggregation with all box inputs temp-only."""
    hole = load_hole()
    require_common(hole, ("metadata", "stats", "segments"))
    key = choose_feature_key(hole)

    for po in hole:
        move_permanent_to_temp(po)

    with suppress_integration_writes():
        hole.create_dhole_features(key)

    check(key in hole.product_datasets, f"feature generation did not create {key!r}")
    result = hole.product_datasets[key]
    check(result.ext == ".npz", "generated feature product did not retain masked-array extension")
    check(np.ma.isMaskedArray(result.data), "generated feature product is not a masked array")
    check(result.data.size > 0, "generated feature product is empty")



def choose_pdf_key(hole: HoleObject) -> str:
    """Find a common dataset that can be rendered from active data."""
    common = common_keys(hole)
    preferred = ["display", "mask"]
    candidates = preferred + sorted(common - set(preferred))

    for key in candidates:
        if key not in common:
            continue

        if key.endswith(("LEGEND", "CLUSTERS")):
            continue

        try:
            if all(
                po.get_pil_image(key, resize=False) is not None
                for po in hole
            ):
                return key
        except Exception:
            continue

    raise SkipTest("no common image-renderable dataset is available")



def test_pdf_booklets_temp_only() -> None:
    """Generate both booklet types when every source dataset is temp-only."""
    hole = load_hole()
    selected_key = choose_pdf_key(hole)

    for po in hole:
        move_permanent_to_temp(po)

        # The hole overview consumes cached thumbnails.
        po.build_thumb(selected_key)

        check(
            po.has_temp(selected_key),
            f"{selected_key!r} is not temp-only in {po.basename}",
        )
        check(
            po.temp_datasets[selected_key].thumb is not None,
            f"could not build in-memory thumbnail for "
            f"{selected_key!r} in {po.basename}",
        )

    with tempfile.TemporaryDirectory(prefix="csv_api_pdf_") as directory:
        output_dir = Path(directory)

        hole_pdf = create_hole_pdf_booklet(
            hole,
            [selected_key],
            output_dir / "hole_api_test.pdf",
            boxes_per_page=3,
            include_downhole_plots=False,
        )
        assert_pdf(hole_pdf)

        po = load_po()
        move_permanent_to_temp(po)

        # Prove at least one temp-only dataset is renderable. Otherwise a
        # title-only PDF could falsely pass.
        renderable = [
            key
            for key in po.keys()
            if not key.endswith(("LEGEND", "CLUSTERS"))
            and po.get_pil_image(key, resize=False) is not None
        ]
        check(
            renderable,
            "single PO has no temp-only renderable dataset",
        )
        check(
            all(po.has_temp(key) for key in renderable),
            "single-PO renderable inputs are not temp-only",
        )

        po_pdf = create_po_pdf_booklet(po, output_dir)
        assert_pdf(po_pdf)


def null_api_available() -> bool:
    return "null" in inspect.signature(ProcessedObject.add_temp_dataset).parameters


def test_null_temp_only_contract() -> None:
    """Exercise tombstone filtering without targeting any real disk dataset."""
    if not null_api_available():
        raise SkipTest("add_temp_dataset() has no null parameter")

    source = load_po()
    po = ProcessedObject.new(source.root_dir, source.basename + "_API_TEST")
    po.add_temp_dataset(TEST_KEY, np.arange(4), ext=".npy")
    check(po.has(TEST_KEY), "synthetic temp setup failed")

    po.add_temp_dataset(TEST_KEY, null=True)
    check(po.has_temp(TEST_KEY), "null temp is not tracked as a transaction")
    check(po.has_temps, "null temp is not included by has_temps")
    check(not po.has(TEST_KEY), "public has() exposed a null temp")
    check(TEST_KEY not in po.keys(), "public keys() exposed a null temp")

    try:
        po[TEST_KEY]
    except KeyError:
        pass
    else:
        raise AssertionError("__getitem__ exposed a null temp")

    po.clear_temps()
    check(not po.has_temps, "clear_temps() retained a null temp")
    check(not po.has(TEST_KEY), "clearing temp-only tombstone created a dataset")


TESTS = [
    ("permanent public contract", test_permanent_contract),
    ("missing-key contract", test_missing_contract),
    ("new temp and clear", test_new_temp_and_clear),
    ("replacement and clear", test_replacement_and_clear),
    ("commit new temp in memory", test_commit_new_temp_in_memory),
    ("commit replacement in memory", test_commit_replacement_in_memory),
    ("hole mixed-registry contract", test_hole_contract),
    ("hole create base from temp-only inputs", test_hole_create_base_temp_only),
    ("hole create mineral map from temp-only inputs", test_hole_create_minmap_temp_only),
    ("hole create feature from temp-only inputs", test_hole_create_feature_temp_only),
    ("PDF booklets from temp-only inputs", test_pdf_booklets_temp_only),
    ("null/tombstone public contract", test_null_temp_only_contract),
]


def validate_paths() -> None:
    if not PO_PATH:
        raise ValueError('Set PO_PATH at the top of the script, e.g. r"C:\\data\\box_metadata.json"')
    if not HOLE_PATH:
        raise ValueError('Set HOLE_PATH at the top of the script, e.g. r"C:\\data\\hole"')

    po_path = Path(PO_PATH)
    hole_path = Path(HOLE_PATH)
    check(po_path.is_file(), f"PO_PATH is not a file: {po_path}")
    check(hole_path.is_dir(), f"HOLE_PATH is not a directory: {hole_path}")


def main() -> int:
    validate_paths()

    print(f"PO_PATH:   {Path(PO_PATH)}")
    print(f"HOLE_PATH: {Path(HOLE_PATH)}")
    print("Source-data writes: disabled; generated PDFs use a temporary directory\n")

    passed = 0
    skipped = 0
    failed = 0

    for name, test in TESTS:
        try:
            test()
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP  {name}: {exc}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS  {name}")

    print(f"\nResult: {passed} passed, {skipped} skipped, {failed} failed")
    if not null_api_available():
        print("Note: tombstone checks were skipped because this build has no null API.")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
