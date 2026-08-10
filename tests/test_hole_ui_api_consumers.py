"""Regression tests for ProcessedObject API use by the Hole UI.

Set HOLE_PATH to a directory accepted by HoleObject.build_from_parent_dir(),
activate the application environment, and run from the project root:

    python test_hole_ui_api_consumers.py

Qt runs offscreen. The tests do not show dialogs, generate products, or write
source data. Synthetic datasets exist only in each box's temp registry.
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app.models.context import CurrentContext
from app.models.hole_object import HoleObject
from app.models.processed_object import ProcessedObject
from app.ui.hole_page import HoleBoxTable, HolePage
from app.ui.display_text import gen_display_text


HOLE_PATH = "c:/Users/Hyperspectral/Desktop/DemoData/UCD_Course/UCD_Demo_data/Hole"

TEMP_PRODUCT = "UI_API_PRODUCT"
PARTIAL_PRODUCT = "UI_API_PARTIAL"
FEATURE_KEY = "UI_API_FEATURE_POS"
INDEX_KEY = "UI_API_MINMAP_INDEX"
LEGEND_KEY = "UI_API_MINMAP_LEGEND"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def combo_data(combo) -> set[str]:
    """Return selectable raw keys, excluding heading rows."""
    return {
        value
        for i in range(combo.count())
        if (value := combo.itemData(i, Qt.UserRole)) is not None
    }


def load_hole() -> HoleObject:
    return HoleObject.build_from_parent_dir(Path(HOLE_PATH))


def stage_ui_fixtures(hole: HoleObject) -> None:
    """Create temp-only keys that permanent-registry enumeration cannot see."""
    boxes = list(hole)
    check(boxes, "hole contains no boxes")

    image = np.arange(20, dtype=np.uint8).reshape(4, 5)
    feature = np.ma.array(image.astype(float), mask=np.zeros_like(image, bool))
    index = np.zeros((4, 5), dtype=np.uint8)
    legend = {0: ["UI test", [255, 0, 0]]}

    with patch.object(ProcessedObject, "build_thumb", return_value=None):
        for po in boxes:
            po.add_temp_dataset(TEMP_PRODUCT, image, ext=".npy")
            po.add_temp_dataset(FEATURE_KEY, feature, ext=".npz")
            po.add_temp_dataset(INDEX_KEY, index, ext=".npy")
            po.add_temp_dataset(LEGEND_KEY, legend, ext=".json")

        # Union-only fixture: present in one box, absent from all others.
        boxes[0].add_temp_dataset(PARTIAL_PRODUCT, image, ext=".npy")

    for po in boxes:
        for key in (TEMP_PRODUCT, FEATURE_KEY, INDEX_KEY, LEGEND_KEY):
            check(po.has_temp(key), f"{key} was not staged temp-only")
            check(key not in po.datasets, f"{key} unexpectedly exists permanently")
    check(PARTIAL_PRODUCT not in boxes[0].datasets,
          "partial fixture unexpectedly exists permanently")


def make_page(hole: HoleObject) -> HolePage:
    cxt = CurrentContext()
    cxt.ho = hole
    # Avoid image work while constructing/refeshing the real widgets. Thumbnail
    # resolution has its own focused test below.
    with patch.object(HoleBoxTable, "_get_thumb_pixmap", autospec=True):
        page = HolePage()
        page.cxt = cxt
        page._refresh_from_hole()
    QApplication.processEvents()
    return page


def capture_dialog_items(callable_) -> list[str]:
    """Run a selector handler, capture its offered labels, then cancel it."""
    captured = {}

    def fake_get_item(parent, title, label, items, current=0, editable=False):
        captured["title"] = title
        captured["items"] = list(items)
        return "", False

    with patch("app.ui.hole_page.QInputDialog.getItem", side_effect=fake_get_item):
        callable_()
    check("items" in captured, "selector did not open its item dialogue")
    return captured["items"]


def test_control_panel_receives_active_union(page: HolePage) -> None:
    """_set_dataset_keys includes temp-only and partially available products."""
    panel = page._control_panel
    panel._set_dataset_keys()
    received = combo_data(panel.secondary_combo)
    for key in (TEMP_PRODUCT, PARTIAL_PRODUCT, FEATURE_KEY, INDEX_KEY):
        check(key in received, f"secondary selector did not receive {key}")
    check(LEGEND_KEY not in received,
          "non-visual LEGEND was incorrectly offered for display")


def test_refresh_receives_active_union(page: HolePage, hole: HoleObject) -> None:
    """_refresh_from_hole sends active union keys to every strip header."""
    page._refresh_from_hole()
    expected_rows = len(hole.boxes)
    check(page._box_table.rowCount() == expected_rows,
          "default box table has the wrong row count")
    check(page.extra_columns, "HolePage did not create its extra strip")

    for table_name, table in [
        ("default", page._box_table),
        ("extra", page.extra_columns[0]),
    ]:
        check(table._header_combo is not None,
              f"{table_name} strip has no dataset header combo")
        received = combo_data(table._header_combo)
        for key in (TEMP_PRODUCT, PARTIAL_PRODUCT, FEATURE_KEY, INDEX_KEY):
            check(key in received,
                  f"{table_name} strip header did not receive {key}")
        check(LEGEND_KEY not in received,
              f"{table_name} strip offered non-visual LEGEND")


def test_overview_selector_receives_active_union(page: HolePage) -> None:
    """export_overview_image offers eligible temp-only union products."""
    items = capture_dialog_items(page._control_panel.export_overview_image)
    expected = {gen_display_text(TEMP_PRODUCT), gen_display_text(PARTIAL_PRODUCT)}
    check(expected.issubset(set(items)),
          f"overview selector is missing {sorted(expected - set(items))}")
    check(gen_display_text(LEGEND_KEY) not in items,
          "overview selector offered a LEGEND")


def test_feature_selector_receives_temp_only_key(page: HolePage) -> None:
    """dhole_feats_create offers a feature that exists only as temp data."""
    items = capture_dialog_items(page._control_panel.dhole_feats_create)
    check(gen_display_text(FEATURE_KEY) in items,
          "feature selector did not receive the temp-only POS product")


def test_minmap_selector_receives_temp_only_family(page: HolePage) -> None:
    """dhole_minmaps_create pairs temp-only INDEX and LEGEND products."""
    items = capture_dialog_items(page._control_panel.dhole_minmaps_create)
    check(gen_display_text(INDEX_KEY) in items,
          "mineral-map selector did not receive the temp-only INDEX/LEGEND family")


def test_thumbnail_prefers_temp_replacement(page: HolePage, hole: HoleObject) -> None:
    """HoleBoxTable renders the active temp thumbnail over the permanent one."""
    po = next(iter(hole))
    candidates = [key for key in po.datasets if po.datasets[key].thumb is not None]
    if not candidates:
        # The selector tests above are the changed API-enforcement surface. This
        # focused check creates an in-memory permanent fixture only when needed.
        with patch.object(ProcessedObject, "build_thumb", return_value=None):
            po.add_dataset("UI_API_THUMB", np.zeros((4, 5), dtype=np.uint8))
        key = "UI_API_THUMB"
    else:
        key = candidates[0]

    permanent = po.datasets[key]
    permanent.thumb = Image.new("RGB", (8, 8), (0, 0, 255))
    with patch.object(ProcessedObject, "build_thumb", return_value=None):
        po.add_temp_dataset(key, np.zeros((4, 5), dtype=np.uint8), ext=permanent.ext)
    po.temp_datasets[key].thumb = Image.new("RGB", (8, 8), (255, 0, 0))

    table = page._box_table
    table.dataset_key = key
    with patch.object(po, "load_thumbs", return_value=None):
        pixmap = table._get_thumb_pixmap(po)

    check(not pixmap.isNull(), "thumbnail lookup returned a null pixmap")
    colour = pixmap.toImage().pixelColor(4, 4)
    check(colour.red() > colour.blue(),
          "thumbnail table rendered the permanent thumbnail instead of temp")


TESTS = [
    ("control-panel active key union", test_control_panel_receives_active_union),
    ("strip headers after hole refresh", test_refresh_receives_active_union),
    ("overview export selector", test_overview_selector_receives_active_union),
    ("downhole feature selector", test_feature_selector_receives_temp_only_key),
    ("downhole mineral-map selector", test_minmap_selector_receives_temp_only_family),
    ("thumbnail temp replacement", test_thumbnail_prefers_temp_replacement),
]


def validate_path() -> None:
    check(bool(HOLE_PATH.strip()), "set HOLE_PATH near the top of this script")
    check(Path(HOLE_PATH).is_dir(), f"HOLE_PATH is not a directory: {HOLE_PATH}")


def main() -> int:
    try:
        validate_path()
    except Exception as exc:
        print(f"SETUP ERROR: {exc}")
        return 2

    app = QApplication.instance() or QApplication(sys.argv[:1])

    try:
        hole = load_hole()
        stage_ui_fixtures(hole)
        page = make_page(hole)
    except Exception as exc:
        print(f"SETUP ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2

    print(f"HOLE_PATH: {HOLE_PATH}")
    print("Qt platform: offscreen")
    print("Source-data writes: disabled; all fixtures are in-memory temp datasets")

    passed = failed = 0
    for name, test in TESTS:
        try:
            if test in (test_refresh_receives_active_union,
                        test_thumbnail_prefers_temp_replacement):
                test(page, hole)
            else:
                test(page)
            print(f"PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    page.close()
    QApplication.processEvents()
    print(f"Result: {passed} passed, 0 skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
