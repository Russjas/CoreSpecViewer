"""
Created on Fri Nov 14 11:37:13 2025

@author: russj
"""



import sys

import numpy as np
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QStandardItem
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import HoleObject
from .base_page import BasePage
from .util_windows import ClosableWidgetWrapper, busy_cursor


class HoleBoxTable(QTableWidget):
    """
    Configurable table showing ProcessedObjects in a HoleObject.
    Supported column keys: 'box', 'thumb', (future: 'depth', etc.)
    """

    def __init__(self, page: "HolePage", parent=None, columns=None, dataset_key='savgol'):
        self.columns = columns or ["box", "thumb"]
        self._page = page
        self.dataset_key = dataset_key
        super().__init__(0, len(self.columns), parent)

        self._header_combo = None  # for thumb column dataset chooser

        # pretty header labels
        labels = []
        for c in self.columns:
            if c == "box":
                labels.append("Box")
            elif c == "thumb":
                labels.append(self.dataset_key)
            else:
                labels.append(c.capitalize())
        self.setHorizontalHeaderLabels(labels)

        # common table setup
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setShowGrid(False)

        # thumbnail display defaults
        self.setIconSize(QSize(400, 120))
        self.verticalHeader().setDefaultSectionSize(80)

        hdr = self.horizontalHeader()
        hdr.setStretchLastSection(False)

        # sensible defaults per column type
        for idx, name in enumerate(self.columns):
            if name == "box":
                hdr.setSectionResizeMode(idx, hdr.ResizeToContents)
            elif name == "thumb":
                hdr.setSectionResizeMode(idx, hdr.Interactive)
            else:
                hdr.setSectionResizeMode(idx, hdr.ResizeToContents)

        # keep header-combo positioned correctly
        hdr.sectionResized.connect(self._update_header_combo_geometry)
        hdr.sectionMoved.connect(self._update_header_combo_geometry)

    # ------------------------------------------------------------------
    def populate_from_hole(self):
        self.setRowCount(0)
        self.cxt = self._page.cxt
        if self.cxt is None or self.cxt.ho is None:
            return
        hole = self.cxt.ho
        try:
            items = sorted(hole.iter_items(), key=lambda kv: kv[0])
        except AttributeError:
            print('error-ed out here')
            items = sorted(hole.boxes.items(), key=lambda kv: kv[0])

        for row, (box_num, po) in enumerate(items):
            self.insertRow(row)

            # track row height after thumbnail is created
            desired_row_height = 0

            col_index = 0
            for col_name in self.columns:

                # ---- BOX NUMBER -----------------------------------------
                if col_name == "box":
                    item = QTableWidgetItem(str(box_num))
                    item.setData(Qt.UserRole, int(box_num))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.setItem(row, col_index, item)

                # ---- THUMBNAIL -----------------------------------------
                elif col_name == "thumb":
                    pix = self._get_thumb_pixmap(po)

                    if not pix.isNull():
                        pix = pix.scaled(
                            self.iconSize(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation,
                        )
                        desired_row_height = max(desired_row_height, pix.height())

                    item = QTableWidgetItem()
                    if not pix.isNull():
                        item.setIcon(QIcon(pix))
                    self.setItem(row, col_index, item)

                # ---- FUTURE COLUMNS ------------------------------------
                else:
                    self.setItem(row, col_index, QTableWidgetItem(""))

                col_index += 1

            if desired_row_height:
                self.setRowHeight(row, desired_row_height + 4)

        # Make thumb column wide if it exists
        if "thumb" in self.columns:
            thumb_col = self.columns.index("thumb")
            self.setColumnWidth(thumb_col, self.iconSize().width() + 5)

        if self.rowCount() > 0:
            self.setCurrentCell(0, 0)

        # refresh header combo label if needed
        self._update_header_label()

    # ------------------------------------------------------------------
    def _get_thumb_pixmap(self, po):
        po.load_thumbs()
        key = self.dataset_key
        ds = getattr(po, "temp_datasets", {}).get(key)
        if ds is None:
            ds = getattr(po, "datasets", {}).get(key)
            if ds is None:
                return QPixmap()

        if ds.thumb is None:
            try:
                print('building')
                po.build_thumb(key)
            except Exception:
                return QPixmap()

        if ds.thumb is None:
            return QPixmap()

        try:
            from io import BytesIO
            buf = BytesIO()
            ds.thumb.convert("RGB").save(buf, format="JPEG")
            pix = QPixmap()
            pix.loadFromData(buf.getvalue(), "JPEG")
            return pix
        except Exception:
            return QPixmap()

    def set_dataset_key(self, key):
        """
        Change the dataset key used for thumbnails (e.g. 'savgol_cr', 'savgol', 'RGB').
        Repopulates immediately if a HoleObject is present.
        """
        self.dataset_key = key
        self._update_header_label()
        self.cxt = self._page.cxt
        if self.cxt is not None and self.cxt.ho is not None:
            self.populate_from_hole()

    # ------------------------------------------------------------------
    # Header combobox handling
    # ------------------------------------------------------------------
    def set_header_dataset_keys(self, keys: list[str]):
        """
        Create or update a QComboBox in the thumb-column header to select
        which dataset key is used for thumbnails.
        """
        if "thumb" not in self.columns:
            return

        thumb_idx = self.columns.index("thumb")
        header = self.horizontalHeader()

        # lazy-create combo
        if self._header_combo is None:
            combo = QComboBox(header)
            combo.currentTextChanged.connect(self._on_header_dataset_changed)
            self._header_combo = combo
        else:
            combo = self._header_combo

        combo.blockSignals(True)
        combo.clear()
        def add_header_item(combo, text):
            model = combo.model()
            row = model.rowCount()
            model.insertRow(row)
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsEnabled)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            model.setItem(row, 0, item)

        base_whitelist = {"savgol", "savgol_cr", "mask", "segments", "cropped"}
        unwrap_prefixes = ("Dhole",)  # DholeAverage, DholeMask, DholeDepths
        non_vis_suff = {'LEGEND', 'CLUSTERS', "stats", "bands", 'metadata' }
        base = []
        unwrapped = []
        products = []
        non_vis = []

        for k in sorted(keys):  # stable order
            if k in base_whitelist:
                base.append(k)
            elif any(k.startswith(pfx) for pfx in unwrap_prefixes):
                unwrapped.append(k)
            elif any(k.endswith(sfx) for sfx in non_vis_suff):
                non_vis.append(k)
            else:
                products.append(k)

        add_header_item(combo, "---Base data---")
        for k in base:
            combo.addItem(k)
        add_header_item(combo, "---Products---")
        for k in products:
            combo.addItem(k)




        default = self.dataset_key
        if default is not None:
            idx = combo.findText(default)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                self.dataset_key = default

        combo.blockSignals(False)

        # clear any existing text in that header item
        item = self.horizontalHeaderItem(thumb_idx)
        if item is not None:
            item.setText("")

        self._reposition_header_combo()

    def _on_header_dataset_changed(self, key: str):
        key = key.strip()
        if not key:
            return
        # this will also repopulate from hole
        self.set_dataset_key(key)

    def _reposition_header_combo(self):
        if self._header_combo is None or "thumb" not in self.columns:
            return
        header = self.horizontalHeader()
        thumb_idx = self.columns.index("thumb")
        section_pos = header.sectionPosition(thumb_idx)
        section_size = header.sectionSize(thumb_idx)
        h = header.height()
        self._header_combo.setGeometry(
            section_pos + 2,
            1,
            max(40, section_size - 4),
            h - 2,
        )
        self._header_combo.show()

    def _update_header_combo_geometry(self, *args):
        # args are (logicalIndex, oldSize, newSize) or (logicalIndex, oldPos, newPos)
        self._reposition_header_combo()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_header_combo()

    def _update_header_label(self):
        """If no header combo is used, keep the thumb header text in sync."""
        if "thumb" not in self.columns:
            return
        if self._header_combo is not None:
            # combo visible, label handled by combo
            return
        thumb_idx = self.columns.index("thumb")
        item = self.horizontalHeaderItem(thumb_idx)
        if item is not None:
            item.setText(self.dataset_key)


class HoleControlPanel(QWidget):
    """
    Narrow side panel for hole-level controls.

    - Displays basic hole metadata (ID, #boxes, depth range)
    - Lets the user choose the dataset key used in the secondary strip table.
    """
    def __init__(self, page: "HolePage", parent=None):
        super().__init__(parent or page)
        self._page = page
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(8)

        # ---- Hole info -------------------------------------------------
        self.lbl_hole_id = QLabel("—")
        self.lbl_box_count = QLabel("—")
        self.lbl_depth_range = QLabel("—")

        info_layout = QFormLayout()
        info_layout.setLabelAlignment(Qt.AlignLeft)
        info_layout.addRow("Hole ID:", self.lbl_hole_id)
        info_layout.addRow("# boxes:", self.lbl_box_count)
        info_layout.addRow("Depth range:", self.lbl_depth_range)
        self.layout.addLayout(info_layout)

        # --- Extra column control panel---
        # Create a local container for the combo + button
        combo_block = QWidget(self)
        combo_layout = QVBoxLayout(combo_block)
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.setSpacing(1)   # tight vertical spacing

        # Label
        label = QLabel("Add extra columns:", combo_block)
        combo_layout.addWidget(label)

        # Combo box
        self.secondary_combo = QComboBox(combo_block)
        self.secondary_combo.setToolTip(
            "Controls which dataset is used to build thumbnails in the "
            "new strip table."
        )

        combo_layout.addWidget(self.secondary_combo)

        # Button directly below combo
        self.create_button = QPushButton("Show", combo_block)
        self.create_button.clicked.connect(self._on_display_btn_clicked)
        combo_layout.addWidget(self.create_button)

        # Now add the *whole block* to your main layout
        self.layout.addWidget(combo_block)


        self.layout.addStretch(1)

    def _set_dataset_keys(self):
        """Populate the combobox without firing change signals."""
        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()
        keys = set()
        try:
            if self.cxt.ho.boxes:
                for box in self.cxt.ho:
                    keys = keys | box.datasets.keys() | box.temp_datasets.keys()
        except Exception:
            pass
        if keys:
            combo = self.secondary_combo
            def add_header_item(combo, text):
                model = combo.model()
                row = model.rowCount()
                model.insertRow(row)
                item = QStandardItem(text)
                item.setFlags(Qt.ItemIsEnabled)
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                model.setItem(row, 0, item)

            base_whitelist = {"savgol", "savgol_cr", "mask", "segments", "cropped"}
            unwrap_prefixes = ("Dhole",)  # DholeAverage, DholeMask, DholeDepths
            non_vis_suff = {'LEGEND', 'CLUSTERS', "stats", "bands", 'metadata' }
            base = []
            unwrapped = []
            products = []
            non_vis = []

            for k in sorted(keys):  # stable order
                if k in base_whitelist:
                    base.append(k)
                elif any(k.startswith(pfx) for pfx in unwrap_prefixes):
                    unwrapped.append(k)
                elif any(k.endswith(sfx) for sfx in non_vis_suff):
                    non_vis.append(k)
                else:
                    products.append(k)

            add_header_item(combo, "---Base data---")
            for k in base:
                combo.addItem(k)
            add_header_item(combo, "---Products---")
            for k in products:
                combo.addItem(k)

        self.secondary_combo.blockSignals(False)

    # ------------------------------------------------------------------
    def _on_display_btn_clicked(self, key: str):
        """
        User changed the dataset key for the secondary strip. Rebuild the
        second table using this key.
        """
        text = self.secondary_combo.currentText()

        key = text.strip()
        if not key:
            return
        self.cxt = self._page.cxt
        if self.cxt is None or self.cxt.ho is None:
            return

        self._page.add_column(dataset_key=key)


    # ------------------------------------------------------------------
    def update_for_hole(self):
        """
        Refresh labels and available dataset keys when a new HoleObject is set.
        Does NOT trigger repopulation of the tables; that only happens on
        user interaction.
        """
        # ---- labels ----------------------------------------------------
        self.cxt = self._page.cxt
        if self.cxt is None or self.cxt.ho is None:
            self.lbl_hole_id.setText("—")
            self.lbl_box_count.setText("—")
            self.lbl_depth_range.setText("—")
            return

        self.lbl_hole_id.setText(str(self.cxt.ho.hole_id or "—"))
        self.lbl_box_count.setText(str(self.cxt.ho.num_box))

        # Depth range from per-box metadata if present
        starts = []
        stops = []
        for meta in self.cxt.ho.hole_meta.values():
            try:
                s = float(meta.get("core depth start", "nan"))
                if np.isfinite(s):
                    starts.append(s)
            except Exception:
                pass
            try:
                e = float(meta.get("core depth stop", "nan"))
                if np.isfinite(e):
                    stops.append(e)
            except Exception:
                pass

        if starts and stops:
            dmin = min(starts)
            dmax = max(stops)
            self.lbl_depth_range.setText(f"{dmin:.2f}–{dmax:.2f} m")
        else:
            self.lbl_depth_range.setText("—")
        self._set_dataset_keys()







class HolePage(BasePage):
    """
    Hole-level view.

    Left panel: table listing all ProcessedObjects (boxes) in the current HoleObject,
                showing the savgol thumbnail and box number.
    Right/third panels can be added later (downhole plots, metadata, etc.).
    """
    changeView = pyqtSignal(str)
    def __init__(self, parent=None):

        super().__init__(parent)
        self._syncing_scroll = False
        self._scroll_tables: list[HoleBoxTable] = []

        #Details and buttons
        self._control_panel = HoleControlPanel(self)
        self._add_left(self._control_panel)
        #Default thumb column
        self._box_table = HoleBoxTable(self, columns=["box", "thumb"])
        self._add_closable_widget(self._box_table, '', closeable = False)
        self._register_scroll_table(self._box_table)
        #Extra, dynamic thumb columns
        self.extra_columns = []
        self.add_column()

        # When a row is selected, update the CurrentContext.po, (default column only, for now)
        self._box_table.cellDoubleClicked.connect(self._on_box_selected)
        self._box_table.cellClicked.connect(self._on_box_clicked)

        btn_open  = QPushButton("Open hole dataset", self)
        self._control_panel.layout.addWidget(btn_open)
        btn_open.clicked.connect(self.open_hole)

        btn_save = QPushButton("Save all changes", self)
        self._control_panel.layout.addWidget(btn_save)
        btn_save.clicked.connect(self.save_changes)


    #====== Dynamic column handling =========================
    def add_column(self, dataset_key = 'mask'):
        new_col = HoleBoxTable(self, columns=["thumb"], dataset_key=dataset_key)
        new_col.cellDoubleClicked.connect(self._on_box_selected)
        new_col.cellClicked.connect(self._on_box_clicked)
        self._add_closable_widget(new_col, '')
        self.extra_columns.append(new_col)
        self._register_scroll_table(new_col)
        self._refresh_from_hole()

    def remove_widget(self, w: QWidget):
        """
        Override BasePage.remove_widget so that when a closable thumb column
        is closed, we keep self.extra_columns in sync and drop references to
        the underlying HoleBoxTable.
        """
        inner = None
        if isinstance(w, ClosableWidgetWrapper):
            inner = getattr(w, "wrapped_widget", None)

        if isinstance(inner, HoleBoxTable):
            try:
                self.extra_columns.remove(inner)
            except ValueError:
                # Already removed or was never registered; ignore
                pass
            self._unregister_scroll_table(inner)
        super().remove_widget(w)

    def _register_scroll_table(self, table: HoleBoxTable):
        """
        Add a HoleBoxTable to the scroll-sync group.
        """
        if table in self._scroll_tables:
            return  # already registered
        self._scroll_tables.append(table)
        vbar = table.verticalScrollBar()
        vbar.valueChanged.connect(self._on_any_table_scrolled)

    def _unregister_scroll_table(self, table: HoleBoxTable):
        """
        Remove a HoleBoxTable from the scroll-sync group and disconnect signals.
        Called when a column is closed.
        """
        if table in self._scroll_tables:
            self._scroll_tables.remove(table)
        try:
            vbar = table.verticalScrollBar()
            vbar.valueChanged.disconnect(self._on_any_table_scrolled)
        except Exception:
            # Already disconnected or table is being destroyed
            pass

    def _on_any_table_scrolled(self, value: int):
        """
        Keep all registered HoleBoxTables vertically aligned.
        Any table can act as the scroll driver.
        """
        if self._syncing_scroll:
            return

        self._syncing_scroll = True
        src_vbar = self.sender()

        for table in self._scroll_tables:
            vbar = table.verticalScrollBar()
            if vbar is src_vbar:
                continue
            vbar.setValue(value)

        self._syncing_scroll = False
    # ------------------------------------------------------------------
    # Context handling
    # ------------------------------------------------------------------

    def open_hole(self):
        path = QFileDialog.getExistingDirectory(
                   self,
                   "Select hole directory of processed data",
                   "",
                   QFileDialog.ShowDirsOnly
                   )
        if not path:
            return
        with busy_cursor('loading...', self):
            hole = HoleObject.build_from_parent_dir(path)
            print('hole loaded')
            self.set_hole(hole)

    def set_hole(self, hole):
        """Set the HoleObject and repopulate the left table."""
        self.cxt.ho = hole
        print('hole set')
        self._refresh_from_hole()

    def _refresh_from_hole(self):
        """
        Populate the box table from the current HoleObject, if any.
        """
        self._control_panel.update_for_hole()
        self._box_table.populate_from_hole()
        for col in self.extra_columns:
            col.populate_from_hole()
        self._control_panel.update_for_hole()


        keys = set()
        try:
            if self.cxt.ho.boxes:
                for box in self.cxt.ho:
                    keys = keys | box.datasets.keys() | box.temp_datasets.keys()

        except Exception:
            pass
        if keys:
            self._box_table.set_header_dataset_keys(keys)
            for col in self.extra_columns:
                col.set_header_dataset_keys(keys)


    # ------------------------------------------------------------------
    # Slots / handlers
    # ------------------------------------------------------------------
    def _on_box_selected(self, row: int, column: int):
        """
        Called when the user selects a different row in the box table.
        Sets CurrentContext.po to the corresponding ProcessedObject, if present.
        """
        if self.cxt is None or self.cxt.ho is None:
            return
        box_num_item = self._box_table.item(row, 0)
        if not box_num_item:
            return
        box_num = box_num_item.text()
        try:
            box_num = int(box_num)
        except ValueError:
            return
        ho = self.cxt.ho
        po = ho.boxes.get(box_num)
        if po is None:
            return

        self.cxt.current = po
        self.changeView.emit('vis')

    def _on_box_clicked(self, row: int, column: int):
        """
        Called when the user selects a different row in the box table.
        Sets CurrentContext.po to the corresponding ProcessedObject, if present.
        """
        if self.cxt is None or self.cxt.ho is None:
            return
        box_num_item = self._box_table.item(row, 0)
        if not box_num_item:
            return
        box_num = box_num_item.text()
        try:
            box_num = int(box_num)
        except ValueError:
            return
        ho = self.cxt.ho
        po = ho.boxes.get(box_num)
        if po is None:
            return

        self.cxt.current = po


    def save_changes(self):

        if self.cxt is not None and self.cxt.ho is not None:
            with busy_cursor('Saving.....', self):
                for po in self.cxt.ho:
                    if po.has_temps:
                        po.commit_temps()
                        po.save_all()
                        po.reload_all()
                        po.load_thumbs()
                self._refresh_from_hole()


    def update_display(self, key=''):
        self._refresh_from_hole()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HolePage()
    #test = HoleObject.build_from_parent_dir('D:/Clonminch_swir/')
    #test = HoleObject.build_from_parent_dir('D:/Multi_process_test')
    #test.get_all_thumbs()
    #viewer.set_hole(test)
    viewer.show()
    sys.exit(app.exec_())
