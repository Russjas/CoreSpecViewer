# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 11:37:13 2025

@author: russj
"""
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QSize, pyqtSignal
from PIL import ImageQt

from PyQt5.QtWidgets import QHeaderView, QAbstractItemView, QGridLayout, QLabel
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
from objects import HoleObject, ProcessedObject

import os, sys
import numpy as np
from PyQt5.QtWidgets import (QSplitter, QVBoxLayout, QHBoxLayout, QTableWidgetItem,QTableWidget,
                             QApplication, QWidget, QToolBar, QPushButton, QFileDialog,
                             QTableView, QMessageBox, QInputDialog, QComboBox,
                             QDialog, QFormLayout)

from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from util_windows import (SpectralImageCanvas, ImageCanvas2D, 
                          InfoTable, SpectrumWindow, busy_cursor,
                          IdSetFilterProxy, two_choice_box)
from tool_dispatcher import ToolDispatcher
import tools as t
from context import CurrentContext
from pages import BasePage
from PyQt5.QtGui import QPixmap, QIcon

class HoleBoxTable(QTableWidget):
    """
    Configurable table showing ProcessedObjects in a HoleObject.
    Supported column keys: 'box', 'thumb', (future: 'depth', etc.)
    """
    box_selected = pyqtSignal(int)
    def __init__(self, page: "HolePage", parent=None, columns=None, dataset_key='savgol'):
        self.columns = columns or ["box", "thumb"]
        self._page = page
        self.dataset_key = dataset_key
        super().__init__(0, len(self.columns), parent)

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

        self.currentCellChanged.connect(self._on_current_cell_changed)

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

    # ------------------------------------------------------------------
    def _get_thumb_pixmap(self, po):
        po.load_thumbs()
        key = self.dataset_key
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
        Optionally repopulates immediately if a HoleObject is provided.
        """
        self.dataset_key = key
        self.cxt = self._page.cxt
        if self.cxt is not None and self.cxt.ho is not None:
            self.populate_from_hole()  

    # ------------------------------------------------------------------
    def _on_current_cell_changed(self, r, c, pr, pc):
        if r < 0:
            return
        # Find the "box" column if it exists
        if "box" not in self.columns:
            return
        box_col = self.columns.index("box")
        item = self.item(r, box_col)
        if item is None:
            return
        box_num = item.data(Qt.UserRole)
        if box_num is not None:
            try:
                self.box_selected.emit(int(box_num))
            except Exception:
                pass

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
        
        
        # ---- Secondary strip dataset ----------------------------------
        self.layout.addSpacing(12)
        self.layout.addWidget(QLabel("Secondary strip dataset:", self))

        self.secondary_combo = QComboBox(self)
        self.secondary_combo.setToolTip(
            "Controls which dataset is used to build thumbnails in the "
            "second strip table."
        )
        self.secondary_combo.currentTextChanged.connect(
            self._on_secondary_dataset_changed
        )
        self.layout.addWidget(self.secondary_combo)

        self.layout.addStretch(1)

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
            self._set_dataset_keys([])
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

        # ---- dataset keys for secondary strip -------------------------
        keys = set()
        try:
            if self.cxt.ho.boxes:
                for box in self.cxt.ho:
                    keys = keys | box.datasets.keys() | box.temp_datasets.keys()
                    
        except Exception:
            pass

        self._set_dataset_keys(keys)

    # ------------------------------------------------------------------
    def _set_dataset_keys(self, keys: list[str]):
        """Populate the combobox without firing change signals."""
        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()
        for k in keys:
            self.secondary_combo.addItem(k)

        # try to default to savgol_cr, then savgol, then first key
        default = None
        for cand in ("savgol_cr", "savgol"):
            if cand in keys:
                default = cand
                break
        if default is None and keys:
            default = keys[0]

        if default is not None:
            idx = self.secondary_combo.findText(default)
            if idx >= 0:
                self.secondary_combo.setCurrentIndex(idx)

        self.secondary_combo.blockSignals(False)

    # ------------------------------------------------------------------
    def _on_secondary_dataset_changed(self, key: str):
        """
        User changed the dataset key for the secondary strip. Rebuild the
        second table using this key.
        """
        key = key.strip()
        if not key:
            return
        self.cxt = self._page.cxt
        if self.cxt is None or self.cxt.ho is None:
            return

        # Reuse existing populate logic on the second table
        self._page._box_table2.set_dataset_key(key)

        
        
class HolePage(BasePage):
    """
    Hole-level view.

    Left panel: table listing all ProcessedObjects (boxes) in the current HoleObject,
                showing the savgol thumbnail and box number.
    Right/third panels can be added later (downhole plots, metadata, etc.).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Replace the default left canvas with our box table
        self._box_table = HoleBoxTable(self, columns=["box", "thumb"])
        self._add_left(self._box_table)
        self._box_table2 = HoleBoxTable(self, columns=["thumb"], dataset_key = 'savgol_cr')
        self._add_right(self._box_table2)
        self._control_panel = HoleControlPanel(self)
        self._add_third(self._control_panel)
        # When a row is selected, update the CurrentContext.po
        self._box_table.cellDoubleClicked.connect(self._on_box_selected)
           
        btn_open  = QPushButton("Open hole dataset", self)
        self._control_panel.layout.addWidget(btn_open)
        btn_open.clicked.connect(self.open_hole)
        
        btn_save = QPushButton("Save all changes", self)
        self._control_panel.layout.addWidget(btn_save)
        btn_save.clicked.connect(self.save_changes)
        
        
        
        self.canvas = ImageCanvas2D()
        #self._add_right(self.canvas)
        
        #scroll sync between the two tables
        self._syncing_scroll = False
        vbar1 = self._box_table.verticalScrollBar()
        vbar2 = self._box_table2.verticalScrollBar()
        vbar1.valueChanged.connect(self._on_table1_scrolled)
        vbar2.valueChanged.connect(self._on_table2_scrolled)
        
    # ------------------------------------------------------------------
    # Context handling
    # ------------------------------------------------------------------
    
    def test_func(self, *args):
        print(*args, 'test func')
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
        self._box_table.populate_from_hole()
        self._box_table2.populate_from_hole()
        self._control_panel.update_for_hole()
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
        # This will also update .current via CurrentContext logic
        self.cxt.current = po
    
    def save_changes(self):
        print('this called')
        
        if self.cxt is not None and self.cxt.ho is not None:
            with busy_cursor('Saving.....', self):
                print('saving')
                for po in self.cxt.ho:
                    if po.has_temps:
                        print(po.metadata['box number'])
                        po.commit_temps()
                        po.build_all_thumbs()
                        po.save_all_thumbs()
                        po.save_all()
                self._refresh_from_hole()
                        
    
    def update_display(self, key=''):
        self._refresh_from_hole()
        
        
    
        
        
    # sync table scroll handles
    def _on_table1_scrolled(self, value: int):
        if self._syncing_scroll:
            return
        self._syncing_scroll = True
        self._box_table2.verticalScrollBar().setValue(value)
        self._syncing_scroll = False

    def _on_table2_scrolled(self, value: int):
        if self._syncing_scroll:
            return
        self._syncing_scroll = True
        self._box_table.verticalScrollBar().setValue(value)
        self._syncing_scroll = False
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HolePage()
    #test = HoleObject.build_from_parent_dir('D:/Clonminch_swir/')
    #test = HoleObject.build_from_parent_dir('D:/Multi_process_test')
    #test.get_all_thumbs()
    #viewer.set_hole(test)
    viewer.show()
    sys.exit(app.exec_())
