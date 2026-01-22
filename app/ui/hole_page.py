"""
UI page for viewing multi-box holes and downhole data.

Allows selection of boxes, linking of datasets, and merged downhole outputs.
"""
import sys
import logging

import numpy as np
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QStandardItem, QPalette 
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
    QStyledItemDelegate,
    QStyle,
    QMessageBox,
    QInputDialog,
    QLineEdit,
    QFrame
)

from ..models import HoleObject
from ..interface import profile_tools as pt
from ..config import feature_keys
from .base_page import BasePage
from .util_windows import ClosableWidgetWrapper, busy_cursor, ImageCanvas2D
from .cluster_window import ClusterWindow
from .band_math_dialogue import BandMathsDialog
from .display_text import gen_display_text

logger = logging.getLogger(__name__)

class NoSelectionDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Remove the State_Selected flag so Qt won't draw highlight
        option.state &= ~QStyle.State_Selected
        super().paint(painter, option, index)

class HoleBoxTable(QTableWidget):
    """
    A specialized table widget for displaying a list of boxes (ProcessedObjects) 
    associated with the current HoleObject.

    Features:
    - Renders columns for box numbers and image thumbnails (ProcessObject datasets).
    - Supports dynamic switching of the displayed thumbnail dataset via a custom header combobox.
    - Handles thumbnail generation, resizing, and aspect ratio preservation.
    - Intended to be used as a vertical "strip" log, often synchronized with other tables.
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
        self.setItemDelegate(NoSelectionDelegate(self))
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
            logger.error(f"Failed to build thumb for {key}", exc_info=True)
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
            combo.currentIndexChanged.connect(self._on_header_dataset_changed)
            self._header_combo = combo
        else:
            combo = self._header_combo

        combo.blockSignals(True)
        combo.clear()
        
        def add_header_item(combo: QComboBox, text: str):
            model = combo.model()
            row = model.rowCount()
            model.insertRow(row)
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsEnabled)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            model.setItem(row, 0, item)
    
        def add_key_item(raw_key: str):
            # Display text in the UI, raw key in UserRole (via userData)
            combo.addItem(gen_display_text(raw_key), raw_key)

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
            add_key_item(k)
        add_header_item(combo, "---Products---")
        for k in products:
            add_key_item(k)




        default = self.dataset_key
        if default is not None:
            idx = combo.findData(default, role=Qt.UserRole)
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
    def _on_header_dataset_changed(self, _idx: int):
        """
        Slot called when the header dataset combo changes.
    
        Reads the raw dataset key from Qt.UserRole, not from display text.
        """
        combo = self._header_combo
        if combo is None:
            return
    
        raw_key = combo.currentData(Qt.UserRole)
        if not raw_key:
            return
    
        # Avoid redundant updates
        if raw_key == self.dataset_key:
            return
    
        self.set_dataset_key(raw_key)
        
        
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
            item.setText(gen_display_text(self.dataset_key))


class HoleControlPanel(QWidget):
    """
    Side panel for displaying hole metadata and executing hole-level operations.

    Displays:
    - Basic metadata: Hole ID, total box count, and depth range.

    Controls:
    - Box Level: Select and add extra visualization columns (strips) to the main view.
    - Full Hole: 
        - Visualize downhole datasets (products, mineral maps) and legends.
        - Set resampling window steps.
        - Generate base datasets, downhole features, and mineral maps from box data.
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
        separator1 = QFrame(self)
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator1)

        # --- Box level control panel---
        combo_block = QWidget(self)
        combo_layout = QVBoxLayout(combo_block)
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.setSpacing(1)
        # Label
        label = QLabel("Add extra columns:", combo_block)
        combo_layout.addWidget(label)
        # Combo box
        self.secondary_combo = QComboBox(combo_block)
        self.secondary_combo.setToolTip(
            "Controls which dataset is used to build thumbnails in the "
            "new strip table.")
        combo_layout.addWidget(self.secondary_combo)
        # Button directly below combo
        self.create_button = QPushButton("Show", combo_block)
        self.create_button.clicked.connect(self._on_display_btn_clicked)
        combo_layout.addWidget(self.create_button)
        self.layout.addWidget(combo_block)
        separator2 = QFrame(self)
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator2)
        # --- Downhole datasets control panal
        combo_block_full = QWidget(self)
        combo_layout_full = QVBoxLayout(combo_block_full)
        combo_layout_full.setContentsMargins(0, 0, 0, 0)
        combo_layout_full.setSpacing(1)
        # Label
        label = QLabel("Down hole datasets:", combo_block_full)
        combo_layout_full.addWidget(label)
        self.full_data_combo = QComboBox(combo_block_full)
        self.full_data_combo.setToolTip(
            "Controls which full downhole datasets can be displayed")
        combo_layout_full.addWidget(self.full_data_combo)
        
        show_dhole_button = QPushButton("Show", combo_block_full)
        show_dhole_button.clicked.connect(self.show_downhole)
        combo_layout_full.addWidget(show_dhole_button)
        
        btn_set_step = QPushButton("Set resampling window", combo_block_full)
        btn_set_step.clicked.connect(self.set_step)
        combo_layout_full.addWidget(btn_set_step)
        
        gen_base_button = QPushButton("Generate base datasets", combo_block_full)
        gen_base_button.clicked.connect(self.gen_base_datasets)
        combo_layout_full.addWidget(gen_base_button)
        
        self.layout.addWidget(combo_block_full)
        separator3 = QFrame(self)
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator3)
        # ----- box-derived dataset controls
        box_derived_block = QWidget(self)
        box_derived_layout = QVBoxLayout(box_derived_block)
        box_derived_layout.setContentsMargins(0, 0, 0, 0)
        box_derived_layout.setSpacing(1)
        
        label = QLabel("Box derived dataset controls:", box_derived_block)
        box_derived_layout.addWidget(label)
        
        gen_min_map_button = QPushButton("Generate Downhole MinMap datasets", box_derived_block)
        gen_min_map_button.clicked.connect(self.dhole_minmaps_create)
        box_derived_layout.addWidget(gen_min_map_button)
        
        gen_feats_button = QPushButton("Generate Downhole feature datasets", box_derived_block)
        gen_feats_button.clicked.connect(self.dhole_feats_create)
        box_derived_layout.addWidget(gen_feats_button)
        
        self.layout.addWidget(box_derived_block)
        separator4 = QFrame(self)
        separator4.setFrameShape(QFrame.HLine)
        separator4.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator4)
        # ----- profile-derived dataset controls
        profile_derived_block = QWidget(self)
        profile_derived_layout = QVBoxLayout(profile_derived_block)
        profile_derived_layout.setContentsMargins(0, 0, 0, 0)
        profile_derived_layout.setSpacing(1)
        
        label = QLabel("Profile derived dataset controls:", profile_derived_block)
        profile_derived_layout.addWidget(label)
        
        gen_kmeans = QPushButton("Calculate profile k-means", profile_derived_block)
        gen_kmeans.clicked.connect(self.prof_kmeans)
        profile_derived_layout.addWidget(gen_kmeans)
               
        # Extract feature section
        feature_label = QLabel("Extract Features:", profile_derived_block)
        feature_label.setStyleSheet("QLabel { font-style: italic; margin-left: 10px; }")
        profile_derived_layout.addWidget(feature_label)
        
        # Combo box with feature keys
        self.feature_combo = QComboBox(profile_derived_block)
        self.feature_combo.setToolTip("Select a feature to extract")
        for key in feature_keys:
            self.feature_combo.addItem(key)
        profile_derived_layout.addWidget(self.feature_combo)
        
        # Button to run extraction
        extract_button = QPushButton("Extract Feature", profile_derived_block)
        extract_button.clicked.connect(self.prof_feature_extraction)
        profile_derived_layout.addWidget(extract_button)
        
        bmath_button = QPushButton("Band Maths", profile_derived_block)
        bmath_button.clicked.connect(self.profile_band_maths)
        profile_derived_layout.addWidget(bmath_button)
                
        self.layout.addWidget(profile_derived_block)
        
        separator5 = QFrame(self)
        separator5.setFrameShape(QFrame.HLine)
        separator5.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(separator5)
        self.layout.addStretch(1)
        

#---------initiation and refresh logic-----------------------------------------

    def _set_dataset_keys(self):
        """Populate the combobox without firing change signals."""
        self.secondary_combo.blockSignals(True)
        self.secondary_combo.clear()
        self.full_data_combo.blockSignals(True)
        self.full_data_combo.clear()
        keys = set()
        try:
            if self.cxt.ho.boxes:
                for box in self.cxt.ho:
                    keys = keys | box.datasets.keys() | box.temp_datasets.keys()
        except Exception:
            pass
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
        
        def add_key_item(combo, raw_key: str):
            # Display text in the UI, raw key in UserRole (via userData)
            combo.addItem(gen_display_text(raw_key), raw_key)
            
        if keys:
            combo = self.secondary_combo
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
                add_key_item(combo, k)
            add_header_item(combo, "---Products---")
            for k in products:
                add_key_item(combo, k)
        
        add_header_item(self.full_data_combo, "---Base data---")
        for k in self.cxt.ho.base_datasets.keys():
            add_key_item(self.full_data_combo, k)
        add_header_item(self.full_data_combo, "---Product data---")
        for k in self.cxt.ho.product_datasets.keys():
            add_key_item(self.full_data_combo, k)
        self.secondary_combo.blockSignals(False)
        self.full_data_combo.blockSignals(False)

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


# ---------downhole data control handlers---------------------------------------------------------
    def _on_display_btn_clicked(self):
        """
        User changed the dataset key for the secondary strip. Rebuild the
        second table using this key.
        """
        
        key = self.secondary_combo.currentData(Qt.UserRole)
        
        if not key:
            return
        self.cxt = self._page.cxt
        if self.cxt is None or self.cxt.ho is None:
            logger.info(f"Button clicked: Display change for box columns to {key} failed because no hole data loaded")
            return
        logger.info(f"Button clicked: Display change for box columns to {key}")
        self._page.add_column(dataset_key=key)


    def set_step(self):
        if not self.cxt.ho:
            return
        dlg = QInputDialog(self)
        dlg.setInputMode(QInputDialog.DoubleInput)
        dlg.setWindowTitle("Resampling window")
        dlg.setLabelText("Enter resampling window in metres:")
        
        # Access the line edit and set placeholder
        line_edit = dlg.findChild(QLineEdit)
        if line_edit:
            line_edit.setPlaceholderText(str(self.cxt.ho.step))
        
        if dlg.exec():
            value = dlg.doubleValue()
            self.cxt.ho.step = value
            logger.info(f"Button clicked: Downhole resample window changed to {value}")

         
    def show_downhole(self):
        """Display downhole product dataset."""
        logger.info(f"Button clicked: Show downhole data")
        if not self.cxt.ho:
            logger.warning(f"Show data cancelled, no hole loaded")
            return
        key = self.full_data_combo.currentData(Qt.UserRole)
        
        if not key:
            logger.warning(f"Show data cancelled, no key selected")
            return
        if key.endswith("CLUSTERS"):
            logger.info(f"Cluster dataset selected")
            self.show_clusters(key)
            return
        try:
            with busy_cursor("Resampling donwhole dataset...", self):
                depths, values, dominant = self.cxt.ho.step_product_dataset(key)
        except ValueError as e:
            logger.error(f"Failed to resample {gen_display_text(key)}", exc_info=True)
            QMessageBox.warning(self, "Failed operation", f"Failed to resample {gen_display_text(key)}: {e}")
            return
        
        # Create canvas
        canvas = ImageCanvas2D()
        
        # Route based on suffix
        if key.endswith("FRACTIONS"):
            # Display fractions as stacked area
            try:
                legend_key = key.replace("FRACTIONS", "LEGEND")
                legend = self.cxt.ho.product_datasets[legend_key].data
                canvas.display_fractions(depths, values, legend, include_unclassified=True)
                
            except ValueError as e:
                logger.error(f"Failed to plot {gen_display_text(key)}", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to plot: {e}")
                return
        
        elif key.endswith("DOM-MIN"):
            # Display indices as categorical bars (both DOM-MIN and INDEX)
            try:
                legend_key = key.replace("DOM-MIN", "LEGEND")
                legend = self.cxt.ho.product_datasets[legend_key].data
                canvas.display_discrete(depths, dominant, legend)
                
            except ValueError as e:
                logger.error(f"Failed to plot {gen_display_text(key)}", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to plot: {e}")
                return
        
        elif key.endswith("INDEX"):
            # Display indices as categorical bars (both DOM-MIN and INDEX)
            try:
                legend_key = key.replace("INDEX", "LEGEND")
                legend = self.cxt.ho.product_datasets[legend_key].data
                canvas.display_discrete(depths, values, legend)
                
            except ValueError as e:
                logger.error(f"Failed to plot {gen_display_text(key)}", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to plot: {e}")
                return
        else:
            # Display continuous as line plot
            try:
                canvas.display_continuous(depths, values, gen_display_text(key))
                
            except ValueError as e:
                logger.error(f"Failed to plot {gen_display_text(key)}", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to plot: {e}")
                return
        # Add to page with wrapper
        self._page.add_dhole_display(key, canvas)
        logger.info(f"{gen_display_text(key)} displayed")
    
    
    def gen_base_datasets(self):
        logger.info(f"Button clicked: Generate base datasets")
        if not self.cxt.ho:
            logger.info(f"Generate base datasets cancelled because no hole loaded")
            return
        with busy_cursor("Generating downhole base datasets...", self):
            try:
                self.cxt.ho.create_base_datasets()
                logger.info(f"Generated base datasets for {self.cxt.ho.hole_id}")
            except ValueError as e:
                logger.error(f"Failed to create base data", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to create base data: {e}")
                return
        self.update_for_hole()
        return
    
    
# ---------Box-derived level control handlers---------------------------------------------------------    
    def dhole_feats_create(self):
        logger.info(f"Button clicked: Generate downhole features")
        if not self.cxt.ho:
            logger.info(f"Generate downhole features cancelled because no hole loaded")
            return
        keys = set()
        if not self.cxt.ho.boxes:
            logger.info(f"Generate downhole features cancelled because no hole loaded")
            return
        for box in self.cxt.ho:
            keys = keys | box.datasets.keys() | box.temp_datasets.keys()
        
        suffixes = ("POS", "DEP")
        raw_names = [x for x in keys if x.endswith(suffixes)]
        display_to_key = {gen_display_text(k): k for k in raw_names}
        
        choice, ok = QInputDialog.getItem(
            self,
            "Select Feature",
            "Features:",
            sorted(display_to_key.keys()),
            0,
            False
        )
        if not choice:
            logger.info(f"Generate downhole features cancelled in dialogue")
            return
        try:
            with busy_cursor("Creating donwhole feature dataset...", self):
                raw_key = display_to_key[choice]
                self.cxt.ho.create_dhole_features(raw_key)
        except ValueError as e:
            logger.error(f"Failed to create downhole features", exc_info=True)
            QMessageBox.warning(self, "Failed operation", f"Failed to create downhole feature: {e}")
            return
        logger.info(f"Generated downhole features for {self.cxt.ho.hole_id} {raw_key}")
        self.update_for_hole()   
        return
    
    def dhole_minmaps_create(self):
        logger.info(f"Button clicked: Generate downhole mineral maps")
        if not self.cxt.ho:
            logger.info(f"Generate downhole mineral maps cancelled because no hole loaded")
            return
        keys = set()
        if not self.cxt.ho.boxes:
            logger.info(f"Generate downhole mineral maps cancelled because no hole loaded")
            return
        for box in self.cxt.ho:
            keys = keys | box.datasets.keys() | box.temp_datasets.keys()
        
        suffixes = ("INDEX",)
        raw_names = [x for x in keys if x.endswith(suffixes)]
        display_to_key = {gen_display_text(k): k for k in raw_names}
        
        choice, ok = QInputDialog.getItem(self, "Select Mineral Map", "MinMaps:", sorted(display_to_key.keys()), 0, False)
        if not choice:
            logger.info(f"Generate downhole mineral maps cancelled in dialogue")
            return
        try:
            with busy_cursor("Creating donwhole minmap dataset...", self):
                key = display_to_key[choice]
                self.cxt.ho.create_dhole_minmap(key)
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to create downhole mineral maps", exc_info=True)
            QMessageBox.warning(self, "Failed operation", f"Failed to create downhole mineral maps: {e}")
            return
        logger.info(f"Generated downhole features for {self.cxt.ho.hole_id} {key}")
        self.update_for_hole()   
        return
# ---------Box-derived level control handlers---------------------------------------------------------  
    def prof_kmeans(self):
        logger.info(f"Button clicked: Full hole k-means")
        if not self.cxt.ho:
            logger.info(f"Full hole k-means cancelled because no hole loaded")
            return
        clusters, ok1 = QInputDialog.getInt(self, "KMeans Clustering",
            "Enter number of clusters:",value=5, min=1, max=50)
        if not ok1:
            logger.info(f"Full hole k-means cancelled because no n value selected")
            return
        iters, ok2 = QInputDialog.getInt(self, "KMeans Clustering",
            "Enter number of iterations:", value=50, min=1, max=1000)
        if not ok2:
            logger.info(f"Full hole k-means cancelled because no iterations selected")
            return
    
        with busy_cursor('clustering...', self):
            try:
                pt.profile_kmeans(self.cxt.ho, clusters, iters)
            except ValueError as e:
                logger.error(f"Failed to create downhole clustering", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to create downhole clustering: {e}")
                return
        logger.info(f"Full hole k-means ({clusters}, {iters}) for {self.cxt.ho.hole_id} created")
        self.update_for_hole()   
        
    def prof_feature_extraction(self, key):
        logger.info(f"Button clicked: Full hole feature extraction")
        if not self.cxt.ho:
            logger.info(f"Full hole feature extraction cancelled because no hole loaded")
            return
        key = self.feature_combo.currentText().strip()
        if not key:
            logger.info(f"Full hole feature extraction cancelled because failed to find key")
            return
        logger.info(f"Full hole feature extraction using: {key}")
        try:
            with busy_cursor(f'extracting {key}...', self):
                pt.run_feature_extraction(self.cxt.ho, key)
        except ValueError as e:
            logger.error(f"Failed to extract full hole feature extraction", exc_info=True)
            QMessageBox.warning(self, "Failed operation", f"Failed to create perform feature extraction: {e}")
            return
        logger.info(f"Full hole feature {key} created")
        self.update_for_hole() 
    
    def show_clusters(self, key):
        print("cluster window not implemented for profile k-means yet")
    
    def profile_band_maths(self):
        """
        - ask user for a band-maths expression + name
        - pass them, along with the current object, to the interface layer
        """
        logger.info(f"Button clicked: Full hole Band Maths")
        if self.cxt.ho is None:
            logger.info(f"Full hole feature Band Maths cancelled because no hole loaded")
            QMessageBox.information(self, "Band Maths", "No Hole loaded")
            return
           
        ok, name, expr, cr = BandMathsDialog.get_expression(
           parent=self,
           default_name="Custom band index",
           default_expr="2300-1400",
        )
        if not ok:
            logger.info("Band maths operation cancelled from dialogue")
            return
        with busy_cursor('calculating...', self):
            try:
                pt.band_math_interface(self.cxt.ho, name, expr, cr=cr)
            except Exception as e:
                logger.error(f"Band maths operation using {expr} for {self.cxt.ho.hole_id} evaluated on CR = {cr} has failed", exc_info=True)
                QMessageBox.warning(self, "Failed operation", f"Failed to evalute expression: {e}")
                return
        logger.info(f"Band maths operation using {expr} for {self.cxt.ho.hole_id} is done. Evaluated on CR = {cr}")
        self.update_for_hole()
        
    
class HolePage(BasePage):
    """
    The main view for interacting with a multi-box 'HoleObject'.

    Layout:
    - Left: A Control Panel for metadata and dataset operations.
    - Main Area: A collection of vertical tables ('strips'), starting with a default 
      Box/Thumbnail table. Users can add additional strips to compare datasets side-by-side.

    Functionality:
    - Synchronized scrolling across all open strip tables.
    - Selection handling: Clicking a box updates the global context (CurrentContext) 
      to that specific ProcessedObject.
    - Downhole Visualization: Supports generating and displaying downhole plots and 
      mineral maps in pop-out capable widgets.
    - Data Management: Loading hole directories and saving changes (committing temporary datasets).
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
    
    def add_dhole_display(self, key, canvas):
        
        disp = gen_display_text(key)
        wrapper = self._add_closable_widget(
            canvas,
            title=f"Downhole: {disp}",
            popoutable = True
        )
        wrapper.popout_requested.connect(self._handle_popout_request)
              

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
        logger.info("Button clicked: Open Hole Dataset")
        path = QFileDialog.getExistingDirectory(
                   self,
                   "Select hole directory of processed data",
                   "",
                   QFileDialog.ShowDirsOnly
                   )
        if not path:
            logger.info("Opening cancelled in dialogue")
            return
        with busy_cursor('loading...', self):
            hole = HoleObject.build_from_parent_dir(path)
            self.set_hole(hole)
        logger.info(f"Opened {self.cxt.ho.hole_id}")

    def set_hole(self, hole):
        """Set the HoleObject and repopulate the left table."""
        self.cxt.ho = hole
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
        logger.info("Button clicked: dbl click on box in hole page")
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
        logger.info(f"{self.cxt.po.basename} displayed in vis page")
        

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
        logger.info("Button clicked: Save changes (hole page)")
        if self.cxt is not None and self.cxt.ho is not None:
            with busy_cursor('Saving.....', self):
                self.cxt.ho.save_product_datasets()
                for po in self.cxt.ho:
                    if po.has_temps:
                        po.commit_temps()
                        po.save_all()
                        po.reload_all()
                        po.load_thumbs()
                        logger.info(f"{po.basename} saved")
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
