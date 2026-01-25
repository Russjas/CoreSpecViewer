"""
Cluster page for examining cluster centres returned from unsupervised clustering methods.

Supports viewing cluster centre spectrum and classifying using correlation techniques.
Works with both ProcessedObject (box-level) and HoleObject (profile-level) data sources.
"""
import logging

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QMessageBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..interface import tools as t
from .base_page import BasePage
from .util_windows import SpectrumWindow, busy_cursor

logger = logging.getLogger(__name__)

class ClusterWindow(BasePage):
    """
    Standalone window for inspecting k-means cluster centres.
    
    Supports both ProcessedObject (box-level) and HoleObject (profile-level)
    cluster data sources.
    
    Do not instantiate directly - use factory methods:
    - ClusterWindow.from_processed_object() for box-level clusters
    - ClusterWindow.from_hole_object() for profile-level clusters
    
    Table layout:
        Col 0: Class ID
        Col 1: Pixel count
        Col 2: Pearson Match
        Col 3: Pearson confidence
        Col 4: SAM Match
        Col 5: SAM confidence
        Col 6: MSAM Match
        Col 7: MSAM confidence
        Col 8: User match
    """

    def __init__(self, parent=None, cxt=None):
        """
        Private constructor. Use factory methods instead.
        
        Parameters
        ----------
        parent : QWidget, optional
        cxt : Context
            Application context (for library access)
        """
        super().__init__(parent)

        # Context
        if cxt is not None:
            self.cxt = cxt
        
        # Data source attributes (set by factory methods)
        self.data_source = None  # Will be ProcessedObject or HoleObject
        self.data_source_type: str = ""  # "processed" or "hole"
        
        self.cluster_key: str = ""
        self.index_key: str | None = None
        self.legend_key: str | None = None

        # Cluster data
        self.centres: np.ndarray | None = None  # (m, B)
        self.pixel_counts: np.ndarray | None = None
        self.bands: np.ndarray | None = None  # Cached bands
        
        # Match results
        self.matches_msam: dict[int, tuple[int, str, float]] = {}  # class_id : (lib_id, min name, confidence)
        self.matches_sam: dict[int, tuple[int, str, float]] = {}
        self.matches_pearson: dict[int, tuple[int, str, float]] = {}

        self._loaded: bool = False

        self.spec_win: SpectrumWindow | None = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Factory methods                                                    #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_processed_object(cls, parent=None, cxt=None, po=None, cluster_key: str = ""):
        """
        Create a ClusterWindow for box-level cluster data.
        
        Parameters
        ----------
        parent : QWidget, optional
        cxt : Context
        po : ProcessedObject
            Processed box object containing cluster data
        cluster_key : str
            Key for cluster centres dataset (e.g., "kmeans-5-50CLUSTERS")
        
        Returns
        -------
        ClusterWindow
        """
        if po is None:
            logger.warning("No ProcessedObject passed")
            raise ValueError("ProcessedObject cannot be None")
        
        window = cls(parent=parent, cxt=cxt)
        window.data_source = po
        window.data_source_type = "processed"
        window.cluster_key = cluster_key
        
        # Set window title
        box_label = getattr(po, "basename", None) or getattr(po, "name", "")
        base_title = "Cluster centres"
        if box_label:
            window.setWindowTitle(f"{base_title} — {box_label}")
        else:
            window.setWindowTitle(base_title)
        
        return window
    
    @classmethod
    def from_hole_object(cls, parent=None, cxt=None, ho=None, cluster_key: str = ""):
        """
        Create a ClusterWindow for profile-level cluster data.
        
        Parameters
        ----------
        parent : QWidget, optional
        cxt : Context
        ho : HoleObject
            Hole object containing profile cluster data
        cluster_key : str
            Key for cluster centres dataset (e.g., "PROF-kmeans-5-50CLUSTERS")
        
        Returns
        -------
        ClusterWindow
        """
        if ho is None:
            logger.warning("No HoleObject passed")
            raise ValueError("HoleObject cannot be None")
        
        window = cls(parent=parent, cxt=cxt)
        window.data_source = ho
        window.data_source_type = "hole"
        window.cluster_key = cluster_key
        
        # Set window title
        box_label = ho.hole_id
        base_title = "Cluster centres"
        if box_label:
            window.setWindowTitle(f"{base_title} — {box_label}")
        else:
            window.setWindowTitle(base_title)
        
        return window

    # ------------------------------------------------------------------ #
    # BasePage lifecycle                                                 #
    # ------------------------------------------------------------------ #
    def activate(self):
        """
        Called by controller just after construction.
        Loads data from the pinned data source.
        """
        super().activate()
        if not self._loaded:
            try:
                self._load_data()
                self._populate_table()
                self._loaded = True
            except Exception as e:
                logger.error(f"Could not load clusters for key '{self.cluster_key}':\n{e}", exc_info=True)
                QMessageBox.critical(
                    self,
                    "Cluster load error",
                    f"Could not load clusters for key '{self.cluster_key}':\n{e}",
                )

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        """
        Build a compact splitter-left style layout inside this window:
        a table + buttons.
        """
        container = QWidget(self)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)

        # Table
        self.clus_table = QTableView(container)
        self.model = QStandardItemModel(self)
        self.clus_table.setModel(self.model)
        self.clus_table.setSelectionBehavior(QTableView.SelectRows)
        self.clus_table.setSelectionMode(QTableView.SingleSelection)
        self.clus_table.setSortingEnabled(True)
        
        self.clus_table.doubleClicked.connect(self._on_row_double_clicked)

        header = self.clus_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.btn_pearson = QPushButton("Pearson against library", container)
        self.btn_pearson.clicked.connect(self._pearson_lib)
        btn_row.addWidget(self.btn_pearson)
        
        self.btn_sam = QPushButton("SAM against library", container)
        self.btn_sam.clicked.connect(self._sam_lib)
        btn_row.addWidget(self.btn_sam)
        
        self.btn_MSAM = QPushButton("MSAM against library", container)
        self.btn_MSAM.clicked.connect(self._msam_lib)
        btn_row.addWidget(self.btn_MSAM)
        
        self.btn_lib = QPushButton("Add Clusters to library", container)
        self.btn_lib.clicked.connect(self._add_clusters_lib)
        btn_row.addWidget(self.btn_lib)

        vbox.addLayout(btn_row)
        vbox.addWidget(self.clus_table, 1)
        
        # Use BasePage helper to put this on the left side of the splitter
        self._add_left(container)

    # ------------------------------------------------------------------ #
    # Loading data from source                                           #
    # ------------------------------------------------------------------ #
    def _load_data(self):
        """
        Load cluster centres and index map from the data source.
        Handles both ProcessedObject and HoleObject sources.
        """
        if self.data_source is None:
            raise RuntimeError("ClusterWindow has no data source set.")
        
        if not self.cluster_key:
            raise RuntimeError("ClusterWindow.cluster_key is not set.")
        
        # Derive related keys for this clustering run
        base = self.cluster_key.replace("CLUSTERS", "")
        self.index_key = base + "INDEX"
        self.legend_key = base + "LEGEND"
        
        # Load data based on source type
        if self.data_source_type == "processed":
            self._load_from_processed_object()
        elif self.data_source_type == "hole":
            self._load_from_hole_object()
        else:
            raise RuntimeError(f"Unknown data source type: {self.data_source_type}")
    
    def _load_from_processed_object(self):
        """Load cluster data from ProcessedObject."""
        po = self.data_source
        
        # Get cluster centres
        centres = po.get_data(self.cluster_key)
        
        if centres.ndim != 2:
            raise ValueError(
                f"Cluster centres expected to be 2D (m x bands), got shape {centres.shape}"
            )
        self.centres = centres
        
        # Get bands
        self.bands = getattr(po, "bands", None)
        
        # Get pixel counts from index
        try:
            idx = po.get_data(self.index_key)
            self.pixel_counts = t.compute_pixel_counts(idx, centres.shape[0])
        except KeyError:
            m = centres.shape[0]
            self.pixel_counts = np.zeros(m, dtype=int)
    
    def _load_from_hole_object(self):
        """Load cluster data from HoleObject."""
        ho = self.data_source
        
        # Get cluster centres from product_datasets
        centres_dataset = ho.product_datasets.get(self.cluster_key)
        if centres_dataset is None:
            raise KeyError(f"No dataset found for key: {self.cluster_key}")
        
        centres = centres_dataset.data
        
        if centres.ndim != 2:
            raise ValueError(
                f"Cluster centres expected to be 2D (m x bands), got shape {centres.shape}"
            )
        self.centres = centres
        
        # Get bands from first box
        first_box = ho.boxes.get(ho.first_box)
        if first_box is None:
            raise ValueError("No boxes available in HoleObject")
        self.bands = first_box.bands
        
        # Get pixel counts from index
        try:
            index_dataset = ho.product_datasets.get(self.index_key)
            if index_dataset is not None:
                idx = index_dataset.data
                self.pixel_counts = t.compute_pixel_counts(idx, centres.shape[0])
            else:
                m = centres.shape[0]
                self.pixel_counts = np.zeros(m, dtype=int)
        except (KeyError, AttributeError):
            m = centres.shape[0]
            self.pixel_counts = np.zeros(m, dtype=int)

    # ------------------------------------------------------------------ #
    # Table population                                                   #
    # ------------------------------------------------------------------ #
    def _populate_table(self):
        if self.centres is None or self.pixel_counts is None:
            return

        m, _B = self.centres.shape

        self.model.clear()
        self.model.setHorizontalHeaderLabels(
            [
                "Class",
                "Pixels",
                "Pearson Match",
                "Pearson confidence",
                "SAM Match",
                "SAM confidence",
                "MSAM Match",
                "MSAM confidence",
                "User match"
            ]
        )

        for cid in range(m):
            row_items = []

            it_class = QStandardItem(str(cid))
            it_class.setEditable(False)
            row_items.append(it_class)

            it_pix = QStandardItem(str(int(self.pixel_counts[cid])))
            it_pix.setEditable(False)
            row_items.append(it_pix)

            # Best-match columns start empty
            for _ in range(3):
                it_name = QStandardItem("")
                it_name.setEditable(False)
                row_items.append(it_name)
                
                it_score = QStandardItem("")
                it_score.setEditable(False)
                row_items.append(it_score)
            
            it_user = QStandardItem("")
            it_user.setEditable(True)
            row_items.append(it_user)

            self.model.appendRow(row_items)

    # ------------------------------------------------------------------ #
    # Row interactions                                                   #
    # ------------------------------------------------------------------ #
    def _row_to_class_id(self, row: int) -> int:
        idx = self.model.index(row, 0)
        val = self.model.data(idx)
        return int(val)

    def _on_row_double_clicked(self, index):
        if not index.isValid():
            return
    
        row = index.row()
        col = index.column()
        class_id = self._row_to_class_id(row)
    
        if col in (0, 1):
            logger.info(f"Button clicked: Double click display cluster centre class {class_id}")
            self._show_cluster_spectrum(class_id)
            return
    
        metric_by_col = {
            2: "pearson", 3: "pearson",
            4: "sam",     5: "sam",
            6: "msam",    7: "msam",
        }
    
        metric = metric_by_col.get(col)
        if metric is None:
            return
        
        self._show_lib_spec(class_id, metric)

    def _show_lib_spec(self, class_id: int, metric: str):
        """Show library spectrum that matched this cluster class."""
        if self.centres is None or self.bands is None or not self.cxt.library:
            return
        if not (0 <= class_id < self.centres.shape[0]):
            return
    
        metric_map = {
            "pearson": self.matches_pearson,
            "sam": self.matches_sam,
            "msam": self.matches_msam,
        }
        matches = metric_map.get(metric)
        if not matches:
            return
    
        tup = matches.get(class_id)
        if tup is None:
            return
    
        sample_id, sample_name, _score = tup
        if sample_id < 0:
            return
    
        x_nm, y = self.cxt.library.get_spectrum(sample_id)
        display_spectra = t.match_spectra(x_nm, y, self.bands)
    
        title = f"CR Spectra for: {sample_name} (ID: {sample_id})"
        if self.spec_win is None:
            self.spec_win = SpectrumWindow(self)
    
        self.spec_win.plot_spectrum(self.bands, t.get_cr(display_spectra), title)
        self.spec_win.ax.set_ylabel("CR Reflectance (Unitless)")
        logger.info(f"Button clicked: Double click display library match {sample_name}")

    def _show_cluster_spectrum(self, class_id: int):
        """
        Show the cluster centre spectrum for class cid in a SpectrumWindow.
        If correlation has been run, title includes the best-match label.
        """
        if self.centres is None:
            return
        if class_id < 0 or class_id >= self.centres.shape[0]:
            return

        y = self.centres[class_id, :]

        if self.bands is None or np.size(self.bands) != y.size:
            x = np.arange(y.size)
            x_label = "Band index"
        else:
            x = np.asarray(self.bands, dtype=float)
            x_label = "Wavelength (nm)"

        if self.spec_win is None:
            self.spec_win = SpectrumWindow(self)

        title = f"Cluster {class_id} centre"
        self.spec_win.plot_spectrum(x, y, title)
        self.spec_win.ax.set_ylabel("CR Reflectance (Unitless)")
        self.spec_win.ax.set_xlabel(x_label)
        
    # ------------------------------------------------------------------ #
    # Interaction to library                                             #
    # ------------------------------------------------------------------ #
    def _add_clusters_lib(self):
        """Add all cluster centres to the spectral library."""
        logger.info("Button clicked: Add cluster centres to library")
        if self.centres is None:
            return
        if self.cluster_key is None:
            return
        if self.bands is None:
            QMessageBox.warning(
                self,
                "No bands",
                "Data source has no wavelength bands available."
            )
            return
        
        # Check if library is open
        if not self.cxt.library or not self.cxt.library.is_open():
            logger.warning("No library database is open.")
            QMessageBox.warning(
                self,
                "No Library",
                "No library database is open."
            )
            return
        
        num_clusters = self.centres.shape[0]
        success_count = 0
        errors = []
        
        with busy_cursor(f'Adding {num_clusters} cluster centres to library...', self):
            for i in range(num_clusters):
                try:
                    spectrum = self.centres[i, :]
                    wavelengths_nm = self.bands
                            
                    name = f"Class {i}"
                    metadata = {}
                    metadata['SampleNum'] = f"Class {i} from {self.cluster_key}"
                    
                    sample_id = self.cxt.library.add_sample(
                        name=name,
                        wavelengths_nm=wavelengths_nm,
                        reflectance=spectrum,
                        metadata=metadata
                    )
                    success_count += 1
                        
                except Exception as e:
                    errors.append(f"Class {i}: {str(e)}")
        
        # Show summary
        if errors:
            error_msg = "\n".join(errors[:5])  # Show first 5 errors
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            logger.warning(f"Added {success_count}/{num_clusters} cluster centres.\n\nErrors:\n{error_msg}")
            QMessageBox.warning(
                self,
                "Add to Library - Partial Success",
                f"Added {success_count}/{num_clusters} cluster centres.\n\nErrors:\n{error_msg}"
            )
        else:
            logger.info(f"Successfully added all {num_clusters} cluster centre(s) to library.")
            QMessageBox.information(
                self,
                "Add to Library - Success",
                f"Successfully added all {num_clusters} cluster centre(s) to library."
            )
    
    def _select_collection_exemplars(self):
        """Return (exemp_ids, exemplars, bands) or None if user cancels / no data."""
        if not self.cxt.library:
            return None
    
        names = sorted(self.cxt.library.collections.keys())
        if not names:
            QMessageBox.information(
                self,
                "No collections",
                "Create a collection first via 'Add Selected → Collection'.",
            )
            return None
    
        if len(names) == 1:
            name = names[0]
        else:
            name, ok = QInputDialog.getItem(
                self, "Select Collection", "Collections:", names, 0, False
            )
            if not ok:
                return None
    
        if not name:
            return None
    
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            return None
    
        exemp_ids = list(exemplars.keys())
        
        if self.bands is None:
            QMessageBox.warning(self, "No bands", "Data source has no bands available.")
            return None
    
        return exemp_ids, exemplars, self.bands
    
    def _run_lib_match(self, fn, match_store: dict[int, tuple[int, str, float]]):
        """Run a library matching function against cluster centres."""
        if self.centres is None:
            return
    
        sel = self._select_collection_exemplars()
        if sel is None:
            return
        exemp_ids, exemplars, bands = sel
    
        index, score = fn(self.centres, exemplars, bands)
    
        for i, (idx_val, s) in enumerate(zip(index, score)):
            if idx_val < 0:
                match_store[i] = (-999, "No match", s)
            else:
                sample_id = exemp_ids[idx_val]
                name = self.cxt.library.get_sample_name(sample_id)
                match_store[i] = (sample_id, name, s)
    
        self._update_matches_in_table()
    
    def _msam_lib(self):
        """Run MSAM correlation against library."""
        logger.info("Button clicked: MSAM match")
        self._run_lib_match(t.wta_min_map_MSAM_direct, self.matches_msam)
    
    def _sam_lib(self):
        """Run SAM correlation against library."""
        logger.info("Button clicked: SAM match")
        self._run_lib_match(t.wta_min_map_SAM_direct, self.matches_sam)
    
    def _pearson_lib(self):
        """Run Pearson correlation against library."""
        logger.info("Button clicked: Pearson match")
        self._run_lib_match(t.wta_min_map_direct, self.matches_pearson)

    def _update_matches_in_table(self):
        """
        Update the Pearson/SAM/MSAM match columns in the cluster table from:
    
            self.matches_pearson[cid] -> (sample_idx, name, score)
            self.matches_sam[cid]     -> (sample_idx, name, score)
            self.matches_msam[cid]    -> (sample_idx, name, score)
    
        Table columns:
            0 : Class
            1 : Pixels
            2 : Pearson Match
            3 : Pearson confidence
            4 : SAM Match
            5 : SAM confidence
            6 : MSAM Match
            7 : MSAM confidence
            8 : User match   (left untouched here)
        """
        if self.centres is None:
            return
    
        m = self.centres.shape[0]
    
        for row in range(m):
            class_id = self._row_to_class_id(row)
            
            def write_pair(col_name: int, col_score: int, tup):
                """
                tup is either (idx, name, score) or None.
                """
                name_idx = self.model.index(row, col_name)
                score_idx = self.model.index(row, col_score)
    
                if tup is None:
                    # No match for this metric / class
                    self.model.setData(name_idx, "")
                    self.model.setData(score_idx, "")
                    return
    
                idx_val, name, score = tup

                self.model.setData(name_idx, name)
                if score is None:
                    self.model.setData(score_idx, "")
                else:
                    self.model.setData(score_idx, f"{float(score):.3f}")
    
            # Look up tuples for this class ID in each metric dict
            pearson_tup = self.matches_pearson.get(class_id)
            sam_tup     = self.matches_sam.get(class_id)
            msam_tup    = self.matches_msam.get(class_id)
    
            # Pearson → cols 2,3
            write_pair(2, 3, pearson_tup)
            # SAM     → cols 4,5
            write_pair(4, 5, sam_tup)
            # MSAM    → cols 6,7
            write_pair(6, 7, msam_tup)