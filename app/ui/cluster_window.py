"""
Cluster page for examing cluster centres returned from unsupervised clustering methods.

Supports viewing cluster centre spectrum and classifying using correlation techniques.
"""




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



class ClusterWindow(BasePage):
    """
    Standalone window for inspecting k-means cluster centres for a specific
    ProcessedObject.

    - Pinned to a single ProcessedObject (`self.po`) at creation.
    - Does *not* follow `cxt.current` as the user changes box.
    - Uses `cxt.library` (LibraryManager) for correlation.

    Table layout:
        Col 0: Class ID
        Col 1: Pixel count
        Col 2: Best match 1
        Col 3: Score 1
        Col 4: Best match 2
        Col 5: Score 2
        Col 6: Best match 3
        Col 7: Score 3
    """

    def __init__(self, parent=None, cxt=None, po=None, cluster_key: str = ""):
        super().__init__(parent)

        # Context + pinned PO
        if cxt is not None:
            self.cxt = cxt
        self.po = po or getattr(self.cxt, "current", None)

        self.cluster_key: str = cluster_key
        self.index_key: str | None = None
        self.legend_key: str | None = None

        self.centres: np.ndarray | None = None     # (m, B)
        self.pixel_counts: np.ndarray | None = None
        self.matches_msam: dict[int, tuple[int, str, float]] = {} # class_id : (lib_id, min name, confidence)
        self.matches_sam: dict[int, tuple[int, str, float]] = {}
        self.matches_pearson: dict[int, tuple[int, str, float]] = {}

        self._loaded: bool = False  # have we pulled data from self.po yet?

        self.spec_win: SpectrumWindow | None = None

        self._build_ui()

        # If we know the box identity, put it in the title
        box_label = getattr(self.po, "basename", None) or getattr(self.po, "name", "")
        base_title = "Cluster centres"
        if box_label:
            self.setWindowTitle(f"{base_title} – {box_label}")
        else:
            self.setWindowTitle(base_title)

    # ------------------------------------------------------------------ #
    # BasePage lifecycle                                                 #
    # ------------------------------------------------------------------ #
    def activate(self):
        """
        Called by controller just after construction. We deliberately *do not*
        follow self.cxt.current; we stay pinned to self.po.
        """
        super().activate()
        if not self._loaded:
            try:
                self._load_from_po()
                self._populate_table()
                self._loaded = True
            except Exception as e:
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
    # Loading from the pinned PO                                         #
    # ------------------------------------------------------------------ #
    def _load_from_po(self):
        """
        Pull cluster centres and index map from the pinned ProcessedObject (`self.po`).
        """
        po = self.po
        if po is None:
            raise RuntimeError("ClusterWindow.po is None (no ProcessedObject pinned).")

        if not self.cluster_key:
            raise RuntimeError("ClusterWindow.cluster_key is not set.")

        # Derive related keys for this clustering run
        base = self.cluster_key.replace("CLUSTERS", "")
        self.index_key = base + "INDEX"
        #TODO: No legend exists for quick cluster by default. May use this window to populate a LEGEND dataset
        self.legend_key = base + "LEGEND" 
        
        centres = po.get_data(self.cluster_key)
                
        if centres.ndim != 2:
            raise ValueError(
                f"Cluster centres expected to be 2D (m x bands), got shape {centres.shape}"
            )
        self.centres = centres
        try:
            idx = po.get_data(self.index_key)
            self.pixel_counts = t.compute_pixel_counts(idx, centres.shape[0])
            
        except KeyError:
            
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
        if self.centres is None or self.po is None or not self.cxt.library:
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
    
        bands = getattr(self.po, "bands", None)
        if bands is None:
            return
    
        x_nm, y = self.cxt.library.get_spectrum(sample_id)
        display_spectra = t.match_spectra(x_nm, y, bands)
    
        title = f"CR Spectra for: {sample_name} (ID: {sample_id})"
        if self.spec_win is None:
            self.spec_win = SpectrumWindow(self)
    
        self.spec_win.plot_spectrum(bands, t.get_cr(display_spectra), title)
        self.spec_win.ax.set_ylabel("CR Reflectance (Unitless)")


    def _show_cluster_spectrum(self, class_id: int):
        """
        Show the cluster centre spectrum for class cid in a SpectrumWindow.
        If correlation has been run, title includes the best-match label.
        """
        if self.centres is None:
            return
        if class_id < 0 or class_id >= self.centres.shape[0]:
            return

        po = self.po
        if po is None:
            return

        bands = getattr(po, "bands", None)
        y = self.centres[class_id, :]

        if bands is None or np.size(bands) != y.size:
            x = np.arange(y.size)
            x_label = "Band index"
        else:
            x = np.asarray(bands, dtype=float)
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
        if self.centres is None:
            return
        if self.cluster_key is None:
            return
        if self.cxt.current is None:
            return
        for i in range(self.centres.shape[0]):
            try:
                spectrum = self.centres[i, :]
                wavelengths_nm = self.cxt.current.bands
                         
                
                name = f"Class {i}"
                metadata = {}
                metadata['SampleNum'] = f"Class {i} from {self.cluster_key}"
                with busy_cursor('Adding to library...', self):
                    sample_id = self.cxt.library.add_sample(
                        name=name,
                        wavelengths_nm=wavelengths_nm,
                        reflectance=spectrum,
                        metadata=metadata
                    )
                
                
                
            except Exception as e:
                QMessageBox.warning(
                    self, 
                    "Add to Library", 
                    f"Failed to add spectrum to library: {e}"
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
        bands = getattr(self.po, "bands", None)
        if bands is None:
            QMessageBox.warning(self, "No bands", "Pinned ProcessedObject has no 'bands' attribute.")
            return None
    
        return exemp_ids, exemplars, bands
    
    def _run_lib_match(self, fn, match_store: dict[int, tuple[int, str, float]]):
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
        self._run_lib_match(t.wta_min_map_MSAM_direct, self.matches_msam)
    
    def _sam_lib(self):
        self._run_lib_match(t.wta_min_map_SAM_direct, self.matches_sam)
    
    def _pearson_lib(self):
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
