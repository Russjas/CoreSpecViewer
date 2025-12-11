"""
Library page for working with spectral exemplar collections.

Supports importing, viewing, renaming, and exporting spectral libraries
used for correlation and MSAM/WTA classification.
"""
import os
import sqlite3

import numpy as np
from PyQt5.QtCore import QModelIndex
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel
from PyQt5.QtWidgets import QComboBox, QFileDialog, QInputDialog, QMessageBox, QPushButton, QTableView, QToolBar

from ..interface import tools as t
from .base_page import BasePage
from .util_windows import IdSetFilterProxy, ImageCanvas2D, SpectrumWindow, busy_cursor, RightClick_Table

# In the 'samples' table (which is displayed in the QTableView):
ID_COLUMN_INDEX = 0   # Column containing SampleID (used for the lookup)
NAME_COLUMN_INDEX = 1 # Column containing Name (used for the plot title)

# In the 'spectra' table (where the BLOB data is stored):
SAMPLE_TABLE_NAME = "Samples"
SPECTRA_TABLE_NAME = "Spectra"
WAVELENGTH_BLOB_COL = "XData"
REFLECTANCE_BLOB_COL = "YData"
# CORRECTED: NumPy dtype for 4-byte (32-bit) little-endian float.
BLOB_DTYPE = '<f4'


# --- Main Viewer Class ---

class LibraryPage(BasePage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CoreSpecViewer")
        self.setGeometry(100, 100, 1000, 600)

        # State
        self.spec_win = None
        self.spec_win_cr = None
        
# ===== UI set up==============================================================
        header = QToolBar("Database / Collection", self)
        self.layout().insertWidget(0, header)   # put header above the splitter

        btn_open  = QPushButton("Open DB…", self); header.addWidget(btn_open)
        btn_filter = QPushButton("Filter to current band range", self);header.addWidget(btn_filter)
        btn_filter.setToolTip("Show only spectra that fully cover the current object's band range")
        btn_new = QPushButton("Create blank DB…", self); header.addWidget(btn_new)

        btn_add   = QPushButton("Add Selected → Collection", self); header.addWidget(btn_add)
        btn_save  = QPushButton("Save Collection as DB…", self); header.addWidget(btn_save)
        btn_clear = QPushButton("Clear Collection", self); header.addWidget(btn_clear)
        btn_correlate = QPushButton("Correlate", self);header.addWidget(btn_correlate)
        btn_delete = QPushButton("Delete Selected", self)
        header.addWidget(btn_delete)
        btn_delete.clicked.connect(self.delete_selected)
        btn_filter.clicked.connect(self.filter_to_current_bands)
        btn_open.clicked.connect(self.open_database_dialog)
        btn_new.clicked.connect(self.create_blank)
        btn_add.clicked.connect(self.add_selected_to_collection)
        btn_save.clicked.connect(lambda: self.save_collection_as_db(None))
        btn_clear.clicked.connect(self.clear_collection)
        btn_correlate.clicked.connect(self.correlate)
        self.view_selector = QComboBox(self); header.addWidget(self.view_selector)
        self.view_selector.currentTextChanged.connect(self._on_view_changed)

 # Left pane = the table
        self.table_view = RightClick_Table(self)
        self.table_view.setSortingEnabled(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.ExtendedSelection)
        self.table_view.doubleClicked.connect(self.handle_double_click)
        self.table_view.rightClicked.connect(self.handle_right_click)
        self.table_view.setSearchColumn(NAME_COLUMN_INDEX)
        
        self._add_left(self.table_view)
        
        self._proxy = IdSetFilterProxy(ID_COLUMN_INDEX, parent=self)
        self.table_view.setModel(self._proxy)

        db_path = self._find_default_database()
        
        if db_path is not None and os.path.exists(db_path):
            self.load_db(db_path)
            
        else:
            QMessageBox.information(self, "Open a database",
                                    "No default database found. Click 'Open DB…' to select a file.")

    
    def _find_default_database(self, search_dir: str = '.', pattern: str = ".db") -> str:

        """
        Return the first matching .db file found under search_dir (recursive).
        Keep this pure (no UI) so it's easy to test.
        """
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.lower().endswith(pattern):
                    self.load_db(os.path.join(root, f))
                    return os.path.join(root, f)
                    
        # Fall back to current working directory as a secondary search
        for root, _, files in os.walk(os.getcwd()):
            for f in files:
                if f.lower().endswith(pattern):
                    self.load_db(os.path.join(root, f))
                    return os.path.join(root, f)
                    
        return None
    
    def _refresh_collection_selector(self):
        """Refresh the 'Show:' selector entries."""
        sel = self.view_selector
        if sel is None:
            return
        current = sel.currentText()
        sel.blockSignals(True)
        sel.clear()
        sel.addItem("Opened database")
        for name in sorted(self.cxt.library.collections.keys()):
            sel.addItem(name)
        # try to preserve previous choice if still valid
        idx = sel.findText(current)
        sel.setCurrentIndex(idx if idx >= 0 else 0)
        sel.blockSignals(False)
    
    
    def create_blank(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Create New SQLite DB", "", "SQLite DB (*.db);;All Files (*)"
        )
        if not path:
            return
        
        # Optionally ensure .db extension
        if not path.endswith('.db'):
            path += '.db'
        
        try:
            self.cxt.library.new_db(path)
        except Exception as e:
            QMessageBox.critical(self, "Database Error",
                                 f"Failed to create new blank database:\n{e}")
            return
        self.load_db(path)
    
    
    
    def open_database_dialog(self):
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SQLite DB", "", "SQLite DB (*.db);;All Files (*)"
        )
        if not path:
            return
        self.load_db(path)
        
    def load_db(self, path: str):
        """
        (Re)open a SQLite DB file and bind the QSqlTableModel/QTableView to it.
        """
        try: 
            self.cxt.library.open_database(path)
        except Exception as e:
            QMessageBox.critical(self, "Database Error",
                                 f"bad load {e}")
            return
        
        self._proxy.setSourceModel(self.cxt.library.model)
        self.table_view.resizeColumnsToContents()
        self.setWindowTitle(f"PyQt5 SQLite Viewer: {os.path.basename(path)}")
        self._refresh_collection_selector()
        if self.spec_win:
            self.spec_win.close()
            self.spec_win = None
        if self.spec_win_cr:
            self.spec_win_cr.close()
            self.spec_win_cr = None
            

    def delete_selected(self):
        """Delete selected samples from the database."""
        ids = self._selected_sample_ids()
        if not ids:
            QMessageBox.information(self, "No Selection", "Select one or more rows to delete.")
            return
        
        reply = QMessageBox.question(
            self, 
            "Confirm Delete",
            f"Delete {len(ids)} sample(s)? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        failed = []
        for sample_id in ids:
            try:
                self.cxt.library.delete_sample(sample_id)
            except Exception as e:
                failed.append((sample_id, str(e)))
        
        if failed:
            msg = "Failed to delete:\n" + "\n".join(f"ID {sid}: {err}" for sid, err in failed)
            QMessageBox.warning(self, "Delete Errors", msg)
        else:
            QMessageBox.information(self, "Deleted", f"Successfully deleted {len(ids)} sample(s).")
    

    def handle_double_click(self, index: QModelIndex):
        """
        Handles the double-click event. Retrieves SampleID and Name.
        """

        if index.isValid():
            m = self.table_view.model()
            # 1. Get the SampleID (key for the spectra table)
            id_index = m.index(index.row(), ID_COLUMN_INDEX)
            sample_id = m.data(id_index)

            # 2. Get the Name for the plot title
            name_index = m.index(index.row(), NAME_COLUMN_INDEX)
            item_name = m.data(name_index)

            if sample_id is None:
                QMessageBox.warning(self, "Error", "Could not retrieve SampleID.")
                return

            self.display_spectra(sample_id, item_name)
            
            
    def handle_right_click(self, index):
        if index.isValid():
            m = self.table_view.model()
            # 1. Get the SampleID (key for the spectra table)
            id_index = m.index(index.row(), ID_COLUMN_INDEX)
            sample_id = m.data(id_index)

            # 2. Get the Name for the plot title
            name_index = m.index(index.row(), NAME_COLUMN_INDEX)
            item_name = m.data(name_index)

            if sample_id is None:
                QMessageBox.warning(self, "Error", "Could not retrieve SampleID.")
                return

            self.display_spectra(sample_id, item_name, hull_rem=True)
            
            
    def display_spectra(self, sample_id, item_name, hull_rem = False):
        """Queries the spectra table, unpacks BLOBs using the correct dtype, and launches the plot window."""
        lib = self.cxt.library
        if lib is None:
            return

        try:
            x_nm, y = lib.get_spectrum(sample_id)
        except Exception as e:
            QMessageBox.critical(
                self,
                "SQL Query Error",
                f"Failed to fetch spectrum for ID {sample_id}:\n{e}",
            )
            return

        # Launch the plot window
        if hull_rem:
            title = f"CR Spectra for: {item_name} (ID: {sample_id})"
            if self.spec_win_cr is None:
                self.spec_win_cr = SpectrumWindow(self)

            self.spec_win_cr.plot_spectrum(x_nm, t.get_cr(y), title)
            self.spec_win_cr.ax.set_ylabel("CR Reflectance (Unitless)")
        else:
            title = f"Spectra for: {item_name} (ID: {sample_id})"
            if self.spec_win is None:
                self.spec_win = SpectrumWindow(self)
    
            self.spec_win.plot_spectrum(x_nm*1000, y, title)
            
            
    def _selected_sample_ids(self):
        """Return list of SampleID values from selected rows."""
        m = self.table_view.model()
        if m is None:
            return []
        sel = self.table_view.selectionModel()
        if sel is None:
            return []

        # Selected rows -> pull SampleID using ID_COLUMN_INDEX
        rows = {ix.row() for ix in sel.selectedRows(ID_COLUMN_INDEX)} or {ix.row() for ix in sel.selectedRows()}
        ids = []
        for r in rows:
            idx = m.index(r, ID_COLUMN_INDEX)
            val = m.data(idx)
            try:
                ids.append(int(val))
            except (TypeError, ValueError):
                pass
        return ids

#TODO: Think about this. This needs to be duplicated in main?
    def _choose_existing_collection(self, title="Select collection"):
        names = sorted(self.cxt.library.collections.keys())
        if not names:
            QMessageBox.information(self, "No collections", "Create a collection first via 'Add Selected → Collection'.")
            return None
        if len(names) == 1:
            return names[0]
        name, ok = QInputDialog.getItem(self, title, "Collections:", names, 0, False)
        return name if ok else None

    def _prompt_collection_name(self, title="Collection name", allow_new=True):
        names = sorted(self.cxt.library.collections.keys())
        if names and allow_new:
            # Let user pick existing or type new
            name, ok = QInputDialog.getItem(self, title, "Choose collection (or type a new name):",
                                            names, 0, True)  # editable combo
        else:
            name, ok = QInputDialog.getText(self, title, "Enter collection name:")
        if not ok or not name.strip():
            return None
        return name.strip()

    def add_selected_to_collection(self):
        ids = self._selected_sample_ids()
        if not ids:
            QMessageBox.information(self, "No Selection", "Select one or more rows first.")
            return
        name = self._prompt_collection_name("Add to collection")
        if not name:
            return
        added, len_coll = self.cxt.library.add_to_collection(name, ids)
        self._refresh_collection_selector()
        QMessageBox.information(self, "Added to Collection",
                                f"Added {added} new items to '{name}'.\nSize now: {len_coll}")
    
    def _on_view_changed(self, text: str):
       if text == "Opened database":
           self._proxy.set_allowed_ids(None)
       elif text in self.cxt.library.collections.keys():
           self._proxy.set_allowed_ids(self.cxt.library.collections.get(text))
       else:
           self._proxy.set_allowed_ids(None)
           
    def filter_to_current_bands(self):
        """
        Build/refresh a special in-memory collection '__filtered__' containing SampleIDs
        whose spectral XData fully covers the current_obj.bands range.
        Applies the proxy filter to show only those rows.
        """
        if self.current_obj is None:
            QMessageBox.critical(self, "No Data", "No processed data loaded to get band range")
            return
            
        # target range: current_obj.bands is in nm → convert to µm to compare with XData
        bands_nm = np.asarray(self.current_obj.bands, dtype=float)
        if bands_nm.size == 0 or np.all(np.isnan(bands_nm)):
            QMessageBox.information(self, "No bands", "Current bands are empty.")
            return
        
        ok_ids = self.cxt.library.filter_ids_covering_bands(bands_nm)

        self.cxt.library.add_to_collection("in range", ok_ids)
        self._refresh_collection_selector()
        self._on_view_changed("in range")

        count = len(ok_ids)
        rng = f"{np.nanmin(bands_nm):.1f}–{np.nanmax(bands_nm):.1f} nm"
        QMessageBox.information(self, "Filtered", f"{count} spectra cover {rng}.")        
            

    def correlate(self):
        if self.current_obj is None:
            QMessageBox.critical(self, "Selection error",
                                 "You do not have a scan dataset loaded")
        if self.current_obj.is_raw:
            QMessageBox.critical(self, "Selection error",
                                 "You cannot correlate on a raw dataset, you must process first")
            return
        ids = self._selected_sample_ids()
        if len(ids) == 0:
            return
        if len(ids) > 1:
            QMessageBox.critical(self, "Selection error",
                                 "Too many spectra selected for single correlation\nCreate a collection for min id from exemplars")
            return

        sample_id = int(ids[0])

        # --- Get the mineral name from the selected row (fallback to DB) ---
        mineral_name = self.cxt.library.get_sample_name(sample_id)
        try:
            x_nm, y = self.cxt.library.get_spectrum(sample_id)
        except Exception as e:
            QMessageBox.critical(
                self,
                "SQL Query Error",
                f"Failed to fetch spectrum for ID {sample_id}:\n{e}",
            )
        corr_canvas = ImageCanvas2D()

        self.corr_wrapper = self._add_closable_widget(
            raw_widget=corr_canvas,
            title="Correlation"
        )
        key = f"{mineral_name}-(ID:-{sample_id})-MINCORR"
        with busy_cursor('correlating...', self):
            _, key = t.quick_corr(self.current_obj, x_nm, y, key = key)
            print(key, self.current_obj.temp_datasets[key].path, self.current_obj.temp_datasets[key].ext)
            
            corr_canvas.show_rgb(self.current_obj.get_data(key))
            corr_canvas.ax.set_title(f"{mineral_name} (ID: {sample_id})", fontsize=11)


    def clear_collection(self):
        name = self._choose_existing_collection("Delete collection")
        if not name:
            return
        if QMessageBox.question(self, "Confirm delete",
                                f"Delete collection '{name}'? This cannot be undone.") != QMessageBox.Yes:
            return
        self.cxt.library.collections.pop(name, None)
        QMessageBox.information(self, "Deleted", f"Collection '{name}' removed.")
        self._refresh_collection_selector()

    def save_collection_as_db(self, key: str | None = None):
        if key is None:
            key = self._choose_existing_collection("Save collection as DB")
        if not key:
            return
        ids_set = self.cxt.library.collections.get(key, set())
        if not ids_set:
            QMessageBox.information(self, "Empty Collection", f"No items in collection '{key}'.")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Save new SQLite DB", "", "SQLite DB (*.db);;All Files (*)")
        if not out_path:
            return
        try:
            self.cxt.library.export_collection_to_db(key, out_path)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save DB:\n{e}")
            return

        QMessageBox.information(self, "Saved",
                                f"New database written to:\n{out_path}\n\n"
                                f"{len(ids_set)} items saved from '{key}'.")

    def update_display(self, key='mask'):
        self._refresh_collection_selector()
        
    def teardown(self):
        # any per-teardown cleanup (close SpectrumWindow, etc.)
        if self.spec_win:
            self.spec_win.close()
            self.spec_win = None
        if self.spec_win_cr:
            self.spec_win_cr.close()
            self.spec_win_cr = None
        super().teardown()



