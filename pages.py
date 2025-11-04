# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:47:46 2025

@author: russj
"""

# pages.py
# Drop-in, embeddable pages that mirror your three windows:
# - RawPage (ProcessRaw-like)
# - MaskPage (MaskingAndDepth-like)
# - VisualisePage (WorkingWindow-like)
import os, sys
import numpy as np
from PyQt5.QtWidgets import (QSplitter, QVBoxLayout, QTableWidgetItem,QTableWidget,
                             QApplication, QWidget, QToolBar, QPushButton, QFileDialog,
                             QTableView, QMessageBox, QDialog, QLabel)
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtSql import QSqlDatabase, QSqlTableModel, QSqlQuery
from util_windows import SpectralImageCanvas, ImageCanvas2D, InfoTable, SpectrumWindow, busy_cursor
from tool_dispatcher import ToolDispatcher
import tools as t
# Optional: your data classes (kept as forward references – pass instances from the controller)
# from objects import RawObject, ProcessedObject


class BasePage(QWidget):
    """
    Common base: holds a QSplitter with left/right/(optional)third widgets,
    a per-page ToolDispatcher, and a safe teardown().
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._splitter = QSplitter(Qt.Horizontal, self)
        self._left = None     # SpectralImageCanvas
        self._right = None    # ImageCanvas2D
        self._third = None    # InfoTable or other QWidget
        self._dispatcher = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._splitter)

        # Data models available to the page (set by controller)
        self.current_obj = None           # RawObject
       

    # --- building helpers ----------------------------------------------------
    def _add_left(self, w: QWidget):
        self._left = w
        self._splitter.addWidget(w)

    def _add_right(self, w: QWidget):
        self._right = w
        self._splitter.addWidget(w)

    def _add_third(self, w: QWidget):
        self._third = w
        self._splitter.addWidget(w)

    # --- lifecycle -----------------------------------------------------------
    def activate(self):
        """
        Called when the page becomes visible/active.
        Recreate dispatcher so tools can (re)bind safely.
        """
        if isinstance(self._left, SpectralImageCanvas):
            self._dispatcher = ToolDispatcher(self._left)
        else:
            self._dispatcher = None

    def teardown(self):
        """
        Must be called on tab switch (or when closing the page).
        Cancels any active tools and disconnects temporary bindings.
        """
        # Rect selector / canvas interactions
        if isinstance(self._left, SpectralImageCanvas):
            # Cancel an active RectangleSelector cleanly
            self._left.cancel_rect_select()
            # Clear any temporary tool callbacks
            if self._dispatcher:
                self._dispatcher.clear()

        # Nothing to explicitly disconnect on ImageCanvas2D/InfoTable by default
    def _add_closable_widget(self, raw_widget: QWidget, title: str):
        """
        Wraps a widget in a ClosableWidgetWrapper and adds it as a *secondary*
        widget to the QSplitter, usually alongside self._right or self._third.
        """
        from util_windows import ClosableWidgetWrapper # Import locally for clean API
        
        wrapper = ClosableWidgetWrapper(raw_widget, title=title, parent=self)
        
        # Connect the wrapper's closed signal to the page's removal handler
        wrapper.closed.connect(self.remove_widget) 
        
        # Add the wrapper to the splitter
        # Note: We do *not* set self._right or self._third here, as this is a secondary view
        self._splitter.addWidget(wrapper)
        
        # Optional: Set the initial size policy for the new widget
        # wrapper.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        # Return the wrapper for external reference, if needed
        return wrapper
        

    def remove_widget(self, w: QWidget):
        """
        Safely remove a widget (which might be the ClosableWidgetWrapper) 
        from the QSplitter and clean up its memory. (Same as previous version)
        """
        """
        Safely remove a widget from the QSplitter and clean up its memory.
        This is crucial for dynamically added widgets like correlation canvases.
        """
        # 1. Find the widget in the splitter (it might be a wrapped item)
        idx = self._splitter.indexOf(w)
        if idx == -1:
            return

        # 2. Remove from layout and disconnect from Python
        w.setParent(None)
        w.deleteLater()
        
        # 3. If the removed widget was one of the three primary slots, clear the reference
        if w is self._left:
            self._left = None
        elif w is self._right:
            self._right = None
        elif w is self._third:
            self._third = None
    # --- accessors for the controller ---------------------------------------
    @property
    def left_canvas(self) -> SpectralImageCanvas:
        return self._left

    @property
    def right_canvas(self) -> ImageCanvas2D:
        return self._right

    @property
    def table(self) -> QWidget:
        return self._third

    @property
    def dispatcher(self) -> ToolDispatcher:
        return self._dispatcher


class RawPage(BasePage):
    """
    Mirrors your ProcessRaw composition:
      left  = SpectralImageCanvas  (reflectance cube view, dbl-click spectra)
      right = ImageCanvas2D        (product/preview)
      third = InfoTable            (cache/status)
    Use ribbon actions to call methods on the active page via the controller.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Build the three-pane layout
        self._add_left(SpectralImageCanvas(self))
        

    def update(self, key = None):
        self.left_canvas.show_rgb(self.current_obj.get_display_reflectance(), self.current_obj.bands)

 
        
class VisualisePage(BasePage):
    """
    Mirrors your WorkingWindow central content:
      left  = SpectralImageCanvas  (master image; dbl-click spectra)
      right = ImageCanvas2D        (derived product)
      third = InfoTable            (cached products table)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._add_left(SpectralImageCanvas(self))
        self._add_right(ImageCanvas2D(self))
        tbl = QTableWidget(0, 1, self)
        tbl.setHorizontalHeaderLabels(["Cached Products"])
        tbl.horizontalHeader().setStretchLastSection(True)
        self._add_third(tbl)
        
        self._HDR_ROLE = Qt.UserRole + 1  # marks header rows
                
        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 5)
        self._splitter.setStretchFactor(2, 2)

        self.cache = set()
        self.table.cellDoubleClicked.connect(self._on_row_activated)
        
        self._mpl_cids = []  # store mpl connection ids if you add them
        self._sync_lock = False
    def activate(self):
        super().activate()
        
        def _sync_now(src_ax, dst_ax):
            if self._sync_lock: return
            self._sync_lock = True
            try:
                dst_ax.set_xlim(src_ax.get_xlim())
                dst_ax.set_ylim(src_ax.get_ylim())
                dst_ax.figure.canvas.draw_idle()
            finally:
                self._sync_lock = False
    
        def _sync_from_event(ev):
            if ev.canvas is self.left_canvas.canvas:
                _sync_now(self.left_canvas.ax, self.right_canvas.ax)
            elif ev.canvas is self.right_canvas.canvas:
                _sync_now(self.right_canvas.ax, self.left_canvas.ax)
    
        # Important: capture all cids and canvases
        for cv in (self.left_canvas.canvas, self.right_canvas.canvas):
            self._bind_mpl(cv, "button_release_event", _sync_from_event)
            self._bind_mpl(cv, "scroll_event",         _sync_from_event)
            self._bind_mpl(cv, "key_release_event",    _sync_from_event)
        
        if self.current_obj is not None and not self.current_obj.is_raw and self.dispatcher:
            def _right_click(y, x):
                spec = self.current_obj.savgol_cr[y, x, :]
                if not hasattr(self, "spec_win"):
                    self.spec_win = SpectrumWindow(self)
                title = "CR Spectrum Viewer"
                self.spec_win.plot_spectrum(self.current_obj.bands, spec, title=title)
            self.dispatcher.set_right_click(_right_click)
            
            self._set_cache()
    def teardown(self):
        super().teardown()
        # Disconnect any mpl events you added in activate()
        if self._mpl_cids:
            for cv, cid in self._mpl_cids:
                try:
                    cv.mpl_disconnect(cid)
                except Exception:
                    pass
            self._mpl_cids.clear()
    
    def update(self, key = 'mask'):
        self.left_canvas.show_rgb(self.current_obj.savgol, self.current_obj.bands)
        self.right_canvas.show_rgb(self.current_obj.get_data(key))
    
    def _on_row_activated(self, row: int, col: int):
        """
        Ignore header rows; on item rows, open the product on the right canvas.
        """
        it = self.table.item(row, 0)
        if not it:
            return
        if it.data(self._HDR_ROLE):
            # header row → do nothing
            return
    
        key = it.text().strip()
        if not key:
            return
    
        # Defensive: show whatever the dataset is (2D, RGB, or false-colour from 3D+)
        try:
            data = self.current_obj.get_data(key)
        except Exception:
            return

        self.right_canvas.show_rgb(data)
    def add_to_cache(self, key: str):
        if not key:
            return
        
        # single source of truth
        self.cache.add(str(key))
        # build grouped view from cache
        self.refresh_cache_table()
    
    def remove_product(self, key: str):
        if key in self.cache:
            self.cache.discard(key)
            self.refresh_cache_table()
    
    
    
    def refresh_cache_table(self):
        """
        Rebuild the Cached Products table grouped into:
          - Base processed
          - Unwrapped
          - Products
        Uses existing self.cache contents; no changes to add_to_cache/remove_product needed.
        """
        # --- categorize keys from current cache ---
        base_whitelist = {"savgol", "savgol_cr", "mask", "segments", "stats", "bands", "cropped"}  # tweak as you like
        unwrap_prefixes = ("Dhole",)  # DholeAverage, DholeMask, DholeDepths
    
        base = []
        unwrapped = []
        products = []
        if self.current_obj is not None and not self.current_obj.is_raw:
            try:
                table_title = f'{self.current_obj.metadata["borehole id"]} {self.current_obj.metadata["box number"]}'
            except KeyError:
                table_tite = 'Cached products'
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem(table_title))
    
        for k in sorted(self.cache):  # stable order
            if k in base_whitelist:
                base.append(k)
            elif any(k.startswith(pfx) for pfx in unwrap_prefixes):
                unwrapped.append(k)
            else:
                products.append(k)
    
        # --- helper creators ---
        def _insert_header(text: str):
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(text)
            # make header visually distinct and inert
            it.setFlags(Qt.NoItemFlags)
            f = it.font(); f.setBold(True); it.setFont(f)
            self.table.setItem(r, 0, it)
    
        def _insert_item(text: str):
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(text)
            it.setTextAlignment(Qt.AlignCenter)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)
    
        # --- rebuild table ---
        self.table.setRowCount(0)
    
        if base:
            _insert_header("Base processed")
            for k in base:
                _insert_item(k)
    
        if unwrapped:
            _insert_header("Unwrapped")
            for k in unwrapped:
                _insert_item(k)
    
        if products:
            _insert_header("Products")
            for k in products:
                _insert_item(k)
    
        self.table.resizeRowsToContents()
            
    def _set_cache(self):
        keys = set(self.current_obj.datasets.keys()) | set(self.current_obj.temp_datasets.keys())
        for key in keys:
            try:
                data = self.current_obj.get_data(key)  # only ndarray keys succeed
            except KeyError:
                continue  
            
            if hasattr(data, "shape") and getattr(data, "shape", ())[0] > 1:
                self.cache.add(key)
        self.refresh_cache_table()
        
    def _bind_mpl(self, canvas, event, handler):
        cid = canvas.mpl_connect(event, handler)
        self._mpl_cids.append((canvas, cid))
        return cid
    
import sqlite3
# --- Configuration Based on Schema Inspection ---
DEFAULT_DATABASE_FILE = "models/minerals_ecostress.db"

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
        self.setWindowTitle("PyQt5 SQLite Viewer")
        self.setGeometry(100, 100, 1000, 600)
        self.current_obj = None
        # State
        self.spec_win = None
        self.db_path = None
        self.db = QSqlDatabase()   # placeholder; real one is created in open_database()
        self.model = None
        self.collection_ids = set()
        self.exemplars = {}
    
# =============================================================================
        header = QToolBar("Database / Collection", self)
        self.layout().insertWidget(0, header)   # put header above the splitter
         
        btn_open  = QPushButton("Open DB…", self); header.addWidget(btn_open)
        btn_add   = QPushButton("Add Selected → Collection", self); header.addWidget(btn_add)
        btn_save  = QPushButton("Save Collection as DB…", self); header.addWidget(btn_save)
        btn_clear = QPushButton("Clear Collection", self); header.addWidget(btn_clear)
        btn_exemplars = QPushButton("Select Exemplars", self); header.addWidget(btn_exemplars)
        btn_correlate = QPushButton("Correlate", self);header.addWidget(btn_correlate)
         
        btn_open.clicked.connect(self.open_database_dialog)
        btn_add.clicked.connect(self.add_selected_to_collection)
        btn_save.clicked.connect(self.save_collection_as_db)
        btn_clear.clicked.connect(self.clear_collection)
        btn_exemplars.clicked.connect(self.get_collection_exemplars)
        btn_correlate.clicked.connect(self.correlate)

        # Left pane = the table
        self.table_view = QTableView(self)
        self.table_view.setSortingEnabled(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.ExtendedSelection)
        self.table_view.doubleClicked.connect(self.handle_double_click)
        self._add_left(self.table_view)

        if os.path.exists(DEFAULT_DATABASE_FILE):
            self.open_database(DEFAULT_DATABASE_FILE)
        else:
            QMessageBox.information(self, "Open a database",
                                    "No default database found. Click 'Open DB…' to select a file.")
        
    def handle_double_click(self, index: QModelIndex):
        """
        Handles the double-click event. Retrieves SampleID and Name.
        """
        if index.isValid():
            # 1. Get the SampleID (key for the spectra table)
            id_index = self.model.index(index.row(), ID_COLUMN_INDEX)
            sample_id = self.model.data(id_index)
            
            # 2. Get the Name for the plot title
            name_index = self.model.index(index.row(), NAME_COLUMN_INDEX)
            item_name = self.model.data(name_index)
            
            if sample_id is None:
                QMessageBox.warning(self, "Error", "Could not retrieve SampleID.")
                return

            self.display_spectra(sample_id, item_name)
            
    def correlate(self):
        if self.current_obj is None:
            return
        ids = self._selected_sample_ids()
        if len(ids) == 0:
            return
        if len(ids) > 1:
            QMessageBox.critical(self, "Selection error", 
                                 f"Too many spectra selected for single correlation\nCreate a collection for min id from exemplars")
            return
        
        sample_id = (ids[0])
        
        # --- Get the mineral name from the selected row (fallback to DB) ---
        mineral_name = "Unknown"
        sel = self.table_view.selectionModel()
        if sel is not None:
            # Prefer rows selected in the ID column; otherwise any selected row
            rows = {ix.row() for ix in sel.selectedRows(ID_COLUMN_INDEX)} or \
                   {ix.row() for ix in sel.selectedRows()}
            if rows:
                r = next(iter(rows))
                name_idx = self.model.index(r, NAME_COLUMN_INDEX)
                val = self.model.data(name_idx)
                if val:
                    mineral_name = str(val)
         
        if mineral_name == "Unknown":
            # Fallback: query Samples table for the name
            q_name = QSqlQuery(self.db)
            q_name.prepare(f"SELECT Name FROM {SAMPLE_TABLE_NAME} WHERE SampleID = ?")
            q_name.addBindValue(sample_id)
            if q_name.exec_() and q_name.next():
                n = q_name.value(0)
                if n:
                    mineral_name = str(n)
        
        
        
        query = QSqlQuery(self.db)
        sql = (f"SELECT {WAVELENGTH_BLOB_COL}, {REFLECTANCE_BLOB_COL} "
               f"FROM {SPECTRA_TABLE_NAME} WHERE SampleID = {sample_id};")
        if not query.exec_(sql):
            QMessageBox.critical(self, "SQL Query Error", 
                                 f"Failed to query spectra for ID {sample_id}.\n\nError: {query.lastError().text()}")
            return
        if query.next():
            # Retrieve BLOBs as QByteArray/bytearray
            x_data_bytes = query.value(0)
            y_data_bytes = query.value(1)
            
            if x_data_bytes is None or y_data_bytes is None:
                QMessageBox.information(self, "No Data", 
                                       f"XData or YData BLOBs are NULL for SampleID {sample_id}.")
                return
            try:
                x_data = np.frombuffer(x_data_bytes, dtype=BLOB_DTYPE)
                y_data = np.frombuffer(y_data_bytes, dtype=BLOB_DTYPE)
            except ValueError as e:
                QMessageBox.critical(self, "Data Unpacking Error", 
                                     f"Could not unpack BLOB data. Check the data type '{BLOB_DTYPE}'.\nError: {e}")
                return
            except Exception as e:
                QMessageBox.critical(self, "Processing Error", f"An unexpected error occurred: {e}")
                return
        
        
        # Add a new ImageCanvas2D for correlation
        corr_canvas = ImageCanvas2D()
        
        # USE NEW FLEXIBLE METHOD:
        # 1. Use the new helper to create, wrap, connect, and add the widget.
        self.corr_wrapper = self._add_closable_widget(
            raw_widget=corr_canvas, 
            title=f"Correlation"
        )
        
        # 2. Display the data using the *original* canvas reference
        # corr_canvas.show_rgb(...)
        #self.corr = ImageCanvas2D()
       # self._add_right(self.corr)
        #self.right_canvas.show_rgb(t.quick_corr(self.current_obj, x_data*1000, y_data))
        with busy_cursor('correlating...', self):
            corr_canvas.show_rgb(t.quick_corr(self.current_obj, x_data*1000, y_data))
            
            corr_canvas.ax.set_title(f"{mineral_name} (ID: {sample_id})", fontsize=11)
        
            
    def display_spectra(self, sample_id, item_name):
        """Queries the spectra table, unpacks BLOBs using the correct dtype, and launches the plot window."""
        query = QSqlQuery(self.db)
        sql = (f"SELECT {WAVELENGTH_BLOB_COL}, {REFLECTANCE_BLOB_COL} "
               f"FROM {SPECTRA_TABLE_NAME} WHERE SampleID = {sample_id};")
        
        if not query.exec_(sql):
            QMessageBox.critical(self, "SQL Query Error", 
                                 f"Failed to query spectra for ID {sample_id}.\n\nError: {query.lastError().text()}")
            return

        if query.next():
            # Retrieve BLOBs as QByteArray/bytearray
            x_data_bytes = query.value(0)
            y_data_bytes = query.value(1)
            
            if x_data_bytes is None or y_data_bytes is None:
                QMessageBox.information(self, "No Data", 
                                       f"XData or YData BLOBs are NULL for SampleID {sample_id}.")
                return

            try:
                x_data = np.frombuffer(x_data_bytes, dtype=BLOB_DTYPE)
                y_data = np.frombuffer(y_data_bytes, dtype=BLOB_DTYPE)
                
            except ValueError as e:
                QMessageBox.critical(self, "Data Unpacking Error", 
                                     f"Could not unpack BLOB data. Check the data type '{BLOB_DTYPE}'.\nError: {e}")
                return
            except Exception as e:
                QMessageBox.critical(self, "Processing Error", f"An unexpected error occurred: {e}")
                return

        else:
            QMessageBox.information(self, "No Spectra Found", 
                                   f"No matching spectra found in the '{SPECTRA_TABLE_NAME}' table for SampleID {sample_id}.")
            return
        
        # Launch the plot window
        title = f"Spectra for: {item_name} (ID: {sample_id})"
        if self.spec_win is None:
            self.spec_win = SpectrumWindow(self)
        f"Spectra for: {item_name} (ID: {sample_id})"
        self.spec_win.plot_spectrum(x_data*1000, y_data, title)
     
    def _selected_sample_ids(self):
        """Return list of SampleID values from selected rows."""
        if self.model is None:
            return []
        sel = self.table_view.selectionModel()
        if sel is None:
            return []
    
        # Selected rows -> pull SampleID using ID_COLUMN_INDEX
        rows = {ix.row() for ix in sel.selectedRows(ID_COLUMN_INDEX)} or {ix.row() for ix in sel.selectedRows()}
        ids = []
        for r in rows:
            idx = self.model.index(r, ID_COLUMN_INDEX)
            val = self.model.data(idx)
            if val is not None:
                ids.append(val)
        return ids
    
    def add_selected_to_collection(self):
        ids = self._selected_sample_ids()
        if not ids:
            QMessageBox.information(self, "No Selection", "Select one or more rows first.")
            return
        before = len(self.collection_ids)
        self.collection_ids.update(ids)
        added = len(self.collection_ids) - before
        QMessageBox.information(self, "Added to Collection",
                                f"Added {added} new items.\nCollection size: {len(self.collection_ids)}")
    
    def clear_collection(self):
        self.collection_ids.clear()
        QMessageBox.information(self, "Cleared", "Collection is now empty.")   
     
    def save_collection_as_db(self):
        if not self.collection_ids:
            QMessageBox.information(self, "Empty Collection", "No items in the collection to save.")
            return
    
        out_path, _ = QFileDialog.getSaveFileName(self, "Save new SQLite DB", "", "SQLite DB (*.db);;All Files (*)")
        if not out_path:
            return
        src_path = self.db_path or DEFAULT_DATABASE_FILE
        try:
            self._export_subset_db(
                src_path=src_path,
                dst_path=out_path,
                sample_ids=list(self.collection_ids),
                samples_table= SAMPLE_TABLE_NAME,
                spectra_table=SPECTRA_TABLE_NAME,  # "spectra"
                spectra_fk_col="SampleID",         # FK/Join column in spectra
                samples_pk_col=self._column_name_by_index(SAMPLE_TABLE_NAME, ID_COLUMN_INDEX)  # usually "SampleID"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save DB:\n{e}")
            return
    
        QMessageBox.information(self, "Saved",
                                f"New database written to:\n{out_path}\n\n"
                                f"{len(self.collection_ids)} items saved.")
    
    
    def _column_name_by_index(self, table_name: str, col_index: int) -> str:
        """
        Resolve a column name from the model header (safer than hardcoding).
        Falls back to PRAGMA if needed.
        """
        # Try the Qt model header first (works for visible columns in self.model)
        if table_name == SAMPLE_TABLE_NAME:
            name = self.model.headerData(col_index, Qt.Horizontal)
            if name:
                return str(name)
    
        # Fallback: PRAGMA
        q = QSqlQuery(self.db)
        if not q.exec_(f"PRAGMA table_info({table_name});"):
            raise RuntimeError(q.lastError().text())
        cols = []
        while q.next():
            cols.append(q.value(1))  # name
        if col_index < 0 or col_index >= len(cols):
            raise IndexError(f"Column index {col_index} out of range for {table_name}")
        return cols[col_index]
    
    
    def _export_subset_db(self, src_path, dst_path, sample_ids,
                          samples_table, spectra_table, spectra_fk_col, samples_pk_col):
        """
        Create a new SQLite DB with identical schema, then insert subset rows:
        - All rows from `samples_table` where samples_pk_col IN sample_ids
        - All rows from `spectra_table` where spectra_fk_col IN sample_ids
        Also copies indices, triggers, and views.
        """
        # Safety: ensure path is new/empty
        if os.path.exists(dst_path):
            # Overwrite? If you prefer to block, raise instead.
            os.remove(dst_path)
    
        src = sqlite3.connect(src_path)
        dst = sqlite3.connect(dst_path)
    
        try:
            src.row_factory = sqlite3.Row
            s_cur = src.cursor()
            d_cur = dst.cursor()
    
            # 1) Clone schema (tables, indices, triggers, views)
            #    Skip internal sqlite_ objects.
            schema_rows = s_cur.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
                "ORDER BY CASE type "
                " WHEN 'table' THEN 0 WHEN 'index' THEN 1 WHEN 'trigger' THEN 2 WHEN 'view' THEN 3 ELSE 4 END, name;"
            ).fetchall()
    
            dst.execute("PRAGMA foreign_keys=OFF;")
            dst.execute("BEGIN;")
            for r in schema_rows:
                ddl = r["sql"]
                # Create empty schema first
                d_cur.execute(ddl)
    
            # 2) Determine column lists for inserts (preserve all columns)
            def table_cols(conn, tname):
                r = conn.execute(f"PRAGMA table_info({tname});").fetchall()
                return [row[1] for row in r]  # name column
    
            samples_cols = table_cols(src, samples_table)
            spectra_cols = table_cols(src, spectra_table)
    
            # 3) Prepare subset selects
            # Build parameter list safely
            ids = list(sample_ids)
            placeholders = ",".join(["?"] * len(ids))
    
            # Pull subset from samples
            samples_sql = f"SELECT {', '.join(samples_cols)} FROM {samples_table} WHERE {samples_pk_col} IN ({placeholders})"
            sample_rows = s_cur.execute(samples_sql, ids).fetchall()
    
            # Pull subset from spectra
            spectra_sql = f"SELECT {', '.join(spectra_cols)} FROM {spectra_table} WHERE {spectra_fk_col} IN ({placeholders})"
            spectra_rows = s_cur.execute(spectra_sql, ids).fetchall()
    
            # 4) Bulk insert into destination
            ins_samples = f"INSERT INTO {samples_table} ({', '.join(samples_cols)}) VALUES ({', '.join(['?']*len(samples_cols))})"
            ins_spectra = f"INSERT INTO {spectra_table} ({', '.join(spectra_cols)}) VALUES ({', '.join(['?']*len(spectra_cols))})"
    
            # Insert samples first (FK safety)
            for row in sample_rows:
                d_cur.execute(ins_samples, [row[c] for c in samples_cols])
    
            for row in spectra_rows:
                d_cur.execute(ins_spectra, [row[c] for c in spectra_cols])
    
            dst.commit()
    
        except Exception:
            dst.rollback()
            raise
        finally:
            src.close()
            dst.close()
    
    def open_database_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SQLite DB", "", "SQLite DB (*.db);;All Files (*)"
        )
        if not path:
            return
        self.open_database(path)
    
    def open_database(self, path: str):
        """
        (Re)open a SQLite DB file and bind the QSqlTableModel/QTableView to it.
        """
        # If already open on same path, no-op
        if self.db_path and os.path.abspath(self.db_path) == os.path.abspath(path):
            return
    
        # Tear down previous model & connection cleanly
        self._close_current_db()
    
        # Create a fresh connection (unique name avoids Qt caching issues)
        conn_name = f"conn_{id(self)}"
        self.db = QSqlDatabase.addDatabase("QSQLITE", conn_name)
        self.db.setDatabaseName(path)
    
        if not self.db.open():
            err = self.db.lastError().text()
            QMessageBox.critical(self, "Database Error",
                                 f"Could not open database:\n{path}\n\nError: {err}")
            # Remove failed connection handle
            QSqlDatabase.removeDatabase(conn_name)
            return
    
        # Basic schema sanity check (samples + spectra exist)
        if not self._table_exists(SAMPLE_TABLE_NAME) or not self._table_exists(SPECTRA_TABLE_NAME):
            self.db.close()
            QSqlDatabase.removeDatabase(conn_name)
            QMessageBox.critical(
                self, "Schema Error",
                f"Required tables 'samples' and '{SPECTRA_TABLE_NAME}' not found in:\n{path}"
            )
            return
    
        # Bind model to this connection
        self.model = QSqlTableModel(self, self.db)
        self.model.setTable(SAMPLE_TABLE_NAME)
        if not self.model.select():
            QMessageBox.critical(self, "SQL Error",
                                 f"Failed to load 'samples' table: {self.model.lastError().text()}")
            self.db.close()
            QSqlDatabase.removeDatabase(conn_name)
            return
    
        self.table_view.setModel(self.model)
        self.table_view.resizeColumnsToContents()
        self.db_path = path
        self.setWindowTitle(f"PyQt5 SQLite Viewer: {os.path.basename(path)}")
    
        # Reset per-DB state
        self.collection_ids.clear()
        if self.spec_win:
            self.spec_win.close()
            self.spec_win = None
    
    def _table_exists(self, name: str) -> bool:
        q = QSqlQuery(self.db)
        ok = q.exec_(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'")
        if not ok:
            return False
        return q.next()
    
    def _close_current_db(self):
        # Detach model from view so Qt can safely drop the connection
        if hasattr(self, "table_view") and self.table_view is not None:
            self.table_view.setModel(None)
    
        if hasattr(self, "model") and self.model is not None:
            self.model.deleteLater()
            self.model = None
    
        if hasattr(self, "db") and self.db.isValid():
            conn_name = self.db.connectionName()
            if self.db.isOpen():
                self.db.close()
            QSqlDatabase.removeDatabase(conn_name)
    
    def teardown(self):
        # any per-teardown cleanup (close SpectrumWindow, etc.)
        if self.spec_win:
            self.spec_win.close()
            self.spec_win = None
        super().teardown()
        if self.db.isOpen():
            self.db.close()
            
    def get_collection_exemplars(self):
        
        """
        Read currently selected SampleIDs from self.collection_ids,
        fetch (Name, XData, YData) from Spectra table, return a list of dicts:
          {"label": "<Name or ID>", "x_nm": 1D float array (nm), "y": 1D float array}
        """
        
        if not self.db.isOpen() or not self.collection_ids:
            return []
        
        ids = list(self.collection_ids)
        
        # Map SampleID -> Name
        id_to_name = {}
        q = QSqlQuery(self.db)
        placeholders = ",".join("?" for _ in ids)
        
        q.prepare(f"SELECT SampleID, Name FROM {SAMPLE_TABLE_NAME} WHERE SampleID IN ({placeholders})")
        for v in ids: q.addBindValue(int(v))
        if q.exec_():
            while q.next():
                id_to_name[int(q.value(0))] = str(q.value(1))
                print('id_to_name', id_to_name)
    
        exemplars = []
        q2 = QSqlQuery(self.db)
        q2.prepare(f"SELECT SampleID, {WAVELENGTH_BLOB_COL}, {REFLECTANCE_BLOB_COL} "
                   f"FROM {SPECTRA_TABLE_NAME} WHERE SampleID IN ({placeholders})")
        for v in ids: q2.addBindValue(int(v))
    
        if not q2.exec_():
            return []
    
        while q2.next():
            sid = int(q2.value(0))
            x_bytes = q2.value(1); y_bytes = q2.value(2)
            if x_bytes is None or y_bytes is None:
                continue
            x = np.frombuffer(bytes(x_bytes), dtype=BLOB_DTYPE)
            y = np.frombuffer(bytes(y_bytes), dtype=BLOB_DTYPE)
            # Your DB stores µm; your viewer multiplies by 1000 for plotting → convert to nm here
            x_nm = (x * 1000.0).astype(np.float32)
    
            self.exemplars[sid] = (id_to_name.get(sid, f"ID{sid}"),   x_nm, y.astype(np.float32))      
    
        print(self.exemplars)
        return self.exemplars
        

# --- Application Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = LibraryPage()
    viewer.show()
    sys.exit(app.exec_())