# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:37:08 2025

@author: russj
"""

from PyQt5.QtWidgets import QTableWidgetItem,QTableWidget
from PyQt5.QtCore import Qt                
              
from .util_windows import SpectralImageCanvas, ImageCanvas2D, SpectrumWindow
from .base_page import BasePage



class VisualisePage(BasePage):
    """
    Mirrors WorkingWindow central content:
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
        
        
                
        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 5)
        self._splitter.setStretchFactor(2, 2)

        self.cache = set()
        self.table.cellDoubleClicked.connect(self._on_row_activated)
        
        self._mpl_cids = []  # store mpl connection ids if you add them
        self._sync_lock = False
        
    def activate(self):
        super().activate()
        if self.current_obj is None:
            return
        if self.current_obj.is_raw:
            return
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
            self.dispatcher.set_right_click(_right_click, temporary=False)
            
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
        self.cache.clear()
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem("Cached Products"))

        
    def update_display(self, key='mask'):
        if self.current_obj is None:
            return
        if self.current_obj.is_raw:
            return
        self.left_canvas.show_rgb(self.current_obj.savgol, self.current_obj.bands)
        
        self._set_cache()
    
        # Mineral map branch
        if key.endswith("INDEX"):
            legend_key = key[:-5] + "LEGEND"  # replace 'INDEX' with 'LEGEND'
            index = self.current_obj.get_data(key)
            legend = None
            # robust legend fetch: prefer temp, else permanent
            if self.current_obj.has(legend_key):
                legend = self.current_obj[legend_key].data
    
            if index is not None and getattr(index, "ndim", 0) == 2:
                self.right_canvas._show_index_with_legend(index, self.current_obj.mask, legend)
                return
    
        # Fallback for everything else
        self.right_canvas.show_rgb(self.current_obj.get_data(key))
        
     
    
    
    
    
    def _on_row_activated(self, row: int, col: int):
        """
        Ignore header rows; on item rows, open the product on the right canvas.
        """
        it = self.table.item(row, 0)
        if not it:
            return
        
    
        key = it.text().strip()
        if not key:
            return
    
        self.update_display(key=key)
        
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
        base_whitelist = {"savgol", "savgol_cr", "mask", "segments", "cropped"}
        unwrap_prefixes = ("Dhole",)  # DholeAverage, DholeMask, DholeDepths
        non_vis_suff = {'LEGEND', 'CLUSTERS', "stats", "bands" }
        base = []
        unwrapped = []
        products = []
        non_vis = []
        if self.current_obj is not None and not self.current_obj.is_raw:
            try:
                table_title = f'{self.current_obj.metadata["borehole id"]} {self.current_obj.metadata["box number"]}'
            except:
                table_title = 'Cached products'
        else:
            table_title = 'Cached products'
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem(table_title))
    
        for k in sorted(self.cache):  # stable order
            if k in base_whitelist:
                base.append(k)
            elif any(k.startswith(pfx) for pfx in unwrap_prefixes):
                unwrapped.append(k)
            elif any(k.endswith(sfx) for sfx in non_vis_suff):
                non_vis.append(k)
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
        if products:
            _insert_header("Products")
            for k in products:
                _insert_item(k)
        if unwrapped:
            _insert_header("Unwrapped")
            for k in unwrapped:
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
    





