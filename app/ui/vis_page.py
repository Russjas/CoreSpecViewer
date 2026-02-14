"""
UI page for visualising processed data.

Displays spectral products (RGB, masks, MWL maps, classifications)
and provides pixel inspection tools.
"""
import logging

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QTableWidgetItem, QMenu

from .base_page import BasePage
from ..interface import ToolDispatcher
from .util_windows import (ImageCanvas2D, 
                           SpectralImageCanvas, 
                           SpectrumWindow, 
                           RightClick_TableWidget,
                           ClosableWidgetWrapper)
from .display_text import gen_display_text

logger = logging.getLogger(__name__)

class VisualisePage(BasePage):
    """
    Page for visualising derived content from core box scans
    """
    clusterRequested = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main canvases (non-closable)
        self._left = SpectralImageCanvas(self)
        self._add_closable_widget(
            self._left,
            title="Smoothed image",
            popoutable=False, closeable=False
        )
        
        self._right = ImageCanvas2D(self)
        self._add_closable_widget(
            self._right,
            title="Mask",
            popoutable=False, closeable=False
        )
        
        # Track all canvases for synchronization (including closable ones)
        self._sync_canvases = [self.left_canvas, self.right_canvas]
        
        tbl = RightClick_TableWidget(0, 1, self)
        tbl.setHorizontalHeaderLabels(["Cached Products"])
        tbl.horizontalHeader().setStretchLastSection(True)
        self._add_third(tbl)

        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 5)
        self._splitter.setStretchFactor(2, 2)

        self.cache = set()
        self.table.cellDoubleClicked.connect(self._on_row_activated)
        self.table.rightClicked.connect(self.tbl_right_click_handler)

        self._mpl_cids = []  # store mpl connection ids
        self._sync_lock = False

    def activate(self):
        super().activate()
        # No bindings or display if there is no dataset loaded    
        if self.current_obj is None:
            return
        if self.current_obj.is_raw:
            return
        #Use the centralised logic for binding sync now that we are all closeable
        for canvas in self._sync_canvases:
            self._register_sync_canvas(canvas)
        #Set the right click cr spectrum up
        if self.current_obj is not None and not self.current_obj.is_raw and self.dispatcher:
            def _right_click(y, x):
                spec = self.current_obj.savgol_cr[y, x, :]
                if not hasattr(self, "spec_win"):
                    self.spec_win = SpectrumWindow(self)
                title = "CR Spectrum Viewer"
                self.spec_win.plot_spectrum(self.current_obj.bands, spec, title=title)
            self.dispatcher.set_right_click(_right_click, temporary=False)
            self.refresh_cache_table()

    def teardown(self):
        super().teardown()
        # Disconnect any mpl events
        if self._mpl_cids:
            for cv, cid in self._mpl_cids:
                try:
                    cv.mpl_disconnect(cid)
                except Exception:
                    pass
            self._mpl_cids.clear()
        
        # Clear sync canvases but keep the main two
        self._sync_canvases = [self.left_canvas, self.right_canvas]
        
        self.cache.clear()
        self.table.setRowCount(0)
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem("Cached Products"))

    def remove_widget(self, w):
        """
        Override to handle removal of closable canvas widgets.
        Unregister from sync and unbind mpl events.
        """
              
        inner = None
        if isinstance(w, ClosableWidgetWrapper):
            inner = getattr(w, "wrapped_widget", None)
        
        # If it's a canvas, remove from sync list
        if isinstance(inner, (ImageCanvas2D, SpectralImageCanvas)):
            if inner in self._sync_canvases:
                self._sync_canvases.remove(inner)
                # Unbind its mpl events
                self._unbind_mpl_for_canvas(inner)
        
        super().remove_widget(w)

    def _unbind_mpl_for_canvas(self, canvas):
        """Disconnect mpl events for a specific canvas."""
        if not hasattr(canvas, 'canvas'):
            return
            
        canvas_obj = canvas.canvas
        # Remove all cids associated with this canvas
        self._mpl_cids = [(cv, cid) for cv, cid in self._mpl_cids 
                          if cv is not canvas_obj]

    def _register_sync_canvas(self, canvas):
        """
        Add a new canvas to the sync group and bind events.
        """
        if canvas not in self._sync_canvases:
            self._sync_canvases.append(canvas)
        
        # Bind sync events if we're active
        if not hasattr(canvas, 'canvas'):
            return
            
        def _sync_now(src_ax, dst_ax):
            if self._sync_lock: 
                return
            self._sync_lock = True
            try:
                dst_ax.set_xlim(src_ax.get_xlim())
                dst_ax.set_ylim(src_ax.get_ylim())
                dst_ax.figure.canvas.draw_idle()
            finally:
                self._sync_lock = False

        def _sync_from_event(ev):
            src_canvas = None
            for c in self._sync_canvases:
                if hasattr(c, 'canvas') and ev.canvas is c.canvas:
                    src_canvas = c
                    break
            
            if src_canvas is None:
                return
                
            src_ax = src_canvas.ax
            for c in self._sync_canvases:
                if c is not src_canvas and hasattr(c, 'ax'):
                    _sync_now(src_ax, c.ax)

        self._bind_mpl(canvas.canvas, "button_release_event", _sync_from_event)
        self._bind_mpl(canvas.canvas, "scroll_event", _sync_from_event)
        self._bind_mpl(canvas.canvas, "key_release_event", _sync_from_event)

    def update_display(self, key='mask'):
        if self.current_obj is None:
            return
        if self.current_obj.is_raw:
            return
        import time
        start = time.perf_counter()
        logger.debug(f"PROFILE VIS PAGE UPDATE DISPLAY: Start : {start:.4f}s")
        self.left_canvas.show_rgb_direct(self.current_obj.display)
        checkpoint_1 = time.perf_counter()
        logger.debug(f"PROFILE UPDATE DISPLAY: Left canvas displayed: {checkpoint_1 - start:.4f}s")
        self.refresh_cache_table()
        checkpoint_2 = time.perf_counter()
        logger.debug(f"PROFILE UPDATE DISPLAY: cache table refreshed: {checkpoint_2 - checkpoint_1:.4f}s")
        self._display_product_in_canvas(self.right_canvas, 'mask')
        checkpoint_3 = time.perf_counter()
        logger.debug(f"PROFILE UPDATE DISPLAY: right canvas displayed: {checkpoint_3 - checkpoint_1:.4f}s")
        logger.debug(f"PROFILE UPDATE DISPLAY: TOTAL update display : {checkpoint_3 - start:.4f}s")

    def _on_row_activated(self, row: int, col: int):
        """
        On double-click: create a new closable widget with the selected product.
        """
        
        it = self.table.item(row, 0)
        if not it:
            return

        key = it.data(Qt.UserRole)
        if not key:
            return
        logger.info(f"Button clicked: Product table dbl clicked {key}")
        if key.endswith('CLUSTERS'):
            self.clusterRequested.emit(key)
            return
        disp = gen_display_text(key)
        # Create a new closable canvas
        canvas = ImageCanvas2D(self)
        wrapper = self._add_closable_widget(
            canvas,
            title=f"Product: {disp}",
            popoutable=True, index=self._splitter.count() -1
        )
        wrapper.popout_requested.connect(self._handle_popout_request)
        # Register for sync
        self._register_sync_canvas(canvas)
        
        # Display the product
        self._display_product_in_canvas(canvas, key)
        logger.info(f"{key} displayed in vis page.")

    def _display_product_in_canvas(self, canvas, key):
        """
        Display the specified product in the given canvas.
        """
        if self.current_obj is None or self.current_obj.is_raw:
            return

        # Mineral map branch
        if key.endswith("INDEX"):
            legend_key = key[:-5] + "LEGEND"
            index = self.current_obj.get_data(key)
            legend = None
            if self.current_obj.has(legend_key):
                legend = self.current_obj[legend_key].data

            if index is not None and getattr(index, "ndim", 0) == 2:
                canvas._show_index_with_legend(index, self.current_obj.mask, legend)
                return

        # Fallback for everything else
        try:
            disp_data = self.current_obj.get_data(key)
        except KeyError:
            return
        canvas.show_rgb(disp_data)

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
        """
        base_whitelist = {"savgol", "savgol_cr", "mask", "segments", "cropped"}
        unwrap_prefixes = ("Dhole",)
        non_vis_suff = {'LEGEND', "stats", "bands", "metadata", "MAPPING"}
        base = []
        unwrapped = []
        products = []
        non_vis = []
        self.cache = set(self.current_obj.datasets.keys()) | set(self.current_obj.temp_datasets.keys())

        if self.current_obj is not None and not self.current_obj.is_raw:
            try:
                table_title = f'{self.current_obj.metadata["borehole id"]} {self.current_obj.metadata["box number"]}'
            except KeyError:
                table_title = 'Cached products'
        else:
            table_title = 'Cached products'
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem(table_title))

        for k in sorted(self.cache):
            if k in base_whitelist:
                base.append(k)
            elif any(k.startswith(pfx) for pfx in unwrap_prefixes):
                unwrapped.append(k)
            elif any(k.endswith(sfx) for sfx in non_vis_suff):
                non_vis.append(k)
            else:
                products.append(k)

        def _insert_header(text: str):
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(text)
            it.setFlags(Qt.NoItemFlags)
            f = it.font()
            f.setBold(True)
            it.setFont(f)
            self.table.setItem(r, 0, it)

        def _insert_item(key: str):
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(gen_display_text(key))
            it.setData(Qt.UserRole, key)
            
            it.setTextAlignment(Qt.AlignCenter)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, it)

        self.table.setRowCount(0)

        if base:
            _insert_header("Base processed")
            for k in sorted(base):
                _insert_item(k)
        if products:
            _insert_header("Products")
            for k in sorted(products):
                _insert_item(k)
        if unwrapped:
            _insert_header("Unwrapped")
            for k in sorted(unwrapped):
                _insert_item(k)

        self.table.resizeRowsToContents()

    def tbl_right_click_handler(self, row, column):
        if self.cxt.ho is not None:
            return
        it = self.table.item(row, 0)
        if not it:
            return
        key = it.data(Qt.UserRole)
        if not key:
            return
        menu = QMenu(self)
        
        act_delete = menu.addAction("Delete row")
        action = menu.exec_(QCursor.pos())
        
        if action == act_delete:
            self.current_obj.delete_dataset(key)
        self.update_display()

    def _bind_mpl(self, canvas, event, handler):
        cid = canvas.mpl_connect(event, handler)
        self._mpl_cids.append((canvas, cid))
        return cid