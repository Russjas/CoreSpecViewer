"""
UI page for visualising processed data.

Displays spectral products (RGB, masks, MWL maps, classifications)
and provides pixel inspection tools.
"""
import logging

from matplotlib.cbook import _Stack
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QTableWidgetItem, QMenu
from PyQt5 import sip
from .base_page import BasePage
from ..interface import ToolDispatcher
from .util_windows import (SpectrumWindow, 
                           RightClick_TableWidget,
                           ClosableWidgetWrapper)
from .display_canvases import ImageCanvas2D, SpectralImageCanvas
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
        
        mask_canvas = self._create_flagged_canvas("mask")
        self._add_closable_widget(
        mask_canvas,
        title="Mask",
        popoutable=False, closeable=True
        )
        
        
        # Track all canvases for synchronization (including closable ones)
        self._sync_canvases = [self.left_canvas, mask_canvas]
        # Track wether to default to keeping displayed extent
        self.change_display_lims = True
        tbl = RightClick_TableWidget(0, 1, self)
        tbl.setHorizontalHeaderLabels(["Cached Products"])
        tbl.horizontalHeader().setStretchLastSection(True)
        self._add_third(tbl)

        self._splitter.setStretchFactor(0, 5)
        self._splitter.setStretchFactor(1, 5)
        self._splitter.setStretchFactor(2, 2)

        self.cache = set()
        self.table.cellActivated.connect(self._on_row_activated)
        #self.table.cellDoubleClicked.connect(self._on_row_activated)
        self.table.rightClicked.connect(self.tbl_right_click_handler)

        self._mpl_cids = []  # store mpl connection ids
        self._sync_lock = False
        self._view_stack = _Stack()

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
        #Set the right click and dbl-click cr/reflectance spectrum up
        if self.dispatcher:
            def _right_click(y, x):
                if self.current_obj is None or self.current_obj.is_raw:
                    return
                spec = self.current_obj.savgol_cr[y, x, :]
                if not hasattr(self, "cr_spec_win"):
                    self.cr_spec_win = SpectrumWindow(self)
                title = "CR Spectrum Viewer"
                self.cr_spec_win.plot_spectrum(self.current_obj.bands, spec, title=title, label=f"({y}, {x})")
            self.dispatcher.set_right_click(_right_click, temporary=False)
            
            def _double_click(y, x):
                if self.current_obj is None or self.current_obj.is_raw:
                    return                                       
                spec = self.current_obj.savgol[y, x, :]
                if not hasattr(self, "spec_win"):
                    self.spec_win = SpectrumWindow(self)
                title = "Spectrum Viewer"
                self.spec_win.plot_spectrum(self.current_obj.bands, spec, title=title, label=f"({y}, {x})")
            self.dispatcher.set_double_click(_double_click, temporary=False) 

            

    def teardown(self):
        super().teardown()
        self._view_stack.clear()
        # Disconnect any mpl events
        if self._mpl_cids:
            for cv, cid in self._mpl_cids:
                try:
                    cv.mpl_disconnect(cid)
                except Exception:
                    pass
            self._mpl_cids.clear()
        # Remove all closable widgets (except left canvas and table)
        widgets_to_remove = []
        for i in range(self._splitter.count()):
            widget = self._splitter.widget(i)
            if isinstance(widget, ClosableWidgetWrapper):
                # Keep the left canvas wrapper
                if getattr(widget, 'wrapped_widget', None) is not self.left_canvas:
                    widgets_to_remove.append(widget)
        
        for widget in widgets_to_remove:
            self.remove_widget(widget)

        # Recreate the mask canvas fresh
        mask_canvas = self._create_flagged_canvas("mask")
        self._add_closable_widget(
            mask_canvas,
            title="Mask",
            popoutable=False, closeable=True, 
            index=self._splitter.count() -1
            )
        mask_canvas.mask_flag = True
        self._sync_canvases = [self.left_canvas, mask_canvas]
        self._view_stack.clear()

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
                self.dispatcher.remove_canvas(inner) # de-register from the dispatcher 
                # Unbind its mpl events
                self._unbind_mpl_for_canvas(inner)
        
        super().remove_widget(w)
    

    def _purge_dead_refs(self, *_):
        super()._purge_dead_refs()
        self._sync_canvases = [c for c in self._sync_canvases if not sip.isdeleted(c)]
        self._mpl_cids = [(cv, cid) for cv, cid in self._mpl_cids if not sip.isdeleted(cv)]

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
        

        self.dispatcher.add_canvas(canvas)

        # add to the shared navigation stack group
        tb = getattr(canvas, "toolbar", None)
        if tb is not None:
            tb._nav_stack = self._view_stack
            tb._nav_group = self

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

    def sync_members(self):
        """Canvases sharing the view sync and the shared nav history."""
        return list(self._sync_canvases)

    def _seed_nav_home(self):
        """Reset the shared history so Home == the current (full-extent) views."""
        self._view_stack.clear()
        members = self.sync_members()
        if members and getattr(members[0], "toolbar", None) is not None:
            members[0].toolbar.push_current()

    def _add_to_view_group(self, canvas):
        """Make a drawn canvas a full member: shared nav stack, current group
        view, and a history entry. Call after the canvas has been drawn."""
        if canvas not in self._sync_canvases or not hasattr(canvas, "ax"):
            return
        tb = getattr(canvas, "toolbar", None)
        if tb is not None:
            tb._nav_stack = self._view_stack
            tb._nav_group = self
        ref = next((c for c in self._sync_canvases
                    if c is not canvas and hasattr(c, "ax")), None)
        if ref is not None:                       # adopt the group's live view
            self._sync_lock = True
            try:
                canvas.ax.set_xlim(ref.ax.get_xlim())
                canvas.ax.set_ylim(ref.ax.get_ylim())
                canvas.canvas.draw_idle()
            finally:
                self._sync_lock = False
        if tb is not None:
            tb.push_current()                     # combined entry now includes it

    def update_display(self, key='mask'):
        if self.current_obj is None:
            return
        if self.current_obj.is_raw:
            return
        #To manage retaining zoom position if the dataset has not changed.
        images = self.left_canvas.ax.images
        displayed_shape  = images[-1].get_array().shape if images else (0,0)
        same_dims = (self.current_obj.display.shape[:2]==displayed_shape[:2])
        header_item = self.table.horizontalHeaderItem(0)
        displayed_header = header_item.text() if header_item else ""
        current_header = (
                f'{self.current_obj.metadata["borehole id"]} '
                f'{self.current_obj.metadata["box number"]}'
                        )
        same_box = current_header == displayed_header
        if same_dims and same_box:
            self.change_display_lims = False
            
        else:
            self.change_display_lims = True
            
        #end of zoom management set up

        ann = self.current_obj['annotations'].data if self.current_obj.has('annotations') else {}
        self.left_canvas.show_rgb_direct(self.current_obj.display, annotations=ann, lims = self.change_display_lims)
        
        self.refresh_cache_table()
        
        for canvas in list(self._sync_canvases):
            if hasattr(canvas, "product_flag"):
                self._display_product_in_canvas(canvas, canvas.product_flag)
                
        if self.change_display_lims:
            self._seed_nav_home()


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
        canvas = self._create_flagged_canvas(key)
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
        self._add_to_view_group(canvas)


    def _display_product_in_canvas(self, canvas, key):
        """
        Display the specified product in the given canvas.
        """
        if self.current_obj is None or self.current_obj.is_raw:
            return
        ann = self.current_obj['annotations'].data if self.current_obj.has('annotations') else {}
        # Mineral map branch
        if key.endswith("INDEX"):
            try:
                legend_key = key[:-5] + "LEGEND"
                index = self.current_obj.get_data(key)
                legend = None
                if self.current_obj.has(legend_key):
                    legend = self.current_obj[legend_key].data

                if index is not None and getattr(index, "ndim", 0) == 2:
                    canvas.set_annotations(ann)
                    canvas._show_index_with_legend(index, self.current_obj.mask, legend, lims = self.change_display_lims)
                    return
            except KeyError:
                wrapper = canvas.parent()
                if isinstance(wrapper, ClosableWidgetWrapper):
                    wrapper.close()
                return

        # get display data for everything else
        try:
            disp_data = self.current_obj.get_data(key)
        except KeyError:
            wrapper = canvas.parent()
            if isinstance(wrapper, ClosableWidgetWrapper):
                wrapper.close()
            return
        canvas.set_annotations(ann)
        stretch = self.current_obj.get_stretch_values(key)
        canvas.show_rgb(disp_data, stretch = stretch, lims = self.change_display_lims)


    def _create_flagged_canvas(self, product_key):
        """Create a canvas tagged with its product key for auto-refresh."""
        canvas = ImageCanvas2D(self)
        canvas.product_flag = product_key
        return canvas
    

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
        non_vis_suff = {'LEGEND', "stats", "bands", "metadata", "MAPPING", "display", "annotations"}
        base = []
        unwrapped = []
        products = []
        non_vis = []
        self.cache = set(self.current_obj.datasets.keys()) | set(self.current_obj.temp_datasets.keys())
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

        def _insert_depth_header(text: str):
            """Special header for depth range - centered and italic"""
            r = self.table.rowCount()
            self.table.insertRow(r)
            it = QTableWidgetItem(text)
            it.setFlags(Qt.ItemIsEnabled)
            it.setTextAlignment(Qt.AlignCenter)
            f = it.font()
            f.setItalic(True)  # Make it italic to differentiate from section headers
            it.setFont(f)
            self.table.setItem(r, 0, it)


        if self.current_obj is not None and not self.current_obj.is_raw:
            try:
                unit = self.current_obj.get_units()
                table_title = f'{self.current_obj.metadata["borehole id"]} {self.current_obj.metadata["box number"]}'
                first_row = f'{self.current_obj.metadata["core depth start"]}{unit} - {self.current_obj.metadata["core depth stop"]}{unit}'
            except KeyError:
                table_title = 'Cached products'
                first_row = ""
        else:
            table_title = 'Cached products'
            first_row = ""
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

        

        self.table.setRowCount(0)
        _insert_depth_header(first_row)

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