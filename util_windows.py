# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 12:11:50 2025

@author: russj
"""


import numpy as np
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QSortFilterProxyModel
from PyQt5.QtWidgets import (QApplication,
    QMainWindow, QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QLineEdit, QDialog, QDialogButtonBox,
    QMessageBox, QToolBar, QLabel, QAction
)
from contextlib import contextmanager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationTool
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Patch
import matplotlib

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')

import spectral_functions as sf

#==========reference passing and cache update======================
@contextmanager
def busy_cursor(msg=None, window=None):
    """Temporarily set the cursor to busy; restores automatically."""
    QApplication.setOverrideCursor(Qt.WaitCursor)
    if window and hasattr(window, "statusBar") and msg:
        window.statusBar().showMessage(msg)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()
        if window and hasattr(window, "statusBar"):
            window.statusBar().clearMessage()

class EventBus(QObject):
    dataset_discovered = pyqtSignal(object)  # payload: {"box": str, "kind": "raw"|"processed", "path": str}

BUS = EventBus()

def post_dataset_ref(box: str, kind: str, path: str):
    
    """Lightweight, validated emit used by child windows."""
    if not box or kind not in ("raw", "processed") or not path:
        return
    
    BUS.dataset_discovered.emit({"box": box, "kind": kind, "path": path})

class IdSetFilterProxy(QSortFilterProxyModel):
    """
    Show only rows whose SampleID (at id_col) is in allowed_ids.
    If allowed_ids is None or empty, show all rows.
    """
    def __init__(self, id_col: int, allowed_ids: set = None, parent=None):
        super().__init__(parent)
        self._id_col = int(id_col)
        self._allowed = set(allowed_ids) if allowed_ids else None
        self.setDynamicSortFilter(True)

    def set_allowed_ids(self, ids: set | None):
        self._allowed = set(ids) if ids else None
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._allowed:
            return True
        src = self.sourceModel()
        if src is None:
            return True
        idx = src.index(source_row, self._id_col, source_parent)
        val = src.data(idx, Qt.DisplayRole)
        try:
            sid = int(val)
        except (TypeError, ValueError):
            return False
        return sid in self._allowed

def two_choice_box(text, left_choice_text, right_choice_text):
    m = QMessageBox()
    m.setWindowTitle("Choose")
    m.setText(text)

    left_btn  = m.addButton(left_choice_text, QMessageBox.YesRole)
    right_btn = m.addButton(right_choice_text, QMessageBox.NoRole)
    cancel_btn = m.addButton(QMessageBox.Cancel)

    m.setDefaultButton(left_btn)
    m.setEscapeButton(cancel_btn)

    m.exec_()  # exec_() if you’re strictly on PyQt5

    clicked = m.clickedButton()
    if clicked is left_btn:
        return "left"
    if clicked is right_btn:
        return "right"
    return "cancel"  # always explicit

class InfoTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.context_dict = False
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["File", "Type"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Fixed dropdown rows
        #self.fixed_input_rows = ["File Reference"]
        

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        

    def add_row(self, key, value, editable=False):
        r = self.table.rowCount()
        self.table.insertRow(r)

        key_item = QTableWidgetItem(str(key))
        key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 0, key_item)

        val_item = QTableWidgetItem(str(value))
        if not editable:
            val_item.setFlags(val_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 1, val_item)

            
    def set_from_dict(self, d):
        self.table.setRowCount(0)
        # Normal rows
        for k, v in d.items():
            self.add_row(k, v, editable=False)
        



class ImageCanvas2D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
         
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.toolbar = NavigationTool(self.canvas, self)
        layout.addWidget(self.toolbar)

    def show_rgb(self, image):
        
        shp = getattr(image, "shape", None)
        if not shp or len(shp) == 1:
            return  # ignore 1D/unknown
        
        if len(shp) == 2:
            rgb = image
            self.ax.clear()
            self.ax.imshow(rgb, cmap=my_map, origin="upper", vmin = np.min(rgb), vmax = np.max(rgb))
        
        elif len(shp) == 3 and shp[2] == 3:
            rgb = image
            self.ax.clear()
            self.ax.imshow(rgb, origin="upper")
        
        elif len(shp) == 3 and shp[2] > 3:
            rgb = sf.get_false_colour(image)
            self.ax.clear()
            self.ax.imshow(rgb, origin="upper")
        else:
            return
        self.ax.set_axis_off()
        self.canvas.draw()

    def popup(self, title="Image"):
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle(title)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint |
                                      Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.show()
        return self

    def _show_index_with_legend(self, index_2d: np.ndarray, legend: list[dict]):
        """
        Render an indexed mineral map with a discrete legend.
    
        legend items contain ONLY:
            {"index": int, "label": str}
        Colors are generated deterministically from matplotlib's tab20.
        """
        if index_2d.ndim != 2:
            raise ValueError("index_2d must be a 2-D integer array of class indices.")
    
        H, W = index_2d.shape
        if H == 0 or W == 0:
            self.ax.clear(); self.ax.set_axis_off(); self.canvas.draw(); return
    
        # ---- normalize legend (index->label), dedup by index (last wins)
        idx_to_label = {}
        for row in legend or []:
            try:
                idx = int(row.get("index"))
                lab = str(row.get("label", f"class {idx}"))
                idx_to_label[idx] = lab
            except Exception:
                continue
    
        # ---- decide K from actual indices present (ignore negatives)
        present = np.unique(index_2d[index_2d >= 0])
        if present.size == 0:
            self.ax.clear(); self.ax.set_axis_off(); self.canvas.draw(); return
        max_idx = int(present.max())
        K = max_idx + 1
    
        # ---- build labels array for 0..K-1 (fallback to "class i" if missing)
        labels = [idx_to_label.get(i, f"class {i}") for i in range(K)]
    
        # ---- deterministic colors from tab20
        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        colors_rgb = (np.array([cmap(i % 20)[:3] for i in range(K)]) * 255).astype(np.uint8)  # (K,3)
    
        # ---- make RGB image; treat negatives as transparent-ish background
        idx_img = index_2d.copy()
        neg_mask = idx_img < 0
        idx_img = np.clip(idx_img, 0, K - 1)
        rgb = colors_rgb[idx_img]
        if neg_mask.any():
            # paint negatives light gray (or leave as-is if you prefer)
            rgb[neg_mask] = np.array([220, 220, 220], dtype=np.uint8)
    
        # ---- draw
        self.ax.clear()
        self.ax.imshow(rgb, origin="upper")
        self.ax.set_axis_off()
    
        # ---- legend includes only classes actually present (>=0) in the image
        present = np.unique(idx_img[~neg_mask])
        handles = [Patch(facecolor=(colors_rgb[i]/255.0), edgecolor='k', label=labels[i])
                   for i in present.tolist()]
        if handles:
            
    
        
            # make space on the right
            self.canvas.figure.subplots_adjust(right=0.80)  # ~20% for legend
            leg = self.ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.00),
                borderaxespad=0.0,
                frameon=True,
                framealpha=0.9,
                fontsize=9,
                handlelength=1.8,
                handletextpad=0.6,
            )
        
    
        if leg:
            leg.set_title("Mineral", prop={"size": 9})
            leg.set_draggable(True)  # users can move it if they want
    
        self.canvas.draw_idle()
    
    
    
    
    
        self.canvas.draw()

class SpectralImageCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.spec_win = None
        self.cube = None
        self.bands = None

        # --- rectangle selection state / API ---
        self.rect_selector = None
        self.on_rectangle_selected = None   # assign a callable(y0, y1, x0, x1) from parent
        self._last_rect = None              # pollable: (y0, y1, x0, x1)

        # Single and right click wiring
        self.on_single_click = None         # callable(y, x) -> None
        self.on_right_click  = None         # callable(y, x) -> None

        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.toolbar = NavigationTool(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.canvas.mpl_connect("button_press_event", self.on_image_click)

    def show_rgb(self, cube, bands):
        self._last_rect = None  # reset any previous ROI
        self.cube = cube
        self.bands = bands
        rgb = sf.get_false_colour(cube)
        
        self.ax.clear()
        self.ax.imshow(rgb, origin="upper")
        self.ax.set_axis_off()
        self.canvas.draw()

    # -------- Double-click → spectrum (per-canvas window) --------
    def on_image_click(self, event):
        
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if getattr(self.toolbar, "mode", "") or self.rect_selector is not None:
            return
        if event.xdata is None or event.ydata is None:
            return
        r = int(round(event.ydata))
        c = int(round(event.xdata))
        if self.cube is None:# or self.bands is None:
            return
        # Double clicks hard-wired to spectrum display
        if event.dblclick:
            spec = self.cube[r, c]
            
            if self.spec_win is None:
                self.spec_win = SpectrumWindow(self)
            title = "Spectrum Viewer"
            self.spec_win.plot_spectrum(self.bands, spec, title=title)
            return
        # hardwire_
        if event.button == 3 and callable(self.on_right_click):
            self.on_right_click(r,c)
            return
        # pass single clicks back to parent using parent-assigned callable
        if event.button == 1 and callable(self.on_single_click):
            self.on_single_click(r, c)
    
    
    
    
    # -------- Rectangle selection: start/cancel, callback, polling --------
    def start_rect_select(self, minspan=(5, 5), interactive=True):
        # avoid conflicts with pan/zoom from toolbar
        if getattr(self.toolbar, "mode", ""):
            return
        self.cancel_rect_select()
        self.rect_selector = RectangleSelector(
            self.ax, self._on_rect_select,
            useblit=True, button=[1],
            minspanx=minspan[0], minspany=minspan[1],
            spancoords='pixels', interactive=interactive
        )
        self.canvas.draw_idle()

    def cancel_rect_select(self):
        if self.rect_selector:
            self.rect_selector.set_active(False)
            self.rect_selector.disconnect_events()
            self.rect_selector = None
            self.canvas.draw_idle()

    def _on_rect_select(self, eclick, erelease):
        # raw coords → sorted → clamped
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x0, x1 = sorted((x1, x2))
        y0, y1 = sorted((y1, y2))

        if self.cube is not None:
            h, w = self.cube.shape[:2]
            x0 = max(0, min(x0, w-1)); x1 = max(1, min(x1, w))
            y0 = max(0, min(y0, h-1)); y1 = max(1, min(y1, h))

        self._last_rect = (y0, y1, x0, x1)  # row/col order
        cb = self.on_rectangle_selected
        self.cancel_rect_select()
        if callable(cb):
            cb(y0, y1, x0, x1)

    # pollable helpers for parents that don't want callbacks
    def rect_props(self):
        """Return (y0, y1, x0, x1) or None."""
        return self._last_rect

    def rect_slices(self):
        """Return (rows_slice, cols_slice) or None."""
        if self._last_rect is None:
            return None
        y0, y1, x0, x1 = self._last_rect
        return slice(y0, y1), slice(x0, x1)

class ClosableWidgetWrapper(QWidget):
    """
    Wraps a widget (like ImageCanvas2D) with a close button/action.
    The parent page connects to the closed signal to remove this wrapper.
    """
    # Signal emitted when the close button is clicked, carries a reference to self
    closed = pyqtSignal(object) 
    
    def __init__(self, wrapped_widget: QWidget, title: str = "", parent=None):
        super().__init__(parent)
        self.wrapped_widget = wrapped_widget
        
        # 1. Create a toolbar for the close button
        self.toolbar = QToolBar(self)
        self.toolbar.setStyleSheet("QToolBar { border: none; padding: 2px; }")
        self.toolbar.setMovable(False)

        # 2. Add a title/label
        self.label = QLabel(title); self.toolbar.addWidget(self.label)
        self.toolbar.addSeparator()
        #self.toolbar.addStretch()

        # 3. Add the close action
        close_action = QAction("✕ Close", self)
        close_action.setToolTip(f"Close {title}")
        close_action.triggered.connect(self._emit_closed)
        self.toolbar.addAction(close_action)

        # 4. Main layout (Toolbar above, Wrapped Widget below)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.wrapped_widget)

    def _emit_closed(self):
        """Emits the signal that the parent should handle."""
        self.closed.emit(self)





class SpectrumWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pixel spectrum")
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.add_subplot(111)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toolbar = NavigationTool(self.canvas, central)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        

        self.setCentralWidget(central)
        
    def clear_all(self):
        self.ax.clear()
        self._series_count = 0
        self.canvas.draw()
      
    def plot_spectrum(self, x, y, title=""):
        if x is not None:
            self.ax.plot(x, y)
        else:
            self.ax.plot(y)
        self.ax.set_xlabel("Wavelength (nm)" if x is not None else "Band")
        self.ax.set_ylabel("Reflectance")
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.show()  
        
    def closeEvent(self, ev):
        self.clear_all()
        
class MetadataDialog(QDialog):
    def __init__(self, meta=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Specim Metadata")

        layout = QVBoxLayout(self)

        self.hole_edit = QLineEdit()
        self.box_edit = QLineEdit()
        self.from_edit = QLineEdit()
        self.to_edit   = QLineEdit()

        self.hole_edit.setPlaceholderText("Hole ID")
        self.box_edit.setPlaceholderText("Box number")
        self.from_edit.setPlaceholderText("Depth from")
        self.to_edit.setPlaceholderText("Depth to")

        # Pre-fill existing values
        if meta:
            self.hole_edit.setText(meta.get('borehole id', ''))
            self.box_edit.setText(meta.get('box number', ''))
            self.from_edit.setText(meta.get('core depth start', ''))
            self.to_edit.setText(meta.get('core depth stop', ''))

        for w in (self.hole_edit, self.box_edit, self.from_edit, self.to_edit):
            layout.addWidget(w)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_result(self):
        return {
            "hole": self.hole_edit.text().strip(),
            "box": self.box_edit.text().strip(),
            "depth_from": self.from_edit.text().strip(),
            "depth_to": self.to_edit.text().strip(),
        }
                
        
        