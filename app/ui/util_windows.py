"""
Auxiliary pop-up windows and modal dialogues.

Contains file choosers, legend editors, statistics viewers, and interactive tools
used outside and embedded in the main pages.
"""


from contextlib import contextmanager

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationTool
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.widgets import PolygonSelector, RectangleSelector

import numpy as np

from PyQt5.QtCore import (
        QSortFilterProxyModel, 
        Qt, 
        pyqtSignal, 
        QModelIndex, 
        QDateTime)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
    
)

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')

from ..interface import tools as t
from ..spectral_ops import spectral_functions as sf


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

class RightClick_Table(QTableView):
    rightClicked = pyqtSignal(QModelIndex)
    def __init__(self, parent = None):
        super().__init__(parent)
        self._search_column = 0
        self._type_ahead = ""
        self._last_key_time = 0
        
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                self.rightClicked.emit(index)
            # Important: still pass it on so selection/focus/double-click logic works
        super().mousePressEvent(event)
    
    
    def setSearchColumn(self, col: int):
        """Column index (in the *proxy* model) to use for type-ahead search."""
        self._search_column = max(0, int(col))
        
        
    def keyPressEvent(self, event):
        text = event.text()
        if text and not text.isspace():
            now = QDateTime.currentMSecsSinceEpoch()
            if now - self._last_key_time > 1000:
                self._type_ahead = ""
            self._last_key_time = now

            self._type_ahead += text.lower()

            model = self.model()
            if model is not None:
                row_count = model.rowCount()
                if row_count:
                    current = self.currentIndex()
                    start_row = current.row() if current.isValid() else 0
                    for offset in range(1, row_count + 1):
                        r = (start_row + offset) % row_count
                        idx = model.index(r, self._search_column)
                        val = model.data(idx)
                        if val is None:
                            continue
                        if str(val).lower().startswith(self._type_ahead):
                            self.setCurrentIndex(idx)
                            self.scrollTo(idx, QTableView.PositionAtCenter)
                            break
            return
        super().keyPressEvent(event)


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

def choice_box(text: str, choices: list[str]) -> int | None:
    """
    Display a QMessageBox with an arbitrary list of choice buttons.
    """
    m = QMessageBox()
    m.setWindowTitle('Choose')
    m.setText(text)

    btns = []

    for label in choices:
        btn = m.addButton(label, QMessageBox.AcceptRole)
        btns.append(btn)

    cancel_btn = m.addButton(QMessageBox.Cancel)
    m.setEscapeButton(cancel_btn)

    m.exec_()

    clicked = m.clickedButton()

    if clicked is cancel_btn:
        return None

    # Find which choice index was clicked
    for i, btn in enumerate(btns):
        if clicked is btn:
            return i

    return None  # fallback (shouldn't happen)

class InfoTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.context_dict = False
        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["File", "Type"])
        self.table.horizontalHeader().setStretchLastSection(True)

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
            a = image.astype(float)
            amin = np.nanmin(a)
            amax = np.nanmax(a)
        
            if amax > amin:
                norm = (a - amin) / (amax - amin)
            else:
                norm = np.zeros_like(a, dtype=float)
        
            
            self.ax.clear()
            self.ax.imshow(norm, cmap=my_map, origin="upper", vmin=0.0, vmax=1.0)


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

    def _show_index_with_legend(self, index_2d: np.ndarray, mask: np.ndarray, legend: list[dict]):
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
         # ---- derive max index from data (non-negative only)
        data_positive = index_2d[index_2d >= 0]
        if data_positive.size == 0:
            # nothing to show
            self.ax.clear()
            self.ax.set_axis_off()
            self.canvas.draw()
            return
        max_idx_data = int(data_positive.max())
        # ---- normalize legend (index->label), dedup by index (last wins)
        idx_to_label = {}
        max_idx_legend = -1
        for row in legend or []:
            try:
                idx = int(row.get("index"))
                lab = str(row.get("label", f"class {idx}"))
            except Exception:
                continue
            idx_to_label[idx] = lab
            if idx > max_idx_legend:
                max_idx_legend = idx
        max_idx = max(max_idx_data, max_idx_legend)
        K = max_idx + 1

        # ---- build labels array for 0..K-1 (fallback to "class i" if missing)
        labels = [idx_to_label.get(i, f"class {i}") for i in range(K)]

        # ---- deterministic colors from tab20
        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        colors_rgb = (np.array([cmap(i % 20)[:3] for i in range(K)]) * 255).astype(np.uint8)  # (K,3)

        # ---- make RGB image; treat negatives as transparent-ish background
        idx_img = index_2d.copy()
        neg_mask = idx_img < 0
        neg_mask[mask==1] = 1
        idx_img = np.clip(idx_img, 0, K - 1)
        rgb = colors_rgb[idx_img]
        if neg_mask.any():
            rgb[neg_mask] = np.array([0, 0, 0], dtype=np.uint8)


        # ---- draw
        self.ax.clear()
        self.ax.imshow(rgb, origin="upper")
        self.ax.set_axis_off()

        # ---- legend includes only classes actually present (>=0) in the image
        valid = ~neg_mask
        handles = []
        leg = None

        if valid.any():
            counts = np.bincount(idx_img[valid].ravel(), minlength=K)
            present = np.nonzero(counts)[0]  # indices with at least 1 pixel

            if present.size > 0:
                # sort by count desc, then index asc
                present_sorted = sorted(
                    present.tolist(),
                    key=lambda i: (-int(counts[i]), int(i)),
                )

                total = int(valid.sum())

                def _pct(i: int) -> float:
                    return (counts[i] / total * 100.0) if total > 0 else 0.0

                handles = [
                    Patch(
                        facecolor=(colors_rgb[i] / 255.0),
                        edgecolor="k",
                        label=f"{labels[i]} — {int(counts[i])} px ({_pct(i):.1f}%)",
                    )
                    for i in present_sorted
                ]

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
    
    
    def show_fraction_stack(
        self,
        depths: np.ndarray,
        fractions: np.ndarray,   # (H, K+1)
        legend: list[dict],      # length K, same order used in fractions
        include_unclassified: bool = True,
    ):
        """
        Show a vertical stacked mineral-fraction log.

        Parameters
        ----------
        depths : (H,)
            Depth per row (same length as fractions.shape[0]).
        fractions : (H, K+1)
            Output from compute_fullhole_mineral_fractions.
            Columns 0..K-1 correspond to legend entries (in order).
            Column K is 'unclassified' remainder.
        legend : list of dict
            [{'index': int, 'label': str}, ...], length K.
        include_unclassified : bool
            If False, hides the last 'unclassified' column.
        """
        depths = np.asarray(depths)
        frac = np.asarray(fractions)
        H, C = frac.shape
        K = len(legend)

        if C != K + 1:
            return
        if depths.shape[0] != H:
            return

        # Ensure depth increases downward visually
        if depths[0] > depths[-1]:
            depths = depths[::-1]
            frac = frac[::-1, :]

        self.ax.clear()
        # give some room for a legend on the right, but not as much as the map
        self.canvas.figure.subplots_adjust(right=0.80)
        self.ax.set_axis_on()

        # Which columns to plot
        cols_to_plot = list(range(K))
        if include_unclassified:
            cols_to_plot.append(K)  # last column = remainder

        frac_use = frac[:, cols_to_plot]      # (H, M)
        cum = np.cumsum(frac_use, axis=1)    # (H, M)
        left = np.hstack([np.zeros((H, 1)), cum[:, :-1]])
        right = cum

        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        handles = []

        for band_idx, col_idx in enumerate(cols_to_plot):
            if col_idx < K:
                cid = int(legend[col_idx]["index"])   # library ID
                name = str(legend[col_idx]["label"])
                color = cmap(cid % 20)
            else:
                cid = None
                name = "Unclassified"
                color = (0.3, 0.3, 0.3, 1.0)  # dark grey

            self.ax.fill_betweenx(
                depths,
                left[:, band_idx],
                right[:, band_idx],
                step="pre",
                facecolor=color,
                edgecolor="none",
                label=name,
            )

        self.ax.set_xlim(0.0, 1.0)
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Fraction of row width")
        self.ax.set_ylabel("Depth")
        self.ax.grid(True, axis="x", alpha=0.2)

        # Legend: one entry per band
        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.9,
            fontsize=9,
            handlelength=1.8,
            handletextpad=0.6,
        )

        self.canvas.draw_idle()
    
    
    
    
    
    def clear_memmap_refs(self):
        """Clear any matplotlib artists that might hold data references."""
        self.ax.clear()
        self.canvas.draw_idle()

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

        # polygon selector
        self._poly_selector = None
        self.on_polygon_finished = None

        # UI elements
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
        if getattr(self, "_poly_selector", None) is not None:
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

    # polygon selection methods
    def start_polygon_select(self):

        self.cancel_rect_select()
        self.cancel_polygon_select()

        def _on_select(verts):
            # verts is [(x,y), ...] in data coords
            if callable(self.on_polygon_finished):
                v_rc = [(int(round(y)), int(round(x))) for (x, y) in verts]
                self.on_polygon_finished(v_rc)
            self.cancel_polygon_select()
            self.canvas.draw()

        self._poly_selector = PolygonSelector(
        self.ax,
        onselect=_on_select,
        useblit=True,
        props=dict(color="orange", alpha=0.9, linewidth=1.5),
        handle_props=dict(marker="o", markersize=4,
                          mec="k", mfc="orange", alpha=0.9),
        grab_range=5,
        draw_bounding_box=False,
    )

        try:
            self.canvas.widgetlock(self._poly_selector)
            self.canvas.draw()
        except ValueError:
            return

    def cancel_polygon_select(self):
        
        """Tear down an active polygon tool, if any."""
        if self._poly_selector is not None:
            try:
                self.canvas.widgetlock.release(self._poly_selector)
            except Exception:
                pass
            try:
                self._poly_selector.disconnect_events()
                self._poly_selector = None
            except Exception:
                self._poly_selector = None



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
    def clear_memmap_refs(self):
        """Release any memmap references held by this canvas."""
        self.cube = None
        self.bands = None
        self.ax.clear()  # Also clear matplotlib artists
        self.canvas.draw_idle()

class ClosableWidgetWrapper(QWidget):
    """
    Wraps a widget (like ImageCanvas2D) with a close button/action.
    The parent page connects to the closed signal to remove this wrapper.
    """
    # Signal emitted when the close button is clicked, carries a reference to self
    closed = pyqtSignal(object)

    def __init__(self, wrapped_widget: QWidget, title: str = "", parent=None, closeable = True):
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
        if closeable:
            close_action = QAction("✕ Close", self)
            close_action.setToolTip(f"Close {title}")
            close_action.triggered.connect(self._emit_closed)
            self.toolbar.addAction(close_action)
        else:
            default_label = QAction("Default", self)
            default_label.setToolTip("Default cannot be closed")
            self.toolbar.addAction(default_label)

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


class WavelengthRangeDialog(QDialog):
    """
    Dialog to request a start/stop wavelength (nm).
    Usage:
        ok, start, stop = WavelengthRangeDialog.get_values(parent, 2100, 2300)
    """

    def __init__(self, parent=None, start_default=None, stop_default=None):
        super().__init__(parent)

        self.setWindowTitle("Select Wavelength Range")

        # --- Widgets ---
        start_label = QLabel("Start (nm):")
        stop_label = QLabel("Stop (nm):")

        self.start_edit = QLineEdit()
        self.stop_edit = QLineEdit()

        if start_default is not None:
            self.start_edit.setText(str(start_default))
        if stop_default is not None:
            self.stop_edit.setText(str(stop_default))

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            orientation=Qt.Horizontal,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # --- Layout ---
        start_layout = QHBoxLayout()
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_edit)

        stop_layout = QHBoxLayout()
        stop_layout.addWidget(stop_label)
        stop_layout.addWidget(self.stop_edit)

        main_layout = QVBoxLayout()
        main_layout.addLayout(start_layout)
        main_layout.addLayout(stop_layout)
        main_layout.addWidget(buttons)

        self.setLayout(main_layout)

    def get_values(self):
        """Return (start_nm, stop_nm) as floats, or (None, None) if invalid."""
        try:
            start = float(self.start_edit.text())
            stop = float(self.stop_edit.text())
        except ValueError:
            return None, None
        return start, stop

    @classmethod
    def get_range(cls, parent=None, start_default=None, stop_default=None):
        """
        Convenience one-shot:
            ok, start, stop = WavelengthRangeDialog.get_range(...)
        """
        dlg = cls(parent, start_default, stop_default)
        result = dlg.exec_()
        if result == QDialog.Accepted:
            start, stop = dlg.get_values()
            return True, start, stop
        return False, None, None



class AutoSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(420, 320)
        cfg = t.get_config()

        # Build dynamic table: Key | Value
        self.tbl = QTableWidget(len(cfg), 2)
        self.tbl.setHorizontalHeaderLabels(["Setting", "Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)

        for row, (k, v) in enumerate(cfg.items()):
            key_item = QTableWidgetItem(k)
            key_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            val_item = QTableWidgetItem(str(v))
            self.tbl.setItem(row, 0, key_item)
            self.tbl.setItem(row, 1, val_item)

        btn_save = QPushButton("Save")
        btn_cancel = QPushButton("Cancel")
        btn_save.clicked.connect(self._on_save)
        btn_cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_cancel)
        row.addWidget(btn_save)

        root = QVBoxLayout(self)
        root.addWidget(self.tbl)
        root.addLayout(row)

    def _on_save(self):
        for r in range(self.tbl.rowCount()):
            key = self.tbl.item(r, 0).text()
            val = self.tbl.item(r, 1).text()
            t.modify_config(key, val)

        self.accept()
