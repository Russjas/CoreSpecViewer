"""
Display canvas widgets for CoreSpecViewer.

Provides a base matplotlib canvas class with shared infrastructure:
    - Figure / axes / FigureCanvas boilerplate
    - Annotation overlay system (draw, clear, toggle)
    - Shape selector methods (rect, polygon, circle, line)
    - popup() utility

Subclasses:
    ImageCanvas2D       — display-only canvas for products, downhole plots,
                          mineral maps. Uses BaseCanvasToolbar (annotation toggle).
    SpectralImageCanvas — interactive canvas for the main image view. Adds
                          click/rect/polygon dispatcher wiring, spectrum display,
                          contrast/equalise/reset controls, and cube reference.
                          Uses SpectralCanvasToolbar.

Toolbar hierarchy:
    NavigationTool  (matplotlib)
        └── BaseCanvasToolbar       — annotation toggle button
                └── SpectralCanvasToolbar   — + Contrast+, Equalize, Reset
"""

import logging
import uuid

import matplotlib
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.facecolor'] = 'white'

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationTool,
)
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.widgets import PolygonSelector, RectangleSelector

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut
)
from PyQt5.QtGui import QKeySequence

from ..spectral_ops.visualisation import get_false_colour
from .util_windows import SpectrumWindow
logger = logging.getLogger(__name__)

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')


# ============================================================================
# Annotation label style — shared constant
# ============================================================================
_ANN_LABEL_KW = dict(
    fontsize=8,
    fontweight='bold',
    color='white',
    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
    zorder=6,
)

# Per-shape colours
_SHAPE_COLOURS = {
    "point":   "red",
    "line":    "blue",
    "rect":    "green",
    "polygon": "yellow",
    "circle":  "orange",
}

# ---- Canvas display decorators ----
# These decorators are tightly coupled to BaseMatplotlibCanvas and its subclasses.
# They assume self.toolbar._ann_btn, self._show_annotations,
# self._last_annotations, self.draw_annotations(), self.clear_annotations()
# are all present on the instance. Do not use outside of display_canvases.py.


def spatial_display(func):
        def wrapper(self, *args, **kwargs):
            self.toolbar._ann_btn.setEnabled(True)   # 1. enable button
            result = func(self, *args, **kwargs)      # 2. ax.clear(), imshow(), canvas.draw()
            if self._show_annotations:               # 3. after canvas.draw() has fired
                self.draw_annotations(self._last_annotations)
            return result
        return wrapper

def chart_display(func):
    def wrapper(self, *args, **kwargs):
        self._show_annotations = False
        self.toolbar._ann_btn.setChecked(False)
        self.toolbar._ann_btn.setEnabled(False)
        self.clear_annotations()
        return func(self, *args, **kwargs)
    return wrapper

# ============================================================================
# Toolbars
# ============================================================================

class BaseCanvasToolbar(NavigationTool):
    """
    Extends the matplotlib navigation toolbar with an annotation toggle button.
    Wired to parent.`_toggle_annotations` — parent must be a BaseMatplotlibCanvas.
    """

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.addSeparator()
        self._ann_btn = QPushButton("Annotations", self)
        self._ann_btn.setToolTip("Toggle annotation overlay")
        self._ann_btn.setCheckable(True)
        self._ann_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4a90d9;
                color: white;
                border: 1px solid #2a70b9;
                border-radius: 3px;
            }
            QPushButton:!checked {
                background-color: none;
            }
        """)
        self._ann_btn.clicked.connect(parent._toggle_annotations)
        self.addWidget(self._ann_btn)


class SpectralCanvasToolbar(BaseCanvasToolbar):
    """
    Extends BaseCanvasToolbar with image adjustment buttons:
    Contrast+, Equalize, Reset.
    Wired to parent methods — parent must be a SpectralImageCanvas.
    """

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.addSeparator()

        contrast_btn = QPushButton("Contrast+", self)
        contrast_btn.setToolTip("Increase image contrast (2-98 percentile stretch) (Alt+C)")
        contrast_btn.clicked.connect(lambda: parent.increase_contrast())
        self.addWidget(contrast_btn)

        hist_btn = QPushButton("Equalize", self)
        hist_btn.setToolTip("Histogram equalization (enhance detail) (Alt+E)")
        hist_btn.clicked.connect(lambda: parent.equalize_histogram())
        self.addWidget(hist_btn)

        reset_btn = QPushButton("Reset", self)
        reset_btn.setToolTip("Reset to original image")
        reset_btn.clicked.connect(lambda: parent.reset_display())
        self.addWidget(reset_btn)


# ============================================================================
# Base canvas
# ============================================================================

class BaseMatplotlibCanvas(QWidget):
    """
    Base class for all matplotlib-backed canvas widgets.

    Provides:
    - Figure / axes / FigureCanvas construction
    - Toolbar instantiation via toolbar_class class attribute
    - popup() utility
    - Annotation overlay: draw_annotations(), clear_annotations(),
      _toggle_annotations(), set_annotations()
    - Shape selector methods: start/cancel for rect, polygon, circle, line
    - clear_memmap_refs() default implementation

    Subclasses set toolbar_class to control which toolbar is used.
    """

    toolbar_class = BaseCanvasToolbar  # override in subclasses

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- annotation state ----
        # Data owned by PO, refreshed on every update_display call
        self._last_annotations = {}
        # User toggle — never touched by page-level code
        self._show_annotations = False
        # Matplotlib artist handles for cleanup only
        self._annotation_artists = []

        # ---- shape selector state ----
        # Rectangle
        self.rect_selector = None
        self.on_rectangle_selected = None   # callable(y0, y1, x0, x1)
        self._last_rect = None

        # Polygon
        self._poly_selector = None
        self.on_polygon_finished = None     # callable(vertices_rc)

        # Circle (drag)
        self._circle_centre = None
        self._circle_preview = None
        self._circle_press_cid = None
        self._circle_motion_cid = None
        self._circle_release_cid = None
        self.on_circle_selected = None      # callable(cy, cx, r)

        # Line (drag)
        self._line_start = None
        self._line_preview = None
        self._line_press_cid = None
        self._line_motion_cid = None
        self._line_release_cid = None
        self.on_line_selected = None        # callable(y0, x0, y1, x1)

        # ---- matplotlib figure ----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.toolbar = self.toolbar_class(self.canvas, self)
        layout.addWidget(self.toolbar)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------


    def clear_memmap_refs(self):
        """
        Release any held data references and wipe the axes.
        Subclasses that hold cube/bands refs should override and call super().
        """
        self._last_annotations = {}   # data gone; flag preserved
        self.ax.clear()
        self.clear_annotations()      # removes artists, draw_idle

    # ------------------------------------------------------------------
    # Annotation overlay
    # ------------------------------------------------------------------

    def set_annotations(self, annotations: dict):
        """
        Called by the page on every update_display.
        Stores the fresh PO annotations and redraws if overlay is active.

        annotations: {"ann_<uuid>": {"shape": str, "label": str, ...}, ...}
        """
        self._last_annotations = annotations
        if self._show_annotations:
            self.draw_annotations(annotations)

    def _toggle_annotations(self, checked: bool):
        """Slot wired to the toolbar annotation button."""
        self._show_annotations = checked
        if checked:
            self.draw_annotations(self._last_annotations)
        else:
            self.clear_annotations()

    def draw_annotations(self, annotations: dict):
        """
        Render annotation overlays onto the current axes.
        Clears any existing annotation artists first.

        annotations: {
            "ann_<uuid>": {
                "shape": "point"|"line"|"rect"|"polygon"|"circle",
                "label": str,
                ... geometry keys ...
            }
        }
        """
        # Remove existing artists
        for artist in self._annotation_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._annotation_artists = []

        for entry in annotations.values():
            shape = entry.get("shape", "point")
            label = entry.get("label", "")
            colour = _SHAPE_COLOURS.get(shape, "yellow")
            txt = None

            try:
                if shape == "point":
                    dot = self.ax.plot(
                        entry["x"], entry["y"], "o",
                        color=colour, markersize=6,
                        markeredgecolor="black", markeredgewidth=0.5,
                        zorder=5,
                    )[0]
                    txt = self.ax.annotate(
                        label,
                        xy=(entry["x"], entry["y"]),
                        xytext=(8, -8), textcoords="offset points",
                        **_ANN_LABEL_KW,
                    )
                    self._annotation_artists.extend([dot, txt])

                elif shape == "line":
                    line = self.ax.plot(
                        [entry["x0"], entry["x1"]],
                        [entry["y0"], entry["y1"]],
                        color=colour, linewidth=1.5, zorder=5,
                    )[0]
                    mx = (entry["x0"] + entry["x1"]) / 2
                    my = (entry["y0"] + entry["y1"]) / 2
                    txt = self.ax.annotate(
                        label,
                        xy=(mx, my),
                        xytext=(4, -8), textcoords="offset points",
                        **_ANN_LABEL_KW,
                    )
                    self._annotation_artists.extend([line, txt])

                elif shape == "rect":
                    patch = matplotlib.patches.Rectangle(
                        (entry["x0"], entry["y0"]),
                        entry["x1"] - entry["x0"],
                        entry["y1"] - entry["y0"],
                        fill=False, edgecolor=colour,
                        linewidth=1.5, zorder=5,
                    )
                    self.ax.add_patch(patch)
                    txt = self.ax.annotate(
                        label,
                        xy=(entry["x0"], entry["y0"]),
                        xytext=(4, -8), textcoords="offset points",
                        **_ANN_LABEL_KW,
                    )
                    self._annotation_artists.extend([patch, txt])

                elif shape == "polygon":
                    verts = entry["vertices"]   # [[y, x], ...]
                    xy = [(v[1], v[0]) for v in verts]
                    patch = matplotlib.patches.Polygon(
                        xy, closed=True, fill=False,
                        edgecolor=colour, linewidth=1.5, zorder=5,
                    )
                    self.ax.add_patch(patch)
                    cx = sum(v[1] for v in verts) / len(verts)
                    cy = sum(v[0] for v in verts) / len(verts)
                    txt = self.ax.annotate(
                        label,
                        xy=(cx, cy),
                        xytext=(4, -8), textcoords="offset points",
                        **_ANN_LABEL_KW,
                    )
                    self._annotation_artists.extend([patch, txt])

                elif shape == "circle":
                    patch = matplotlib.patches.Circle(
                        (entry["cx"], entry["cy"]), entry["r"],
                        fill=False, edgecolor=colour,
                        linewidth=1.5, zorder=5,
                    )
                    self.ax.add_patch(patch)
                    txt = self.ax.annotate(
                        label,
                        xy=(entry["cx"], entry["cy"] - entry["r"]),
                        xytext=(4, -8), textcoords="offset points",
                        **_ANN_LABEL_KW,
                    )
                    
  
                    self._annotation_artists.extend([patch, txt])

                else:
                    logger.warning(f"Unknown annotation shape: {shape}")

            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to draw annotation entry {entry}: {e}")
                continue

        self.canvas.draw_idle()

    def clear_annotations(self):
        """
        Remove annotation artists from axes.
        Does NOT touch _show_annotations — that is user state only.
        """
        for artist in self._annotation_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._annotation_artists = []
        self.canvas.draw_idle()

    
    # ------------------------------------------------------------------
    # Rectangle selector
    # ------------------------------------------------------------------

    def start_rect_select(self, minspan=(5, 5), interactive=True, useblit=True):
        if getattr(self.toolbar, "mode", ""):
            return
        self.cancel_rect_select()
        self.rect_selector = RectangleSelector(
            self.ax, self._on_rect_select,
            useblit=useblit, button=[1],
            minspanx=minspan[0], minspany=minspan[1],
            spancoords="pixels", interactive=interactive,
        )
        self.canvas.draw_idle()

    def cancel_rect_select(self):
        if self.rect_selector:
            self.rect_selector.set_active(False)
            self.rect_selector.disconnect_events()
            self.rect_selector = None
            self.canvas.draw_idle()

    def _on_rect_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x0, x1 = sorted((x1, x2))
        y0, y1 = sorted((y1, y2))
        # Clamp to cube bounds if available
        cube = getattr(self, "cube", None)
        if cube is not None:
            h, w = cube.shape[:2]
            x0 = max(0, min(x0, w - 1)); x1 = max(1, min(x1, w))
            y0 = max(0, min(y0, h - 1)); y1 = max(1, min(y1, h))
        self._last_rect = (y0, y1, x0, x1)
        cb = self.on_rectangle_selected
        self.cancel_rect_select()
        if callable(cb):
            cb(y0, y1, x0, x1)

    def rect_props(self):
        """Return (y0, y1, x0, x1) or None."""
        return self._last_rect

    def rect_slices(self):
        """Return (rows_slice, cols_slice) or None."""
        if self._last_rect is None:
            return None
        y0, y1, x0, x1 = self._last_rect
        return slice(y0, y1), slice(x0, x1)

    # ------------------------------------------------------------------
    # Polygon selector
    # ------------------------------------------------------------------

    def start_polygon_select(self):
        self.cancel_rect_select()
        self.cancel_polygon_select()

        def _on_select(verts):
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
        """Tear down an active polygon selector, if any."""
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

    # ------------------------------------------------------------------
    # Circle selector (drag: press = centre, release = radius)
    # ------------------------------------------------------------------

    def start_circle_select(self):
        self._circle_centre = None
        self._remove_circle_preview()
        self._circle_press_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_circle_press
        )
        self._circle_motion_cid = self.canvas.mpl_connect(
            "motion_notify_event", self._on_circle_motion
        )
        self._circle_release_cid = self.canvas.mpl_connect(
            "button_release_event", self._on_circle_release
        )

    def _on_circle_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        self._circle_centre = (event.xdata, event.ydata)

    def _on_circle_motion(self, event):
        if self._circle_centre is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        cx, cy = self._circle_centre
        r = ((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2) ** 0.5
        self._remove_circle_preview()
        self._circle_preview = matplotlib.patches.Circle(
            (cx, cy), r,
            fill=False, edgecolor="yellow",
            linewidth=1.5, linestyle="--", zorder=7,
        )
        self.ax.add_patch(self._circle_preview)
        self.canvas.draw_idle()

    def _on_circle_release(self, event):
        if self._circle_centre is None or event.button != 1:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        cx, cy = self._circle_centre
        r = ((event.xdata - cx) ** 2 + (event.ydata - cy) ** 2) ** 0.5
        self._remove_circle_preview()
        self._disconnect_circle_events()
        if callable(self.on_circle_selected):
            self.on_circle_selected(int(cy), int(cx), int(r))

    def _remove_circle_preview(self):
        if self._circle_preview is not None:
            try:
                self._circle_preview.remove()
            except Exception:
                pass
            self._circle_preview = None

    def _disconnect_circle_events(self):
        for attr in ("_circle_press_cid", "_circle_motion_cid", "_circle_release_cid"):
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._circle_centre = None

    def cancel_circle_select(self):
        self._remove_circle_preview()
        self._disconnect_circle_events()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Line selector (drag: press = start, release = end)
    # ------------------------------------------------------------------

    def start_line_select(self):
        self._line_start = None
        self._remove_line_preview()
        self._line_press_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_line_press
        )
        self._line_motion_cid = self.canvas.mpl_connect(
            "motion_notify_event", self._on_line_motion
        )
        self._line_release_cid = self.canvas.mpl_connect(
            "button_release_event", self._on_line_release
        )

    def _on_line_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        self._line_start = (event.xdata, event.ydata)

    def _on_line_motion(self, event):
        if self._line_start is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x0, y0 = self._line_start
        self._remove_line_preview()
        self._line_preview = self.ax.plot(
            [x0, event.xdata], [y0, event.ydata],
            color="yellow", linewidth=1.5, linestyle="--", zorder=7,
        )[0]
        self.canvas.draw_idle()

    def _on_line_release(self, event):
        if self._line_start is None or event.button != 1:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x0, y0 = self._line_start
        self._remove_line_preview()
        self._disconnect_line_events()
        if callable(self.on_line_selected):
            self.on_line_selected(int(y0), int(x0), int(event.ydata), int(event.xdata))

    def _remove_line_preview(self):
        if self._line_preview is not None:
            try:
                self._line_preview.remove()
            except Exception:
                pass
            self._line_preview = None

    def _disconnect_line_events(self):
        for attr in ("_line_press_cid", "_line_motion_cid", "_line_release_cid"):
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    self.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._line_start = None

    def cancel_line_select(self):
        self._remove_line_preview()
        self._disconnect_line_events()
        self.canvas.draw_idle()


# ============================================================================
# ImageCanvas2D
# ============================================================================

class ImageCanvas2D(BaseMatplotlibCanvas):
    """
    Display-only canvas for products, mineral maps, and downhole plots.

    Uses BaseCanvasToolbar — has annotation toggle but no interaction tools.
    No dispatcher, no click wiring, no cube reference.
    """

    toolbar_class = BaseCanvasToolbar

    def __init__(self, parent=None):
        super().__init__(parent)

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------
    @spatial_display
    def show_rgb(self, image):
        if image.dtype == bool:
            image = image.astype(int)
        shp = getattr(image, "shape", None)
        if not shp or len(shp) == 1:
            return

        if len(shp) == 2:
            lo = np.nanpercentile(image.data, 2)
            hi = np.nanpercentile(image.data, 98)
            clipped = np.clip(image, lo, hi) if hi > lo else image
            self.ax.clear()
            self.ax.imshow(clipped, cmap=my_map, origin="upper")

        elif len(shp) == 3 and shp[2] == 3:
            self.ax.clear()
            self.ax.imshow(image, origin="upper")

        elif len(shp) == 3 and shp[2] > 3:
            self.ax.clear()
            self.ax.imshow(get_false_colour(image), origin="upper")

        else:
            return

        self.ax.set_axis_off()
        self.canvas.draw()

    @spatial_display
    def _show_index_with_legend(self, index_2d: np.ndarray, mask: np.ndarray, legend: list[dict]):
        """Render an indexed mineral map with a discrete colour legend."""
        if index_2d.ndim != 2:
            raise ValueError("index_2d must be a 2-D integer array of class indices.")

        H, W = index_2d.shape
        if H == 0 or W == 0:
            self.ax.clear(); self.ax.set_axis_off(); self.canvas.draw(); return

        data_positive = index_2d[index_2d >= 0]
        if data_positive.size == 0:
            self.ax.clear(); self.ax.set_axis_off(); self.canvas.draw(); return

        max_idx_data = int(data_positive.max())
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
        labels = [idx_to_label.get(i, f"class {i}") for i in range(K)]

        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        colors_rgb = (np.array([cmap(i % 20)[:3] for i in range(K)]) * 255).astype(np.uint8)

        idx_img = index_2d.copy()
        neg_mask = idx_img < 0
        neg_mask[mask == 1] = 1
        idx_img = np.clip(idx_img, 0, K - 1)
        rgb = colors_rgb[idx_img]
        if neg_mask.any():
            rgb[neg_mask] = np.array([0, 0, 0], dtype=np.uint8)

        self.ax.clear()
        self.ax.imshow(rgb, origin="upper")
        self.ax.set_axis_off()

        valid = ~neg_mask
        handles = []
        leg = None

        if valid.any():
            counts = np.bincount(idx_img[valid].ravel(), minlength=K)
            present = np.nonzero(counts)[0]
            if present.size > 0:
                present_sorted = sorted(
                    present.tolist(),
                    key=lambda i: (-int(counts[i]), int(i)),
                )
                total = int(valid.sum())

                def _pct(i):
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
            self.canvas.figure.subplots_adjust(right=0.80)
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
            leg.set_draggable(True)

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Downhole display methods
    # ------------------------------------------------------------------
    @chart_display
    def display_fractions(
        self,
        depths: np.ndarray,
        fractions: np.ndarray,
        legend: list[dict],
        include_unclassified: bool = True,
    ):
        """Show a vertical stacked mineral-fraction log."""


        depths = np.asarray(depths)
        frac = np.asarray(fractions)
        H, C = frac.shape
        K = len(legend)
        if C != K + 1 or depths.shape[0] != H:
            return
        if depths[0] > depths[-1]:
            depths = depths[::-1]
            frac = frac[::-1, :]

        self.ax.clear()
        self.canvas.figure.subplots_adjust(right=0.80)
        self.ax.set_axis_on()

        cols_to_plot = list(range(K))
        if include_unclassified:
            cols_to_plot.append(K)

        frac_use = frac[:, cols_to_plot]
        cum = np.cumsum(frac_use, axis=1)
        left = np.hstack([np.zeros((H, 1)), cum[:, :-1]])
        right = cum

        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]

        for band_idx, col_idx in enumerate(cols_to_plot):
            if col_idx < K:
                cid = int(legend[col_idx]["index"])
                name = str(legend[col_idx]["label"])
                color = cmap(cid % 20)
            else:
                name = "Unclassified"
                color = (0.7, 0.7, 0.7, 1.0)

            self.ax.fill_betweenx(
                depths,
                left[:, band_idx],
                right[:, band_idx],
                step="pre",
                facecolor=color,
                edgecolor="none",
                label=name,
            )

        self.ax.set_ylim(depths.min(), depths.max())
        self.ax.invert_yaxis()
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_xlabel("Fraction of row width")
        self.ax.set_ylabel("Depth")
        self.ax.grid(True, axis="x", alpha=0.2)
        self.ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True, framealpha=0.9,
            fontsize=9, handlelength=1.8, handletextpad=0.6,
        )
        self.canvas.draw_idle()

    @chart_display
    def display_discrete(
        self,
        depths: np.ndarray,
        dominant_indices: np.ndarray,
        legend: list[dict],
        width: float = 0.1,
    ):
        """Display a categorical log track based on dominant mineral indices."""
        depths = np.asarray(depths)
        dominant_indices = np.asarray(dominant_indices)
        if depths[0] > depths[-1]:
            depths = depths[::-1]
            dominant_indices = dominant_indices[::-1]

        self.ax.clear()
        self.canvas.figure.subplots_adjust(right=0.80)

        cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]
        index_to_color = {}
        legend_handles = []
        legend_labels = []

        for i, entry in enumerate(legend):
            try:
                mineral_id = int(entry["index"])
            except (TypeError, ValueError):
                continue
            color = cmap(mineral_id % 20)
            index_to_color[i] = color
            legend_handles.append(matplotlib.patches.Patch(facecolor=color))
            legend_labels.append(entry["label"])

        no_data_color = (1.0, 1.0, 1.0, 1.0)
        index_to_color[-1] = no_data_color
        legend_handles.append(matplotlib.patches.Patch(facecolor=no_data_color))
        legend_labels.append("No Dominant / Gap")

        H = dominant_indices.shape[0]
        for i in range(H):
            idx = dominant_indices[i]
            z_top = depths[i]
            z_bottom = (
                depths[i + 1] if i + 1 < H
                else depths[-1] + (depths[-1] - depths[-2])
            )
            color = index_to_color.get(idx, (0.5, 0.5, 0.5, 1.0))
            self.ax.barh(
                y=z_top, width=width, height=z_bottom - z_top,
                left=0, align="edge", color=color, edgecolor="none",
            )

        self.ax.set_ylim(depths.min(), depths.max())
        self.ax.invert_yaxis()
        self.ax.set_ylabel("Depth")
        self.ax.set_xlabel("Dominant Mineral")
        self.ax.set_xlim(0.0, width)
        self.ax.set_xticks([])
        self.ax.set_xticklabels([])
        self.ax.legend(
            handles=legend_handles, labels=legend_labels,
            loc="upper left", bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0, frameon=True, framealpha=0.9,
            fontsize=9, handlelength=1.8, handletextpad=0.6,
        )
        self.canvas.draw_idle()

    @chart_display
    def display_continuous(self, depths, values, key):
        """Display a continuous depth-vs-value profile."""
        if depths.shape != values.shape:
            return
        if depths[0] > depths[-1]:
            depths = depths[::-1]
            values = values[::-1]
        self.ax.clear()
        self.ax.plot(values, depths, "o-", markersize=3)
        self.ax.invert_yaxis()
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_xlabel(key)
        self.ax.grid(True, alpha=0.3)
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    def clear_memmap_refs(self):
        """Release data references and clear axes."""
        self._last_annotations = {}
        self.ax.clear()
        self.clear_annotations()


# ============================================================================
# SpectralImageCanvas
# ============================================================================

class SpectralImageCanvas(BaseMatplotlibCanvas):
    """
    Interactive canvas for the main hyperspectral image view.

    Extends BaseMatplotlibCanvas with:
    - Hyperspectral cube reference (cube, bands)
    - Click dispatcher wiring (on_single_click, on_right_click)
    - Double-click spectrum display
    - Image adjustment: increase_contrast, equalize_histogram, reset_display
    - show_rgb (derives false colour from cube)
    - show_rgb_direct (accepts pre-computed RGB + annotations)

    Uses SpectralCanvasToolbar.
    """

    toolbar_class = SpectralCanvasToolbar

    def __init__(self, parent=None):
        super().__init__(parent)

        # Cube reference
        self.spec_win = None
        self.cube = None
        self.bands = None
        self._current_rgb = None

        # Click wiring — assigned by ToolDispatcher
        self.on_single_click = None     # callable(y, x)
        self.on_right_click = None      # callable(y, x)

        # Wire click handler
        self.canvas.mpl_connect("button_press_event", self.on_image_click)

        #Shortcuts for visual display changes
        QShortcut(QKeySequence("Alt+E"), self,
          activated=self.equalize_histogram,
          context=Qt.ApplicationShortcut)

        QShortcut(QKeySequence("Alt+C"), self,
                activated=self.increase_contrast,
                context=Qt.ApplicationShortcut)

    # ------------------------------------------------------------------
    # Display methods
    # ------------------------------------------------------------------
    @spatial_display
    def show_rgb(self, cube, bands):
        """Derive and display false colour from a hyperspectral cube."""
        self._last_rect = None
        self.cube = cube
        self.bands = bands
        rgb = get_false_colour(cube)
        self._current_rgb = rgb
        self.ax.clear()
        self.ax.imshow(rgb, origin="upper")
        self.ax.set_axis_off()
        self.canvas.draw()

    @spatial_display
    def show_rgb_direct(self, rgb_array, cube, bands, annotations=None):
        """
        Display a pre-computed RGB array and refresh annotation overlay.

        annotations: {"ann_<uuid>": {...}, ...} — passed from page on every
        update_display. If None, falls back to empty dict (safe for RawPage
        which never calls this method).
        """
        self._last_rect = None
        self.cube = cube
        self.bands = bands
        self._current_rgb = rgb_array
        self._last_annotations = annotations if annotations is not None else {}
        self.ax.clear()
        self.ax.imshow(rgb_array, origin="upper")
        self.ax.set_axis_off()
        self.canvas.draw()

    @spatial_display
    def increase_contrast(self):
        """Apply a 2–98 percentile contrast stretch to the current RGB."""
        if self._current_rgb is None:
            return
        rgb = self._current_rgb
        p_low, p_high = np.percentile(rgb, (2, 98))
        rgb_contrast = np.clip((rgb - p_low) / (p_high - p_low), 0, 1)
        self.ax.clear()
        self.ax.imshow(rgb_contrast, origin="upper")
        self.ax.set_axis_off()
        self.canvas.draw()

    @spatial_display
    def equalize_histogram(self):
        """Apply per-channel histogram equalisation to the current RGB."""
        if self._current_rgb is None:
            return
        rgb = self._current_rgb.astype(np.float32) / 255.0
        rgb_eq = np.zeros_like(rgb)
        for i in range(3):
            channel = rgb[:, :, i]
            hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 1))
            cdf = hist.cumsum()
            cdf = cdf / cdf[-1]
            rgb_eq[:, :, i] = np.interp(channel.flatten(), bins[:-1], cdf).reshape(channel.shape)
        self.ax.clear()
        self.ax.imshow(rgb_eq, origin="upper")
        self.ax.set_axis_off()
        self.canvas.draw()

    @spatial_display
    def reset_display(self):
        """Reset to original un-adjusted RGB."""
        if self._current_rgb is None:
            return
        self.ax.clear()
        self.ax.imshow(self._current_rgb, origin="upper")
        self.ax.set_axis_off()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def on_image_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if getattr(self.toolbar, "mode", "") or self.rect_selector is not None:
            return
        if getattr(self, "_poly_selector", None) is not None:
            return
        r = int(round(event.ydata))
        c = int(round(event.xdata))
        if self.cube is None:
            return
        # Double-click → spectrum
        if event.dblclick:
            
            spec = self.cube[r, c]
            if self.spec_win is None:
                self.spec_win = SpectrumWindow(self)
            self.spec_win.plot_spectrum(self.bands, spec, title="Spectrum Viewer")
            return
        if event.button == 3 and callable(self.on_right_click):
            self.on_right_click(r, c)
            return
        if event.button == 1 and callable(self.on_single_click):
            self.on_single_click(r, c)

    # ------------------------------------------------------------------
    # clear_memmap_refs — override to also drop cube/bands
    # ------------------------------------------------------------------

    def clear_memmap_refs(self):
        """Release cube/bands memmap references, clear axes and annotations."""
        self.cube = None
        self.bands = None
        self._last_annotations = {}   # data cleared; _show_annotations preserved
        self.ax.clear()
        self.cancel_circle_select()
        self.cancel_line_select()
        self.clear_annotations()      # removes artists, draw_idle
