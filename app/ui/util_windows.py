"""
Auxiliary pop-up windows and modal dialogues.

Contains file choosers, legend editors, statistics viewers, and interactive tools
used outside and embedded in the main pages.
"""


from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.facecolor'] = 'white'
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationTool
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.widgets import PolygonSelector, RectangleSelector

import numpy as np
import logging


from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import (
        QSortFilterProxyModel, 
        Qt, 
        pyqtSignal, 
        QModelIndex, 
        QDateTime,
        QTimer,
        QThread)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
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
    QComboBox,
    QDoubleSpinBox,
    QShortcut
    
)

logger = logging.getLogger(__name__)

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')

from .display_text import gen_display_text
from ..interface import tools as t
from ..spectral_ops.visualisation import get_false_colour

#==========reference passing and cache update======================
@contextmanager
def busy_cursor(msg=None, window=None):
    """
    Show busy cursor with optional animated status message.
    
    Backward compatible - all existing calls work unchanged:
        with busy_cursor("Loading...", self):
            do_work()
    
    Extended functionality - yield progress object for dynamic updates:
        with busy_cursor("Processing boxes", self) as progress:
            for i, po in enumerate(hole):
                progress.messages[i] = f"Processing {po.basename}"
                progress.update(i)
                po.do_thing()
    
    Args:
        msg: Status message to display
        window: Window with statusBar() for messages
    
    Yields:
        ProgressHelper object (only if used as `with ... as progress:`)
    """
    
    # Create progress helper
    helper = _ProgressHelper(window, msg)
    
    try:
        yield helper
    finally:
        helper.cleanup()


class _ProgressHelper:
    """
    Shows a small floating window with animated message after 3 seconds.
    
    Window appears after 3 seconds if operation is still running.
    Updates via .set() method and animates with dots.
    """
    
    def __init__(self, parent, base_message):
        
        self.parent = parent
        self.base_message = base_message or "Processing..."
        
        self.current_message = base_message
        
        self.dialog = None
        self.label = None
        self.window_visible = False
        
        # Set busy cursor immediately
        QApplication.setOverrideCursor(Qt.WaitCursor)
               
        # Create dialog 
        
    
    def _create_dialog(self):
        """Create dialog on main thread (hidden initially)"""
        
        self.dialog = QDialog(self.parent)
        self.dialog.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool
        )
        self.dialog.setModal(False)
        
        self.label = QLabel(self.base_message)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 20px 30px;
                border-radius: 8px;
                font-size: 13px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dialog.setLayout(layout)
    
    def _show_window(self):
        """Show dialog"""
        
        if not self.dialog:
            return
        
        self.dialog.adjustSize()
        
        # Center on parent
        if self.parent:
            parent_geo = self.parent.geometry()
            self.dialog.move(
                parent_geo.center().x() - self.dialog.width() // 2,
                parent_geo.center().y() - self.dialog.height() // 2
            )
        
        self.dialog.show()
        self.dialog.raise_()
        self.window_visible = True
        
              
        QApplication.processEvents()
    
    
    
    def set(self, message):
        """
        Update the displayed message.
        
        Args:
            message: New message to display
        """
        
        if not self.window_visible:
            self._create_dialog()
            self._show_window()
        if self.label:
            self.label.setText(message)
            self.dialog.adjustSize()  # Resize to fit new text
            QApplication.processEvents()
    
    def cleanup(self):
        """Clean up dialog, and cursor"""
           
        if self.dialog:
            self.dialog.close()
            self.dialog.deleteLater()
        
        QApplication.restoreOverrideCursor()

class PopoutWindow(QMainWindow):
    """
    A simple top-level window to host a popped-out widget.
    It takes ownership of the content widget and ensures it's resized.
    """
    def __init__(self, content_widget: QWidget, title: str = "Popout Window", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        content_widget.setParent(self) 
        
        self.setCentralWidget(content_widget)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        
        self.resize(content_widget.sizeHint() * 1.5)


class RightClick_TableWidget(QTableWidget):
    rightClicked = pyqtSignal(int, int)  # row, column
    
    def __init__(self, rows=0, cols=1, parent=None):
        super().__init__(rows, cols, parent)
        self._search_column = 0
        self._type_ahead = ""
        self._last_key_time = 0
        
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            item = self.itemAt(event.pos())
            if item is not None:
                self.rightClicked.emit(item.row(), item.column())
        super().mousePressEvent(event)



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


class ClosableWidgetWrapper(QWidget):
    """
    Wraps a widget (like ImageCanvas2D) with a close button/action.
    The parent page connects to the closed signal to remove this wrapper.
    """
    # Signal emitted when the close button is clicked, carries a reference to self
    closed = pyqtSignal(object)
    popout_requested = pyqtSignal(object)
    def __init__(self, wrapped_widget: QWidget, title: str = "", parent=None, closeable = True, popoutable = False):
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
        
        if popoutable:
            popout_action = QAction("⇱", self) # Using U+21f1 (North West Arrow and South East Arrow)
            popout_action.setToolTip(f"Show {title} in a separate window")
            popout_action.triggered.connect(self._emit_popout)
            self.toolbar.addAction(popout_action)
        
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
        
    def _emit_popout(self):
        """Emits the signal that the parent should handle to undock the widget."""
        self.popout_requested.emit(self)

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
        self._series_count = 0 

    def clear_all(self):
        self.ax.clear()
        self._series_count = 0
        self.canvas.draw()

    def plot_spectrum(self, x, y, title="", label=None):
        lbl = label if label else (title if title else f"Series {self._series_count + 1}")
        if x is not None:
            self.ax.plot(x, y, label=lbl)
        else:
            self.ax.plot(y, label=lbl)
        self._series_count += 1 
        self.ax.set_xlabel("Wavelength (nm)" if x is not None else "Band")
        self.ax.set_ylabel("Reflectance")
        if title:
            self.ax.set_title(title)
        if self._series_count > 1:                                     
            self.ax.legend(loc="best", framealpha=0.9,                
                           fontsize=9, draggable=True)             
        else:                                                         
            legend = self.ax.get_legend()                              
            if legend:                                                 
                legend.remove()                                       
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
        self.show()

    def closeEvent(self, ev):
        self.clear_all()


class LibMetadataDialog(QDialog):
    """
    Dialog that dynamically builds metadata fields from a list of column names.
    You provide:
        - columns: list[str] of column names from the Samples table
        - existing meta: optional dict of pre-filled values
    """
    def __init__(self, meta=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Library Metadata")
        #Naughty hard coded schema
        columns = ["Name", 
                   "Type", 
                   "Class", 
                   "SubClass",
                   "ParticleSize", 
                   "Owner",
                   "Origin",
                   "Phase", 
                   "Description"]
       
        layout = QVBoxLayout(self)

        self.edit_fields = {}   # column_name → QLineEdit
        
        for col in columns:
            edit = QLineEdit()
            edit.setPlaceholderText(col)

            # Pre-fill if metadata exists
            if meta and col in meta:
                edit.setText(str(meta[col]))

            self.edit_fields[col] = edit
            layout.addWidget(edit)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_metadata(self):
        """
        Return dict {column_name: value} for all editable fields.
        Columns with empty text still return "", so caller may filter.
        """
        return {col: w.text().strip() for col, w in self.edit_fields.items()}



class MetadataDialog(QDialog):
    """
    Dialog that requests mandatory metadata values
    """
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


class ProfileExportDialog(QDialog):
    """
    Dialog to choose:
      - a dataset key (dropdown)
      - a step value (numeric)
      - an output directory (browse)

    Usage:
        ok, key, step, out_dir, mode = ProfileExportDialog.get_values(
            parent=self,
            keys=keys,
            step_default=hole.step,
            dir_default=hole.root / "profiles",
            title="Export profiles")
    """

    def __init__(
        self,
        parent=None,
        keys=None,
        step_default=None,
        dir_default=None,
        title="Export profile csv",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.keys = list(keys or [])
        self.display_keys = [gen_display_text(key) for key in self.keys]
        self.key_map = dict(zip(self.display_keys, self.keys))
        
        dir_default = Path(dir_default) if dir_default is not None else None

        # --- Widgets ---
        key_label = QLabel("Dataset key:")
        self.key_combo = QComboBox()
        self.key_combo.addItems([str(k) for k in self.display_keys])

        step_label = QLabel("Step:")
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(2)
        self.step_spin.setSingleStep(0.01)
        self.step_spin.setMinimum(0.01)        # avoid zero unless meaningful
        self.step_spin.setMaximum(1_000_000.0) # arbitrary large ceiling
        if step_default is not None:
            self.step_spin.setValue(float(step_default))

        dir_label = QLabel("Output folder:")
        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(False)  # set True if you want browse-only
        if dir_default is not None:
            self.dir_edit.setText(str(dir_default))

        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_for_dir)

        export_modes = ["full", "stepped", "both"]
        export_labels = ["Every pixel", "Resampled data", "Both"]
        self.mode_map = dict(zip(export_labels, export_modes))
        mode_label = QLabel("What do you want to export?")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([str(k) for k in export_labels])
        

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            orientation=Qt.Horizontal,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # --- Layout ---
        row_key = QHBoxLayout()
        row_key.addWidget(key_label)
        row_key.addWidget(self.key_combo)

        row_step = QHBoxLayout()
        row_step.addWidget(step_label)
        row_step.addWidget(self.step_spin)

        row_dir = QHBoxLayout()
        row_dir.addWidget(dir_label)
        row_dir.addWidget(self.dir_edit)
        row_dir.addWidget(self.browse_btn)
        
        row_mode = QHBoxLayout()
        row_mode.addWidget(mode_label)
        row_mode.addWidget(self.mode_combo)

        main = QVBoxLayout()
        main.addLayout(row_key)
        main.addLayout(row_step)
        main.addLayout(row_dir)
        main.addLayout(row_mode)
        main.addWidget(buttons)
        self.setLayout(main)

    def _browse_for_dir(self):
        start_dir = self.dir_edit.text().strip() or ""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            start_dir,
        )
        if chosen:
            self.dir_edit.setText(chosen)

    def values(self):
        """
        Return (key, step, out_dirpath) or (None, None, None) if invalid.
        """
        display = self.key_combo.currentText().strip()
        step = float(self.step_spin.value())
        out_text = self.dir_edit.text().strip()
        mode_selected = self.mode_combo.currentText().strip()
    
        if not display or not out_text:
            return None, None, None, None
    
        key = self.key_map.get(display)
        if key is None:
            return None, None, None, None
        
        mode = self.mode_map.get(mode_selected)
    
        return key, step, Path(out_text), mode

    @classmethod
    def get_values(cls, parent=None, keys=None, step_default=None, dir_default=None, title=None):
        dlg = cls(
            parent=parent,
            keys=keys,
            step_default=step_default,
            dir_default=dir_default,
            title=title or "Select export options",
        )
        result = dlg.exec_()
        if result == QDialog.Accepted:
            key, step, out_dir, mode = dlg.values()
            return True, key, step, out_dir, mode
        return False, None, None, None, None

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
            logger.info(f"Config setting {key} changed to {val}")
        self.accept()
