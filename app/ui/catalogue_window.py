"""
Standalone window for browsing directories and selecting scan files.

Provides drive/root view and emits selection signals back to the UI.
"""
from pathlib import Path

from PyQt5.QtCore import QDir, QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileSystemModel,
    QMainWindow,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


class CatalogueWindow(QMainWindow):
    """
    Standalone file/directory browser window.

    Emits
    -----
    fileActivated(str)
        A file was double-clicked.
    dirActivated(str)
        A directory was double-clicked.
    """

    fileActivated = pyqtSignal(str)
    dirActivated = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        start_folder=None,
        name_filters=None,
        show_only_filtered_files=True,
        auto_navigate_on_dir_double_click=True,
    ):
        super().__init__(parent)

        self.setWindowTitle("Catalogue View")
        self.resize(300, 900)

        self._auto_nav = auto_navigate_on_dir_double_click

        # ----- central widget / layout
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        # ----- model
        self.model = QFileSystemModel(self)
        self.model.setRootPath(QDir.rootPath())

        if name_filters is None:
            # tweak this to whatever you care about
            name_filters = ["*.json", "*.hdr", "*.csv"]

        self.model.setNameFilters(name_filters)
        self.model.setNameFilterDisables(not show_only_filtered_files)

        # ----- view
        self.tree = QTreeView(self)
        self.tree.setModel(self.model)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)
        self.tree.setColumnWidth(0, 320)
        self.tree.doubleClicked.connect(self._on_double_clicked)

        layout.addWidget(self.tree)
        self.tree.setRootIndex(QModelIndex())

    # ------------------------------------------------------------------ API

    def set_root(self, folder):
        """Set the visible root directory."""
        folder = str(Path(folder))
        idx = self.model.index(folder)
        if idx.isValid():
            self.tree.setRootIndex(idx)
            self.tree.expand(idx)
            self.tree.scrollTo(idx)
            self.tree.setCurrentIndex(idx)

    # ------------------------------------------------------------------ internals

    def _on_double_clicked(self, index):
        path = self.model.filePath(index)
        if not path:
            return

        if self.model.isDir(index):
            self.dirActivated.emit(path)
        else:
            self.fileActivated.emit(path)

    def closeEvent(self, event):
        """
        Ensure the main GUI knows this window is gone.

        Does NOT quit the app (unlike your old GUI's behaviour).
        """

        parent = self.parent()
        if parent is not None and hasattr(parent, "_catalogue_window"):
            parent._catalogue_window = None

        event.accept()
