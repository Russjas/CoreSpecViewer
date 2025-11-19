"""
Main toolbar (ribbon) used to trigger high-level spectral and mask tools.

Provides button groups for crop, mask, processing, classification, and saving.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class Ribbon(QTabWidget):
    """Ribbon with three tabs: Raw, Mask, Visualise.
    Entry formats:
      ("button", label, callback)
      ("menu",   label, [(item_label, item_callback), ...])
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.North)
        self.setMovable(False)
        self.setDocumentMode(True)
        self.bars = {}


    def _create_bar(self):
        bar = QToolBar()
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        return bar

    def wrap(self, bar):
        page = QWidget(); lay = QHBoxLayout(page); lay.setContentsMargins(8,8,8,8)
        lay.addWidget(bar); lay.addStretch(1); return page

    def _populate(self, bar: QToolBar, spec):
        for kind, label, payload in spec:
            if kind == "button":
                act = QAction(label, bar)
                act.triggered.connect(payload)
                bar.addAction(act)
            elif kind == "menu":
                top = QAction(label, bar)
                menu = QMenu(label, bar)
                for sublabel, subcb in payload:
                    sub = QAction(sublabel, bar); sub.triggered.connect(subcb)
                    menu.addAction(sub)
                top.setMenu(menu)
                bar.addAction(top)
    def add_tab(self, name, entries):
        new_tab = self._create_bar()
        self.bars[name] = new_tab
        self.addTab(self.wrap(new_tab), name)
        self._populate(self.bars[name], entries)

    def add_global_actions(self, perm_act_list, pos='left'):
        """Add permanent buttons (e.g., Open/Save) to the ribbon corner."""
        tb = QToolBar(self)
        tb.setMovable(False)
        tb.setFloatable(False)
        #tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        #tb.setStyleSheet("QToolBar { background: transparent; border: 0; }"
         #                "QToolButton { padding: 2px; margin: 0; }")
        tb.setStyleSheet("""
QToolBar {
    background: #d8d8d8;   /* slightly lighter grey than Qt default (~#d4d4d4) */
    border: none;
}

QToolButton {
    padding: 2px;
    margin: 0;
}
""")
        if pos=='left':
            for a in perm_act_list:
                tb.addAction(a)
            self.setCornerWidget(tb, Qt.TopLeftCorner)
        else:
            for a in perm_act_list:
                tb.addAction(a)
            self.setCornerWidget(tb, Qt.TopRightCorner)

class Groups(QTabWidget):
    """
    Ribbon-style control that shows all 'tabs' as grouped blocks
    inside a single visible tab.

    Public API compatible with Ribbon:
      - add_tab(name, entries)
      - add_global_actions(perm_act_list, pos='left'|'right')
      - _create_bar(), _populate(), wrap()
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTabPosition(QTabWidget.North)
        self.setMovable(False)
        self.setDocumentMode(True)

        self.bars = {}  # name -> QToolBar for that group

        # ----- Single tab that will host all the groups in one row -----
        page = QWidget(self)
        self._group_layout = QHBoxLayout(page)
        self._group_layout.setContentsMargins(8, 8, 8, 8)
        self._group_layout.setSpacing(0)  # space between groups
        self._group_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        # Visible single tab; label it however you like
        self.addTab(page, "CoreSpecViewer Controls")

    # ---------- API-compat helpers ----------

    def _create_bar(self):
        """
        For each 'tab' we create a separate QToolBar that will live inside
        its own group widget.
        """
        bar = QToolBar(self)
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return bar

    def wrap(self, bar):
        """
        Kept only for API compatibility with Ribbon.
        In this implementation, the bar's parent is the group widget.
        """
        return bar.parent() if bar is not None else None

    def _populate(self, bar: QToolBar, spec):
        """
        Populate the given toolbar with actions from the spec.
        """
        for kind, label, payload in spec:
            if kind == "button":
                act = QAction(label, bar)
                act.triggered.connect(payload)
                bar.addAction(act)

            elif kind == "menu":
                top = QAction(label, bar)
                menu = QMenu(label, bar)
                for sublabel, subcb in payload:
                    sub = QAction(sublabel, bar)
                    sub.triggered.connect(subcb)
                    menu.addAction(sub)
                top.setMenu(menu)
                bar.addAction(top)

    # ---------- Public API (same signatures as Ribbon) ----------

    def add_tab(self, name, entries):
        """
        Create a new 'group' block inside the single tab:
    
            | Raw        |
            | buttons... |   | Masking    |   | Visualise |   | Hole operations |
        """

        # Create toolbar for this group
        bar = self._create_bar()
        self.bars[name] = bar

        # ----- Group widget structure -----
        group_widget = QWidget(self)

        # Outer layout: [ VLine |  inner VBox  | VLine ]
        outer = QHBoxLayout(group_widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        left_sep = QFrame(self)
        left_sep.setFrameShape(QFrame.VLine)
        left_sep.setFrameShadow(QFrame.Sunken)
        left_sep.setLineWidth(1)
        left_sep.setMidLineWidth(0)

        right_sep = QFrame(self)
        right_sep.setFrameShape(QFrame.VLine)
        right_sep.setFrameShadow(QFrame.Sunken)
        right_sep.setLineWidth(1)
        right_sep.setMidLineWidth(0)

        inner = QVBoxLayout()
        inner.setContentsMargins(10, 0, 10, 0)  # padding inside group
        inner.setSpacing(2)

        label = QLabel(name)
        label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        label.setStyleSheet("QLabel { font-weight: bold; }")

        inner.addWidget(label)
        inner.addWidget(bar)

        outer.addWidget(left_sep)
        outer.addLayout(inner)
        outer.addWidget(right_sep)

        # Populate toolbar
        self._populate(bar, entries)

        # Let the group size itself to its content, then fix that width
        group_widget.adjustSize()
        group_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        # Add the group to the horizontal layout (left/top aligned globally)
        self._group_layout.addWidget(group_widget)

    def add_global_actions(self, perm_act_list, pos='left'):
        """
        Identical to Ribbon.add_global_actions – keeps your green
        Open/Save/Info/Settings toolbars exactly as they are.
        """
        tb = QToolBar(self)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
        tb.setStyleSheet("""
QToolBar {
    background: #d8d8d8;
    border: none;
}
QToolButton {
    padding: 2px;
    margin: 0;
}
""")
        for a in perm_act_list:
            tb.addAction(a)

        if pos == 'left':
            self.setCornerWidget(tb, Qt.TopLeftCorner)
        else:
            self.setCornerWidget(tb, Qt.TopRightCorner)
