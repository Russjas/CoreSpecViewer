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
    """Ribbon with multiple tabs, each containing a toolbar.
    
    Entry formats:
      ("button", label, callback)
      ("button", label, callback, tooltip)
      ("menu",   label, [(item_label, item_callback), ...])
      ("menu",   label, [(item_label, item_callback), ...], tooltip)
      
    Submenu items can be:
      (sublabel, subcb)
      (sublabel, subcb, subtooltip)
      ("menu", sublabel, nested_submenu_list)
      ("menu", sublabel, nested_submenu_list, tooltip)
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
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(bar, 0, Qt.AlignCenter)  # Center align the toolbar
        return page

    def _populate(self, bar: QToolBar, spec):
        """
        Populate the given toolbar with actions from the spec.
    
        Each entry in spec can be:
    
          ("button", label, callback)
          ("button", label, callback, tooltip)
    
          ("menu", label, submenu_list)
          ("menu", label, submenu_list, tooltip)
    
        where submenu_list is:
          [
            (sublabel, subcb),
            (sublabel, subcb, subtooltip),
            ("menu", sublabel, nested_submenu_list),
            ("menu", sublabel, nested_submenu_list, tooltip),
            ...
          ]
        """
        for entry in spec:
            if not entry:
                continue
    
            kind = entry[0]
    
            if kind == "button":
                # ("button", label, callback[, tooltip])
                if len(entry) == 4:
                    _kind, label, callback, tooltip = entry
                else:
                    _kind, label, callback = entry
                    tooltip = None
    
                act = QAction(label, bar)
                act.triggered.connect(callback)
                if tooltip:
                    act.setToolTip(tooltip)
                    act.setStatusTip(tooltip)
                bar.addAction(act)
    
            elif kind == "menu":
                # ("menu", label, submenu_list[, tooltip])
                if len(entry) == 4:
                    _kind, label, submenu, tooltip = entry
                else:
                    _kind, label, submenu = entry
                    tooltip = None
    
                top = QAction(label, bar)
                if tooltip:
                    top.setToolTip(tooltip)
                    top.setStatusTip(tooltip)
    
                menu = QMenu(label, bar)
    
                # Process submenu items (may include nested menus)
                self._populate_menu(menu, submenu, bar)
    
                top.setMenu(menu)
                bar.addAction(top)

    def _populate_menu(self, menu: QMenu, items, bar: QToolBar):
        """
        Recursively populate a QMenu with items.
        
        Items can be:
          (label, callback)
          (label, callback, tooltip)
          ("menu", label, submenu_list)
          ("menu", label, submenu_list, tooltip)
        """
        for item in items:
            if not item:
                continue
            
            # Check if this is a nested menu
            if item[0] == "menu":
                # ("menu", label, submenu_list[, tooltip])
                if len(item) == 4:
                    _kind, label, submenu_items, tooltip = item
                else:
                    _kind, label, submenu_items = item
                    tooltip = None
                
                submenu = QMenu(label, menu)
                if tooltip:
                    submenu.setToolTip(tooltip)
                    submenu.setStatusTip(tooltip)
                
                # Recursively populate the submenu
                self._populate_menu(submenu, submenu_items, bar)
                menu.addMenu(submenu)
            
            else:
                # Regular menu item (action)
                if len(item) == 3:
                    label, callback, tooltip = item
                else:
                    label, callback = item
                    tooltip = None

                action = QAction(label, bar)
                action.triggered.connect(callback)
                if tooltip:
                    action.setToolTip(tooltip)
                    action.setStatusTip(tooltip)
                menu.addAction(action)

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
        tb.setToolButtonStyle(Qt.ToolButtonIconOnly)
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
        if pos == 'left':
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

        Each entry in spec can be:

          ("button", label, callback)
          ("button", label, callback, tooltip)

          ("menu", label, submenu_list)
          ("menu", label, submenu_list, tooltip)

        where submenu_list is:
          [
            (sublabel, subcb),
            (sublabel, subcb, subtooltip),
            ("menu", submenu_label, nested_submenu_list),  # <-- NEW: nested menu
            ("menu", submenu_label, nested_submenu_list, tooltip),
            ...
          ]
        """
        for entry in spec:
            if not entry:
                continue

            kind = entry[0]

            if kind == "button":
                # ("button", label, callback[, tooltip])
                if len(entry) == 4:
                    _kind, label, callback, tooltip = entry
                else:
                    _kind, label, callback = entry
                    tooltip = None

                act = QAction(label, bar)
                act.triggered.connect(callback)
                if tooltip:
                    act.setToolTip(tooltip)
                    act.setStatusTip(tooltip)
                bar.addAction(act)

            elif kind == "menu":
                # ("menu", label, submenu_list[, tooltip])
                if len(entry) == 4:
                    _kind, label, submenu, tooltip = entry
                else:
                    _kind, label, submenu = entry
                    tooltip = None

                top = QAction(label, bar)
                if tooltip:
                    top.setToolTip(tooltip)
                    top.setStatusTip(tooltip)

                menu = QMenu(label, bar)

                # Process submenu items (may include nested menus)
                self._populate_menu(menu, submenu, bar)

                top.setMenu(menu)
                bar.addAction(top)

    def _populate_menu(self, menu: QMenu, items, bar: QToolBar):
        """
        Recursively populate a QMenu with items.
        
        Items can be:
          (label, callback)
          (label, callback, tooltip)
          ("menu", label, submenu_list)
          ("menu", label, submenu_list, tooltip)
        """
        for item in items:
            if not item:
                continue
            
            # Check if this is a nested menu
            if item[0] == "menu":
                # ("menu", label, submenu_list[, tooltip])
                if len(item) == 4:
                    _kind, label, submenu_items, tooltip = item
                else:
                    _kind, label, submenu_items = item
                    tooltip = None
                
                submenu = QMenu(label, menu)
                if tooltip:
                    submenu.setToolTip(tooltip)
                    submenu.setStatusTip(tooltip)
                
                # Recursively populate the submenu
                self._populate_menu(submenu, submenu_items, bar)
                menu.addMenu(submenu)
            
            else:
                # Regular menu item (action)
                if len(item) == 3:
                    label, callback, tooltip = item
                else:
                    label, callback = item
                    tooltip = None

                action = QAction(label, bar)
                action.triggered.connect(callback)
                if tooltip:
                    action.setToolTip(tooltip)
                    action.setStatusTip(tooltip)
                menu.addAction(action)

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
            
class GroupedRibbon(QTabWidget):
    """
    Hybrid ribbon that uses tabbed navigation with grouped layouts.
    
    Compatible API with Ribbon and Groups:
      - add_tab(name, entries)
      - add_global_actions(perm_act_list, pos='left'|'right')
      
    Special feature: Use None as an entry to create a visual group separator
    
    Example:
        ribbon.add_tab('Processing', [
            ("button", "Crop", self.crop),
            ("button", "Process", self.process),
            None,  # <- Group separator
            ("button", "New mask", self.mask),
            ("button", "Enhance", self.enhance),
        ])
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.North)
        self.setMovable(False)
        self.setDocumentMode(True)
        self.bars = {}  # tab_name -> list of QToolBars

    def _create_bar(self):
        """Create a toolbar for a group."""
        bar = QToolBar(self)
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        bar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return bar

    def _populate(self, bar: QToolBar, spec):
        """
        Populate the given toolbar with actions from the spec.
    
        Each entry in spec can be:
    
          ("button", label, callback)
          ("button", label, callback, tooltip)
    
          ("menu", label, submenu_list)
          ("menu", label, submenu_list, tooltip)
    
        where submenu_list is:
          [
            (sublabel, subcb),
            (sublabel, subcb, subtooltip),
            ("menu", sublabel, nested_submenu_list),
            ("menu", sublabel, nested_submenu_list, tooltip),
            ...
          ]
        """
        for entry in spec:
            if not entry:
                continue
    
            kind = entry[0]
    
            if kind == "button":
                # ("button", label, callback[, tooltip])
                if len(entry) == 4:
                    _kind, label, callback, tooltip = entry
                else:
                    _kind, label, callback = entry
                    tooltip = None
    
                act = QAction(label, bar)
                act.triggered.connect(callback)
                if tooltip:
                    act.setToolTip(tooltip)
                    act.setStatusTip(tooltip)
                bar.addAction(act)
    
            elif kind == "menu":
                # ("menu", label, submenu_list[, tooltip])
                if len(entry) == 4:
                    _kind, label, submenu, tooltip = entry
                else:
                    _kind, label, submenu = entry
                    tooltip = None
    
                top = QAction(label, bar)
                if tooltip:
                    top.setToolTip(tooltip)
                    top.setStatusTip(tooltip)
    
                menu = QMenu(label, bar)
    
                # Process submenu items (may include nested menus)
                self._populate_menu(menu, submenu, bar)
    
                top.setMenu(menu)
                bar.addAction(top)

    def _populate_menu(self, menu: QMenu, items, bar: QToolBar):
        """
        Recursively populate a QMenu with items.
        
        Items can be:
          (label, callback)
          (label, callback, tooltip)
          ("menu", label, submenu_list)
          ("menu", label, submenu_list, tooltip)
        """
        for item in items:
            if not item:
                continue
            
            # Check if this is a nested menu
            if item[0] == "menu":
                # ("menu", label, submenu_list[, tooltip])
                if len(item) == 4:
                    _kind, label, submenu_items, tooltip = item
                else:
                    _kind, label, submenu_items = item
                    tooltip = None
                
                submenu = QMenu(label, menu)
                if tooltip:
                    submenu.setToolTip(tooltip)
                    submenu.setStatusTip(tooltip)
                
                # Recursively populate the submenu
                self._populate_menu(submenu, submenu_items, bar)
                menu.addMenu(submenu)
            
            else:
                # Regular menu item (action)
                if len(item) == 3:
                    label, callback, tooltip = item
                else:
                    label, callback = item
                    tooltip = None

                action = QAction(label, bar)
                action.triggered.connect(callback)
                if tooltip:
                    action.setToolTip(tooltip)
                    action.setStatusTip(tooltip)
                menu.addAction(action)

    def _create_group_widget(self, bar: QToolBar, group_name: str = None):
        """
        Wrap a toolbar in a group widget with optional label and separators.
        
        Args:
            bar: The toolbar to wrap
            group_name: Optional name to display above the toolbar
        
        Returns:
            QWidget containing the styled group
        """
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
        inner.setContentsMargins(10, 0, 10, 0)
        inner.setSpacing(2)
        
        if group_name:
            label = QLabel(group_name)
            label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            label.setStyleSheet("QLabel { font-weight: bold; }")
            inner.addWidget(label)
        
        inner.addWidget(bar)
        
        outer.addWidget(left_sep)
        outer.addLayout(inner)
        outer.addWidget(right_sep)
        
        group_widget.adjustSize()
        group_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        
        return group_widget

    def add_tab(self, name, entries):
        """
        Create a new tab with grouped toolbars.
        
        Use None in the entries list to create group separators.
        
        Args:
            name: Name of the tab
            entries: List of button/menu specs, with None for group breaks
        
        Example:
            ribbon.add_tab('Processing', [
                ("button", "Crop", self.crop),
                None,  # Group break
                ("button", "Mask", self.mask),
            ])
        """
        # Split entries into groups based on None separators
        groups = []
        current_group = []
        
        for entry in entries:
            if entry is None:
                if current_group:
                    groups.append(current_group)
                    current_group = []
            else:
                current_group.append(entry)
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        # If no separators were used, treat as single group
        if not groups:
            groups = [entries]
        
        # Create the tab page
        page = QWidget(self)
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(0)
        page_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        # Store all bars for this tab
        self.bars[name] = []
        
        # Create a toolbar for each group
        for i, group_entries in enumerate(groups):
            bar = self._create_bar()
            self.bars[name].append(bar)
            
            # Populate the toolbar
            self._populate(bar, group_entries)
            
            # Wrap in styled group widget (no label for cleaner look)
            group_widget = self._create_group_widget(bar)
            
            # Add to page layout
            page_layout.addWidget(group_widget)
        
        # Add stretch at the end
        page_layout.addStretch(1)
        
        # Add the tab
        self.addTab(page, name)

    def add_global_actions(self, perm_act_list, pos='left'):
        """Add permanent buttons (e.g., Open/Save) to the ribbon corner."""
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
