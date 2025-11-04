from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QTabWidget, QWidget, QHBoxLayout, QToolBar, QAction, QMenu,QToolButton

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
            
        
        
 