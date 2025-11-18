"""
Created on Wed Oct 22 09:47:46 2025

@author: russj
"""


from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget

from ..interface import ToolDispatcher
from ..models import CurrentContext
from .util_windows import ClosableWidgetWrapper, ImageCanvas2D, SpectralImageCanvas


class BasePage(QWidget):
    """
    Common base: holds a QSplitter with left/right/(optional)third widgets,
    a per-page ToolDispatcher, and a safe teardown().
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._splitter = QSplitter(Qt.Horizontal, self)
        self._left = None     # SpectralImageCanvas
        self._right = None    # ImageCanvas2D
        self._third = None    # InfoTable or other QWidget
        self._dispatcher = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._splitter)

        # Data models available to the page (set by controller)
        self._cxt = CurrentContext()

    @property
    def cxt(self) -> CurrentContext | None:
        return self._cxt

    @cxt.setter
    def cxt(self, new_cxt: CurrentContext | None):
        self._cxt = new_cxt

    # all existing code can keep using self.current_obj
    @property
    def current_obj(self):
        return self._cxt.current if self._cxt else None



    # --- building helpers ----------------------------------------------------
    def _add_left(self, w: QWidget):
        self._left = w
        self._splitter.addWidget(w)

    def _add_right(self, w: QWidget):
        self._right = w
        self._splitter.addWidget(w)

    def _add_third(self, w: QWidget):
        self._third = w
        self._splitter.addWidget(w)

    # --- lifecycle -----------------------------------------------------------
    def activate(self):
        """
        Called when the page becomes visible/active.
        Recreate dispatcher so tools can (re)bind safely.
        """
        if isinstance(self._left, SpectralImageCanvas):
            self._dispatcher = ToolDispatcher(self._left)
        else:
            self._dispatcher = None

    def teardown(self):
        """
        Must be called on tab switch (or when closing the page).
        Cancels any active tools and disconnects temporary bindings.
        """
        # Rect selector / canvas interactions
        if isinstance(self._left, SpectralImageCanvas):
            # Cancel an active RectangleSelector cleanly
            self._left.cancel_rect_select()
            # Clear any temporary tool callbacks
            if self._dispatcher:
                self._dispatcher.clear()

        # Nothing to explicitly disconnect on ImageCanvas2D/InfoTable by default
    def _add_closable_widget(self, raw_widget: QWidget, title: str, closeable=True):
        """
        Wraps a widget in a ClosableWidgetWrapper and adds it as a *secondary*
        widget to the QSplitter, usually alongside self._right or self._third.
        """
         # Import locally for clean API

        wrapper = ClosableWidgetWrapper(raw_widget, title=title, parent=self, closeable=closeable)

        # Connect the wrapper's closed signal to the page's removal handler
        wrapper.closed.connect(self.remove_widget)

        # Add the wrapper to the splitter

        self._splitter.addWidget(wrapper)

        return wrapper


    def remove_widget(self, w: QWidget):
        """
        Safely remove a widget (which might be the ClosableWidgetWrapper) 
        from the QSplitter and clean up its memory. (Same as previous version)
        """
        """
        Safely remove a widget from the QSplitter and clean up its memory.
        This is crucial for dynamically added widgets like correlation canvases.
        """
        # 1. Find the widget in the splitter (it might be a wrapped item)
        idx = self._splitter.indexOf(w)
        if idx == -1:
            return

        # 2. Remove from layout and disconnect from Python
        w.setParent(None)
        w.deleteLater()

        # 3. If the removed widget was one of the three primary slots, clear the reference
        if w is self._left:
            self._left = None
        elif w is self._right:
            self._right = None
        elif w is self._third:
            self._third = None

    def update_display(self, key='mask'):
        pass
    # --- accessors for the controller ---------------------------------------
    @property
    def left_canvas(self) -> SpectralImageCanvas:
        return self._left

    @property
    def right_canvas(self) -> ImageCanvas2D:
        return self._right

    @property
    def table(self) -> QWidget:
        return self._third

    @property
    def dispatcher(self) -> ToolDispatcher:
        return self._dispatcher

