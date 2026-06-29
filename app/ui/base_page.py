"""
Base class for CoreSpecViewer pages.

Provides access to shared context, tool dispatch, and common UI layout helpers.
"""
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter, QVBoxLayout, QWidget
from PyQt5 import sip

from ..interface import ToolDispatcher
from ..models import CurrentContext
from .util_windows import (ClosableWidgetWrapper, 
                            PopoutWindow)
from .display_canvases import ImageCanvas2D, SpectralImageCanvas, BaseMatplotlibCanvas
logger = logging.getLogger(__name__)
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
        self._popouts = []

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
        self._dispatcher = ToolDispatcher()
        if self._left is not None and isinstance(self._left, BaseMatplotlibCanvas):
            self._dispatcher.add_canvas(self._left)
    def teardown(self):
        """
        Must be called on tab switch (or when closing the page).
        Cancels any active tools and disconnects temporary bindings.
        """
        if self._dispatcher:
            self._dispatcher.clear()
        for w in list(self._popouts):
            w.close()
        self._popouts.clear()

        # Nothing to explicitly disconnect on ImageCanvas2D/InfoTable by default
    def _add_closable_widget(self, raw_widget: QWidget, title: str, closeable=True, popoutable=False,
                             index = None):
        """
        Wraps a widget in a ClosableWidgetWrapper and adds it as a *secondary*
        widget to the QSplitter, usually alongside self._right or self._third.
        """
         # Import locally for clean API

        wrapper = ClosableWidgetWrapper(raw_widget, title=title, parent=self, closeable=closeable,
                                        popoutable=popoutable)

        # Connect the wrapper's closed signal to the page's removal handler
        wrapper.closed.connect(self.remove_widget)
        wrapper.destroyed.connect(self._purge_dead_refs)  # truth: object is gone
        # Add the wrapper to the splitter
        # Add the wrapper to the splitter at the specified index
        if index is None:
            self._splitter.addWidget(wrapper)
        else:
            self._splitter.insertWidget(index, wrapper)
       

        return wrapper


    def remove_widget(self, w: QWidget):
        """
        Safely remove a widget (which might be the ClosableWidgetWrapper) 
        from the QSplitter and clean up its memory.
        """
        if w in self._popouts:
            self._popouts.remove(w)
            w.setParent(None)
            w.deleteLater()
            return
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

    def _purge_dead_refs(self, *_):
        # destroyed has fired — the wrapper's C++ side is gone. Drop it by identity.
        self._popouts = [w for w in self._popouts if not sip.isdeleted(w)]

    
    def _handle_popout_request(self, wrapper: ClosableWidgetWrapper):
        """
        Handles the signal from a ClosableWidgetWrapper to pop its content out 
        into a new, independent QMainWindow. This is a generic handler 
        for all pages. Canvas remains registered on page, only de-registered on close
        """
        wrapper.setParent(None)
        wrapper.setWindowFlags(Qt.Window)
        wrapper.setWindowTitle(wrapper.label.text())
        wrapper.show()
        self._popouts.append(wrapper)
        logger.info(f"{wrapper.label.text()} popped out")



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

