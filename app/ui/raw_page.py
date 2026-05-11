"""
UI page for displaying and interacting with RawObject data.

Shows RGB preview, allows reflectance conversion, and box-level operations.
"""

import logging
logger = logging.getLogger(__name__)

from .base_page import BasePage
from .display_canvases import SpectralImageCanvas
from .util_windows import SpectrumWindow


class RawPage(BasePage):
    """
    
      left  = SpectralImageCanvas  (reflectance cube view, dbl-click spectra)
      
    Use ribbon actions to call methods on the active page via the controller.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Build the three-pane layout
        self._add_left(SpectralImageCanvas(self))


    def update_display(self, key = 'mask'):
        if self.current_obj is None:
            logger.warning(f"Current object is None")
            return
        if not self.current_obj.is_raw:
            logger.warning(f"Current object is not raw")
            return
        logger.debug(f"shape of display reflectance {self.current_obj.get_display_reflectance().shape}")
        self.left_canvas.show_rgb(self.current_obj.get_display_reflectance())

    def activate(self):
        super().activate()
        if self.dispatcher:
            def _double_click(y, x):
                if self.current_obj is None:
                    return
                spec = self.current_obj.get_display_reflectance()[y, x, :]
                if not hasattr(self, "spec_win"):
                    self.spec_win = SpectrumWindow(self)
                self.spec_win.plot_spectrum(self.current_obj.bands, spec, title="Spectrum Viewer", label=f"({y}, {x})")
            self.dispatcher.set_double_click(_double_click, temporary=False)



