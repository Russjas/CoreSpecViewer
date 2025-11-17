# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:37:07 2025

@author: russj
"""

from .util_windows import SpectralImageCanvas
from .base_page import BasePage

class RawPage(BasePage):
    """
    Mirrors your ProcessRaw composition:
      left  = SpectralImageCanvas  (reflectance cube view, dbl-click spectra)
      right = ImageCanvas2D        (product/preview)
      third = InfoTable            (cache/status)
    Use ribbon actions to call methods on the active page via the controller.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Build the three-pane layout
        self._add_left(SpectralImageCanvas(self))
        

    def update_display(self, key = 'mask'):
        if self.current_obj is None:
            return
        if not self.current_obj.is_raw:
            return
        self.left_canvas.show_rgb(self.current_obj.get_display_reflectance(), self.current_obj.bands)

    
        
