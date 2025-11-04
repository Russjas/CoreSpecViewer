from util_windows import SpectralImageCanvas

class ToolDispatcher:
    """Allow multiple tools to share SpectralImageCanvas callbacks safely."""
    def __init__(self, canvas: SpectralImageCanvas):
        self.canvas = canvas
        self._on_click = None
        self._on_right = None
        self._on_rect = None
        # Bind shims
        self.canvas.on_single_click = self._shim_click
        self.canvas.on_right_click  = self._shim_right
        self.canvas.on_rectangle_selected = self._shim_rect

    def set_single_click(self, func): self._on_click = func
    def set_right_click(self, func): self._on_right = func
    def set_rect(self, func): self._on_rect = func
    def clear(self): self._on_click = self._on_right = self._on_rect = None

    def _shim_click(self, y, x):
        if callable(self._on_click): self._on_click(y, x)
    def _shim_right(self, y, x):
        if callable(self._on_right): self._on_right(y, x)
    def _shim_rect(self, y0, y1, x0, x1):
        if callable(self._on_rect): self._on_rect(y0, y1, x0, x1)
