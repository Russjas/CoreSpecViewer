from util_windows import SpectralImageCanvas



class ToolDispatcher:
    def __init__(self, canvas):
        self.canvas = canvas
        # permanent
        self._perm_click = None
        self._perm_right = None
        self._perm_rect  = None
        # temporary (tool)
        self._tmp_click = None
        self._tmp_right = None
        self._tmp_rect  = None
        # bind shims
        self.canvas.on_single_click = self._shim_click
        self.canvas.on_right_click  = self._shim_right
        self.canvas.on_rectangle_selected = self._shim_rect

    # setters — choose temporary (default) or permanent
    def set_single_click(self, func, *, temporary=True):
        if temporary:
            self._tmp_click = func
        else:
            self._perm_click = func
    def set_right_click(self, func, *, temporary=True):
        if temporary:
            self._tmp_right = func 
        else:
            self._perm_right = func
    def set_rect(self, func, *, temporary=True):
        if temporary:
            self._tmp_rect = func
        else:
            self._perm_rect = func

    # granular clears for tools
    def clear_temp_click(self): self._tmp_click = None
    def clear_temp_right(self): self._tmp_right = None
    def clear_temp_rect(self):  self._tmp_rect  = None
    def clear_all_temp(self):
        self._tmp_click = self._tmp_right = self._tmp_rect = None

    # clear all for teardown methods
    def clear(self):
        self._perm_click = self._perm_right = self._perm_rect = None
        self.clear_all_temp()

    # shims prefer temp, then perm
    def _shim_click(self, y, x):
        f = self._tmp_click or self._perm_click
        if callable(f): f(y, x)
    def _shim_right(self, y, x):
        f = self._tmp_right or self._perm_right
        if callable(f): f(y, x)
    def _shim_rect(self, y0, y1, x0, x1):
        f = self._tmp_rect or self._perm_rect
        if callable(f): f(y0, y1, x0, x1)