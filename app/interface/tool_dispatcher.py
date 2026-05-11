"""
Dispatches UI actions to the correct data object.

Maps button and interaction signals to tool functions,
ensuring the correct object (Raw, Processed, Hole) receives the operation.
"""


class ToolDispatcher:
    """
   Lightweight router for canvas events.

   Accepts any canvas object that exposes the expected callback registration
   API (set_click_callback, set_rect_callback, etc.). It does not depend on
   Qt or SpectralImageCanvas directly.
   """
    def __init__(self):
        self._canvases = []
        # permanent
        self._perm_click = None
        self._perm_right = None
        self._perm_double = None
        self._perm_rect  = None
        self._perm_poly  = None
        self._perm_circle = None
        self._perm_line = None
        # temporary (tool)
        self._tmp_click = None
        self._tmp_right = None
        self._tmp_double = None 
        self._tmp_rect  = None
        self._tmp_poly  = None
        self._tmp_circle = None
        self._tmp_line = None
        # bind shims

    # Manage the canvas list

    def add_canvas(self, canvas):
        if canvas not in self._canvases:
            self._canvases.append(canvas)
            self._wire(canvas)

    def remove_canvas(self, canvas):
        if canvas in self._canvases:
            self._canvases.remove(canvas)
            canvas.cancel_rect_select()
            canvas.cancel_polygon_select()
            canvas.cancel_circle_select()
            canvas.cancel_line_select()

    def _wire(self, canvas):
        canvas.on_single_click       = self._shim_click
        canvas.on_right_click        = self._shim_right
        canvas.on_double_click       = self._shim_double
        canvas.on_rectangle_selected = self._shim_rect
        canvas.on_polygon_finished   = self._shim_polygon
        canvas.on_circle_selected    = self._shim_circle
        canvas.on_line_selected      = self._shim_line

    # Iteralte all canvases to start selectors
    def start_rect_select(self, **kwargs):
        for canvas in self._canvases:
            canvas.start_rect_select(**kwargs)

    def start_polygon_select(self):
        for canvas in self._canvases:
            canvas.start_polygon_select()

    def start_circle_select(self):
        for canvas in self._canvases:
            canvas.start_circle_select()

    def start_line_select(self):
        for canvas in self._canvases:
            canvas.start_line_select()


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
    def set_double_click(self, func, *, temporary=True):   # ← new
        if temporary:
            self._tmp_double = func
        else:
            self._perm_double = func
    def set_rect(self, func, *, temporary=True):
        if temporary:
            self._tmp_rect = func
        else:
            self._perm_rect = func
    def set_polygon(self, func, *, temporary=True):
        if temporary:
            self._tmp_poly = func
        else:
            self._perm_poly = func
    def set_circle(self, func, *, temporary=True):
        if temporary:
            self._tmp_circle = func
        else:
            self._perm_circle = func
    def set_line(self, func, *, temporary=True):
        if temporary:
            self._tmp_line = func
        else:
            self._perm_line = func

    # granular clears for tools
    def clear_temp_click(self): self._tmp_click = None
    def clear_temp_right(self): self._tmp_right = None
    def clear_temp_double(self): self._tmp_double = None
    def clear_temp_rect(self):  self._tmp_rect  = None
    def clear_temp_polygon(self): self._tmp_poly = None
    def clear_temp_line(self): self._tmp_line = None


    # Iterate through canvases and clear all selectors
    def clear_all_temp(self):
        self._tmp_click = self._tmp_right = self._tmp_double = None
        self._tmp_rect  = self._tmp_poly  = self._tmp_circle = self._tmp_line = None
        for canvas in self._canvases:
            canvas.cancel_rect_select()
            canvas.cancel_polygon_select()
            canvas.cancel_circle_select()
            canvas.cancel_line_select()

    # clear all for teardown methods
    def clear(self):
        self._perm_click = self._perm_right = self._perm_double = None
        self._perm_rect  = self._perm_poly  = self._perm_circle = self._perm_line = None
        self.clear_all_temp()

    # shims prefer temp, then perm
    def _shim_click(self, y, x):
        f = self._tmp_click or self._perm_click
        if callable(f): f(y, x)
    def _shim_right(self, y, x):
        f = self._tmp_right or self._perm_right
        if callable(f): f(y, x)
    def _shim_double(self, y, x):                          
        f = self._tmp_double or self._perm_double
        if callable(f): f(y, x)
    def _shim_rect(self, y0, y1, x0, x1):
        f = self._tmp_rect or self._perm_rect
        if callable(f): f(y0, y1, x0, x1)
    def _shim_polygon(self, vertices_rc):
        f = self._tmp_poly or self._perm_poly
        if callable(f): f(vertices_rc)
    def _shim_circle(self, cy, cx, r):
        f = self._tmp_circle or self._perm_circle
        if callable(f): f(cy, cx, r)
    def _shim_line(self, y0, x0, y1, x1):
        f = self._tmp_line or self._perm_line
        if callable(f): f(y0, x0, y1, x1)
