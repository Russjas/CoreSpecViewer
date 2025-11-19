"""
CoreSpecViewer Interface Package
==================================

This package contains the lightweight interface layer that connects the
GUI event system (mouse clicks, selections, polygon tools, etc.) with the
higher-level application logic.

It provides:

- ``ToolDispatcher``  
  A small controller that routes UI events from a SpectralImageCanvas
  to the currently-active tool. Tools may register temporary or permanent
  handlers for:
      * single-click events
      * right-click events
      * rectangle selections
      * polygon selections

- ``tools``  
  A collection of stateless utility functions implementing core actions
  such as cropping, masking, feature extraction, unwrapping, k-means,
  and winner-takes-all mineral mapping. These operate directly on
  ``ProcessedObject`` and ``RawObject`` instances.

Design Notes
------------
The interface layer intentionally contains *no* data model logic and *no*
UI components. Its job is simply to:

1. Expose high-level operations to the rest of the application.
2. Translate GUI gestures into tool actions using ``ToolDispatcher``.
3. Provide a clean separation between:
      - UI widgets  
      - tool behaviours  
      - data objects  

This keeps the viewer modular and makes tools easy to swap, disable,
or stack without modifying the underlying canvas.

Typical Usage
-------------
The main window constructs a ``SpectralImageCanvas`` and wraps it in a
``ToolDispatcher``. Individual pages or toolbars then register the
event handlers they need, for example::

    disp = ToolDispatcher(canvas)
    disp.set_single_click(handle_pick)
    disp.set_polygon(handle_polygon, temporary=True)

Everything else in the application interacts with the data layer via
functions in :mod:`tools`.

"""

from .tool_dispatcher import ToolDispatcher
