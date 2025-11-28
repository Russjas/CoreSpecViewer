"""
UI module for CoreSpecViewer.

This package contains all Qt-based user-interface components used by the
application. It provides the full collection of page-level views that make up
the main workflow:

- RawPage:
    Tools for inspecting raw hyperspectral cubes, browsing reflectance data,
    and viewing pixel-level spectra.

- VisualisePage:
    The primary processed-data viewer. Supports false-colour display, product
    previews, cached product management, and interactive spectral inspection.

- LibraryPage:
    Spectral-library manager and correlation workspace. Enables SQLite-based
    spectral browsing, collection building, exemplar selection, and correlation
    workflows.

- HolePage:
    Full hole-level visualisation using thumbnail strips, metadata panels,
    and selectable ProcessedObjects for multi-box navigation.

- CatalogueWindow:
    A standalone file/directory browser for opening datasets outside the main
    view workflow.

All UI pages inherit from `BasePage` and integrate with the central controller
(via the ToolDispatcher) to provide a consistent set of actions across the
application. This module exposes the concrete pages so that the main application
can instantiate and register them without importing their internal details.
"""

from .catalogue_window import CatalogueWindow
from .hole_page import HolePage
from .lib_page import LibraryPage
from .raw_page import RawPage
from .ribbon import Groups, Ribbon
from .util_windows import (
    AutoSettingsDialog,
    ImageCanvas2D,
    InfoTable,
    MetadataDialog,
    SpectralImageCanvas,
    SpectrumWindow,
    busy_cursor,
    choice_box,
    two_choice_box,
    WavelengthRangeDialog
)
from .vis_page import VisualisePage

__all__ = [
    "RawPage",
    "VisualisePage",
    "LibraryPage",
    "HolePage",
    "CatalogueWindow",
    "Ribbon",
    "Groups",
    "SpectralImageCanvas",
    "ImageCanvas2D",
    "InfoTable",
    "SpectrumWindow",
    "AutoSettingsDialog",
    "MetadataDialog",
    "two_choice_box",
    "choice_box",
    "InfoTable",
    "busy_cursor",
    "WavelengthRangeDialog"
]
