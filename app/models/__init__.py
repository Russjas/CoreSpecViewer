"""
CoreSpecViewer app.models package.

Core data structures for representing and manipulating hyperspectral core-scan datasets.

This module defines the foundational classes used across the GSI Hyperspectral Core Box
ecosystem. These objects encapsulate both raw Specim Lumo directories and processed
reflectance-derived datasets, providing a consistent interface for file discovery,
in-memory manipulation, and metadata management.

Classes
-------
Dataset
    Generic container for a single data file (NumPy array, JSON, JPEG, or NPZ masked array).
ProcessedObject
    Logical grouping of processed datasets derived from a single core box.
RawObject
    Representation of a Specim Lumo export directory with dark, white, and data frames.
HoleObject
    Aggregator class representing a borehole composed of multiple processed boxes.



Notes
-----
These classes form the bridge between file-level datasets (e.g., `.npy`, `.hdr`)
and higher-level geological entities (boxes, holes). They support robust path handling,
automatic loading/saving, and downstream integration with the GUI and processing
workflows.
"""

from .context import CurrentContext
from .dataset import Dataset
from .hole_object import HoleObject
from .processed_object import ProcessedObject
from .raw_object import RawObject

__all__ = [
    "Dataset",
    "ProcessedObject",
    "RawObject",
    "HoleObject",
    "CurrentContext",
]
