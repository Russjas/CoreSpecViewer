"""
Tracks the active dataset context for the UI.

Holds the current RawObject, ProcessedObject, and HoleObject,
and signals which object is currently active for visualisation or editing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .hole_object import HoleObject
from .processed_object import ProcessedObject
from .raw_object import RawObject
from .lib_manager import LibraryManager


@dataclass
class CurrentContext:
    """
    Lightweight container for the application's current working state.

    Holds references to whichever data objects are active in this session.
    Each field may be None.

    Attributes
    ----------
    po : ProcessedObject | None
        The currently loaded processed dataset.
    ro : RawObject | None
        The currently loaded raw dataset.
    ho : HoleObject | None
        The currently loaded hole-level dataset.
    project_root : Path | None
        Optional project-level root directory.
    active_tool : str | None
        Name of the currently active tool (for UI/tool dispatcher).
    """
    # Class attributes, common patterns in state validation
    SCAN = ('scan',)
    PROCESSED = ('processed',)
    RAW = ('raw',)
    HOLE = ('hole',)
    LIBRARY = ('library',) 
    COLLECTIONS = ('collections',)
    MASKING_WITH_STATS = ('processed', 'has:stats')
    CORRELATION_SINGLE = ('processed', 'collections')
    CORRELATION_MULTI = ('hole', 'collections')
    ADD_TO_LIBRARY = ('processed', 'library')
    UNWRAP = ('processed', 'has:stats')
    MANDATORY_META = ('meta:borehole id', 'meta:box number', 
                         'meta:core depth start', 'meta:core depth stop')

    #Instance attributes
    _po: Optional["ProcessedObject"] = None
    _ro: Optional["RawObject"] = None
    _ho: Optional["HoleObject"] = None
    library: LibraryManager = field(default_factory=LibraryManager)
    _review_log: Path | None = None
    _project_root: Path | None = None
    active: str | None = None

    #----- properties for enforcing active set on assignment
    @property
    def po(self): return self._po

    @po.setter
    def po(self, obj):
        self._po = obj
        self.active = "po" if obj is not None else None
        

    @property
    def ro(self): return self._ro

    @ro.setter
    def ro(self, obj):
        self._ro = obj
        self.active = "ro" if obj is not None else None

    @property
    def ho(self): return self._ho

    @ho.setter
    def ho(self, obj):
        self._ho = obj

    #current object can only ever be PO or RO - access point for tools that can use either
    @property
    def current(self) -> Any | None:
        if self.active == "po": return self._po
        if self.active == "ro": return self._ro
        return None

    #current object can only ever be PO or RO - access point for tools that can use either
    @current.setter
    def current(self, obj):
        if obj is None:
            self.active = None
            return

        is_raw = getattr(obj, "is_raw", None)
        if is_raw is True:
            self.ro = obj
            return
        if is_raw is False:
            self.po = obj
            return

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------

    @property
    def has_processed(self) -> bool:
        return self.po is not None

    @property
    def has_raw(self) -> bool:
        return self.ro is not None

    @property
    def has_hole(self) -> bool:
        return self.ho is not None


    @property
    def metadata(self) -> dict | None:
        if self._ho is not None: return self._ho.metadata
        if self._po is not None: return self._po.metadata
        if self._ro is not None: return self._ro.metadata
        return None
    
    def requires(self, *requirements) -> bool:
        """
        Check multiple requirements at once.
        
        Args:
            *requirements: Variable list of requirement strings, iterables, or mix
            parent: Optional QWidget for showing error dialog
            
        Returns:
            True if all requirements pass, False otherwise
            
        Examples:
            # Multiple arguments
            if not self.cxt.requires('processed', 'has:stats', parent=self):
                return
            
            # Single iterable
            reqs = ('processed', 'has:stats')
            if not self.cxt.requires(reqs, parent=self):
                return
            
            # Mix of iterable and strings
            reqs = ['processed', 'scanned']
            if not self.cxt.requires(reqs, 'hole', parent=self):
                return
            
            # Multiple iterables
            base = ['processed']
            extra = ['hole', 'library']
            if not self.cxt.requires(base, extra, parent=self):
                return
        """
        # Flatten all iterables in requirements (one level only)
        flattened = []
        for req in requirements:
            # Check if it's an iterable but NOT a string
            if hasattr(req, '__iter__') and not isinstance(req, str):
                flattened.extend(req)
            else:
                flattened.append(req)
        
        # Validate all flattened requirements
        for req in flattened:
            is_valid, error_msg = self._check_requirement(req)
            if not is_valid:
                return False, error_msg
        return True, ""

    def _check_requirement(self, requirement: str) -> tuple[bool, str]:
        """
        Check a single requirement string. Returns (is_valid, error_message).
        
        Note: self (the context) always exists, so we only validate:
          - self.current (can be None)
          - self.ho (can be None)
          - self.library.is_loaded (can be False)
        
        Supported requirement strings:
          'scan'              - Any scan loaded
          'processed'         - Processed scan loaded
          'raw'               - Raw scan loaded
          'hole'              - Hole loaded for multi-box ops
          'library'           - Library database loaded
          'collections'       - Library has collections
          'has:dataset_key'   - Current scan has specific dataset (e.g., 'has:stats')
          'meta:metadata_key' - Current scan has specific metadata (e.g., 'meta:borehole id')
        """
        
        # Basic scan existence
        if requirement == 'scan':
            if self.current is None:
                return False, "No scan loaded"
            return True, ""
        
        # Processed scan
        if requirement == 'processed':
            if self.current is None:
                return False, "No scan loaded"
            if self.current.is_raw:
                return False, "Open a processed dataset first"
            return True, ""
        
        # Raw scan
        if requirement == 'raw':
            if self.current is None:
                return False, "No scan loaded"
            if not self.current.is_raw:
                return False, "This operation requires a raw dataset"
            return True, ""
        
        # Hole for multi-box operations
        if requirement == 'hole':
            if self.ho is None:
                return False, "No hole loaded for multi-box operation"
            return True, ""
        
        # Library loaded
        if requirement == 'library':
            if self.library is None:
                return False, "No library loaded"
            if hasattr(self.library, 'is_open') and not self.library.is_open():
                return False, "No library loaded"
            return True, ""
        
        # Library has collections
        if requirement == 'collections':
            # First check library is loaded
            lib_valid, lib_msg = self._check_requirement('library')
            if not lib_valid:
                return False, lib_msg
            
            # Then check for collections
            if not self.library.collections:
                return False, "No collections in library. Create a collection first."
            return True, ""
        
        # Dataset presence on current scan (e.g., 'has:stats', 'has:mask')
        if requirement.startswith('has:'):
            dataset_key = requirement.split(':', 1)[1]
            if self.current is None:
                return False, "No scan loaded"
            if not self.current.has(dataset_key):
                return False, f"Required dataset '{dataset_key}' not found. Run prerequisite operation first."
            return True, ""
        
        # Metadata presence on current scan (e.g., 'meta:borehole id')
        if requirement.startswith('meta:'):
            meta_key = requirement.split(':', 1)[1]
            if self.current is None:
                return False, "No scan loaded"
            if not self.current.metadata:
                return False, f"No metadata available"
            if not self.current.metadata.get(meta_key):
                return False, f"Required metadata '{meta_key}' is missing"
            return True, ""
        
        # Unknown requirement
        return False, f"Unknown requirement: '{requirement}'"























