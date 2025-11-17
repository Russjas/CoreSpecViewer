# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:28:04 2025

@author: russj
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from .raw_object import RawObject
from .processed_object import ProcessedObject
from .hole_object import HoleObject



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

    _po: Optional["ProcessedObject"] = None
    _ro: Optional["RawObject"] = None
    _ho: Optional["HoleObject"] = None
    _review_log: Optional[Path] = None
    _project_root: Optional[Path] = None
    active: Optional[str] = None
    
    #----- properties for enforcing active set on assignment
    @property
    def po(self): return self._po
    
    @po.setter
    def po(self, obj):
        self._po = obj
        self.active = "po" if obj is not None else None
        print('po changed')

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
    def current(self) -> Optional[Any]:
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
    def metadata(self) -> Optional[dict]:
        if self._ho is not None: return self._ho.metadata
        if self._po is not None: return self._po.metadata
        if self._ro is not None: return self._ro.metadata
        return None

   

    