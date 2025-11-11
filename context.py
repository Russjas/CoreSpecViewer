# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:28:04 2025

@author: russj
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

# forward references avoid circular imports
if False:
    from objects import RawObject, ProcessedObject, HoleObject
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
    review_log : Path | None
        Path to a batch-processing log or database.
    project_root : Path | None
        Optional project-level root directory.
    active_tool : str | None
        Name of the currently active tool (for UI/tool dispatcher).
    """

    po: Optional["ProcessedObject"] = None
    ro: Optional["RawObject"] = None
    ho: Optional["HoleObject"] = None
    review_log: Optional[Path] = None
    project_root: Optional[Path] = None
    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------
    @property
    def obj(self) -> Optional[Any]:
        """Return whichever object is currently active (PO > RO > HO)."""
        return self.po or self.ro or self.ho

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
    def summary(self) -> str:
        """Compact human-readable summary for debug/logging."""
        parts = []
        if self.po:
            parts.append(f"PO: {getattr(self.po, 'name', type(self.po).__name__)}")
        if self.ro:
            parts.append(f"RO: {getattr(self.ro, 'name', type(self.ro).__name__)}")
        if self.ho:
            parts.append(f"HO: {getattr(self.ho, 'hole_id', type(self.ho).__name__)}")
        if self.review_log:
            parts.append(f"log: {self.review_log.name}")
        return " | ".join(parts) if parts else "(empty context)"

    # ------------------------------------------------------------------
    # simple state management
    # ------------------------------------------------------------------
    #TODO: This is generic, develop as I find a need for it
    def clear(self, level: str = "all") -> None:
        """
        Clear parts of the context.

        level = "po" clears only processed object
        level = "ro" clears raw + processed
        level = "ho" clears everything
        level = "all" clears everything including review_log
        """
        if level in ("po", "ro", "ho", "all"):
            self.po = None
        if level in ("ro", "ho", "all"):
            self.ro = None
        if level in ("ho", "all"):
            self.ho = None
        if level == "all":
            self.review_log = None

    