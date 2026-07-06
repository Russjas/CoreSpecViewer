"""
DholeView — a detached, single-box downhole workbench.

Reuses the multi-box hole machinery (HoleControlPanel + the downhole
display canvases) against an *ephemeral* one-box HoleObject, so a single
box can be previewed downhole with the same controls (step, resample,
feature/minmap generation, plotting) as a full hole — without touching
the real application context or the box's on-disk data.

Design contract
---------------
- DholeView IS a BasePage (for closable-widget wrappers, popouts, the
  splitter, add_dhole_display) but is NOT a registered page. It must
  never be passed to MainWindow.add_page() or included in the
  distribute-context broadcast, or its ephemeral cxt would be
  overwritten with the real one. It is a construct-and-show top-level
  window that owns its own lifecycle.
- Its cxt is a private CurrentContext whose .ho is a scratch-rooted,
  '_temp'-id one-box hole. current/active/po are deliberately left
  unset: nothing in the downhole path reads them, and leaving them None
  guarantees no reach-back into the real box through .current.
"""
from __future__ import annotations
import tempfile
import logging
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QLabel

from ..models import CurrentContext, ProcessedObject, HoleObject
from ..interface import tools
from ..interface.profile_tools import build_ephemeral_hole
from .base_page import BasePage
from .hole_page import HoleControlPanel

logger = logging.getLogger(__name__)




class DholeView(BasePage):
    """
    Detached single-box downhole viewer.

    Do not register this with the page controller. Instantiate via
    DholeView.from_box(po) and show() it; it manages its own teardown
    and cleans up the ephemeral hole's scratch directory on close.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scratch dir backing the ephemeral hole; cleaned up on close.
        self._scratch_dir: Path | None = None

        # Embed the (unmodified) hole control panel. It reaches data only
        # through self._page.cxt.ho, so pointing our cxt at the ephemeral
        # hole is all it needs.
        self._control_panel = HoleControlPanel(self)

        # The panel is tall; give it a scroll area so the window stays sane.
        scroller = QScrollArea(self)
        scroller.setWidgetResizable(True)
        scroller.setWidget(self._control_panel)
        self._add_left(scroller)

        # Downhole canvases are added on demand to the right of the panel
        # via add_dhole_display (called back by the panel's show_downhole).
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        # Trim the panel down to the downhole-relevant controls.
        self._suppress_multibox_controls()

        self.setWindowTitle("Downhole preview")
        self.resize(1100, 800)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        logger.info(f"Parent = {parent}")
    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_box(cls, po: "ProcessedObject", parent=None) -> "DholeView":
        """
        Build a detached downhole viewer for a single, saved box.

        The box must have unwrap stats/segments and no unsaved temp
        datasets (create_base_datasets calls reload_all, which drops
        temps). tools.build_ephemeral_hole enforces isolation: it roots
        the throwaway hole in a scratch dir and tags its id '_temp' so no
        hole-named files land in the real box directory.
        """
        view = cls(parent=parent)

        ho = build_ephemeral_hole(po)          # scratch-rooted, '_temp' id
        view._scratch_dir = ho.root_dir

        temp_cxt = CurrentContext()
        temp_cxt.ho = ho
        # current/active/po intentionally left unset — see module docstring.
        view.cxt = temp_cxt                          # inert setter; nothing re-pushes over it

        # Not driven by the page switcher, so activate ourselves once.
        view.activate()
        view._control_panel.update_for_hole()
        view.add_box_number()
        wid = po.metadata.get("borehole id", "?")
        box = po.metadata.get("box number", "?")
        view.setWindowTitle(f"Downhole preview — {wid} box {box}")
        return view

    # ------------------------------------------------------------------
    # Page-level callbacks the control panel expects on self._page
    # ------------------------------------------------------------------
    def add_box_number(self):
        """
        Assumes ephemeral single box cxt
        """
        cp = self._control_panel
        bx_nm_lbl = QLabel("—")
        cp.info_layout.addRow("Box number:", bx_nm_lbl)
        if self.cxt.ho is not None:
            bx_nm_lbl.setText(str(self.cxt.ho.first_box))
    def add_dhole_display(self, key, canvas):
        """
        Host a downhole plot canvas as a closable, pop-out-able widget.
        Mirrors HolePage.add_dhole_display so the panel's show_downhole
        works unchanged.
        """
        from .display_text import gen_display_text
        disp = gen_display_text(key)
        wrapper = self._add_closable_widget(
            canvas,
            title=f"Downhole: {disp}",
            popoutable=False,
        )
        wrapper.popout_requested.connect(self._handle_popout_request)

    def add_column(self, dataset_key: str = "mask"):
        """
        No-op. The panel's 'Add extra columns' control targets the
        multi-box strip tables, which a single-box view has no use for.
        The control is hidden in _suppress_multibox_controls; this
        override exists only so a stray signal can't AttributeError.
        """
        logger.debug("add_column ignored in DholeView (single-box)")

    # ------------------------------------------------------------------
    # Surface trimming
    # ------------------------------------------------------------------
    def _suppress_multibox_controls(self):
        """
        Hide the panel controls that don't belong in a single-box preview:
          - 'Add extra columns' (multi-box strip tables)
          - Export/Archive menu (writes into the scratch hole; nonsensical
            here, and Archive on a '_temp' hole is a footgun)

        Only stored widgets can be hidden cleanly. If you want this fully
        tidy, give HoleControlPanel a `mode`/capability flag so these
        blocks are never built; this method is the pragmatic version.
        """
        cp = self._control_panel
        for attr in ("combo_block", "separator2", "export_button"):
            w = getattr(cp, attr, None)
            if w is not None:
                w.hide()
        

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        try:
            self.teardown()                          # closes popouts, clears dispatcher
        finally:
            self._cleanup_scratch()
        super().closeEvent(event)

    def _cleanup_scratch(self):
        d = self._scratch_dir
        self._scratch_dir = None
        if d and Path(d).exists():
            try:
                shutil.rmtree(d, ignore_errors=True)
                logger.debug(f"Removed downhole preview scratch dir {d}")
            except Exception:
                logger.warning(f"Failed to remove scratch dir {d}", exc_info=True)
