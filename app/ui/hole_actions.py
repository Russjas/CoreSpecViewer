"""
Callback handler for Masking Actions.

"""
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QInputDialog, QFileDialog, QDialog

from ..interface import tools as t
from .base_actions import BaseActions
from . import busy_cursor, LibMetadataDialog, WavelengthRangeDialog, LibMetadataDialog
from .band_math_dialogue import BandMathsDialog
from ..config import feature_keys as FEATURE_KEYS


class HoleActions(BaseActions):
    """Raw data operations"""
    def __init__(self, context, ribbon, parent=None, box_ops=None):
        self.box_ops = box_ops
        super().__init__(context, ribbon, parent)
        
    def stage_ribbon(self):
        self.extract_feature_list_multi = []
        for key in FEATURE_KEYS:
            self.extract_feature_list_multi.append((key, lambda _, k=key: self.box_ops.run_feature_extraction(k, multi=True)))
        self.extract_feature_list_multi.append(("Custom Feature...", lambda: self.box_ops.custom_feature_act(multi=True))) 
        self.extract_feature_list_multi.append(("Cache Features...", lambda: self.box_ops.cache_features(multi=True))) 
        self._register_group('Hole operations', [
            ("button", "Previous", self.hole_prev_box, "View previous box in hole", "Left"),
            ("button", "Next", self.hole_next_box, "View next box in hole", "Right"),
            ("button", "Return to Raw", self.hole_return_to_raw, "Open the raw dataset to replace this box"),
            ("menu", "Iterative Operations", [
                ("Quick Cluster", lambda: self.box_ops.act_kmeans(multi=True)),
                ("menu", "Fullhole Correlations", [
                    ("MineralMap Pearson (Winner-takes-all)", lambda: self.box_ops.act_vis_correlation(multi=True)),
                    ("MineralMap SAM (Winner-takes-all)", lambda: self.box_ops.act_vis_sam(multi=True)),
                    ("MineralMap MSAM (Winner-takes-all)", lambda: self.box_ops.act_vis_msam(multi=True)),
                    ("Multi-range check (Winner-takes-all)", lambda: self.box_ops.act_vis_multirange(multi=True), "Performs custom multi-window matching"),
                    ("select range", lambda: self.box_ops.act_subrange_corr(multi=True), "Performs correlation on a chosed wavelength range"),
                ]),
                ("menu", "Fullhole Features", self.extract_feature_list_multi),
                ("Band Maths", lambda: self.box_ops.act_band_maths(multi=True)),
                ("Export to ENVI", lambda: self.box_ops.export_envi(multi=True)),
            ], "Performs operations iteratively on each box in hole. Commits and saves any unsaved box edits without confirmation."),
            ("button", "Save All", self.save_all_changes, "This will commit and save all temp datasets without asking for confirmation"),
            ("button", "Generate Images", lambda: self.box_ops.gen_images(multi=True))
        ])
            # --- HOLE actions ---
    def hole_next_box(self):
        logger.info(f"Button clicked: Next Box")     
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Hole Navigation", msg)
            return
        try:
            box_num = int(self.cxt.current.metadata.get("box number"))
        except Exception:
            box_num = None
        if box_num is not None:
            next_box = None
            found = False
            for bn, po in self.cxt.ho.iter_items():
                if found:
                    next_box = po
                    break
                if bn == box_num:
                    found = True

            if next_box is None:
                return   # at the last box
           
            self.cxt.current = next_box
            active_page = self.controller.active_page()
           
            if hasattr(active_page, "teardown"):
                active_page.teardown()
            if hasattr(active_page, "activate"):
                active_page.activate()
            self.controller.refresh()  
           

    def hole_prev_box(self):
        logger.info(f"Button clicked: Previous Box")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Hole Navigation", msg)
            return
        try:
            box_num = int(self.cxt.current.metadata.get("box number"))
        except Exception:
            box_num = None
        if box_num is not None:
            prev = None
            for bn, po in self.cxt.ho.iter_items():
                if bn == box_num:
                    break
                prev = po

            if prev is None:
                return   # at the first box
            self.cxt.current = prev
            active_page = self.controller.active_page()
            if hasattr(active_page, "teardown"):
                active_page.teardown()
            if hasattr(active_page, "activate"):
                active_page.activate()
            self.controller.refresh()

    def hole_return_to_raw(self):
        logger.info(f"Button clicked: Return to Raw")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Hole Navigation", msg)
            return
        path = QFileDialog.getExistingDirectory(
                   self.controller,
                   "Select directory",
                   "",
                   QFileDialog.ShowDirsOnly
                   )
        if not path:
            return
        box_num = None
        try:
            box_num = int(self.cxt.current.metadata.get("box number"))
        except Exception:
            box_num = None

        try:
            with busy_cursor('loading...', self.controller):
                loaded_obj = t.load(path)
                if not loaded_obj.is_raw:
                    self._show_error("Return to Raw",
                        "Selected path is not a raw Lumo directory.")
                    return
                new_po = loaded_obj.process()
                new_po.update_root_dir(self.cxt.current.root_dir)
                new_po.build_all_thumbs()
                new_po.save_all_thumbs()
                if new_po.metadata.get("borehole id") != self.cxt.ho.hole_id:
                    self._show_error("Return to Raw",
                        "The new loaded scan is from a different hole")
                    return
                if box_num not in self.cxt.ho.boxes:
                    self._show_error("Return to Raw",
                             f"Box {box_num} not found in current hole.")
                    return
                self.cxt.ho.boxes[box_num] = new_po
                self.cxt.ho.hole_meta[box_num] = new_po.metadata

                self.cxt.current = new_po

                self.controller.refresh(view_key = "vis")

        except Exception as e:
            self._show_error("Open dataset", f"Failed to open dataset: {e}")
            return

    def save_all_changes(self):
        logger.info(f"Button clicked: Save All")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Save hole", msg)
            return
        with busy_cursor('Saving.....', self.controller) as progress:
            self.cxt.ho.save_product_datasets()
            
            for po in self.cxt.ho:
                po.save_all_thumbs()
                if po.has_temps:
                    progress.set(f"Saving {po.metadata['box number']} box number")
                    logger.info(f"{po.metadata['box number']} box number")
                    po.commit_temps()
                    po.save_all()
                    logger.info('saved all, reloading')
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Data saved for {po.basename}")
            self.controller.refresh(view_key = "hol")