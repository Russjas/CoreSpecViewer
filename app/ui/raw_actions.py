"""
Callback handler for Raw Actions.

"""
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog

from ..interface import tools as t
from .base_actions import BaseActions
from . import busy_cursor , MetadataDialog


class RawActions(BaseActions):
    """Raw data operations"""
    
    def stage_ribbon(self):
        """Define and register ribbon buttons"""
        self._register_group('Raw', [
            ("button", "Auto Crop", self.automatic_crop, 
             "Faster on Raw than Processed data.\nUses image analysis to automatically detect core box - NB. is very flaky"),
            ("button", "Crop", self.crop_current_image, 
             "Faster on Raw than Processed data.\nManually crop the image"),
            ("button", "Process", self.process_raw, 
             "Produce a processed dataset from this raw dataset"),
        ])

    def crop_current_image(self):
        logger.info(f"Button clicked: Crop")
        valid_state, msg = self.cxt.requires(self.cxt.SCAN)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Crop", msg)
            return
        p = self.controller.active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return
    
        # Ask the page to collect a rectangle and pass back coords
        def _on_rect(y0, y1, x0, x1):
            try:
                with busy_cursor('cropping...', self.controller):
                    self.cxt.current = t.crop(self.cxt.current, y0, y1, x0, x1)
                    logger.info(f"{self.cxt.current.basename} cropped to Y {y0}:{y1}, X {x0}:{x1}")
                    self.controller.refresh()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()
    
    def automatic_crop(self):
        logger.info(f"Button clicked: Auto-Crop")
        valid_state, msg = self.cxt.requires(self.cxt.SCAN)
        if not valid_state:
            logger.warning(msg)
            self._show_error( "Cropping", msg)
            return
        with busy_cursor('cropping...', self.controller):
            self.cxt.current = t.crop_auto(self.cxt.current)
            logger.info(f"{self.cxt.current.basename} cropped using autocrop")
        self.controller.refresh()
    
    
    def process_raw(self):
        logger.info(f"Button clicked: Process")
        valid_state, msg = self.cxt.requires(self.cxt.RAW)
        if not valid_state:
            logger.warning(msg)
            self._show_error( "Process", msg)
            return
        meta_check, _ = self.cxt.requires(self.cxt.MANDATORY_META) 
        if not meta_check:
            dlg = MetadataDialog(self.cxt.current.metadata, parent=self.controller)
            if dlg.exec() == QDialog.Accepted:
                result = dlg.get_result()
                self.cxt.current.metadata['borehole id'] = result['hole']
                self.cxt.current.metadata['box number'] = result['box']
                self.cxt.current.metadata['core depth start'] = result['depth_from']
                self.cxt.current.metadata['core depth stop'] = result['depth_to']
                logger.info("mandatory metadata added manually")
            else:
                logger.info("Processing cancelled - metadata dialog closed")
                return
        try:
            with busy_cursor('processing...', self.controller):
    
                self.cxt.po = self.cxt.current.process()
    
    
        except Exception as e:
            logger.error(f"Failed to process {self.cxt.current.basename}", exc_info=True)
            self._show_error("Process", f"Failed to process/save: {e}")
            return
    
        self.controller.refresh(view_key="vis")
        
