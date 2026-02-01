"""
Callback handler for Masking Actions.

"""
import logging
logger = logging.getLogger(__name__)


from ..interface import tools as t
from .base_actions import BaseActions
from . import busy_cursor



class MaskActions(BaseActions):
    """Raw data operations"""
    
    def stage_ribbon(self):
        """Define and register ribbon buttons"""
        
        self._register_group('Masking', [
            ("button", "New mask", lambda: self.act_mask_point('new'), "Creates a blank mask,\n then masks by correlation with selected pixel."),
            ("button", "Enhance mask", lambda: self.act_mask_point('enhance'), "Adds to existing mask by correlation with selected pixel"),
            ("button", "Mask line", lambda: self.act_mask_point('line'), "Adds a masked vertical line to existing mask"),
            ("button", "Mask region", self.act_mask_rect, "Adds a masked rectangle to existing mask"),
            ("menu", "Freehand mask region", [
                ("Mask outside selected", lambda: self.act_mask_polygon(mode="mask outside"), "With exising mask, masks all pixels outside of selected region"),
                ("Mask inside selected", lambda: self.act_mask_polygon(mode="mask inside"), "With exising mask, masks all pixels inside of selected region")
            ]),
            ("button", "Despeckle", self.despeck_mask, "Remove speckles from mask"),
            ("button", "Improve", self.act_mask_improve, "Heuristically improves the mask"),
            ("button", "Calc stats", self.act_mask_calc_stats, "Calculates connected components used for downhole unwrapping"),
            ("button", "unwrap preview", self.unwrap, 'Produces "unwrapped" coreboxes by vertical concatenation: Right→Left, Top→Bottom')
        ])
    
    # -------- MASK actions --------

    def act_mask_rect(self):
        logger.info(f"Button clicked: Mask Region")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        p = self.controller.active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        def _on_rect(y0, y1, x0, x1):
            try:
                self.cxt.current = t.mask_rect(self.cxt.current, y0, y1, x0, x1 )
                logger.info(f"{self.cxt.current.basename} masked at Y {y0}:{y1}, X {x0}:{x1}")
                self.controller.refresh()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()

    def act_mask_point(self, mode):
        logger.info(f"Button clicked: Mask point {mode}")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        p = self.controller.active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            return

        def handle_point_click(y, x):
            try:
                with busy_cursor('trying mask correlation...', self.controller):
                    self.cxt.current = t.mask_point(self.cxt.current, mode, y, x)
                    logger.info(f"{self.cxt.current.basename} masked by correlation using mode {mode} and pixel ({y},{x})")
                self.controller.refresh()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_single_click(handle_point_click)


    def act_mask_improve(self):
        logger.info(f"Button clicked: Improve Mask")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        self.cxt.current = t.improve_mask(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} Mask improved heuristically")
        self.controller.refresh()

    def despeck_mask(self):
        logger.info(f"Button clicked: Despeckle Mask")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        self.cxt.current = t.despeckle_mask(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} Mask despeckled")
        self.controller.refresh()

    def act_mask_polygon(self, mode = "mask outside"):
        logger.info(f"Button clicked: Freehand Mask mode {mode}")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        p = self.controller.active_page()
        if not p or not p.dispatcher or self.cxt.current is None:
            return
        def _on_finish(vertices_rc):
            self.cxt.current = t.mask_polygon(self.cxt.current, vertices_rc, mode = mode)
            logger.info(f"{self.cxt.current.basename} Mask enhanced with freehand polygon {vertices_rc}")
            self.controller.refresh()
            p.dispatcher.clear_all_temp()
        p.dispatcher.set_polygon(_on_finish, temporary=True)
        p.left_canvas.start_polygon_select()


    def act_mask_calc_stats(self):
        logger.info(f"Button clicked: Calc stats")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        self.cxt.current = t.calc_unwrap_stats(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} connected components calculated for unwrapping stats")
        self.controller.refresh()


    def unwrap(self):
        logger.info(f"Button clicked: Unwrap")
        valid_state, msg = self.cxt.requires(self.cxt.UNWRAP)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        with busy_cursor('unwrapping...', self.controller):
            self.cxt.current = t.unwrapped_output(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} unwrapped using connected components stats")
        self.controller.refresh()
