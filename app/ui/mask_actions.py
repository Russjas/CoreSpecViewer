"""
Callback handler for Masking Actions.

"""
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QInputDialog
from .display_canvases import ImageCanvas2D
from ..interface import tools as t
from .base_actions import BaseActions
from . import busy_cursor

class MaskActions(BaseActions):
    """Raw data operations"""
    
    def stage_ribbon(self):
        """Define and register ribbon buttons"""
        
        self._register_group('Masking', [
            ("button", "Auto-mask by cluster", self.act_mask_by_cluster, "Clusters image into 2 classes, then masks the selected class. New, and experimental.", "Ctrl+Alt"),
            ("button", "New mask", lambda: self.act_mask_point('new'), "Creates a blank mask,\n then masks by correlation with selected pixel.", "Ctrl+W"),
            ("button", "Enhance mask", lambda: self.act_mask_point('enhance'), "Adds to existing mask by correlation with selected pixel", "Ctrl+E"),
            ("button", "Mask region", self.act_mask_rect, "Adds a masked rectangle to existing mask", "Ctrl+R"),
            ("menu", "Freehand mask region", [
                ("Mask inside selected", lambda: self.act_mask_polygon(mode="mask inside"), "With existing mask, masks all pixels inside of selected region", "Ctrl+F"),
                ("Mask outside selected", lambda: self.act_mask_polygon(mode="mask outside"), "With existing mask, masks all pixels outside of selected region", "Ctrl+Shift+F"),
                ("Unmask inside selected", lambda: self.act_mask_polygon(mode="unmask inside"), "With existing mask, unmasks all pixels inside of selected region", "Ctrl+A"),
                ("Unmask outside selected", lambda: self.act_mask_polygon(mode="unmask outside"), "With existing mask, unmasks all pixels outside of selected region", "Ctrl+Shift+A"), 
            ]),
            ("button", "Despeckle", self.despeck_mask, "Remove speckles from mask"),
            ("menu", "Improve", [
                ("Vertical", lambda: self.act_mask_improve(mode="vertical"), "Heuristically improves the mask, only on true-vertical boxes"),
                ("Connect Lines", lambda: self.act_mask_improve(mode="hough"), "Heuristically connect the lines using hough line connection", "Alt+S")
            ]),
            ("button", "Add depth anchor", self.act_depth_anchor, "Add a known depth point to constrain depth registration"),
            ("button", "Clear all depth anchors", self.act_clear_depth_anchors, "Clear all depth anchors from metadata"),
            ("button", "Calc stats", self.act_mask_calc_stats, "Calculates connected components used for downhole unwrapping", "Ctrl+D"),
            ("button", "Mask line", lambda: self.act_mask_point('line'), "Adds a masked vertical line to existing mask"),
            ("button", "unwrap preview", self.unwrap, 'Produces "unwrapped" coreboxes by vertical concatenation: Right→Left, Top→Bottom', "Ctrl+G"),
            ("button", "Mask all", self.act_mask_all, "Masks all pixels (inverse workflow: unmask what you need)"),
            ("button", "Unmask region", lambda: self.act_mask_rect(unmask=True), "Unmasks a rectangle in existing mask", "Ctrl+Shift+R"),
            ("button", "Invert mask", self.act_invert_mask, "Inverts mask: masked ↔ unmasked"),
            ("button", "re-generate thumbs (slow)", self.re_thumb, 'Regenerates all thumbnail images. Slow process, but shouldnt be needed often'),
            ("button", "Rim", self.rim_mask, "Apply a rim to the mask", "Ctrl+Alt+R")
        ])
    
    # -------- MASK actions --------

    def act_mask_rect(self, unmask = False):
        label = "Unmask Region" if unmask else "Mask Region"
        logger.info(f"Button clicked: Mask Region")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        p = self.controller.active_page()
        if not p or not p.dispatcher:
            return

        def _on_rect(y0, y1, x0, x1):
            try:
                self.cxt.current = t.mask_rect(self.cxt.current, y0, y1, x0, x1, unmask = unmask)
                verb = "unmasked" if unmask else "masked"
                logger.info(f"{self.cxt.current.basename} {verb} at Y {y0}:{y1}, X {x0}:{x1}")
                self.controller.refresh()
            finally:
                p.dispatcher.clear_all_temp()
        p.dispatcher.set_rect(_on_rect)
        p.dispatcher.start_rect_select()

    def act_mask_point(self, mode):
        logger.info(f"Button clicked: Mask point {mode}")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        p = self.controller.active_page()
        if not p or not p.dispatcher:
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


    def act_mask_improve(self, mode = "vertical"):
        logger.info(f"Button clicked: Improve Mask with mode={mode}")  # ADD THIS
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        logger.info(f"Calling improve_mask with mode={mode}")  # ADD THIS
        self.cxt.current = t.improve_mask(self.cxt.current, mode = mode)
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

    
    def rim_mask(self):
        logger.info(f"Button clicked: Rim Mask")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        self.cxt.current = t.rim(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} Mask rimmed")
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
        p.dispatcher.start_polygon_select()


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


    def act_mask_all(self):
        logger.info("Button clicked: Mask All")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        self.cxt.current = t.mask_all(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} All pixels masked")
        self.controller.refresh()


    def act_invert_mask(self):
        logger.info("Button clicked: Invert Mask")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Masking", msg)
            return
        self.cxt.current = t.invert_mask(self.cxt.current)
        logger.info(f"{self.cxt.current.basename} Mask inverted")
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

    
    def re_thumb(self):
        logger.info(f"Button clicked: Regenerate Thumbs")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("thumbnails", msg)
            return
        with busy_cursor('unwrapping...', self.controller):
            self.cxt.po.build_all_thumbs(force=True)

    # Experimental auto masking
    def act_mask_by_cluster(self):
        logger.info("Button clicked: Auto-mask by cluster")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Auto-mask by cluster", msg)
            return

        with busy_cursor('Clustering (k=2)...', self.controller):
            po, cluster_array = t.mask_clusters(self.cxt.current)

        logger.info(f"{po.basename} k=2 clustering complete for auto-mask")

        dlg = ClusterMaskDialog(cluster_array, po.mask, parent=self.controller)
        chosen_idx = dlg.run()

        if chosen_idx is None:
            logger.info("Auto-mask by cluster cancelled by user")
            return

        logger.info(f"{po.basename} masking cluster index {chosen_idx}")
        self.cxt.current = t.mask_by_cluster(po, cluster_array, chosen_idx)
        logger.info(f"{po.basename} auto-mask by cluster complete, class {chosen_idx} masked")
        self.controller.refresh()

    def act_depth_anchor(self):
        logger.info("Button clicked: Add depth anchor")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Depth Anchor", msg)
            return

        p = self.controller.active_page()
        if not p or not p.dispatcher:
            return

        def handle_click(y, x):
            try:
                depth, ok = QInputDialog.getDouble(
                    self.controller, "Depth Anchor",
                    f"Enter depth (m) for point ({int(x)}, {int(y)}):",
                    decimals=2
                )
                if not ok:
                    return
                self.cxt.current = t.add_depth_anchor(self.cxt.current, x, y, depth)
                
                import uuid
                ann = dict(self.cxt.current['annotations'].data) if self.cxt.current.has('annotations') else {}
                ann[f"ann_{uuid.uuid4().hex[:8]}"] = {
                    "shape": "point",
                    "x": int(x),
                    "y": int(y),
                    "label": f"Depth Anchor: {depth:.3f}m"
                }
                self.cxt.current.add_temp_dataset('annotations', ann, ext='.json')

                self.controller.refresh()
            except Exception as e:
                logger.error("Failed to add depth anchor", exc_info=True)
                self._show_error("Depth Anchor", f"Failed to add depth anchor: {e}")
            finally:
                p.dispatcher.clear_all_temp()

        p.dispatcher.set_single_click(handle_click, temporary=True)


    def act_clear_depth_anchors(self):
        logger.info("Button clicked: Clear depth anchors")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Clear Depth Anchors", msg)
            return

        if not self.cxt.current.metadata.get('anchors'):
            return

        try:
            #Clear from metadata
            metadata = dict(self.cxt.current.metadata)
            metadata.pop('anchors', None)
            self.cxt.current.add_temp_dataset('metadata', metadata, ext='.json')
            logger.info(f"Depth anchors cleared for {self.cxt.current.basename}")
            
            # Clear from annotations
            if self.cxt.current.has('annotations'):
                ann = dict(self.cxt.current['annotations'].data)
                ann = {k: v for k, v in ann.items() 
                            if "depth anchor" not in v.get("label", "").lower()}
                self.cxt.current.add_temp_dataset('annotations', ann, ext='.json')

            self.controller.refresh()
        except Exception as e:
            logger.error("Failed to clear depth anchors", exc_info=True)
            self._show_error("Clear Depth Anchors", f"Failed to clear depth anchors: {e}")


# ======= Dialogue for experimental auto masking
class ClusterMaskDialog(QDialog):
    """
    Modal dialog displaying a k=2 cluster map and asking the user
    which class to mask. Returns the chosen class index (0 or 1) via run(),
    or None if cancelled.
    """

    def __init__(self, cluster_array, existing_mask, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-mask by cluster")
        self.setModal(True)
        self._chosen_idx = None

        legend = [
            {"index": 0, "label": "Class 1"},
            {"index": 1, "label": "Class 2"},
        ]

        layout = QVBoxLayout(self)

        label = QLabel("Which class should be masked?")
        label.setStyleSheet("font-size: 13px; padding: 6px;")
        layout.addWidget(label)

        self._canvas = ImageCanvas2D(self)
        self._canvas.setMinimumSize(600, 400)
        self._canvas._show_index_with_legend(cluster_array, existing_mask, legend)
        layout.addWidget(self._canvas)

        btn_row = QHBoxLayout()

        btn_class1 = QPushButton("Mask Class 1")
        btn_class1.clicked.connect(lambda: self._select(0))
        btn_row.addWidget(btn_class1)

        btn_class2 = QPushButton("Mask Class 2")
        btn_class2.clicked.connect(lambda: self._select(1))
        btn_row.addWidget(btn_class2)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

    def _select(self, idx):
        self._chosen_idx = idx
        self.accept()

    def run(self):
        """
        Execute the dialog and return the chosen cluster index (0 or 1),
        or None if the user cancelled.
        """
        self.exec_()
        return self._chosen_idx