"""
Operations that work on single box or full hole.
Stateful class that holds context and controller references.
"""
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QInputDialog, QFileDialog, QDialog, QMessageBox

from ..models import CurrentContext
from ..interface import tools as t
from .base_actions import BaseActions
from . import busy_cursor, LibMetadataDialog, WavelengthRangeDialog, LibMetadataDialog
from .band_math_dialogue import BandMathsDialog



class BoxOperations:
    """Handles spectral operations in single or multi-box mode"""
    
    def __init__(self, context: CurrentContext, controller):
        self.cxt = context
        self.controller = controller
        
    def _show_error(self, title: str, message: str):
        """Show a consistent error dialog."""
        QMessageBox.warning(self.controller, title, message)
    
    def _show_info(self, title: str, message: str):
        """Show a consistent info dialog."""
        QMessageBox.information(self.controller, title, message)
        
    def ask_collection_name(self):
        valid_state, msg = self.cxt.requires(self.cxt.COLLECTIONS)
        if not valid_state:
            return None
        names = sorted(self.cxt.library.collections.keys())
        if not names:
            self._show_error("No collections", "Create a collection first via 'Add Selected → Collection'.")
            return None
        if len(names) == 1:
            return names[0]
        name, ok = QInputDialog.getItem(self.controller, "Select Collection", "Collections:", names, 0, False)
        return name if ok else None
    
    
    def run_feature_extraction(self, key, multi = False):
        logger.info(f"Button clicked: Extract feature {key}, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Feature Extraction", msg)
            return
        if multi:
            with busy_cursor(f'feature extraction {key}....', self.controller):
                for po in self.cxt.ho:
                    t.run_feature_extraction(po, key)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Extract feature {key} for {po.basename} done")
                
                self.controller.refresh()
            return
        with busy_cursor(f'extracting {key}...', self.controller):
            self.cxt.current = t.run_feature_extraction(self.cxt.current, key)
        logger.info(f"Extract feature {key} for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")


    def act_vis_correlation(self, multi = False):
        logger.info(f"Button clicked: Mineral Map Pearson Correlation, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Mineral Mapping", msg)
            return
        name = self.ask_collection_name()
        if not name:
            logger.warning("Correlation cancelled, no collection selected")
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            logger.error(f"Correlation cancelled, failed to collect exemplar for collection {name}", exc_info=True)
            return
        if multi:
            logger.info(f"Correlation is using collection {name} in multi mode")
            with busy_cursor('correlation...', self.controller):
                for po in self.cxt.ho:
                    t.wta_min_map(po, exemplars, name)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Pearson with collection {name} for {po.basename} done")
                    
            self.controller.refresh(view_key="hol")
            return
        
        logger.info(f"Correlation is using collection {name} on single box")
        with busy_cursor('correlation...', self.controller):
            self.cxt.current = t.wta_min_map(self.cxt.current, exemplars, name)
        logger.info(f"Pearson with collection {name} for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")


    def act_vis_sam(self, multi = False):
        logger.info(f"Button clicked: Mineral Map SAM Correlation, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Mineral Mapping", msg)
            return
        name = self.ask_collection_name()
        if not name:
            logger.warning("Correlation cancelled, no collection selected")
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            logger.error(f"Correlation cancelled, failed to collect exemplar for collection {name}", exc_info=True)
            return
        if multi:
            logger.info(f"Correlation is using collection {name} in multi mode")
            with busy_cursor('correlation...', self.controller):
                for po in self.cxt.ho:
                    t.wta_min_map_SAM(po, exemplars, name)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"SAM with collection {name} for {po.basename} done")
            self.controller.refresh(view_key="hol")
            return
        logger.info(f"Correlation is using collection {name} on single box")
        with busy_cursor('correlation...', self.controller):
            self.cxt.current = t.wta_min_map_SAM(self.cxt.current, exemplars, name)
        logger.info(f"SAM with collection {name} for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")
        
        
    def act_vis_msam(self, multi = False):
        logger.info(f"Button clicked: Mineral Map MSAM Correlation, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Mineral Mapping", msg)
            return
        name = self.ask_collection_name()
        if not name:
            logger.warning("Correlation cancelled, no collection selected")
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            logger.error(f"Correlation cancelled, failed to collect exemplar for collection {name}", exc_info=True)
            return
        if multi:
            logger.info(f"Correlation is using collection {name} in multi mode")
            with busy_cursor('correlation...', self.controller):
                for po in self.cxt.ho:
                    t.wta_min_map_MSAM(po, exemplars, name)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"MSAM with collection {name} for {po.basename} done")
                    
            self.controller.refresh(view_key="hol")
            return
        logger.info(f"Correlation is using collection {name} on single box")
        with busy_cursor('correlation...', self.controller):
            self.cxt.current = t.wta_min_map_MSAM(self.cxt.current, exemplars, name)
        logger.info(f"MSAM with collection {name} for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")

    def act_vis_multirange(self, multi = False):
        modes = ['pearson', 'sam', 'msam']
        logger.info(f"Button clicked: Multi-range mineral mapping, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Mineral Mapping", msg)
            return
        name = self.ask_collection_name()
        if not name:
            logger.warning("Correlation cancelled, no collection selected")
            return
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            logger.error(f"Correlation cancelled, failed to collect exemplar for collection {name}", exc_info=True)
            return
        mode, ok = QInputDialog.getItem(self.controller, "Select Match Mode", "Options:", modes, 0, False)
        if not ok or not mode:
            logger.warning("Correlation cancelled, no mode selected")
            return
        if multi:
            
            logger.info(f"Correlation is using {mode} and collection {name} in multibox mode")
            with busy_cursor('correlation...', self.controller):
                for po in self.cxt.ho:
                    t.wta_multi_range_minmap(po, exemplars, name, mode=mode)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Multirange with collection {name} using {mode} for {po.basename} done")
            self.controller.refresh(view_key="hol")
            return
        
        logger.info(f"Correlation is using {mode} and  collection {name} for single box")
        with busy_cursor('correlation...', self.controller):
            self.cxt.current = t.wta_multi_range_minmap(self.cxt.current, exemplars, name, mode=mode)
        logger.info(f"Multirange with collection {name} using {mode} for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")
        
        
    def act_subrange_corr(self, multi = False):
        modes = ['pearson', 'sam', 'msam']
        logger.info(f"Button clicked: User-defined range correlation, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Mineral Mapping", msg)
            return
        name = self.ask_collection_name()
        if not name:
            logger.warning("Correlation cancelled, no collection selected")
            return
        logger.info(f"Correlation is using collection {name}")
        exemplars = self.cxt.library.get_collection_exemplars(name)
        if not exemplars:
            logger.error(f"Correlation cancelled, failed to collect exemplar for collection {name}", exc_info=True)
            return
        mode, ok = QInputDialog.getItem(self.controller, "Select Match Mode", "Options:", modes, 0, False)
        if not ok or not mode:
            logger.warning("Correlation cancelled, no mode selected")
            return
        logger.info(f"Correlation is using {mode} and  collection {name}")
        ok, start_nm, stop_nm = WavelengthRangeDialog.get_range(
            parent=self.controller,
            start_default=0,
            stop_default=20000,
        )
        if not ok:
            logger.warning("Correlation cancelled, no range selected")
            return
        if multi:
            logger.info(f"Correlation is using {mode}, collection {name} and range ({start_nm}:{stop_nm}) in multibox mode")
            with busy_cursor('correlation...', self.controller):
                for po in self.cxt.ho:
                    try:
                        t.wta_min_map_user_defined(po, exemplars, name, [start_nm, stop_nm], mode=mode)
                        po.commit_temps()
                        po.save_all()
                        po.reload_all()
                        po.load_thumbs()
                        logger.info(f"Defined range correlation with collection {name} using {mode} and range ({start_nm}:{stop_nm}) for {po.basename} done")
                    except ValueError:
                        logger.warn(f"Defined range correlation with collection {name} using {mode} and range ({start_nm}:{stop_nm}) for {po.basename} done", exc_info=True)
                        continue      
            self.controller.refresh(view_key="hol")
            return
        logger.info(f"Correlation is using {mode}, collection {name} and range ({start_nm}:{stop_nm}) for single box")
        with busy_cursor('correlation...', self.controller):
            try:
                self.cxt.current = t.wta_min_map_user_defined(self.cxt.current, exemplars, name, [start_nm, stop_nm], mode=mode)
            except Exception as e:
                logger.error(f"Defined range correlation with collection {name} using {mode} and range ({start_nm}:{stop_nm}) for {self.cxt.current.basename} done", exc_info=True)
                self._show_error("Failed operation", f"Failed to use band range: {e}")
                return
        logger.info(f"Defined range correlation with collection {name} using {mode} and range ({start_nm}:{stop_nm}) for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")

    
    def act_kmeans(self, multi = False):
        logger.info(f"Button clicked: Quick Cluster, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Clustering", msg)
            return
        clusters, ok1 = QInputDialog.getInt(self.controller, "KMeans Clustering",
                "Enter number of clusters:",value=5, min=1, max=50)
        if not ok1:
            logger.warning("Clustering cancelled, n value not selected")
            return
        iters, ok2 = QInputDialog.getInt(self.controller, "KMeans Clustering",
            "Enter number of iterations:", value=50, min=1, max=1000)
        if not ok2:
            logger.warning("Clustering cancelled, interations number value not selected")
            return
        if multi:
            logger.info(f"Clustering started using clusters {clusters} and iters {iters} mutlti box")
            with busy_cursor('clustering...', self.controller):
                for po in self.cxt.ho:
                    t.kmeans_caller(po, clusters, iters)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Clustering ({clusters}, {iters}) done for {po.basename}")
            self.controller.refresh(view_key="hol")
            return
        
        logger.info(f"Clustering started using clusters {clusters} and iters {iters}")
        with busy_cursor('clustering...', self.controller):
            self.cxt.current = t.kmeans_caller(self.cxt.current, clusters, iters)
        logger.info(f"Clustering ({clusters}, {iters}) done for {self.cxt.current.basename}")
        self.controller.refresh(view_key="vis")
        
    def act_band_maths(self, multi = False):
        
        """
        Triggered from the ribbon/menu:
        - ask user for a band-maths expression + name
        - pass them, along with the current object, to the interface layer
        """
        logger.info(f"Button clicked: Band Maths, multi-mode: {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Band maths", msg)
            return
        ok, name, expr, cr = BandMathsDialog.get_expression(
           parent=self.controller,
           default_name="Custom band index",
           default_expr="R2300-R1400",
        )
        if not ok:
            logger.info("Band maths operation cancelled from dialogue")
            return
        if multi:
            logger.info(f"Band Maths started using {expr} mutlti box")
            with busy_cursor('clustering...', self.controller):
                for po in self.cxt.ho:
                    t.band_math_interface(po, name, expr, cr=cr) 
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Band maths operation using {expr} for {self.cxt.current.basename} is done for {po.basename}. Evaluated on CR = {cr}")
            self.controller.refresh(view_key="hol")
            return
        with busy_cursor('Calculating...', self.controller):
            try:
                self.cxt.current = t.band_math_interface(self.cxt.current, name, expr, cr=cr)
            except Exception as e:
                logger.error(f"Band maths operation using {expr} for {self.cxt.current.basename} evaluated on CR = {cr} has failed", exc_info=True)
                self._show_error("Failed operation", f"Failed to evalute expression: {e}")
                return
        logger.info(f"Band maths operation using {expr} for {self.cxt.current.basename} is done. Evaluated on CR = {cr}")
        self.controller.refresh(view_key="vis")
    
    def gen_images(self, multi = False):
        logger.info(f"Button clicked: Generate Images, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Generate Images", msg)
            return
        if multi:
            with busy_cursor("Exporting jpgs....", self.controller):
                for po in self.cxt.ho:
                    try:
                        po.save_all()
                        po.export_images()
                        po.reload_all()
                        po.load_thumbs()
                        logger.info(f"Exported images for {po.basename}")
                    except ValueError as e:
                        logger.error(f"failed to export images for {po.basename}")
                        continue
                return
        with busy_cursor("Exporting jpgs....", self.controller):
            try:
                self.cxt.current.export_images()
            except ValueError as e:
                logger.error(f"failed to export images for {self.cxt.current.basename}")
                return
        logger.info(f"Exported images for {self.cxt.current.basename}")
        
        