"""
Operations that work on single box or full hole.
Stateful class that holds context and controller references.
"""
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import (QInputDialog, 
                             QFileDialog, 
                             QDialog, 
                             QMessageBox, 
                             QVBoxLayout, 
                             QHBoxLayout, 
                             QLabel, 
                             QLineEdit, 
                             QDoubleSpinBox, 
                             QDialogButtonBox)

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
    
    def custom_feature_act(self, multi=False):
        """Get custom feature parameters from user and run extraction."""
        logger.info(f"Button clicked: Custom Feature Extraction, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Custom Feature Extraction", msg)
            return
        # Get custom feature parameters from dialog
         # Get custom feature dict from single dialog
        custom_feature = CustomFeatureDialog.get_custom_feature(parent=self.controller)
        
        if not custom_feature:
            logger.warning("Custom feature cancelled by user")
            return
        feature_name = list(custom_feature.keys())[0]
        if multi:
            with busy_cursor(f'custom feature extraction {feature_name}....', self.controller):
                for po in self.cxt.ho:
                    t.run_feature_extraction(po, custom_feature)
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Extract custom feature {feature_name} for {po.basename} done")
                self.controller.refresh()
            return
        
        with busy_cursor(f'extracting {feature_name}...', self.controller):
            self.cxt.current = t.run_feature_extraction(self.cxt.current, custom_feature)
        logger.info(f"Extract custom feature {feature_name} for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")

    def cache_features(self, multi = False):
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Checking features", msg)
            return
        if multi:
            with busy_cursor(f'feature checking....', self.controller) as progress:
                for po in self.cxt.ho:
                    try:
                        progress.set(f"Caching features for {po.basename}")
                        t.cache_feature_map(po)
                        po.commit_temps()
                        po.save_all()
                        po.reload_all()
                        po.load_thumbs()
                    except AssertionError as e:
                        logger.error(f"Cache features for {po.basename} failed", exc_info = True)
                        continue
                    logger.info(f"cached features for {po.basename} done")
                
                self.controller.refresh()
            return
        with busy_cursor(f'Caching ...', self.controller):
            try:
                self.cxt.current = t.cache_feature_map(self.cxt.current)
            except AssertionError as e:
                logger.error(f"cache features for {self.cxt.current.basename} failed", exc_info = True)
                self._show_error("cache features", "Could not find required bands, try different features or adjust ranges")
                return
        logger.info(f"Cache Features for {self.cxt.current.basename} done")
        self.controller.refresh(view_key="vis")

    def run_feature_extraction(self, key, multi = False):
        logger.info(f"Button clicked: Extract feature {key}, Multi-mode = {multi}")
        valid_state, msg = self.cxt.requires(self.cxt.HOLE if multi else self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Feature Extraction", msg)
            return
        if multi:
            with busy_cursor(f'feature extraction {key}....', self.controller) as progress:
                for po in self.cxt.ho:
                    try:
                        progress.set(f"Extracting feature {key} for {po.basename}")
                        t.run_feature_extraction(po, key)
                        po.commit_temps()
                        po.save_all()
                        po.reload_all()
                        po.load_thumbs()
                    except AssertionError as e:
                        logger.error(f"Extract feature {key} for {po.basename} failed", exc_info = True)
                        continue
                    logger.info(f"Extract feature {key} for {po.basename} done")
                
                self.controller.refresh()
            return
        with busy_cursor(f'extracting {key}...', self.controller):
            try:
                self.cxt.current = t.run_feature_extraction(self.cxt.current, key)
            except AssertionError as e:
                logger.error(f"Extract feature {key} for {self.cxt.current.basename} failed", exc_info = True)
                self._show_error("Feature Extraction", "Could not find required bands, try different features or adjust ranges")
                return
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
            with busy_cursor('correlation...', self.controller) as progress:
                for po in self.cxt.ho:
                    progress.set(f"Pearson with collection {name} for {po.basename}")
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
            with busy_cursor('correlation...', self.controller) as progress:
                for po in self.cxt.ho:
                    progress.set("SAM with collection {name} for {po.basename}")
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
            with busy_cursor('correlation...', self.controller) as progress:
                for po in self.cxt.ho:
                    progress.set(f"MSAM with collection {name} for {po.basename}")
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
            with busy_cursor('correlation...', self.controller) as progress:
                for po in self.cxt.ho:
                    progress.set(f"Multirange with collection {name} using {mode} for {po.basename}")
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
            with busy_cursor('correlation...', self.controller) as progress:
                for po in self.cxt.ho:
                    try:
                        progress.set(f"Defined range correlation with collection {name} using {mode} and range ({start_nm}:{stop_nm}) for {po.basename}")
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
            with busy_cursor('Clustering...', self.controller) as progress:
                for po in self.cxt.ho:
                    progress.set(f"Clustering ({clusters}, {iters}) for {po.basename}")
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
            with busy_cursor('clustering...', self.controller) as progress:
                for po in self.cxt.ho:
                    progress.set(f"Band maths operation using {expr} for {po.basename} evaluating on CR = {cr}")
                    t.band_math_interface(po, name, expr, cr=cr) 
                    po.commit_temps()
                    po.save_all()
                    po.reload_all()
                    po.load_thumbs()
                    logger.info(f"Band maths operation using {expr} is done for {po.basename}. Evaluated on CR = {cr}")
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
            with busy_cursor("Exporting jpgs....", self.controller) as progress:
                for po in self.cxt.ho:
                    try:
                        progress.set(f"Exporting images for {po.basename}")
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
        
class CustomFeatureDialog(QDialog):
    """Dialog to get custom feature parameters from user."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Feature Definition")
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Feature name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Feature Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Custom2300")
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # Wavelength min
        wav_min_layout = QHBoxLayout()
        wav_min_layout.addWidget(QLabel("Feature Wavelength Min (nm):"))
        self.wav_min_spin = QDoubleSpinBox()
        self.wav_min_spin.setRange(0, 20000)
        self.wav_min_spin.setDecimals(1)
        self.wav_min_spin.setValue(2185.0)
        self.wav_min_spin.setSingleStep(10.0)
        wav_min_layout.addWidget(self.wav_min_spin)
        layout.addLayout(wav_min_layout)
        
        # Wavelength max
        wav_max_layout = QHBoxLayout()
        wav_max_layout.addWidget(QLabel("Feature Wavelength Max (nm):"))
        self.wav_max_spin = QDoubleSpinBox()
        self.wav_max_spin.setRange(0, 20000)
        self.wav_max_spin.setDecimals(1)
        self.wav_max_spin.setValue(2215.0)
        self.wav_max_spin.setSingleStep(10.0)
        wav_max_layout.addWidget(self.wav_max_spin)
        layout.addLayout(wav_max_layout)
        
        # CR crop min
        cr_min_layout = QHBoxLayout()
        cr_min_layout.addWidget(QLabel("CR Crop Min (nm):"))
        self.cr_min_spin = QDoubleSpinBox()
        self.cr_min_spin.setRange(0, 20000)
        self.cr_min_spin.setDecimals(1)
        self.cr_min_spin.setValue(2120.0)
        self.cr_min_spin.setSingleStep(10.0)
        cr_min_layout.addWidget(self.cr_min_spin)
        layout.addLayout(cr_min_layout)
        
        # CR crop max
        cr_max_layout = QHBoxLayout()
        cr_max_layout.addWidget(QLabel("CR Crop Max (nm):"))
        self.cr_max_spin = QDoubleSpinBox()
        self.cr_max_spin.setRange(0, 20000)
        self.cr_max_spin.setDecimals(1)
        self.cr_max_spin.setValue(2245.0)
        self.cr_max_spin.setSingleStep(10.0)
        cr_max_layout.addWidget(self.cr_max_spin)
        layout.addLayout(cr_max_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def validate_and_accept(self):
        """Validate inputs before accepting."""
        import re
        from PyQt5.QtWidgets import QMessageBox
        
        name = self.name_edit.text().strip()
        
        # Check name is not empty
        if not name:
            QMessageBox.warning(self, "Invalid Input", "Feature name cannot be empty.")
            return
        
        # Validate wavelength ordering
        wav_min = self.wav_min_spin.value()
        wav_max = self.wav_max_spin.value()
        cr_min = self.cr_min_spin.value()
        cr_max = self.cr_max_spin.value()
        
        if wav_min >= wav_max:
            QMessageBox.warning(self, "Invalid Range", "Feature wavelength min must be less than max.")
            return
        
        if cr_min >= cr_max:
            QMessageBox.warning(self, "Invalid Range", "CR crop min must be less than max.")
            return
        
        if not (cr_min <= wav_min and wav_max <= cr_max):
            QMessageBox.warning(
                self, 
                "Invalid Range", 
                "Feature wavelength range must be within CR crop range.\n"
                f"CR range: [{cr_min}, {cr_max}]\n"
                f"Feature range: [{wav_min}, {wav_max}]"
            )
            return
        
        self.accept()
    
    def get_feature_dict(self):
        """Return the custom feature dict in the expected format."""
        import re
        name = self.name_edit.text().strip()
        # Auto-sanitize the name
        name = re.sub(r'[\\/:*?"<>|_]', '-', name)
        
        if not name:
            return None
        
        return {
            name: [
                self.wav_min_spin.value(),
                self.wav_max_spin.value(),
                self.cr_min_spin.value(),
                self.cr_max_spin.value()
            ]
        }
    
    @staticmethod
    def get_custom_feature(parent=None):
        """Static method to show dialog and return result."""
        dialog = CustomFeatureDialog(parent)
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_feature_dict()
        return None