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

FEATURE_KEYS = [
    '1400W', '1480W', '1550W', '1760W', '1850W',
    '1900W', '2080W', '2160W', '2200W', '2250W',
    '2290W', '2320W', '2350W', '2390W', '2950W',
    '2950AW', '2830W', '3000W', '3500W', '4000W',
    '4000WIDEW', '4470TRUEW', '4500SW', '4500CW',
    '4670W', '4920W', '4000V_NARROWW', '4000shortW', '2950BW'
]

class VisActions(BaseActions):
    """Raw data operations"""
    def __init__(self, context, ribbon, parent=None, box_ops=None):
        self.box_ops = box_ops
        self.legend_mapping_path = None
        super().__init__(context, ribbon, parent)
    
    
    def stage_ribbon(self):
        
        self.extract_feature_list = []
        for key in FEATURE_KEYS:
            self.extract_feature_list.append((key, lambda _, k=key: self.box_ops.run_feature_extraction(k)))
        
        self._register_group('Visualise', [
    ("button", "Quick Cluster", self.box_ops.act_kmeans, "Performs unsupervised k-means clustering"),
    ("menu", "Correlation", [
        ("MineralMap Pearson (Winner-takes-all)", self.box_ops.act_vis_correlation, "Performs Pearson correlation against selected collection from the library"),
        ("MineralMap SAM (Winner-takes-all)", self.box_ops.act_vis_sam, "Performs Spectral Angle Mapping against selected collection from the library"),
        ("MineralMap MSAM (Winner-takes-all)", self.box_ops.act_vis_msam, "Performs Modified Spectral Angle Mapping against selected collection from the library"),
        ("Multi-range check (Winner-takes-all)", self.box_ops.act_vis_multirange, "Performs custom multi-window matching"),
        ("select range", self.box_ops.act_subrange_corr, "Performs correlation on a chosed wavelength range"),
        ("Re-map legends", self._remap_legends)
    ]),
    ("menu", "Features", self.extract_feature_list, "Performs Minimum Wavelength Mapping"),
    ("button", "Band Maths", self.box_ops.act_band_maths, "Open the band maths expression window"),
    ("menu", "Library building", [
        ("Add spectra", self.act_lib_pix, "Add a single pixel spectra to the current library\n WARNING: This will modify the library on disk, use a back up"),
        ("Add region average", self.act_lib_region, "Add the average spectra of a region to the current library\n WARNING: This will modify the library on disk, use a back up"),
    ]),
    ("button", "Generate Images", self.box_ops.gen_images, "Generates full size images of all products and base datasets in an outputs folder"),
])
        
    # -------- VISUALISE actions --------
    
    

    def _remap_legends(self):
        logger.info(f"Button clicked: Remap legends")
        if self.legend_mapping_path is None:
            
            self._show_info(
            "Legend remapping",
            "Legend remapping groups detailed spectral hits into "
            "interpretable mineral classes.\n\n"
            "You can choose *any* JSON file that defines these classes.\n"
            "A recommended default lives in the 'resources' folder.\n\n"
            "Please select a remapping file now."
        )

        # Default directory for the QFileDialog
        

            path, _ = QFileDialog.getOpenFileName(
                self.controller,
                "Select legend mapping JSON",
                ".",
                "JSON files (*.json)"
            )
            if not path:
                return  # user cancelled — do nothing safely
            self.legend_mapping_path = path
        
        valid_state, msg = self.cxt.requires(self.cxt.PROCESSED)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Add to Library", msg)
            return
        try:
            self.cxt.current = t.clean_legends(self.cxt.current, self.legend_mapping_path)
        except Exception as e:
            logger.error(f"There is no processed dataset loaded", exc_info=True)
            self._show_error("Failed operation", f"Failed to remap legends: {e}")
            return
        logger.info(f"Legends remapped")
        self.controller.refresh()
    
    def act_lib_pix(self):
        """Add a single pixel spectrum to the current library."""
        logger.info(f"Button clicked: Add pixel to library")
        valid_state, msg = self.cxt.requires(self.cxt.ADD_TO_LIBRARY)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Add to Library", msg)
            return
           
        p = self.controller.active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            logger.warning("Library add cancelled, due to absence of either page, dispatcher or canvas")
            return
        
        def handle_point_click(y, x):
            try:
                spectrum = self.cxt.current.savgol[int(y), int(x), :]
                wavelengths_nm = self.cxt.current.bands
                        
                # Ask for metadata
                dlg = LibMetadataDialog(parent=self.controller)
                if dlg.exec() != QDialog.Accepted:
                    logger.warning("Library add cancelled in dialogue")
                    return  # user cancelled
                
                metadata = dlg.get_metadata()
                name = metadata.get("Name", "").strip()
                if not name:
                    logger.warning("Library add cancelled, no metadata added")
                    self._show_error(
                        "Add to Library", 
                        "Name is a mandatory field"
                    )
                    return
                
                metadata['SampleNum'] = f"Hole: {self.cxt.current.metadata.get('borehole id', 'Unknown')} Box: {self.cxt.current.metadata.get('box number', 'Unknown')} Pixel: ({int(y)}, {int(x)})"
                with busy_cursor('Adding to library...', self.controller):
                    sample_id = self.cxt.library.add_sample(
                        name=name,
                        wavelengths_nm=wavelengths_nm,
                        reflectance=spectrum,
                        metadata=metadata
                    )
                self.controller.refresh(view_key="lib")
                logger.info(f"Added spectrum '{name}' to library (ID: {sample_id})")
                                
            except Exception as e:
                logger.error("Failed to add pixel spectra to library", exc_info=True)
                self._show_error(
                    "Add to Library", 
                    f"Failed to add spectrum to library: {e}"
                )
            finally:
                p.dispatcher.clear_all_temp()
        
        p.dispatcher.set_single_click(handle_point_click)
        
        

        
    def act_lib_region(self):
        """Add the average spectrum of a region to the current library."""
        logger.info(f"Button clicked: Add region average to library")
        valid_state, msg = self.cxt.requires(self.cxt.ADD_TO_LIBRARY)
        if not valid_state:
            logger.warning(msg)
            self._show_error("Add to Library", msg)
            return
        
        p = self.controller.active_page()
        if not p or not p.dispatcher or not p.left_canvas:
            logger.warning("Library add cancelled, due to absence of either page, dispatcher or canvas")
            return
        
        def _on_rect(y0, y1, x0, x1):
            try:
                # Extract region and compute average spectrum
                region = self.cxt.current.savgol[y0:y1, x0:x1, :]
                
                # Use mask if available to exclude masked pixels
                if self.cxt.current.has('mask'):
                    mask_region = self.cxt.current.mask[y0:y1, x0:x1]
                    valid_pixels = region[mask_region == 0]
                    if valid_pixels.size == 0:
                        logger.warning("Library add cancelled, Selected region contains no valid (unmasked) pixels.")
                        self._show_error(
                            "Add to Library", 
                            "Selected region contains no valid (unmasked) pixels."
                        )
                        return
                    avg_spectrum = valid_pixels.mean(axis=0)
                    pixel_count = len(valid_pixels)
                else:
                    avg_spectrum = region.reshape(-1, region.shape[-1]).mean(axis=0)
                    pixel_count = (y1 - y0) * (x1 - x0)
                
                wavelengths_nm = self.cxt.current.bands
                
                dlg = LibMetadataDialog(parent=self.controller)
                
                if dlg.exec() != QDialog.Accepted:
                    logger.warning("Library add cancelled in dialogue")
                    return  # user cancelled
                
                metadata = dlg.get_metadata()
                name = metadata.get("Name", "").strip()
                if not name:
                    logger.warning("Library add cancelled, no metadata added")
                    self._show_error(
                        "Add to Library", 
                        "Name is a mandatory field"
                    )
                    return
                metadata['SampleNum'] = f"Hole: {self.cxt.current.metadata.get('borehole id', 'Unknown')} Box: {self.cxt.current.metadata.get('box number', 'Unknown')} Region: ({y0}-{y1},{x0}-{x1})"
                
                
                with busy_cursor('Adding to library...', self.controller):
                    sample_id = self.cxt.library.add_sample(
                        name=name,
                        wavelengths_nm=wavelengths_nm,
                        reflectance=avg_spectrum,
                        metadata=metadata
                    )
                    self.controller.refresh(view_key="lib")
                logger.info(f"Added averaged spectrum '{name}' ({pixel_count} pixels) to library (ID: {sample_id})")
                                
            except Exception as e:
                logger.error(f"Added averaged spectrum '{name}' ({pixel_count} pixels) to library (ID: {sample_id})", exc_info=True)
                self._show_error(
                    "Add to Library", 
                    f"Failed to add spectrum to library: {e}"
                )
            finally:
                p.dispatcher.clear_all_temp()
        
        p.dispatcher.set_rect(_on_rect)
        p.left_canvas.start_rect_select()
        