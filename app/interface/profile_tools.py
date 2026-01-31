"""
High-level utility functions for spectral analysis on spectral profile data.
Used by UI pages to manipulate HoleObject base datasets.
"""
from pathlib import Path
import re
import logging

from matplotlib.path import Path as mpl_path
import numpy as np

from .. import config
from ..models import ProcessedObject, RawObject, HoleObject
from ..spectral_ops import spectral_functions as sf
from ..spectral_ops import band_maths as bm
from ..spectral_ops import export_ops as exp

logger = logging.getLogger(__name__)
# Tools ready to go on profiles
def profile_kmeans(obj, clusters = 5, iters = 50):
    if not isinstance(obj, HoleObject):
        raise ValueError("Passed object is not a HoleObject")
    if "AvSpectra" not in list(obj.base_datasets.keys()):
        raise ValueError("Base datasets have not been calculated for this hole")
    h, b = obj.base_datasets["AvSpectra"].data.shape
    data = sf.cr(obj.base_datasets["AvSpectra"].data)
    data = data[np.newaxis, :,:]
    img, classes = sf.kmeans_spectral_wrapper(data, clusters, iters)
    img = np.squeeze(img)
    obj.add_product_dataset(f'PROF-kmeans-{clusters}-{iters}INDEX', img.astype(np.int16), '.npy')
    obj.add_product_dataset(f'PROF-kmeans-{clusters}-{iters}CLUSTERS', classes, '.npy')
    legend = []
    for i in np.unique(img):
        legend.append({"index" : int(i), "label" : f"Class {i}"})
    obj.add_product_dataset(f'PROF-kmeans-{clusters}-{iters}LEGEND', legend, '.json')
        
    return obj

def run_feature_extraction(obj, key):
    """
    Estimate minimum wavelength (MWL) position and corresponding absorption depth
    for a specified short-wave infrared absorption feature using multiple
    possible fitting techniques.
    """
    data = obj.base_datasets["AvSpectra"].data
    data = data[np.newaxis, ...]
    data_cr = sf.cr(data)
    mask = np.zeros((data.shape[:2]))
    bands = obj[obj.first_box].bands
    
    pos, dep, feat_mask = sf.Combined_MWL(data, data_cr, mask, bands, key, technique = 'POLY')
    
    obj.add_product_dataset(f'PROF-{key}POS', 
                            np.ma.masked_array(np.squeeze(pos), mask = np.squeeze(feat_mask)), 
                            '.npz')
    obj.add_product_dataset(f'PROF-{key}DEP', 
                            np.ma.masked_array(np.squeeze(dep), mask = np.squeeze(feat_mask)), 
                            '.npz')
    return obj


def band_math_interface(obj, name, expr, cr = False):
    """
    Takes a processed object, a name and an expression and uses the band_maths
    submodule to parse and evaluate the expression on reflectance data. Optionally 
    evaluate the expression on continuum removed data.
    """
    if not cr:
        cube = obj.base_datasets["AvSpectra"].data[np.newaxis, ...]
    else:
        cube = sf.cr(obj.base_datasets["AvSpectra"].data)[np.newaxis, ...]
    bands = obj[obj.first_box].bands
    out = bm.evaluate_expression(expr, cube, bands)
    clean_key = re.sub(r'[\\/:*?"<>|_]', '-', name)
    obj.add_product_dataset(clean_key, np.squeeze(out), '.npy')
    return obj

def wta_min_map(obj, exemplars, coll_name, mode='numpy'):
    """
    Compute a winner-takes-all Pearson class index and best-corr map.

    Parameters
    ----------
    obj : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    
    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMap-pearson-{coll_name}"
    data = sf.remove_hull(obj.base_datasets["AvSpectra"].data)
    bands_nm = obj.get_bands()
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_wta_strict(data[np.newaxis,...], exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]
    obj.add_product_dataset(f'PROF-{key_prefix}INDEX', np.squeeze(index), '.npy')    
    obj.add_product_dataset(f'PROF-{key_prefix}CONF', np.squeeze(confidence), '.npy') 
    obj.add_product_dataset(f'PROF-{key_prefix}LEGEND', legend, '.json') 
    
    return obj

def wta_min_map_SAM(obj, exemplars, coll_name, mode='numpy'):
    """
    Compute a winner-takes-all SAM class index and best-corr map.

    Parameters
    ----------
    obj : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    
    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMap-SAM-{coll_name}"
    data = sf.remove_hull(obj.base_datasets["AvSpectra"].data)
    bands_nm = obj.get_bands()
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_wta_sam_strict(data[np.newaxis,...], exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]
    obj.add_product_dataset(f'PROF-{key_prefix}INDEX', np.squeeze(index), '.npy')    
    obj.add_product_dataset(f'PROF-{key_prefix}CONF', np.squeeze(confidence), '.npy') 
    obj.add_product_dataset(f'PROF-{key_prefix}LEGEND', legend, '.json') 

    return obj


def wta_min_map_MSAM(obj, exemplars, coll_name, mode='numpy'):
    """
    Compute a winner-takes-all MSAM class index and best-corr map.

    Parameters
    ----------
    obj : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    
    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMap-MSAM-{coll_name}"
    data = sf.remove_hull(obj.base_datasets["AvSpectra"].data)
    bands_nm = obj.get_bands()
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_wta_msam_strict(data[np.newaxis,...], exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]
    obj.add_product_dataset(f'PROF-{key_prefix}INDEX', np.squeeze(index), '.npy')    
    obj.add_product_dataset(f'PROF-{key_prefix}CONF', np.squeeze(confidence), '.npy') 
    obj.add_product_dataset(f'PROF-{key_prefix}LEGEND', legend, '.json') 

    return obj


def wta_min_map_user_defined(obj, exemplars, coll_name, ranges, mode='pearson'):
    """
    Compute a winner-takes-all map on a user selected range.

    Parameters
    ----------
    obj : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    coll_name : str text name of the collection passed
    ranges : list[float(min), float(max)]
    mode : str (pearson, sam, msam)
    
    
    """
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMap-{ranges[0]}-{ranges[1]}-{mode}-{coll_name}"
    data = sf.remove_hull(obj.base_datasets["AvSpectra"].data)
    bands_nm = obj.get_bands()
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_subrange(data[np.newaxis,...], exemplar_stack, bands_nm, ranges, mode=mode)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]
    obj.add_product_dataset(f'PROF-{key_prefix}INDEX', np.squeeze(index), '.npy')    
    obj.add_product_dataset(f'PROF-{key_prefix}CONF', np.squeeze(confidence), '.npy') 
    obj.add_product_dataset(f'PROF-{key_prefix}LEGEND', legend, '.json') 

    return obj


"""
CSV Export Tools for HoleObject Product Datasets
"""


def export_profile_to_csv(
    hole: HoleObject,
    key: str,
    output_dir: Path | str | None = None,
    mode: str = "both",
    step: float | None = None
) -> list[Path]:
    """
    Export a HoleObject product dataset to CSV format.
    
    This is the main coordination function that handles type detection,
    validation, and delegates to appropriate CSV writers in spectral_ops.
    
    Parameters
    ----------
    hole : HoleObject
        Hole object containing the dataset to export
    key : str
        Product dataset key to export
    output_dir : Path or str
        Directory where CSV files will be written
    mode : {"full", "stepped", "both"}, default="both"
        Export mode:
        - "full": Raw concatenated data at irregular depths from all boxes
        - "stepped": Resampled to regular grid using hole.step
        - "both": Export both versions
    step : float, optional
        Step size for resampling (in meters). Uses hole.step if None.
        Only relevant if mode is "stepped" or "both".
    
    Returns
    -------
    list[Path]
        Paths to created CSV files
    
    Raises
    ------
    KeyError
        If dataset key not found in product_datasets
    ValueError
        If required data (depths) is missing or data validation fails
    Notes
    -----
    Export is only supported for numeric profile datasets. LEGEND and CLUSTERS
    datasets are skipped with a warning. The function automatically determines
    the appropriate CSV format based on the dataset key suffix:
    
    - *_FRACTIONS: Multi-column fractions with legend
    - *_DOM-MIN or *_INDEX: Categorical indices with legend
    - Other: Continuous values
    """
    if output_dir is None:
        output_dir = Path(hole.root_dir) / "profiles/" 
        logger.info(f'Export path using default {output_dir}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    # Check dataset exists
    if key not in hole.product_datasets:
        available = list(hole.product_datasets.keys())
        logger.error(f"Dataset '{key}' not found in product_datasets. Available keys: {available}")
        raise KeyError(f"Dataset '{key}' not found in product_datasets. Available keys: {available}")
    
    # Check depths dataset exists
    if 'depths' not in hole.base_datasets:
        logger.error(f"Cannot export: 'depths' dataset missing from base_datasets for {hole.hole_id}.")
        raise ValueError(f"Cannot export: 'depths' dataset missing from base_datasets for {hole.hole_id}.")
    
    # Skip non-exportable types
    if key.endswith("LEGEND"):
        logger.warning(f"Cannot export '{key}': LEGEND is metadata (JSON), not a numeric profile")
        return []
    
    if key.endswith("CLUSTERS"):
        logger.warning(f"Cannot export '{key}': CLUSTERS data not supported for CSV export")
        return []
    
    # Validate mode
    valid_modes = {"full", "stepped", "both"}
    if mode not in valid_modes:
        logger.error(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
    
    # SETUP
    step = step or hole.step
    dataset = hole.product_datasets[key]
    depths_full = hole.base_datasets['depths'].data
    created_files = []
    # Determine export type from key suffix
    export_type = _determine_export_type(key)
    logger.info(f"Determined export type '{export_type}' for key '{key}'")
    legend = _get_legend_for_key(hole, key, export_type)
   
    # EXPORT FULL VERSION
    if mode in ("full", "both"):
        csv_path = output_dir / f"{hole.hole_id}-{key}-full.csv"
        
        try:
            exp.export_profile_csv(
                export_type,
                csv_path,
                depths_full,
                dataset.data,
                legend,
                title=key)
            created_files.append(csv_path)
            logger.info(f"Exported full profile: {csv_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to export full profile for '{key}': {e}", exc_info=True)
            raise ValueError(f"Full export failed for '{key}': {e}") from e
    
    # EXPORT STEPPED VERSION
    if mode in ("stepped", "both"):
        csv_path = output_dir / f"{hole.hole_id}-{key}-stepped.csv"
        
        try:
            depths_s, values_s, dominant_s = hole.step_product_dataset(key)
            if key.endswith("DOM-MIN"):
                export_values = dominant_s
            else:
                export_values = values_s
            exp.export_profile_csv(
                export_type,
                csv_path,
                depths_s,
                export_values,
                legend,
                title=key)
            created_files.append(csv_path)
            logger.info(f"Exported stepped profile: {csv_path.name} (step={step}m)")
        except Exception as e:
            logger.error(f"Failed to export stepped profile for '{key}': {e}", exc_info=True)
            raise ValueError(f"Stepped export failed for '{key}': {e}") from e
    
    return created_files

def _determine_export_type(key: str) -> str:
    """
    Determine CSV export format from dataset key suffix.
    
    Uses naming conventions to classify datasets:
    - *_FRACTIONS → "fractions" (2D mineral fractions)
    - *_DOM-MIN or *_INDEX → "categorical" (1D mineral indices)
    - Everything else → "continuous" (1D continuous values)
    
    Parameters
    ----------
    key : str
        Dataset key
    
    Returns
    -------
    str
        Export type: "continuous", "fractions", or "categorical"
    """
    if key.endswith("FRACTIONS"):
        return "fractions"
    
    if key.endswith(("DOM-MIN", "INDEX")):
        return "categorical"
    
    # Default to continuous for unknown types
    # This handles: AvSpectra, feature positions/depths, custom datasets
    return "continuous"


def _get_legend_for_key(hole: HoleObject, key: str, export_type: str) -> dict | None:
    """
    Retrieve legend dictionary for datasets that need mineral name lookup.
    
    Legends are stored as separate *_LEGEND datasets and are referenced
    by naming convention (e.g., SWIR-MIN_FRACTIONS uses SWIR-MIN_LEGEND).
    
    Parameters
    ----------
    hole : HoleObject
        Hole object to search for legend
    key : str
        Dataset key (e.g., 'SWIR-MIN_FRACTIONS')
    export_type : str
        Export type determined from key
    
    Returns
    -------
    dict or None
        Legend dictionary mapping indices to mineral names.
        Format: {0: "Mineral1", 1: "Mineral2", ..., N: "Unclassified"}
        Returns None if legend not found or not needed.
    """
    if export_type == "fractions":
        # FRACTIONS needs LEGEND for column headers
        legend_key = key.replace("FRACTIONS", "LEGEND")
        legend_ds = hole.product_datasets.get(legend_key)
        
        if legend_ds:
            return legend_ds.data
        else:
            logger.warning(
                f"No legend found for '{key}' (expected '{legend_key}'). "
                f"CSV will use generic column names."
            )
            return None
    
    elif export_type == "categorical":
        # DOM-MIN and INDEX need LEGEND for mineral name column
        if key.endswith("DOM-MIN"):
            legend_key = key.replace("DOM-MIN", "LEGEND")
        elif key.endswith("INDEX"):
            legend_key = key.replace("INDEX", "LEGEND")
        else:
            return None
        
        legend_ds = hole.product_datasets.get(legend_key)
        
        if legend_ds:
            return legend_ds.data
        else:
            logger.warning(f"No legend found for '{key}' (expected '{legend_key}'). Export will only include numeric indices."
            )
            return None
    
    # Continuous data doesn't need legend
    return None
























