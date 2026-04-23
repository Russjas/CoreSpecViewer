"""
High-level utility functions for cropping, masking, unwrapping, and feature extraction.
Used by UI pages to manipulate RawObject and ProcessedObject datasets.
"""
from pathlib import Path
import re
import logging
import time

from matplotlib.path import Path as mpl_path
import numpy as np

from ..config import config
from ..models import ProcessedObject, RawObject
from ..spectral_ops.visualisation import get_false_colour
from ..spectral_ops.processing import resample_spectrum, unwrap_from_stats, remove_cont

from ..spectral_ops import analysis as sa
from ..spectral_ops import masking as sm
from ..spectral_ops import remap_legend as rl
from ..spectral_ops import band_maths as bm

logger = logging.getLogger(__name__)

#======Getting and setting app configs ========================================


def get_config():
    """
    Loads the config dictionary - a single mutable dictionary of config
    patterns used accross the app
    """
    return config.as_dict()

def modify_config(key, value):
    """
    Sets user selected values in the config dictionary - a single mutable 
    dictionary of config patterns used accross the app
    """
    config.set(key, value)

#==== Data loading helper functions ===========================================

def load(path):
    """
    Load a RawObject or ProcessedObject depending on path type.
    - directory  → RawObject.from_Lumo_directory
    - single file → ProcessedObject.from_path
    Returns the created object or None.
    """
    if not path:
        return None

    p = Path(path)
    if p.is_dir():
        return RawObject.from_Lumo_directory(p)
    elif p.is_file():
        return ProcessedObject.from_path(p)
    else:
        return None


def discover_lumo_directories(root_dir: Path) -> list[Path]:
    """
    Recursively discover all subdirectories under `root_dir`.
    Excludes capture and metadata subdirectories inside lumo parent directories
    to avoid double processing.

    Parameters
    ----------
    root_dir : Path
        A pathlib.Path object representing the starting directory.

    Returns
    -------
    list[Path]
        A sorted list of absolute Path objects including the root itself.
    """
    if not root_dir.is_dir():
        raise NotADirectoryError(f"{root_dir} is not a valid directory.")

    # Use rglob('*') for recursive traversal, filtering for directories only
    dirs = [root_dir.resolve()]  # include the root
    try:
        for p in root_dir.rglob('*'):
                if p.is_dir():
                    rel = p.relative_to(root_dir).as_posix().lower()
                    if "capture" not in rel and 'metadata' not in rel and "calibrations" not in rel:
                        dirs.append(p.resolve())
    except PermissionError:
        pass


    return sorted(set(dirs))


#======= Cropping and reset functions for RO or PO data =======================




def crop(obj, y_min, y_max, x_min, x_max):
    """
    Generic, window-agnostic spatial crop.

    - For RawObject → create temp_reflectance (preview).
    - For ProcessedObject → create temp datasets for all 2D/3D arrays.
    """
    start_time = time.perf_counter()
    
    if isinstance(obj, RawObject):
        if not hasattr(obj, "reflectance") or obj.reflectance is None:
            refl_start = time.perf_counter()
            obj.get_reflectance()
            logger.debug(f"get_reflectance took {time.perf_counter() - refl_start:.3f}s")
            
        if hasattr(obj, "temp_reflectance") and obj.temp_reflectance is not None:
            arr = obj.temp_reflectance
        else:
            arr = obj.reflectance

        crop_start = time.perf_counter()
        obj.temp_reflectance = arr[y_min:y_max, x_min:x_max]
        logger.debug(f"RawObject crop took {time.perf_counter() - crop_start:.3f}s")
        
        logger.debug(f"crop(RawObject) total: {time.perf_counter() - start_time:.3f}s")
        return obj

    elif isinstance(obj, ProcessedObject):
        # union of base + temps
        keys = set(obj.datasets.keys()) | set(obj.temp_datasets.keys())
        
        # Ensure mask is processed first for thumbnail generation
        ordered_keys = ['mask'] if 'mask' in keys else []
        ordered_keys.extend([k for k in keys if k != 'mask'])
        
        total_crop_time = 0
        for key in ordered_keys:
            key_start = time.perf_counter()
            logger.info(f"cropping {obj.basename} {key} dataset")
            
            # choose source: prefer temp if present
            src = obj.temp_datasets[key].data if obj.has_temp(key) else obj.datasets[key].data
            
            if isinstance(src, np.ndarray) and src.ndim > 1:
                array_copy_start = time.perf_counter()
                cropped_copy = np.array(src[y_min:y_max, x_min:x_max, ...])
                copy_time = time.perf_counter() - array_copy_start
                
                if obj.has_temp(key):
                    obj.temp_datasets[key].data = cropped_copy
                else:
                    add_start = time.perf_counter()
                    obj.add_temp_dataset(key, cropped_copy)
                    add_time = time.perf_counter() - add_start
                    logger.debug(f"  add_temp_dataset({key}) took {add_time:.3f}s")
                
                key_time = time.perf_counter() - key_start
                total_crop_time += key_time
                logger.debug(f"  {key}: copy={copy_time:.3f}s, total={key_time:.3f}s, shape={cropped_copy.shape}")
        
        logger.debug(f"crop(ProcessedObject) total: {time.perf_counter() - start_time:.3f}s ({len(ordered_keys)} datasets)")
        return obj

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")


def crop_auto(obj):
    """
    Window-agnostic auto crop using the detect rectangles method as I have
    nothing better for now.

    - For RawObject → create temp_reflectance (preview).
    - For ProcessedObject → create temp datasets for all 2D/3D arrays.
    """
    if isinstance(obj, RawObject):
        if not hasattr(obj, "reflectance") or obj.reflectance is None:
            obj.get_reflectance()
        if hasattr(obj, "temp_reflectance") and obj.temp_reflectance is not None:
            arr = obj.temp_reflectance
        else:
            arr = obj.reflectance
        img = get_false_colour(arr)
        img = (img*255).astype(np.uint8)
        cropped, slicer = sm.detect_slice_rectangles_robust(img)
        if slicer is None:
            return obj
        try:
            test = arr[slicer]
        except Exception:
            return obj
        if not isinstance(test, np.ndarray) or test.ndim < 2 or 0 in test.shape:
            return obj
        obj.temp_reflectance = test
        return obj

    elif isinstance(obj, ProcessedObject):
        # Detect crop region using savgol
        base = getattr(obj, "savgol", None)
        if not isinstance(base, np.ndarray) or base.ndim < 2 or 0 in base.shape:
            return obj
        img = get_false_colour(base)
        img = np.asarray(img)
        if img.ndim < 2 or 0 in img.shape:
            return obj
        img = (img * 255).astype(np.uint8, copy=False)

        cropped, slicer = sm.detect_slice_rectangles_robust(img)
        if slicer is None:
            return obj
        try:
            test_ref = base[slicer]
        except Exception:
            return obj
        if not isinstance(test_ref, np.ndarray) or test_ref.ndim < 2 or 0 in test_ref.shape:
            return obj
        
        # slicer is valid → apply to all datasets
        keys = set(obj.datasets.keys()) | set(obj.temp_datasets.keys())
        
        # Ensure mask is processed first for thumbnail generation
        ordered_keys = ['mask'] if 'mask' in keys else []
        ordered_keys.extend([k for k in keys if k != 'mask'])
        
        for key in ordered_keys:
            logger.info(f"auto-cropping {obj.basename} {key} dataset")
            
            src = obj.temp_datasets[key].data if obj.has_temp(key) else obj.datasets[key].data
            
            if getattr(src, "ndim", 0) <= 1:
                continue
            try:
                cropped = src[slicer]
            except Exception:
                continue
            if 0 in cropped.shape:
                continue

            cropped_copy = np.array(cropped)

            if obj.has_temp(key):
                obj.temp_datasets[key].data = cropped_copy
            else:
                obj.add_temp_dataset(key, cropped_copy)
        
        return obj
    
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")


def reset(obj):
    """
    Clears temporary datasets from RO or PO
    """
    if obj.is_raw:
        obj.temp_reflectance = None
    else:
        obj.clear_temps()
    return obj

# =============Masking tools===================================================

def mask_rect(obj, ymin, ymax, xmin, xmax):
    """
    Adds a user selected rectangle to the mask.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    msk = np.array(obj.mask)
    msk[ymin:ymax, xmin:xmax] = 1
    obj.add_temp_dataset('mask', data = msk)
    return obj


def mask_point(obj, mode, y, x):
    """
    Uses a user defined point to either;
    new:      Create a new mask and mask where correlation between all spectra 
              and the user selected spectra are >0.9

    enhance: Using the existing mask additionally mask where correlation 
             between all spectra and the user selected spectra are >0.9

    line:    Using the existing mask additionally mask the user selected column
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    if mode == 'new':
        msk = np.zeros(obj.savgol.shape[:2])
        pixel_vec = obj.savgol_cr[y, x, :]
        corr = sa.numpy_pearson(obj.savgol_cr, pixel_vec)
        msk[corr > 0.9] = 1
        obj.add_temp_dataset('mask', data = msk)
        return obj
    if mode == 'enhance':
        msk = np.array(obj.mask)
        pixel_vec = obj.savgol_cr[y, x, :]
        corr = sa.numpy_pearson(obj.savgol_cr, pixel_vec)
        msk[corr > 0.9] = 1
        obj.add_temp_dataset('mask', data = msk)
        return obj
    if mode == 'line':
        msk = np.array(obj.mask)
        msk[:, x] = 1
        obj.add_temp_dataset('mask', data = msk)
        return obj


def mask_polygon(obj, vertices_rc, mode = "mask outside"):
    """
    Given polygon vertices in (row, col) image indices, set outside to 1 (masked).
    Creates/updates a temp 'mask' dataset.

    - If no mask exists, starts from zeros.
    - Keeps interior as-is (commonly 0), sets outside to 1.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    if obj.is_raw:
        return obj
    H, W = obj.savgol.shape[:2]

    poly = np.asarray(vertices_rc, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        return obj  # ignore bad polygons

    rr = np.arange(H)
    cc = np.arange(W)
    grid_c, grid_r = np.meshgrid(cc, rr)           # (H,W)
    pts = np.column_stack([grid_c.ravel(), grid_r.ravel()])  # (H*W,2) in (x=col, y=row)
    inside = mpl_path(poly[:, ::-1]).contains_points(pts)        # flip to (x,y)
    inside = inside.reshape(H, W)
    if mode == "mask outside":
        # outside = ~inside  -> set to 1
        msk = np.array(obj.mask)
        msk[~inside] = 1
    elif mode == "mask inside":
        msk = np.array(obj.mask)
        msk[inside] = 1
    elif mode == "unmask outside":
        msk = np.array(obj.mask)
        msk[~inside] = 0
    elif mode == "unmask inside":
        msk = np.array(obj.mask)
        msk[inside] = 0

    obj.add_temp_dataset('mask', data = msk)
    return obj


def improve_mask(obj, mode="vertical"):
    """
    Heuristically thicken a mask column-wise using simple occupancy.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    logger.info(f"improve_mask called with mode={mode}")  # ADD THIS
    if mode=="vertical":
        logger.info("Running improve_mask_from_graph")  # ADD THIS
        msk = sm.improve_mask_from_graph(obj.mask) 
    else:
        logger.info("Running hough_line_connection")  # ADD THIS
        msk = sm.hough_line_connection(obj.mask)
    obj.add_temp_dataset('mask', data = msk)
    return obj


def despeckle_mask(obj):
    """
    Heuristically thicken a mask column-wise using simple occupancy.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    msk = sm.despeckle_mask(obj.mask)
    obj.add_temp_dataset('mask', data = msk)
    return obj


def mask_all(obj):
    """
    Set entire mask to 1 (all pixels masked).
    Useful for starting with everything masked, then selectively unmasking.
    """
    if obj.is_raw:
        return obj
    H, W = obj.savgol.shape[:2]
    msk = np.ones((H, W), dtype=np.uint8)
    obj.add_temp_dataset('mask', data=msk)
    return obj

def invert_mask(obj):
    """
    Flip mask values: 0 → 1, 1 → 0.
    Useful for inverting imported masks or switching mask conventions.
    """
    if obj.is_raw:
        return obj
    msk = (~obj.mask.astype(bool)).astype(np.uint8)
    obj.add_temp_dataset('mask', data=msk)
    return obj

#============ Unwrapping tools ================================================

def calc_unwrap_stats(obj):
    """
    Compute connected components on the (eroded) inverse of a mask and sets the
    returned stats to a dataset for use in future unwrapping operations.
    Also creates a dataset image of the derived segments for user inspection
    """
    label_image, stats = sm.get_stats_from_mask(obj.mask)
    label_image = label_image / np.max(label_image)
    obj.add_temp_dataset('stats', stats, '.npy')
    obj.add_temp_dataset('segments', label_image, '.npy')

    return obj


def unwrapped_output(obj):
    """
    Uses previously computed unwrap stats to produce a vertically concatenated
    core box spectral cube and mask. Calculates mask-aware per pixel depths
    using depth values held in the metadata
    """
    dhole_reflect = unwrap_from_stats(obj.mask, obj.savgol, obj.stats)
    dhole_depths = np.linspace(float(obj.metadata['core depth start']), float(obj.metadata['core depth stop']),
                                    dhole_reflect.shape[0])

    obj.add_temp_dataset('DholeAverage', dhole_reflect.data, '.npy')
    obj.add_temp_dataset('DholeMask', dhole_reflect.mask, '.npy')
    obj.add_temp_dataset('DholeDepths', dhole_depths, '.npy')

    return obj
#==========pass through helpers===============================================
def get_cr(spectra):
    return remove_cont(spectra)

#========= Reflectance interpretation tools ===================================

def run_feature_extraction(obj, key):
    """
    Estimate minimum wavelength (MWL) position and corresponding absorption depth
    for a specified short-wave infrared absorption feature using multiple
    possible fitting techniques.
    
    Parameters
    ----------
    key : str or dict
        Either a string key from the standard features (e.g., '2200W')
        OR a dict in format {feature_name: [wav_min, wav_max, cr_min, cr_max]}
    """
    cache_available = ('feature_indices' in obj.datasets and 
            'feature_heights' in obj.datasets)
    if cache_available:
        cached_arrays = (
            obj.datasets['feature_indices'].data,
            obj.datasets['feature_heights'].data
        )
    else:
        cached_arrays = None
    if isinstance(key, dict):
        # Custom feature - extract the name for dataset keys
        feature_name = list(key.keys())[0]
        pos, dep, feat_mask = sa.Combined_MWL(obj.savgol, obj.savgol_cr, obj.mask, obj.bands, key, technique='POLY', cached_arrays=cached_arrays)
        obj.add_temp_dataset(f'{feature_name}POS', np.ma.masked_array(pos, mask=feat_mask), '.npz')
        obj.add_temp_dataset(f'{feature_name}DEP', np.ma.masked_array(dep, mask=feat_mask), '.npz')
    else:
        pos, dep, feat_mask = sa.Combined_MWL(obj.savgol, obj.savgol_cr, obj.mask, obj.bands, key, technique = 'POLY', cached_arrays=cached_arrays)
        obj.add_temp_dataset(f'{key}POS', np.ma.masked_array(pos, mask = feat_mask), '.npz')
        obj.add_temp_dataset(f'{key}DEP', np.ma.masked_array(dep, mask = feat_mask), '.npz')
    return obj


def cache_feature_map(obj, max_feats=20):
    """
    Compute and cache default features for a ProcessedObject.
    
    This is called once per box to enable fast feature extraction.
    """
    
    
    logger.info(f"Computing feature map for {obj.basename}...")
    
    feature_indices, feature_heights, feature_counts = sa.compute_feature_map(
        obj.savgol_cr, max_feats=max_feats
    )
    
    # Store as datasets on the object
    obj.add_temp_dataset('feature_indices', feature_indices, ext='.npy')
    obj.add_temp_dataset('feature_heights', feature_heights, ext='.npy')
    obj.add_temp_dataset('feature_counts', feature_counts, ext='.npy')
    return obj
    
    


def quick_corr(obj, x, y, key):
    """
    Runs a pearson correlation of a user selected spectum against the objects
    continuum removed dataset.
    Currently result is stored as a masked array in the temp dataset.
    Database mineral names are used as the key, but often contain characters that are
    illegal in file paths. As the key is used in the save path of the resulting dataset
    it needs to be sanitised.
    The clean key is returned in addion to the processed object, so the caller has reference
    the generated dataset.
    """
    clean_key = re.sub(r'[\\/:*?"<>|_]', '-', key)
    if obj.is_raw:
        return None
    res_y = resample_spectrum(x, y, obj.bands)
    corr = np.ma.masked_array(sa.numpy_pearson(obj.savgol_cr, remove_cont(res_y)), mask = obj.mask)
    obj.add_temp_dataset(clean_key, corr, '.npz')
    return obj, clean_key


def wta_multi_range_minmap(obj, exemplars, coll_name, mode='pearson'):
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMapMulti-{mode}-{coll_name}"
    data = obj.savgol
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    best_idx, best_score, best_window = sa.mineral_map_multirange(data,
                                                               exemplar_stack,
                                                               bands_nm,
                                                               mode=mode
                                                               )
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]
    obj.add_temp_dataset(f"{key_prefix}INDEX", best_idx.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', best_score, '.npy',)
    obj.add_temp_dataset(f'{key_prefix}WINDOW', best_window, '.npy')
    
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
    data = obj.savgol_cr
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_subrange(data, exemplar_stack, bands_nm, ranges, mode=mode)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

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
    data = obj.savgol_cr
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_wta_msam_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

    return obj


def wta_min_map_MSAM_direct(arr, exemplars, bands,  mode='numpy'):
    """
    Compute a winner-takes-all MSAM class index and best-corr map.
    This direct variation returns an array directly, rather than adding to the
    model

    Parameters
    ----------
    array : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    
    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    data = np.array(arr[np.newaxis,...])
    bands_nm = np.array(bands)
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_wta_msam_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    return np.squeeze(index), np.squeeze(confidence)


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
    data = obj.savgol_cr
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_wta_sam_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

    return obj


def wta_min_map_SAM_direct(arr, exemplars, bands,  mode='numpy'):
    """
    Compute a winner-takes-all SAM class index and best-corr map.
    This direct variation returns an array directly, rather than adding to the
    model

    Parameters
    ----------
    array : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    
    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    data = np.array(arr[np.newaxis,...])
    bands_nm = np.array(bands)
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_wta_sam_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    return np.squeeze(index), np.squeeze(confidence)


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
    data = obj.savgol_cr
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_wta_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

    return obj


def wta_min_map_direct(arr, exemplars, bands,  mode='numpy'):
    """
    Compute a winner-takes-all Pearson class index and best-corr map.
    This direct variation returns an array directly, rather than adding to the
    model

    Parameters
    ----------
    array : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    
    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    data = np.array(arr[np.newaxis,...])
    bands_nm = np.array(bands)
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = remove_cont(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sa.mineral_map_wta_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    return np.squeeze(index), np.squeeze(confidence)


def clean_legends(obj, onto_path):
    """
    Function for creating new mineral mapping datasets with ontologically re-mapped
    legends. A default ontology is provided, but the path to it, or to user created mapping
    must be supplied
    """
    
    for key in obj.keys():
        if key.endswith('LEGEND'):
            leg_key = key
            base_key = key[:-6]
            ind_key = key[:-6] + "INDEX"
            index_array = obj[ind_key].data
            
            legend = obj[leg_key].data
                        
            new_index, new_legend, debug_map = rl.remap_index_with_ontology(
                index_array=index_array,
                legend = legend,
                ontology_path = onto_path,
                keep_unmatched_as_original = False,
                unknown_label = "Unclassified"
            )
            clean_key_prefix = base_key+"-clean-"
            obj.add_temp_dataset(f"{clean_key_prefix}INDEX", new_index.astype(np.int16),  ".npy")
            obj.add_temp_dataset(f"{clean_key_prefix}LEGEND", new_legend, ".json")
            obj.add_temp_dataset(f'{clean_key_prefix}MAPPING', debug_map, '.json')
    return obj


def match_spectra(spectra_x, spectra_y, bands_nm):
    """
    passthrough fuction for matching a spectrum to a band range
    """
    y_res = resample_spectrum(np.asarray(spectra_x, float), np.asarray(spectra_y, float), bands_nm)
    
    return y_res


def kmeans_caller(obj, clusters = 5, iters = 50):
    """
    Calls an implementation of k-means using user-defined cluster and 
    iteration values
    """
    H,W,B = obj.savgol.shape
    data = obj.savgol_cr
    mask = obj.mask.astype(bool)
    valid_mask = ~mask
    valid_mask &= np.isfinite(data).all(axis=2)
    valid_mask &= ~np.isnan(data).any(axis=2)
    # 2) flatten & extract valid pixels
    flat = data.reshape(-1, B)
    vm = valid_mask.ravel()
    idx = np.nonzero(vm)[0]
    X = flat[idx]
    #spectral demands 3d array
    X_3d = X.reshape(-1, 1, B)
    img, classes = sa.kmeans_spectral_wrapper(X_3d, clusters, iters)
    img = np.squeeze(img)  # (N_valid,)
    # 4) rebuild labels to (H, W)
    labels_full = np.full(flat.shape[0], -1, dtype=int)
    labels_full[idx] = img
    clustered_map = labels_full.reshape(H, W)

    obj.add_temp_dataset(f'kmeans-{clusters}-{iters}INDEX', clustered_map.astype(np.int16), '.npy')
    obj.add_temp_dataset(f'kmeans-{clusters}-{iters}CLUSTERS', classes, '.npy')
    return obj



def compute_pixel_counts(idx: np.ndarray, m: int) -> np.ndarray:
    """
    Count pixels per cluster ID using a H x W index map.
    Negative IDs are treated as background and ignored.
    """
    flat = np.asarray(idx).ravel()
    flat = flat[flat >= 0]
    if flat.size == 0:
        return np.zeros(m, dtype=int)
    counts = np.bincount(flat, minlength=m)
    return counts[:m]


def band_math_interface(obj, name, expr, cr = False):
    """
    Takes a processed object, a name and an expression and uses the band_maths
    submodule to parse and evaluate the expression on reflectance data. Optionally 
    evaluate the expression on continuum removed data.
    """
    if not cr:
        cube = obj.savgol
    else:
        cube = obj.savgol_cr
    
    out = bm.evaluate_expression(expr, cube, obj.bands)
    clean_key = re.sub(r'[\\/:*?"<>|_]', '-', name)
    obj.add_temp_dataset(clean_key, np.ma.masked_array(out, obj.mask), '.npz')
    return obj
    
    
    