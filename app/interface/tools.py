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

from ..config import config, VALID_CONVENTIONS, CONVENTION_DISPLAY, FEATURE_BOUNDS
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

def save_config():
    """
    Persists the users changed config values to dict for consistent config
    between sessions
    """
    try:
        config.save()
    except Exception as e:
        logger.warning(f"Failed to persist config values {e}", exc_info=True)

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
        
    if isinstance(obj, RawObject):
        if not hasattr(obj, "reflectance") or obj.reflectance is None:
            refl_start = time.perf_counter()
            obj.get_reflectance()
            logger.debug(f"get_reflectance took {time.perf_counter() - refl_start:.3f}s")
            
        if hasattr(obj, "temp_reflectance") and obj.temp_reflectance is not None:
            arr = obj.temp_reflectance
        else:
            arr = obj.reflectance

        obj.temp_reflectance = arr[y_min:y_max, x_min:x_max]
        
        return obj

    elif isinstance(obj, ProcessedObject):
        # union of base + temps
        keys = obj.keys()
        
        # Ensure mask is processed first for thumbnail generation
        ordered_keys = ['mask'] if 'mask' in keys else []
        ordered_keys.extend([k for k in keys if k != 'mask'])
        base_uncropped_shape = obj.savgol.shape[:2]
        total_crop_time = 0
        for key in ordered_keys:
            key_start = time.perf_counter()
            logger.info(f"cropping {obj.basename} {key} dataset")
            
            # choose source: temp first enforced by PO API
            src = obj.get_data(key)
            ext = obj[key].ext
            if isinstance(src, np.ndarray) and src.shape[:2] == base_uncropped_shape:
                array_copy_start = time.perf_counter()
                sliced = src[y_min:y_max, x_min:x_max, ...]
                cropped_copy = sliced.copy()
                obj.add_temp_dataset(key, cropped_copy, ext = ext)
                params = {"Spatial Crop" : "Manual",
                          "bounds": {
                        'y_min' : y_min,
                        'y_max' : y_max,
                        'x_min' : x_min,
                        'x_max' : x_max,
                        }}
                obj.update_lineage(key, key, params)
                    
        obj.regenerate_display()
        return obj

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")


def crop_auto(obj,mode='references'):
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
        cropped, slicer = sm.auto_crop(img, mode=mode)
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
        base = getattr(obj, "savgol", None)
        if not isinstance(base, np.ndarray) or base.ndim < 2 or 0 in base.shape:
                return obj
        # Use pre-computed display dataset if available, fall back to get_false_colour on savgol
        if obj.has('display'):
            img = obj.display  # already uint8 [0, 255], (H, W, 3)
        else:
            img = get_false_colour(base)
            img = np.asarray(img)
            if img.ndim < 2 or 0 in img.shape:
                return obj
            img = (img * 255).astype(np.uint8, copy=False)
        base_uncropped_shape = img.shape[:2]
        cropped, slicer = sm.auto_crop(img, mode=mode)
        if slicer is None:
            return obj
        try:
            test_ref = base[slicer]
        except Exception:
            return obj
        if not isinstance(test_ref, np.ndarray) or test_ref.ndim < 2 or 0 in test_ref.shape:
            return obj
        
        # slicer is valid → apply to all datasets
        keys = obj.keys()
        
        # Ensure mask is processed first for thumbnail generation
        ordered_keys = ['mask'] if 'mask' in keys else []
        ordered_keys.extend([k for k in keys if k != 'mask'])
        
        for key in ordered_keys:
            logger.info(f"auto-cropping {obj.basename} {key} dataset")
            
            src = obj.get_data(key)
            ext = obj[key].ext
            if isinstance(src, np.ndarray) and src.shape[:2] == base_uncropped_shape:
                try:
                    cropped = src[slicer]
                except Exception:
                    continue
                if 0 in cropped.shape:
                    continue

                cropped_copy = cropped.copy()
                obj.add_temp_dataset(key, cropped_copy, ext = ext)
                params = {"Spatial Crop" : "Auto",
                          "Mode" : mode,}
                y_slice, x_slice = slicer
                params["bounds"] = {
                "y_min": y_slice.start,
                "y_max": y_slice.stop,
                "x_min": x_slice.start,
                "x_max": x_slice.stop,
                }
                obj.update_lineage(key, key, params)
            
                
        obj.regenerate_display()
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

#==========Experimental auto-masking tools ====================================
def mask_clusters(obj):
    """
    Perform ephemeral k=2 clustering on the ProcessedObject for use in
    auto-masking. The cluster map is NOT stored on the object — it is
    returned directly to the caller for UI-layer coordination.

    Valid pixels are determined by the existing mask and data finiteness.
    Masked pixels receive label -1 in the returned array.

    Returns
    -------
    obj : ProcessedObject
        Unchanged — returned for call-site consistency.
    cluster_array : np.ndarray, shape (H, W), dtype int
        Cluster label map. Values are 0 or 1 for valid pixels, -1 for masked.
    """
    H, W, B = obj.savgol.shape
    data = obj.savgol_cr
    mask = obj.mask.astype(bool)

    valid_mask = ~mask
    valid_mask &= np.isfinite(data).all(axis=2)
    valid_mask &= ~np.isnan(data).any(axis=2)

    flat = data.reshape(-1, B)
    vm = valid_mask.ravel()
    idx = np.nonzero(vm)[0]
    X = flat[idx].reshape(-1, 1, B)

    img, _ = sa.kmeans_spectral_wrapper(X, 2, 50)
    img = np.squeeze(img)

    labels_full = np.full(flat.shape[0], -1, dtype=int)
    labels_full[idx] = img
    cluster_array = labels_full.reshape(H, W)

    return obj, cluster_array



def mask_by_cluster(obj, cluster_array, index):
    """
    Mask all pixels belonging to the given cluster index.
    ORs with the existing mask — does not clobber prior masking work.
    Mask convention: 0 = valid, 1 = masked.

    Parameters
    ----------
    obj : ProcessedObject
    cluster_array : np.ndarray, shape (H, W)
        Label map returned by mask_clusters().
    index : int
        Cluster label to mask (0 or 1).

    Returns
    -------
    obj : ProcessedObject
    """
    msk = np.array(obj.mask)
    msk[cluster_array == index] = 1
    obj.add_temp_dataset('mask', data = msk)
    params = {
    "masking type": "Mask by cluster",
    "cluster index": int(index),
    "comments": "Experimental procedure; cluster assignment is not preserved.",
}
    obj.update_lineage("mask", "mask", params )
    return obj

# ===========Proven masking tools ============================================

def mask_rect(obj, ymin, ymax, xmin, xmax, unmask = False):
    """
    Adds (or removes) a user selected rectangle to/from the mask.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    msk = np.array(obj.mask)
    msk[ymin:ymax, xmin:xmax] = 0 if unmask else 1
    obj.add_temp_dataset('mask', data = msk)
    params = {"masking type": "Rectangle", 
              'ymin' : ymin,
              'ymax' : ymax,
              'xmin' : xmin,
              'xmax' : xmax,
              'mask inside' : not unmask
              }
    obj.update_lineage("mask", "mask", params )
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
        params = {"masking type": "New mask by point",
                  'x': x,
                  'y': y
                  }
        obj.update_lineage("mask", ["mask", "savgol_cr"], params)
        return obj
    if mode == 'enhance':
        msk = np.array(obj.mask)
        pixel_vec = obj.savgol_cr[y, x, :]
        corr = sa.numpy_pearson(obj.savgol_cr, pixel_vec)
        msk[corr > 0.9] = 1
        obj.add_temp_dataset('mask', data = msk)
        params = {"masking type": "Enhance mask by point",
                          'x': x,
                          'y': y
                          }
        obj.update_lineage("mask", ["mask", "savgol_cr"], params)
        return obj
    if mode == 'line':
        msk = np.array(obj.mask)
        msk[:, x] = 1
        obj.add_temp_dataset('mask', data = msk)
        params = {"masking type": "Mask column by point",
                                  'x': x,
                                  'y': y
                                  }
        obj.update_lineage("mask", "mask", params)
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
    params = {"masking type": "Mask by polygon",
              "mode" : mode,
              "polygon" : f'{vertices_rc}'}
    obj.add_temp_dataset('mask', data = msk)
    obj.update_lineage("mask", "mask", params)
    return obj


def improve_mask(obj, mode="vertical"):
    """
    Heuristically thicken a mask column-wise using simple occupancy.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    logger.info(f"improve_mask called with mode={mode}")
    params = {"masking type": "CV heuristics"}
    if mode=="vertical":
        logger.info("Running improve_mask_from_graph")  
        msk = sm.improve_mask_from_graph(obj.mask) 
        params["mode"] = "vertical"
    else:
        logger.info("Running hough_line_connection")
        msk = sm.hough_line_connection(obj.mask)
        params["mode"] = "hough line connection"
    obj.add_temp_dataset('mask', data = msk)
    obj.update_lineage("mask", "mask", params)
    
    return obj


def despeckle_mask(obj):
    """
    Despeckle the mask using CV techniques
    """
    msk = sm.despeckle_mask(obj.mask)
    params = {"masking type": "CV heuristics",
              "mode" : "despeckle"}
    obj.add_temp_dataset('mask', data = msk)
    obj.update_lineage("mask", "mask", params)
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
    params = {"masking type": "Mask all"}
    obj.update_lineage("mask", "mask", params)
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
    params = {"masking type": "Logical inversion of mask"}
    obj.update_lineage("mask", "mask", params)
    return obj


def rim(obj):
    """
    Write a rim around the cropped cube. This prevents small angular segments
    on slanted boxes.
    """
    if obj.is_raw:
        return obj
    msk = np.array(obj.mask)
    rim_px = 5                        # rim width in pixels — adjustable here

    h, w = msk.shape[:2]

    # Guard: a rim wider than half the frame would mask everything.
    if h <= 2 * rim_px or w <= 2 * rim_px:
        logger.warning(f"rim: mask {h}x{w} too small for a {rim_px}px rim; skipping")
        return obj

    # 1 = masked/background. Force the outer border to background so no core
    # content reaches the image edge, clipping the thin angular slivers that
    # appear at the corners of a slanted/rotated box.
    msk[:rim_px, :]  = 1   # top
    msk[-rim_px:, :] = 1   # bottom
    msk[:, :rim_px]  = 1   # left
    msk[:, -rim_px:] = 1   # right

    obj.add_temp_dataset('mask', data = msk)
    params = {"masking type": "Mask rim of image",
              "Rim width" : rim_px}
    obj.update_lineage("mask", "mask", params)
    logger.info(f"rim: applied {rim_px}px background rim to mask")
    return obj

#============ Unwrapping tools ================================================

def calc_unwrap_stats(obj):
    """
    Compute connected components on the (eroded) inverse of a mask and sets the
    returned stats to a dataset for use in future unwrapping operations.
    Also creates a dataset image of the derived segments for user inspection
    """
    label_image, stats = sm.get_stats_from_mask(obj.mask)
    #label_image = label_image / np.max(label_image)
    obj.add_temp_dataset('stats', stats, '.npy')
    obj.add_temp_dataset('segments', label_image, '.npy')
    metadata = obj.metadata.copy()
    metadata['box_convention'] = config.box_convention
    obj.add_temp_dataset('metadata', metadata, ext='.json')
    
    logger.info(f"Box convention '{config.box_convention}' embedded in metadata for {obj.basename}")
    params = {"Operation" : "calculate stats for unwrapping",
              "Hard-coded defaults" : {"Erosion iterations": 2,
                                        "Erosion kernel": [3, 3],
                                        "Erosion anchor": [0, 0],
                                        "Connectivity": 8}}
    obj.update_lineage(["stats", "segments"], "mask", params)

    return obj


def add_depth_anchor(obj, x, y, depth):
    metadata = dict(obj.metadata)
    anchors = list(metadata.get('anchors') or [])
    anchors.append({'x': int(x), 'y': int(y), 'depth': float(depth)})
    metadata['anchors'] = anchors
    obj.add_temp_dataset('metadata', metadata, ext='.json')
    logger.info(f"Depth anchor added to {obj.basename}: x={x}, y={y}, depth={depth}m "
                f"({len(anchors)} total)")
    return obj

def unwrapped_output(obj):
    """
    Uses previously computed unwrap stats to produce a vertically concatenated
    core box spectral cube and mask. Calculates mask-aware per pixel depths
    using depth values held in the metadata. 
    This preview of the unwrapped result also produces an image map of the depth
    registration
    """
    convention = obj.metadata.get('box_convention', None)
    depth_start, depth_stop, anchors = obj.get_depth_params_in_m()
    dhole_reflect, dhole_depths, dmap = unwrap_from_stats(obj.mask, obj.savgol, obj.stats, obj.segments, 
                                                    convention=convention,
                                                    anchors = anchors,
                                                    depth_start = depth_start,
                                                    depth_stop = depth_stop,
                                                    return_map=True)
    dmask = dhole_reflect.mask[:,:,0]

    obj.add_temp_dataset('DholeMask', dmask, '.npy')
    obj.add_temp_dataset('DholeAverage', dhole_reflect.data, '.npy')
    obj.add_temp_dataset('DholeDepths', dhole_depths, '.npy')
    if dmap is not None:
        obj.add_temp_dataset('DepthMap', dmap, '.npz')
    
    params = {"Unwrapping" : "Image previews",
              "Anchors": anchors,
              "Box start depth" : depth_start,
              "Box end depth" : depth_stop,
              "Output depth units": "m",
              "Source depth units": obj.get_units(),
              "Box convention": convention or "rl_tb",
              "MIN_AREA" : config.min_seg_area,
              "MIN_WIDTH" : config.min_seg_width,
              "Hard-coded defaults": {
                        "Lane gap minimum": 10,
                        "Lane gap width proportion": 0.25,
                        "Lane gap fallback": 25,
                        "Depth interpolation": "linear",
                        "Segment padding": "centred",
                    }}
    output_keys = ['DholeAverage', 'DholeMask', 'DholeDepths']
    if dmap is not None:
        output_keys.append('DepthMap')
    obj.update_lineage(output_keys,
                       ["mask", "savgol", "stats", "segments"],
                       params
                        )
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
    if isinstance(key, dict):
        if len(key) != 1:
            logger.warning("Custom feature definition must contain exactly one feature")
            return obj
        feature_name, feature_bounds = next(iter(key.items()))
        feature_definition = {
            feature_name: list(feature_bounds)
        }
        feature_type = "custom"
    else:
        feature_name = key
        feature_definition = {
            feature_name: list(FEATURE_BOUNDS[key])
        }
        feature_type = "standard"
    
    inputs = ["savgol", "savgol_cr", "mask", "bands"]

    cache_available = (obj.has('feature-indices') and
                       obj.has('feature-heights')
                       )
    if cache_available:
        cached_arrays = (
            obj.get_data('feature-indices'),
            obj.get_data('feature-heights')
        )
        inputs.extend(["feature-indices", "feature-heights"])
        extraction_method = "cached feature detection"
    else:
        cached_arrays = None
        extraction_method = "direct feature detection"

    try:
        pos, dep, feat_mask = sa.Combined_MWL(obj.savgol, obj.savgol_cr, obj.mask, obj.bands, key, technique='POLY', cached_arrays=cached_arrays)
        pos_key = f'{feature_name}POS'
        dep_key = f'{feature_name}DEP'
        obj.add_temp_dataset(pos_key, np.ma.masked_array(pos, mask=feat_mask), '.npz')
        obj.add_temp_dataset(dep_key, np.ma.masked_array(dep, mask=feat_mask), '.npz')
        params = {
                    "Operation": "minimum wavelength feature extraction",
                    "Feature type": feature_type,
                    "Feature definition": feature_definition,
                    "Feature extraction method": extraction_method,
                    "Fitting technique": "POLY",
                    "Feature detection threshold": config.feature_detection_threshold,
                    "Width filtering": False,
                }
        
        if not cache_available:
            params["Hard-coded defaults"] = {
                "Peak detection tolerance": 11,
            }

        obj.update_lineage(
            [pos_key, dep_key],
            inputs,
            params,
        )

    except AssertionError as e:
        logger.warning(f"hylite error: {e}")
    except ValueError as e:
        logger.warning(f"Error calculating MWL: {e}")
    
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
    
    bands = np.asarray(obj.bands)

    # Store raw cache arrays
    obj.add_temp_dataset('feature-indices', feature_indices, ext='.npy')
    obj.add_temp_dataset('feature-heights', feature_heights, ext='.npy')
    obj.add_temp_dataset('feature-counts', np.ma.masked_array(feature_counts, mask=obj.mask), ext='.npz')

    params = {"Operation" : "Detected features caching",
              "Feature detection threshold" : config.feature_detection_threshold,
              "Maximum number of features" : max_feats,
              "Feature ordering": "descending absorption depth",
              "Peak detection": "scipy.signal.find_peaks",}
    obj.update_lineage(['feature-indices', 'feature-heights'],
                       "savgol_cr",
                       params)
    obj.update_lineage('feature-counts',
                       ["savgol_cr", "mask"],
                       params)

    # Store position and depth for top 3 features as displayable datasets
    three_keys = ['deepest', 'second-deepest', 'third-deepest']
    for k, label in enumerate(three_keys):
        idx_slice = feature_indices[:, :, k].astype(float)
        idx_slice[idx_slice < 0] = 0  # avoid invalid index into bands
        pos = bands[idx_slice.astype(int)]
        pos[feature_indices[:, :, k] < 0] = np.nan
        dep = feature_heights[:, :, k].copy()

        obj.add_temp_dataset(
            f'{label}-featurePOS',
            np.ma.masked_array(pos, mask=obj.mask),
            ext='.npz'
        )
        obj.add_temp_dataset(
            f'{label}-featureDEP',
            np.ma.masked_array(dep, mask=obj.mask),
            ext='.npz'
        )
        params_ind = {"Operation" : "Indexed cached features",
                      "Feature indexed" : label}
        obj.update_lineage([f'{label}-featurePOS',  f'{label}-featureDEP'],
                           ["feature-indices", "feature-heights", "bands", "mask"],
                           params_ind)
    return obj
    
    


def quick_corr(obj, x, y, key, ids = None):
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
    params = {"Operation" : "Single mineral correlation",
              "Correlation method" : "Pearson",
              "Library used" : config.library_path,
              "Exemplar spectra" : key,
              "Exemplar processing": "resampled to object bands and continuum removed"}
    obj.update_lineage(clean_key, ["savgol_cr", "mask", "bands"], params)
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
    params = {"Correlation" : "Multi-range",
                "Mode" : mode,
                "Library used" : config.library_path,
                "Collection name" : coll_name,
                "Collection IDs" : {
                            str(sample_id): str(label)
                            for sample_id, (label, _, _) in exemplars.items()
                        },
                "Exemplar processing": "resampled to object bands continuum removed independently within each window",
                "Continuum removal" : "gfit 0.2", 
                "Hardcoded parameters" : {"Windows": [(1350, 1500),
                                                    (1850, 2000), 
                                                    (2140, 2230), 
                                                    (2230, 2320), 
                                                    (2305, 2500)],
                                        "Pearson minimum score": 0.70,
                                        "MSAM minimum score": 0.70,
                                        "SAM maximum angle (degrees)": 8.0,}
                        }
    obj.update_lineage([f"{key_prefix}INDEX", f"{key_prefix}LEGEND", f'{key_prefix}CONF', f'{key_prefix}WINDOW'],
                       ["savgol", "bands"], 
                       params
                       )
    return obj


def wta_min_map_user_defined(obj, exemplars, coll_name, ranges, mode='pearson'):
    """
    Compute a winner-takes-all map on a user selected range.

    Parameters
    ----------
    obj : ProcessedObject   (needs .savgol (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    coll_name : str text name of the collection passed
    ranges : list[float(min), float(max)]
    mode : str (pearson, sam, msam)
    
    
    """
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMap-{ranges[0]}-{ranges[1]}-{mode}-{coll_name}"
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
    index, confidence = sa.mineral_map_subrange(data, exemplar_stack, bands_nm, ranges, mode=mode)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)
    params = {"Correlation" : "Custom band range - winner takes all",
                    "Mode" : mode,
                    "Library used" : config.library_path,
                    "Collection name" : coll_name,
                    "Collection IDs" : {
                                str(sample_id): str(label)
                                for sample_id, (label, _, _) in exemplars.items()
                            },
                    "Custom band range": [float(x) for x in ranges],
                    "Exemplar processing": "array and exemplars sliced to custom range before continuum removal",
                    "Continuum removal" : "gfit 0.2", 
                    "Hardcoded parameters" : {"Pearson minimum score": 0.70,
                                            "MSAM minimum score": 0.70,
                                            "SAM maximum angle (degrees)": 8.0,}
                            }
    obj.update_lineage([f"{key_prefix}INDEX", f"{key_prefix}LEGEND", f'{key_prefix}CONF'],
                           ["savgol", "bands"], 
                           params
                           )
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
    params = {"Correlation" : "Modified Spectral Angle Mapping - winner takes all",
                        "Library used" : config.library_path,
                        "Collection name" : coll_name,
                        "Collection IDs" : {
                                    str(sample_id): str(label)
                                    for sample_id, (label, _, _) in exemplars.items()
                                },
                        "Exemplar processing": "resampled to bands and continuum removed",
                        "Continuum removal" : "gfit 0.2", 
                        "Hardcoded parameters" : {"MSAM minimum score": 0.70,}
                                }
    obj.update_lineage([f"{key_prefix}INDEX", f"{key_prefix}LEGEND", f'{key_prefix}CONF'],
                               ["savgol_cr", "bands"], 
                               params
                               )
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
    params = {"Correlation" : "Spectral Angle Mapping - winner takes all",
                            "Library used" : config.library_path,
                            "Collection name" : coll_name,
                            "Collection IDs" : {
                                        str(sample_id): str(label)
                                        for sample_id, (label, _, _) in exemplars.items()
                                    },
                            "Exemplar processing": "resampled to bands and continuum removed",
                            "Continuum removal" : "gfit 0.2", 
                            "Hardcoded parameters" : {"SAM maximum angle (degrees)": 8.0,}
                                    }
    obj.update_lineage([f"{key_prefix}INDEX", f"{key_prefix}LEGEND", f'{key_prefix}CONF'],
                                   ["savgol_cr", "bands"], 
                                   params
                                   )
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
    params = {"Correlation" : "Pearson Correlation - winner takes all",
                                "Library used" : config.library_path,
                                "Collection name" : coll_name,
                                "Collection IDs" : {
                                            str(sample_id): str(label)
                                            for sample_id, (label, _, _) in exemplars.items()
                                        },
                                "Exemplar processing": "resampled to bands and continuum removed",
                                "Continuum removal" : "gfit 0.2", 
                                "Hardcoded parameters" : {"Pearson minimum score": 0.70}
                                        }
    obj.update_lineage([f"{key_prefix}INDEX", f"{key_prefix}LEGEND", f'{key_prefix}CONF'],
                                       ["savgol_cr", "bands"], 
                                       params
                                       )
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

def custom_false_colour(obj, bands: list):
    """
    Produce a custom 3-band false colour image, using the spectral library get_rgb call.
    obj must be a processed object and bands must be a list of integers
    """
    if len(bands) != 3:
        logger.warning(f"A list of three bands is required, {len(bands)} were provided")
        return obj
    if not all(isinstance(x, int) for x in bands):
        logger.warning(f"A list of three integers is required but wrong types were provided")
        return obj
    if not all(0 <= x < obj.savgol.shape[2] for x in bands):
        logger.warning(f"Band numbers must be between 0 and {obj.savgol.shape[2] - 1}")
    data = obj.savgol
    fc = get_false_colour(data, bands = bands)
    fc[obj.mask == 1] = (0,0,0)
    key = "-".join(f"{obj.bands[x]:.2f}" for x in bands)
    obj.add_temp_dataset(key, fc, '.npy')
    return obj