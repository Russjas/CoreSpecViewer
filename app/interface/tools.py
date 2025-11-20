"""
High-level utility functions for cropping, masking, unwrapping, and feature extraction.
Used by UI pages to manipulate RawObject and ProcessedObject datasets.
"""
from pathlib import Path

from matplotlib.path import Path as mpl_path
import numpy as np

from .. import config
from ..models import ProcessedObject, RawObject
from ..spectral_ops import spectral_functions as sf

#======Getting and setting app configs ========================================


def get_config():
    """
    Loads the config dictionary - a single mutable dictionary of config
    patterns used accross the app
    """
    return sf.con_dict

def modify_config(key, value):
    """
    Sets user selected values in the config dictionary - a single mutable 
    dictionary of config patterns used accross the app
    """
    config.set_value(key, value)

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
            obj.get_reflectance()
        if hasattr(obj, "temp_reflectance") and obj.temp_reflectance is not None:
            arr = obj.temp_reflectance

        else:
            arr = obj.reflectance

        obj.temp_reflectance = arr[y_min:y_max, x_min:x_max]
        return obj

    elif isinstance(obj, ProcessedObject):
        # union of base + temps
        keys = set(obj.datasets.keys()) | set(obj.temp_datasets.keys())

        for key in keys:
            # choose source: prefer temp if present
            if obj.has_temp(key):
                src = obj.temp_datasets[key].data
            else:
                ds = obj.datasets.get(key)
                src = getattr(ds, "data", None) if ds else None

            if isinstance(src, np.ndarray) and src.ndim > 1:
                cropped = src[y_min:y_max, x_min:x_max, ...]
                cropped_copy = np.array(cropped)
                if obj.has_temp(key):
                    # keep the same wrapper; just update data
                    obj.temp_datasets[key].data = cropped_copy
                else:
                    # first temp for this key
                    obj.add_temp_dataset(key, cropped_copy)
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
        img = sf.get_false_colour(arr)
        img = (img*255).astype(np.uint8)
        cropped, slicer = sf.detect_slice_rectangles_robust(img)
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
        # union of base + temps
        keys = set(obj.datasets.keys()) | set(obj.temp_datasets.keys())
        base = getattr(obj, "savgol", None)
        if not isinstance(base, np.ndarray) or base.ndim < 2 or 0 in base.shape:
            return obj
        img = sf.get_false_colour(base)
        img = np.asarray(img)
        if img.ndim < 2 or 0 in img.shape:
            return obj
        img = (img * 255).astype(np.uint8, copy=False)

        cropped, slicer = sf.detect_slice_rectangles_robust(img)
        if slicer is None:
            return obj
        try:
            test_ref = base[slicer]
        except Exception:
            return obj
        if not isinstance(test_ref, np.ndarray) or test_ref.ndim < 2 or 0 in test_ref.shape:
            return obj
        # slicer is valid & non-empty for the reference → now apply per dataset
        for key in keys:
            if obj.has_temp(key):
                src = obj.temp_datasets[key].data
            else:
                ds = obj.datasets.get(key)
                src = getattr(ds, "data", None) if ds else None

            if not (isinstance(src, np.ndarray) and src.ndim > 1):
                continue
            try:
                cropped = src[slicer]
            except Exception:
                continue
            if 0 in cropped.shape:
                continue

            cropped_copy = np.array(cropped)  # materialise

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
        corr = sf.numpy_pearson(obj.savgol_cr, pixel_vec)
        msk[corr > 0.9] = 1
        obj.add_temp_dataset('mask', data = msk)
        return obj
    if mode == 'enhance':
        msk = np.array(obj.mask)
        pixel_vec = obj.savgol_cr[y, x, :]
        corr = sf.numpy_pearson(obj.savgol_cr, pixel_vec)
        msk[corr > 0.9] = 1
        obj.add_temp_dataset('mask', data = msk)
        return obj
    if mode == 'line':
        msk = np.array(obj.mask)
        msk[:, x] = 1
        obj.add_temp_dataset('mask', data = msk)
        return obj


def mask_polygon(obj, vertices_rc):
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

    # outside = ~inside  -> set to 1
    msk = np.array(obj.mask)
    msk[~inside] = 1
    obj.add_temp_dataset('mask', data = msk)
    return obj


def improve_mask(obj):
    """
    Heuristically thicken a mask column-wise using simple occupancy.
    Mask values follow the convention 0 = valid, 1 = masked.
    """
    msk = sf.improve_mask_from_graph(obj.mask)
    obj.add_temp_dataset('mask', data = msk)
    return obj

#============ Unwrapping tools ================================================

def calc_unwrap_stats(obj):
    """
    Compute connected components on the (eroded) inverse of a mask and sets the
    returned stats to a dataset for use in future unwrapping operations.
    Also creates a dataset image of the derived segments for user inspection
    """
    label_image, stats = sf.get_stats_from_mask(obj.mask)
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
    dhole_reflect = sf.unwrap_from_stats(obj.mask, obj.savgol, obj.stats)
    dhole_depths = np.linspace(float(obj.metadata['core depth start']), float(obj.metadata['core depth stop']),
                                    dhole_reflect.shape[0])

    obj.add_temp_dataset('DholeAverage', dhole_reflect.data, '.npy')
    obj.add_temp_dataset('DholeMask', dhole_reflect.mask, '.npy')
    obj.add_temp_dataset('DholeDepths', dhole_depths, '.npy')

    return obj
#==========pass through helpers===============================================
def get_cr(spectra):
    return sf.cr(spectra)

#========= Reflectance interpretation tools ===================================

def run_feature_extraction(obj, key):
    """
    Estimate minimum wavelength (MWL) position and corresponding absorption depth
    for a specified short-wave infrared absorption feature using multiple
    possible fitting techniques.
    """
    pos, dep, feat_mask = sf.Combined_MWL(obj.savgol, obj.savgol_cr, obj.mask, obj.bands, key, technique = 'QUAD')
    obj.add_temp_dataset(f'{key}POS', np.ma.masked_array(pos, mask = feat_mask), '.npz')
    obj.add_temp_dataset(f'{key}DEP', np.ma.masked_array(dep, mask = feat_mask), '.npz')
    return obj

#TODO: currently these are held entirely in the scope of lib window and not persistes
# think about adding as datasets -> need to consider the display window logic
def quick_corr(obj, x, y):
    """
    Runs a pearson correlation of a user selected spectum against the objects
    continuum removed dataset.
    Currently returns masked array directly rather than adding to obj.temp_datasets
    """
    if obj.is_raw:
        return None
    res_y = sf.resample_spectrum(x, y, obj.bands)
    return np.ma.masked_array(sf.numpy_pearson(obj.savgol_cr, sf.cr(res_y)), mask = obj.mask)


#TODO: Old method, maybe delete. Indexing is currently handled directly by display
#and thumbnail logic. 
def _colorize_indexed(class_idx: np.ndarray, labels: list[str]):
    """
    Specific helper for index maps.
    Return RGB image (uint8) and a color table aligned to labels.
    """
    import matplotlib
    tab = matplotlib.colormaps['tab20']
    K = max(int(class_idx.max()) + 1, len(labels))
    colors = np.array([tab(i % 20)[:3] for i in range(K)], dtype=np.float32)
    colors_rgb = (colors * 255).astype(np.uint8)  # (K,3)
    rgb = colors_rgb[class_idx]                   # (H,W,3) uint8
    return rgb, colors_rgb



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
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_wta_msam_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

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
    data = obj.savgol_cr
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_wta_sam_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

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
    data = obj.savgol_cr
    bands_nm = obj.bands
    labels, bank = [], []
    for _, (label, x_nm, y) in exemplars.items():
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_wta_strict(data, exemplar_stack)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

    return obj


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
    img, classes = sf.kmeans_spectral_wrapper(X_3d, clusters, iters)
    img = np.squeeze(img)  # (N_valid,)
    # 4) rebuild labels to (H, W)
    labels_full = np.full(flat.shape[0], -1, dtype=int)
    labels_full[idx] = img
    clustered_map = labels_full.reshape(H, W)

    obj.add_temp_dataset(f'kmeans-{clusters}-{iters}INDEX', clustered_map.astype(np.int16), '.npy')
    obj.add_temp_dataset(f'kmeans-{clusters}-{iters}CLUSTERS', classes, '.npy')
    return obj





