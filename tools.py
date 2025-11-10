# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:33:11 2025

@author: russj
"""
from pathlib import Path
import numpy as np
from objects import ProcessedObject, RawObject, Dataset
import spectral_functions as sf
from PIL import Image
from matplotlib.path import Path as mpl_path

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
                if obj.has_temp(key):
                    # keep the same wrapper; just update data
                    obj.temp_datasets[key].data = cropped
                else:
                    # first temp for this key
                    obj.add_temp_dataset(key, cropped)
        return obj

    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

def crop_auto(obj):
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
        obj.temp_reflectance = arr[slicer]
        return obj
    elif isinstance(obj, ProcessedObject):
        # union of base + temps
        keys = set(obj.datasets.keys()) | set(obj.temp_datasets.keys())
        img = sf.get_false_colour(obj.savgol)
        img = (img*255).astype(np.uint8)
        cropped, slicer = sf.detect_slice_rectangles_robust(img)
        for key in keys:
            # choose source: prefer temp if present
            if obj.has_temp(key):
                src = obj.temp_datasets[key].data
            else:
                ds = obj.datasets.get(key)
                src = getattr(ds, "data", None) if ds else None

            if isinstance(src, np.ndarray) and src.ndim > 1:
                cropped = src[slicer]
                if obj.has_temp(key):
                    # keep the same wrapper; just update data
                    obj.temp_datasets[key].data = cropped
                else:
                    # first temp for this key
                    obj.add_temp_dataset(key, cropped)
        return obj




def reset(obj):
    if obj.is_raw:
        obj.temp_reflectance = None
    else:
        obj.clear_temps()
    return obj

# Masking tools

def mask_rect(obj, ymin, ymax, xmin, xmax):
    if not obj.has_temp('mask'):
        obj.add_temp_dataset('mask')
    obj.mask[ymin:ymax, xmin:xmax] = 1
    return obj

def mask_point(obj, mode, y, x):
    if mode == 'new':
        obj.add_temp_dataset('mask', data = np.zeros((obj.savgol.shape[:2])))
        pixel_vec = obj.savgol_cr[y, x, :]
        corr = sf.numpy_pearson(obj.savgol_cr, pixel_vec)
        obj.mask[corr > 0.9] = 1
        return obj
    if mode == 'enhance':
        if not obj.has_temp('mask'):
            obj.add_temp_dataset('mask')
        pixel_vec = obj.savgol_cr[y, x, :]
        corr = sf.numpy_pearson(obj.savgol_cr, pixel_vec)
        obj.mask[corr > 0.9] = 1
        return obj
    if mode == 'line':
        if not obj.has_temp('mask'):
            obj.add_temp_dataset('mask')
        obj.mask[:, x] = 1
        return obj

def mask_polygon(obj, vertices_rc):
    """
    Given polygon vertices in (row, col) image indices, set outside to 1 (masked).
    Creates/updates a temp 'mask' dataset.

    - If no mask exists, starts from zeros.
    - Keeps interior as-is (commonly 0), sets outside to 1.
    """
    if obj.is_raw:
        H, W = obj.get_display_reflectance().shape[:2]
    else:
        H, W = obj.savgol.shape[:2]

    # ensure temp mask exists (0=keep, 1=masked)
    if not obj.has_temp('mask'):
        z = np.zeros((H, W), dtype=np.uint8)
        obj.add_temp_dataset('mask', data=z)

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
    m = obj.mask if obj.has_temp('mask') else obj.datasets['mask'].data
    m[~inside] = 1

    return obj


def improve_mask(obj):
    if not obj.has_temp('mask'):
        obj.add_temp_dataset('mask')
    obj.temp_datasets['mask'].data = sf.improve_mask_from_graph(obj.mask)
    return obj
    
def calc_unwrap_stats(obj):
    label_image, stats = sf.get_stats_from_mask(obj.mask)
    label_image = label_image / np.max(label_image)
    obj.add_dataset('stats', stats, '.npy')
    obj.add_dataset('segments', label_image, '.npy')
    
    return obj

def unwrapped_output(obj):
    seg_image = sf.seg_from_stats(obj.segments, obj.stats)
    dhole_reflect = sf.unwrap_from_stats(obj.mask, obj.savgol,obj.stats)
    dhole_depths = np.linspace(float(obj.metadata['core depth start']), float(obj.metadata['core depth stop']),  
                                    dhole_reflect.shape[0])
    
    obj.add_dataset('DholeAverage', dhole_reflect.data, '.npy')
    obj.add_dataset('DholeMask', dhole_reflect.mask, '.npy')
    obj.add_dataset('DholeDepths', dhole_depths, '.npy')


    return obj

def run_feature_extraction(obj, key): 
    pos, dep, feat_mask = sf.Combined_MWL(obj.savgol, obj.savgol_cr, obj.mask, obj.bands, key)
    obj.add_dataset(f'{key}POS', np.ma.masked_array(pos, mask = feat_mask), '.npz')
    obj.add_dataset(f'{key}DEP', np.ma.masked_array(dep, mask = feat_mask), '.npz')
    return obj
    
def quick_corr(obj, x, y):
    res_y = sf.resample_spectrum(x, y, obj.bands)
    return np.ma.masked_array(sf.numpy_pearson(obj.savgol_cr, sf.cr(res_y)), mask = obj.mask)

# --- Mineral map (winner-takes-all Pearson) -------------------------------
def _colorize_indexed(class_idx: np.ndarray, labels: list[str]):
    """Return RGB image (uint8) and a color table aligned to labels."""
    import matplotlib
    tab = matplotlib.colormaps['tab20']
    K = max(int(class_idx.max()) + 1, len(labels))
    colors = np.array([tab(i % 20)[:3] for i in range(K)], dtype=np.float32)
    colors_rgb = (colors * 255).astype(np.uint8)  # (K,3)
    rgb = colors_rgb[class_idx]                   # (H,W,3) uint8
    return rgb, colors_rgb



def wta_min_map(obj, exemplars, coll_name, mode='numpy'):
    """
    Compute a winner-takes-all Pearson class index and best-corr map.

    Parameters
    ----------
    obj : ProcessedObject   (needs .savgol_cr (H,W,B) and .bands (B,))
    exemplars : dict[int, (label:str, x_nm:1D, y:1D)]
        Usually from LibraryPage.get_collection_exemplars().
    use_cr : bool
        If True, continuum-removes each exemplar to match obj.savgol_cr.

    Returns
    -------
    class_idx : (H,W) int32
    best_corr : (H,W) float32
    labels    : list[str]
    """
    coll_name = coll_name.replace('_', '')
    key_prefix = f"MinMap{coll_name}"
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
    if mode == 'numpy':
        index, confidence = sf.numpy_pearson_stackexemplar_threshed(data, exemplar_stack)
    else:
        index, confidence = sf.mineral_map_wta_strict(data, exemplar_stack)
    
    def _stage(key: str, ext: str, data):
        ds = Dataset(
            base=obj.basename,
            key=key,
            path=str(obj.root_dir) + '/' +  f"{str(obj.basename)}_{key}{ext}",
            suffix=key,
            ext=ext,
            data=data
        )
        obj.temp_datasets[key] = ds
        return key
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]
    
    idx_key   = _stage(f"{key_prefix}{mode}INDEX",  ".npy",  index.astype(np.int16))
    legend_key = _stage(f"{key_prefix}{mode}LEGEND",  ".json",  legend)
    confidence_key = _stage(f'{key_prefix}{mode}CONF', '.npy', confidence)
    
    return obj



   



    