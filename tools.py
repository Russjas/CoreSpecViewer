# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:33:11 2025

@author: russj
"""
from pathlib import Path
import numpy as np
from objects import ProcessedObject, RawObject
import spectral_functions as sf
from PIL import Image

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
    print('OBJECT TYPE', type(obj))
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

     
    



    