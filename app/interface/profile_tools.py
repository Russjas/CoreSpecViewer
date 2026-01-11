"""
High-level utility functions for spectral analysis on spectral profile data.
Used by UI pages to manipulate HoleObject base datasets.
"""
from pathlib import Path
import re

from matplotlib.path import Path as mpl_path
import numpy as np

from .. import config
from ..models import ProcessedObject, RawObject, HoleObject
from ..spectral_ops import spectral_functions as sf
from ..spectral_ops import band_maths as bm

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
        legend.append({"index" : i, "label" : f"Class {i}"})
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



# Existing tools that need to be refactored to accept profile dataset from HO


#=============Mineral mapping from libraries===================================
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
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_subrange(data, exemplar_stack, bands_nm, ranges, mode=mode)
    legend = [{"index": i, "label": labels[i]} for i in range(len(labels))]

    obj.add_temp_dataset(f"{key_prefix}INDEX", index.astype(np.int16),  ".npy")
    obj.add_temp_dataset(f"{key_prefix}LEGEND", legend, ".json")
    obj.add_temp_dataset(f'{key_prefix}CONF', confidence, '.npy',)

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
        y_res = sf.resample_spectrum(np.asarray(x_nm, float), np.asarray(y, float), bands_nm)
        y_res = sf.cr(y_res[np.newaxis, :])[0]
        labels.append(str(label))
        bank.append(y_res.astype(np.float32))
    if not bank:
        raise ValueError("No exemplars provided.")
    exemplar_stack = np.vstack(bank)
    index, confidence = sf.mineral_map_subrange(data, exemplar_stack, bands_nm, ranges, mode=mode)
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






















