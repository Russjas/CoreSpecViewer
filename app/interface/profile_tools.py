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
from ..spectral_ops import remap_legend as rl
from ..spectral_ops import band_maths as bm


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