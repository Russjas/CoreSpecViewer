# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:46:18 2025

@author: russj
"""
import spectral.io.envi as envi
import spectral as sp
import numpy as np
import scipy as sc

from numba import jit
from PIL import Image
from datetime import datetime, date
import xml.etree.ElementTree as ET 
import json

import csv
import math
import os, glob
import pickle
import pathlib
from gfit.util import remove_hull
import hylite
from hylite.analyse import minimum_wavelength
import cv2



Lumo_swir_slice = slice(13, 262, None)
Lumo_mwir_slice = slice(5, 142, None)
Lumo_rgb_slice = slice(0,-1,None)
stellenbosch_slice = slice(5,-5,None)


def read_envi_header(file):
    return envi.read_envi_header(file)



def parse_lumo_metadata(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    pairs = []
    # Grab only <key> elements that actually have the attribute
    for key in root.findall('.//key'):
        fld = key.attrib.get('field')
        if fld is not None:
            pairs.append((fld, (key.text or '').strip()))

    # Also flatten simple non-<key> blocks (e.g., <corrected_metadata>)
    flat = {}
    for block in root:
        if block.tag in ('header', 'userdefined', 'core'):
            continue  # these were captured via <key field="...">
        for child in block:
            if len(child) == 0:  # leaf node
                flat[child.tag] = (child.text or '').strip()

    out = dict(pairs)
    out.update(flat)
    
    return out

def get_false_colour(array, bands=None):
    '''
    accepts a numpy array and returns 3 band image

    '''
    return sp.get_rgb(array, bands=bands)



def derive_display_bands(b):
    '''
    derives the first last and middle bands to display, if default bands are not 
    provided, or are otherwise problematic.

    '''
    return [0, b // 2, b - 1]


def reflect_correct(data, white, dark):
    """
    Convert raw radiance-like data to reflectance (%) using a white/dark reference.

    Parameters
    ----------
    data : ndarray, shape (H, W, B)
        Measured hyperspectral cube.
    white : ndarray, shape (H, W, B) or broadcastable to (H, W, B)
        White reference image/cube.
    dark : ndarray, shape (H, W, B) or broadcastable to (H, W, B)
        Dark reference image/cube.

    Returns
    -------
    ndarray
        Reflectance cube in percent, same shape as `data`. Values < 0 are clamped to 0.

    Notes
    -----
    The formula is `(data - mean(dark)) / (mean(white) - mean(dark)) * 100`.  
    """

    M, N, B = data.shape
    wmax = np.mean(white, axis=0)
    dmin = np.mean(dark, axis=0)
    result = np.divide(np.subtract(data, dmin), np.subtract(wmax, dmin))
    result[result < 0] = 0
    return result * 100

def bands_from_snr(white, dark, wavelengths=None, snr_thresh=20.0, min_run=20):
    """
    Estimate usable band range(s) by SNR and return the longest contiguous run.

    Parameters
    ----------
    white : ndarray
        White reference cube/stack; last axis is bands.
    dark : ndarray
        Dark reference cube/stack; last axis is bands.
    wavelengths : ndarray or list, optional
        (Unused in this version) Wavelengths corresponding to bands (nm).
    snr_thresh : float, optional
        Minimum SNR to consider a band usable; default is 20.0.
    min_run : int, optional
        Minimum contiguous run length to consider; used as a soft threshold.

    Returns
    -------
    slice
        Slice selecting the longest contiguous band run above threshold.
    ndarray
        1D array of SNR per band.

    Raises
    ------
    ValueError
        If the band counts of `white` and `dark` (last axis) do not match.

    Notes
    -----
    SNR is computed as `(mean(white) - mean(dark)) / std(dark)` with spatial
    averaging over all non-band axes. 
    """

    
    white = np.asarray(white)
    dark  = np.asarray(dark)
    
    if white.shape[-1] != dark.shape[-1]:
        raise ValueError(f"Band mismatch: white B={white.shape[-1]}, dark B={dark.shape[-1]}")

    # reduce over all spatial dims, keep only band axis
    reduce_axes_w = tuple(range(white.ndim - 1))
    reduce_axes_d = tuple(range(dark.ndim  - 1))
    w_bar = np.nanmean(white, axis=reduce_axes_w)  # shape (B,)
    d_bar = np.nanmean(dark,  axis=reduce_axes_d)  # shape (B,)
    d_std = np.nanstd(dark,   axis=reduce_axes_d) + 1e-9

    snr = (w_bar - d_bar) / d_std                    # shape (B,)
    good = snr >= snr_thresh

    

    # largest contiguous True run
    starts, ends, in_run = [], [], False
    for i, ok in enumerate(good.tolist() + [False]):
        if ok and not in_run: s = i; in_run = True
        elif not ok and in_run: starts.append(s); ends.append(i); in_run = False

    if not starts:
        B = good.size
        return slice(int(0.1*B), int(0.9*B)), snr

    lengths = np.array(ends) - np.array(starts)
    idx = int(np.argmax(lengths))
    if lengths[idx] < min_run:
        # optional: relax or just return the longest anyway
        pass
    
    thresh_test = np.where(snr > snr_thresh)
    return slice(thresh_test[0][0]-1, thresh_test[0][-1]+1), snr

def find_snr_and_reflect(header_path, white_path, dark_path, QAQC=False): # TODO: When the separate QAQC GUIS get integrated, they can call this func with QAQC=True to check SNR
    """
    Load raw data and references, pick a band window from SNR, and compute reflectance.

    Parameters
    ----------
    header_path : str or path-like
        Path to the main ENVI header (.hdr).
    white_path : str or path-like
        Path to the white reference ENVI header.
    dark_path : str or path-like
        Path to the dark reference ENVI header.
    QAQC : Boolean. 
        If true will calculate bands above threshold, if False will use fixed slice
        for technique

    Returns
    -------
    tuple
        `(data_reflect, bands, header)` where
        - data_reflect : ndarray, reflectance (%) cube cropped to usable bands
        - bands : ndarray, wavelengths for the selected band slice
        - header : dict, parsed header from `parse_header`.

    Notes
    -----
    Uses `bands_from_snr` to determine a contiguous band slice prior to
    reflectance correction.
    """

    box = envi.open(header_path)
    white_ref = envi.open(white_path)
    dark_ref  = envi.open(dark_path)
     
    header = envi.read_envi_header(header_path)
    
    bands = np.array([float(x) for x in header['wavelength']])
    data = np.array(box.load())
    white_ref = np.array(white_ref.load())
    dark_ref = np.array(dark_ref.load())
    if QAQC:
        band_slice, snr = bands_from_snr(white_ref, dark_ref, snr_thresh=20.0)
    else:
        print(header['sensor type'])
        if 'SWIR' in header['sensor type']:
            band_slice = Lumo_swir_slice
            snr = None
        elif 'RGB' in header['sensor type']:
            band_slice = Lumo_rgb_slice
            snr = None
        elif 'Specim Hyperspectral Sensor' in header['sensor type']:
            band_slice = stellenbosch_slice
            snr = None
        else:
            band_slice = Lumo_mwir_slice
            snr = None
        
    data  = data[:, :, band_slice]
    white = white_ref[:, :, band_slice]
    dark  = dark_ref[:, :, band_slice]
    bands = bands[band_slice]
    
    data_reflect = reflect_correct(data, white, dark)
    
    return data_reflect, bands, snr

# Actual processing funcs========================================================

def process(cube):
    
    savgol = sc.signal.savgol_filter(cube, 10, 2)
    savgol_cr = remove_hull(savgol)
    mask = np.zeros((cube.shape[0], cube.shape[1]))
    return savgol, savgol_cr, mask
        
def improve_mask_from_graph(mask):
    """
    Heuristically thicken a mask column-wise using simple occupancy.

    Parameters
    ----------
    mask : ndarray of {0,1}
        Binary mask image with 1s indicating core.

    Returns
    -------
    ndarray
        A copy of `mask` where columns with sufficient occupancy
        (sum above ~H/3) are set to 1 across all rows.  
    """

    line = np.sum(mask, axis = 0)
    new_mask=mask.copy()
    for i in range(line.shape[0]):
        
        if line[i]>int(mask.shape[0]/3):
            new_mask[:,i] = 1
    return new_mask


def get_stats_from_mask(mask, proportion=16, iters=2):
    """
    Compute connected components on the (eroded) inverse of a mask.

    Parameters
    ----------
    mask : ndarray of {0,1}
        Binary mask with 1 = core region.
    proportion : int, optional
        (Unused here) placeholder for future scaling.
    iters : int, optional
        Erosion iterations applied to the inverse mask before labeling.

    Returns
    -------
    labels : ndarray
        Labeled image (int) of connected components.
    stats : ndarray, shape (N, 5)
        Component stats from OpenCV: (x, y, width, height, area).
    """

    inv_mask = 1-mask
    kernel = np.ones((3,3),np.uint8)
    erod_im = cv2.erode(inv_mask, kernel, anchor=(0, 0), iterations=iters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erod_im.astype(np.uint8), connectivity=8)
    return labels, stats

@jit(nopython=True)
def numpy_pearson(data, exemplar):
    """
    Compute per-pixel Pearson correlation between each spectrum and an exemplar.

    Parameters
    ----------
    data : ndarray, shape (H, W, B)
        Hyperspectral cube (float).
    exemplar : ndarray, shape (B,)
        1D band vector to correlate with.

    Returns
    -------
    ndarray, shape (H, W)
        Pearson correlation coefficient for each pixel; NaNs are set to 0.
    """

    m, n, b = data.shape
    coeffs = np.zeros((m,n))
    for i in range(m):
        
        for j in range(n):  
            
            x = np.corrcoef(data[i,j], exemplar)[1,0]
            y = np.isnan(x)
            if y == False:
                coeffs[i,j] = x
            else:
                coeffs[i,j] = 0
    return coeffs

def seg_from_stats(image, stats, MIN_AREA=300, MIN_WIDTH=10):
    """
    Extract bounding boxes from component stats, pad to a common width,
    and stack them vertically (right-to-left, then top-to-bottom ordering).

    Parameters
    ----------
    image : ndarray
        Input 2D or 3D image to segment.
    stats : ndarray, shape (N, 5)
        (x, y, width, height, area) rows from `cv2.connectedComponentsWithStats`.
    MIN_AREA : int, optional
        Minimum area to keep.
    MIN_WIDTH : int, optional
        Minimum width to keep.

    Returns
    -------
    ndarray
        Vertically concatenated segments (plain array; padding filled with zeros).

    Notes
    -----
    Segments are sorted into columns using an x-based binning tolerance,
    then into rows by ascending y. 
    """

  
    segments = []
    
    for i in range(1, stats.shape[0]): # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or w < MIN_WIDTH:
            continue # Skip small regions
        else:
            segment = image[y:y+h, x:x+w]
                        # Store top-left x, y for sorting
            segments.append(((x, y), segment))
           
    # Sort segments: right to left (x descending), top to bottom (y ascending)
    tolerance = 15
    segments_sorted = sorted(segments, key=lambda s: (round(-s[0][0]/tolerance), s[0][1]))
    
    
    
    # Determine max width
    max_width = max(s[1].shape[1] for s in segments_sorted)
    # Pad segments to same width
    padded_segments = []
    
    for _, seg in segments_sorted:
        h, w = seg.shape[:2]
        pad_total = max_width - w
    
        if pad_total > 0:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            if seg.ndim == 2:
                pad_shape = ((0, 0), (pad_left, pad_right))
            else:
                pad_shape = ((0, 0), (pad_left, pad_right), (0, 0))
    
            seg_padded = np.pad(seg, pad_shape, mode='constant', constant_values=0)
        else:
            seg_padded = seg

        padded_segments.append(seg_padded)
    
    
    # Stack vertically
    concatenated = np.vstack(padded_segments)
    
    return concatenated


def unwrap_from_stats(mask, image, stats, MIN_AREA=300, MIN_WIDTH=10):
    """
    Unwrap core segments into a vertically stacked, width-normalized masked array.

    Parameters
    ----------
    mask : ndarray of {0,1} or bool
        Binary mask (same H×W as `image`), 1/True = core pixels.
    image : ndarray
        Input 2D or 3D image/cube.
    stats : ndarray, shape (N, 5)
        (x, y, width, height, area) rows from `cv2.connectedComponentsWithStats`.
    MIN_AREA : int, optional
        Minimum area to keep.
    MIN_WIDTH : int, optional
        Minimum width to keep.

    Returns
    -------
    np.ma.MaskedArray
        Vertically concatenated masked array of segments; padded regions are masked
        and original non-core pixels remain masked.

    Notes
    -----
    - Sorting is right-to-left (columns) then top-to-bottom (rows).
    - Padding is symmetric to match the maximum width across segments, applied
      to both data and mask before stacking. 
    """

    full_mask = np.zeros_like(image)
    full_mask[mask==1] = 1
    segments = []
    
    for i in range(1, stats.shape[0]): # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or w < MIN_WIDTH:
            continue # Skip small regions
        else:
            segment = np.ma.masked_array(image, mask = full_mask)[y:y+h, x:x+w]
            
                        # Store top-left x, y for sorting
            segments.append(((x, y), segment))
           
    # Sort segments: right to left (x descending), top to bottom (y ascending)
    tolerance = 10
    segments_sorted = sorted(segments, key=lambda s: (round(-s[0][0]/tolerance), s[0][1]))
    
    
    
    # Determine max width
    max_width = max(s[1].shape[1] for s in segments_sorted)
    # Pad segments to same width
    padded_segments = []
    
    for _, seg in segments_sorted:
        h, w = seg.shape[:2]
        pad_total = max_width - w
    
        if pad_total > 0:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            if seg.ndim == 2:
                pad_shape = ((0, 0), (pad_left, pad_right))
            else:
                pad_shape = ((0, 0), (pad_left, pad_right), (0, 0))
    
            seg_pad = np.pad(seg.data, pad_shape, mode='constant', constant_values=0)
            seg_mask_padded = np.pad(seg.mask, pad_shape, mode='constant', constant_values=1)
            seg_padded = np.ma.masked_array(seg_pad, mask = seg_mask_padded)
        else:
            
            seg_padded = seg
        
        padded_segments.append(seg_padded)
    
    padded_seg_data = [x.data for x in padded_segments]
    padded_seg_mask = [x.mask for x in padded_segments]
    # Stack vertically
    concatenated_data = np.vstack(padded_seg_data)
    concatenated_mask = np.vstack(padded_seg_mask)
    concatenated= np.ma.masked_array(concatenated_data, mask = concatenated_mask)
    
    return concatenated

def est_peaks_cube_scipy_thresh(data, bands, wavrange=(2300, 2340), thresh = 0.3):
    w, l, b = data.shape
    arr = np.zeros((w,l))
    for i in range(w):
        for j in range(l):
            peak_indices, peak_dict = sc.signal.find_peaks(1-data[i,j], height=(None, None))
            #print(peak_dict.keys())
            peak_heights = peak_dict['peak_heights']
            #print(type(peak_heights), type(peak_heights)[0])
            x = [bands[peak_indices[i]] for i in range(len(peak_indices)) if peak_heights[i] >thresh ]
            #print(len(x))
            for k in x:        
                if k > wavrange[0] and k < wavrange[1]:
                    arr[i,j] = k
                    break
                else:
                    arr[i,j] = -999
    return arr

feats = {'1400W':(	1387,	1445,	1350,	1450),
'1480W':(	1471,	1491,	1440,	1520),
'1550W':(	1520,	1563,	1510,	1610),
'1760W':(	1751,	1764,	1730,	1790),
'1850W':(	1749,	1949,	1820,	1880),
'1900W': (1840,1990, 1850, 1970), 
'2080W':(	1980,	2180,	2060,	2100),
'2160W':(	2159,	2166,	2138,	2179),
'2200W':(	2185,	2215,	2120,	2245),
'2250W':(	2248,	2268,	2230,	2280),
'2290W':(	2279,	2338,	2270,	2320),
'2320W':(	2300,	2340,	2295,	2355),
'2350W':(	2320,	2366,	2310,	2370),
'2390W':(	2377,	2406,	2375,	2435),
'2950W':(2920, 2980, 2900, 3000),
'2950AW':(2900, 2960, 2900, 3000),
'2830W':(	2677,	2890,	2790,	2890),
'3000W':(	2900,	3100,	2795,	3900),
'3500W':(	3400,	3600,	3300,	3700),#NOT from laukamp
'4000W':(	3930,	4150,	3800,	4200),
'4000WIDEW':(3910,	4150,	3800,	4200),
'4470TRUEW':(	4460,	4490,	4350,	4550),
'4500SW':(4570,	4850,	4090,	5040),
'4500CW':(4625,	4770,	4090,	5040),
'4670W': (4300, 4800, 4300, 4800),
'4920W': (4850, 5100, 4850, 5157),
'4000V_NARROWW': (3930,4150,3800,4200),
'4000shortW': (3850,4000,3800,4200),
'2950BW':(2920, 2990, 2790, 3200),
}

def est_peaks_cube_scipy(data, bands, wavrange=(2300, 2340)):
    w, l, b = data.shape
    arr = np.zeros((w,l))
    for i in range(w):
        for j in range(l):
            peak_indices, peak_dict = sc.signal.find_peaks(1-data[i,j], height=(None, None))
            x = [bands[i] for i in peak_indices]
            for k in x:        
                if k > wavrange[0] and k < wavrange[1]:
                     arr[i,j] = k
                     break
                else:
                    arr[i,j] = -999
    return arr


def Combined_MWL(savgol, savgol_cr, mask, bands, feature, technique = 'QUAD', 
                 thresh=0.2):
    
    feats = {'1400W':(	1387,	1445,	1350,	1450),
    '1480W':(	1471,	1491,	1440,	1520),
    '1550W':(	1520,	1563,	1510,	1610),
    '1760W':(	1751,	1764,	1730,	1790),
    '1850W':(	1749,	1949,	1820,	1880),
    '1900W': (1840,1990, 1850, 1970), 
    '2080W':(	1980,	2180,	2060,	2100),
    '2160W':(	2159,	2166,	2138,	2179),
    '2200W':(	2185,	2215,	2120,	2245),
    '2250W':(	2248,	2268,	2230,	2280),
    '2290W':(	2279,	2338,	2270,	2320),
    '2320W':(	2300,	2340,	2295,	2355),
    '2350W':(	2320,	2366,	2310,	2370),
    '2390W':(	2377,	2406,	2375,	2435),
    '2950W':(2920, 2980, 2900, 3000),
    '2950AW':(2900, 2960, 2900, 3000),
    '2830W':(	2677,	2890,	2790,	2890),
    '3000W':(	2900,	3100,	2795,	3900),
    '3500W':(	3400,	3600,	3300,	3700),#NOT from laukamp
    '4000W':(	3930,	4150,	3800,	4200),
    '4000WIDEW':(3910,	4150,	3800,	4200),
    '4470TRUEW':(	4460,	4490,	4350,	4550),
    '4500SW':(4570,	4850,	4090,	5040),
    '4500CW':(4625,	4770,	4090,	5040),
    '4670W': (4300, 4800, 4300, 4800),
    '4920W': (4850, 5100, 4850, 5157),
    '4000V_NARROWW': (3930,4150,3800,4200),
    '4000shortW': (3850,4000,3800,4200),
    '2950BW':(2920, 2990, 2790, 3200),
    }
    cr_crop_min = feats[feature][2]
    cr_crop_max = feats[feature][3]
    cr_crop_min_index = np.argmin(np.abs(np.array(bands)-(feats[feature][2])))
    cr_crop_max_index = np.argmin(np.abs(np.array(bands)-(feats[feature][3])))
    wav_min = feats[feature][0]
    wav_max = feats[feature][1]
    wav_min_index = np.argmin(np.abs(np.array(bands)-(feats[feature][0])))
    wav_max_index = np.argmin(np.abs(np.array(bands)-(feats[feature][1])))
    
    check_response =  est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min, wav_max))
    #check_response =  est_peaks_cube_scipy_thresh(savgol_cr, bands, wavrange=(wav_min, wav_max), thresh = thresh)
    
    if technique.upper() == 'QND': 
        print(technique)
        new_bands = bands[cr_crop_min_index:cr_crop_max_index]
        m, n, b = savgol_cr.shape
        data = remove_hull(savgol_cr[:,:, cr_crop_min_index:cr_crop_max_index])
        minsA = np.zeros((data.shape[0], data.shape[1]))
        minsA = np.argmin(data, axis=2)
        minsB = np.zeros((data.shape[0], data.shape[1]), dtype=float)
        for i in range(new_bands.shape[0]):
            minsB[minsA==i] = new_bands[i]
        position = minsB
        depth = 1-np.min(data, axis=2)
        
    elif technique.upper() == 'POLY':  
        print(technique)        
        hiswir = hylite.HyImage(savgol)
        hiswir.set_wavelengths(bands)
        Mpoly = minimum_wavelength( hiswir, float(cr_crop_min), float(cr_crop_max), 
                                   n=1, method='poly', log=False, vb=False, minima=True)
        depth = Mpoly.__getitem__([0,'depth'])
        position = Mpoly.__getitem__([0,'pos'])
        width = Mpoly.__getitem__([0,'width'])
    elif technique.upper() == 'GAUS':
        print(technique)          
        hiswir = hylite.HyImage(savgol)
        hiswir.set_wavelengths(bands)
        Mpoly = minimum_wavelength( hiswir, float(cr_crop_min), float(cr_crop_max), 
                                   n=1, method='gaussian', log=False, vb=True, minima=True)
        depth = Mpoly.__getitem__([0,'depth'])
        position = Mpoly.__getitem__([0,'pos'])
        width = Mpoly.__getitem__([0,'width'])
    elif technique.upper() == 'QUAD':   
        print(technique)
        hiswir = hylite.HyImage(savgol)
        hiswir.set_wavelengths(bands)
        Mpoly = minimum_wavelength( hiswir, float(cr_crop_min), float(cr_crop_max), 
                                   n=1, method='quad', log=False, vb=False, minima=True)
        depth = Mpoly.__getitem__([0,'depth'])
        position = Mpoly.__getitem__([0,'pos'])
        width = Mpoly.__getitem__([0,'width'])
    feature_mask = mask.copy()
    feature_mask[check_response < 0] =1
    #feature_mask[position>wav_max] = 1
    #feature_mask[position<wav_min] = 1
    #if thresh:
        #feature_mask[depth<thresh] = 1
    
        
    
    
    return position, depth, feature_mask


def carbonate_facies(savgol, savgol_cr, mask, bands, technique = 'QUAD'):
    
    cr_crop_min_22 = feats['2200W'][2]
    cr_crop_max_22 = feats['2200W'][3]
    cr_crop_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][2])))
    cr_crop_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][3])))
    wav_min_22 = feats['2200W'][0]
    wav_max_22 = feats['2200W'][1]
    wav_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][0])))
    wav_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][1])))
    
    cr_crop_min_23 = feats['2320W'][2]
    cr_crop_max_23 = feats['2320W'][3]
    cr_crop_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][2])))
    cr_crop_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][3])))
    wav_min_23 = feats['2320W'][0]
    wav_max_23 = feats['2320W'][1]
    wav_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][0])))
    wav_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][1])))
    
    
    print('checking dirty or clean')
    dirty_or_clean = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_22, wav_max_22))
    print('checking how calcitic')
    calcitic_or_not = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_23, wav_max_23))
    #carb wavelength position
    print('MWL-ing')
    calc_or_dolo, _, feat_mask = Combined_MWL(savgol, savgol_cr, mask, bands, '2320W', technique = technique)
  
                
    # ==========#Facies colours"===================================================================
    clean_calcite = [0, 0, 255] #1 in data
    clean_dolomitic_calcite = [255, 0, 0]#2 in data
    clean_calcitic_dolomite = [0, 255, 255] #3 in data           
    clean_dolomite =[204, 255, 153]#4 in data
    dirty_calcite = [0, 255, 0]#5 in data
    dirty_dolomitic_calcite = [255, 255, 0]#6 in data
    dirty_calcitic_dolomite = [255, 0, 255]#7 in data
    dirty_dolomite = [204, 153, 255]#8 in data
# =============================================================================
    M, N, B = savgol_cr.shape
    output_image = np.zeros((M,N, 3))
    output_data = np.zeros((M,N))
#decision tree            
# 8 part facies                
    for i in range(M):
        for j in range(N):
            if dirty_or_clean[i,j] > 0:
                #dirty
                if calc_or_dolo[i,j] >= 2330:           
                    output_image[i, j] = dirty_calcite
                    output_data[i, j] = 5
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = dirty_dolomitic_calcite
                    output_data[i, j] = 6
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = dirty_calcitic_dolomite
                    output_data[i, j] = 7
                else:                                   
                    output_image[i, j] = dirty_dolomite
                    output_data[i, j] = 8
            else:                                       
                #Clean
                if calc_or_dolo[i,j] >= 2330:           
                    output_image[i, j] = clean_calcite
                    output_data[i, j] = 1
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = clean_dolomitic_calcite
                    output_data[i, j] = 2
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = clean_calcitic_dolomite
                    output_data[i, j] = 3
                else:                                   
                    output_image[i, j] = clean_dolomite
                    output_data[i, j] = 4
            if calcitic_or_not[i,j] < 0: 
                if dirty_or_clean[i,j] < 0:
                    #not calcitic not siliciclastic
                   output_image[i, j] = [255, 255, 255]# 10 non-carbonaceous response
                   output_data[i, j] = 10
                else:
                    #not carbonaceous but siliciclastic
                    output_image[i, j] = [96, 96, 96] # 9
                    output_data[i, j] = 9
        
    output_data[mask==1] = 0
    output_image[mask==1] = [0,0,0]
    return output_data, output_image

import numpy as np

def carbonate_facies_original(savgol, savgol_cr, mask, bands, technique = 'QUAD'):
    
    cr_crop_min_22 = feats['2200W'][2]
    cr_crop_max_22 = feats['2200W'][3]
    cr_crop_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][2])))
    cr_crop_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][3])))
    wav_min_22 = feats['2200W'][0]
    wav_max_22 = feats['2200W'][1]
    wav_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][0])))
    wav_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][1])))
    
    cr_crop_min_23 = feats['2320W'][2]
    cr_crop_max_23 = feats['2320W'][3]
    cr_crop_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][2])))
    cr_crop_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][3])))
    wav_min_23 = feats['2320W'][0]
    wav_max_23 = feats['2320W'][1]
    wav_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][0])))
    wav_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][1])))
    
    
    print('checking dirty or clean')
    dirty_or_clean = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_22, wav_max_22))
    print('checking how calcitic')
    calcitic_or_not = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_23, wav_max_23))
    #carb wavelength position
    print('MWL-ing')
    print(bands[cr_crop_min_index_23:cr_crop_max_index_23].shape)
    calc_or_dolo, _ = get_SQM_peak_finder_vectorized(remove_hull(savgol[:,:,cr_crop_min_index_23:cr_crop_max_index_23]), bands[cr_crop_min_index_23:cr_crop_max_index_23])
             
    # ==========#Facies colours"===================================================================
    clean_calcite = [0, 0, 255] #1 in data
    clean_dolomitic_calcite = [255, 0, 0]#2 in data
    clean_calcitic_dolomite = [0, 255, 255] #3 in data           
    clean_dolomite =[204, 255, 153]#4 in data
    dirty_calcite = [0, 255, 0]#5 in data
    dirty_dolomitic_calcite = [255, 255, 0]#6 in data
    dirty_calcitic_dolomite = [255, 0, 255]#7 in data
    dirty_dolomite = [204, 153, 255]#8 in data
# =============================================================================
    M, N, B = savgol_cr.shape
    output_image = np.zeros((M,N, 3))
    output_data = np.zeros((M,N))
#decision tree            
# 8 part facies                
    for i in range(M):
        for j in range(N):
            if dirty_or_clean[i,j] > 0:
                #dirty
                if calc_or_dolo[i,j] >= 2330:           
                    output_image[i, j] = dirty_calcite
                    output_data[i, j] = 5
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = dirty_dolomitic_calcite
                    output_data[i, j] = 6
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = dirty_calcitic_dolomite
                    output_data[i, j] = 7
                else:                                   
                    output_image[i, j] = dirty_dolomite
                    output_data[i, j] = 8
            else:                                       
                #Clean
                if calc_or_dolo[i,j] >= 2330:           
                    output_image[i, j] = clean_calcite
                    output_data[i, j] = 1
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = clean_dolomitic_calcite
                    output_data[i, j] = 2
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = clean_calcitic_dolomite
                    output_data[i, j] = 3
                else:                                   
                    output_image[i, j] = clean_dolomite
                    output_data[i, j] = 4
            if calcitic_or_not[i,j] < 0: 
                if dirty_or_clean[i,j] < 0:
                    #not calcitic not siliciclastic
                   output_image[i, j] = [255, 255, 255]# 10 non-carbonaceous response
                   output_data[i, j] = 10
                else:
                    #not carbonaceous but siliciclastic
                    output_image[i, j] = [96, 96, 96] # 9
                    output_data[i, j] = 9
        
    output_data[mask==1] = 0
    output_image[mask==1] = [0,0,0]
    return output_data, output_image



def get_SQM_peak_finder_vectorized(data, bands, atol=1e-12):
    """
    Vectorised implementation of the Simple Quadratic Method (SQM).
    Inputs:
        data  : (M,N,B) or (N,B) array of continuum-removed spectra (baseline ~1), cropped to a single feature.
        bands : (B,) array of band centres (same units as desired output, e.g., nm).
        atol  : numerical tolerance for flatness / zero-division guards.
    Returns:
        SQM(np.squeeze(tru), np.squeeze(dep))
    Notes:
        - Edge minima (b==0 or b==B-1) and flat triplets fall back to the discrete band centre;
          depth is left as 0.0 for those pixels (as in your original).
        - Depth is computed from the fitted parabola: depth = 1 - f(min).
    """

    # Ensure 3D
    if data.ndim == 2:
        data = np.expand_dims(data, 0)  # (1,N,B)
    M, N, B = data.shape
    print(M,N,B, bands.shape)
    # Argmin index at each pixel
    b = np.argmin(data, axis=-1)  # (M,N)
    
    # Neighbor indices
    bL = np.clip(b - 1, 0, B - 1)
    bR = np.clip(b + 1, 0, B - 1)

    # Gather spectral values at L, 0, R
    # Build broadcast indices
    ii, jj = np.indices((M, N), sparse=False)
    D0  = data[ii, jj, b ]   # at minimum band
    DL  = data[ii, jj, bL]   # left neighbor
    DR  = data[ii, jj, bR]   # right neighbor

    # Gather wavelengths
    W0  = bands[b ]
    WL  = bands[bL]
    WR  = bands[bR]

    # Start outputs with fallbacks (band centre), depth 0
    tru = W0.copy()
    dep = np.zeros_like(D0, dtype=float)

    # Valid pixels: not at edges and not flat around the min
    not_edge = (b > 0) & (b < (B - 1))
    not_flat = (~np.isclose(D0, DR, atol=atol)) & (~np.isclose(DL, D0, atol=atol))
    mask = not_edge & not_flat

    if np.any(mask):
        # Compute quadratic coefficients using the 3-point formula (Rodger et al. 2012)
        # A = an/ad, Bc = ...
        an = (D0[mask] - DL[mask]) * (WL[mask] - WR[mask]) + (DR[mask] - D0[mask]) * (W0[mask] - WL[mask])
        ad = (WL[mask] - WR[mask]) * (W0[mask]**2 - WL[mask]**2) + (W0[mask] - WL[mask]) * (WR[mask]**2 - WL[mask]**2)

        # Guards against degenerate geometry (tiny denominator or identical wavelengths)
        good_geom = (~np.isclose(ad, 0.0, atol=atol)) & (~np.isclose(W0[mask] - WL[mask], 0.0, atol=atol))

        # Submask of truly valid pixels
        if np.any(good_geom):
            A = an[good_geom] / ad[good_geom]
            Bc = ((D0[mask][good_geom] - DL[mask][good_geom]) - A * (W0[mask][good_geom]**2 - WL[mask][good_geom]**2)) / (W0[mask][good_geom] - WL[mask][good_geom])

            # Avoid division by ~0 for A
            nondeg = ~np.isclose(A, 0.0, atol=atol)

            # Final valid set
            valid = np.zeros_like(mask, dtype=bool)
            # Map good_geom & nondeg back into full image mask positions
            idx_mask = np.argwhere(mask)
            idx_good = idx_mask[good_geom]
            idx_final = idx_good[nondeg]
            if idx_final.size > 0:
                vi = idx_final[:, 0]
                vj = idx_final[:, 1]

                A  = A[nondeg]
                Bc = Bc[nondeg]
                # Vertex (refined minimum)
                m = -Bc / (2.0 * A)

                # Compute C via left point, then depth = 1 - f(m)
                C = DL[vi, vj] - (A * (WL[vi, vj]**2)) - (Bc * WL[vi, vj])
                fmin = A * (m**2) + Bc * m + C
                d = 1.0 - fmin

                tru[vi, vj] = m
                dep[vi, vj] = d

    # Report fallback usage (edges/flat/degenerate)
    total = M * N
    used_band_centre = np.count_nonzero(dep == 0.0)  # depth stays 0 only where we fell back (by construction)
    

    return np.squeeze(tru), np.squeeze(dep)


#TODO: Is this called anywhere? Delete
def crop_with_mask_cv2(cube, mask, margin=0, invert=False, min_area=0):
    """
    Crop a hyperspectral cube to the bounding box of a boolean mask.

    Parameters
    ----------
    cube : np.ndarray
        3D array of shape (y, x, bands)
    mask : np.ndarray
        2D boolean array, same (y, x) shape as cube. True = foreground.
    margin : int
        Extra pixels of padding around the detected box.
    invert : bool
        If True, treat False as foreground instead of True.
    min_area : int
        Ignore connected components smaller than this area (optional).

    Returns
    -------
    cropped : np.ndarray
        Cropped hyperspectral cube (y, x, bands)
    bbox : tuple[int]
        (y0, y1, x0, x1) bounding box used for cropping
    """
    if cube.ndim != 3:
        raise ValueError("cube must be 3D (y, x, bands)")
    if mask.shape != cube.shape[:2]:
        raise ValueError("mask must match cube spatial dimensions (y, x)")
    if mask.dtype != bool:
        raise ValueError("mask must be boolean")

    # Optionally invert
    if invert:
        mask = ~mask

    # Convert for cv2 (expects 0/255 uint8)
    m = (mask.astype(np.uint8)) * 255

    # Find external contours (connected regions)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    # Optionally ignore tiny components
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            raise ValueError("No contours above min_area")

    # Largest component
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Add margin and clip to image bounds
    H, W = mask.shape
    x0 = max(0, x - margin); y0 = max(0, y - margin)
    x1 = min(W, x + w + margin); y1 = min(H, y + h + margin)

    cropped = cube[y0:y1, x0:x1, :]
    return cropped, (y0, y1, x0, x1)

HERE = pathlib.Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "Box_picker_YOLO.pt"


print(MODEL_PATH)

def get_coords(box):
    yolo_x, yolo_y, yolo_width, yolo_height = box.xywhn[0].tolist()
    image_height, image_width = box.orig_shape
    center_x = yolo_x * image_width
    center_y = yolo_y * image_height
    left_top_x = center_x - ((yolo_width * image_width)/2)
    left_top_y = center_y - ((yolo_height * image_height)/2)
    right_bottom_x = box.xyxyn[0].tolist()[2] * image_width
    right_bottom_y = box.xyxyn[0].tolist()[3] * image_height
    cropping_coords = np.array((left_top_y, right_bottom_y,left_top_x, right_bottom_x )).astype(np.uint32)
    return int(left_top_y)-5, int(right_bottom_y)+5,int(left_top_x)-5, int(right_bottom_x)+5





def auto_crop(radiance):
    m, c = sp.kmeans(np.nan_to_num(radiance, nan=0), 12, 3)
    #sp.imshow(radiance, classes=m)
    quickmask = np.zeros(radiance.shape[:2])
    quickmask[m==0] = 1
    kernel = np.ones((15, 15), np.uint8)  # adjust for your resolution
    mask_clean = cv2.morphologyEx(quickmask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cropped, (y0, y1, x0, x1) = crop_with_mask_cv2(radiance, mask_clean.astype(bool), invert=True)
    return cropped
    

   
    
def resample_spectrum(x_src_nm: np.ndarray, y_src: np.ndarray, x_tgt_nm: np.ndarray) -> np.ndarray:
    """
    Fast 1D linear resample onto target band centers (nm).
    Clamps to edges; returns finite array (NaNs filled with 0).
    """
    y = np.interp(x_tgt_nm, x_src_nm, y_src, left=y_src[0], right=y_src[-1]).astype(float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y  
    
def cr(spectra):
    '''helper function to keep scientific dependencies contained'''
    return remove_hull(spectra)


    
    
    
def detect_slice_rectangles_robust(
    image,                  #numpy array uint8, 3-channel (H, W, 3)
    min_area_frac=0.0005,     # min polygon area as a fraction of image area
    canny_sigma=0.33,         # auto-Canny thresholds from median
    approx_eps_frac=0.02,     # polygon approximation tolerance
    close_kernel=5,           # morphological close kernel size (pixels)
    use_otsu=True,            # binarize before Canny (good for UI/scans)
    allow_rotated=True        # also accept minAreaRect boxes
):
    img = image
    
    H, W = img.shape[:2]
    min_area = H * W * min_area_frac

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    src = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] if use_otsu else gray
    v = np.median(src); lo = int(max(0, (1.0 - canny_sigma) * v)); hi = int(min(255, (1.0 + canny_sigma) * v))
    edges = cv2.Canny(src, lo, hi)

    if close_kernel and close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects_xywh, rects_poly = [], []

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_eps_frac * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            rects_xywh.append((x, y, w, h))
            rects_poly.append(approx[:,0,:].astype(np.float32))
        elif allow_rotated:
            r = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = r
            if rw * rh >= min_area:
                box = cv2.boxPoints(r).astype(np.float32)
                x, y, w, h = cv2.boundingRect(box.astype(np.int32))
                rects_xywh.append((x, y, w, h))
                rects_poly.append(box)

    # simple NMS by IoU on axis-aligned boxes
    def iou(a,b):
        ax,ay,aw,ah = a; bx,by,bw,bh = b
        ax2, ay2, bx2, by2 = ax+aw, ay+ah, bx+bw, by+bh
        ix1, iy1, ix2, iy2 = max(ax,bx), max(ay,by), min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        return inter / (aw*ah + bw*bh - inter + 1e-9)

    keep = []
    used = [False]*len(rects_xywh)
    order = sorted(range(len(rects_xywh)), key=lambda i: rects_xywh[i][2]*rects_xywh[i][3], reverse=True)
    for i in order:
        if used[i]: continue
        keep.append(i)
        for j in order:
            if used[j] or j==i: continue
            if iou(rects_xywh[i], rects_xywh[j]) > 0.6:
                used[j] = True

    rects_xywh = [rects_xywh[i] for i in keep]
    rects_poly = [rects_poly[i] for i in keep]
    
    areas = np.array([cv2.contourArea(p) for p in rects_poly])
    index = np.argmax(areas)
    x, y, w, h = rects_xywh[index]
    y1, y2 = y, y + h
    x1, x2 = x, x + w
    crop_slice = (slice(y, y + h), slice(x, x + w))
    return image[crop_slice], crop_slice
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    