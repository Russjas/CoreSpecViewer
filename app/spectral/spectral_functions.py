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
import xml.etree.ElementTree as ET 
from gfit.util import remove_hull
import hylite
from hylite.analyse import minimum_wavelength
import cv2
import matplotlib

from ..config import con_dict  # live shared dict

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')

def _slice_from_sensor(sensor_type: str):
    s = sensor_type or ""
    print(s)
    if "SWIR" in s:
        start, stop = con_dict["swir_slice_start"], con_dict["swir_slice_stop"]
    elif "RGB" in s:
        start, stop = con_dict["rgb_slice_start"], con_dict["rgb_slice_stop"]
    elif "FX50" in s: 
        start, stop = con_dict["mwir_slice_start"], con_dict["mwir_slice_stop"]
    else:
        start, stop = con_dict["default_slice_start"], con_dict["default_slice_stop"]
    print(slice(start, stop, None))
    return slice(start, stop, None)

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
    denom = np.subtract(wmax, dmin) # Calculate denominator

    result = np.where(denom != 0, np.divide(np.subtract(data, dmin), denom), 0.0)
    #result = np.divide(np.subtract(data, dmin), np.subtract(wmax, dmin))
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
        sensor = header['sensor type']
        band_slice = _slice_from_sensor(sensor)
        snr = None
        
        
    data  = data[:, :, band_slice]
    white = white_ref[:, :, band_slice]
    dark  = dark_ref[:, :, band_slice]
    bands = bands[band_slice]
    
    data_reflect = reflect_correct(data, white, dark)
    
    return data_reflect, bands, snr

# Actual processing funcs========================================================

def process(cube):
    win = int(con_dict["savgol_window"])
    poly = int(con_dict["savgol_polyorder"])
    savgol = sc.signal.savgol_filter(cube, win, poly)
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
    
 
@jit(nopython=True)
def numpy_pearson_stackexemplar_threshed(data, exemplar_stack):
    num = exemplar_stack.shape[0]
    coeffs = np.zeros((data.shape[0], data.shape[1]))
    confidence = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            c_list = np.zeros((num))
            for n in range(num):
                c_list[n] = np.corrcoef(data[i,j], exemplar_stack[n])[1,0]
            if np.max(c_list) > 0.7:
                coeffs[i,j] = np.argmax(c_list)
                confidence[i,j] = np.max(c_list)
                
            else:
                coeffs[i,j] = -999
                confidence[i,j] = np.max(c_list)
    return coeffs, confidence
    

def mineral_map_wta(data, exemplar_stack, thresh=0.70, invalid_value=-999):
    """
    Vectorized winner-takes-all Pearson to match np.corrcoef semantics.

    data:            (H, W, B) float
    exemplar_stack:  (K, B)    float
    Returns:
        class_idx: (H, W) int32  (=-999 for low confidence)
        best_corr: (H, W) float32  Pearson r in [-1, 1]
    """
    H, W, B = data.shape
    K = exemplar_stack.shape[0]

    # --- z-score with sample std (ddof=1) to match np.corrcoef
    # data -> (N, B)
    X = data.reshape(-1, B).astype(np.float32)
    X_mean = X.mean(axis=1, keepdims=True)
    X_std  = X.std(axis=1, ddof=1, keepdims=True)
    X_std  = np.where(X_std < 1e-12, 1e-12, X_std)
    Xz = (X - X_mean) / X_std                              # (N, B)

    # exemplars -> (K, B)
    E = exemplar_stack.astype(np.float32)
    E_mean = E.mean(axis=1, keepdims=True)
    E_std  = E.std(axis=1, ddof=1, keepdims=True)
    E_std  = np.where(E_std < 1e-12, 1e-12, E_std)
    Ez = (E - E_mean) / E_std                              # (K, B)

    # --- Pearson matrix: divide by (B-1) to match corrcoef scaling
    corr = (Xz @ Ez.T) / max(B - 1, 1)                    # (N, K), in [-1, 1]
    best_corr = corr.max(axis=1)
    idx = corr.argmax(axis=1).astype(np.int32)

    # --- apply threshold like your nested loops
    idx = np.where(best_corr >= thresh, idx, invalid_value).astype(np.int32)

    return idx.reshape(H, W), best_corr.reshape(H, W).astype(np.float32)

def mineral_map_wta_strict(data, exemplar_stack, thresh=0.70, invalid_value=-999):
    """
    Vectorized WTA Pearson that replicates np.corrcoef semantics:
      - float64 math
      - sample std (ddof=1)
      - divide by (B-1)
      - produces NaN where corrcoef would (zero-variance vectors)
      - applies threshold like the loop: <= thresh -> -999
    Returns (class_idx: int32 (H,W), best_corr: float32 (H,W))
    """
    H, W, B = data.shape
    K = exemplar_stack.shape[0]

    X = data.reshape(-1, B).astype(np.float64)   # (N,B)
    E = exemplar_stack.astype(np.float64)        # (K,B)

    # Means & sample std (ddof=1)
    X_mean = X.mean(axis=1, keepdims=True)
    E_mean = E.mean(axis=1, keepdims=True)
    X_std  = X.std(axis=1, ddof=1, keepdims=True)
    E_std  = E.std(axis=1, ddof=1, keepdims=True)

    # Zero-variance masks (these produce NaN in corrcoef)
    X_zero = (X_std == 0)                         # (N,1)
    E_zero = (E_std == 0)                         # (K,1)

    # z-scores; where std==0, leave as 0 then we will set NaNs via masks later
    Xz = (X - X_mean) / np.where(X_std == 0, 1, X_std)
    Ez = (E - E_mean) / np.where(E_std == 0, 1, E_std)

    # Pearson matrix with (B-1) divisor to match corrcoef scaling
    corr = (Xz @ Ez.T) / max(B - 1, 1)           # (N,K) float64

    # Inject NaNs where corrcoef would be NaN (zero variance in either vector)
    if X_zero.any():
        corr[X_zero[:, 0], :] = np.nan
    if E_zero.any():
        corr[:, E_zero[:, 0]] = np.nan

    # Best corr and argmax with NaN-aware handling
    best_corr = np.nanmax(corr, axis=1)          # (N,)
    # For rows that are all-NaN, nanargmax would error; handle explicitly
    all_nan = np.isnan(best_corr)
    idx = np.empty_like(best_corr, dtype=np.int32)
    if (~all_nan).any():
        idx[~all_nan] = np.nanargmax(corr[~all_nan], axis=1).astype(np.int32)
    if all_nan.any():
        idx[all_nan] = invalid_value

    # Apply threshold exactly like the loop (> 0.70 keeps, else -999)
    keep = best_corr > float(thresh)
    idx = np.where(keep, idx, invalid_value).astype(np.int32)

    return idx.reshape(H, W), best_corr.reshape(H, W).astype(np.float32)   

    
def kmeans_spectral_wrapper(data, clusters, iters):
    m, c = sp.kmeans(data, clusters, iters)
    return m, c

#TODO: Come back and refactor this after time checks
def mk_thumb_sidelines(arr, baseheight=90, basewidth=800, mask=None):
    """
    Create a PIL thumbnail image from an array, using the same
    visual conventions as ImageCanvas2D.show_rgb.

    Parameters
    ----------
    arr : np.ndarray
        Shape (H, W, B), (H, W) or (H, W, 3).
    baseheight : int
        Max height of the thumbnail (pixels).
    basewidth : int
        Max width of the thumbnail (pixels).
    mask : np.ndarray[bool], optional
        Boolean mask of shape (H, W). True = masked (black).

    Returns
    -------
    PIL.Image.Image
        RGB thumbnail image, ready to save as JPEG.
    """
    arr = np.asarray(arr)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported array shape {arr.shape}; expected 2D or 3D.")
    if 0 in arr.shape:
        raise ValueError(f"arr shape {arr.shape} cannot have a zero size dim")

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match array spatial shape {arr.shape[:2]}."
            )
    if arr.shape[0] > arr.shape[1]:
        arr = np.flip(np.swapaxes(arr, 0, 1), axis=0)
        if mask is not None:
            mask = np.flip(np.swapaxes(mask, 0, 1), axis=0)
    # ---- Convert to RGB (H, W, 3), matching ImageCanvas2D.show_rgb logic
    if arr.ndim == 2:
        # 2D -> use the same colormap as show_rgb (my_map, with min/max scaling)
        a = arr.astype(float)
        amin = np.nanmin(a)
        amax = np.nanmax(a)
        if amax > amin:
            norm = (a - amin) / (amax - amin)
        else:
            norm = np.zeros_like(a, dtype=float)
        rgb = my_map(norm)[..., :3]  # (H, W, 3) float in [0,1]

        # scale to 0–255
        rgb8 = np.nan_to_num(rgb * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)

    else:
        # 3D
        H, W, C = arr.shape

        # Hyperspectral -> use the same false-colour function as the viewer
        if C > 3:
            fc = get_false_colour(arr)  # should return (H, W, 3)
            fc = np.asarray(fc)
            if fc.ndim != 3 or fc.shape[2] != 3:
                raise ValueError("sf.get_false_colour must return an (H, W, 3) array.")
            # handle float vs uint8
            if np.issubdtype(fc.dtype, np.integer):
                rgb8 = np.clip(fc, 0, 255).astype(np.uint8)
            else:
                # float: assume arbitrary range, normalise to [0,1]
                vmin = np.nanmin(fc)
                vmax = np.nanmax(fc)
                if vmax > vmin:
                    rgb = (fc - vmin) / (vmax - vmin)
                else:
                    rgb = np.zeros_like(fc, dtype=float)
                rgb8 = np.nan_to_num(rgb * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)

        else:
            # C == 1 or C == 3
            a = arr

            if C == 1:
                a = np.repeat(a, 3, axis=2)

            # Now a is (H, W, 3)
            if np.issubdtype(a.dtype, np.integer):
                # Assume uint8-style data
                rgb8 = np.clip(a, 0, 255).astype(np.uint8)
            else:
                # float: we don't know the range -> normalise like imshow defaults
                vmin = np.nanmin(a)
                vmax = np.nanmax(a)
                if vmax > vmin:
                    rgb = (a - vmin) / (vmax - vmin)
                else:
                    rgb = np.zeros_like(a, dtype=float)
                rgb8 = np.nan_to_num(rgb * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)

    # ---- Apply mask (masked regions = black)
    if mask is not None:
        rgb8[mask] = 0

    # ---- Create PIL image and resize with aspect-ratio preserved
    h, w = rgb8.shape[:2]
    scale = min(basewidth / float(w), baseheight / float(h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = Image.fromarray(rgb8, mode="RGB")
    im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return im

import time

def mk_thumb(arr, baseheight=90, basewidth=800, mask=None):
    print('make thumb called')
    t0 = time.perf_counter()

    def checkpoint(label):
        print(f"[mk_thumb] {label}: {time.perf_counter() - t0:.4f}s")

    checkpoint("start")

    arr = np.asarray(arr)
    checkpoint("after np.asarray(arr)")

    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported array shape {arr.shape}; expected 2D or 3D.")
    if 0 in arr.shape:
        raise ValueError(f"arr shape {arr.shape} cannot have a zero size dim")

    # mask check
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match array spatial shape {arr.shape[:2]}."
            )
    checkpoint("after mask validation")

    # orientation flip
    if arr.shape[0] > arr.shape[1]:
        arr = np.flip(np.swapaxes(arr, 0, 1), axis=0)
        if mask is not None:
            mask = np.flip(np.swapaxes(mask, 0, 1), axis=0)
    checkpoint("after optional orientation flip")

    # ---- Convert to RGB
    if arr.ndim == 2:
        # 2D → colormap
        a = arr.astype(float)
        checkpoint("after arr.astype(float)")

        amin = np.nanmin(a)
        amax = np.nanmax(a)
        checkpoint("after nanmin/max")

        if amax > amin:
            norm = (a - amin) / (amax - amin)
        else:
            norm = np.zeros_like(a, dtype=float)
        checkpoint("after normalisation")

        rgb = my_map(norm)[..., :3]
        checkpoint("after colormap call (my_map(norm))")

        rgb8 = np.nan_to_num(rgb * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
        checkpoint("after rgb→uint8 conversion")

    else:
        # 3D
        H, W, C = arr.shape

        if C > 3:
            # hyperspectral false-colour conversion
            fc = get_false_colour(arr)
            checkpoint("after get_false_colour(arr)")

            fc = np.asarray(fc)
            checkpoint("after np.asarray(fc)")

            if fc.ndim != 3 or fc.shape[2] != 3:
                raise ValueError("sf.get_false_colour must return (H, W, 3) array.")

            if np.issubdtype(fc.dtype, np.integer):
                rgb8 = np.clip(fc, 0, 255).astype(np.uint8)
                checkpoint("after clip+astype for integer false-colour")
            else:
                vmin = np.nanmin(fc)
                vmax = np.nanmax(fc)
                checkpoint("after nanmin/max on false-colour")

                if vmax > vmin:
                    rgb = (fc - vmin) / (vmax - vmin)
                else:
                    rgb = np.zeros_like(fc, dtype=float)
                checkpoint("after false-colour normalisation")

                rgb8 = np.nan_to_num(rgb * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
                checkpoint("after false-colour float→uint8")

        else:
            # C == 1 or C == 3
            a = arr

            if C == 1:
                a = np.repeat(a, 3, axis=2)
                checkpoint("after repeat single band to RGB")

            if np.issubdtype(a.dtype, np.integer):
                rgb8 = np.clip(a, 0, 255).astype(np.uint8)
                checkpoint("after integer RGB clip+astype")
            else:
                vmin = np.nanmin(a)
                vmax = np.nanmax(a)
                checkpoint("after nanmin/max for RGB")

                if vmax > vmin:
                    rgb = (a - vmin) / (vmax - vmin)
                else:
                    rgb = np.zeros_like(a, dtype=float)
                checkpoint("after float RGB normalisation")

                rgb8 = np.nan_to_num(rgb * 255.0, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
                checkpoint("after float RGB→uint8 conversion")

    # ---- Apply mask
    if mask is not None:
        rgb8[mask] = 0
        checkpoint("after applying mask")

    # resize
    h, w = rgb8.shape[:2]
    scale = min(basewidth / float(w), baseheight / float(h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = Image.fromarray(rgb8, mode="RGB")
    checkpoint("after Image.fromarray")

    im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    checkpoint("after resize (LANCZOS)")

    checkpoint("END")
    return im
    
    
    
    
    
    
    
    
    
    
    
    
    
    