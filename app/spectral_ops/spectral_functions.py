"""
Large, monolithic module for performing operations. A few of these functions are
useless without the GUI (con_dict, slice_from_sensor), but the vast majority are
UI agnostic and can be run on any hyperspectral cube in npy format.

NB. This will be broken up in a re-factor eventually.
"""
import time
import xml.etree.ElementTree as ET


import cv2
from gfit.util import remove_hull
import hylite
from hylite.analyse import minimum_wavelength
from hylite.sensors import Fenix as HyliteFenix
import matplotlib
from numba import jit
import numpy as np
from PIL import Image
import scipy as sc
import spectral as sp
import spectral.io.envi as envi
from ..config import con_dict  # live shared dict

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')

#===================io funcs===================================================

def _slice_from_sensor(sensor_type: str):
    """
    Derive the edge bands to slice out using the config dict
    """
    s = sensor_type or ""
    print(s)
    if "SWIR" in s:
        start, stop = con_dict["swir_slice_start"], con_dict["swir_slice_stop"]
    elif "RGB" in s:
        start, stop = con_dict["rgb_slice_start"], con_dict["rgb_slice_stop"]
    elif "FX50" in s:
        start, stop = con_dict["mwir_slice_start"], con_dict["mwir_slice_stop"]
    elif "FENIX" in s:
        start, stop = con_dict["fenix_slice_start"], con_dict["fenix_slice_stop"]
    else:
        start, stop = con_dict["default_slice_start"], con_dict["default_slice_stop"]
    print(slice(start, stop, None))
    return slice(start, stop, None)

def read_envi_header(file):
    """
    Reads an envi header file
    """
    return envi.read_envi_header(file)

def parse_lumo_metadata(xml_file):
    """
    Parser specifically for metadata produced using the Specim Lumo 
    aquisition system
    """
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


#========= library spectra helper methods =====================================

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


#======== Visualisation helpers =========================================================
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


def index_to_rgb(index_2d: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Convert an indexed mineral map + optional mask into an RGB image.

    Parameters
    ----------
    index_2d : (H, W) integer array
        Class indices. Negative values are treated as background.
    mask : (H, W) bool or 0/1 array, optional
        Additional mask. True/1 = masked (drawn as background).

    Returns
    -------
    rgb8 : (H, W, 3) uint8
        Color image ready for PIL / imshow.
    """
    idx = np.asarray(index_2d)
    if idx.ndim != 2:
        raise ValueError(f"index_to_rgb expects a 2-D index map; got {idx.shape}")

    H, W = idx.shape
    if H == 0 or W == 0:
        raise ValueError("index_to_rgb got zero-sized index map.")

    # ---- derive K purely from data (non-negative indices)
    positive = idx[idx >= 0]
    if positive.size == 0:
        # nothing valid, return black
        return np.zeros((H, W, 3), dtype=np.uint8)

    max_idx = int(positive.max())
    K = max_idx + 1

    # ---- deterministic colors from tab20, wrapping every 20 classes
    cmap = matplotlib.colormaps["tab20"]
    colors_rgb = (np.array([cmap(i % 20)[:3] for i in range(K)]) * 255).astype(np.uint8)  # (K,3)

    # ---- build RGB image with background/negatives + mask
    idx_img = idx.copy()
    neg_mask = idx_img < 0

    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (H, W):
            raise ValueError(f"Mask shape {m.shape} does not match index map {idx.shape}.")
        neg_mask |= m.astype(bool)

    idx_img = np.clip(idx_img, 0, K - 1)
    rgb = colors_rgb[idx_img]  # (H,W,3), uint8

    # paint background+masked pixels black
    if neg_mask.any():
        rgb[neg_mask] = np.array([0, 0, 0], dtype=np.uint8)

    return rgb


def mk_thumb(
    arr,
    baseheight: int = 90,
    basewidth: int = 800,
    mask: np.ndarray | None = None,
    index_mode: bool = False,
    resize: bool = True
):
    """
    Create a PIL thumbnail image from an array.

    Parameters
    ----------
    arr : np.ndarray
        Shape (H, W, B), (H, W) or (H, W, 3).
        - If index_mode=False: numeric image or cube.
        - If index_mode=True: 2D integer index map (negative = background).
    baseheight : int
        Max height of the thumbnail (pixels).
    basewidth : int
        Max width of the thumbnail (pixels).
    mask : np.ndarray[bool] or 0/1, optional
        Boolean mask of shape (H, W). True/1 = masked (black).
    index_mode : bool, optional
        If True, treat arr as an indexed mineral map and use tab20 colors
        (via index_to_rgb), instead of colormap/false-colour.

    Returns
    -------
    PIL.Image.Image
        RGB thumbnail image, ready to save as JPEG.
    """
    
   

    # ---- to ndarray + sanity checks
    arr = np.asarray(arr)
    
    

    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported array shape {arr.shape}; expected 2D or 3D.")
        
    if 0 in arr.shape:
        
        raise ValueError(f"arr shape {arr.shape} cannot have a zero size dim")

    # ---- mask validation
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:2]:
            
            raise ValueError(
                f"Mask shape {mask.shape} does not match array spatial shape {arr.shape[:2]}."
            )
    

    # ---- orientation flip
    if arr.shape[0] > arr.shape[1]:
        arr = np.flip(np.swapaxes(arr, 0, 1), axis=0)
        if mask is not None:
            mask = np.flip(np.swapaxes(mask, 0, 1), axis=0)
    

    # ------------------------------------------------------------------
    # 1) INDEX MODE: use classification colour map (tab20)
    # ------------------------------------------------------------------
    if index_mode:
        if arr.ndim != 2:
            raise ValueError(
                f"index_mode=True requires a 2-D index map; got shape {arr.shape}"
            )

        rgb8 = index_to_rgb(arr, mask=mask)
        

    # ------------------------------------------------------------------
    # 2) NORMAL MODE: original mk_thumb behaviour
    # ------------------------------------------------------------------
    else:
        if arr.ndim == 2:
            # 2D → colormap
            if mask is not None:
                a = np.ma.masked_array(arr, mask = mask).astype(float)
            else:
                a = np.ma.array(arr, dtype=float)
            

            amin = np.nanmin(a)
            amax = np.nanmax(a)
            

            if amax > amin:
                norm = (a - amin) / (amax - amin)
            else:
                norm = np.zeros_like(a, dtype=float)
            
            norm = np.ma.array(norm, mask=a.mask)
            rgb = my_map(norm)[..., :3]
            

            rgb8 = np.nan_to_num(
                rgb * 255.0,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            ).astype(np.uint8)
            

        else:
            # 3D
            H, W, C = arr.shape

            if C > 3:
                # hyperspectral false-colour conversion
                fc = get_false_colour(arr)
                

                fc = np.asarray(fc)
                

                if fc.ndim != 3 or fc.shape[2] != 3:
                    raise ValueError("get_false_colour must return (H, W, 3) array.")

                if np.issubdtype(fc.dtype, np.integer):
                    rgb8 = np.clip(fc, 0, 255).astype(np.uint8)
                    
                else:
                    vmin = np.nanmin(fc)
                    vmax = np.nanmax(fc)
                    

                    if vmax > vmin:
                        rgb = (fc - vmin) / (vmax - vmin)
                    else:
                        rgb = np.zeros_like(fc, dtype=float)
                    

                    rgb8 = np.nan_to_num(
                        rgb * 255.0,
                        nan=0.0,
                        posinf=255.0,
                        neginf=0.0,
                    ).astype(np.uint8)
                    

            else:
                # C == 1 or C == 3
                a = arr

                if C == 1:
                    a = np.repeat(a, 3, axis=2)
                    

                if np.issubdtype(a.dtype, np.integer):
                    rgb8 = np.clip(a, 0, 255).astype(np.uint8)
                    
                else:
                    vmin = np.nanmin(a)
                    vmax = np.nanmax(a)
                    

                    if vmax > vmin:
                        rgb = (a - vmin) / (vmax - vmin)
                    else:
                        rgb = np.zeros_like(a, dtype=float)
                    

                    rgb8 = np.nan_to_num(
                        rgb * 255.0,
                        nan=0.0,
                        posinf=255.0,
                        neginf=0.0,
                    ).astype(np.uint8)
                    

        # ---- apply mask (normal mode only; index_mode already handled it)
        if mask is not None:
            rgb8[mask] = 0
            

    # ---- final resize (PIL, as in original)
    h, w = rgb8.shape[:2]
    scale = min(basewidth / float(w), baseheight / float(h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = Image.fromarray(rgb8, mode="RGB")
    

    if (new_w, new_h) != (w, h) and resize:
        im = im.resize((new_w, new_h), Image.LANCZOS)
    

    
    return im


#======== Preprocessing corrections ===========================================

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


# TODO: When the separate QAQC GUIS get integrated, they can call this func with QAQC=True to check SNR
def find_snr_and_reflect(header_path, white_path, dark_path, QAQC=False,
                         data_data_path = None,
                         white_data_path = None,
                         dark_data_path = None): 
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
        
    optional data path arguments, for mangled datasets with inconsistent file names.

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

    if data_data_path:
        box = envi.open(header_path, image=data_data_path)
    else:
        box = envi.open(header_path)

    if white_data_path:
        white_ref = envi.open(white_path, image=white_data_path)
    else:
        white_ref = envi.open(white_path)

    if dark_data_path:
        dark_ref = envi.open(dark_path, image=dark_data_path)
    else:
        dark_ref = envi.open(dark_path)

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


def get_fenix_reflectance(path):

    hyimg = HyliteFenix.correct_folder(str(path))
    
    if isinstance(hyimg, tuple):
        hyimg = hyimg[0]
    
    reflectance = hyimg.data          # (H, W, B), float32
    bands       = hyimg.get_wavelengths()
    snr         = None # snr workflows not implemented yet
    band_slice = _slice_from_sensor("FENIX Sensor")
    reflectance = np.rot90(reflectance, 2)
    return reflectance[:,:, band_slice]*100, bands[band_slice], snr




#================= Actual processing funcs=====================================

def process(cube):
    """
    Perform Savitzky-Golay smoothing and continuum removal on reflectance data
    and return the products with blank mask
    """
    win = int(con_dict["savgol_window"])
    poly = int(con_dict["savgol_polyorder"])
    savgol = sc.signal.savgol_filter(cube, win, poly)
    savgol_cr = remove_hull(savgol)
    mask = np.zeros((cube.shape[0], cube.shape[1]))
    return savgol, savgol_cr, mask

#=============== Mask tooling =================================================

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


def despeckle_mask(mask):
    """
    Remove small speckles by operating on the inverted mask.
    mask: boolean array
    """
    mask_bool = mask.astype(bool)
    inv = ~mask_bool
    bw = inv.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area = 50 # remove only small pixel speckles 
    clean = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    clean_bool = ~(clean.astype(bool))
    
    return clean_bool.astype(np.uint8)


#============= The app auto-crop method (until something better!)==============

def detect_slice_rectangles_robust(
    image,                  #numpy array uint8, 3-channel (H, W, 3)
    min_area_frac=0.0005,     # min polygon area as a fraction of image area
    canny_sigma=0.33,         # auto-Canny thresholds from median
    approx_eps_frac=0.02,     # polygon approximation tolerance
    close_kernel=5,           # morphological close kernel size (pixels)
    use_otsu=True,            # binarize before Canny (good for UI/scans)
    allow_rotated=True        # also accept minAreaRect boxes
):
    """
    Detect and crop the largest rectangular slice/region in an RGB image using
    robust edge-based contour extraction.

    The function:
    1. Converts the image to grayscale and denoises via bilateral filtering.
    2. Applies either Otsu thresholding or raw grayscale input to Canny edge
       detection with adaptive high/low thresholds.
    3. Uses morphological closing to join fragmented edges (optional).
    4. Extracts contours and retains only convex quadrilaterals or rotated
       rectangles above a minimum area threshold.
    5. Performs simple non-maximum suppression (NMS) using IoU to remove
       overlapping rectangles.
    6. Selects the largest remaining rectangle, computes a crop slice, and
       returns the cropped region.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), dtype uint8
        Input BGR/RGB image (only shape and intensity matter). Should represent
        a core scan, object, or similar high-contrast rectangular target.

    min_area_frac : float, optional
        Minimum rectangle area as a fraction of total image area. Defaults to
        0.0005.

    canny_sigma : float, optional
        Sigma factor for adaptive Canny thresholds computed from the median
        intensity of the source. Default is 0.33.

    approx_eps_frac : float, optional
        Polygon approximation tolerance as a fraction of contour perimeter
        (used in `cv2.approxPolyDP`). Default is 0.02.

    close_kernel : int, optional
        Kernel size (pixels) for morphological closing to join edges. Set to 0
        or 1 to disable. Default is 5.

    use_otsu : bool, optional
        If True, apply Otsu binarization before Canny. Often more robust for
        segmentation of UI scans or uneven illumination. Default is True.

    allow_rotated : bool, optional
        If True, fallback to rotated bounding boxes (`cv2.minAreaRect`) when
        convex quadrilaterals are not found. Default is True.

    Returns
    -------
    cropped : np.ndarray
        Cropped image containing the selected rectangle.

    crop_slice : tuple of slice
        Tuple `(slice(rows), slice(cols))` that can be used to apply the same
        crop to other arrays (e.g. hyperspectral cubes).

    Notes
    -----
    - IoU threshold for NMS is fixed at 0.6 to discard heavily overlapping
      candidate rectangles.
    - The returned crop is always axis-aligned even when `allow_rotated=True`.
    - Bilateral filtering is used to preserve edge structure while reducing
      noise for more stable polygon detection.
    """
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
    crop_slice = (slice(y, y + h), slice(x, x + w))
    return image[crop_slice], crop_slice

#=============== Unwrapping (mostly using cv connected components) ============

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

    inv_mask = 1-mask.astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    erod_im = cv2.erode(inv_mask, kernel, anchor=(0, 0), iterations=iters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erod_im.astype(np.uint8), connectivity=8)
    return labels, stats

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

#====== Functions for working with unwrapped datasets =========================
def compute_downhole_mineral_fractions(
    index_map: np.ndarray,
    mask: np.ndarray,
    legend: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute row-based mineral fractions and dominant mineral per row.

    - Uses mask to exclude non-core pixels (mask==1).
    - Fractions are normalised over *core* pixels only in each row.
    - Columns 0..K-1 correspond to legend entries (in order).
    - Column K is 'unclassified': core pixels whose index is not in legend.
    - dominant[i] is index into legend (0..K-1), or -1 if no classified pixels.
    
    - Legend must be an ordered list of dictionaries ["index" : idx,
                                                      "label" : "label text"]
    
    
    """
    if index_map.ndim != 2:
        raise ValueError(f"index_map must be 2D, got {index_map.shape}")
    if mask.shape != index_map.shape:
        raise ValueError("mask and index_map must have the same shape")

    idx = np.asarray(index_map, dtype=int)
    msk = np.asarray(mask, dtype=bool)
    H, W = idx.shape

    class_ids = np.array([row["index"] for row in legend], dtype=int)
    K = len(class_ids)

    fractions = np.zeros((H, K + 1), dtype=float)
    dominant = np.full(H, -1, dtype=int)

    for i in range(H):
        row = idx[i]
        row_mask = msk[i]

        # core pixels only: not masked, and index >= 0
        valid_mask = (~row_mask) & (row >= 0)
        if not np.any(valid_mask):
            continue  # leave zeros; dominant[i] stays -1

        valid_vals = row[valid_mask]
        total_valid = valid_vals.size

        # Count *all* core values in this row
        max_val = int(valid_vals.max())
        counts_all = np.bincount(valid_vals, minlength=max_val + 1)

        # Extract counts for legend classes in legend order
        counts = np.zeros(K, dtype=float)
        for j, cid in enumerate(class_ids):
            if 0 <= cid < counts_all.size:
                counts[j] = counts_all[cid]

        total_classified = counts.sum()
        unclassified = total_valid - total_classified

        # Fractions over core width
        fractions[i, :K] = counts / total_valid
        fractions[i, K] = unclassified / total_valid

        if total_classified > 0:
            dominant[i] = int(np.argmax(fractions[i, :K]))
        else:
            dominant[i] = -1

    return fractions, dominant



#========= Functions for interpreting reflectance data ========================
##        Functions currently used in the app =================================


def mineral_map_subrange(cube: np.ndarray,            # (H, W, B_data)
    exemplar_stack: np.ndarray,       # (K, B_lib)
    wl_data: np.ndarray,         # (B_data,)
    ranges: list[tuple[float, float]],  # [(wmin, wmax), ...]
    mode: str = "pearson",       # "pearson", "sam", "msam"
    invalid_value: int = -999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    selected-range winner-takes-all mineral map.
    """
    try:
        if ranges[1]>ranges[0]:
            start = np.argmin(np.abs(wl_data - ranges[0]))
            stop = np.argmin(np.abs(wl_data - ranges[1]))
        else:
            start = np.argmin(np.abs(wl_data - ranges[1]))
            stop = np.argmin(np.abs(wl_data - ranges[0]))
        cube = cube[..., start:stop]
        exemplar_stack = exemplar_stack[..., start:stop]
    except IndexError:
        raise ValueError("range selection failed on this data")
    if mode=='pearson' :  
        index, confidence = mineral_map_wta_strict(cube, exemplar_stack)
    elif mode == "msam":
        index, confidence = mineral_map_wta_sam_strict(cube, exemplar_stack)
    elif mode == "sam":
        index, confidence = mineral_map_wta_msam_strict(cube, exemplar_stack)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'pearson', 'sam' or 'msam'")
    return index, confidence

def mineral_map_multirange(
    cube: np.ndarray,            # (H, W, B_data)
    exemplars: np.ndarray,       # (K, B_lib)
    wl_data: np.ndarray,         # (B_data,)
    #windows: list[tuple[float, float]],  # [(wmin, wmax), ...]
    mode: str = "pearson",       # "pearson", "sam", "msam"
    invalid_value: int = -999,
) -> tuple[np.ndarray, np.ndarray]:
    windows = [
        (1350, 1500),  # 1.4 µm OH / hydration
        (1850, 2000),  # 1.9 µm H2O
        (2140, 2230),  # Al-OH clays / micas
        (2230, 2320),  # Mg/Fe-OH, chlorite/epidote/amphiboles
        (2305, 2500),  # carbonates + Mg/Fe-OH long-λ structure
    ]
    """
    Multi-range winner-takes-all mineral map.

    For each wavelength window:
      - resample library to instrument wavelengths (once),
      - slice cube & library to that window,
      - continuum-remove both,
      - run the chosen minmap_wta_*_strict,
      - keep the best score across windows per pixel.

    The definition of "best" depends on the mode:

      - pearson, msam: larger score = better match
      - sam: smaller angle (degrees) = better match

    Returns
    -------
    best_idx : (H, W) int
        Index into exemplar stack, or invalid_value where no match.
    best_score : (H, W) float
        Score of the winning match in the metric's natural units:
          - pearson: correlation coefficient
          - msam   : 0..1 (1 = perfect)
          - sam    : angle in degrees (0 = perfect)
    best_window : (H, W) int
        Index of the window that produced the winning match
        (0..len(windows)-1), or -1 where no match.
    """
    H, W, B = cube.shape

    # --- Choose underlying WTA function and comparison semantics ---
    if mode == "pearson":
        wta = mineral_map_wta_strict
        # For Pearson, larger score is better, so initialise best_score very low.
        best_score = np.full((H, W), -np.inf, dtype=np.float32)

        def is_better(score_w: np.ndarray, best_score: np.ndarray) -> np.ndarray:
            """Return mask where new score is better (Pearson: larger is better)."""
            return score_w > best_score

    elif mode == "msam":
        wta = mineral_map_wta_msam_strict
        # For MSAM, larger score is better (1 = perfect), same as Pearson.
        best_score = np.full((H, W), -np.inf, dtype=np.float32)

        def is_better(score_w: np.ndarray, best_score: np.ndarray) -> np.ndarray:
            """Return mask where new score is better (MSAM: larger is better)."""
            return score_w > best_score

    elif mode == "sam":
        wta = mineral_map_wta_sam_strict
        # For SAM, smaller angle (degrees) is better, so initialise best_score very high.
        best_score = np.full((H, W), np.inf, dtype=np.float32)

        def is_better(score_w: np.ndarray, best_score: np.ndarray) -> np.ndarray:
            """Return mask where new score is better (SAM: smaller angle is better)."""
            return score_w < best_score

    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'pearson', 'sam' or 'msam'")

    # --- Output arrays: index + winning window ---
    best_idx = np.full((H, W), invalid_value, dtype=np.int32)
    best_window = np.full((H, W), -1, dtype=np.int16)

    # --- Resample library to instrument wavelengths once ---
    
    # ex_resampled: (K, B_data)

    # --- Loop over wavelength windows ---
    for w_idx, (wmin, wmax) in enumerate(windows):

        # Select bands in this window
        band_mask = (wl_data >= wmin) & (wl_data <= wmax)
        if not np.any(band_mask):
            # No bands in this range for this instrument; skip.
            continue

        cube_slice = cube[:, :, band_mask]          # (H, W, Bw)
        ex_slice = exemplars[:, band_mask]       # (K, Bw)

        cube_cr = cr(cube_slice)
        ex_cr = cr(ex_slice)

        # Run chosen strict WTA matcher on this window
        idx_w, score_w = wta(
            cube_cr,
            ex_cr,
            )

        # Mask of pixels that have a valid match in this window
        valid = idx_w != invalid_value
        # Optional safety: ignore NaNs in score_w
        valid &= np.isfinite(score_w)

        # Decide where this window's match is better than the current best
        better = valid & is_better(score_w, best_score)

        # Update winners
        best_idx[better] = idx_w[better]
        best_score[better] = score_w[better]
        best_window[better] = w_idx

    return best_idx, best_score, best_window





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


def mineral_map_wta_sam_strict(data, exemplar_stack, max_angle_deg=8.0, invalid_value=-999):
    """
    Winner-takes-all Spectral Angle Mapper (SAM).

    Parameters
    ----------
    data : np.ndarray
        Hyperspectral cube, shape (H, W, B), float.
    exemplar_stack : np.ndarray
        Library spectra, shape (K, B), float.
    max_angle_deg : float
        Maximum allowed SAM angle (in degrees). Pixels with best_angle > max_angle_deg
        are set to invalid_value in the class map.
    invalid_value : int
        Fill value for invalid / no-match pixels in the class map.

    Returns
    -------
    class_idx : np.ndarray, int32, shape (H, W)
        Winner-takes-all class index for each pixel. invalid_value where no valid match.
    best_angle : np.ndarray, float32, shape (H, W)
        SAM angle (degrees) of the winning class. Smaller = better match, NaN where undefined.
    """
    H, W, B = data.shape
   
    # Flatten pixels to (N, B)
    X = data.reshape(-1, B).astype(np.float64)   # (N, B)
    E = exemplar_stack.astype(np.float64)        # (K, B)

    # L2 norms
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)  # (N, 1)
    E_norm = np.linalg.norm(E, axis=1, keepdims=True)  # (K, 1)

    # Zero-norm masks (angle undefined)
    X_zero = (X_norm == 0)   # (N, 1)
    E_zero = (E_norm == 0)   # (K, 1)

    # Normalise; where norm==0, divide by 1 and mask later
    Xn = X / np.where(X_norm == 0, 1.0, X_norm)
    En = E / np.where(E_norm == 0, 1.0, E_norm)

    # Cosine similarity matrix: (N, K)
    cos_sim = Xn @ En.T

    # Inject NaNs where angle is undefined (zero vector)
    if X_zero.any():
        cos_sim[X_zero[:, 0], :] = np.nan
    if E_zero.any():
        cos_sim[:, E_zero[:, 0]] = np.nan

    # Clip to valid domain for arccos
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Best (maximum) cosine per pixel – equivalent to minimum angle
    best_cos = np.nanmax(cos_sim, axis=1)    # (N,)

    # Handle rows that are all-NaN
    all_nan = np.isnan(best_cos)
    idx = np.empty_like(best_cos, dtype=np.int32)
    if (~all_nan).any():
        idx[~all_nan] = np.nanargmax(cos_sim[~all_nan], axis=1).astype(np.int32)
    if all_nan.any():
        idx[all_nan] = invalid_value

    # Convert best cosine similarity to angle in degrees
    best_angle = np.empty_like(best_cos, dtype=np.float64)
    valid = ~all_nan
    if valid.any():
        best_angle[valid] = np.degrees(np.arccos(best_cos[valid]))
    if all_nan.any():
        best_angle[all_nan] = np.nan

    # Apply angle threshold: keep if angle <= max_angle_deg
    keep = (best_angle <= float(max_angle_deg))
    # Also drop NaNs
    keep &= ~np.isnan(best_angle)

    idx = np.where(keep, idx, invalid_value).astype(np.int32)

    return idx.reshape(H, W), best_angle.reshape(H, W).astype(np.float32)


def mineral_map_wta_msam_strict(data, members, thresh=0.70, invalid_value=-999):
    """
    Modified after Spectral Python package MSAM algorithm.
    Winner-takes-all classification using Modified SAM (MSAM) scores
    following Oshigami et al. (2013).

    Parameters
    ----------
    data : np.ndarray
        Hyperspectral cube, shape (H, W, B), float-like.
    members : np.ndarray
        Library spectra / endmembers, shape (K, B), float-like.
    thresh : float
        Minimum MSAM score to accept a match. Pixels with best_score <= thresh
        are assigned `invalid_value` in the class map.
        MSAM score is in [0, 1], with 1 = perfect match (zero angle).
    invalid_value : int
        Fill value for pixels with no valid match (or undefined score).

    Returns
    -------
    class_idx : np.ndarray, int32, shape (H, W)
        Winner-takes-all class index for each pixel. `invalid_value` where no
        valid match.
    best_score : np.ndarray, float32, shape (H, W)
        MSAM score of the winning class. Range [0, 1]; NaN where undefined.
    """
    H, W, B = data.shape
    
    assert members.shape[1] == B, "Matrix dimensions are not aligned."

    # Flatten pixels: (N, B)
    X = data.reshape(-1, B).astype(np.float64)     # (N, B)
    M = members.astype(np.float64)                 # (K, B)

    # --- Normalise endmembers (demean + unit length) ---
    M_mean = M.mean(axis=1, keepdims=True)         # (K, 1)
    M_demean = M - M_mean                          # (K, B)
    M_norm = np.linalg.norm(M_demean, axis=1, keepdims=True)  # (K, 1)
    M_zero = (M_norm == 0)                         # (K, 1)
    M_unit = M_demean / np.where(M_norm == 0, 1.0, M_norm)    # (K, B)

    # --- Normalise pixels (demean + unit length) ---
    X_mean = X.mean(axis=1, keepdims=True)         # (N, 1)
    X_demean = X - X_mean                          # (N, B)
    X_norm = np.linalg.norm(X_demean, axis=1, keepdims=True)  # (N, 1)
    X_zero = (X_norm == 0)                         # (N, 1)
    X_unit = X_demean / np.where(X_norm == 0, 1.0, X_norm)    # (N, B)

    # --- Cosine similarity (after MSAM normalisation) ---
    # (N, K) matrix of dot products between pixel and each member
    cos_sim = X_unit @ M_unit.T                    # (N, K)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    # Inject NaNs where MSAM is undefined (zero norm vectors)
    if X_zero.any():
        cos_sim[X_zero[:, 0], :] = np.nan
    if M_zero.any():
        cos_sim[:, M_zero[:, 0]] = np.nan

    # --- MSAM score: 1 - (angle / (pi/2)), so 1 = perfect match ---
    # angle = arccos(cos_sim) in [0, pi]
    angle = np.arccos(cos_sim)                     # (N, K), radians
    msam_score = 1.0 - (angle / (np.pi / 2.0))     # (N, K)
    # For invalid (NaN cos_sim), msam_score stays NaN

    # --- Winner-takes-all selection ---
    best_score = np.nanmax(msam_score, axis=1)     # (N,)
    all_nan = np.isnan(best_score)

    idx = np.empty_like(best_score, dtype=np.int32)
    if (~all_nan).any():
        idx[~all_nan] = np.nanargmax(msam_score[~all_nan], axis=1).astype(np.int32)
    if all_nan.any():
        idx[all_nan] = invalid_value

    # --- Thresholding in MSAM space ---
    # Keep only pixels with score > thresh, like your Pearson WTA.
    keep = (best_score > float(thresh)) & (~np.isnan(best_score))
    idx = np.where(keep, idx, invalid_value).astype(np.int32)

    return idx.reshape(H, W), best_score.reshape(H, W).astype(np.float32)

# ==== Clustering functions ===================================================
def kmeans_spectral_wrapper(data, clusters, iters):
    """
    Run k-means clustering on spectral data using Spectral Python (SPy).

    This wraps :func:`spectral.kmeans` for convenience and returns both the
    cluster map and cluster centers.
    """
    
    m, c = sp.kmeans(data, clusters, iters)
    return m, c

# ==== minimim wavelenth mapping =============================================

def Combined_MWL(savgol, savgol_cr, mask, bands, feature, technique = 'QUAD', use_width=False):
    """
    Estimate minimum wavelength (MWL) position and corresponding absorption depth
    for a specified short-wave infrared absorption feature using multiple
    possible fitting techniques.
 
    This function:
    1) Looks up feature-specific wavelength windows from a pre-defined dictionary
       (e.g. '2200W', '2320W', '4000W', etc.).
       Dict is derived from values in:
           
       Laukamp, C., Rodger, A., LeGras, M., Lampinen, H., Lau, I. C., 
       Pejcic, B., Stromberg, J., Francis, N., & Ramanaidou, E. (2021). 
       Mineral physicochemistry underlying feature-based extraction of 
       mineral abundance and composition from shortwave, mid and thermal 
       infrared reflectance spectra. Minerals, 11(4), 347. 
       https://doi.org/10.3390/min11040347
       
    2) Computes a preliminary peak detection response using a SciPy-based peak
       finder (`est_peaks_cube_scipy`) to identify invalid pixels.
    3) Applies one of several minimum-wavelength estimation methods to each
       pixel spectrum:
         - ``'QND'`` – a quick non-derivative method using continuum removal +
           argmin (no fit).
         - ``'POLY'`` – polynomial fitting (via `hylite.analyse.minimum_wavelength`).
         - ``'GAUS'`` – Gaussian fitting.
         - ``'QUAD'`` – quadratic fitting (default; recommended).
    4) Returns pixel-wise MWL position (nm), absorption depth, and an updated mask
       where poorly-defined pixels are flagged.
 
    Parameters
    ----------
    savgol : np.ndarray, shape (M, N, B)
        Smoothed reflectance spectra (e.g. Savitzky-Golay filtered). Used for
        model fitting in the chosen technique.
 
    savgol_cr : np.ndarray, shape (M, N, B)
        Continuum-removed version of ``savgol``. Used to determine peak response
        and for the ``'QND'`` method.
 
    mask : np.ndarray[bool], shape (M, N)
        Boolean mask of invalid pixels (True = masked). This is copied internally
        and updated based on peak detection failure.
 
    bands : np.ndarray, shape (B,)
        Wavelength values (in nm) corresponding to the last axis of ``savgol``.
 
    feature : str
        Key specifying the target absorption feature. Must exist in the internal
        `feats` dictionary (e.g. '2200W', '2320W', '4000W', etc.).
 
    technique : {'QND', 'POLY', 'GAUS', 'QUAD'}, optional
        Minimum-wavelength fitting method to use. Default is ``'QUAD'``.
            * ``'QND'`` – fast, no fitting.
            * ``'POLY'`` – polynomial fit.
            * ``'GAUS'`` – Gaussian fit.
            * ``'QUAD'`` – quadratic fit (default).
 
    thresh : taken from the configuration dictionary, default is 0.7
        Minimum absorption depth threshold used by some alternative masking
        options. Currently applied in the returned mask.
    
    use_width : Boolean
        Experimental gating of valid features. Off by default until thoroughly tested
 
    Returns
    -------
    position : np.ndarray, shape (M, N)
        Estimated minimum wavelength position (in nm) for the selected feature.
 
    depth : np.ndarray, shape (M, N)
        Estimated absorption depth (unitless), method-dependent.
 
    feature_mask : np.ndarray[bool], shape (M, N)
        Updated mask where invalid/poorly detected pixels are True.
 
    Notes
    -----
    - Pixel validity is first checked using `est_peaks_cube_scipy` on the
      continuum-removed cube. Failure yields masked pixels.
    - The `'POLY'`, `'GAUS'`, and `'QUAD'`` methods wrap
      :func:`hylite.analyse.minimum_wavelength`.
    - The `'QND'`` method uses a coarse argmin over continuum-removed spectra
      without fitting; depth is computed as ``1 - min(cr)``.
    """
    thresh = con_dict["feature detection threshold"]
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
    width_props = {"2080W": {
        "label":         "clay / OH",
        "depth_factor":  1.0,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    },
    "2160W": {
        "label":         "clay / OH",
        "depth_factor":  1.0,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    },
    "2200W": {
        "label":         "Al–OH",
        "depth_factor":  1.0,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  None,
    },
    "2250W": {
        "label":         "Al–OH / Mg–OH",
        "depth_factor":  1.0,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    },
    "2290W": {
        "label":         "Mg–Fe–OH",
        "depth_factor":  1.0,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    },
    "2320W": {
        "label":         "Mg–Fe–OH",
        "depth_factor":  1.0,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    },
    "2350W": {
        "label":         "carbonate / OH",
        "depth_factor":  1.1,   # slightly stricter
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    },
    "2390W": {
        "label":         "carbonate",
        "depth_factor":  1.1,
        "use_width":     True,
        "width_min_nm":  8.0,
        "width_max_nm":  80.0,
    }}
    cr_crop_min = feats[feature][2]
    cr_crop_max = feats[feature][3]
    cr_crop_min_index = np.argmin(np.abs(np.array(bands)-(feats[feature][2])))
    cr_crop_max_index = np.argmin(np.abs(np.array(bands)-(feats[feature][3])))
    wav_min = feats[feature][0]
    wav_max = feats[feature][1]
    wav_min_index = np.argmin(np.abs(np.array(bands)-(feats[feature][0])))
    wav_max_index = np.argmin(np.abs(np.array(bands)-(feats[feature][1])))

    #check_response =  est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min, wav_max))
    check_response =  est_peaks_cube_scipy_thresh(savgol_cr, bands, wavrange=(wav_min, wav_max), thresh = thresh)

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
    feature_mask[position>wav_max] = 1
    feature_mask[position<wav_min] = 1
    if thresh:
        feature_mask[depth<thresh] = 1
        
    # Experimental non-feature masking. Off by default. To use in GUI, hack the default parameter
    if use_width and technique.upper() in ('POLY', 'GAUS', 'QUAD') and feature in width_props:
        wmin = width_props[feature]["width_min_nm"]
        wmax = width_props[feature]["width_max_nm"]

        if wmin is not None:
            feature_mask[width < wmin] = 1
        if wmax is not None:
            feature_mask[width > wmax] = 1
    

    return position, depth, feature_mask


def est_peaks_cube_scipy(data, bands, wavrange=(2300, 2340)):
    """
    Detect spectral absorption peaks within a specified wavelength range
    for every pixel in a hyperspectral cube using SciPy peak detection.

    For each pixel spectrum, local maxima of ``(1 - reflectance)`` are
    identified using :func:`scipy.signal.find_peaks`. The first peak whose
    wavelength lies inside ``wavrange`` is returned; if no such peak is found,
    a sentinel value of ``-999`` is assigned.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral image or array of continuum-removed or inverted
        reflectance values. The last axis must represent spectral bands.

    bands : np.ndarray, shape (B,)
        Wavelength values (in nm) corresponding to the third axis of ``data``.

    wavrange : tuple(float, float), optional
        Minimum and maximum wavelength (in nm) to search for peaks.
        Default is ``(2300, 2340)``.

    Returns
    -------
    arr : np.ndarray, shape (H, W)
        Estimated peak wavelength for each pixel. If no valid peak is found
        in the target range, the value is ``-999``.

    Notes
    -----
    - Peaks are detected on ``1 - data[i, j]``, making absorption dips into
      positive peaks.
    - Only the *first* valid peak inside ``wavrange`` is reported per pixel.
    - No thresholding by peak height or prominence is applied; downstream
      filtering may be required for reliability.
    """
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







#=============Functions I dont think are actually called=======================

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
            if not y:
                coeffs[i,j] = x
            else:
                coeffs[i,j] = 0
    return coeffs



def est_peaks_cube_scipy_thresh(data, bands, wavrange=(2300, 2340), thresh = 0.3):
    """
    Detect spectral absorption peaks within a wavelength window using SciPy,
    applying a minimum peak height (depth) threshold per pixel.

    For each pixel spectrum, peaks of ``(1 - reflectance)`` are detected via
    :func:`scipy.signal.find_peaks`. Only peaks whose amplitude exceeds
    ``thresh`` are retained, and the first peak whose wavelength lies inside
    ``wavrange`` is returned. If no valid peak is found, the value ``-999`` is
    assigned.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral cube or continuum-removed/inverted reflectance array.
        The last axis corresponds to spectral bands.

    bands : np.ndarray, shape (B,)
        Wavelength values (in nm) corresponding to the third axis of ``data``.

    wavrange : tuple(float, float), optional
        Minimum and maximum wavelength (nm) to search for peaks. Default is
        ``(2300, 2340)``.

    thresh : float, optional
        Minimum absorption depth required for a peak to be considered. The
        threshold is applied to the peak height returned by SciPy. Default is
        0.3.

    Returns
    -------
    arr : np.ndarray, shape (H, W)
        Estimated peak wavelength for each pixel. Pixels without a detected
        peak above threshold inside the target range contain ``-999``.

    Notes
    -----
    - Peaks are detected on ``1 - data[i, j]`` (i.e., treating absorption dips
      as positive peaks).
    - Only the first valid peak within ``wavrange`` is reported per pixel.
    - This method ignores peak prominence, width, or signal-to-noise; it only
      filters by amplitude. Post-processing or masking may be required for
      robust interpretation.
    """
    w, l, b = data.shape
    arr = np.full((w,l), -999)
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

def est_peaks_cube_scipy_multi_thresh(
    data,
    bands,
    wavrange=(2300, 2340),
    depth_thresh=0.3,
    prom_thresh=None,
    min_width_nm=10.0,
    max_width_nm=None,
):
    """
    Robust peak detector for use as a 'real feature?' gate.

    Parameters
    ----------
    data : (H, W, B) ndarray
        Continuum-removed reflectance cube (values ~1 with dips).
    bands : (B,) ndarray
        Wavelengths in nm.
    wavrange : (float, float)
        Target feature window [min_nm, max_nm].
    depth_thresh : float
        Minimum feature depth (1 - R_CR at the minimum).
    prom_thresh : float or None
        Minimum prominence. If None, defaults to depth_thresh.
    min_width_nm : float or None
        Minimum allowed feature width (FWHM) in nm.
    max_width_nm : float or None
        Maximum allowed feature width in nm (optional).

    Returns
    -------
    arr : (H, W) ndarray
        For each pixel, the wavelength (nm) of the best peak in wavrange,
        or -999.0 if no acceptable feature was found.
    """
    H, W, B = data.shape
    arr = np.full((H, W), -999.0, dtype=float)

    if prom_thresh is None:
        prom_thresh = depth_thresh

    # Approximate band step (assumed almost regular)
    band_step = float(np.median(np.diff(bands)))

    for i in range(H):
        for j in range(W):
            spec = data[i, j, :]
            y = 1.0 - spec  # turn absorption into positive peaks

            # Get peaks + measurements.
            # height filters by depth; prominence/width=0 just request props.
            peaks, props = sc.signal.find_peaks(
                y,
                height=depth_thresh,
                prominence=0,
                width=0,
            )

            if peaks.size == 0:
                # Leave arr[i,j] = -999.0 (no feature)
                continue

            heights = props["peak_heights"]
            prom    = props.get("prominences", np.zeros_like(heights))
            widths_idx = props.get("widths", np.zeros_like(heights))
            widths_nm  = widths_idx * band_step
            lambdas    = bands[peaks]

            # Basic quality masks
            valid = np.ones_like(heights, dtype=bool)
            valid &= heights >= depth_thresh
            valid &= prom    >= prom_thresh

            if min_width_nm is not None:
                valid &= widths_nm >= min_width_nm
            if max_width_nm is not None:
                valid &= widths_nm <= max_width_nm

            # Restrict to the target wavelength window
            if np.any(valid):
                valid &= (lambdas >= wavrange[0]) & (lambdas <= wavrange[1])

            if not np.any(valid):
                # No valid peak in the window → leave -999
                continue

            # Choose the 'best' peak: highest prominence (or depth if you prefer)
            v_idx = np.where(valid)[0]
            best_local = v_idx[np.argmax(prom[v_idx])]
            arr[i, j] = lambdas[best_local]

    return arr




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


@jit(nopython=True)
def numpy_pearson_stackexemplar_threshed(data, exemplar_stack):
    """
    Classify each pixel spectrum by maximum Pearson correlation against a stack
    of exemplar spectra, applying a fixed correlation threshold.

    For every pixel spectrum in ``data``, Pearson correlation coefficients are
    computed against each exemplar in ``exemplar_stack``. The pixel is assigned
    to the exemplar with the highest coefficient **only if the maximum
    correlation exceeds 0.7**. Otherwise, the pixel is marked as unclassified
    using a sentinel index (``-999``). Correlation scores are returned in a
    separate confidence array.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral data cube (typically reflectance or continuum-removed
        spectra). Pixel spectra are along the last axis.

    exemplar_stack : np.ndarray, shape (N, B)
        Stack of N reference/exemplar spectra to match against. Each spectrum
        must have the same band length as ``data`` along the spectral axis.

    Returns
    -------
    coeffs : np.ndarray, shape (H, W)
        Index of the best-matching exemplar for each pixel. Pixels failing the
        correlation threshold are assigned ``-999``.

    confidence : np.ndarray, shape (H, W)
        Maximum Pearson correlation coefficient for each pixel, regardless of
        whether it passed the threshold.

    Notes
    -----
    - Pearson correlation is computed via ``np.corrcoef`` for each exemplar.
    - Threshold is currently fixed at 0.7; modify in the code for different
      confidence limits.
    - No spectral preprocessing is performed—spectra should ideally be
      normalized or continuum-removed beforehand to avoid bias.
    - Returned ``coeffs`` is an integer index map; it can be used to build
      classification images or masks.
    """
    num = exemplar_stack.shape[0]
    coeffs = np.zeros((data.shape[0], data.shape[1]))
    confidence = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            c_list = np.zeros(num)
            for n in range(num):
                c_list[n] = np.corrcoef(data[i,j], exemplar_stack[n])[1,0]
            if np.max(c_list) > 0.7:
                coeffs[i,j] = np.argmax(c_list)
                confidence[i,j] = np.max(c_list)

            else:
                coeffs[i,j] = -999
                confidence[i,j] = np.max(c_list)
    return coeffs, confidence


# ======== Unsure why these are in here, but leave for now ====================
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


#TODO: This has been replaced by strict version, dont think this is called
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

#======== Author specific functions - unlikely to have wide use

def carbonate_facies(savgol, savgol_cr, mask, bands, technique = 'QUAD'):
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


def carbonate_facies_original(savgol, savgol_cr, mask, bands, technique = 'QUAD'):
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


# Older version of unwrap_from_stats, not mask aware. Dont think it is used in GUI
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








