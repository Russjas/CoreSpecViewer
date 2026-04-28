"""
Visualisation helpers for hyperspectral data in CoreSpecViewer.

Converts hyperspectral cubes and classified index maps into RGB images and
thumbnails for display in the GUI. No processing or analysis logic — pure
array-to-image transforms. No GUI dependencies; all functions return numpy
arrays or PIL Images.

Functions
---------
get_false_colour        Extract a 3-band RGB image from a hyperspectral cube
                        using spectral.get_rgb(). Guarantees finite output by
                        zero-filling NaN and Inf values.
get_false_colour_fast   Lightweight alternative to get_false_colour — directly
                        indexes three bands without the spectral library overhead.
                        Used where performance matters (e.g. live display updates).
derive_display_bands    Return sensible default band indices (first, middle, last)
                        when no display bands are specified.
index_to_rgb            Convert a 2D integer class-index map into a uint8 RGB
                        image using a deterministic tab20 colormap. Handles
                        background, invalid indices, and an optional mask.
mk_thumb                Generate a normalised thumbnail PIL Image from a cube,
                        index map, or masked array. Handles false-colour,
                        index-mode colourisation, optional masking, and resize.
                        Internally uses get_false_colour_fast and index_to_rgb.

Notes
-----
mk_thumb is the single entry point used by ProcessedObject.export_image() for
all thumbnail generation. get_false_colour is used by the live canvas display
and autocrop pipeline.
"""

import logging
import time

import numpy as np
import spectral as sp
from PIL import Image
import matplotlib

logger = logging.getLogger(__name__)

my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')

DISPLAY_RANGE = {
    '1400W':  (1387, 1445),
    '1480W':  (1471, 1491),
    '1550W':  (1520, 1563),
    '1760W':  (1751, 1764),
    '1850W':  (1749, 1949),
    '1900W':  (1840, 1990), 
    '2080W':  (1980, 2180), 
    '2160W':  (2159, 2166),
    '2200W':  (2185, 2215),
    '2250W':  (2248, 2268),
    '2290W':  (2279, 2310), 
    '2320W':  (2300, 2340),
    '2350W':  (2320, 2366),
    '2390W':  (2377, 2406),
    '2830W':  (2677, 2890),  
    '2950W':  (2920, 2980),
    '2950AW': (2900, 2960),
    '2950BW': (2920, 2990),
    '3000W':  (2900, 3100),  
    '3500W':  (3400, 3600),
    '4000W':  (3930, 4150),
    '4000WIDEW': (3910, 4150),
    '4000V_NARROWW': (3930, 4150),
    '4000shortW': (3850, 4000),
    '4470TRUEW': (4460, 4490),
    '4500SW': (4570, 4850),
    '4500CW': (4625, 4770),
    '4670W':  (4300, 4800),
    '4920W':  (4850, 5100),
    
}


def get_false_colour(array, bands=None):
    """
    spectral.get_rgb(), but guarantees finite output by replacing NaN/Inf with 0.
    """
    rgb = sp.get_rgb(array, bands=bands)

    if not np.isfinite(rgb).all():
        logger.debug("Non-finite values in display output; zero-filling")
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0, copy=True)

    return rgb


def get_false_colour_fast(array, bands=None):
    """
    Extract RGB bands from hyperspectral cube.
    Replaces spectral.get_rgb() to avoid performance issues.
    """
    if bands is None:
        # Default: first, middle, last band
        C = array.shape[2]
        bands = [0, C // 2, C - 1]
    
    # Just extract the 3 bands directly
    rgb = array[:, :, bands]
    
    return rgb


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
    
    t0 = time.perf_counter()
    
    # ---- to ndarray + sanity checks
    t1 = time.perf_counter()
    arr = np.asarray(arr)
    logger.debug(f"[{time.perf_counter() - t0:.4f}s] Array conversion (+{time.perf_counter() - t1:.4f}s)")
    
    t1 = time.perf_counter()
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported array shape {arr.shape}; expected 2D or 3D.")
        
    if 0 in arr.shape:
        raise ValueError(f"arr shape {arr.shape} cannot have a zero size dim")
    logger.debug(f"[{time.perf_counter() - t0:.4f}s] Shape validation (+{time.perf_counter() - t1:.4f}s)")

    # ---- mask validation
    t1 = time.perf_counter()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match array spatial shape {arr.shape[:2]}."
            )
    logger.debug(f"[{time.perf_counter() - t0:.4f}s] Mask validation (+{time.perf_counter() - t1:.4f}s)")
    
    # ---- orientation flip
    t1 = time.perf_counter()
    if arr.shape[0] > arr.shape[1]:
        arr = np.flip(np.swapaxes(arr, 0, 1), axis=0)
        if mask is not None:
            mask = np.flip(np.swapaxes(mask, 0, 1), axis=0)
    logger.debug(f"[{time.perf_counter() - t0:.4f}s] Orientation flip (+{time.perf_counter() - t1:.4f}s)")
    
    # ------------------------------------------------------------------
    # 1) INDEX MODE: use classification colour map (tab20)
    # ------------------------------------------------------------------
    if index_mode:
        t1 = time.perf_counter()
        if arr.ndim != 2:
            raise ValueError(
                f"index_mode=True requires a 2-D index map; got shape {arr.shape}"
            )

        rgb8 = index_to_rgb(arr, mask=mask)
        logger.debug(f"[{time.perf_counter() - t0:.4f}s] Index mode: index_to_rgb (+{time.perf_counter() - t1:.4f}s)")
        

    # ------------------------------------------------------------------
    # 2) NORMAL MODE: original mk_thumb behaviour
    # ------------------------------------------------------------------
    else:
        if arr.ndim == 2:
            # 2D → colormap
            t1 = time.perf_counter()
            if mask is not None:
                a = np.ma.masked_array(arr, mask = mask).astype(float)
            else:
                a = np.ma.array(arr, dtype=float)
            logger.debug(f"[{time.perf_counter() - t0:.4f}s] 2D mode: masked array creation (+{time.perf_counter() - t1:.4f}s)")
            
            t1 = time.perf_counter()
            
            # Auto-detect appropriate range based on data
            data_min = np.nanmin(a)
            data_max = np.nanmax(a)
            logger.debug(f"======================== Data min {data_min}, Data Max {data_max}============================")
            logger.debug(f"Number greater than 1: {np.sum(a > 1)}, full size of array: {a.shape[0]*a.shape[1]}")
            # Check if the vast majority of values are between 0 and 1
            if data_min >= 0 and np.sum((a.compressed() >= 0) & (a.compressed() <= 1)) >= 0.95 * a.compressed().size:
                amin, amax = 0.0, 1.0
                logger.debug("Values in [0,1], using 0-1 stretch")
                
            else:
                # Check if majority of values fall within any known feature position range
                in_range = False
                compressed = a.compressed()  # Only non-masked values
                for range_min, range_max in DISPLAY_RANGE.values():
                    valid_data = compressed[(compressed >= range_min) & (compressed <= range_max)]
                    if valid_data.size > 0.7 * compressed.size:  # 70% of valid pixels
                        amin, amax = range_min, range_max
                        logger.debug(f"Majority of data in [{range_min}, {range_max}], global feature stretch")
                        in_range = True
                        break

                if not in_range:
                    # Band maths or other - use local stretch
                    amin, amax = data_min, data_max
                    logger.debug("Using local stretch")
            
            logger.debug(f"[{time.perf_counter() - t0:.4f}s] 2D mode: range determination (+{time.perf_counter() - t1:.4f}s)")
            
            t1 = time.perf_counter()
            if amax > amin:
                norm = np.clip((a - amin) / (amax - amin), 0, 1)
            else:
                norm = np.zeros_like(a, dtype=float)
            logger.debug(f"[{time.perf_counter() - t0:.4f}s] 2D mode: normalization (+{time.perf_counter() - t1:.4f}s)")
            
            t1 = time.perf_counter()
            norm = np.ma.array(norm, mask=a.mask)
            rgb = my_map(norm)[..., :3]
            logger.debug(f"[{time.perf_counter() - t0:.4f}s] 2D mode: colormap application (+{time.perf_counter() - t1:.4f}s)")
            
            t1 = time.perf_counter()
            rgb8 = np.nan_to_num(
                rgb * 255.0,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            ).astype(np.uint8)
            logger.debug(f"[{time.perf_counter() - t0:.4f}s] 2D mode: RGB8 conversion (+{time.perf_counter() - t1:.4f}s)")
            

        else:
            # 3D
            H, W, C = arr.shape
            logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D mode: shape = ({H}, {W}, {C})")

            if C > 3:
                # hyperspectral false-colour conversion
                t1 = time.perf_counter()
                fc = get_false_colour_fast(arr)
                logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: get_false_colour_fast (+{time.perf_counter() - t1:.4f}s)")
                
                t1 = time.perf_counter()
                fc = np.asarray(fc)
                logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: asarray(fc) (+{time.perf_counter() - t1:.4f}s)")
                
                t1 = time.perf_counter()
                if fc.ndim != 3 or fc.shape[2] != 3:
                    raise ValueError("get_false_colour_fast must return (H, W, 3) array.")

                if np.issubdtype(fc.dtype, np.integer):
                    rgb8 = np.clip(fc, 0, 255).astype(np.uint8)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: integer clip/convert (+{time.perf_counter() - t1:.4f}s)")
                    
                else:
                    t2 = time.perf_counter()
                    vmin = np.nanmin(fc)
                    vmax = np.nanmax(fc)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: nanmin/nanmax on fc (+{time.perf_counter() - t2:.4f}s)")
                    
                    t2 = time.perf_counter()
                    if vmax > vmin:
                        rgb = (fc - vmin) / (vmax - vmin)
                    else:
                        rgb = np.zeros_like(fc, dtype=float)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: normalization (+{time.perf_counter() - t2:.4f}s)")
                    
                    t2 = time.perf_counter()
                    rgb8 = np.nan_to_num(
                        rgb * 255.0,
                        nan=0.0,
                        posinf=255.0,
                        neginf=0.0,
                    ).astype(np.uint8)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: nan_to_num + uint8 (+{time.perf_counter() - t2:.4f}s)")
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D hyperspectral: float path total (+{time.perf_counter() - t1:.4f}s)")
                    

            else:
                # C == 1 or C == 3
                t1 = time.perf_counter()
                a = arr

                if C == 1:
                    a = np.repeat(a, 3, axis=2)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D C=1: np.repeat (+{time.perf_counter() - t1:.4f}s)")
                    
                t1 = time.perf_counter()
                if np.issubdtype(a.dtype, np.integer):
                    rgb8 = np.clip(a, 0, 255).astype(np.uint8)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D C<=3: integer clip/convert (+{time.perf_counter() - t1:.4f}s)")
                    
                else:
                    t2 = time.perf_counter()
                    vmin = np.nanmin(a)
                    vmax = np.nanmax(a)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D C<=3: nanmin/nanmax (+{time.perf_counter() - t2:.4f}s)")
                    
                    t2 = time.perf_counter()
                    if vmax > vmin:
                        rgb = (a - vmin) / (vmax - vmin)
                    else:
                        rgb = np.zeros_like(a, dtype=float)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D C<=3: normalization (+{time.perf_counter() - t2:.4f}s)")
                    
                    t2 = time.perf_counter()
                    rgb8 = np.nan_to_num(
                        rgb * 255.0,
                        nan=0.0,
                        posinf=255.0,
                        neginf=0.0,
                    ).astype(np.uint8)
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D C<=3: nan_to_num + uint8 (+{time.perf_counter() - t2:.4f}s)")
                    logger.debug(f"[{time.perf_counter() - t0:.4f}s] 3D C<=3: float path total (+{time.perf_counter() - t1:.4f}s)")
                    

        # ---- apply mask (normal mode only; index_mode already handled it)
        t1 = time.perf_counter()
        if mask is not None:
            rgb8[mask] = 0
        logger.debug(f"[{time.perf_counter() - t0:.4f}s] Apply mask to rgb8 (+{time.perf_counter() - t1:.4f}s)")
            

    # ---- final resize (PIL, as in original)
    t1 = time.perf_counter()
    h, w = rgb8.shape[:2]
    scale = min(basewidth / float(w), baseheight / float(h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = Image.fromarray(rgb8, mode="RGB")
    logger.debug(f"[{time.perf_counter() - t0:.4f}s] PIL Image.fromarray (+{time.perf_counter() - t1:.4f}s)")
    
    t1 = time.perf_counter()
    if (new_w, new_h) != (w, h) and resize:
        im = im.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(f"[{time.perf_counter() - t0:.4f}s] PIL resize ({w}x{h} -> {new_w}x{new_h}) (+{time.perf_counter() - t1:.4f}s)")
    else:
        logger.debug(f"[{time.perf_counter() - t0:.4f}s] No resize needed (resize={resize}, same_size={(new_w, new_h) == (w, h)})")
    
    logger.debug(f"[{time.perf_counter() - t0:.4f}s] ===== TOTAL mk_thumb time (shape={arr.shape}, index_mode={index_mode}, resize={resize}) =====")
    
    return im


