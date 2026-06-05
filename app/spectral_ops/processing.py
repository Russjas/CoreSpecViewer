"""
Core spectral processing operations for CoreSpecViewer.

Pure numerical transforms operating on numpy arrays. No GUI dependencies.
All functions are UI-agnostic and can be used in scripting or batch contexts.

Functions
---------
process             Apply Savitzky-Golay smoothing and continuum removal to a
                    reflectance cube. Returns smoothed cube, continuum-removed
                    cube, and a blank mask. Window and polynomial order are read
                    from AppConfig at call time.
remove_cont         Thin wrapper around gfit continuum removal (remove_hull).
                    Kept here to contain the scientific dependency.
resample_spectrum   1D linear resample of a reference spectrum onto target
                    band centres. Used to align library spectra to sensor bands.
unwrap_from_stats   Unwrap masked core segments into a vertically stacked,
                    width-normalised masked array for downhole analysis.
compute_downhole_mineral_fractions
                    Compute per-row mineral fractions and dominant mineral
                    from a classified index map and mask.

Dependencies
------------
Reads AppConfig for savgol_window and savgol_polyorder.
"""

import logging

from gfit.util import remove_hull
import numpy as np
import scipy as sc

from ..config import config  # mutable module singleton

logger = logging.getLogger(__name__)


def process(cube):
    """
    Perform Savitzky-Golay smoothing and continuum removal on reflectance data
    and return the products with blank mask
    """
    win = config.savgol_window
    poly = config.savgol_polyorder
    savgol = sc.signal.savgol_filter(cube, win, poly)
    savgol_cr = remove_hull(savgol)
    mask = np.zeros((cube.shape[0], cube.shape[1]))
    return savgol, savgol_cr, mask


def remove_cont(spectra):
    '''helper function to keep scientific dependencies contained'''
    return remove_hull(spectra)


def resample_spectrum(x_src_nm: np.ndarray, y_src: np.ndarray, x_tgt_nm: np.ndarray) -> np.ndarray:
    """
    Fast 1D linear resample onto target band centers (nm).
    Clamps to edges; returns finite array (NaNs filled with 0).
    """
    y = np.interp(x_tgt_nm, x_src_nm, y_src, left=y_src[0], right=y_src[-1]).astype(float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def unwrap_from_stats(mask, image, stats, convention = None, MIN_AREA=300, MIN_WIDTH=10):
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
    convention : str or None, optional
        Box reading convention controlling segment sort order.
        'rl_tb': right-to-left, top-to-bottom (default)
        'lr_tb': left-to-right, top-to-bottom
        'rl_bt': right-to-left, bottom-to-top
        'lr_bt': left-to-right, bottom-to-top
        None falls back to 'rl_tb' for backward compatibility.
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
    - Default Sorting is right-to-left (columns) then top-to-bottom (rows),
      with three other conventions to ensure correct depth registration.
    - Padding is symmetric to match the maximum width across segments, applied
      to both data and mask before stacking. 
    """
    _SORT_KEYS = {
        'rl_tb': lambda s: ( round(-s[0][0] / 10),  s[0][1]),
        'lr_tb': lambda s: ( round( s[0][0] / 10),  s[0][1]),
        'rl_bt': lambda s: ( round(-s[0][0] / 10), -s[0][1]),
        'lr_bt': lambda s: ( round( s[0][0] / 10), -s[0][1]),
    }
    sort_key = _SORT_KEYS.get(convention, _SORT_KEYS['rl_tb'])
    full_mask = np.zeros_like(image)
    full_mask[mask==1] = 1
    segments = []

    for i in range(1, stats.shape[0]): # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or w < MIN_WIDTH:
            continue # Skip small regions
        else:
            segment = np.ma.masked_array(image, mask = full_mask)[y:y+h, x:x+w]
            segments.append(((x, y), segment))

    # Sort segments: right to left (x descending), top to bottom (y ascending)
    
    segments_sorted = sorted(segments, key=sort_key)

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


def compute_downhole_mineral_fractions(
    index_map: np.ndarray,
    mask: np.ndarray,
    legend: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute row-based mineral fractions and dominant mineral per row.

    - Uses mask to exclude non-core pixels (mask==1).
    - Fractions are normalised over *core* pixels only in each row (including -999 unclassified).
    - Columns 0..K-1 correspond to legend entries (in order).
    - Column K is 'unclassified': core pixels whose index is not in legend, including -999.
    - dominant[i] is index into legend (0..K-1), or -1 if no classified pixels.
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

    # FIX 1: Initialize with NaN. This prevents 0.0 sums from corrupting the average
    # in the resampling step when a row is entirely a gap.
    fractions = np.full((H, K + 1), np.nan, dtype=float)
    dominant = np.full(H, -1, dtype=int)

    for i in range(H):
        row = idx[i]
        row_mask = msk[i]

        # FIX 2: Include the UNCLASSIFIED index ID (-999) in the valid core count.
        # Invalid value is -999 in all spectral ops, but it is a default argument
        # rather than enforced. No gui operation will pass a different argument.
        # valid_mask = (~row_mask) AND ( (row >= 0) OR (row == -999) )
        valid_mask = (~row_mask) & ((row >= 0) | (row == -999))
        
        if not np.any(valid_mask):
            # If no valid pixels, the row remains NaN (due to FIX 1)
            continue

        valid_vals = row[valid_mask]
        total_valid = valid_vals.size

        # We use only non-negative values for bincount (as is standard)
        positive_vals = valid_vals[valid_vals >= 0]
        max_val = int(positive_vals.max()) if positive_vals.size > 0 else 0
        counts_all = np.bincount(positive_vals, minlength=max_val + 1)

        # Extract counts for legend classes in legend order
        counts = np.zeros(K, dtype=float)
        for j, cid in enumerate(class_ids):
            if 0 <= cid < counts_all.size:
                counts[j] = counts_all[cid]

        total_classified = counts.sum()
        
        # Unclassified count is now correctly calculated as the remainder of 
        # ALL valid core pixels (total_valid) minus those classified by the legend.
        unclassified = total_valid - total_classified

        # Fractions over core width (now total_valid includes -999 pixels)
        fractions[i, :K] = counts / total_valid
        fractions[i, K] = unclassified / total_valid

        if total_classified > 0:
            dominant[i] = int(np.argmax(fractions[i, :K]))
        else:
            dominant[i] = -1

    return fractions, dominant