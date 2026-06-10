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


def _sort_segments_by_runs(segments, convention="rl_tb"):
    """
    Sort existing unwrap segments into a physical depth path.

    Input and output format is unchanged:
        [((x, y), seg), ...]

    Logic:
        1. Measure actual unmasked component pixels.
        2. Group nearby x positions into lanes.
        3. Sort lanes left/right by convention.
        4. Sort segments within each lane top/bottom by convention.
    """

    if not segments:
        return []

    convention = convention or "rl_tb"

    right_to_left = convention in ("rl_tb", "rl_bt")
    top_to_bottom = convention in ("rl_tb", "lr_tb")

    measured = []

    for item in segments:
        (x, y), seg = item

        seg_mask = np.ma.getmaskarray(seg)

        # Collapse cube mask to spatial mask
        if seg_mask.ndim > 2:
            seg_mask = seg_mask.any(axis=2)

        rr, cc = np.nonzero(~seg_mask)

        # Degenerate segment: keep it, but fall back to bbox origin
        if rr.size == 0:
            measured.append({
                "item": item,
                "median_x": float(x),
                "min_y": int(y),
                "max_y": int(y + seg.shape[0] - 1),
            })
            continue

        global_x = x + cc
        global_y = y + rr

        measured.append({
            "item": item,
            "median_x": float(np.median(global_x)),
            "min_y": int(np.min(global_y)),
            "max_y": int(np.max(global_y)),
            "width": int(np.max(global_x) - np.min(global_x) + 1)
        })
    

    #Calclate typical core run widths to threshold
    widths = [r["width"] for r in measured if r["width"] > 0]
    lane_gap = max(10, 0.25 * np.median(widths)) if widths else 25

    # ---- build lanes from actual component x positions ----
    measured_by_x = sorted(measured, key=lambda r: r["median_x"])

    lanes = []

    for rec in measured_by_x:
        if not lanes:
            lanes.append({
                "x_center": rec["median_x"],
                "records": [rec],
            })
            continue

        nearest_lane = min(
            lanes,
            key=lambda lane: abs(rec["median_x"] - lane["x_center"])
        )

        if abs(rec["median_x"] - nearest_lane["x_center"]) <= lane_gap:
            nearest_lane["records"].append(rec)
            nearest_lane["x_center"] = float(np.median(
                [r["median_x"] for r in nearest_lane["records"]]
            ))
        else:
            lanes.append({
                "x_center": rec["median_x"],
                "records": [rec],
            })

    # ---- sort lanes according to convention ----
    lanes = sorted(
        lanes,
        key=lambda lane: lane["x_center"],
        reverse=right_to_left
    )

    # ---- sort inside each lane by y according to convention ----
    sorted_segments = []

    for lane in lanes:
        lane_records = sorted(
            lane["records"],
            key=lambda r: r["min_y"] if top_to_bottom else -r["max_y"]
        )

        sorted_segments.extend(r["item"] for r in lane_records)

    return sorted_segments



def unwrap_from_stats(mask, image, stats, labels,
                                anchors=None,
                                convention=None, 
                                depth_start = None,
                                depth_stop = None,
                                return_map = False
                                ):
    """
    Extended unwrap_from_stats that resolves anchor (x, y) positions to
    output row indices, then performs the standard unwrap.

    

    Parameters
    ----------
    mask, image, stats : pre-requisite datasets 
    labels : ndarray
        Connected-component label image matching stats. Older normalised
        segment images are accepted and reconstructed for compatibility.
    anchors : list of dict or None
        Each dict has keys 'x', 'y', 'depth' — pixel-space position of a
        known depth point on the tray image.
    convention : str or None
        Sort convention: 'rl_tb' (default), 'lr_tb', 'rl_bt', 'lr_bt'.
    MIN_AREA, MIN_WIDTH : int
        Filter thresholds, pulled from the config and user definable.

    Returns
    -------
    concatenated : np.ma.MaskedArray
        Same stacked output as the original function.
    resolved_anchors : list of dict
        Each input anchor dict extended with:
            'row'     : resolved output row index (int)
            'snapped' : True if the point was outside every valid segment
            'segment' : index of the owning segment in the sorted list
    """
    MIN_AREA = config.min_seg_area
    MIN_WIDTH = config.min_seg_width
    _SORT_KEYS = {
        'rl_tb': lambda s: ( round(-s[0][0] / 10),  s[0][1]),
        'lr_tb': lambda s: ( round( s[0][0] / 10),  s[0][1]),
        'rl_bt': lambda s: ( round(-s[0][0] / 10), -s[0][1]),
        'lr_bt': lambda s: ( round( s[0][0] / 10), -s[0][1]),
    }
    sort_key = _SORT_KEYS.get(convention, _SORT_KEYS['rl_tb'])

    anchors = anchors or []
    
    # Adjustments to the unwrapping code now requires originally label image
    # to avoid mis-registration on overlapping bboxes
    # For backwards compatibility old, normalised label images are accepted and sanitised here
    # calc_unwrap_stats no longer normalises the label image

    max_label = stats.shape[0] - 1
    if max_label <= 0:
        labels = np.zeros_like(labels, dtype=np.int32)
    elif np.nanmax(labels) <= 1.0:
        labels =  np.rint(labels * max_label).astype(np.int32)
    else:
        labels =  labels.astype(np.int32)

    # ---- collect valid segments — stored as ((x, y), segment) matching
    #      production tuple structure; w/h derived from segment.shape ----
    segments = []
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or w < MIN_WIDTH:
            continue
        sub      = image[y:y+h, x:x+w]
        seg_mask = labels[y:y+h, x:x+w] != i
        if sub.ndim > 2:                                  # broadcast across bands for a cube
            seg_mask = np.repeat(seg_mask[:, :, None], sub.shape[2], axis=2)
        seg = np.ma.masked_array(sub, mask=seg_mask)
        segments.append(((x, y), seg))

    #segments_sorted = sorted(segments, key=sort_key)
    #segments_sorted = sorted(segments,
    #key=lambda s: _segment_path_key(s, convention=convention or "rl_tb"))
    segments_sorted = _sort_segments_by_runs(
    segments,
    convention=convention or "rl_tb")


    # ---- cumulative row offsets — height from segment.shape[0] ----
    cumulative_rows = []
    running = 0
    for (sx, sy), seg in segments_sorted:
        cumulative_rows.append(running)
        running += seg.shape[0]

    
    for idx, ((sx, sy), seg) in enumerate(segments_sorted):
        sh, sw = seg.shape[:2]
        

    # ---- resolve each anchor to an output row ----
    resolved_anchors = []

    for anc in anchors:
        ax, ay, adepth = anc['x'], anc['y'], anc['depth']
        result = dict(anc)
        result['snapped'] = False

        # Check whether anchor falls inside a valid segment bounding box
        hit_idx = None
        for idx, ((sx, sy), seg) in enumerate(segments_sorted):
            sh, sw = seg.shape[:2]
            if sx <= ax < sx + sw and sy <= ay < sy + sh:
                hit_idx = idx
                break

        if hit_idx is not None:
            (sx, sy), seg = segments_sorted[hit_idx]
            within_row        = ay - sy
            result['row']     = cumulative_rows[hit_idx] + within_row
            result['segment'] = hit_idx
            
        else:
            # Snap: find nearest segment in sort order by column bin,
            # breaking ties by y proximity
            anchor_col_bin = sort_key(((ax, ay), None))[0]
            seg_col_bins   = [sort_key(s)[0] for s in segments_sorted]

            best_idx = min(
                range(len(segments_sorted)),
                key=lambda i: (
                    abs(seg_col_bins[i] - anchor_col_bin),
                    abs(segments_sorted[i][0][1] - ay)
                )
            )
            result['row']     = cumulative_rows[best_idx]
            result['segment'] = best_idx
            result['snapped'] = True
            

        resolved_anchors.append(result)

    # ---- pad and stack — identical to production ----
    max_width = max(seg.shape[1] for _, seg in segments_sorted)
    padded_segments = []

    for _, seg in segments_sorted:
        h, w      = seg.shape[:2]
        pad_total = max_width - w

        if pad_total > 0:
            pad_left  = pad_total // 2
            pad_right = pad_total - pad_left
            pad_shape = ((0, 0), (pad_left, pad_right)) if seg.ndim == 2 \
                        else ((0, 0), (pad_left, pad_right), (0, 0))
            seg_pad   = np.pad(seg.data, pad_shape, mode='constant', constant_values=0)
            seg_mask_ = np.pad(seg.mask, pad_shape, mode='constant', constant_values=1)
            padded_segments.append(np.ma.masked_array(seg_pad, mask=seg_mask_))
        else:
            padded_segments.append(seg)

    concatenated = np.ma.masked_array(
        np.vstack([s.data for s in padded_segments]),
        mask=np.vstack([s.mask for s in padded_segments])
    )

    depths = None
    if depth_start is not None and depth_stop is not None:
        n_rows = concatenated.shape[0]
        if resolved_anchors:
            anchor_rows   = [0]           + [r['row']   for r in resolved_anchors] + [n_rows - 1]
            anchor_depths = [depth_start] + [r['depth'] for r in resolved_anchors] + [depth_stop]
            order         = np.argsort(anchor_rows)
            anchor_rows   = np.array(anchor_rows)[order]
            anchor_depths = np.array(anchor_depths)[order]
            depths        = np.interp(np.arange(n_rows), anchor_rows, anchor_depths)
        else:
            depths = np.linspace(depth_start, depth_stop, n_rows)

    if not return_map:
        return concatenated, depths

    depth_map = (build_depth_map(depths, segments_sorted, cumulative_rows, image.shape[:2])
                if depths is not None else None)
    return concatenated, depths, depth_map


def build_depth_map(depths, segments_sorted, cumulative_rows, out_shape):
    """
    Reverse the unwrap: give every original-image pixel its registered depth.

    Built from the same segment geometry the forward pass used, so it is a
    true inverse of *this* unwrap, not a re-derivation. The forward pass
    assigns one depth per stacked row, i.e. per original row within a segment,
    so depth is constant across a segment's width and every core pixel inherits
    its row's depth.

    Parameters
    ----------
    depths : np.ndarray, shape (n_rows,)
        Per-stacked-row depths; n_rows == sum of segment heights.
    segments_sorted : list of ((x, y), np.ma.MaskedArray)
        Convention-sorted segments; value is the masked bbox crop, origin
        (x, y) in original-image coordinates.
    cumulative_rows : list of int
        Starting stacked-row offset per segment, aligned with segments_sorted.
    out_shape : tuple of int
        (H, W) of the original (cropped) image being registered.

    Returns
    -------
    np.ma.MaskedArray, shape (H, W), float
        Per-pixel depth, masked wherever no kept-segment core pixel contributed.
    """
    H, W = out_shape
    depth_map = np.ma.masked_all((H, W), dtype=float)

    for i, ((sx, sy), seg) in enumerate(segments_sorted):
        h, w = seg.shape[:2]
        off = cumulative_rows[i]
        row_depths = depths[off:off + h]                   # (h,) one depth per row

        seg_mask = np.ma.getmaskarray(seg)                 # full bool, seg.shape
        if seg_mask.ndim > 2:                              # cube: collapse bands
            seg_mask = seg_mask.any(axis=tuple(range(2, seg_mask.ndim)))
        core_r, core_c = np.nonzero(~seg_mask)             # local coords of core

        depth_map[sy + core_r, sx + core_c] = row_depths[core_r]

    
    return depth_map




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