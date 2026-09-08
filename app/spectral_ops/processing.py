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
    mask = np.zeros((cube.shape[0], cube.shape[1])).astype(int)
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


# ================== unwrapping and depth registration =========================================
def _sort_segments_by_runs(segments, convention="rl_tb"):
    """
    Group connected-component segments into lanes and order those lanes
    according to the box convention.

    Parameters
    ----------
    segments : list
        Each item is ``((x, y), seg)``, where ``x`` and ``y`` are the
        absolute image coordinates of the segment bounding-box origin and
        ``seg`` is its masked bounding-box crop.

    convention : str
        One of:
            ``rl_tb`` : lanes right-to-left, rows top-to-bottom
            ``lr_tb`` : lanes left-to-right, rows top-to-bottom
            ``rl_bt`` : lanes right-to-left, rows bottom-to-top
            ``lr_bt`` : lanes left-to-right, rows bottom-to-top

    Returns
    -------
    lanes_sorted : list[list[tuple]]
        Lanes in physical downhole order. Each segment is returned as::

            ((x, y_start, y_stop), seg)

        ``y_start`` and ``y_stop`` are absolute image-row coordinates;
        ``y_stop`` is exclusive. Segments in the same lane therefore retain
        enough spatial information to identify overlapping absolute y rows.
    """
    if not segments:
        return []

    convention = convention or "rl_tb"

    if convention not in {"rl_tb", "lr_tb", "rl_bt", "lr_bt"}:
        convention = "rl_tb"

    right_to_left = convention.startswith("rl")
    top_to_bottom = convention.endswith("tb")

    measured = []

    for (x, y), seg in segments:
        spatial_mask = _spatial_mask(seg)
        rows, cols = np.nonzero(~spatial_mask)

        image_y_start = int(y)
        image_y_stop = int(y + seg.shape[0])

        if rows.size == 0:
            # Retain a degenerate segment using its bounding-box geometry.
            median_x = float(x)
            min_y = image_y_start
            max_y = image_y_stop - 1
            width = 0
        else:
            absolute_x = x + cols
            absolute_y = y + rows

            median_x = float(np.median(absolute_x))
            min_y = int(np.min(absolute_y))
            max_y = int(np.max(absolute_y))
            width = int(np.max(absolute_x) - np.min(absolute_x) + 1)

        measured.append((
            median_x,
            min_y,
            max_y,
            width,
            int(x),
            image_y_start,
            image_y_stop,
            seg,
        ))

    widths = [record[3] for record in measured if record[3] > 0]

    lane_gap = max(20, 0.6 * np.median(widths)) if widths else 25

    measured.sort(key=lambda record: record[0])

    # Each lane is stored temporarily as:
    # [current_x_center, [measured records]]
    lanes = []

    for record in measured:
        median_x = record[0]

        if not lanes:
            lanes.append([median_x, [record]])
            continue

        nearest_lane = min(
            lanes,
            key=lambda lane: abs(median_x - lane[0]),
        )

        if abs(median_x - nearest_lane[0]) <= lane_gap:
            nearest_lane[1].append(record)
            nearest_lane[0] = float(np.median([
                item[0]
                for item in nearest_lane[1]
            ]))
        else:
            lanes.append([median_x, [record]])

    # Put the lanes into physical downhole order.
    lanes.sort(
        key=lambda lane: lane[0],
        reverse=right_to_left,
    )

    lanes_sorted = []

    for _, lane_records in lanes:
        if top_to_bottom:
            lane_records.sort(key=lambda record: record[1])
        else:
            lane_records.sort(
                key=lambda record: record[2],
                reverse=True,
            )

        lane_segments = [
            (
                (
                    record[4],  # absolute x
                    record[5],  # absolute y_start
                    record[6],  # absolute y_stop, exclusive
                ),
                record[7],      # masked segment crop
            )
            for record in lane_records
        ]

        lanes_sorted.append(lane_segments)

    return lanes_sorted

def _spatial_mask(seg):
    """
    Return a 2D True=masked/invalid spatial mask for a segment.
    """
    seg_mask = np.ma.getmaskarray(seg)

    if seg_mask.ndim > 2:
        seg_mask = seg_mask.any(axis=tuple(range(2, seg_mask.ndim)))

    return seg_mask



def unwrap_from_stats(
    mask,
    image,
    stats,
    labels,
    anchors=None,
    convention=None,
    depth_start=None,
    depth_stop=None,
    return_map=False,
):
    """
    Reconstruct connected components within their physical lanes and unwrap
    those lanes into a vertically concatenated image.

    Components assigned to the same lane retain their absolute image-row
    positions. Components sharing an absolute y-coordinate therefore share
    an output row and registered depth.

    Parameters
    ----------
    mask : np.ndarray
        Spatial mask matching ``image``. Non-zero values are invalid.

    image : np.ndarray or np.ma.MaskedArray
        Data to unwrap. May be a 2D image or a 3D spectral cube.

    stats : np.ndarray
        Connected-component statistics returned by
        ``cv2.connectedComponentsWithStats``.

    labels : np.ndarray
        Connected-component label image corresponding to ``stats``.
        Older normalised label images are accepted.

    anchors : list of dict or None
        Depth-control points in the original image coordinate system.
        Each dictionary must contain ``x``, ``y`` and ``depth``.

    convention : str or None
        One of:
            ``rl_tb`` : lanes right-to-left, rows top-to-bottom
            ``lr_tb`` : lanes left-to-right, rows top-to-bottom
            ``rl_bt`` : lanes right-to-left, rows bottom-to-top
            ``lr_bt`` : lanes left-to-right, rows bottom-to-top

    depth_start, depth_stop : float or None
        Start and stop depths for the box.

    return_map : bool
        If True, also return a masked depth map in the original image space.

    Returns
    -------
    concatenated : np.ma.MaskedArray
        Lane-reconstructed and vertically concatenated image.

    depths : np.ndarray or None
        One registered depth per row of ``concatenated``.

    depth_map : np.ma.MaskedArray, optional
        Per-pixel registered depths in the original image coordinates.
    """
    anchors = anchors or []
    convention = convention or "rl_tb"

    valid_conventions = {
        "rl_tb",
        "lr_tb",
        "rl_bt",
        "lr_bt",
    }

    if convention not in valid_conventions:
        raise ValueError(
            f"Unknown box convention {convention!r}. "
            f"Expected one of {sorted(valid_conventions)}."
        )

    bottom_to_top = convention.endswith("bt")

    MIN_AREA = config.min_seg_area
    MIN_WIDTH = config.min_seg_width

    # ------------------------------------------------------------------
    # Sanitise the connected-component label image.
    #
    # Current files contain integer component labels. Older files may
    # contain the same labels normalised into the range 0–1.
    # ------------------------------------------------------------------
    max_label = stats.shape[0] - 1

    if max_label <= 0:
        labels = np.zeros_like(labels, dtype=np.int32)

    elif np.nanmax(labels) <= 1.0:
        labels = np.rint(
            labels * max_label
        ).astype(np.int32)

    else:
        labels = labels.astype(np.int32)

    # ------------------------------------------------------------------
    # Reconstruct retained connected components.
    #
    # Each item has the existing form:
        #((absolute_x, absolute_y), masked_segment)
    # ------------------------------------------------------------------
    segments = []

    for label_id in range(1, stats.shape[0]):
        x, y, w, h, area = stats[label_id]
        if area < MIN_AREA or w < MIN_WIDTH:
            continue
        sub      = image[y:y+h, x:x+w]
        seg_mask = (labels[y:y+h, x:x+w] != label_id) | mask[y:y+h, x:x+w].astype(bool)
        if sub.ndim > 2:                                  # broadcast across bands for a cube
            seg_mask = np.repeat(seg_mask[:, :, None], sub.shape[2], axis=2)
        seg = np.ma.masked_array(sub, mask=seg_mask)
        segments.append(((x, y), seg))

    if not segments:
        raise ValueError(
            "No connected components passed the configured area and "
            "width thresholds."
        )

    # ------------------------------------------------------------------
    # Assign components to lanes.
    #
    # The returned structure is:
    #
    # [
    #     [
    #         ((x, y_start, y_stop), seg),
    #         ...
    #     ],
    #     ...
    # ]
    #
    # The outer list is in physical downhole lane order. y_stop is
    # exclusive.
    # ------------------------------------------------------------------
    lanes_sorted = _sort_segments_by_runs(
        segments,
        convention=convention,
    )

    if not lanes_sorted:
        raise ValueError(
            "No lanes could be constructed from the retained segments."
        )

    # The reconstructed lane images and their corresponding absolute
    # source-image y-coordinates.
    lane_images = []
    lane_source_rows = []

    # ------------------------------------------------------------------
    # Reconstruct each lane.
    # ------------------------------------------------------------------
    for lane in lanes_sorted:
        # Produce one lane row for every unique absolute image y-position
        # occupied by a segment in this lane.
        source_rows = np.unique(
            np.concatenate([
                np.arange(
                    y_start,
                    y_stop,
                    dtype=np.int32,
                )
                for (x, y_start, y_stop), seg in lane
            ])
        )

        source_rows.sort()

        # Preserve the original relative x-positions of every component.
        lane_x_start = min(x for (x, y_start, y_stop), seg in lane)
        lane_x_stop = max(x + seg.shape[1] for (x, y_start, y_stop), seg in lane)
        lane_height = source_rows.size
        lane_width = lane_x_stop - lane_x_start

        if image.ndim == 2:
            lane_shape = (lane_height, lane_width)
        elif image.ndim == 3:
            bands = image.shape[2]
            lane_shape = (lane_height, lane_width, bands)
        else:
            raise ValueError(
                f"Expected a 2D image or 3D cube, got shape {image.shape}"
            )

        lane_data = np.zeros(lane_shape, dtype=image.dtype,)
        lane_mask = np.ones(lane_shape, dtype=bool,)

        # Fast lookup from an absolute image y-coordinate to a row in the
        # reconstructed lane.
        output_row_for_y = np.full(image.shape[0], -1, dtype=np.int32,)
        output_row_for_y[source_rows] = np.arange(lane_height, dtype=np.int32,)

        for (x, y_start, y_stop), seg in lane:
            destination_x_start = x - lane_x_start
            destination_x_stop = (destination_x_start + seg.shape[1])

            segment_data = np.ma.getdata(seg)
            segment_mask = np.ma.getmaskarray(seg)

            for local_row in range(seg.shape[0]):
                absolute_y = y_start + local_row
                destination_row = output_row_for_y[absolute_y]

                target_data = lane_data[
                    destination_row,
                    destination_x_start:destination_x_stop,
                    ...
                ]

                target_mask = lane_mask[
                    destination_row,
                    destination_x_start:destination_x_stop,
                    ...
                ]

                source_data = segment_data[local_row]
                source_mask = segment_mask[local_row]
                source_valid = ~source_mask

                # Copy only real component pixels. Masked bbox pixels must
                # not overwrite pixels contributed by another component.
                np.copyto(
                    target_data,
                    source_data,
                    where=source_valid,
                )

                target_mask[source_valid] = False

        lane_image = np.ma.masked_array(
            lane_data,
            mask=lane_mask,
        )

        # The lane was constructed in ascending original-image y order.
        # Reverse both the image and its source-row mapping when the box
        # convention runs bottom-to-top.
        if bottom_to_top:
            lane_image = lane_image[::-1]
            source_rows = source_rows[::-1]

        lane_images.append(lane_image)
        lane_source_rows.append(source_rows)

    # ------------------------------------------------------------------
    # Pad complete lanes to a common width.
    # ------------------------------------------------------------------
    widest_lane = max(
        lane.shape[1]
        for lane in lane_images
    )

    padded_lanes = []

    for lane in lane_images:
        lane_height, lane_width = lane.shape[:2]

        pad_total = widest_lane - lane_width
        pad_left = pad_total // 2

        padded_shape = (lane_height, widest_lane, *lane.shape[2:],)

        padded = np.ma.masked_all(padded_shape, dtype=lane.dtype,)

        padded[:, pad_left:pad_left + lane_width, ...] = lane

        padded_lanes.append(padded)

    # Stack complete lanes rather than individual components.
    concatenated = np.ma.concatenate(
        padded_lanes,
        axis=0,
    )

    # Starting output row of every lane in the concatenated image.
    lane_offsets = np.cumsum([0] + [source_rows.size
            for source_rows in lane_source_rows
        ]
    )[:-1]

    # ------------------------------------------------------------------
    # Resolve depth anchors against lanes.
    # ------------------------------------------------------------------
    lane_centres = []

    for lane in lanes_sorted:
        segment_centres = []

        for (x, y_start, y_stop), seg in lane:
            segment_spatial_mask = _spatial_mask(seg)
            rows, cols = np.nonzero(~segment_spatial_mask)

            if cols.size:
                segment_centres.append(float(np.median(x + cols)))
            else:
                segment_centres.append(float(x + (seg.shape[1] - 1) / 2))

        lane_centres.append(float(np.median(segment_centres)))

    resolved_anchors = []

    for anchor in anchors:
        ax = int(anchor["x"])
        ay = int(anchor["y"])
        adepth = float(anchor["depth"])

        hit_lane = None

        # First attempt an exact hit on a valid component pixel.
        for lane_index, lane in enumerate(lanes_sorted):
            for (x, y_start, y_stop), seg in lane:
                segment_width = seg.shape[1]

                if not (
                    x <= ax < x + segment_width
                    and y_start <= ay < y_stop
                ):
                    continue

                local_row = ay - y_start
                local_col = ax - x

                segment_spatial_mask = _spatial_mask(seg)

                if not segment_spatial_mask[
                    local_row,
                    local_col,
                ]:
                    hit_lane = lane_index
                    break

            if hit_lane is not None:
                break

        snapped = hit_lane is None

        # If the point missed every component, select the nearest lane by
        # its representative x-position.
        if hit_lane is None:
            hit_lane = min(
                range(len(lane_centres)),
                key=lambda lane_index: abs(
                    ax - lane_centres[lane_index]
                ),
            )

        source_rows = lane_source_rows[hit_lane]

        # Exact hits find ay itself. Missed anchors retain ay when another
        # fragment in that lane occupies the row, otherwise they snap to
        # the nearest occupied source row.
        local_lane_row = int(np.argmin(np.abs(source_rows - ay)))

        resolved_source_y = int(source_rows[local_lane_row])

        output_row = int(lane_offsets[hit_lane] + local_lane_row)

        resolved_anchors.append({
            **anchor,
            "depth": adepth,
            "row": output_row,
            "lane": hit_lane,
            "source_y": resolved_source_y,
            "snapped": snapped,
        })

    # ------------------------------------------------------------------
    # Register depths against concatenated lane rows.
    # ------------------------------------------------------------------
    depths = None

    if depth_start is not None and depth_stop is not None:
        n_rows = concatenated.shape[0]

        if resolved_anchors:
            control_rows = np.array( [0] + [anchor["row"] for anchor in resolved_anchors] + [n_rows - 1], dtype=int,)

            control_depths = np.array([depth_start] + [anchor["depth"] for anchor in resolved_anchors]  + [depth_stop], dtype=float,)

            order = np.argsort(control_rows, kind="stable", )

            control_rows = control_rows[order]
            control_depths = control_depths[order]

            depths = np.interp(np.arange(n_rows), control_rows, control_depths,)

        else:
            depths = np.linspace(depth_start,depth_stop, n_rows,)

    if not return_map:
        return concatenated, depths

    # ------------------------------------------------------------------
    # Paint the registered depths back into original image coordinates.
    # ------------------------------------------------------------------
    depth_map = None

    if depths is not None:
        depth_map = np.ma.masked_all(
            image.shape[:2],
            dtype=float,
        )

        for lane_index, lane in enumerate(lanes_sorted):
            source_rows = lane_source_rows[lane_index]
            lane_offset = lane_offsets[lane_index]
            lane_height = source_rows.size

            # Lookup from absolute source y to registered depth. The order
            # of source_rows already reflects the tb/bt convention.
            depth_for_source_y = np.ma.masked_all(
                image.shape[0],
                dtype=float,
            )

            depth_for_source_y[source_rows] = depths[
                lane_offset:lane_offset + lane_height
            ]

            for (x, y_start, y_stop), seg in lane:
                segment_spatial_mask = _spatial_mask(seg)

                local_rows, local_cols = np.nonzero(
                    ~segment_spatial_mask
                )

                absolute_rows = y_start + local_rows
                absolute_cols = x + local_cols

                depth_map[
                    absolute_rows,
                    absolute_cols,
                ] = depth_for_source_y[absolute_rows]

    return concatenated, depths, depth_map


# ==================================================================================================

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