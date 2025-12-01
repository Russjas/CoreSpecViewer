import numpy as np

def resample_fractions_and_dominant_by_step(
    depths_row: np.ndarray,
    fractions_row: np.ndarray,   # (N, K+1)
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin row-based fractions into regular depth intervals and recompute dominant.

    Parameters
    ----------
    depths_row : (N,)
        Depth per unwrapped row (same length as fractions_row).
    fractions_row : (N, K+1)
        Fractions per row (output of compute_fullhole_mineral_fractions
        after concatenating all boxes). Columns 0..K-1 are legend classes,
        column K is 'unclassified'.
    step : float
        Bin size in depth units (e.g. 0.05 for 5 cm).

    Returns
    -------
    depths_bin : (M,)
        Depth for each bin (bin centres).
    fractions_bin : (M, K+1)
        Mean fractions over rows falling into each bin.
    dominant_bin : (M,)
        Dominant mineral per bin, as an index 0..K-1 into the legend.
        -1 if no classified pixels in that bin (i.e. only unclassified).
    """
    depths_row = np.asarray(depths_row, dtype=float)
    frac = np.asarray(fractions_row, dtype=float)
    N, C = frac.shape

    if depths_row.shape[0] != N:
        raise ValueError("depths_row and fractions_row must have same length.")

    if N == 0:
        return (
            np.zeros((0,), dtype=float),
            np.zeros((0, C), dtype=float),
            np.zeros((0,), dtype=int),
        )

    d_min = depths_row.min()
    d_max = depths_row.max()

    # Bin edges
    bins = np.arange(d_min, d_max + step, step)
    bin_idx = np.digitize(depths_row, bins) - 1  # 0..len(bins)-2

    depths_out = []
    frac_out = []
    dom_out = []

    K = C - 1  # last column = unclassified

    for k in range(len(bins) - 1):
        rows_in_bin = (bin_idx == k)
        if not np.any(rows_in_bin):
            continue  # skip empty bins

        frac_mean = frac[rows_in_bin].mean(axis=0)
        depth_mid = 0.5 * (bins[k] + bins[k + 1])

        # dominant in this bin = max of classified fractions only
        classified = frac_mean[:K]
        if classified.sum() > 0:
            dom_idx = int(np.argmax(classified))
        else:
            dom_idx = -1

        depths_out.append(depth_mid)
        frac_out.append(frac_mean)
        dom_out.append(dom_idx)

    if not depths_out:
        return (
            np.zeros((0,), dtype=float),
            np.zeros((0, C), dtype=float),
            np.zeros((0,), dtype=int),
        )

    return (
        np.array(depths_out),
        np.vstack(frac_out),
        np.array(dom_out, dtype=int),
    )

def bin_features_by_step(
    depths_row: np.ndarray,
    features_row: np.ndarray,
    step: float,
    agg: str = "mean",
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin row-based features into regular depth intervals of size `step`.

    Parameters
    ----------
    depths_row : (N,)
        Depth for each row (same length as features_row).
    features_row : (N,) or (N, F)
        Features per row (e.g. band position, depth, strength, etc.).
        NaNs are treated as missing values.
    step : float
        Bin size in depth units (e.g. 0.05 for 5 cm).
    agg : {"mean", "median"}
        Aggregation function to apply within each bin.
    min_count : int
        Minimum number of *non-NaN* samples required in a bin to
        produce a value. Bins with fewer valid samples are set to NaN.

    Returns
    -------
    depths_bin : (M,)
        Depth for each bin (bin centres).
    features_bin : (M,) or (M, F)
        Binned features. Same feature dimension as `features_row`.
    """
    depths_row = np.asarray(depths_row, dtype=float)
    feats = np.asarray(features_row, dtype=float)

    if depths_row.ndim != 1:
        raise ValueError("depths_row must be 1D.")

    N = depths_row.shape[0]

    if feats.ndim == 1:
        feats = feats.reshape(N, 1)
        squeeze = True
    elif feats.ndim == 2 and feats.shape[0] == N:
        squeeze = False
    else:
        raise ValueError("features_row must be shape (N,) or (N, F).")

    N, F = feats.shape

    if N == 0:
        return (
            np.zeros((0,), dtype=float),
            np.zeros((0, F), dtype=float) if not squeeze else np.zeros((0,), dtype=float),
        )

    if step <= 0:
        raise ValueError("step must be positive.")

    d_min = depths_row.min()
    d_max = depths_row.max()

    # Bin edges [d_min, d_min+step, ..., d_max+step]
    bins = np.arange(d_min, d_max + step, step)
    if bins.size < 2:
        # Degenerate case: everything in one bin
        bins = np.array([d_min, d_max])

    # For each row, which bin does it fall into? (0..len(bins)-2)
    bin_idx = np.digitize(depths_row, bins) - 1

    depths_out = []
    feats_out = []

    for k in range(len(bins) - 1):
        rows_in_bin = (bin_idx == k)
        if not np.any(rows_in_bin):
            continue  # skip empty bins completely

        block = feats[rows_in_bin]          # (n_bin_rows, F)
        # Mask NaNs
        valid_mask = ~np.isnan(block)
        valid_counts = valid_mask.sum(axis=0)

        # If too few valid samples, set NaN for that feature
        vals = np.full(F, np.nan, dtype=float)

        if agg == "mean":
            # Avoid warnings by using where + counts
            sums = np.nansum(block, axis=0)
            with np.errstate(invalid="ignore"):
                vals = np.where(valid_counts >= min_count, sums / valid_counts, np.nan)
        elif agg == "median":
            for j in range(F):
                col = block[:, j]
                col_valid = col[~np.isnan(col)]
                if col_valid.size >= min_count:
                    vals[j] = np.median(col_valid)
        else:
            raise ValueError("agg must be 'mean' or 'median'.")

        depth_mid = 0.5 * (bins[k] + bins[k + 1])

        depths_out.append(depth_mid)
        feats_out.append(vals)

    if not depths_out:
        return (
            np.zeros((0,), dtype=float),
            np.zeros((0, F), dtype=float) if not squeeze else np.zeros((0,), dtype=float),
        )

    depths_bin = np.array(depths_out)
    features_bin = np.vstack(feats_out)  # (M, F)

    if squeeze:
        features_bin = features_bin[:, 0]

    return depths_bin, features_bin

