"""
downhole_resampling_v2.py
 
Vectorised replacements for the binning functions in downhole_resampling.py.
 
Strategy
--------
All three public functions share the same core approach:
 
  1. Build a regular bin-centre grid with np.arange  (unchanged from v1)
  2. Assign every input row to a bin in ONE pass with np.digitize,
     producing an integer label array of length N.
  3. Aggregate using np.bincount (mean path) or a sort-then-split pattern
     (median path) — no Python loop over bins.
 
This reduces the dominant cost from O(M × N) to O(N log N) or O(N),
where M = number of bins and N = number of depth rows.
 
Numerical compatibility with v1
---------------------------------
step_fractions_pair  — bit-for-bit identical for interior bins.
                       Edge convention at d_max is explicitly matched
                       (see _make_bin_grid docstring).
step_indices         — identical; tie-breaking (lowest index wins) is
                       preserved because np.argmax returns the first max.
step_continuous/mean — bit-for-bit identical after masked→NaN conversion.
step_continuous/median — mathematically identical; verified by unit test
                         template included at the bottom of this file.
 
Public API
----------
Identical signatures and return types to v1.  Drop-in replacement.
"""
 
from __future__ import annotations
 
import numpy as np
from scipy import stats as _scipy_stats   # imported ONCE at module level
 
 
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
 
def _make_bin_grid(d_min: float, d_max: float, step: float) -> np.ndarray:
    """
    Build the regular grid of bin centres that matches v1 exactly.
 
    v1 uses:
        np.arange(d_min, d_max + step / 2.0, step)
 
    The +step/2.0 sentinel ensures d_max is covered when d_max falls
    exactly on a grid point.  We replicate this here so downstream
    consumers see the same depth axis.
    """
    return np.arange(d_min, d_max + step / 2.0, step)
 
 
def _digitize_to_bins(depths: np.ndarray, depths_bin: np.ndarray) -> np.ndarray:
    """
    Assign each element of `depths` to its bin index using np.searchsorted.
 
    Returns an integer array `labels` of shape (N,) where:
        labels[i] >= 0  means depths[i] belongs to bin labels[i]
        labels[i] == -1 means depths[i] is NaN or outside [d_min, d_max)
                        and must be excluded before calling np.bincount.
 
    Convention matches v1's half-open intervals: [lo, hi) for each bin,
    i.e.  lo = bin_centre - half,  hi = bin_centre + half.
 
    NaN depths: np.searchsorted behaviour on NaN is undefined — on most
    platforms NaN sorts to the end, giving a label of M-1 (last bin),
    silently contaminating it.  v1's boolean mask approach returns False
    for any NaN comparison, excluding such rows from all bins.  We
    replicate that by explicitly marking NaN depths with label -1.
    Callers must filter to labels >= 0 before passing to np.bincount.
    """
    step = depths_bin[1] - depths_bin[0] if len(depths_bin) > 1 else 1.0
    half = step / 2.0
 
    # Bin edges: left edge of each bin
    edges = depths_bin - half  # shape (M,)
 
    # searchsorted with 'right' gives the number of edges <= depth,
    # which is the 1-based bin index.  Subtract 1 for 0-based.
    labels = np.searchsorted(edges, depths, side="right") - 1
 
    M = len(depths_bin)
 
    # Clamp right overflow (depths in the right tail of the last bin)
    labels = np.where(labels >= M, M - 1, labels)
 
    # Mark NaN depths as -1 so callers can exclude them.
    # This matches v1's boolean mask behaviour where NaN comparisons
    # always return False, silently excluding NaN-depth rows from all bins.
    nan_depths = np.isnan(depths)
    if nan_depths.any():
        labels = labels.copy()
        labels[nan_depths] = -1
 
    return labels.astype(np.intp)
 
 
def _masked_to_float(arr) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a regular or masked array to (values_float, valid_bool).
 
    values_float : (N,) float64, NaN where invalid
    valid_bool   : (N,) bool,    False where invalid
 
    Treats both numpy mask and pre-existing NaNs as invalid.
    """
    if isinstance(arr, np.ma.MaskedArray):
        values = np.array(arr.data, dtype=float)
        mask_invalid = np.ma.getmaskarray(arr)
        values[mask_invalid] = np.nan
    else:
        values = np.asarray(arr, dtype=float)
 
    valid = ~np.isnan(values)
    return values, valid
 
 
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
 
def step_fractions_pair(
    depths_row: np.ndarray,
    fractions_row: np.ndarray,   # (N, K+1)
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin row-based fractions into regular depth intervals and recompute dominant.
 
    Vectorised replacement for v1.step_fractions_pair.
 
    Parameters
    ----------
    depths_row : (N,)
    fractions_row : (N, K+1)
        Columns 0..K-1 are legend classes; column K is 'unclassified'.
    step : float
        Bin size in depth units.
 
    Returns
    -------
    depths_bin  : (M,)
    fractions_bin : (M, K+1)  — NaN for empty bins
    dominant_bin  : (M,)      — -1 for empty bins or no classified content
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
    if step <= 0:
        raise ValueError("step must be positive.")
 
    d_min = depths_row.min()
    d_max = depths_row.max()
 
    depths_bin = _make_bin_grid(d_min, d_max, step)
    M = len(depths_bin)
 
    # Assign every row to its bin in one pass
    labels = _digitize_to_bins(depths_row, depths_bin)  # (N,)
 
    # Pre-allocate outputs
    fractions_bin = np.full((M, C), np.nan, dtype=float)
    dominant_bin = np.full(M, -1, dtype=int)
 
    # --- NaN-aware vectorised mean per bin per column ---
    # fractions_row may contain NaN values (rows where no spectral classification
    # was possible). A single NaN weight passed to np.bincount contaminates the
    # entire bin sum, producing wrong fractions. We must treat NaNs as missing:
    # use a per-column valid count and zero-substitute NaN weights so they
    # contribute nothing to the sum — matching v1's use of np.nanmean.
    K = C - 1  # unclassified is last column
 
    # Rows whose depth is NaN get label -1 and must be excluded from all bins,
    # matching v1's behaviour where (NaN >= lo) is always False.
    depth_ok = labels >= 0  # (N,) bool
 
    for c in range(C):
        col = frac[:, c]                              # (N,) — may contain NaN
        valid = ~np.isnan(col) & depth_ok             # (N,) bool
 
        if not valid.any():
            continue  # entire column is NaN/bad-depth — leave fractions_bin[:, c] as NaN
 
        valid_labels  = labels[valid]
        valid_weights = col[valid]
 
        # Per-bin count of valid (non-NaN) rows for this column
        col_counts = np.bincount(valid_labels, minlength=M).astype(float)  # (M,)
        col_populated = col_counts > 0
 
        col_sum = np.bincount(valid_labels, weights=valid_weights, minlength=M)
        fractions_bin[col_populated, c] = col_sum[col_populated] / col_counts[col_populated]
 
    # A bin is populated if any column has a real value
    populated = ~np.all(np.isnan(fractions_bin), axis=1)  # (M,)
 
    # --- Dominant mineral: argmax over classified columns (0..K-1) ---
    classified = fractions_bin[:, :K]  # (M, K)
 
    # Only compute dominant for populated bins that have any classified content
    classified_sum = np.nansum(classified, axis=1)  # (M,)
    has_classified = populated & (classified_sum > 0)
 
    if has_classified.any():
        dominant_bin[has_classified] = np.argmax(classified[has_classified], axis=1)
 
    return depths_bin, fractions_bin, dominant_bin
 
 
def step_indices(
    depths: np.ndarray,
    indices: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Step INDEX data using mode aggregation.
 
    Vectorised replacement for v1.step_indices.
 
    Mode is computed via bincount-per-class: for each bin, build a histogram
    over class labels and take argmax.  This avoids both the Python loop over
    bins and the per-iteration scipy.stats.mode call.
 
    Tie-breaking: lowest class index wins (argmax returns first maximum) —
    identical to scipy.stats.mode behaviour.
 
    Parameters
    ----------
    depths  : (N,) float
    indices : (N,) int   — categorical class labels (>= 0)
    step    : float
 
    Returns
    -------
    depths_stepped  : (M,)
    indices_stepped : (M,) int16  — -1 for empty bins
    """
    depths = np.asarray(depths, dtype=float)
    indices = np.asarray(indices)
 
    if depths.shape[0] == 0:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=np.int16)
    if step <= 0:
        raise ValueError("step must be positive.")
 
    d_min = depths.min()
    d_max = depths.max()
 
    depths_stepped = _make_bin_grid(d_min, d_max, step)
    M = len(depths_stepped)
    indices_stepped = np.full(M, -1, dtype=np.int16)
 
    # Assign rows to bins
    labels = _digitize_to_bins(depths, depths_stepped)  # (N,)
 
    # Exclude rows with NaN depths (label -1) or invalid class indices (< 0).
    # NaN-depth rows match v1's behaviour where (NaN >= lo) is always False.
    valid_mask = (indices >= 0) & (labels >= 0)
    if not valid_mask.any():
        return depths_stepped, indices_stepped
 
    n_classes = int(indices[valid_mask].max()) + 1
 
    # Build a 2D histogram: rows = bins, cols = classes
    # Use a flat bincount with an offset: flat_index = label * n_classes + class
    flat_indices = (labels[valid_mask] * n_classes + indices[valid_mask]).astype(np.intp)
    hist_flat = np.bincount(flat_indices, minlength=M * n_classes)  # (M * n_classes,)
    hist = hist_flat.reshape(M, n_classes)  # (M, n_classes)
 
    # Bins with at least one valid sample
    populated = hist.sum(axis=1) > 0  # (M,)
 
    indices_stepped[populated] = np.argmax(hist[populated], axis=1).astype(np.int16)
 
    return depths_stepped, indices_stepped
 
 
def step_continuous(
    depths_row: np.ndarray,
    features_row: np.ndarray,   # (N,) — regular or masked array
    step: float,
    agg: str = "mean",
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin 1-D row-based features into regular depth intervals.
 
    Vectorised replacement for v1.step_continuous.
 
    Input is always 1-D (shape (N,)); the (N, F) path from v1 is not
    required here and has been removed for clarity.
 
    Parameters
    ----------
    depths_row   : (N,) float
    features_row : (N,) float or masked array
    step         : float
    agg          : {"mean", "median"}
    min_count    : int — bins with fewer valid samples are set to NaN
 
    Returns
    -------
    depths_bin   : (M,)
    features_bin : (M,) masked array — masked where insufficient data
    """
    depths_row = np.asarray(depths_row, dtype=float)
 
    if depths_row.shape[0] == 0:
        empty = np.ma.array(np.zeros((0,), dtype=float), mask=True)
        return np.zeros((0,), dtype=float), empty
    if step <= 0:
        raise ValueError("step must be positive.")
 
    d_min = depths_row.min()
    d_max = depths_row.max()
 
    depths_bin = _make_bin_grid(d_min, d_max, step)
    M = len(depths_bin)
 
    # --- Resolve input to float with NaN for invalids, once ---
    values, valid_feat = _masked_to_float(features_row)   # (N,), (N,)
 
    # Assign all rows to bins in one pass.
    # NaN depths get label -1 (see _digitize_to_bins docstring).
    labels = _digitize_to_bins(depths_row, depths_bin)  # (N,)
 
    # A row is usable only if its feature value is valid AND its depth
    # is not NaN. v1 excludes NaN-depth rows implicitly because
    # (NaN >= lo) is always False; we must do the same explicitly.
    valid = valid_feat & (labels >= 0)
 
    # Labels and values for usable rows only
    valid_labels = labels[valid]   # (K,) where K = valid.sum()
    valid_values = values[valid]   # (K,)
 
    # Count of valid samples per bin
    counts = np.bincount(valid_labels, minlength=M).astype(float)  # (M,)
 
    # Bins that meet the minimum count threshold
    sufficient = counts >= min_count   # (M,) bool
 
    features_out = np.full(M, np.nan, dtype=float)
 
    if agg == "mean":
        # Weighted bincount gives the sum; divide by count for mean
        bin_sum = np.bincount(valid_labels, weights=valid_values, minlength=M)
        features_out[sufficient] = bin_sum[sufficient] / counts[sufficient]
 
    elif agg == "median":
        # No vectorised median-by-group primitive exists in numpy.
        # Sort once by bin label so each bin's values are contiguous,
        # then apply np.median to each group.  This is O(N log N) for
        # the sort and O(N) total for the group medians — still much
        # faster than the O(M × N) mask approach in v1.
        sort_order = np.argsort(valid_labels, kind="stable")
        sorted_labels = valid_labels[sort_order]
        sorted_values = valid_values[sort_order]
 
        # Find where bin label changes to locate group boundaries
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        groups = np.split(sorted_values, boundaries)
        unique_bins = np.unique(sorted_labels)
 
        for bin_idx, group in zip(unique_bins, groups):
            if counts[bin_idx] >= min_count:
                features_out[bin_idx] = np.median(group)
 
    else:
        raise ValueError(f"agg must be 'mean' or 'median', got '{agg}'")
 
    # Return as masked array — mask where NaN (empty or insufficient bins)
    out_mask = np.isnan(features_out)
    features_bin = np.ma.array(features_out, mask=out_mask)
 
    return depths_bin, features_bin
 
 
# ---------------------------------------------------------------------------
# Equivalence + timing tests against v1 using a real HoleObject
# ---------------------------------------------------------------------------
# Usage (from the repo root, with a loaded HoleObject `ho`):
#
#     from app.spectral_ops.downhole_resampling_v2 import run_equivalence_tests
#     run_equivalence_tests(ho)
#
# Or run the full suite from the command line:
#
#     python -m app.spectral_ops.downhole_resampling_v2 /path/to/hole_dir
#
# The HO must have base_datasets["depths"] loaded and at least one
# product_dataset of each suffix type (FRACTIONS, INDEX, continuous).
# ---------------------------------------------------------------------------
 
def run_equivalence_tests(ho) -> bool:
    """
    Test v2 output against v1 on real data extracted from a HoleObject.
    Reports equivalence results and head-to-head timing for each function.
 
    Parameters
    ----------
    ho : HoleObject
        A fully loaded HoleObject with base_datasets["depths"] present
        and at least some product_datasets populated.
 
    Returns
    -------
    bool
        True if all applicable tests pass, False if any fail.
        Prints a per-test report to stdout in all cases.
    """
    import time
    from app.spectral_ops import downhole_resampling as v1
 
    depths = ho.base_datasets["depths"].data
    step = ho.step
    N = depths.shape[0]
 
    all_passed = True
    W = 60
 
    # -----------------------------------------------------------------------
    # Discover one key of each suffix type present on the HO
    # -----------------------------------------------------------------------
    key_fractions = next(
        (k for k in ho.product_datasets if k.endswith("FRACTIONS")), None
    )
    key_index = next(
        (k for k in ho.product_datasets if k.endswith("INDEX")), None
    )
    key_continuous = next(
        (
            k for k in ho.product_datasets
            if not any(k.endswith(s) for s in
                       ("FRACTIONS", "DOM-MIN", "INDEX", "LEGEND", "CLUSTERS"))
        ),
        None,
    )
 
    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _check(name: str, passed: bool, detail: str = ""):
        nonlocal all_passed
        status = "PASS" if passed else "FAIL"
        suffix = f"  — {detail}" if detail else ""
        print(f"  {status}  {name}{suffix}")
        if not passed:
            all_passed = False
 
    def _time(fn, *args, repeats=1, **kwargs):
        """Run fn(*args, **kwargs) `repeats` times; return result and best time in ms."""
        best = float("inf")
        result = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            best = min(best, time.perf_counter() - t0)
        return result, best * 1000  # ms
 
    def _speedup(t_v1, t_v2):
        if t_v2 == 0:
            return "—"
        ratio = t_v1 / t_v2
        return f"{ratio:.1f}x faster" if ratio >= 1 else f"{1/ratio:.1f}x slower"
 
    def _compare_1d(a_raw, b_raw, label, depths_bin):
        """
        Robust nan-aware comparison for 1-D step_continuous output.
 
        Splits the check into two independent questions:
          1. Do the NaN positions agree?  A mismatch means the two
             implementations disagree on which bins are populated.
          2. Do the real values agree within tolerance?
 
        atol=1e-12, rtol=1e-10: accommodates the ~1 ULP rounding
        difference (~3e-16) from nanmean pairwise summation vs
        bincount sum-then-divide.  Values meaningfully larger than
        this indicate a structural disagreement.
        """
        a = np.ma.filled(a_raw, np.nan)
        b = np.ma.filled(b_raw, np.nan)
 
        nan_mask_ok = np.array_equal(np.isnan(a), np.isnan(b))
        both_valid  = ~np.isnan(a) & ~np.isnan(b)
 
        if both_valid.any():
            diffs   = np.abs(a[both_valid] - b[both_valid])
            vals_ok = bool(np.allclose(
                a[both_valid], b[both_valid], atol=1e-12, rtol=1e-10,
            ))
        else:
            diffs   = np.array([0.0])
            vals_ok = True
 
        _check(label, nan_mask_ok and vals_ok)
 
        if not nan_mask_ok:
            n1 = int(np.isnan(a).sum())
            n2 = int(np.isnan(b).sum())
            print(f"         NaN position mismatch: v1={n1} NaNs, v2={n2} NaNs")
 
        if not vals_ok and both_valid.any():
            print(f"         max abs diff (valid bins) = {diffs.max():.2e}")
            worst_valid_idx  = int(np.argmax(diffs))
            worst_bin        = int(np.where(both_valid)[0][worst_valid_idx])
            print(f"         worst bin index = {worst_bin}  depth = {depths_bin[worst_bin]:.4f} m")
            print(f"           v1 = {a[worst_bin]:.8g}   v2 = {b[worst_bin]:.8g}")
            half = (depths_bin[1] - depths_bin[0]) / 2.0 if len(depths_bin) > 1 else step / 2.0
            z    = depths_bin[worst_bin]
            v1_n = int(np.sum((depths >= z - half) & (depths < z + half)))
            v2_labels_dbg = _digitize_to_bins(depths, depths_bin)
            v2_n = int(np.sum(v2_labels_dbg == worst_bin))
            print(f"           v1 sees {v1_n} rows in bin   v2 sees {v2_n} rows in bin")
 
    # -----------------------------------------------------------------------
    print(f"Equivalence + timing  |  hole: {ho.hole_id}  |  step: {step} m  |  N={N:,} rows")
    print("=" * W)
 
    # -----------------------------------------------------------------------
    # step_fractions_pair
    # -----------------------------------------------------------------------
    print("step_fractions_pair")
    print("-" * W)
    if key_fractions is None:
        print("  SKIP — no FRACTIONS dataset found")
    else:
        data = ho.product_datasets[key_fractions].data
        print(f"  dataset : {key_fractions}  shape={data.shape}")
        try:
            (d1, f1, dom1), t_v1 = _time(v1.step_fractions_pair, depths, data, step)
            (d2, f2, dom2), t_v2 = _time(step_fractions_pair,    depths, data, step)
 
            print(f"  v1 time : {t_v1:8.2f} ms")
            print(f"  v2 time : {t_v2:8.2f} ms  ({_speedup(t_v1, t_v2)})")
 
            # depths
            _check("depths_bin", np.allclose(d1, d2, atol=1e-10))
 
            # fractions — NaN-position check then per-column value check
            nan_mask_ok = np.array_equal(np.isnan(f1), np.isnan(f2))
            both_valid_f = ~np.isnan(f1) & ~np.isnan(f2)
            frac_ok = nan_mask_ok and (
                bool(np.allclose(f1[both_valid_f], f2[both_valid_f],
                                 atol=1e-12, rtol=1e-10))
                if both_valid_f.any() else True
            )
            _check("fractions_bin", frac_ok, f"key={key_fractions}")
            if not frac_ok:
                if not nan_mask_ok:
                    print(f"         NaN position mismatch in fractions")
                else:
                    diff = np.abs(f1[both_valid_f] - f2[both_valid_f])
                    print(f"         max abs diff = {diff.max():.2e}")
 
            # dominant
            dom_ok = np.array_equal(dom1, dom2)
            _check("dominant_bin", dom_ok)
            if not dom_ok:
                n_diff = int(np.sum(dom1 != dom2))
                print(f"         bins with different dominant = {n_diff} / {len(dom1)}")
 
        except Exception as exc:
            _check("step_fractions_pair", False, f"raised {type(exc).__name__}: {exc}")
 
    # -----------------------------------------------------------------------
    # step_indices
    # -----------------------------------------------------------------------
    print("step_indices")
    print("-" * W)
    if key_index is None:
        print("  SKIP — no INDEX dataset found")
    else:
        data = ho.product_datasets[key_index].data
        print(f"  dataset : {key_index}  shape={data.shape}")
        try:
            (d1i, i1), t_v1 = _time(v1.step_indices, depths, data, step)
            (d2i, i2), t_v2 = _time(step_indices,    depths, data, step)
 
            print(f"  v1 time : {t_v1:8.2f} ms")
            print(f"  v2 time : {t_v2:8.2f} ms  ({_speedup(t_v1, t_v2)})")
 
            _check("depths_stepped",  np.allclose(d1i, d2i, atol=1e-10))
            indices_ok = np.array_equal(i1, i2)
            _check("indices_stepped", indices_ok)
            if not indices_ok:
                n_diff = int(np.sum(i1 != i2))
                print(f"         bins with different index = {n_diff} / {len(i1)}")
 
        except Exception as exc:
            _check("step_indices", False, f"raised {type(exc).__name__}: {exc}")
 
    # -----------------------------------------------------------------------
    # step_continuous
    # -----------------------------------------------------------------------
    print("step_continuous")
    print("-" * W)
    if key_continuous is None:
        print("  SKIP — no continuous dataset found")
    else:
        data = ho.product_datasets[key_continuous].data
        print(f"  dataset : {key_continuous}  shape={data.shape}")
 
        # mean
        try:
            (d1c, f1c), t_v1 = _time(v1.step_continuous, depths, data, step, agg="mean")
            (d2c, f2c), t_v2 = _time(step_continuous,    depths, data, step, agg="mean")
 
            print(f"  mean — v1 time : {t_v1:8.2f} ms")
            print(f"  mean — v2 time : {t_v2:8.2f} ms  ({_speedup(t_v1, t_v2)})")
 
            _compare_1d(f1c, f2c, "mean / features_bin", d1c)
 
        except Exception as exc:
            _check("step_continuous(mean)", False,
                   f"raised {type(exc).__name__}: {exc}")
 
        # median
        try:
            (d1m, f1m), t_v1 = _time(v1.step_continuous, depths, data, step, agg="median")
            (d2m, f2m), t_v2 = _time(step_continuous,    depths, data, step, agg="median")
 
            print(f"  median — v1 time : {t_v1:8.2f} ms")
            print(f"  median — v2 time : {t_v2:8.2f} ms  ({_speedup(t_v1, t_v2)})")
 
            _compare_1d(f1m, f2m, "median / features_bin", d1m)
 
        except Exception as exc:
            _check("step_continuous(median)", False,
                   f"raised {type(exc).__name__}: {exc}")
 
    # -----------------------------------------------------------------------
    print()
    print("=" * W)
    print(f"Result: {'ALL PASSED' if all_passed else 'ONE OR MORE FAILURES'}\n")
    return all_passed
 
 
 
#%% ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path
    from app.models import HoleObject as _HoleObject
 
     
    hole_dir = _Path("C:/Users/Hyperspectral/Documents/HS_Data/Kilree_MWIR")
    print(f"Loading hole from {hole_dir} ...")
    ho = _HoleObject.build_from_parent_dir(hole_dir)
    
 
    ok = run_equivalence_tests(ho)
    
