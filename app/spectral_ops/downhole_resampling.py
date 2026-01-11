"""
Small module for handing re-sampling of full hole datasets.
Desinged for use with mineral maps and spectral feature products.
May be adapted and extended to handle other downhole datasets
"""
import numpy as np

def step_fractions_pair(
    depths_row: np.ndarray,
    fractions_row: np.ndarray,  # (N, K+1)
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin row-based fractions into regular depth intervals and recompute dominant.
    
    Now returns a complete regular grid with NaN for gaps (bins with no data).
    
    Parameters
    ----------
    depths_row : (N,)
        Depth per unwrapped row (same length as fractions_row).
    fractions_row : (N, K+1)
        Fractions per row (output of compute_fullhole_mineral_fractions after 
        concatenating all boxes). Columns 0..K-1 are legend classes, 
        column K is 'unclassified'.
    step : float
        Bin size in depth units (e.g. 0.05 for 5 cm).
    
    Returns
    -------
    depths_bin : (M,)
        Regular grid of depth bin centres from d_min to d_max with spacing step.
    fractions_bin : (M, K+1)
        Mean fractions over rows falling into each bin. NaN for bins with no data.
    dominant_bin : (M,)
        Dominant mineral per bin, as an index 0..K-1 into the legend.
        -1 if no classified pixels in that bin OR if the bin is a gap (no data).
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
    
    # Create regular grid of bin centres
    depths_bin = np.arange(d_min, d_max + step / 2.0, step)
    M = depths_bin.shape[0]
    
    # Pre-allocate outputs with NaN/default values
    fractions_bin = np.full((M, C), np.nan, dtype=float)
    dominant_bin = np.full(M, -1, dtype=int)
    
    half = step / 2.0
    K = C - 1  # last column = unclassified
    
    for i, z in enumerate(depths_bin):
        lo = z - half
        hi = z + half
        mask = (depths_row >= lo) & (depths_row < hi)
        
        if not np.any(mask):
            # No rows in this interval → stays NaN, dominant stays -1
            continue
        
        # Compute mean fractions for this bin
        frac_mean = np.nanmean(frac[mask, :], axis=0)
        fractions_bin[i, :] = frac_mean
        
        # Determine dominant mineral from classified fractions only
        classified = frac_mean[:K]
        if classified.sum() > 0 and not np.all(np.isnan(classified)):
            dominant_bin[i] = int(np.argmax(classified))
        else:
            dominant_bin[i] = -1
    
    return depths_bin, fractions_bin, dominant_bin


def step_indices(depths, indices, step):
    """
    Step INDEX data using mode aggregation.
    
    Args:
        depths: Original sample depths (1D)
        indices: Categorical indices (1D integers)
        step: Depth interval for binning
    
    Returns:
        depths_stepped: Regular depth grid (1D)
        indices_stepped: Most common index per bin (1D integers)
    """
    from scipy import stats
    
    d_min = depths.min()
    d_max = depths.max()
    
    # Use + step/2.0 to ensure d_max is covered
    depths_stepped = np.arange(d_min, d_max + step / 2.0, step)
    indices_stepped = np.full(len(depths_stepped), -1, dtype=np.int16)
    
    for i, d in enumerate(depths_stepped):
        mask = (depths >= d - step/2) & (depths < d + step/2)
        bin_indices = indices[mask]
        
        if len(bin_indices) > 0:
            indices_stepped[i] = stats.mode(bin_indices, keepdims=True)[0][0]
    
    return depths_stepped, indices_stepped


def step_continuous(
        depths_row: np.ndarray,
        features_row: np.ndarray,  # Can be regular array or masked array
        step: float,
        agg: str = "mean",
        min_count: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bin row-based features into regular depth intervals of size step.
        
        Now returns a complete regular grid with NaN for gaps (bins with insufficient data).
        Properly handles numpy masked arrays, propagating the mask through aggregation.
        
        Parameters
        ----------
        depths_row : (N,)
            Depth for each row (same length as features_row).
        features_row : (N,) or (N, F), array or masked array
            Features per row (e.g. band position, depth, strength, etc.).
            If masked array, masked values are treated as invalid.
            NaNs are also treated as missing values.
        step : float
            Bin size in depth units (e.g. 0.05 for 5 cm).
        agg : {"mean", "median"}
            Aggregation function to apply within each bin.
        min_count : int
            Minimum number of valid (unmasked, non-NaN) samples required in a bin 
            to produce a value. Bins with fewer valid samples are set to NaN.
        
        Returns
        -------
        depths_bin : (M,)
            Regular grid of depth bin centres from d_min to d_max with spacing step.
        features_bin : (M,) or (M, F), masked array
            Binned features as a masked array. Masked entries indicate bins with 
            insufficient valid data or gaps. Same feature dimension as features_row.
        """
        depths_row = np.asarray(depths_row, dtype=float)
        
        # Check if input is a masked array
        is_masked = np.ma.isMaskedArray(features_row)
        
        if is_masked:
            feats = np.ma.asarray(features_row, dtype=float)
        else:
            feats = np.asarray(features_row, dtype=float)
        
        if depths_row.ndim != 1:
            raise ValueError("depths_row must be 1D.")
        
        N = depths_row.shape[0]
        
        # Handle 1D vs 2D features
        if feats.ndim == 1:
            feats = feats.reshape(N, 1)
            squeeze = True
        elif feats.ndim == 2 and feats.shape[0] == N:
            squeeze = False
        else:
            raise ValueError("features_row must be shape (N,) or (N, F).")
        
        N, F = feats.shape
        
        if N == 0:
            empty_result = np.ma.masked_all((0, F) if not squeeze else (0,), dtype=float)
            return np.zeros((0,), dtype=float), empty_result
        
        if step <= 0:
            raise ValueError("step must be positive.")
        
        if agg not in ("mean", "median"):
            raise ValueError("agg must be 'mean' or 'median'.")
        
        d_min = depths_row.min()
        d_max = depths_row.max()
        
        # Create regular grid of bin centres
        depths_bin = np.arange(d_min, d_max + step / 2.0, step)
        M = depths_bin.shape[0]
        
        # Pre-allocate output as masked array with everything masked initially
        features_bin = np.ma.masked_all((M, F), dtype=float)
        
        half = step / 2.0
        
        for i, z in enumerate(depths_bin):
            lo = z - half
            hi = z + half
            depth_mask = (depths_row >= lo) & (depths_row < hi)
            
            if not np.any(depth_mask):
                # No rows in this interval stays masked
                continue
            
            block = feats[depth_mask, :]  # (n_bin_rows, F)
            
            # Determine valid samples: not masked AND not NaN
            if is_masked:
                # For masked arrays, combine the mask with NaN checks
                valid_mask = ~(np.ma.getmaskarray(block) | np.isnan(np.ma.filled(block, np.nan)))
            else:
                # For regular arrays, only check NaN
                valid_mask = ~np.isnan(block)
            
            valid_counts = valid_mask.sum(axis=0)  # (F,)
            
            # Aggregate only if we have enough valid samples
            if agg == "mean":
                # Use masked array mean if available, otherwise nanmean
                if is_masked:
                    means = np.ma.mean(block, axis=0)
                    # Check if we have enough valid counts
                    for j in range(F):
                        if valid_counts[j] >= min_count:
                            features_bin[i, j] = means[j]
                        # else: stays masked
                else:
                    sums = np.nansum(block, axis=0)
                    with np.errstate(invalid="ignore"):
                        vals = np.where(
                            valid_counts >= min_count,
                            sums / valid_counts,
                            np.nan
                        )
                    for j in range(F):
                        if valid_counts[j] >= min_count:
                            features_bin[i, j] = vals[j]
                        # else: stays masked
                        
            elif agg == "median":
                for j in range(F):
                    col = block[:, j]
                    # Get valid (unmasked, non-NaN) values
                    if is_masked:
                        col_valid = np.ma.compressed(col)
                        # Also filter out NaNs that might exist in the data
                        col_valid = col_valid[~np.isnan(col_valid)]
                    else:
                        col_valid = col[~np.isnan(col)]
                    
                    if col_valid.size >= min_count:
                        features_bin[i, j] = np.median(col_valid)
                    # else: stays masked
        
        if squeeze:
            features_bin = features_bin[:, 0]
        
        return depths_bin, features_bin
