"""
Spectral analysis, mineral mapping, and classification for CoreSpecViewer.

All functions operate on numpy arrays and are UI-agnostic. This is the natural
home for new spectral algorithms. Imports cr and resample_spectrum from
spectral_processing for internal use.

Functions
---------
numpy_pearson               Vectorised Pearson correlation of a cube against
                            a 1D exemplar spectrum.
mineral_map_wta_strict      Winner-takes-all Pearson mineral mapping with
                            correlation threshold.
mineral_map_wta_sam_strict  Winner-takes-all Spectral Angle Mapping (SAM)
                            with angle threshold.
mineral_map_wta_msam_strict Winner-takes-all Modified SAM using continuum-
                            removed spectra.
mineral_map_subrange        Single spectral subrange correlation dispatcher —
                            delegates to wta_strict, sam_strict, or msam_strict.
mineral_map_multirange      Multi-window winner-takes-all mapping across user-
                            defined wavelength ranges. Least sensitive to
                            spectrally inactive minerals.
kmeans_spectral_wrapper     Unsupervised k-means clustering of spectral pixels.
Combined_MWL                Multi-window minimum wavelength feature detector.
                            Detects absorption feature position and depth across
                            defined spectral windows. Threshold read from AppConfig.
est_peaks_cube_scipy_thresh Per-pixel peak detection with SNR threshold, used
                            internally by Combined_MWL.

Dependencies
------------
Imports cr, resample_spectrum from spectral_processing.
Reads AppConfig for feature_detection_threshold.
"""

import logging

from numba import jit
import numpy as np
import scipy as sc
import spectral as sp
import hylite
from hylite.analyse import minimum_wavelength
from gfit.util import remove_hull

from .processing import remove_cont
from..config import config

logger = logging.getLogger(__name__)


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

        cube_cr = remove_cont(cube_slice)
        ex_cr = remove_cont(ex_slice)

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
    thresh = config.feature_detection_threshold
    feats = {
    '1400W':  (1387, 1445, 1350, 1450),
    '1480W':  (1471, 1491, 1440, 1520),
    '1550W':  (1520, 1563, 1510, 1610),
    '1760W':  (1751, 1764, 1730, 1790),
    '1850W':  (1749, 1949, 1720, 1980),
    '1900W':  (1840, 1990, 1820, 2010), 
    '2080W':  (1980, 2180, 1950, 2200), 
    '2160W':  (2159, 2166, 2138, 2179),
    '2200W':  (2185, 2215, 2120, 2245),
    '2250W':  (2248, 2268, 2230, 2280),
    '2290W':  (2279, 2310, 2270, 2350), 
    '2320W':  (2300, 2340, 2295, 2355),
    '2350W':  (2320, 2366, 2310, 2370),
    '2390W':  (2377, 2406, 2375, 2435),
    '2830W':  (2790, 2890, 2790, 2920),  
    '2950W':  (2920, 2980, 2900, 3000),
    '2950AW': (2900, 2960, 2900, 3000),
    '2950BW': (2920, 2990, 2790, 3200),
    '3000W':  (2900, 3100, 2795, 3900),  
    '3500W':  (3400, 3600, 3300, 3700), #NOT from Laukamp
    '4000W':  (3930, 4150, 3800, 4200),
    '4000WIDEW': (3910, 4150, 3800, 4200),
    '4000V_NARROWW': (3930, 4150, 3800, 4200),
    '4000shortW': (3850, 4000, 3800, 4200),
    '4470TRUEW': (4460, 4490, 4350, 4550),
    '4500SW': (4570, 4850, 4090, 5040),
    '4500CW': (4625, 4770, 4090, 5040),
    '4670W':  (4300, 4800, 4300, 4800),
    '4920W':  (4850, 5100, 4850, 5157),
    
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
    if isinstance(feature, dict):
        # Extract the single key-value pair
        if len(feature) != 1:
            raise ValueError("Feature dict must contain exactly one key-value pair")
        
        feature_name, bounds = next(iter(feature.items()))
        
        if len(bounds) != 4:
            raise ValueError("Feature bounds must be [wav_min, wav_max, cr_crop_min, cr_crop_max]")
        
        wav_min, wav_max, cr_crop_min, cr_crop_max = bounds
        
    elif isinstance(feature, str):
        # Existing path - look up in feats dictionary
        feature_name = feature
        cr_crop_min = feats[feature][2]
        cr_crop_max = feats[feature][3]
        wav_min = feats[feature][0]
        wav_max = feats[feature][1]
    else:
        raise TypeError("Feature must be either a string or a dictionary")
    
    # Common code for both paths
    cr_crop_min_index = np.argmin(np.abs(np.array(bands) - cr_crop_min))
    cr_crop_max_index = np.argmin(np.abs(np.array(bands) - cr_crop_max))
    wav_min_index = np.argmin(np.abs(np.array(bands) - wav_min))
    wav_max_index = np.argmin(np.abs(np.array(bands) - wav_max))
    

    #check_response =  est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min, wav_max))
    check_response =  est_peaks_cube_scipy_thresh(savgol_cr, bands, wavrange=(wav_min, wav_max), thresh = thresh)

    if technique.upper() == 'QND':
        logger.info(f"Using fit type {technique} for MWL")
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
        logger.info(f"Using fit type {technique} for MWL")
        hiswir = hylite.HyImage(savgol)
        hiswir.set_wavelengths(bands)
        Mpoly = minimum_wavelength( hiswir, float(cr_crop_min), float(cr_crop_max),
                                   n=1, method='poly', log=False, vb=False, minima=True)
        depth = Mpoly.__getitem__([0,'depth'])
        position = Mpoly.__getitem__([0,'pos'])
        width = Mpoly.__getitem__([0,'width'])
    elif technique.upper() == 'GAUS':
        logger.info(f"Using fit type {technique} for MWL")
        hiswir = hylite.HyImage(savgol)
        hiswir.set_wavelengths(bands)
        Mpoly = minimum_wavelength( hiswir, float(cr_crop_min), float(cr_crop_max),
                                   n=1, method='gaussian', log=False, vb=True, minima=True)
        depth = Mpoly.__getitem__([0,'depth'])
        position = Mpoly.__getitem__([0,'pos'])
        width = Mpoly.__getitem__([0,'width'])
    elif technique.upper() == 'QUAD':
        logger.info(f"Using fit type {technique} for MWL")
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
    

    return position, np.clip(depth, 0,1), feature_mask


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
            
            peak_heights = peak_dict['peak_heights']
           
            x = [bands[peak_indices[i]] for i in range(len(peak_indices)) if peak_heights[i] >thresh ]
           
            for k in x:
                if k > wavrange[0] and k < wavrange[1]:
                    arr[i,j] = k
                    break
                else:
                    arr[i,j] = -999
    return arr