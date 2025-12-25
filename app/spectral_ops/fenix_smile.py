"""
Fenix smile correction module. 
This submodule is for calculating fenix camera smile corrections. 
Preference is to use the hard-coded smile corrections in the hylite module,
however they appear to be instrument specifc.
This method is for performing the correction ONLY when no manufacturer provided 
calibration is avaliable.
If you have the instrument specific calpack, either hack the values into hylite, 
or write your own correction function to be called when Fenix is recognised in the gui
(see spectral_function.get_fenix_reflectance in the spectral_ops module).
"""

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter1d
from scipy.signal import correlate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional

def calculate_smile_correction(
    image: np.ndarray,
    reference_band: int = 20,
    n_bands_avg: int = 5,
    poly_degree: int = 3,
    max_shift_pixels: int = 45,
    min_confidence_ratio: float = 0.5,
    threshold_std: float = 3.0,
    smooth_sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate robust smile correction with improved correlation and outlier handling.
    
    Key improvements:
    - Normalized cross-correlation for better peak detection
    - Gaussian smoothing to reduce noise sensitivity
    - Adaptive confidence thresholding
    - Subpixel shift estimation
    
    Parameters:
    - image: 3D array (rows, cols, bands)
    - reference_band: band with clear features
    - n_bands_avg: number of bands to average (increased default for stability)
    - poly_degree: polynomial degree (3 is typically sufficient)
    - max_shift_pixels: initial outlier threshold
    - min_confidence_ratio: minimum confidence threshold
    - threshold_std: outlier rejection threshold (lowered to 3.0)
    - smooth_sigma: Gaussian smoothing parameter for noise reduction
    
    Returns:
    - fitted_shifts: spectral shift correction for every row
    - poly_coeffs_final: final polynomial coefficients
    """
    
    # Extract and average bands with bounds checking
    start_band = max(0, reference_band - n_bands_avg)
    end_band = min(image.shape[2], reference_band + n_bands_avg)
    band_slice = image[:, :, start_band:end_band]
    band_image = band_slice.mean(axis=2)
    
    # Apply gentle smoothing to reduce noise
    if smooth_sigma > 0:
        band_image = gaussian_filter1d(band_image, sigma=smooth_sigma, axis=1)
    
    # Use median row as reference (more robust than mean)
    ref_row = image.shape[0] // 2
    reference = band_image[ref_row, :]
    
    # Normalize reference for better correlation
    reference = (reference - np.mean(reference)) / (np.std(reference) + 1e-10)
    
    shifts = []
    confidences = []
    
    for row in range(image.shape[0]):
        row_data = band_image[row, :]
        
        # Normalize row data
        row_data_norm = (row_data - np.mean(row_data)) / (np.std(row_data) + 1e-10)
        
        # Cross-correlation with normalized data
        corr = correlate(row_data_norm, reference, mode='same', method='fft')
        
        center = len(corr) // 2
        peak_idx = np.argmax(corr)
        
        # Subpixel shift estimation using parabolic interpolation
        if 0 < peak_idx < len(corr) - 1:
            # Parabolic fit around peak for subpixel accuracy
            y1, y2, y3 = corr[peak_idx-1], corr[peak_idx], corr[peak_idx+1]
            subpixel_offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3) if (y1 - 2*y2 + y3) != 0 else 0
            shift = (peak_idx - center) + subpixel_offset
        else:
            shift = peak_idx - center
        
        # Calculate confidence as normalized correlation coefficient
        confidence = corr[peak_idx] / (np.sqrt(np.sum(corr**2)) + 1e-10)
        
        shifts.append(shift)
        confidences.append(confidence)
    
    shifts = np.array(shifts)
    confidences = np.array(confidences)
    rows = np.arange(len(shifts))
    
    # === Enhanced Outlier Rejection ===
    # Use median absolute deviation (MAD) for robust statistics
    median_shift = np.median(shifts)
    mad = np.median(np.abs(shifts - median_shift))
    robust_std = 1.4826 * mad  # MAD to std conversion
    
    # Initial filtering with both hard limits and statistical bounds
    initial_valid_indices = (
        (np.abs(shifts) < max_shift_pixels) &
        (confidences > (min_confidence_ratio * np.max(confidences))) &
        (np.abs(shifts - median_shift) < 5 * robust_std)
    )
    
    if np.sum(initial_valid_indices) < poly_degree + 2:
        raise ValueError(f"Insufficient valid points ({np.sum(initial_valid_indices)}) for polynomial fit")
    
    rows_filt = rows[initial_valid_indices]
    shifts_filt = shifts[initial_valid_indices]
    weights_filt = confidences[initial_valid_indices]
    weights_filt = weights_filt / np.max(weights_filt)
    
    # Initial polynomial fit
    poly_coeffs = np.polyfit(rows_filt, shifts_filt, deg=poly_degree, w=weights_filt)
    fitted_shifts_initial = np.polyval(poly_coeffs, rows)
    
    # Calculate residuals and adaptive threshold
    residuals = np.abs(shifts - fitted_shifts_initial)
    std_residuals = np.std(residuals[initial_valid_indices])
    
    # Final filtering with adaptive threshold
    final_valid_indices = (residuals < (threshold_std * std_residuals))
    
    if np.sum(final_valid_indices) < poly_degree + 2:
        # Fall back to initial filtering if too aggressive
        final_valid_indices = initial_valid_indices
    
    rows_final = rows[final_valid_indices]
    shifts_final = shifts[final_valid_indices]
    weights_final = confidences[final_valid_indices] / np.max(confidences)
    
    # Final polynomial fit
    poly_coeffs_final = np.polyfit(rows_final, shifts_final, deg=poly_degree, w=weights_final)
    fitted_shifts = np.polyval(poly_coeffs_final, rows)
    
    return fitted_shifts, poly_coeffs_final


def calculate_smile_multiband(
    image: np.ndarray,
    band_step: int = 8,
    use_robust_average: bool = True
) -> np.ndarray:
    """
    Calculate smile using multiple bands with improved robustness.
    
    Improvements:
    - Adaptive band selection based on image quality
    - Robust averaging (median or trimmed mean)
    - Better error handling
    - Optimized band sampling
    
    Parameters:
    - image: 3D hyperspectral image
    - band_step: step size for band sampling (smaller = more bands, slower)
    - use_robust_average: use median instead of mean for averaging
    
    Returns:
    - m: array with rows and fitted shifts
    """
    all_shifts = []
    all_quality = []
    
    # Sample bands more intelligently
    n_bands = image.shape[2]
    band_indices = range(n_bands // 10, n_bands - n_bands // 10, band_step)
    
    for band in band_indices:
        if band >= n_bands - 5:  # Need buffer for averaging
            continue
        
        try:
            shifts, coeffs = calculate_smile_correction(
                image, 
                reference_band=band, 
                n_bands_avg=5,
                poly_degree=3,
                threshold_std=3.0
            )
            
            # Estimate quality based on smoothness of shifts
            quality = 1.0 / (np.std(np.diff(shifts)) + 0.01)
            
            all_shifts.append(shifts)
            all_quality.append(quality)
            
        except (np.linalg.LinAlgError, ValueError, RuntimeError, TypeError):
            continue
    
    if len(all_shifts) == 0:
        raise ValueError("No valid bands found for smile correction")
    
    # Convert to arrays for easier manipulation
    all_shifts = np.array(all_shifts)
    all_quality = np.array(all_quality)
    
    # Weight by quality and use robust averaging
    if use_robust_average:
        # Use weighted median or trimmed mean
        mean_shifts = np.average(all_shifts, axis=0, weights=all_quality)
        # Remove outlier bands (top and bottom 10%)
        sorted_indices = np.argsort(all_shifts, axis=0)
        trim = max(1, len(all_shifts) // 10)
        trimmed_shifts = all_shifts[sorted_indices[trim:-trim, 0], :]
        mean_shifts = np.median(trimmed_shifts, axis=0)
    else:
        mean_shifts = np.average(all_shifts, axis=0, weights=all_quality)
    
    # Smooth the final result
    mean_shifts = gaussian_filter1d(mean_shifts, sigma=2.0)
    
    # Fit final curve with lower degree for stability
    rows = np.arange(len(mean_shifts))
    
    try:
        # Use degree 2 or 3 for final fit (more stable)
        valid_mask = ~np.isnan(mean_shifts) & np.isfinite(mean_shifts)
        if np.sum(valid_mask) < 4:
            raise ValueError("Too few valid points for final fit")
        
        poly_coeffs = np.polyfit(rows[valid_mask], mean_shifts[valid_mask], deg=2)
        fitted_shifts = np.polyval(poly_coeffs, rows)
        
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use smoothed mean directly
        fitted_shifts = mean_shifts
    
    # Create output array
    m = np.zeros((len(fitted_shifts), 2))
    m[:, 0] = rows
    m[:, 1] = fitted_shifts
    
    return m


def fenix_smile_correction(image: np.ndarray) -> np.ndarray:
    """
    Apply smile correction to Fenix hyperspectral image.
    Code is modified from hylite.sensors.fenix, with m (horizontal offset array) being derived
    from the image as opposed to using manufacturer supplied values.
    NB. This method is less effective than using manufacturer calpack, or more robustly tested smiles,
    it may over or underfit.
    
    Parameters:
    - image: ndarray, will auto-rotate if needed
    
    Returns:
    - remap: corrected image with smile removed
    """
    # Auto-rotate if needed (prefer wider dimension as columns)
    needs_transpose = image.shape[0] > image.shape[1]
    if needs_transpose:
        image = np.transpose(image, (1, 0, 2))
    
    # Calculate smile correction
    m = calculate_smile_multiband(image, band_step=10)
    
    # Create displacement map
    dmap = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
    dmap[:, :, 0] = -m[:, None, 1]  # x displacements
    dmap[:, :, 0] -= np.min(-m[:, 1])  # ensure non-negative
    dmap[:, :, 1] = (m[:, 0] - np.arange(image.shape[0]))[:, None]  # y displacements
    
    # Calculate output dimensions
    width = int(image.shape[1] + np.max(m[:, 1]) - np.min(m[:, 1]))
    height = int(np.ceil(np.max(m[:, 0])) + 1)
    
    # Resize displacement map
    dmap = cv2.resize(dmap, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Create coordinate mappings
    xx, yy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
    idx = np.dstack([xx, yy]).astype(np.float32)
    idx[:, :, 0] -= dmap[:, :, 0]
    idx[:, :, 1] -= dmap[:, :, 1]
    
    # Apply remapping with band chunking if necessary
    max_bands_per_chunk = 500
    
    if image.shape[-1] <= max_bands_per_chunk:
        remap = cv2.remap(image, idx, None, cv2.INTER_LINEAR)
    else:
        # Process in chunks
        remap_chunks = []
        for start_band in range(0, image.shape[-1], max_bands_per_chunk):
            end_band = min(start_band + max_bands_per_chunk, image.shape[-1])
            chunk = cv2.remap(image[:, :, start_band:end_band], idx, None, cv2.INTER_LINEAR)
            remap_chunks.append(chunk)
        remap = np.dstack(remap_chunks)
    
    return remap




