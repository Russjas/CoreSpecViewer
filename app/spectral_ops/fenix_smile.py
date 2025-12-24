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
from scipy.ndimage import correlate1d
from scipy.signal import correlate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cv2

def calculate_smile_correction(image, reference_band=20, n_bands_avg=5, poly_degree=3, max_shift_pixels=45, min_confidence_ratio=0.5, threshold_std=3.0):
    """
    Calculate robust smile correction by tracking horizontal features across rows
    using an iterative, two-pass polynomial fit with statistical outlier rejection.
    
    Parameters:
    - image: 3D array (rows, cols, bands)
    - reference_band: band with clear features
    - n_bands_avg: number of bands to average for stability
    - poly_degree: degree of the polynomial to fit (e.g., 4 or 5)
    - max_shift_pixels: Initial threshold to reject very extreme shifts.
    - min_confidence_ratio: Minimum confidence (relative to max) required for a point to be included in the initial fit.
    - threshold_std: Statistical threshold (in standard deviations of residuals) for final outlier rejection.
    
    Returns:
    - fitted_shifts: The calculated spectral shift correction curve for every row.
    - poly_coeffs_final: The coefficients of the final fitted polynomial.
    """
    
   
    band_slice = image[:, :, reference_band-n_bands_avg:reference_band+n_bands_avg]
    band_image = band_slice.mean(axis=2)
    ref_row = image.shape[0] // 2
    reference = band_image[ref_row, :]
    
    
    shifts = []
    confidences = []
    
    for row in range(image.shape[0]):
        row_data = band_image[row, :]
        corr = correlate(row_data, reference, mode='same', method='fft')
        
        center = len(corr) // 2
        peak_idx = np.argmax(corr)
        shift = peak_idx - center
        confidence = corr[peak_idx] / np.mean(corr)
        
        shifts.append(shift)
        confidences.append(confidence)
    
    shifts = np.array(shifts)
    confidences = np.array(confidences)
    rows = np.arange(len(shifts))
    
    # === Outlier Rejection and Fitting ===
    initial_valid_indices = (np.abs(shifts) < max_shift_pixels) & \
                            (confidences > (min_confidence_ratio * np.max(confidences)))

    rows_filt = rows[initial_valid_indices]
    shifts_filt = shifts[initial_valid_indices]
    
    weights_filt = confidences[initial_valid_indices] / np.max(confidences)
    poly_coeffs = np.polyfit(rows_filt, shifts_filt, deg=poly_degree, w=weights_filt)
    fitted_shifts_initial = np.polyval(poly_coeffs, rows)

    
    # Calculate absolute residuals for all points from the initial fit
    residuals = np.abs(shifts - fitted_shifts_initial)
    
    # Calculate the standard deviation of residuals using the initially filtered points (inliers)
    std_residuals = np.std(residuals[initial_valid_indices]) 

    # Final set of valid indices: points that are close to the estimated smooth curve
    final_valid_indices = (residuals < (threshold_std * std_residuals))

    rows_final = rows[final_valid_indices]
    shifts_final = shifts[final_valid_indices]
    
    # Normalize and use weights for the final pass
    weights_final = confidences[final_valid_indices] / np.max(confidences)

    #Final Pass Fit 
    poly_coeffs_final = np.polyfit(rows_final, shifts_final, deg=poly_degree, w=weights_final)

    # Evaluate the final fitted curve on ALL rows
    fitted_shifts = np.polyval(poly_coeffs_final, rows)

    return fitted_shifts, poly_coeffs_final

def calculate_smile_multiband(image):
    """
    Calculate smile using multiple bands and average results for robustness
    """
    all_shifts = []
    
    for band in range(0,image.shape[-1], 10):#range(35,80, 5):#
        #print(band)
        if band >= image.shape[2]:
            continue
        
        try:
            shifts, _ = calculate_smile_correction(image, reference_band=band, n_bands_avg=3)
            all_shifts.append(shifts)
        except (np.linalg.LinAlgError, ValueError, RuntimeError, TypeError):
            # Skip bands that fail due to NaNs or other issues
            # maximising the number of bands used 
            continue
    
    if len(all_shifts) == 0:
        raise ValueError("No valid bands found for smile correction")
    
    # Average across bands
    mean_shifts = np.mean(all_shifts, axis=0)
    std_shifts = np.std(all_shifts, axis=0)
    
    # Fit final curve
    rows = np.arange(len(mean_shifts))
    
    try:
        poly_coeffs = np.polyfit(rows, mean_shifts, deg=2)
        fitted_shifts = np.polyval(poly_coeffs, rows)
    except (np.linalg.LinAlgError, ValueError):
        # If polyfit fails, try with NaN removal or lower degree
        valid_mask = ~np.isnan(mean_shifts)
        poly_coeffs = np.polyfit(rows[valid_mask], mean_shifts[valid_mask], deg=3)
        fitted_shifts = np.polyval(poly_coeffs, rows)
    
    m = np.zeros((len(fitted_shifts), 2))
    m[:, 0] = rows
    m[:, 1] = fitted_shifts
        
    return m

def fenix_smile_correction(image):
    '''
    Code is lifted intact from hylite.sensors.fenix, with m (horizontal offset array being derived
    from the image as opposed to using manufacturer supplied values.
    NB. This method is less effective than using manufacturer calpack, or more robustly tested smiles,
    it may over or underfit.
    
    Parameters
    ----------
    image : ndarray, rotated. 
        Expects horizontal smile.

    Returns
    image : ndarray, vertical
        Remapped array with the smile removed.

    '''
    # rough heuristic to ensure image is flipped
    if image.shape[0] > image.shape[1]:
        image = np.transpose(image, (1, 0, 2))
    m = calculate_smile_multiband(image)

    dmap = np.zeros((image.shape[0], image.shape[1], 2))
    dmap[:, :, 0] += -m[:, None, 1]  # displacements in x
    dmap[:, :, 0] -= np.min(-m[:, 1])  # avoid negative displacements
    dmap[:, :, 1] += (m[:, 0] - np.arange(image.shape[0]))[:, None]  # displacements in y

    # calculate width/height of corrected image
    width = int(image.shape[1] + np.max(m[:, 1]) - np.min(m[:, 1]))
    height = int(np.ceil(np.max(m[:, 0])))

    # resize displacement map to output dimensions
    dmap = cv2.resize(dmap, (width, height), cv2.INTER_LINEAR)

    # use displacement vectors to calculate mapping from output coordinates to original coordinates
    xx, yy = np.meshgrid(range(dmap.shape[1]), range(dmap.shape[0]))
    idx = np.dstack([xx, yy]).astype(np.float32)
    idx[:, :, 0] -= dmap[:, :, 0]
    idx[:, :, 1] -= dmap[:, :, 1]
    
    # apply remapping
    if image.shape[-1] < 512:  # open-cv cannot handle more than 512 bands at a time
        remap = cv2.remap(image, idx, None, cv2.INTER_LINEAR)
    else:  # we need to split into different stacks with < 512 bands and then recombine
        remap = []
        mn = 0
        mx = 500
        while mn < image.shape[-1]:
            if mx > image.shape[-1]:
                mx = image.shape[-1]
    
            # apply mapping to slice of bands
            remap.append(cv2.remap(image[:, :, mn:mx], idx, None, cv2.INTER_LINEAR))
    
            # get next slice
            mn = mx
            mx += 500
    
        # stack
        remap = np.dstack(remap)
    return remap

