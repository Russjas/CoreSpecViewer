"""
Mask generation and refinement tools for CoreSpecViewer.

Operates on binary spatial masks (H x W uint8, 1 = masked) and hyperspectral
cubes. Provides automated core segmentation from false-colour images and
post-processing refinements. No GUI dependencies.

Functions
---------
detect_slice_rectangles_robust  Detect core cylinder rectangles in a false-colour
                                image using contour detection and IoU filtering.
                                Primary automated crop/mask entry point.
get_stats_from_mask             Run connected-component analysis on a mask and
                                return per-component statistics for downhole
                                unwrapping.
improve_mask_from_graph         Heuristically thicken a mask column-wise using
                                spatial occupancy — removes isolated holes.
despeckle_mask                  Remove isolated single-pixel artefacts from a mask.

Notes
-----
detect_slice_rectangles_robust is the cv2-based fallback used for distribution.
ML-based segmentation (YOLO, SAM) is available separately but not shipped as
part of this module due to hardware and environment requirements.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def detect_crop_from_variance(src, win_frac=0.02, noise_floor_frac=0.02):
    """
    Derive a four-edge crop slice purely from sliding window variance profiles,
    with no marker detection.
 
    Computes column-wise and row-wise mean intensity profiles across the full
    image, then finds the outermost positions where local variance is still
    above a noise floor. Core content has high variance; dark uniform borders
    do not.
 
    Parameters
    ----------
    src : np.ndarray, shape (H, W, 3), dtype uint8
        Input image, already contrast-stretched.
    win_frac : float, optional
        Sliding window size as a fraction of the relevant image dimension
        (width for columns, height for rows). Default 0.02.
    noise_floor_frac : float, optional
        Fraction of peak variance below which a position is considered border.
        Default 0.02.
 
    Returns
    -------
    crop_slice : tuple of slice or None
        (slice(y_top, y_bottom), slice(x_left, x_right)), or None if the
        variance profile is flat across the entire image.
    """
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float64)
    H, W = gray.shape
 
    def sliding_variance(signal, win):
        n = len(signal)
        var = np.zeros(n)
        half = win // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half)
            var[i] = np.var(signal[lo:hi])
        return var
 
    def first_and_last_active(var_profile, noise_floor_frac):
        threshold = var_profile.max() * noise_floor_frac
        active = np.where(var_profile > threshold)[0]
        if len(active) == 0:
            return None, None
        return active[0], active[-1]
 
    # Column profile — mean across all rows per column
    col_signal = gray.mean(axis=0)
    col_win = max(8, int(W * win_frac))
    col_var = sliding_variance(col_signal, col_win)
    x0, x1 = first_and_last_active(col_var, noise_floor_frac)
 
    if x0 is None:
        logger.warning("detect_crop_from_variance: column profile flat, cannot crop")
        return None
 
    # Row profile — mean across detected x range per row
    row_signal = gray[:, x0:x1 + 1].mean(axis=1)
    row_win = max(8, int(H * win_frac))
    row_var = sliding_variance(row_signal, row_win)
    y0, y1 = first_and_last_active(row_var, noise_floor_frac)
 
    if y0 is None:
        logger.warning("detect_crop_from_variance: row profile flat, cannot crop")
        return None
 
    logger.info(f"detect_crop_from_variance: crop y={y0}:{y1+1}, x={x0}:{x1+1}")
    return (slice(y0, y1 + 1), slice(x0, x1 + 1))


def detect_slice_rectangles(img_eq,
                             min_area_frac=0.0005,
                             canny_sigma=0.33,
                             approx_eps_frac=0.02,
                             close_kernel=5,
                             use_otsu=True,
                             allow_rotated=True):
    """
    Detect the largest rectangular content region in a contrast-stretched
    image using edge-based contour extraction.
 
    Applies bilateral filtering, optional Otsu binarization, and Canny edge
    detection. Extracts convex quadrilateral contours and optionally rotated
    bounding boxes above a minimum area threshold. Non-maximum suppression
    (IoU > 0.6) removes overlapping candidates and the largest surviving
    rectangle is returned.
 
    Parameters
    ----------
    img_eq : np.ndarray, shape (H, W, 3), dtype uint8
        Contrast-stretched input image. Contrast stretching is the
        caller's responsibility.
    min_area_frac : float, optional
        Minimum rectangle area as a fraction of total image area. Default 0.0005.
    canny_sigma : float, optional
        Adaptive Canny threshold factor from median intensity. Default 0.33.
    approx_eps_frac : float, optional
        Polygon approximation tolerance as a fraction of contour perimeter
        (cv2.approxPolyDP). Default 0.02.
    close_kernel : int, optional
        Morphological closing kernel size in pixels. Set 0 or 1 to disable.
        Default 5.
    use_otsu : bool, optional
        Apply Otsu binarization before Canny. Default True.
    allow_rotated : bool, optional
        Accept rotated bounding boxes (cv2.minAreaRect) when no convex
        quadrilaterals are found. Default True.
 
    Returns
    -------
    crop_slice : tuple of slice or None
        (slice(y, y+h), slice(x, x+w)) in img_eq coordinates, or None if
        no rectangle was found.
    """
    H, W = img_eq.shape[:2]
    min_area = H * W * min_area_frac
 
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
 
    src = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if use_otsu else gray
    v = np.median(src)
    lo = int(max(0, (1.0 - canny_sigma) * v))
    hi = int(min(255, (1.0 + canny_sigma) * v))
    edges = cv2.Canny(src, lo, hi)
 
    if close_kernel and close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects_xywh, rects_poly = [], []
 
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_eps_frac * peri, True)
 
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            rects_xywh.append((x, y, w, h))
            rects_poly.append(approx[:, 0, :].astype(np.float32))
        elif allow_rotated:
            r = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = r
            if rw * rh >= min_area:
                box = cv2.boxPoints(r).astype(np.float32)
                x, y, w, h = cv2.boundingRect(box.astype(np.int32))
                rects_xywh.append((x, y, w, h))
                rects_poly.append(box)
 
    def iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / (aw * ah + bw * bh - inter + 1e-9)
 
    keep = []
    used = [False] * len(rects_xywh)
    order = sorted(range(len(rects_xywh)), key=lambda i: rects_xywh[i][2] * rects_xywh[i][3], reverse=True)
    for i in order:
        if used[i]:
            continue
        keep.append(i)
        for j in order:
            if used[j] or j == i:
                continue
            if iou(rects_xywh[i], rects_xywh[j]) > 0.6:
                used[j] = True
 
    rects_xywh = [rects_xywh[i] for i in keep]
    rects_poly  = [rects_poly[i]  for i in keep]
 
    if not rects_xywh:
        logger.info("detect_slice_rectangles: no rectangles found")
        return None
 
    areas = np.array([cv2.contourArea(p) for p in rects_poly])
    x, y, w, h = rects_xywh[np.argmax(areas)]
    return (slice(y, y + h), slice(x, x + w))
 

def auto_crop(image, mode='references'):
    """
    Auto-crop a core scan image to its content region.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), dtype uint8
        Input BGR/RGB image.
    mode : str, optional
        Crop strategy. One of:
            'references' : detect physical reference markers (GSI Sisu-rock
                           scanner). Falls back to 'variance' if markers not found.
            'variance'   : sliding window variance profiles on all four edges.
            'rectangles' : contour-based rectangle detection.
        Default 'references'.

    Returns
    -------
    cropped : np.ndarray
    crop_slice : tuple of slice or None
    """
    p_low, p_high = np.percentile(image, (15, 85))
    img_eq = np.clip((image - p_low) / (p_high - p_low), 0, 1)
    img_eq = (img_eq * 255).astype(np.uint8)

    if mode == 'variance':
        crop_slice = detect_crop_from_variance(image)

    elif mode == 'rectangles':
        crop_slice = detect_slice_rectangles(img_eq)

    else:
        raise ValueError(f"auto_crop: unknown mode '{mode}'")

    if crop_slice is None:
        logger.warning(f"auto_crop: mode '{mode}' returned no crop")
        return image, None

    return image[crop_slice], crop_slice



def get_stats_from_mask(mask, proportion=16, iters=2):
    """
    Compute connected components on the (eroded) inverse of a mask.

    Parameters
    ----------
    mask : ndarray of {0,1}
        Binary mask with 1 = non-core region.
    proportion : int, optional
        (Unused here) placeholder for future scaling.
    iters : int, optional
        Erosion iterations applied to the inverse mask before labeling.

    Returns
    -------
    labels : ndarray
        Labeled image (int) of connected components.
    stats : ndarray, shape (N, 5)
        Component stats from OpenCV: (x, y, width, height, area).
    """

    inv_mask = 1-mask.astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    erod_im = cv2.erode(inv_mask, kernel, anchor=(0, 0), iterations=iters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erod_im.astype(np.uint8), connectivity=8)
    return labels, stats


def improve_mask_from_graph(mask):
    """
    Heuristically thicken a mask column-wise using simple occupancy.

    Parameters
    ----------
    mask : ndarray of {0,1}
        Binary mask image with 1s indicating core.

    Returns
    -------
    ndarray
        A copy of `mask` where columns with sufficient occupancy
        (sum above ~H/3) are set to 1 across all rows.  
    """

    line = np.sum(mask, axis = 0)
    new_mask=mask.copy()
    for i in range(line.shape[0]):

        if line[i]>int(mask.shape[0]/3):
            new_mask[:,i] = 1
    return new_mask


def hough_line_connection(mask):
    """
    Detect lines and connect them based on proximity and angle
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask where lines are white (1) on black (0)
    
    Returns:
    --------
    numpy.ndarray : result as [0,1] binary mask
    """
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(mask.astype(np.uint8),  # Just ensure uint8
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=50,
                            minLineLength=100,
                            maxLineGap=30)
    
    # Create output image
    result = mask.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), 1, 2)  # Draw with value 1, not 255
    
    return result


def despeckle_mask(mask):
    """
    Remove small speckles by operating on the inverted mask.
    mask: boolean array
    """
    mask_bool = mask.astype(bool)
    inv = ~mask_bool
    bw = inv.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area = 50 # remove only small pixel speckles 
    clean = np.zeros_like(bw)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    clean_bool = ~(clean.astype(bool))
    
    return clean_bool.astype(np.uint8)