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


def detect_slice_rectangles_robust(
    image,                  #numpy array uint8, 3-channel (H, W, 3)
    min_area_frac=0.0005,     # min polygon area as a fraction of image area
    canny_sigma=0.33,         # auto-Canny thresholds from median
    approx_eps_frac=0.02,     # polygon approximation tolerance
    close_kernel=5,           # morphological close kernel size (pixels)
    use_otsu=True,            # binarize before Canny (good for UI/scans)
    allow_rotated=True        # also accept minAreaRect boxes
):
    """
    Detect and crop the largest rectangular slice/region in an RGB image using
    robust edge-based contour extraction.

    The function:
    1. Converts the image to grayscale and denoises via bilateral filtering.
    2. Applies either Otsu thresholding or raw grayscale input to Canny edge
       detection with adaptive high/low thresholds.
    3. Uses morphological closing to join fragmented edges (optional).
    4. Extracts contours and retains only convex quadrilaterals or rotated
       rectangles above a minimum area threshold.
    5. Performs simple non-maximum suppression (NMS) using IoU to remove
       overlapping rectangles.
    6. Selects the largest remaining rectangle, computes a crop slice, and
       returns the cropped region.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), dtype uint8
        Input BGR/RGB image (only shape and intensity matter). Should represent
        a core scan, object, or similar high-contrast rectangular target.

    min_area_frac : float, optional
        Minimum rectangle area as a fraction of total image area. Defaults to
        0.0005.

    canny_sigma : float, optional
        Sigma factor for adaptive Canny thresholds computed from the median
        intensity of the source. Default is 0.33.

    approx_eps_frac : float, optional
        Polygon approximation tolerance as a fraction of contour perimeter
        (used in `cv2.approxPolyDP`). Default is 0.02.

    close_kernel : int, optional
        Kernel size (pixels) for morphological closing to join edges. Set to 0
        or 1 to disable. Default is 5.

    use_otsu : bool, optional
        If True, apply Otsu binarization before Canny. Often more robust for
        segmentation of UI scans or uneven illumination. Default is True.

    allow_rotated : bool, optional
        If True, fallback to rotated bounding boxes (`cv2.minAreaRect`) when
        convex quadrilaterals are not found. Default is True.

    Returns
    -------
    cropped : np.ndarray
        Cropped image containing the selected rectangle.

    crop_slice : tuple of slice
        Tuple `(slice(rows), slice(cols))` that can be used to apply the same
        crop to other arrays (e.g. hyperspectral cubes).

    Notes
    -----
    - IoU threshold for NMS is fixed at 0.6 to discard heavily overlapping
      candidate rectangles.
    - The returned crop is always axis-aligned even when `allow_rotated=True`.
    - Bilateral filtering is used to preserve edge structure while reducing
      noise for more stable polygon detection.
    """
    img = image

    H, W = img.shape[:2]
    min_area = H * W * min_area_frac

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    src = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] if use_otsu else gray
    v = np.median(src); lo = int(max(0, (1.0 - canny_sigma) * v)); hi = int(min(255, (1.0 + canny_sigma) * v))
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
            rects_poly.append(approx[:,0,:].astype(np.float32))
        elif allow_rotated:
            r = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = r
            if rw * rh >= min_area:
                box = cv2.boxPoints(r).astype(np.float32)
                x, y, w, h = cv2.boundingRect(box.astype(np.int32))
                rects_xywh.append((x, y, w, h))
                rects_poly.append(box)

    # simple NMS by IoU on axis-aligned boxes
    def iou(a,b):
        ax,ay,aw,ah = a; bx,by,bw,bh = b
        ax2, ay2, bx2, by2 = ax+aw, ay+ah, bx+bw, by+bh
        ix1, iy1, ix2, iy2 = max(ax,bx), max(ay,by), min(ax2,bx2), min(ay2,by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        return inter / (aw*ah + bw*bh - inter + 1e-9)

    keep = []
    used = [False]*len(rects_xywh)
    order = sorted(range(len(rects_xywh)), key=lambda i: rects_xywh[i][2]*rects_xywh[i][3], reverse=True)
    for i in order:
        if used[i]: continue
        keep.append(i)
        for j in order:
            if used[j] or j==i: continue
            if iou(rects_xywh[i], rects_xywh[j]) > 0.6:
                used[j] = True

    rects_xywh = [rects_xywh[i] for i in keep]
    rects_poly = [rects_poly[i] for i in keep]

    areas = np.array([cv2.contourArea(p) for p in rects_poly])
    index = np.argmax(areas)
    x, y, w, h = rects_xywh[index]
    crop_slice = (slice(y, y + h), slice(x, x + w))
    return image[crop_slice], crop_slice


def get_stats_from_mask(mask, proportion=16, iters=2):
    """
    Compute connected components on the (eroded) inverse of a mask.

    Parameters
    ----------
    mask : ndarray of {0,1}
        Binary mask with 1 = core region.
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