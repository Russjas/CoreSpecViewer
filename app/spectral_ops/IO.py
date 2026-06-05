"""
I/O layer for hyperspectral data in CoreSpecViewer.

Handles all reading from disk: ENVI file pairs, Specim Lumo XML metadata,
and Fenix sensor directories. Also contains the sensor-band-slice logic
that trims noisy edge bands on load, driven by the AppConfig singleton.

Functions
---------
read_envi_header        Parse an ENVI .hdr file into a metadata dict.
parse_lumo_metadata     Parse a Specim Lumo acquisition XML into a flat dict.
find_bands              Locate band-centre wavelengths from a metadata dict.
load_envi               Load an ENVI file pair to a numpy array + metadata dict.
                        Falls back to hylite loader on malformed headers.
reflect_correct         Convert raw radiance to percent reflectance using
                        white and dark references.
bands_from_snr          Derive a usable band slice from white/dark SNR.
find_snr_and_reflect    Full pipeline: load raw ENVI data, select bands by
                        sensor type or SNR, apply reflectance correction.
get_fenix_reflectance   Load and correct Fenix sensor data via hylite,
                        applying smile correction on the derived path.
_slice_from_sensor      Internal: map sensor type string to a band slice
                        using AppConfig slice bounds.

Dependencies
------------
Reads AppConfig for all sensor band-slice bounds (swir_slice_*, mwir_slice_*,
rgb_slice_*, fenix_slice_*, default_slice_*).
"""

import os
import glob
import xml.etree.ElementTree as ET
import logging
import re

from hylite.sensors import Fenix as HyliteFenix
from hylite.io.images import loadWithNumpy
import spectral.io.envi as envi

import numpy as np


from ..config import config  # mutable module singleton
from .fenix_smile import fenix_smile_correction

logger = logging.getLogger(__name__)


#===================io funcs===================================================

def _slice_from_sensor(sensor_type: str):
    """
    Derive the edge bands to slice out using the config dict
    """
    s = sensor_type or ""
    logger.info(f"Sensor type {s}")
    if "SWIR" in s:
        start, stop = config.swir_slice_start, config.swir_slice_stop
    elif "RGB" in s:
        start, stop = config.rgb_slice_start, config.rgb_slice_stop
    elif "FX50" in s:
        start, stop = config.mwir_slice_start, config.mwir_slice_stop
    elif "FENIX" in s:
        start, stop = config.fenix_slice_start, config.fenix_slice_stop
    else:
        start, stop = config.default_slice_start, config.default_slice_stop
    
    return slice(start, stop, None)

def read_envi_header(file):
    """
    Reads an envi header file
    """
    return envi.read_envi_header(file)




# Canonical keys to hunt for, mapped to their normalised token
# these are based on keys used in Lumo acquisition software
# they could be extended to match other metadata formats
_BRUTE_TARGETS = {
    "borehole id":      "boreholeid",
    "box number":       "boxnumber",
    "core depth start": "coredepthstart",
    "sensor type":      "sensortype",
    "core depth stop":  "coredepthstop",
}

def _normalise(s: str) -> str:
    """Lowercase and strip all non-alphanumeric characters for fuzzy matching."""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def brute_force_xml_metadata(xml_file) -> dict:
    """
    Metadata xml's have been found to have a variety of formats, this is designed
    to bypass the structure for xml's with unexpected structures.

    Walk every node in an XML tree without structural assumptions.
    For each node, check three locations where a required field name
    could appear: as an attribute value, as a tag name, or as element text.
    The corresponding value is then extracted from the sibling, own text,
    or next-sibling respectively.

    Normalises all candidate strings before matching, so variants like
    'boreholeid', 'borehole_id', 'BOREHOLE ID' all resolve correctly.

    Parameters
    ----------
    xml_file : str or Path
        Path to any XML file.

    Returns
    -------
    dict
        Canonical key -> extracted value, for every target that was found.
        Keys not found are absent from the dict (not None).
        Each value is a dict: {'value': str, 'strategy': str, 'raw_key': str}
        so the caller has a diagnostic trail.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Build reverse lookup: normalised token -> canonical key
    norm_to_canonical = {v: k for k, v in _BRUTE_TARGETS.items()}

    found = {}  # canonical key -> {'value', 'strategy', 'raw_key'}

    def _record(canonical, value, strategy, raw_key):
        """Store a hit only if not already found (first hit wins)."""
        if canonical not in found and value:
            found[canonical] = {
                'value':    value.strip(),
                'strategy': strategy,
                'raw_key':  raw_key,
            }

    def _walk(node):
        # --- Strategy 1: field name appears as an ATTRIBUTE VALUE ---
        # e.g. <item field="borehole id">BH1</item>
        for attr_name, attr_val in node.attrib.items():
            norm_attr_val = _normalise(attr_val)
            if norm_attr_val in norm_to_canonical:
                canonical = norm_to_canonical[norm_attr_val]
                value = (node.text or '').strip()
                _record(canonical, value, f'attribute_value[{attr_name}]', attr_val)

        # --- Strategy 2: field name IS the tag name ---
        # e.g. <boreholeid>BH1</boreholeid>
        norm_tag = _normalise(node.tag)
        if norm_tag in norm_to_canonical:
            canonical = norm_to_canonical[norm_tag]
            value = (node.text or '').strip()
            _record(canonical, value, 'tag_name', node.tag)

        # --- Strategy 3: field name appears as TEXT, value is in next sibling ---
        # e.g. <key>borehole id</key><value>BH1</value>
        children = list(node)
        for i, child in enumerate(children):
            norm_text = _normalise(child.text or '')
            if norm_text in norm_to_canonical and i + 1 < len(children):
                canonical = norm_to_canonical[norm_text]
                value = (children[i + 1].text or '').strip()
                _record(canonical, value, 'sibling_text', child.text)

        for child in node:
            _walk(child)

    _walk(root)

    # Log findings for diagnostics
    for canonical, hit in found.items():
        logger.debug(
            f"brute_force_xml_metadata | found '{canonical}' = '{hit['value']}' "
            f"via {hit['strategy']} (raw key: '{hit['raw_key']}')"
        )
    missing = [k for k in _BRUTE_TARGETS if k not in found]
    if missing:
        logger.warning(f"brute_force_xml_metadata | could not resolve: {missing}")

    # Return flat canonical_key -> value for easy merging
    return {k: v['value'] for k, v in found.items()}


def parse_lumo_metadata(xml_file):
    """
    Parser specifically for metadata produced using the Specim Lumo 
    aquisition system, and contains assumptions about the format of the metadata
    which have proved to not be consistent across sensors and Lumo versions
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    pairs = []
    # Grab only <key> elements that actually have the attribute
    for key in root.findall('.//key'):
        fld = key.attrib.get('field')
        if fld is not None:
            pairs.append((fld, (key.text or '').strip()))

    # Also flatten simple non-<key> blocks (e.g., <corrected_metadata>)
    flat = {}
    for block in root:
        if block.tag in ('header', 'userdefined', 'core'):
            continue  # these were captured via <key field="...">
        for child in block:
            if len(child) == 0:  # leaf node
                flat[child.tag] = (child.text or '').strip()

    out = dict(pairs)
    out.update(flat)

    # Check whether any brute-force targets are missing or blank
    missing = [
        k for k in _BRUTE_TARGETS
        if not out.get(k, '').strip()
    ]
    print(f"TEMP PRINT MISSING {missing}")
    if missing:
        logger.warning(
            f"parse_lumo_metadata structured parse missing {missing} — falling back to brute force for {xml_file}"
        )
        brute = brute_force_xml_metadata(xml_file)
        # Only fill gaps — do not overwrite what structured parsing already found
        for k in missing:
            if k in brute:
                out[k] = brute[k]

    return out


def find_bands(metadata: dict, arr: np.ndarray):
    """
    searches metadata contents structurally for a band-centres list
    """
    
    nbands = arr.shape[-1]
    best = None  

    for key, val in (metadata or {}).items():
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == nbands:
            try:
                arr = np.asarray(val, dtype=float)
                if arr.ndim != 1:
                    continue

                d = np.diff(arr)
                if not (np.all(d >= 0) or np.all(d <= 0)):
                    continue

                span = float(arr.max() - arr.min())
                if best is None or span > best[0]:
                    best = (span, key, arr)

            except Exception:
                pass

    if best is None:
        return None, None
    _, key, arr = best
    return key, arr

def load_envi(head_path, data_path):
    """Passthrough function to the spectral python library for ENVI file loads
    Assumes full post-processing has been performed:
        - Data is reflectance
        - Noisy edge bands have been sliced away
        - Data has been smoothed
    Some ENVI headers do not capture the byte order parameter. SPy is intolerant 
    of this, so fallback to a hylite loader.
    """
    try:
        box = envi.open(head_path, image=data_path)
        data = np.array(box.load())
       
    except Exception:
        # Workaround for hylite matchHeader Windows path separator bug
        # Bug will be fixed in next hylite release
        # TODO: Remove when environment upgrades to numpy 2 and latest hylite
        # See: https://github.com/hifexplo/hylite/issues/17
        path, ext = os.path.splitext(head_path)
        match = glob.glob(path + "*")
        hylite_path =[x for x in match if x.endswith(".hdr")][0]
        #=====================================
        box = loadWithNumpy(hylite_path)
        box = np.array(box.data)  
        data = np.transpose(box, (1, 0, 2))
        
    metadata = read_envi_header(head_path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, metadata

def reflect_correct(data, white, dark):
    """
    Convert raw radiance-like data to reflectance (%) using a white/dark reference.

    Parameters
    ----------
    data : ndarray, shape (H, W, B)
        Measured hyperspectral cube.
    white : ndarray, shape (H, W, B) or broadcastable to (H, W, B)
        White reference image/cube.
    dark : ndarray, shape (H, W, B) or broadcastable to (H, W, B)
        Dark reference image/cube.

    Returns
    -------
    ndarray
        Reflectance cube in percent, same shape as `data`. Values < 0 are clamped to 0.

    Notes
    -----
    The formula is `(data - mean(dark)) / (mean(white) - mean(dark)) * 100`.  
    """

    M, N, B = data.shape
    wmax = np.mean(white, axis=0)
    dmin = np.mean(dark, axis=0)
    denom = np.subtract(wmax, dmin) # Calculate denominator

    result = np.where(denom != 0, np.divide(np.subtract(data, dmin), denom), 0.0)
    #result = np.divide(np.subtract(data, dmin), np.subtract(wmax, dmin))
    result[result < 0] = 0
    return result * 100

def bands_from_snr(white, dark, wavelengths=None, snr_thresh=20.0, min_run=20):
    """
    Estimate usable band range(s) by SNR and return the longest contiguous run.

    Parameters
    ----------
    white : ndarray
        White reference cube/stack; last axis is bands.
    dark : ndarray
        Dark reference cube/stack; last axis is bands.
    wavelengths : ndarray or list, optional
        (Unused in this version) Wavelengths corresponding to bands (nm).
    snr_thresh : float, optional
        Minimum SNR to consider a band usable; default is 20.0.
    min_run : int, optional
        Minimum contiguous run length to consider; used as a soft threshold.

    Returns
    -------
    slice
        Slice selecting the longest contiguous band run above threshold.
    ndarray
        1D array of SNR per band.

    Raises
    ------
    ValueError
        If the band counts of `white` and `dark` (last axis) do not match.

    Notes
    -----
    SNR is computed as `(mean(white) - mean(dark)) / std(dark)` with spatial
    averaging over all non-band axes. 
    """


    white = np.asarray(white)
    dark  = np.asarray(dark)

    if white.shape[-1] != dark.shape[-1]:
        raise ValueError(f"Band mismatch: white B={white.shape[-1]}, dark B={dark.shape[-1]}")

    # reduce over all spatial dims, keep only band axis
    reduce_axes_w = tuple(range(white.ndim - 1))
    reduce_axes_d = tuple(range(dark.ndim  - 1))
    w_bar = np.nanmean(white, axis=reduce_axes_w)  # shape (B,)
    d_bar = np.nanmean(dark,  axis=reduce_axes_d)  # shape (B,)
    d_std = np.nanstd(dark,   axis=reduce_axes_d) + 1e-9

    snr = (w_bar - d_bar) / d_std                    # shape (B,)
    good = snr >= snr_thresh



    # largest contiguous True run
    starts, ends, in_run = [], [], False
    for i, ok in enumerate(good.tolist() + [False]):
        if ok and not in_run: s = i; in_run = True
        elif not ok and in_run: starts.append(s); ends.append(i); in_run = False

    if not starts:
        B = good.size
        return slice(int(0.1*B), int(0.9*B)), snr

    lengths = np.array(ends) - np.array(starts)
    idx = int(np.argmax(lengths))
    if lengths[idx] < min_run:
        # optional: relax or just return the longest anyway
        pass

    thresh_test = np.where(snr > snr_thresh)
    return slice(thresh_test[0][0]-1, thresh_test[0][-1]+1), snr


# TODO: When the separate QAQC GUIS get integrated, they can call this func with QAQC=True to check SNR
def find_snr_and_reflect(header_path, white_path, dark_path, QAQC=False,
                         data_data_path = None,
                         white_data_path = None,
                         dark_data_path = None): 
    """
    Load raw data and references, pick a band window from SNR, and compute reflectance.

    Parameters
    ----------
    header_path : str or path-like
        Path to the main ENVI header (.hdr).
    white_path : str or path-like
        Path to the white reference ENVI header.
    dark_path : str or path-like
        Path to the dark reference ENVI header.
    QAQC : Boolean. 
        If true will calculate bands above threshold, if False will use fixed slice
        for technique
        
    optional data path arguments, for mangled datasets with inconsistent file names.

    Returns
    -------
    tuple
        `(data_reflect, bands, header)` where
        - data_reflect : ndarray, reflectance (%) cube cropped to usable bands
        - bands : ndarray, wavelengths for the selected band slice
        - header : dict, parsed header from `parse_header`.

    Notes
    -----
    Uses `bands_from_snr` to determine a contiguous band slice prior to
    reflectance correction.
    """

    if data_data_path:
        box = envi.open(header_path, image=data_data_path)
    else:
        box = envi.open(header_path)

    if white_data_path:
        white_ref = envi.open(white_path, image=white_data_path)
    else:
        white_ref = envi.open(white_path)

    if dark_data_path:
        dark_ref = envi.open(dark_path, image=dark_data_path)
    else:
        dark_ref = envi.open(dark_path)

    header = envi.read_envi_header(header_path)

    bands = np.array([float(x) for x in header['wavelength']])
    data = np.array(box.load())
    white_ref = np.array(white_ref.load())
    dark_ref = np.array(dark_ref.load())
    if QAQC:
        band_slice, snr = bands_from_snr(white_ref, dark_ref, snr_thresh=20.0)
    else:
        sensor = header['sensor type']
        band_slice = _slice_from_sensor(sensor)
        snr = None


    data  = data[:, :, band_slice]
    white = white_ref[:, :, band_slice]
    dark  = dark_ref[:, :, band_slice]
    bands = bands[band_slice]

    data_reflect = reflect_correct(data, white, dark)

    return data_reflect, bands, snr


def get_fenix_reflectance(path, mode='hylite'):
    
    if mode == 'hylite':
        hyimg = HyliteFenix.correct_folder(str(path), shift=True, lens = True)
    else:
        hyimg = HyliteFenix.correct_folder(str(path), shift=True, lens = False)
    
    if isinstance(hyimg, tuple):
        hyimg = hyimg[0]
    if mode == 'hylite':
        reflectance = hyimg.data
        logger.debug(f"Hylite path reflextance shape {reflectance.shape}")
    else:
        reflectance = fenix_smile_correction(np.transpose(hyimg.data, (1, 0, 2)))          # (H, W, B), float32
        logger.debug(f"derived path reflextance shape {reflectance.shape}")
    bands       = hyimg.get_wavelengths()
    snr         = None # snr workflows not implemented yet
    band_slice = _slice_from_sensor("FENIX Sensor")
    if mode != 'hylite':
        reflectance = np.transpose(reflectance, (1, 0, 2))
        reflectance = np.flip(reflectance, axis = 0)
    else:
        reflectance = np.rot90(reflectance, 2)
    logger.debug(f"reflectance shape {reflectance.shape}, bands shape {bands.shape}, slice: {band_slice} ")
    return reflectance[:,:, band_slice]*100, bands[band_slice], snr


# Writing Envi cubes on export

# ---- ENVI header schema (semantic half) -------------------------------------

_ENVI_SEMANTIC_KEYS = (
    "description",
    "sensor type",
    "wavelength units",
    "z plot titles",
    "reflectance scale factor",
    "acquisition date",
    "acquisition time",
    "map info",
    "coordinate system string",
)

_PER_BAND_KEYS = ("band names", "fwhm", "data gain values")


def translate_metadata(po_metadata, bands, smooth_params=None,
                       include_ignore_value=False, ignore_value=0):
    """
    Build the semantic half of an ENVI header from a metadata dict

    Layout fields (samples/lines/bands/data type/interleave/byte order) are NOT
    set here; they are injected by ``write_envi`` from the array itself.

    Parameters
    ----------
    po_metadata : dict
        ProcessedObject metadata (merged ENVI header + parsed Lumo XML). Only
        recognised semantic keys survive; everything else is dropped.
    bands : array-like, shape (B,)
        Band-centre wavelengths. Authoritative for the 'wavelength' field.
    smooth_params : dict, optional
        Smoothing provenance to stamp as custom fields. Recognised keys:
        'method', 'window', 'polyorder'. Only present keys are written. Pass
        None (default) for reflectance/unsmoothed exports — no fields emitted.
    include_ignore_value : bool, optional
        If True, write 'data ignore value' so ENVI-aware readers treat masked
        pixels as nodata. Default False.
    ignore_value : int or float, optional
        Nodata sentinel to declare. MUST match the value the caller wrote into
        masked pixels. Default 0 (ENVI-native masked convention).

    Returns
    -------
    dict
        Header dict of semantic fields only, ready to be merged with
        array-derived layout fields by ``write_envi``.

    Notes
    -----
    Key matching against ``po_metadata`` is case-insensitive; canonical
    lower-case ENVI names are emitted. 'wavelength units' defaults to 'nm'.
    """
    # Case-insensitive view of the incoming metadata.
    lower_index = {str(k).strip().lower(): v for k, v in (po_metadata or {}).items()}

    header = {}

    # 1. Carry through scalar/short semantic fields.
    for key in _ENVI_SEMANTIC_KEYS:
        if key in lower_index:
            header[key] = lower_index[key]

    # 2. Wavelength axis — bands is the source of truth, always overwrite.
    bands = np.asarray(bands, dtype=float).ravel()
    nbands = bands.size
    header["wavelength"] = [float(b) for b in bands]
    header.setdefault("wavelength units", "nm")

    # 3. Per-band fields: keep only if length still matches the (possibly sliced) axis.
    for key in _PER_BAND_KEYS:
        if key not in lower_index:
            continue
        val = lower_index[key]
        try:
            if len(val) == nbands:
                header[key] = val
            else:
                logger.debug(f"Dropping stale per-band field '{key}' "
                             f"(len {len(val)} != {nbands} bands)")
        except TypeError:
            logger.debug(f"Dropping malformed per-band field '{key}'")

    # 4. 'default bands' (RGB display hint): keep only if indices are in range.
    if "default bands" in lower_index:
        db = lower_index["default bands"]
        try:
            if all(0 <= int(i) < nbands for i in db):
                header["default bands"] = db
            else:
                logger.debug("Dropping 'default bands' (index out of range after slicing)")
        except (TypeError, ValueError):
            logger.debug("Dropping malformed 'default bands'")

    # 5. Smoothing provenance (smoothed exports only).
    if smooth_params:
        if "method" in smooth_params:
            header["smoothing method"] = smooth_params["method"]
        if "window" in smooth_params:
            header["smoothing window"] = smooth_params["window"]
        if "polyorder" in smooth_params:
            header["smoothing polyorder"] = smooth_params["polyorder"]

    # 6. Nodata declaration (masked exports only).
    if include_ignore_value:
        header["data ignore value"] = ignore_value

    logger.debug(
        f"translate_metadata: {len(header)} fields, {nbands} bands, "
        f"smoothed={'yes' if smooth_params else 'no'}, "
        f"ignore_value={'yes' if include_ignore_value else 'no'}"
    )
    return header

def write_envi(array, header, path,
               interleave="bil", byteorder=0, dtype=None, force=False):
    """
    Write a numpy cube and a semantic header dict to an ENVI file pair.

    The layout half of the header
    (samples/lines/bands/data type/byte order/interleave/header offset/file
    type) is derived by spectral python library from the array and the keyword arguments —
    only the semantic header (typically from ``translate_metadata``) is passed
    in. Always writes ``byte order`` explicitly so the malformed-header fallback
    in ``load_envi`` is never needed for files this tool produces.

    Parameters
    ----------
    array : ndarray, shape (H, W, B) or (H, W)
        The cube to write. A 2-D array is stored as a single band.
    header : dict
        Semantic ENVI header fields (e.g. from ``translate_metadata``). Any
        layout keys are ignored — spectral overwrites them from the array.
    head_path : str or path-like
        Output ``.hdr`` path. Must end in ``.hdr``.
    data_path : str or path-like
        Output binary path. Must share the same stem as ``head_path`` (ENVI
        forces the data file to ``<header base> + ext``); its extension sets
        the data-file extension.
    interleave : {'bil', 'bip', 'bsq'}, optional
        Band interleave to write. Default 'bil'.
    byteorder : int or str, optional
        Endianness, 0/'little' or 1/'big'. Default 0 (little-endian).
    dtype : numpy dtype or type string, optional
        Storage dtype. Default None preserves the array's dtype (pass e.g.
        ``np.float32`` to downcast a float64 cube and halve the file size).
    force : bool, optional
        Overwrite existing files if True; otherwise raise. Default False.

    Returns
    -------
    tuple of str
        ``(head_path, data_path)`` actually written.

    Raises
    ------
    ValueError
        If the array is not 2-D/3-D, if its band count disagrees with the
        header's 'wavelength' list, or if the header and data paths do not
        share a stem.
    """
    array = np.asarray(array)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D array, got ndim={array.ndim}")

    # ensure descriptive metadata matches cube shape
    nbands = array.shape[2] if array.ndim == 3 else 1
    wl = header.get("wavelength")
    if wl is not None and len(wl) != nbands:
        raise ValueError(
            f"Band-count mismatch: array has {nbands} bands but header "
            f"'wavelength' lists {len(wl)}. Refusing to write an inconsistent pair."
        )

    kwargs = dict(metadata=header, interleave=interleave,
                  byteorder=byteorder, ext="img", force=force)
    if dtype is not None:
        kwargs["dtype"] = dtype

    envi.save_image(str(path), array, **kwargs)
    logger.info(f"Wrote ENVI pair: {path}"
                f"({array.shape}, {nbands} bands, interleave={interleave})")
    return str(path)

