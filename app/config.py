"""
Global configuration dictionary and default parameters used across CoreSpecViewer.

Stores SAVGOL settings, directory rules, filtering thresholds, and UI behaviour
flags shared by spectral and interface modules.
"""

con_dict = {
    # band slice bounds (inclusive-exclusive)
         
    "swir_slice_start": 13,
    "swir_slice_stop": 262,
    "mwir_slice_start": 5,
    "mwir_slice_stop": 142,
    "rgb_slice_start": 0,
    "rgb_slice_stop": -1,   # keep -1 to mean "to last"
    "default_slice_start": 5,
    "default_slice_stop": -5,
    "fenix_slice_start": 20,
    "fenix_slice_stop": -20,

    # Savitzky–Golay
    "savgol_window": 10,
    "savgol_polyorder": 2,
    #features
    "feature detection threshold": 0.1 
}


feature_keys = [
    '1400W', '1480W', '1550W', '1760W', '1850W',
    '1900W', '2080W', '2160W', '2200W', '2250W',
    '2290W', '2320W', '2350W', '2390W', '2950W',
    '2950AW', '2830W', '3000W', '3500W', '4000W',
    '4000WIDEW', '4470TRUEW', '4500SW', '4500CW',
    '4670W', '4920W', '4000V_NARROWW', '4000shortW', '2950BW'
]

def set_value(key, value):
    if key not in con_dict:
        raise KeyError(key)
    # naive cast
    ty = type(con_dict[key])
    con_dict[key] = ty(value)


def get_all():
    return con_dict



