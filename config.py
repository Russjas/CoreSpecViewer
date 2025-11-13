# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 09:39:39 2025

@author: russj
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

    # Savitzky–Golay
    "savgol_window": 10,   
    "savgol_polyorder": 2,
}



def set_value(key, value):
    if key not in con_dict:
        raise KeyError(key)
    # naive cast
    ty = type(con_dict[key])
    con_dict[key] = ty(value)
    

def get_all():
    return con_dict  