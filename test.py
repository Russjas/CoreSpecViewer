# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 11:00:44 2025

@author: russj
"""


import numpy as np

path = 'D:/Clonminch_swir/24_7_clonminch_0_0m80_3m42_2025-10-13_10-03-03_cropped.npy'

# 1. Load the memory map.
original_memmap = np.load(path, mmap_mode='r', allow_pickle=True) 

# 2. Create the new, in-memory array data.
new_data = original_memmap[20:-10, 20:-10].copy()

# 3. CRITICAL FIX: Access the internal memory-map object (which IS closable) 
#    and explicitly close its file handle.
#    This bypasses the missing public .close() method on the outer memmap array object.
if hasattr(original_memmap, '_mmap') and callable(getattr(original_memmap._mmap, 'close')):
    original_memmap._mmap.close()

# 4. Delete the memmap object's reference.
#del original_memmap 

# 5. Save the new data to the path.
np.save(path, new_data) # This must now succeed.