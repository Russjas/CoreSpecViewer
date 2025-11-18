"""
Created on Mon Nov 17 09:26:57 2025

@author: russj
"""

from dataclasses import dataclass
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Dataset:
    """
    Lightweight wrapper for an individual on-disk dataset.

    Encapsulates both the file path and in-memory data object, handling
    transparent loading and saving for supported formats.

    Parameters
    ----------
    base : str
        Basename common to all datasets in the same ProcessedObject.
    key : str
        Short identifier for the dataset (e.g., 'cropped', 'mask', 'metadata').
    path : Path
        Full filesystem path to the file.
    suffix : str
        Key or descriptive suffix used to form the filename.
    ext : str
        File extension (must be one of `.npy`, `.json`, `.jpg`, `.npz`).
    data : object, optional
        The loaded or assigned data object. If `None`, the dataset is loaded
        automatically if the file exists.

    Attributes
    ----------
    data : object
        The in-memory representation of the dataset (NumPy array, dict, PIL image, etc.).

    Notes
    -----
    For `.npz` files, data are stored as a `numpy.ma.MaskedArray` with separate
    arrays for `data` and `mask`. NumPy `.npy` files are loaded as memmaps
    to minimize memory footprint.
    """
    base:str
    key: str
    path: Path
    suffix: str
    ext: str
    data: object = None
    thumb: Image | None = None
    _memmap_ref: object = None
    def __post_init__(self):
        """Normalize the path and automatically load data if the file exists."""
        self.path = Path(self.path)
        if not self.ext.startswith('.'):
            self.ext = '.' + self.ext

        # Auto-load only if file already exists
        if self.path.is_file() and self.data is None:
            self.load_dataset()

    def close_handle(self) -> None:
        print('close called')
        """
        Explicitly close any outstanding memmap file handle to release the OS lock.
        Attempt to fix an annoying save bug
        """
        if self._memmap_ref is not None:
            print('actually closing')
            self._memmap_ref.close()
        gc.collect()
        self._memmap_ref = None

    def load_dataset(self):
        """
        Load the dataset from disk into memory based on its extension.

        Raises
        ------
        ValueError
            If the file type is not recognized or unsupported.
        """
        if self.ext not in ['.npy', '.json', '.jpg', '.npz']:
            raise ValueError(f"Cannot open {self.ext}, this is an invalid file type")

        if self.ext == '.npy':
            # memmap keeps memory footprint low; disable pickle for safety
            data = np.load(self.path, mmap_mode='r', allow_pickle=False)
            # store a reference to the memmap to explicitly close later
            if isinstance(data, np.memmap):
                self._memmap_ref = data._mmap
            self.data = data


        elif self.ext == '.json':
            self.data =json.loads(self.path.read_text(encoding="utf-8"))

        elif self.ext == '.jpg':
            self.data = Image.open(self.path)

        elif self.ext == '.npz':
            with np.load(self.path, allow_pickle=False) as npz:
                data = npz["data"]
                mask = npz["mask"].astype(bool)

            self.data = np.ma.MaskedArray(data, mask=mask, copy=False)


    def save_dataset(self, new=False):
        """
        Write the dataset to disk in the appropriate format.

        Parameters
        ----------
        new : bool, optional
            If False (default), existing memmaps are not overwritten in-place.

        Raises
        ------
        ValueError
            If no data is loaded or the file type is unsupported.
        """
        if self.data is None:
            raise ValueError("No data loaded or assigned; nothing to save.")

        if self.ext not in ['.npy', '.json', '.jpg', '.npz']:
            raise ValueError(f"Cannot save {self.ext}, this is an invalid file type")

        if self.ext == '.npz':
            np.savez_compressed(
                self.path,
                data=self.data.data,  # raw data
                mask=self.data.mask)

        elif self.ext == '.npy':
            # If it's a memmap and we're not creating new, just return
            print(type(self.data), self.key)
            if isinstance(self.data, np.memmap) and not new:
                return

            print((self._memmap_ref is None), 'mem ref is None')

            np.save(self.path, self.data)


        elif self.ext == '.json':
            text = json.dumps(self.data, indent=2)
            self.path.write_text(text, encoding='utf-8')

        elif self.ext == '.jpg':
            if isinstance(self.data, Image.Image):
                self.data.save(self.path)
        else:
            raise ValueError(f"Cannot save unsupported file type: {self.ext}")

    def copy(self, data=None):
        """
        Create a shallow copy of this Dataset, optionally replacing its data.
        """
        # Force conversion to regular array if memmap
        if data is None:
            if isinstance(self.data, np.memmap):
                new_data = np.array(self.data, copy=True)
            else:
                new_data = self.data.copy()
        else:
            if isinstance(data, np.memmap):
                new_data = np.array(data, copy=True)
            else:
                new_data = data.copy()

        return Dataset(
            base=self.base,
            key=self.key,
            path=self.path,
            suffix=self.suffix,
            ext=self.ext,
            data=new_data
        )




    def save_thumb(self):

        if self.thumb is not None:
            self.thumb.save(str(self.path)[:-len(self.ext)]+'thumb.jpg')
