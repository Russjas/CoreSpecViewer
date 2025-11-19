"""
Created on Mon Nov 17 09:26:57 2025

@author: russj
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..spectral_ops import spectral_functions as sf
from .dataset import Dataset


@dataclass
class ProcessedObject:
    """
    Logical container for all processed datasets derived from a single core box.

    A ProcessedObject groups multiple Dataset instances sharing a common basename
    (e.g., '<hole_name>_<box_number> etc'), providing unified access to their arrays, JSON
    metadata, and derived products.

    Attributes
    ----------
    basename : str
        Shared base name for all component datasets.
    root_dir : Path
        Directory containing all processed files.
    datasets : dict[str, Dataset]
        Mapping of dataset keys to Dataset instances.
    temp_datasets : dict[str, Dataset]
        Temporary or derived datasets not yet written to disk.

    Notes
    -----
    This abstraction enables consistent access via attribute syntax:
    `obj.cropped` returns the underlying NumPy array for that dataset key.
    """
    basename: str
    root_dir: Path
    datasets: dict = field(default_factory=dict)
    temp_datasets: dict = field(default_factory=dict)



    # ---- convenience attribute passthrough ----
    def __getattr__(self, name):
        """Convenience passthrough for accessing `.data` via attribute syntax."""
        if name in self.temp_datasets:
            return self.temp_datasets[name].data
        elif name in self.datasets:
            return self.datasets[name].data
        raise AttributeError(f"{name!r} not found in datasets or attributes")

    # ---- internal: parse a stem into (basename, key) with a special-case for 'savgol_cr' ----
    @staticmethod
    def _parse_stem_with_exception(stem: str):
        """
        Parse a filename stem into (basename, key), preserving '_savgol_cr' suffixes.
        Legacy, unfortunately

        Parameters
        ----------
        stem : str
            Filename stem without extension.

        Returns
        -------
        tuple[str, str]
            Basename and key. Returns (None, None) if no underscore is found.
        """
        if stem.endswith("_savgol_cr"):
            return stem[: -len("_savgol_cr")], "savgol_cr"
        # Fallback: original behavior — split on last underscore
        base, sep, key = stem.rpartition("_")
        if not sep:
            # No underscore -> cannot infer
            return None, None
        return base, key

    @classmethod
    def from_path(cls, path):
        """
        Instantiate a ProcessedObject by discovering all matching datasets.

        Parameters
        ----------
        path : str or Path
            Path to one file in the processed directory.

        Returns
        -------
        ProcessedObject
            Populated instance with all associated Dataset objects.

        Raises
        ------
        ValueError
            If the basename cannot be inferred from the filename.
        """
        p = Path(path)
        root = p.parent

        # Infer basename from the given file
        stem = p.stem
        basename, seed_key = cls._parse_stem_with_exception(stem)
        if basename is None:
            raise ValueError(
                f"Cannot infer basename from {p.name}; expected '<basename>_<suffix>.<ext>'."
            )

        # Discover all matching files in the same directory
        datasets = {}
        for fp in root.iterdir():
            if not fp.is_file():
                continue
            s = fp.stem

            # First try the special-case parser
            b, key = cls._parse_stem_with_exception(s)
            if b is None or b != basename:
                continue  # not part of this basename group
            if key.endswith('thumb'):                continue

            ext = fp.suffix if fp.suffix.startswith(".") else fp.suffix


            ds = Dataset(base=basename, key=key, path=fp, suffix=key, ext=ext)
            datasets[key] = ds
            print(key)

        return cls(basename=basename, root_dir=root, datasets=datasets)

    # ---- disk I/O helpers ----
    def save_all(self, new=False):
        """Save all registered datasets to disk."""
        for dataset in self.datasets.values():
            dataset.save_dataset(new=new)
            dataset.save_thumb()

    @classmethod
    def new(cls, root_dir, basename):
        """Factory for a brand-new ProcessedObject (no files yet)."""
        return cls(basename=basename, root_dir=Path(root_dir))

    def add_dataset(self, key, data, ext=".npy"):
        """Attach an in-memory dataset; not written until save_all()."""
        path = self.root_dir / f"{self.basename}_{key}{ext}"
        ds = Dataset(base=self.basename, key=key, path=path, suffix=key, ext=ext, data=data)
        self.datasets[key] = ds


    def add_temp_dataset(self, key, data=None, ext=".npy"):
        """Attach an in-memory dataset; not written until save_all()."""
        if key in self.datasets.keys():
            self.temp_datasets[key] = self.datasets[key].copy(data=data)
            self.build_thumb(key)
            return
        path = self.root_dir / f"{self.basename}_{key}{ext}"
        ds = Dataset(base=self.basename, key=key, path=path, suffix=key, ext=ext, data=data)
        self.temp_datasets[key] = ds
        print('calling build thum with {key}')
        self.build_thumb(key)

    def update_root_dir(self, path):
        """
        Update the root directory and adjust file paths for all datasets.

        Parameters
        ----------
        path : str or Path
            New directory to assign as the root.
        """
        new_root = Path(path)
        self.root_dir = new_root
        for ds in self.datasets.values():
            filename = f"{self.basename}_{ds.key}{ds.ext}"
            ds.path = new_root.joinpath(filename)
        if self.temp_datasets:
            for ds in self.temp_datasets.values():
                filename = f"{self.basename}_{ds.key}{ds.ext}"
                ds.path = new_root.joinpath(filename)

    def update_dataset(self, key, data):
        """Replace the in-memory data for a given dataset key."""
        self.datasets[key].data = data


    def commit_temps(self):
        """Promote all temporary datasets to permanent and clear temp cache."""
        for key in self.temp_datasets.keys():
            # Close old memmap handle before replacing
            if key in self.datasets:

                self.datasets[key].close_handle()
                self.datasets[key]._memmap_ref = None
                self.datasets[key].data = None
                del self.datasets[key]

            self.datasets[key] = self.temp_datasets[key]

        self.clear_temps()

    def clear_temps(self):
        """Remove all temporary datasets."""
        self.temp_datasets.clear()

    @property
    def is_raw(self) -> bool:
        """Return False; used for interface consistency with RawObject."""
        return False

    @property
    def has_temps(self):
        """Whether the object currently holds temporary datasets."""
        return bool(self.temp_datasets)
    # ---- registry API ----
    def keys(self):
        """Return a sorted list of all dataset keys (base + temp)."""
        return sorted(self.datasets.keys()|self.temp_datasets.keys())

    def has(self, key: str):
        """Return True if the dataset key exists (with valid ndarray data)."""
        return (key in self.temp_datasets) or (key in self.datasets)

    def has_temp(self, key):
        """Check if a temporary dataset exists for the specified key."""
        return key in self.temp_datasets

    def __getitem__(self, key):
        """Return the Dataset object for the given key."""
        if key in self.temp_datasets:
            return self.temp_datasets[key]
        return self.datasets[key]

    def get_data(self, key: str):
        """
        Return the ndarray for a dataset key.
        Respects temp-first when prefer_temp=True.
        Raises KeyError if the key doesn't exist anywhere.
        """
        if key in self.temp_datasets and isinstance(self.temp_datasets[key].data, np.ndarray):
            return self.temp_datasets[key].data
        if key in self.datasets and isinstance(self.datasets[key].data, np.ndarray):
            return self.datasets[key].data

        raise KeyError(f"No dataset '{key}' in temps or base")

    def reload_dataset(self, key):
        """Reload a single dataset from disk."""
        self.datasets[key].load_dataset()

    def reload_all(self):
        """Reload all datasets from disk."""
        for ds in self.datasets.values():
            ds.load_dataset()
    def build_thumb(self, key):
        print(f'build thumb {key}')
        ds = self.temp_datasets.get(key)
        if ds is None:
            ds = self.datasets.get(key)
        if ds is None:
            return

        try:
            if ds.ext == ".npy" and getattr(ds.data, "ndim", 0) > 1:
                if key == "mask":
                    im = sf.mk_thumb(ds.data)
                elif key.endswith("INDEX"):
                    im = sf.mk_thumb(ds.data, mask=self.mask, index_mode=True)
                else:
                    im = sf.mk_thumb(ds.data, mask=self.mask)
                ds.thumb = im

            elif ds.ext == ".npz":
                im = sf.mk_thumb(ds.data.data, mask=ds.data.mask)
                ds.thumb = im

            else:
                return

        except ValueError:
            return

    def build_all_thumbs(self):
        """Build thumbnails for all thumbnail-able datasets."""
        for key in self.datasets.keys()|self.temp_datasets.keys():
            try:
                self.build_thumb(key)
            except Exception:
                continue

    def save_all_thumbs(self):
        """Save any in-memory thumbnails as JPEGs beside their datasets."""
        for ds in self.datasets.values():
            if ds.thumb is not None:
                ds.save_thumb()

    def load_thumbs(self):
        for key, ds in self.datasets.items():
            if Path(str(ds.path)[:-len(ds.ext)]+'thumb.jpg').is_file():
                self.datasets[key].thumb = Image.open(str(ds.path)[:-len(ds.ext)]+'thumb.jpg')

    def load_or_build_thumbs(self):
        for key, ds in self.datasets.items():

            if Path(str(ds.path)[:-len(ds.ext)]+'thumb.jpg').is_file():
                self.datasets[key].thumb = Image.open(str(ds.path)[:-len(ds.ext)]+'thumb.jpg')
            else:
                self.build_thumb(key)
