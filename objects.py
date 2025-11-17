# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:19:02 2025

@author: russj

objects.py

Core data structures for representing and manipulating hyperspectral core-scan datasets.

This module defines the foundational classes used across the GSI Hyperspectral Core Box
ecosystem. These objects encapsulate both raw Specim Lumo directories and processed
reflectance-derived datasets, providing a consistent interface for file discovery,
in-memory manipulation, and metadata management.

Classes
-------
Dataset
    Generic container for a single data file (NumPy array, JSON, JPEG, or NPZ masked array).
ProcessedObject
    Logical grouping of processed datasets derived from a single core box.
RawObject
    Representation of a Specim Lumo export directory with dark, white, and data frames.
HoleObject
    Aggregator class representing a borehole composed of multiple processed boxes.

Constants
---------
SPECIM_LUMO_REQUIRED : dict
    Expected filenames for valid Specim Lumo exports.

Notes
-----
These classes form the bridge between file-level datasets (e.g., `.npy`, `.hdr`)
and higher-level geological entities (boxes, holes). They support robust path handling,
automatic loading/saving, and downstream integration with the GUI and processing
workflows.
"""

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
from collections import Counter
import builtins
import xml.etree.ElementTree as ET 
from PIL import Image
import spectral_functions as sf
from typing import Iterator, Tuple, List, Union, Optional, Dict
import sys, gc

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
    thumb: Optional[Image] = None
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
        import gc, sys
        
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

        return cls(basename=basename, root_dir=root, datasets=datasets)

    # ---- disk I/O helpers ----
    def save_all(self, new=False):
        """Save all registered datasets to disk."""
        for dataset in self.datasets.values():
            dataset.save_dataset(new=new)

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
        
     #TODO remove this debg func  
    def print_refs(self, text):
        """Promote all temporary datasets to permanent and clear temp cache."""
        print(text)
        for key in self.datasets:
                import sys, gc
                if self.datasets[key]._memmap_ref is not None:
                    print(f"Refcount {key}: {sys.getrefcount(self.datasets[key]._memmap_ref)}")
                    refs = gc.get_referrers(self.datasets[key]._memmap_ref)
                    print(f"Referrers {key}: {[type(r).__name__ for r in refs]}")
                
                
            
    
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

        def _build_for(dataset_dict, key):
            ds = dataset_dict.get(key)
            if ds is None:
                return
    
            try:
                if ds.ext == ".npy" and getattr(ds.data, "ndim", 0) > 1:
                    if key == "mask":
                        im = sf.mk_thumb(ds.data)
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
    
        _build_for(self.datasets, key)
        _build_for(self.temp_datasets, key)    


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
        
            
            

SPECIM_LUMO_REQUIRED = {
    "data head":    "{id}.hdr",
    "data raw":     "{id}.raw",
    "data log":     "{id}.log",
    "dark head":    "DARKREF_{id}.hdr",
    "dark raw":     "DARKREF_{id}.raw",
    "dark log":     "DARKREF_{id}.log",
    "white head":   "WHITEREF_{id}.hdr",
    "white raw":    "WHITEREF_{id}.raw",
    "white log":    "WHITEREF_{id}.log",
    "metadata":     "{id}.xml",
}
  
@dataclass
class RawObject:
    """
    Representation of a Specim Lumo raw export directory.

    Provides automatic validation of required files (dark, white, data, metadata)
    and methods to load reflectance cubes and generate processed outputs.

    Attributes
    ----------
    basename : str
        Unique identifier derived from the dataset stem.
    root_dir : Path
        Directory containing the raw export.
    files : dict[str, str]
        Mapping of required file roles to file paths.
    metadata : dict
        Combined metadata parsed from XML and ENVI headers.
    temp_reflectance : np.ndarray, optional
        Optional cropped or masked version of the reflectance cube.
    """
    basename: str
    root_dir: Path
    files: dict = field(default_factory=dict)
    temp_reflectance: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)
    file_issues: dict = field(default_factory=dict)
    def __post_init__(self):
        """On initialization, populate metadata and compute reflectance."""
        self.get_metadata()
        self.get_reflectance()
        
        
    def get_metadata(self):
        """Load and merge Specim XML + ENVI header metadata if available."""
        if 'metadata' in self.files.keys() and 'data head' in self.files.keys():
            self.metadata = sf.parse_lumo_metadata(self.files['metadata']) | sf.read_envi_header(self.files['data head'])
        elif 'metadata' not in self.files.keys() and 'data head' in self.files.keys():
            self.metadata = sf.read_envi_header(self.files['data head'])
    @property
    def is_raw(self) -> bool:
        """Return True; distinguishes from ProcessedObject."""
        return True
        
    @classmethod
    def from_Lumo_directory(cls, directory):
        """
        Build a RawObject from a Specim Lumo export folder.
    
        Validates file presence, detects duplicates or zero-byte files,
        and constructs the file mapping. Missing non-critical files are
        flagged and ignored (continue), while critical files (raw data/refs)
        will still raise an error.
        """
        d = Path(directory)
        all_files = [p for p in d.rglob("*") if p.is_file()]
        if all_files:
            stems = [p.stem.lower() for p in all_files]
            box_id = Counter(stems).most_common(1)[0][0]
        else:
            box_id = d.name.lower()
    
        required = {k: pat.format(id=box_id) for k, pat in SPECIM_LUMO_REQUIRED.items()}
        files = {}
        missing, duplicates, zero_byte = [], {}, {}
        critical_missing = []
    
        # Define files that are CRITICAL for reflectance calculation
        CRITICAL_FILES = ["data head", "data raw", "white head", "white raw", "dark head", "dark raw"]
    
        for role, expected_name in required.items():
            matches = [p for p in all_files if p.name.lower() == expected_name.lower()]
            
            if not matches:
                missing.append(role)
                if role in CRITICAL_FILES:
                    critical_missing.append(role)
            elif len(matches) > 1:
                # Duplicates: Skip file and log the issue
                duplicates[role] = [str(m) for m in matches]
            else:
                f = matches[0]
                if f.stat().st_size <= 0:
                    # Zero-byte: Skip file and log the issue
                    zero_byte[role] = str(f)
                else:
                    files[role] = str(f)
                    
        # CRITICAL CHECK: Still raise an error if raw data/references are missing.
        if critical_missing:
            raise ValueError(
                f"Cannot open raw dataset: Critical files are missing or invalid: {critical_missing}"
            )
            
        # Create the instance, including the file issue report
        raw_object = cls(basename=box_id, root_dir=d, files=files)
        
        # Store all non-critical issues on the object for inspection/logging
        raw_object.file_issues = {
            "missing": missing,
            "duplicates": duplicates,
            "zero_byte": zero_byte,
        }
    
        # Add a print or logging statement for non-critical issues (optional)
        if missing or duplicates or zero_byte:
            print(f"⚠️ Warning: Non-critical files issues found for {box_id}:")
            if missing:
                print(f"  Missing (Skipped): {missing}")
            if duplicates:
                print(f"  Duplicates (Skipped): {duplicates}")
            if zero_byte:
                print(f"  Zero Byte (Skipped): {zero_byte}")
    
        return raw_object


    #TODO Might need a refactor when QAQC functions are integrated - not yet.
    def get_reflectance_QAQC(self, QAQC=True):
        """Load reflectance with optional QA/QC metrics (SNR)."""
        self.reflectance, self.bands, self.snr = sf.find_snr_and_reflect(self.files['data head'], self.files['white head'], self.files['dark head'], QAQC=QAQC)
        return self.reflectance, self.snr

    def get_reflectance(self):
        """Return or compute the reflectance cube (without QA/QC)."""
        if getattr(self, "reflectance", None) is None:
            self.reflectance, self.bands, self.snr = sf.find_snr_and_reflect(self.files['data head'], self.files['white head'], self.files['dark head'], QAQC=False)
        return self.reflectance
    def get_false_colour(self, bands=None):
        """Generate a false-colour RGB composite for visualization."""
        if hasattr(self, "reflectance") and self.reflectance is not None:
            return sf.get_false_colour(self.reflectance, bands=bands)
        
    def process(self):
        """
        Generate a ProcessedObject containing derived products.

        Returns
        -------
        ProcessedObject
            New instance populated with reflectance, bands, Savitzky–Golay,
            continuum-removed, mask, and metadata datasets.
        """
        if not hasattr(self, "reflectance") or self.reflectance is None:
            self.get_reflectance()
        if getattr(self, "temp_reflectance", None) is not None:
            self.reflectance = self.temp_reflectance
        po = ProcessedObject.new(self.root_dir, self.basename)
        po.add_dataset('metadata', self.metadata, ext='.json')
        po.add_dataset('cropped', self.reflectance, ext='.npy')
        po.add_dataset('bands', self.bands, ext='.npy')
        savgol, savgol_cr, mask = sf.process(self.reflectance)
        po.add_dataset('savgol', savgol, ext='.npy')
        po.add_dataset('savgol_cr', savgol_cr, ext='.npy')
        po.add_dataset('mask', mask, ext='.npy')
        return po
     
    def add_temp_reflectance(self, array):
        """
        Stage a temporary reflectance array (e.g., cropped) without committing.

        Parameters
        ----------
        array : np.ndarray
            Array whose last dimension matches the original reflectance bands.
        """
        if getattr(self, "reflectance", None) is not None:
            if array.shape[-1] == self.reflectance.shape[-1]:
                self.temp_reflectance = array
    def get_display_reflectance(self):
        """
        Return the temporary reflectance if present, else the base reflectance.
        """
        if getattr(self, "temp_reflectance", None) is not None:
            return self.temp_reflectance
        else:
            return self.reflectance
    # ---- registry API ----
    def keys(self):
        """Return sorted role names for files in the raw directory mapping."""
        return sorted(self.files.keys())

    def has(self, key: str):
        """Return True if the required file role exists."""
        return key in self.files
    
    def __getitem__(self, key):
        """Return the file path registered under the specified role key."""
        return self.files[key]

from datetime import datetime

def combine_timestamp(meta: dict) -> datetime | None:
    """
    Combine 'time' and 'date' fields in metadata into a datetime object.
    Returns None if either part is missing or invalid.
    """
    date_str = meta.get("date")
    time_str = meta.get("time")

    if not (date_str and time_str):
        return None

    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


@dataclass
class HoleObject:
    hole_id: str
    root_dir: Path
    num_box: int
    first_box: int
    last_box: int
    hole_meta: Dict[int, dict] = field(default_factory=dict)
    boxes: Dict[int, "ProcessedObject"] = field(default_factory=dict)

    @classmethod
    def build_from_box(cls, obj):
        if not isinstance(obj, ProcessedObject):
            obj = ProcessedObject.from_path(obj)
        try:
            hole_id = obj.metadata["borehole id"]
        except Exception as e:
            raise ValueError(f"Cannot extract 'borehole id' from metadata: {e}")
        return cls.build_from_parent_dir(obj.root_dir, hole_id)

    @classmethod
    def build_from_parent_dir(cls, path, hole_id: str = ""):
        root = Path(path)

        # ---- PASS 1: read JSON only (cheap) to detect dominant hole_id if not provided
        hole_ids: List[str] = []
        for fp in sorted(root.glob("*.json")):
            try:
                meta = json.loads(fp.read_text(encoding="utf-8"))
                h_id = meta.get("borehole id")
                if h_id:
                    hole_ids.append(str(h_id))
            except Exception:
                continue

        if not (hole_id and str(hole_id).strip()):
            if hole_ids:
                hole_id = Counter(hole_ids).most_common(1)[0][0]
            else:
                raise ValueError(f"No JSON in {root} contained a 'borehole id'.")

        # fresh, empty hole; counters will be filled by add_box
        hole = cls.new(hole_id=hole_id, root_dir=root)

        # ---- PASS 2: load only boxes; add_box will filter by hole_id & update counters
        for fp in sorted(root.glob("*.json")):
            try:
                po = ProcessedObject.from_path(fp)  # may memmap; acceptable for matching ones
                hole.add_box(po)                    # will skip/raise if hole_id mismatches
            except ValueError:
                # mismatched hole_id or bad metadata -> skip
                continue
            except Exception:
                # any other load error -> skip this file
                continue

        if hole.num_box == 0:
            raise ValueError(f"No boxes in {root} matched borehole id '{hole_id}'.")

        return hole

    @classmethod
    def new(cls,
            hole_id: str = "",
            root_dir: Path = Path("."),
            num_box: int = 0,
            first_box: int = 0,
            last_box: int = 0,
            hole_meta: dict | None = None,
            boxes: dict | None = None):
        return cls(
            hole_id=hole_id,
            root_dir=root_dir,
            num_box=num_box,
            first_box=first_box,
            last_box=last_box,
            hole_meta={} if hole_meta is None else dict(hole_meta),
            boxes={} if boxes is None else dict(boxes),
        )

    def add_box(self, obj) -> int:
        if not isinstance(obj, ProcessedObject):
            obj = ProcessedObject.from_path(obj)

        try:
            meta = obj.metadata
            box_hole_id = meta["borehole id"]
            box_num = int(meta["box number"])
        except Exception as e:
            raise ValueError(f"Box metadata missing required fields: {e}")

        # initialise / validate hole_id
        if not self.hole_id:
            self.hole_id = box_hole_id
        elif self.hole_id != box_hole_id:
            raise ValueError(
                f"Box hole_id '{box_hole_id}' does not match HoleObject.hole_id '{self.hole_id}'"
            )

        # initialise root_dir if empty
        if not getattr(self, "root_dir", None) or str(self.root_dir) == ".":
            self.root_dir = obj.root_dir

        # handle re-scans by box number
        if box_num in self.boxes:
            
            # if same basename, treat as duplicate and ignore
            if self.boxes[box_num].basename == obj.basename:
                return box_num
            # choose newer by timestamp
            old_t = combine_timestamp(self.boxes[box_num].metadata)
            new_t = combine_timestamp(meta)
            if old_t and new_t and new_t <= old_t:
                return box_num  # keep existing
            # else replace with newer
        # INSERT / REPLACE
        self.boxes[box_num] = obj
        self.hole_meta[box_num] = meta

        # update counters after insertion
        keys = sorted(self.boxes.keys())
        self.num_box = len(keys)
        self.first_box = keys[0] if keys else 0
        self.last_box = keys[-1] if keys else 0

        return box_num

    def check_for_all_keys(self, key):
        for i in self:
            tst = i.datasets.get(key)
            if not tst:
                return False
        return True
    
    def get_all_thumbs(self):
        for i in self:
            i.load_or_build_thumbs()
    
    def __iter__(self) -> Iterator["ProcessedObject"]:
        for bn in sorted(self.boxes):
            yield self.boxes[bn]

    def iter_items(self) -> Iterator[Tuple[int, "ProcessedObject"]]:
        for bn in sorted(self.boxes):
            yield bn, self.boxes[bn]

    def __len__(self) -> int:
        return self.num_box

    def __contains__(self, box_number: int) -> bool:
        return box_number in self.boxes

    def __getitem__(self, key: Union[int, slice, List[int]]) -> Union["ProcessedObject", List["ProcessedObject"]]:
        if isinstance(key, int):
            return self.boxes[key]
        elif isinstance(key, slice):
            ordered = sorted(self.boxes)
            selected = ordered[key]
            return [self.boxes[bn] for bn in selected]
        elif isinstance(key, list):
            return [self.boxes[bn] for bn in key]
        else:
            raise TypeError(f"Unsupported key type: {type(key).__name__}")


if __name__ == '__main__':
    test = HoleObject.build_from_parent_dir('D:/Multi_process_test/')
    test.get_all_thumbs()
    #%%
    for i in test:
        i.get_all_thumbs()
   
    
    