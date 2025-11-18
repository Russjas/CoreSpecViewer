"""
Created on Mon Nov 17 09:26:58 2025

@author: russj
"""


from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..spectral_ops import spectral_functions as sf
from .processed_object import ProcessedObject

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
    temp_reflectance: np.ndarray | None = field(default=None, repr=False)
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
        po.build_all_thumbs()
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
