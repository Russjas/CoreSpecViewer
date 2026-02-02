"""
Represents an unprocessed hyperspectral scan.

Holds radiance data, illumination references, wavelengths, and metadata.
Provides reflectance conversion for further processing.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np

from ..spectral_ops import spectral_functions as sf
from .processed_object import ProcessedObject

logger = logging.getLogger(__name__)

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
    sensor: str = "sensor"
    def __post_init__(self):
        """On initialization, populate metadata and compute reflectance."""
        self.get_metadata()
        self.get_reflectance()
        
    def get_metadata(self):
        """Load and merge Specim XML + ENVI header metadata if available."""
        if 'metadata' in self.files.keys() and 'data head' in self.files.keys():
            self.metadata = sf.parse_lumo_metadata(self.files['metadata']) | sf.read_envi_header(self.files['data head'])
            self.sensor = self.metadata['sensor type']
        elif 'metadata' not in self.files.keys() and 'data head' in self.files.keys():
            self.metadata = sf.read_envi_header(self.files['data head'])
            self.sensor = self.metadata['sensor type']
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
            logger.info(f"Warning: Non-critical files issues found for {box_id}:")
            if missing:
                logger.warning(f"  Missing (Skipped): {missing}")
            if duplicates:
                logger.warning(f"  Duplicates (Skipped): {duplicates}")
            if zero_byte:
                logger.warning(f"  Zero Byte (Skipped): {zero_byte}")

        return raw_object


    #TODO Might need a refactor when QAQC functions are integrated - not yet.
    def get_reflectance_QAQC(self, QAQC=True):
        """Load reflectance with optional QA/QC metrics (SNR)."""
        if getattr(self, "reflectance", None) is not None:
            return self.reflectance

        if "fenix" not in self.sensor.lower():
            self.reflectance, self.bands, self.snr = sf.find_snr_and_reflect(
                self.files['data head'],
                self.files['white head'],
                self.files['dark head'],
                QAQC=QAQC,
                data_data_path=self.files['data raw'],
                white_data_path=self.files['white raw'],
                dark_data_path=self.files['dark raw'],
            )
        else:
            logger.debug(f"length of loaded bands {len(self.metadata['wavelength'])}")
            if len(self.metadata['wavelength']) < 400:
                self.reflectance, self.bands, self.snr = sf.get_fenix_reflectance(str(self.root_dir), mode='hylite')
            else:
                self.reflectance, self.bands, self.snr = sf.get_fenix_reflectance(str(self.root_dir), mode='derived')
    
        return self.reflectance

    def get_reflectance(self):
        
        """
        Return or compute the reflectance cube (without QA/QC).
        Fenix systems require fish-eye and other corrections, this is all off-loaded to hylite,
        requiring a different pathway through Raw object.
        """
        if getattr(self, "reflectance", None) is not None:
            return self.reflectance
        logger.info(f"{(self.sensor.lower())} sensor detected")
        if "fenix" not in self.sensor.lower():
            self.reflectance, self.bands, self.snr = sf.find_snr_and_reflect(
                self.files['data head'],
                self.files['white head'],
                self.files['dark head'],
                QAQC=False,
                data_data_path=self.files['data raw'],
                white_data_path=self.files['white raw'],
                dark_data_path=self.files['dark raw'],
            )
        else:
            logger.debug(f"length of loaded bands {len(self.metadata['wavelength'])}")
            if len(self.metadata['wavelength']) < 400:
                self.reflectance, self.bands, self.snr = sf.get_fenix_reflectance(str(self.root_dir), mode='hylite')
            else:
                self.reflectance, self.bands, self.snr = sf.get_fenix_reflectance(str(self.root_dir), mode='derived')
    
        return self.reflectance
    def get_false_colour(self, bands=None):
        """Generate a false-colour RGB composite for visualization."""
        if hasattr(self, "reflectance") and self.reflectance is not None:
            return sf.get_false_colour(self.reflectance, bands=bands)
        
    
    @classmethod  
    def manual_create_from_multiple_paths(
        cls, 
        data_head_path,
        white_head_path,
        dark_head_path,
        metadata_path: str | Path = ""
    ):
        """
        Manually build a RawObject from explicit header paths.
    
        Parameters
        ----------
        data_head_path : str | Path
            Path to the DATA header file (e.g., *.hdr).
        white_head_path : str | Path
            Path to the WHITE reference header file.
        dark_head_path : str | Path
            Path to the DARK reference header file.
        metadata_path : str | Path, optional
            Optional path to a metadata/header file (can be empty).
    
        Notes
        -----
        - The corresponding RAW files are inferred by swapping the suffix
          of each header path to '.raw'.
        - Critical files (data/white/dark head+raw) must exist and be
          non-zero size or a ValueError is raised.
        - Non-critical issues (e.g., missing metadata) are stored on
          `raw_object.file_issues` and printed as warnings.
        """
        from pathlib import Path
    
        # Normalise inputs to Path objects
        data_head_path = Path(data_head_path)
        white_head_path = Path(white_head_path)
        dark_head_path = Path(dark_head_path)
        metadata_path = Path(metadata_path) if metadata_path else None
    
        # Infer a box_id and root_dir from the data header
        box_id = data_head_path.stem.lower()
        root_dir = data_head_path.parent
    
        files: dict[str, str] = {}
        missing: list[str] = []
        duplicates: dict[str, list[str]] = {}  # no real duplicates in manual mode, but keep shape
        zero_byte: dict[str, str] = {}
        critical_missing: list[str] = []
    
        # CRITICAL file roles (same semantics as from_Lumo_directory)
        CRITICAL_FILES = ["data head", "data raw", "white head", "white raw", "dark head", "dark raw"]
    
        def _check_and_add_head_and_raw(role_head: str, head_path: Path) -> None:
            """Validate a head file and its inferred raw companion."""
            nonlocal files, missing, zero_byte, critical_missing
    
            if head_path is None:
                missing.append(role_head)
                critical_missing.append(role_head)
                return
    
            if not head_path.exists():
                missing.append(role_head)
                critical_missing.append(role_head)
                return
    
            if head_path.stat().st_size <= 0:
                zero_byte[role_head] = str(head_path)
                critical_missing.append(role_head)
                return
    
            # Store the header
            files[role_head] = str(head_path)
    
            # Now infer and validate the RAW file
            role_raw = role_head.replace("head", "raw")
            raw_path = head_path.with_suffix(".raw")
    
            if not raw_path.exists():
                missing.append(role_raw)
                critical_missing.append(role_raw)
                return
    
            if raw_path.stat().st_size <= 0:
                zero_byte[role_raw] = str(raw_path)
                critical_missing.append(role_raw)
                return
    
            files[role_raw] = str(raw_path)
    
        # --- Validate critical triplets: data / white / dark ---
        _check_and_add_head_and_raw("data head", data_head_path)
        _check_and_add_head_and_raw("white head", white_head_path)
        _check_and_add_head_and_raw("dark head", dark_head_path)
    
        # --- Optional metadata (non-critical) ---
        if metadata_path is not None:
            if not metadata_path.exists():
                missing.append("metadata")
            elif metadata_path.stat().st_size <= 0:
                zero_byte["metadata"] = str(metadata_path)
            else:
                files["metadata"] = str(metadata_path)
    
        # CRITICAL CHECK: Still raise an error if any critical files are missing/invalid
        if critical_missing:
            logger.error(f"Cannot create raw dataset: Critical files are missing or invalid: {critical_missing}")
            raise ValueError(
                f"Cannot create raw dataset: Critical files are missing or invalid: {critical_missing}"
            )
    
        # Create the instance
        raw_object = cls(basename=box_id, root_dir=root_dir, files=files)
    
        # Store non-critical issues on the object for inspection/logging
        raw_object.file_issues = {
            "missing": missing,
            "duplicates": duplicates,
            "zero_byte": zero_byte,
        }
    
        # Optional: print warnings about non-critical issues
        if missing or duplicates or zero_byte:
            logger.warning(f" Warning: Non-critical files issues found for {box_id}:")
            if missing:
                logger.warning(f"  Missing (Skipped): {missing}")
            if duplicates:
                logger.warning(f"  Duplicates (Skipped): {duplicates}")
            if zero_byte:
                logger.warning(f"  Zero Byte (Skipped): {zero_byte}")
    
        return raw_object
    
    
    @classmethod
    def manual_create_from_critical_paths(
        cls,
        data_head_path,
        data_raw_path,
        white_head_path,
        white_raw_path,
        dark_head_path,
        dark_raw_path,
        metadata_path: str | Path | None = None,
    ):
        """
        Build a RawObject from explicitly provided paths to the six critical
        Specim files, plus an optional metadata file.
    
        Critical roles:
            - data head
            - data raw
            - white head
            - white raw
            - dark head
            - dark raw
    
        Any problem with these critical files (missing or zero-byte)
        leads to a ValueError. The metadata file is optional and only
        generates non-critical file_issues.
        """
        from pathlib import Path
    
        # Normalise to Path objects
        paths = {
            "data head": Path(data_head_path),
            "data raw": Path(data_raw_path),
            "white head": Path(white_head_path),
            "white raw": Path(white_raw_path),
            "dark head": Path(dark_head_path),
            "dark raw": Path(dark_raw_path),
        }
    
        missing = []
        zero_byte = {}
        duplicates = {}          # manual mode → never duplicates, but keep structure
        critical_missing = []
    
        # --- Validate critical files ---
        for role, p in paths.items():
            if not p.exists():
                missing.append(role)
                critical_missing.append(role)
            else:
                if p.stat().st_size <= 0:
                    zero_byte[role] = str(p)
                    critical_missing.append(role)
    
        # Raise for *any* critical failure
        if critical_missing:
            logger.error(f"Cannot create raw dataset: Critical files are missing or invalid: {critical_missing}")
            raise ValueError(
                f"Cannot create raw dataset: Critical files are missing or invalid: {critical_missing}"
            )
    
        # Build the files dict for RawObject
        files = {role: str(p.resolve()) for role, p in paths.items()}
    
        # --- Optional metadata (non-critical) ---
        if metadata_path:
            m = Path(metadata_path)
            if not m.exists():
                missing.append("metadata")
            else:
                if m.stat().st_size <= 0:
                    zero_byte["metadata"] = str(m)
                else:
                    files["metadata"] = str(m.resolve())
    
        # --- Infer basename + root_dir ---
        data_head = paths["data head"]
        parent_dirs = {p.parent for p in paths.values()}
    
        if len(parent_dirs) == 1:
            root_dir = parent_dirs.pop()
        else:
            # Mangled set → default to data head location
            root_dir = data_head.parent
    
        box_id = data_head.stem.lower()
    
        # --- Create the RawObject ---
        raw_object = cls(
            basename=box_id,
            root_dir=root_dir,
            files=files
        )
    
        # --- Attach the file_issues report (non-critical only) ---
        raw_object.file_issues = {
            "missing": missing,
            "duplicates": duplicates,
            "zero_byte": zero_byte,
        }
    
        # Optional warning summary
        if missing or duplicates or zero_byte:
            logger.warning(f"Warning: Non-critical files issues found for {box_id}:")
            if missing:
                logger.warning(f"  Missing (Skipped): {missing}")
            if duplicates:
                logger.warning(f"  Duplicates (Skipped): {duplicates}")
            if zero_byte:
                logger.warning(f"  Zero Byte (Skipped): {zero_byte}")
    
        return raw_object
    
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
