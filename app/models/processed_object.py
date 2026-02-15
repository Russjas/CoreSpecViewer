"""
Represents a processed hyperspectral dataset.

Stores derived products (reflectance, SAVGOL, CR, masks, MWL results,
spectral maps) and supports temporary products, saving .npz outputs,
and UI-friendly dataset access.
"""
from dataclasses import dataclass, field
from pathlib import Path
import logging
import numpy as np
from PIL import Image

from ..spectral_ops import spectral_functions as sf
from .dataset import Dataset

base_datasets = ["cropped", "savgol", "savgol_cr", "mask", "bands", "metadata", "display"]
logger = logging.getLogger(__name__)


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
        logger.error(f"{name} not found in datasets or attributes")
        raise AttributeError(f"{name} not found in datasets or attributes")

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
            logger.error(f"Cannot infer basename from {p.name}; expected '<basename>_<suffix>.<ext>'.")
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
        obj = cls(basename=basename, root_dir=root, datasets=datasets)
        if 'display' not in obj.datasets:
            obj._generate_display()
            obj.datasets['display'].save_dataset()
            obj.reload_all()  # Drop back to memmaps   

        return obj


    @classmethod
    def load_post_processed_envi(cls, head_path, data_path, meta_path = None, smoothed = True):
        """
        Instantiate a ProcessedObject from a post processed dataset stored in an
        ENVI file.

        Parameters
        ----------
        head_path : str or Path
            Path to ENVI header file.
        data_path : str or Path
            Path to ENVI binary file.
        name : str
            name of the dataset (typically a holeID and box_number combination)

        Returns
        -------
        ProcessedObject
            Populated instance with all associated Dataset objects.

        Assumes 
        -------
        Full post-processing has been performed:
            - Data is reflectance
            - Noisy edge bands have been sliced away
            - Data has been smoothed

        Raises
        ------
        ValueError
            If the band names cannot be discovered.
        """
        path = Path(head_path)
        root = path.parent
        name = path.stem
        data, metadata = sf.load_envi(head_path, data_path)
        if meta_path is not None:
            metadata = metadata | sf.parse_lumo_metadata(meta_path)
        
        band_key, bands = sf.find_bands(metadata, data)
        
        if bands is None:
            logger.error("Cannot identify band names from the header file")
            raise ValueError("Cannot identify band names from the header file")
        if smoothed:
            savgol = data
            cropped = np.zeros_like(savgol)
            mask = np.zeros(savgol.shape[:2]).astype(int)
            savgol_cr = sf.cr(savgol)
        else:
            cropped = data
            savgol, savgol_cr, mask = sf.process(cropped)
        po = cls.new(root, name)
        po.add_dataset('metadata', metadata, ext='.json')
        po.add_dataset('cropped', cropped, ext='.npy')
        po.add_dataset('bands', bands, ext='.npy')
        po.add_dataset('savgol', savgol, ext='.npy')
        po.add_dataset('savgol_cr', savgol_cr, ext='.npy')
        po.add_dataset('mask', mask, ext='.npy')
        po._generate_display()
        po.build_all_thumbs()
        return po
        
    def _generate_display(self):
        """Generate RGB display dataset from savgol"""
        logger.info(f"Generating display dataset for {self.basename}")
        
        rgb = sf.get_false_colour(self.savgol)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        self.add_dataset("display", rgb_uint8, ext=".npy") 



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
    

    def export_images(self):
        for key in self.datasets.keys()|self.temp_datasets.keys():
            try:
                
                self.export_image(key)
                logger.info(f"exported {self.basename} {key}")
            except (ValueError, FileNotFoundError):
                logger.debug(f"export failed for  {self.basename} {key}", exc_info=True)
                logger.warning(f"export failed for  {self.basename} {key}")
                continue
        
        
    def export_image(self, key):
        output_dir = Path(self.root_dir) / 'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ds = self.temp_datasets.get(key)
        if ds is None:
            ds = self.datasets.get(key)
        if ds is None:
            return
        if key == "stats":
            logger.error(f"Cannot export {key}")
            return
        final_path = output_dir / f'{self.basename}-{key}.jpg'  
        logger.debug(f"Attempting export {self.basename} {key} to {final_path}")
        try:
            if ds.ext == ".npy" and getattr(ds.data, "ndim", 0) > 1:
                if key.startswith('Dhole'):
                    if key == 'DholeMask':
                        im = sf.mk_thumb(ds.data)
                        im.save(str(final_path), quality = 95)
                        logger.info(f"Exported {self.basename} {key}")
                    else:
                        mask_to_use = getattr(self, 'DholeMask', None)
                        if mask_to_use is not None:
                            if mask_to_use.ndim == 3:
                                mask_data = mask_to_use[:, :, 0]
                            elif mask_to_use.ndim == 2:
                                mask_data = mask_to_use
                            im = sf.mk_thumb(ds.data, mask=self.mask, index_mode=True, resize=False)
                            im.save(str(final_path), quality = 95)
                            logger.info(f"Exported {self.basename} {key}")
                        else:
                            logger.warning(f"Warning: 'DholeMask' not found on self for key {key}. Skipping.")
                            
                            return
                elif key == "mask":
                    im = sf.mk_thumb(ds.data)
                    im.save(str(final_path), quality = 95)
                    
                    logger.info(f"Exported {self.basename} {key}")
                elif key.endswith("INDEX"):
                    im = sf.mk_thumb(ds.data, mask=self.mask, index_mode=True, resize=False)
                    im.save(str(final_path), quality = 95)
                    
                    logger.info(f"Exported {self.basename} {key}")
                else:
                    im = sf.mk_thumb(ds.data, mask=self.mask, resize=False)
                    im.save(str(final_path), quality = 95)
                    
                    logger.info(f"Exported {self.basename} {key}")
                

            elif ds.ext == ".npz":
                im = sf.mk_thumb(ds.data.data, mask=ds.data.mask)
                im.save(str(final_path), quality = 95)
                
                logger.info(f"Exported {self.basename} {key}")
                               
            else:
                logger.warning(f"Failed to export {self.basename} {key}")
                
                return

        except ValueError:
            logger.error(f"ValueError exporting image for {key}")
            return
    
    def build_thumb(self, key):
        
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
                #im = sf.mk_thumb(ds.data)
                ds.thumb = im
            else:
                return

        except ValueError:
            logger.error(f"ValueError building thumb for {key}")
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
    
    def save_archive_file(self, output_dir: Path | str = None, 
                  include_products: bool = False) -> Path:
        """
        Save ProcessedObject as a legacy NPZ archive file.
        
        Creates an NPZ file containing cropped, mask, metadata, and bands datasets
        in a structured format with optional derived products.
        
        Parameters
        ----------
        output_path : Path | str, optional
            Path for the output NPZ file. If None, saves to root_dir with basename.
        include_products : bool, default=False
            If True, includes all derived products in a nested 'products' dictionary.
        
        Returns
        -------
        Path
            Path to the created NPZ file.
        
        Examples
        --------
        >>> po = ProcessedObject.from_path("mydata_cropped.npy")
        >>> # Save only base datasets
        >>> archive_path = po.save_archive_file()
        
        >>> # Save everything including all products
        >>> archive_path = po.save_archive_file(include_products=True)
        >>> 
        >>> # Save to specific location
        >>> po.save_archive_file("E:/Archives/", include_products=True)
        
        Notes
        -----
        NPZ structure:
        - cropped : reflectance data cube (ndarray)
        - mask : binary mask array (ndarray)
        - metadata : metadata dictionary (wrapped as numpy object)
        - bands : wavelength centers (ndarray)
        - products : nested dictionary (wrapped as numpy object) containing:
            - key : ndarray (for regular arrays)
            - keyDATA : ndarray (for masked arrays - data component)
            - keyMASK : ndarray (for masked arrays - mask component)
            - key : dict (for dictionary products like legends)
        
        The 'products' dictionary preserves the original data types:
        - Regular ndarrays are stored directly
        - Masked arrays are split into DATA and MASK components
        - Dictionaries are stored directly (nested within products dict)
        
        All Python objects (metadata dict, products dict, and nested dicts within
        products) are serialized using NumPy's pickle mechanism via dtype=object.
        
        To load:
        >>> with np.load(path, allow_pickle=True) as npz:
        >>>     cropped = npz['cropped']
        >>>     metadata = npz['metadata'].item()
        >>>     if 'products' in npz.files:
        >>>         products = npz['products'].item()
        >>>         # Access regular array:
        >>>         savgol = products['savgol']
        >>>         # Reconstruct masked array:
        >>>         stats = np.ma.array(products['statsDATA'], 
        >>>                            mask=products['statsMASK'])
        >>>         # Access nested dict:
        >>>         legend = products['MinMap-LEGEND']
        
        See Also
        --------
        hydrate_from_archive : Load ProcessedObject from legacy NPZ archive
        save_all : Save datasets to individual files
        """
        if output_dir is None:
            output_path = self.root_dir / f"{self.basename}.npz"
        else:
            output_path = Path(output_dir) / f"{self.basename}.npz"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {self.basename} as archive: {output_path}")
        
        # Build the base data dictionary
        # Arrays are stored directly, dicts are wrapped as numpy objects
        data_dict = {
            'cropped': self.cropped,
            'mask': self.mask,
            'metadata': np.array(self.metadata, dtype=object),
            'bands': self.bands
        }
        
        # Add products in nested structure if requested
        if include_products:
            products = {}
            
            for key, ds in self.datasets.items():
                # Skip the base ones already added
                if key in ('savgol', 'savgol_cr', 'metadata', 'bands', 'cropped', 'mask'):
                    continue
                
                # Load dataset if not already in memory
                if ds.data is None:
                    ds.load_dataset()
                
                # Handle different data types
                if isinstance(ds.data, dict):
                    # Dictionary (e.g., legends) - store directly in products dict
                    # It will be pickled as part of the outer products dict
                    products[key] = ds.data
                    logger.debug(f"Including dict: {key}")
                    
                elif isinstance(ds.data, np.ma.MaskedArray):
                    # Masked array - store data and mask separately with suffix
                    products[f"{key}DATA"] = ds.data.data
                    products[f"{key}MASK"] = ds.data.mask
                    logger.debug(f"Including masked_array: {key} (as {key}DATA, {key}MASK)")
                    
                elif isinstance(ds.data, np.ndarray):
                    # Regular ndarray - store directly
                    products[key] = ds.data
                    logger.debug(f"Including ndarray: {key}")
                    
                else:
                    logger.warning(f"Skipping unknown data type for {key}: {type(ds.data)}")
            
            data_dict['products'] = np.array(products, dtype=object)
            logger.info(f"Added {len(products)} products to archive")
        
        # Save the NPZ
        np.savez(output_path, **data_dict)
        
        logger.info(f"Legacy NPZ saved: {output_path}")
        return output_path
        
    
    @classmethod
    def hydrate_from_archive(cls, npz_path: Path | str, 
                            output_dir: Path | str,
                            basename: str = None,
                            load_products: bool = True) -> "ProcessedObject":
        """
        Create a ProcessedObject from an archive NPZ file.
        
        Loads base datasets and optionally derived products from an NPZ archive
        created by save_archive_file(). Always generates savgol and savgol_cr
        from the cropped data.
        
        Parameters
        ----------
        npz_path : Path | str
            Path to the archive NPZ file.
        output_dir : Path | str
            Directory where the new ProcessedObject files will be saved.
        basename : str, optional
            Name for the ProcessedObject. If None, uses the NPZ filename stem.
        load_products : bool, default=True
            Whether to load additional products from the 'products' dict if present.
        
        Returns
        -------
        ProcessedObject
            New ProcessedObject instance with all datasets extracted and saved.
        
        Examples
        --------
        >>> # Load archive with all products
        >>> po = ProcessedObject.hydrate_from_archive(
        ...     "mydata.npz",
        ...     "D:/HSI_Processed/"
        ... )
        
        >>> # Load to specific location with custom basename
        >>> po = ProcessedObject.hydrate_from_archive(
        ...     "E:/Archives/data.npz",
        ...     "D:/Restored/",
        ...     basename="16.101.HILLSTREET_56"
        ... )
        
        >>> # Load only base datasets, skip products
        >>> po = ProcessedObject.hydrate_from_archive(
        ...     "data.npz",
        ...     "output/",
        ...     load_products=False
        ... )
        
        Notes
        -----
        Expected NPZ structure:
        - cropped : reflectance data cube (ndarray)
        - mask : binary mask array (ndarray)
        - metadata : metadata dictionary (numpy object)
        - bands : wavelength centers (ndarray)
        - products (optional) : nested dictionary (numpy object) containing:
            - key : ndarray (for regular arrays)
            - keyDATA + keyMASK : ndarrays (for masked arrays)
            - key : dict (for dictionary products)
        
        The method will:
        1. Load base datasets (cropped, mask, metadata, bands)
        2. Generate savgol and savgol_cr from cropped data (always)
        3. Load products if load_products=True and products exist in archive
        4. Reconstruct masked arrays from DATA/MASK pairs
        5. Build thumbnails for all datasets
        6. Save all datasets to individual files
        
        Note: savgol and savgol_cr are never stored in the archive - they are
        always regenerated from cropped data to save space and ensure consistency.
        
        Raises
        ------
        FileNotFoundError
            If the NPZ file doesn't exist.
        KeyError
            If required base datasets (cropped, mask, metadata, bands) are missing.
        
        See Also
        --------
        save_archive_file : Create an archive NPZ file
        from_path : Load existing ProcessedObject from directory
        new : Create a new empty ProcessedObject
        """
        npz_path = Path(npz_path)
        output_dir = Path(output_dir)
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Archive NPZ file not found: {npz_path}")
        
        if basename is None:
            basename = npz_path.stem
        
        logger.info(f"Hydrating ProcessedObject from archive: {npz_path.name}")
        
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                try:
                    cropped = npz["cropped"]
                    mask = npz["mask"].astype(int)
                    metadata = npz["metadata"].item()
                    bands = npz["bands"]
                except KeyError as e:
                    raise KeyError(f"Missing required dataset in archive: {e}")
                
                # Create new ProcessedObject
                po = cls.new(output_dir, basename)
                po.add_dataset('metadata', metadata, ext='.json')
                po.add_dataset('cropped', cropped, ext='.npy')
                po.add_dataset('bands', bands, ext='.npy')
                po.add_dataset('mask', mask, ext='.npy')
                
                logger.debug(f"Loaded base datasets for {basename}")
                
                savgol, savgol_cr, _ = sf.process(cropped)
                po.add_dataset('savgol', savgol, ext='.npy')
                po.add_dataset('savgol_cr', savgol_cr, ext='.npy')
                logger.debug(f"Generated savgol and savgol_cr from cropped data")
                
                # Load products if present and requested
                if load_products and 'products' in npz.files:
                    products = npz['products'].item()
                    
                    # Track which keys are part of masked arrays
                    handled = set()
                    
                    # First pass: identify and reconstruct masked arrays
                    for key in list(products.keys()):
                        if key.endswith('DATA'):
                            base_key = key[:-4]  # Remove 'DATA'
                            mask_key = f"{base_key}MASK"
                            
                            if mask_key in products:
                                # Reconstruct masked array
                                data = np.ma.array(products[key], mask=products[mask_key])
                                po.add_dataset(base_key, data, ext='.npz')
                                handled.add(key)
                                handled.add(mask_key)
                                logger.debug(f"Loaded masked_array: {base_key}")
                    
                    # Second pass: load remaining products
                    for key, data in products.items():
                        if key in handled:
                            continue
                        
                        if isinstance(data, dict):
                            # Dictionary product (e.g., legends)
                            po.add_dataset(key, data, ext='.json')
                            logger.debug(f"Loaded dict: {key}")
                        elif isinstance(data, np.ndarray):
                            # Regular array
                            po.add_dataset(key, data, ext='.npy')
                            logger.debug(f"Loaded ndarray: {key}")
                        else:
                            logger.warning(f"Skipping unknown product type: {key} ({type(data)})")
                    
                    logger.info(f"Loaded {len(products)} products from archive")
                
                # Build thumbnails and save all datasets
                if "display" not in po.datasets.keys():
                    po._generate_display()
                po.build_all_thumbs()
                               
                logger.info(f"Successfully hydrated {basename} with {len(po.datasets)} datasets")
                return po
                
        except Exception as e:
            logger.error(f"Failed to hydrate from archive: {npz_path}", exc_info=True)
            raise
    
    
    
    
    def delete_dataset(self, key):
        
        """
        Remove a dataset from the object and optionally delete from disk.
        
        Parameters
        ----------
        key : str
            Dataset key to delete (e.g., 'mask', 'savgol_cr').
           
        Raises
        ------
        KeyError
            If the dataset key doesn't exist.
        
        Examples
        --------
        >>> po.delete_dataset('old_product')  # Delete file and remove from memory
        >>> po.delete_dataset('temp_data', from_disk=False)  # Remove from memory only
        """
        if key in base_datasets:
            return #You dont want to delete these datasets.
        
        # Check both permanent and temporary datasets
        if key in self.datasets:
            ds = self.datasets[key]
            location = self.datasets
        elif key in self.temp_datasets:
            ds = self.temp_datasets[key]
            location = self.temp_datasets
        else:
            raise KeyError(f"Dataset '{key}' not found in object")
        
        if key.endswith("INDEX"):
            try:
                self.delete_dataset(key.replace("INDEX", "LEGEND"))
            except (KeyError, FileNotFoundError):
                pass #not in keys or already deleted
        try:
            ds.delete()
        except FileNotFoundError:
            # File already gone, that's fine
            pass
               
        del location[key]
        