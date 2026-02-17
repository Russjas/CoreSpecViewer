"""
Container for multiple scanned core boxes belonging to a drill hole.

Manages ordering, metadata, merged downhole tables, and propagation of
derived datasets between boxes.
"""
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Union
import logging

import numpy as np

from ..spectral_ops import spectral_functions as sf
from ..spectral_ops import downhole_resampling as res
from .processed_object import ProcessedObject
from .dataset import Dataset

logger = logging.getLogger(__name__)
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
    hole_meta: dict[int, dict] = field(default_factory=dict)
    boxes: dict[int, "ProcessedObject"] = field(default_factory=dict)
    base_datasets: dict = field(default_factory=dict)
    product_datasets: dict = field(default_factory=dict)
    step: float = 0.05
    
    
    #==================fullhole dataset operations=============================
    """
    Hole owned datasets will always have hole_id as the basename
    path = self.root_dir / f"{self.hole_id}_{key}{ext}"
    #ds = Dataset(base=self.hole_id, key=key, path=path, suffix=key, ext=ext, data=data)
    """
    def load_hole_datasets(self):
        for fp in self.root_dir.iterdir():
            if not fp.is_file():
                continue
            s = fp.stem
            base, sep, key = s.rpartition("_")
            
            if base is None or base != self.hole_id:
                continue  # not fullhole dataset
            
            ext = fp.suffix if fp.suffix.startswith(".") else fp.suffix

            ds = Dataset(base=self.hole_id, key=key, path=fp, suffix=key, ext=ext)
            base_keys = ["depths", "AvSpectra"]
            if key in base_keys:
                self.base_datasets[key] = ds
            else:
                self.product_datasets[key] = ds
        return
    
    
    def add_product_dataset(self, key: str, data, ext: str = None):
        """
        Generic method to add a product dataset to the HoleObject.
        
        Args:
            key: The dataset key/identifier
            data: The dataset data to store
            ext: File extension (e.g., '.npy', '.npz', '.json'). 
                 If None, will be inferred from data type:
                 - dict -> '.json'
                 - masked array -> '.npz'
                 - regular array -> '.npy'
        """
        if ext is None:
            # Infer extension from data type
            if isinstance(data, dict):
                ext = ".json"
            elif hasattr(data, 'mask'):  # masked array
                ext = ".npz"
            else:  # regular numpy array
                ext = ".npy"
        
        # Ensure extension starts with a dot
        if not ext.startswith("."):
            ext = f".{ext}"
        
        path = self.root_dir / f"{self.hole_id}_{key}{ext}"
        
        self.product_datasets[key] = Dataset(
            base=self.hole_id,
            key=key,
            path=path,
            suffix=key,
            ext=ext,
            data=data
        )
    
   # Functions for aggregating 1D from 2D 
    def create_base_datasets(self):
        """
        This function will only work if every po in hole has had unwrapped stats calculated.
        Returns HoleObject with unwrapped, concatenated downhole datasets.
        Does not account for missing boxes in this step
        """
        if not self.check_for_all_keys('stats'):
            logger.warning("Missing 'stats' data for one or more boxes in the hole. Calculate stats before calling this method.")
            raise ValueError("Missing 'stats' data for one or more boxes in the hole. Calculate stats before calling this method.")
        full_depths = None
        full_average = None
        try:
            import time
            start = time.perf_counter()
            full_depths_list = []
            full_average_list = []
            
# =============================================================================
            for po in self:
                img = sf.unwrap_from_stats(po.mask, po.savgol, po.stats)
                checkpoint_1 = time.perf_counter()
                logger.debug(f"Unwrapped {po.datasets['metadata'].data['box number']}: {checkpoint_1 - start:.4f}s")
                depths = np.linspace(float(po.metadata['core depth start']), 
                                     float(po.metadata['core depth stop']),
                                     img.shape[0])
                full_depths_list.append(depths)
                full_average_list.append(np.ma.mean(img, axis=1))
                checkpoint_2 = time.perf_counter()
                logger.debug(f"appended results {checkpoint_2 - checkpoint_1:.4f}s")
                po.save_all()
                checkpoint_3 = time.perf_counter()
                logger.debug(f"saved all {checkpoint_3 - checkpoint_2:.4f}s")
                po.reload_all()
                checkpoint_4 = time.perf_counter()
                logger.debug(f"Reloaded all {checkpoint_4 - checkpoint_3:.4f}s")
                logger.info(f"Processed {self.hole_id} box number {po.metadata['box number']}")

        
        except Exception as e:
            logger.error(f'many, many things could have gone wrong', exc_info=True)
            return self
        
        
        full_depths = np.concatenate(full_depths_list)
        full_average = np.ma.vstack(full_average_list)
        checkpoint_final = time.perf_counter()
        logger.debug(f"Total for {self.num_box}: {checkpoint_final - start:.4f}s")
        self.base_datasets['depths'] = Dataset(base=self.hole_id, 
                                          key="depths", 
                                          path=self.root_dir / f"{self.hole_id}_depths.npy", 
                                          suffix="depths", 
                                          ext=".npy", 
                                          data=full_depths)
        self.base_datasets['AvSpectra'] = Dataset(base=self.hole_id, 
                                          key="AvSpectra", 
                                          path=self.root_dir / f"{self.hole_id}_AvSpectra.npy", 
                                          suffix="AvSpectra", 
                                          ext=".npy", 
                                          data=full_average.data)
        
        
        for ds in self.base_datasets.values():
            ds.save_dataset()
        return self
        
    def create_dhole_minmap(self, key):
        """
        Returns HoleObject with unwrapped, concatenated Mineral map datasets.
        Requires all boxes to have a ...INDEX and ...LEGEND style mineral map,
        and that all boxes have identical legends.
        Does not account for missing boxes in this step
        """
        if not self.check_for_all_keys(key):
            logger.warning(f"Missing {key} data for one or more boxes in the hole. Calculate stats before calling this method.")
            raise ValueError(f"{key} dataset is not available for every box in hole")
        if not (key.endswith("INDEX") or key.endswith("LEGEND")):
            logger.warning(f"{key} is an invalid dataset for this operation")
            raise ValueError(f"{key} is an invalid dataset for this operation")
        
        if key.endswith("INDEX"):
            leg_key = key.replace("INDEX", "LEGEND")
            ind_key = key
        elif key.endswith("LEGEND"):
            leg_key = key
            ind_key = key.replace("LEGEND", "INDEX")
            
        #check all legends are the same, not working with different versions
        dicts = [po.datasets[leg_key].data for po in self]
        if not all(d == dicts[0] for d in dicts[1:]):
            logger.warning(f"Boxes with {key} have different Legend entries")
            raise ValueError(f"Boxes with {key} have different Legend entries")
            
        full_fractions = None    # will become (H_total, K+1)
        full_dominant  = None 
        legend = dicts[0]
        
        for po in self:
            seg = sf.unwrap_from_stats(po.mask, po.datasets[ind_key].data, po.stats)
            fractions, dominant = sf.compute_downhole_mineral_fractions(seg.data, seg.mask, 
                                                                     po.datasets[leg_key].data)
            if full_fractions is None:
                # First box → just take it as-is
                full_fractions = fractions      # shape (H_box, K+1)
                full_dominant  = dominant       # shape (H_box,)
            else:
                # Append this box below the existing full arrays
                full_fractions = np.vstack((full_fractions, fractions))
                full_dominant  = np.concatenate((full_dominant, dominant))
            po.reload_all()
        
        fracs_key = ind_key.replace("INDEX", "FRACTIONS")
        dom_key = ind_key.replace("INDEX", "DOM-MIN")
        self.add_product_dataset(fracs_key, full_fractions, ext=".npy")
        self.add_product_dataset(dom_key, full_dominant, ext=".npy")
        self.add_product_dataset(leg_key, legend, ext=".json")
        
        logger.info("Base datasets successfully created!")
        
    def create_dhole_features(self, key):
        """
        Returns HoleObject with unwrapped, concatenated feature datasets.
        Pos and dep datasets must be passed separately, no discovery is included.
        Datasets must me masked arrays.
        Does not account for missing boxes in this step
        """
        if not self.check_for_all_keys(key):
            logger.warning(f"Missing {key} data for one or more boxes in the hole. Calculate stats before calling this method.")
            raise ValueError(f"{key} dataset is not available for every box in hole")
        
        full_feature = None    # will become (H_total, K+1)
        for po in self:
            if po.datasets[key].ext != ".npz":
                logger.warning(f"Box {po.metadata['box number']} {key} dataset is not a masked array.")
                raise ValueError(f"Box {po.metadata['box number']} {key} dataset is not a masked array.")
                
            seg = sf.unwrap_from_stats(po.datasets[key].data.mask, po.datasets[key].data.data, po.stats)
            feat_row = np.ma.mean(seg, axis=1)
            if full_feature is None:
                full_feature = feat_row
            else:
                full_feature  = np.ma.concatenate((full_feature, feat_row))
            po.reload_all()
        
        self.add_product_dataset(key, full_feature, ext=".npz")

#==================================================================
    
    def step_product_dataset(self, key):
        """
        Step product dataset based on suffix.
        
        Returns:
            Tuple format depends on data type:
            - FRACTIONS: (depths, fractions, dominant)
            - DOM-MIN: (depths, dominant_indices, None)
            - INDEX: (depths, indices, None)
            - Continuous: (depths, values, None)
        """
        invalid_keys_suffixes = ("LEGEND", "CLUSTERS")
        if key not in self.product_datasets.keys():
            logger.warning(f"Datasets of type {key} are not present")
            raise ValueError("bounced no dataset")
        for suffix in invalid_keys_suffixes:
            if key.endswith(suffix):
                logger.warning(f"Cannot step {suffix} datasets: '{key}'")
                raise ValueError(f"Cannot step {suffix} datasets: '{key}'")
        depths = self.base_datasets["depths"].data
        data = self.product_datasets[key].data
        
        # 1. FRACTIONS-pair path
        if key.endswith("FRACTIONS"):
            depths_s, fractions_s, dominant_s = res.step_fractions_pair(
                depths, data, self.step
            )
            return depths_s, fractions_s, dominant_s
        
        # 2. DOM-MIN path on (the pair path)
        elif key.endswith("DOM-MIN"):
            data = self.product_datasets[key.replace("DOM-MIN", "FRACTIONS")].data
            depths_s, fractions_s, dominant_s = res.step_fractions_pair(
                depths, data, self.step
            )
            return depths_s, fractions_s, dominant_s
        
        # 3. INDEX path
        elif key.endswith("INDEX"):
            depths_s, indices_s = res.step_indices(
                depths, data, self.step
            )
            return depths_s, indices_s, None
        
        # 4. Continuous path (default)
        else:
            depths_s, values_s = res.step_continuous(
                depths, data, self.step, agg='mean'
            )
            return depths_s, values_s, None



    def save_product_datasets(self):
        for key in self.product_datasets.keys():
            self.product_datasets[key].save_dataset()
        
#================box level functions ==========================================
    
    

    @classmethod
    def build_from_box(cls, obj):
        if not isinstance(obj, ProcessedObject):
            obj = ProcessedObject.from_path(obj)
        try:
            hole_id = obj.metadata["borehole id"]
        except Exception as e:
            logger.error(f"Cannot extract 'borehole id' from metadata: {e}")
            raise ValueError(f"Cannot extract 'borehole id' from metadata: {e}")
        return cls.build_from_parent_dir(obj.root_dir, hole_id)

    @classmethod
    def build_from_parent_dir(cls, path, hole_id: str = ""):
        root = Path(path)

        # ---- PASS 1: read JSON only (cheap) to detect dominant hole_id if not provided
        hole_ids: list[str] = []
        for fp in sorted(root.glob("*_metadata.json")):
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
                logger.error(f"No JSON in {root} contained a 'borehole id'.")
                raise ValueError(f"No JSON in {root} contained a 'borehole id'.")

        # fresh, empty hole; counters will be filled by add_box
        hole = cls.new(hole_id=hole_id, root_dir=root)

        # ---- PASS 2: load only boxes; add_box will filter by hole_id & update counters
        for fp in sorted(root.glob("*_metadata.json")):
            try:
                po = ProcessedObject.from_path(fp)  # may memmap; acceptable for matching ones
                logger.info(f"Loaded {po.basename}")
                hole.add_box(po)                    # will skip/raise if hole_id mismatches
            except ValueError:
                logger.error(f"mismatched hole_id or bad metadata -> skipped {fp}")
                # mismatched hole_id or bad metadata -> skip
                continue
            except Exception:
                logger.error(f"Load error", exc_info=True)
                continue

        if hole.num_box == 0:
            logger.error(f"No boxes in {root} matched borehole id '{hole_id}'.")
            raise ValueError(f"No boxes in {root} matched borehole id '{hole_id}'.")
        hole.load_hole_datasets()
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
            logger.error(f"Box metadata missing required fields")
            raise ValueError(f"Box metadata missing required fields: {e}")

        # initialise / validate hole_id
        if not self.hole_id:
            self.hole_id = box_hole_id
        elif self.hole_id != box_hole_id:
            logger.error(f"Box hole_id '{box_hole_id}' does not match HoleObject.hole_id '{self.hole_id}'")
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

    def get_bands(self):
        """Returns the band centres of the first box of the hole
        Boxes are checked for band consistency on load"""
        if self.first_box is None:
            raise ValueError("No boxes available in HoleObject")
        return self[self.first_box].bands
    
    def save_hole_archive(self, archive_dir: Path | str = None) -> Path:
        """
        Save hole-level products as archive.
        Box archives should be created separately.
        """
        if archive_dir is None:
            archive_dir = self.root_dir / "archives"
        else:
            archive_dir = Path(archive_dir)
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{self.hole_id}_HOLE_PRODUCTS.npz"
        
        # Build products dict
        base_dict = {}
        products = {}
        
        # Add base datasets
        for key, ds in self.base_datasets.items():
            if ds.data is None:
                ds.load_dataset()
            base_dict[key] = ds.data
        
        # Add product datasets
        for key, ds in self.product_datasets.items():
            if ds.data is None:
                ds.load_dataset()
            
            if isinstance(ds.data, np.ma.MaskedArray):
                products[f"{key}DATA"] = ds.data.data
                products[f"{key}MASK"] = ds.data.mask
            elif isinstance(ds.data, dict):
                products[key] = ds.data
            else:
                products[key] = ds.data
        output_dict = {}
        output_dict["base_datasets"] = base_dict
        output_dict["product_datasets"] = products
        # Save archive
        np.savez(
            archive_path,
            hole_id=np.array(self.hole_id, dtype=object),
            num_box=self.num_box,
            first_box=self.first_box,
            last_box=self.last_box,
            hole_meta=np.array(self.hole_meta, dtype=object),
            files=np.array(output_dict, dtype=object)
        )
        
        logger.info(f"Saved hole archive: {archive_path}")
        return archive_path
    
    def save_full_hole_archive(self, archive_dir: Path | str = None,
                               include_box_products: bool = False) -> Path:
        """
        Archive entire hole: all boxes + hole products.
        """
        if archive_dir is None:
            archive_dir = self.root_dir / "archives"
        else:
            archive_dir = Path(archive_dir)
        
        boxes_dir = archive_dir / "boxes"
        boxes_dir.mkdir(parents=True, exist_ok=True)
        
        # Archive all boxes
        logger.info(f"Archiving {len(self.boxes)} boxes...")
        for box_num, po in self.boxes.items():
            po.save_archive_file(
                output_dir=boxes_dir,
                include_products=include_box_products
            )
            logger.debug(f"Archived box {box_num}")
        
        # Archive hole products
        hole_archive = self.save_hole_archive(archive_dir)
        
        logger.info(f"Full hole archive complete: {archive_dir}")
        return archive_dir
    
    
    @classmethod
    def hydrate_hole_from_archive(cls, archive_dir: Path | str, save_dir: Path | str) -> "HoleObject":
        """
        Restore HoleObject from archive directory.
        
        Two modes:
        1. Full hole archive: Loads hole products + boxes from *_HOLE_PRODUCTS.npz + boxes/*.npz
        2. Boxes-only archive: Builds HoleObject from boxes/*.npz only (no hole products)
        
        Parameters
        ----------
        archive_dir : Path | str
            Directory containing either:
            - *_HOLE_PRODUCTS.npz + boxes/ subdirectory (full archive), OR
            - boxes/ subdirectory only (boxes-only archive)
        save_dir : Path | str
            Directory where hydrated data will be saved
            
        Returns
        -------
        HoleObject
            Hydrated hole object with all boxes loaded
            
        Notes
        -----
        Disk space must be available to cache all products to disk.
        
        Full archive includes downhole products (depths, AvSpectra, mineral profiles).
        Boxes-only archive reconstructs HoleObject without these products - they can
        be regenerated later using create_base_datasets(), create_dhole_minmap(), etc.
        """
        archive_dir = Path(archive_dir)
        save_dir = Path(save_dir)
        
        # Check for hole products archive
        hole_npz_files = list(archive_dir.glob("*_HOLE_PRODUCTS.npz"))
        
        if hole_npz_files:
            # ============================================================
            # MODE 1: Full hole archive (hole products + boxes)
            # ============================================================
            logger.info(f"Found hole products archive, loading full hole")
            hole_npz = hole_npz_files[0]
            
            with np.load(hole_npz, allow_pickle=True) as npz:
                hole_id = str(npz['hole_id'].item())
                num_box = int(npz['num_box'])
                first_box = int(npz['first_box'])
                last_box = int(npz['last_box'])
                hole_meta = npz['hole_meta'].item()
                files_dict = npz['files'].item()
            
            # Create HoleObject with metadata from archive
            ho = cls(
                hole_id=hole_id,
                root_dir=save_dir,
                num_box=num_box,
                first_box=first_box,
                last_box=last_box,
                hole_meta=hole_meta
            )
            
            # Restore base datasets
            for key, data in files_dict['base_datasets'].items():
                path = save_dir / f"{hole_id}_{key}.npy"
                ds = Dataset(base=hole_id, key=key, path=path, suffix=key, ext='.npy', data=data)
                ho.base_datasets[key] = ds
            
            # Restore product datasets
            handled = set()
            
            # First pass: reconstruct masked arrays
            for key in list(files_dict['product_datasets'].keys()):
                if key.endswith('DATA'):
                    base_key = key[:-4]
                    mask_key = f"{base_key}MASK"
                    
                    if mask_key in files_dict['product_datasets']:
                        data = np.ma.array(
                            files_dict['product_datasets'][key],
                            mask=files_dict['product_datasets'][mask_key]
                        )
                        path = save_dir / f"{hole_id}_{base_key}.npz"
                        ds = Dataset(base=hole_id, key=base_key, path=path, suffix=base_key, ext='.npz', data=data)
                        ho.product_datasets[base_key] = ds
                        handled.add(key)
                        handled.add(mask_key)
                        logger.debug(f"Restored masked array: {base_key}")
            
            # Second pass: load remaining products (dicts and regular arrays)
            for key, data in files_dict['product_datasets'].items():
                if key in handled:
                    continue
                
                if isinstance(data, dict):
                    path = save_dir / f"{hole_id}_{key}.json"
                    ds = Dataset(base=hole_id, key=key, path=path, suffix=key, ext='.json', data=data)
                    ho.product_datasets[key] = ds
                    logger.debug(f"Restored dict: {key}")
                elif isinstance(data, np.ndarray):
                    path = save_dir / f"{hole_id}_{key}.npy"
                    ds = Dataset(base=hole_id, key=key, path=path, suffix=key, ext='.npy', data=data)
                    ho.product_datasets[key] = ds
                    logger.debug(f"Restored ndarray: {key}")
            
            logger.info(f"Restored {len(ho.base_datasets)} base datasets and {len(ho.product_datasets)} product datasets")
            
        else:
            # ============================================================
            # MODE 2: Boxes-only archive (no hole products)
            # ============================================================
            logger.info(f"No hole products archive found, building HoleObject from boxes only")
            
            # Check that boxes directory exists
            boxes_dir = archive_dir / "boxes"
            if not boxes_dir.exists():
                boxes_dir = archive_dir
            if not boxes_dir.exists():
                raise FileNotFoundError(
                    f"No hole products archive (*_HOLE_PRODUCTS.npz) and no boxes/ directory found in {archive_dir}"
                )
            
            box_archives = sorted(boxes_dir.glob("*.npz"))
            if not box_archives:
                raise FileNotFoundError(f"No box archives found in {boxes_dir}")
            
            # Quickly determine hole_id by checking metadata in all boxes
            hole_ids = []
            for box_archive in box_archives:
                try:
                    with np.load(box_archive, allow_pickle=True) as npz:
                        if 'metadata' in npz:
                            meta = npz['metadata'].item()
                            h_id = meta.get("borehole id")
                            if h_id:
                                hole_ids.append(str(h_id))
                except Exception as e:
                    logger.warning(f"Could not read metadata from {box_archive.name}: {e}")
                    continue
            
            if not hole_ids:
                raise ValueError(
                    f"No box archives in {boxes_dir} contain 'borehole id' in metadata. "
                    "Cannot determine hole_id."
                )
            
            # Use most common hole_id (handles edge cases with mixed boxes)
            from collections import Counter
            hole_id = Counter(hole_ids).most_common(1)[0][0]
            
            if len(set(hole_ids)) > 1:
                logger.warning(
                    f"Multiple hole IDs found in box archives: {set(hole_ids)}. "
                    f"Using most common: {hole_id}"
                )
            
            logger.info(f"Determined hole_id: {hole_id} from {len(box_archives)} box archives")
            
            # Create empty HoleObject (counters will be updated by add_box)
            ho = cls.new(hole_id=hole_id, root_dir=save_dir)
            
        ho._add_archived_boxes(boxes_dir)        
        logger.info(f"Hydrated hole {ho.hole_id} with {ho.num_box} boxes")
        return ho
    
    
    def _add_archived_boxes(self, archive_dir: Path | str):
        archive_dir = Path(archive_dir)
        
        if archive_dir.exists():
            box_archives = sorted(archive_dir.glob("*.npz"))
            logger.info(f"Loading {len(box_archives)} box archives")
            
            for box_archive in box_archives:
                po = ProcessedObject.hydrate_from_archive(box_archive, self.root_dir)
                po.save_all()
                po.reload_all()
                self.add_box(po)
                logger.info(f"added box {po.metadata['box number']} to {self.hole_id}")
        else:
            logger.warning(f"No boxes/ directory found in {archive_dir}")
            raise ValueError(f"No boxes/ directory found in {archive_dir}")
                
            
    
    
    
    
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

    def iter_items(self) -> Iterator[tuple[int, "ProcessedObject"]]:
        for bn in sorted(self.boxes):
            yield bn, self.boxes[bn]

    def __len__(self) -> int:
        return self.num_box

    def __contains__(self, box_number: int) -> bool:
        return box_number in self.boxes

    def __getitem__(self, key: int | slice | list[int]) -> Union["ProcessedObject", list["ProcessedObject"]]:
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
