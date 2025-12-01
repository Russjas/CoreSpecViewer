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

import numpy as np

from ..spectral_ops import spectral_functions as sf
from ..spectral_ops import downhole_resampling as res
from .processed_object import ProcessedObject
from .dataset import Dataset


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

            print(key, ext, base)
            ds = Dataset(base=self.hole_id, key=key, path=fp, suffix=key, ext=ext)
            base = ["depths", "AvSpectra"]
            if key in base:
                self.base_datasets[key] = ds
            else:
                self.product_datasets[key] = ds
        return
    
    
    def create_base_datasets(self):
        """
        This function will only work if every po in hole has had unwrapped stats calculated.
        Returns HoleObject with unwrapped, concatenated downhole datasets.
        Does not account for missing boxes in this step
        """
        if not self.check_for_all_keys('stats'):
            raise ValueError("Missing 'stats' data for one or more boxes in the hole. Calculate stats before calling this method.")
        full_depths = None
        full_average = None
        try:
            for po in self:
                img = sf.unwrap_from_stats(po.mask, po.savgol, po.stats)
                depths = np.linspace(float(po.metadata['core depth start']), 
                                         float(po.metadata['core depth stop']),
                                         img.shape[0])
                if full_depths is None:
                    full_depths = depths      
                    full_average = np.ma.mean(img, axis=1)
                else:
                    full_depths = np.concatenate((full_depths, depths))
                    full_average = np.vstack((full_average, np.ma.mean(img, axis=1)))
                po.save_all()
                po.reload_all()
        except Exception as e:
            print(f'many, many things could have gone wrong - new code.{e}')
            return self
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
        print('called')
        print(key)
        if not self.check_for_all_keys(key):
            raise ValueError(f"{key} dataset is not available for every box in hole")
        if not (key.endswith("INDEX") or key.endswith("LEGEND")):
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
        self.product_datasets[fracs_key] = Dataset(base=self.hole_id, 
                                          key=fracs_key, 
                                          path=self.root_dir / f"{self.hole_id}_{fracs_key}.npy", 
                                          suffix=fracs_key, 
                                          ext=".npy", 
                                          data=full_fractions)
        self.product_datasets[dom_key] = Dataset(base=self.hole_id, 
                                          key=dom_key, 
                                          path=self.root_dir / f"{self.hole_id}_{dom_key}.npy", 
                                          suffix=dom_key, 
                                          ext=".npy", 
                                          data=full_dominant)
        self.product_datasets[leg_key] = Dataset(base=self.hole_id, 
                                          key=leg_key, 
                                          path=self.root_dir / f"{self.hole_id}_{leg_key}.json", 
                                          suffix=leg_key, 
                                          ext=".json", 
                                          data=legend)
        print("datasets successfully created!") 
        
    def create_dhole_features(self, key):
        """
        Returns HoleObject with unwrapped, concatenated feature datasets.
        Pos and dep datasets must be passed separately, no discovery is included.
        Datasets must me masked arrays.
        Does not account for missing boxes in this step
        """
        if not self.check_for_all_keys(key):
            raise ValueError(f"{key} dataset is not available for every box in hole")
        
        full_feature = None    # will become (H_total, K+1)
        for po in self:
            if po.datasets[key].ext != ".npz":
                raise ValueError(f"Box {po.metadata['box number']} {key} dataset is not a masked array.")
                
            seg = sf.unwrap_from_stats(po.datasets[key].data.mask, po.datasets[key].data.data, po.stats)
            feat_row = np.ma.mean(seg, axis=1)
            feat_row = np.ma.masked_less(feat_row, 1)
            if full_feature is None:
                # First box → just take it as-is
                full_feature = feat_row
                
            else:
                full_feature  = np.ma.concatenate((full_feature, feat_row))
            po.reload_all()
        
        self.product_datasets[key] = Dataset(base=self.hole_id, 
                                          key=key, 
                                          path=self.root_dir / f"{self.hole_id}_{key}", 
                                          suffix=key, 
                                          ext=".npz", 
                                          data=full_feature)


    def step_product_dataset(self, key):
        if key not in self.product_datasets.keys():
            raise ValueError("bounced no dataset")
        print(f"{key} passed successfully!")   
        if (key.endswith("FRACTIONS") or key.endswith("DOM-MIN")):
            print('on fractions line')
            if key.endswith("FRACTIONS"):
                dom_key = key.replace("FRACTIONS", "DOM-MIN")
                frac_key = key
            elif key.endswith("DOM-MIN"):
                frac_key = key
                dom_key = key.replace("DOM-MIN", "FRACTIONS")
            depths_stepped, fractions_stepped, dominant_stepped = res.resample_fractions_and_dominant_by_step(
                                                                self.base_datasets["depths"].data,
                                                                self.product_datasets[frac_key].data,
                                                                self.step)
            return depths_stepped, fractions_stepped, dominant_stepped
        else:
            print('on features line')
            depths_stepped, feature_stepped = res.bin_features_by_step(
                          self.base_datasets["depths"].data,
                          self.product_datasets[key].data,
                          self.step)
            return depths_stepped, feature_stepped, None
# =============================================================================
#TODO list

# 
#         Add plotting methods to the gui architecture
#         Wire up the GUI for downhole plots
# =============================================================================
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
            raise ValueError(f"Cannot extract 'borehole id' from metadata: {e}")
        return cls.build_from_parent_dir(obj.root_dir, hole_id)

    @classmethod
    def build_from_parent_dir(cls, path, hole_id: str = ""):
        root = Path(path)

        # ---- PASS 1: read JSON only (cheap) to detect dominant hole_id if not provided
        hole_ids: list[str] = []
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
