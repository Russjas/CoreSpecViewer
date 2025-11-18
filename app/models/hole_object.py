"""
Created on Mon Nov 17 09:26:58 2025

@author: russj
"""
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Union

from .processed_object import ProcessedObject


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
