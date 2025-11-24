"""
Legend Remapping and Class Aggregation for Spectral Mineral Maps
================================================================

This module provides utilities for post-classification legend cleanup (yes, it 
feels like cheating), collapsing raw spectral library matches into meaningful mineral groups that 
are geologically interpretable. The functionality is designed to operate 
*after* MSAM/SAM/Pearson/MinMap classification, using an ontology of 
mineral families and sub-families.

Why this exists
---------------
Direct spectral matching against large reference libraries (e.g. ECOSTRESS) 
often produces rare or overly specific mineral hits (e.g. Howlite, 
Rhodochrosite, Pb-carbonates), even when the underlying spectrum is 
representative of a common mineral family (e.g. Carbonate, White mica, 
Chlorite group). These results are mathematically valid, but not 
geologically useful.

This module resolves that by:
    • Mapping raw “best-match” labels into mineral families/sub-families
      (e.g. Calcite, Dolomite, Indistinct carbonate, White mica, Smectite)
    • Assigning consistent legend indices for display/export
    • Optionally keeping unmatched labels separate or merging them into a 
      single “Unclassified” class
    • Preserving reproducibility by avoiding hidden thresholds or priors

Inputs
------
    • index_array : (H, W) integer array
        Pixel-wise classification result from MinMap/SAM/MSAM/Pearson.
    • legend : list[dict]
        Full raw legend from classification, e.g.:
            [{'index': 0, 'label': 'Calcite CaCO3'}, ...]
    • ontology : dict
        Mineral family/sub-family definitions loaded from a JSON ontology.

Outputs
-------
    • remapped index array with contiguous new indices
    • clean legend using aggregated class labels
    • debug mapping describing which raw minerals contributed to each class

Design Principles
-----------------
    ✓ No black boxes — all rules are transparent and JSON-defined
    ✓ Keep exact mineral matches when geologically distinct (e.g. Calcite vs Dolomite)
    ✓ Collapse rare or ambiguous library entries into meaningful heads (e.g. 
      “Indistinct carbonate”, “Chlorite group”, “White mica”)
    ✓ Optional strict or relaxed handling of unmatched labels
    ✓ Pixel-mapping remains fully reversible for auditing/provenance

Typical Usage
-------------
    >>> new_idx, new_leg, dbg = remap_index_with_ontology(
            raw_index, raw_legend, ontology,
            keep_unmatched_as_original=True
        )

The result can be immediately displayed, exported, or used for further 
economic interpretation (e.g. carbonate–illite mixes, alteration zoning, etc.).

This module does **not** perform spectral matching itself; it only remaps 
classification results for clearer geological interpretation.
"""
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
import json

import numpy as np


def _classify_label_with_ontology(
    label: str,
    ontology: Dict[str, Any],
) -> Tuple[str | None, str | None]:
    """
    Return (family, class_name) for a legend label using the ontology,
    or (None, None) if no pattern matches.

    Matching rule:
      - label is lowercased
      - each ontology class has a list of 'patterns'
      - if any pattern is a substring of the label, it's a match
      - if multiple patterns match, the one with the LONGEST pattern wins
        (more specific).
    """
    text = label.lower()
    best_family = None
    best_class = None
    best_len = 0

    for family, fam_data in ontology.items():
        classes = fam_data.get("classes", {})
        for class_name, cls in classes.items():
            for patt in cls.get("patterns", []) or []:
                p = patt.lower().strip()
                if not p:
                    continue
                if p in text:
                    if len(p) > best_len:
                        best_len = len(p)
                        best_family = family
                        best_class = class_name

    if best_class is None:
        return None, None
    return best_family, best_class


def remap_index_with_ontology(
    index_array: np.ndarray,
    legend: List[Dict[str, Any]],
    ontology_path: str,
    keep_unmatched_as_original: bool = True,
    unknown_label: str = "Unclassified"
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Remap a pixel-wise index array and its legend using a sub-family ontology.

    Parameters
    ----------
    index_array : np.ndarray
        (H, W) integer array of class indices (e.g., minmap/MSAM output).
    legend : list of dict
        Legend entries, each like {'index': int, 'label': str}.
        'index' must match values present in index_array.
    ontology : dict
        Ontology loaded from JSON. Expected structure:
            {
              "Carbonate": {
                "classes": {
                  "Calcite": { "patterns": [...], "members": [...] },
                  "Dolomite": { ... },
                  "Indistinct carbonate": { ... }
                }
              },
              "Al-OH clay/mica": {
                "classes": {
                  "White mica": {...},
                  "Smectite": {...},
                  ...
                }
              },
              ...
            }
    keep_unmatched_as_original : bool, optional
        If True (default), any legend label that doesn't match the ontology
        is kept as its own new class with the original label.
        If False, unmatched labels are mapped to a single 'unknown_label' class.
    unknown_label : str, optional
        Label used when keep_unmatched_as_original=False and a label does not
        match any ontology pattern.

    Returns
    -------
    new_index : np.ndarray
        (H, W) integer array of remapped indices, with new indices in
        [0 .. N_new-1]. Any negative values in the input (e.g. -999) are
        preserved as-is.
    new_legend : list of dict
        New legend: [{'index': int, 'label': str}, ...] corresponding to
        the remapped indices in new_index.
    debug_map : dict
        Debugging information:
          {
            "old_index_to_new_label": { old_idx: new_label, ... },
            "new_label_to_members": {
                "Calcite": [ "Calcite CaCO3", "Calcite CaCO_3", ... ],
                ...
            }
          }
    """
    with open(ontology_path) as f:
        ontology = json.load(f)
    
    if not isinstance(ontology, dict):
        raise ValueError("Ontology did not load correctly")
        
    if index_array.ndim != 2:
        raise ValueError("index_array must be 2D (H, W)")

    # Build quick lookup from old index -> old label
    idx_to_label: Dict[int, str] = {}
    for entry in legend:
        idx = int(entry["index"])
        label = str(entry["label"])
        idx_to_label[idx] = label

    # Map each old index to a NEW label (class head) using the ontology
    old_index_to_new_label: Dict[int, str] = {}
    new_label_to_members: Dict[str, List[str]] = {}

    # If we want a single bucket for unmatched labels and we need it,
    # we'll add it lazily.
    unknown_bucket_created = False
    unknown_new_label = unknown_label

    for old_idx, old_label in sorted(idx_to_label.items()):
        family, cls = _classify_label_with_ontology(old_label, ontology)

        if cls is not None:
            # Use the ontology class name as the new legend label
            new_label = cls
        else:
            if keep_unmatched_as_original:
                # Keep the original legend label as its own class
                new_label = old_label
            else:
                # Map all unmatched to a single "unknown" class
                new_label = unknown_new_label
                unknown_bucket_created = True

        old_index_to_new_label[old_idx] = new_label
        new_label_to_members.setdefault(new_label, []).append(old_label)

    # Now assign contiguous new integer indices in order of first appearance
    label_to_new_idx: "OrderedDict[str, int]" = OrderedDict()
    for old_idx in sorted(idx_to_label.keys()):
        new_label = old_index_to_new_label[old_idx]
        if new_label not in label_to_new_idx:
            label_to_new_idx[new_label] = len(label_to_new_idx)

    # Build mapping array from old index -> new index.
    # We'll handle negative/invalid indices separately.
    if idx_to_label:
        max_old_idx = max(idx_to_label.keys())
    else:
        max_old_idx = -1

    # Start with -1 for everything
    index_map = np.full(max_old_idx + 1, -1, dtype=int)
    for old_idx, new_label in old_index_to_new_label.items():
        new_idx = label_to_new_idx[new_label]
        index_map[old_idx] = new_idx

    # Apply mapping to index_array,
    # but preserve negative indices (e.g. background, no-data).
    new_index = np.empty_like(index_array, dtype=int)

    mask_valid = index_array >= 0
    mask_invalid = ~mask_valid

    new_index[mask_invalid] = index_array[mask_invalid]
    if max_old_idx >= 0:
        new_index[mask_valid] = index_map[index_array[mask_valid]]
    else:
        # No valid legend entries; just copy array
        new_index[mask_valid] = index_array[mask_valid]

    # Build new legend list
    new_legend: List[Dict[str, Any]] = []
    for new_label, new_idx in label_to_new_idx.items():
        new_legend.append({"index": new_idx, "label": new_label})

    debug_map = {
        "old_index_to_new_label": old_index_to_new_label,
        "new_label_to_members": new_label_to_members,
    }

    return new_index, new_legend, debug_map



