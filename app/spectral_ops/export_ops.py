"""
CSV Export Functions for Downhole Profile Data

Pure functions for writing profile datasets (1D downhole data) to CSV format.
These functions are format-agnostic - they don't know or care if data is
full (irregular) or stepped (regular grid).
"""

import csv
from pathlib import Path
import logging

import numpy as np

logger = logging.getLogger(__name__)

def export_profile_csv(
    mode: str,
    csv_path: Path,
    depths: np.ndarray,
    data: np.ndarray,
    legend: list[dict] | None = None,
    title: str = "",
    nodata: int = -999
) -> Path:
    """
    Dispatch to appropriate profile writer based on mode.
    
    Thin wrapper that routes to the correct CSV writer function.
    ------
     raises ValueError
        If mode is not recognized
    """
    if mode == "continuous":
        return write_continuous_profile(
            csv_path=csv_path,
            depths=depths,
            values=data,
            title=title,
            nodata=nodata
        )
    
    elif mode == "fractions":
        return write_fractions_profile(
            csv_path=csv_path,
            depths=depths,
            fractions=data,
            legend=legend,
            title=title,
            nodata=nodata
        )
    
    elif mode == "categorical":
        return write_categorical_profile(
            csv_path=csv_path,
            depths=depths,
            indices=data,
            legend=legend,
            title=title
        )
    
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Must be one of: 'continuous', 'fractions', 'categorical'"
        )

def write_continuous_profile(
    csv_path: Path,
    depths: np.ndarray,
    values: np.ndarray,
    title: str = "",
    nodata: int = -999
) -> Path:
    """
    Write continuous 1D profile data to CSV.
    
    Creates a two-column CSV with optional title row, depth and value columns.
    Handles masked arrays and NaN values by writing nodata value.
     
    Notes
    -----
    - Depth precision: 2 decimal places
    - Value precision: 6 decimal places
    - Masked/NaN values: written as nodata value (default -999)
    """
    if depths.ndim != 1:
        logger.error(f"Depths must be 1D array, got shape {depths.shape}")
        raise ValueError(f"Depths must be 1D array, got shape {depths.shape}")
    if values.ndim != 1:
        logger.error(f"Values must be 1D array, got shape {values.shape}")
        raise ValueError(f"Values must be 1D array, got shape {values.shape}")
    if len(depths) != len(values):
        logger.error(f"Length mismatch: depths has {len(depths)} samples, values has {len(values)} samples")
        raise ValueError(f"Length mismatch: depths has {len(depths)} samples, values has {len(values)} samples")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if title:
            writer.writerow([title])
        writer.writerow(['depth', 'value'])
        for d, v in zip(depths, values):
            depth_str = _format_depth(d)
            value_str = _format_value(v, nodata)
            writer.writerow([depth_str, value_str])
    
    logger.info(f"Exported continuous profile to .csv: {csv_path.name} ({len(depths)} samples)")
    return csv_path


def write_fractions_profile(
    csv_path: Path,
    depths: np.ndarray,
    fractions: np.ndarray,
    legend: list[dict] | None = None,
    title: str = "",
    nodata: int = -999
) -> Path:
    """
    Write mineral fractions profile data to CSV.
    
    Creates multi-column CSV with optional title row, depth and fraction columns 
    for each mineral. The last column is assumed to be "unclassified" fraction.
    
    Notes
    -----
    - Fractions columns correspond to legend entries in list order
    - Legend entry with missing "name" key gets "unnamed mineral"
    - Masked/NaN fractions: written as nodata value (default -999)
    """
    if depths.ndim != 1:
        logger.error(f"Depths must be 1D array, got shape {depths.shape}")
        raise ValueError(f"Depths must be 1D array, got shape {depths.shape}")
    if fractions.ndim != 2:
        logger.error(f"Fractions must be 2D array, got shape {fractions.shape}")
        raise ValueError(f"Fractions must be 2D array, got shape {fractions.shape}")
    if len(depths) != fractions.shape[0]:
        logger.error(f"Length mismatch: depths has {len(depths)} samples, fractions has {fractions.shape[0]} samples")
        raise ValueError(f"Length mismatch: depths has {len(depths)} samples, fractions has {fractions.shape[0]} samples")
    
    num_columns = fractions.shape[1]
    
    # Build header row
    headers = ['depth']
    
    if legend:
        # Legend entries correspond to columns 0..K-1 in order
        # Column K (last) is unclassified
        for entry in legend:
            mineral_name = entry.get("label", "unnamed mineral")
            headers.append(mineral_name)
        
        # Last column is unclassified
        headers.append("unclassified")
    
    else:
        for i in range(num_columns - 1):
            headers.append(f"mineral_{i}")
        headers.append("unclassified")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if title:
            writer.writerow([title])
        writer.writerow(headers)
        for d, frac_row in zip(depths, fractions):
            row = [_format_depth(d)]
            for frac in frac_row:
                row.append(_format_value(frac, nodata))
            writer.writerow(row)
    
    logger.info(f"Wrote fractions profile CSV: {csv_path.name} ({len(depths)} samples X {num_columns} columns)")
    return csv_path


def write_categorical_profile(
    csv_path: Path,
    depths: np.ndarray,
    indices: np.ndarray,
    legend: list[dict] | None = None,
    title: str = ""
    ) -> Path:
    """
    Write categorical profile data (mineral indices) to CSV.
    
    Creates CSV with optional title row, depth, index, and optionally mineral name columns.
    Indices of -1 represent "no dominant" or gaps.
    
    Notes
    -----
    - Index values are positions in the legend list, not the "index" field values
    - Index -1 is special: means no dominant mineral or data gap
    - Masked/NaN indices: converted to -1
    - Without legend: only depth and numeric index columns
    """
    if depths.ndim != 1:
        logger.error(f"Depths must be 1D array, got shape {depths.shape}")
        raise ValueError(f"Depths must be 1D array, got shape {depths.shape}")
    if indices.ndim != 1:
        logger.error(f"Indices must be 1D array, got shape {indices.shape}")
        raise ValueError(f"Indices must be 1D array, got shape {indices.shape}")
    if len(depths) != len(indices):
        logger.error(f"Length mismatch: depths has {len(depths)} samples, indices has {len(indices)} samples")
        raise ValueError(f"Length mismatch: depths has {len(depths)} samples, indices has {len(indices)} samples")

    headers = ['depth', 'index']
    if legend:
        headers.append('mineral_name')
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if title:
            writer.writerow([title])
        writer.writerow(headers)
        for d, idx in zip(depths, indices):
            row = [_format_depth(d)]
            
            # Handle masked/NaN/invalid indices
            if np.ma.is_masked(idx) or not np.isfinite(idx) or idx < 0:
                row.append("-1")
                if legend:
                    row.append('No dominant')
            else:
                idx_int = int(idx)
                row.append(str(idx_int))
                
                if legend:
                    # idx_int is position in legend list
                    if 0 <= idx_int < len(legend):
                        mineral_name = legend[idx_int].get("label", "unnamed mineral")
                    else:
                        mineral_name = f'Unknown (index {idx_int})'
                    row.append(mineral_name)
            
            writer.writerow(row)
    
    logger.info(f"Wrote categorical profile CSV: {csv_path.name} ({len(depths)} samples)")
    return csv_path


# ============================================================================
# FORMATTING 
# ============================================================================

def _format_depth(depth: float) -> str:
    """
    Format depth value for CSV output.
    Uses 2 decimal places (centimeter precision).
    """
    return f"{depth:.2f}"


def _format_value(value: float, nodata: int = -999) -> str:
    """
    Format a numeric value for CSV output.
    Handles masked arrays and NaN by returning nodata value.
    Otherwise formats to 4 decimal places.
    """
    # Check if masked (for masked arrays)
    if np.ma.is_masked(value):
        return str(nodata)
    if not np.isfinite(value):  # Catches both NaN and inf
        return str(nodata)
    return f"{value:.4f}"



















