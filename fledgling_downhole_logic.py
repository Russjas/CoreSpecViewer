# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 10:06:20 2025

@author: russj
"""


from app.spectral_ops import spectral_functions as sf
from app.models.processed_object import ProcessedObject
from app.models.hole_object import HoleObject
import spectral as sp
import numpy as np

hole = HoleObject.build_from_parent_dir('D:/Clonminch_uncropped')



#%%
full_fractions = None    # will become (H_total, K+1)
full_dominant  = None 
for po in hole:
    seg = sf.unwrap_from_stats(po.mask, po.datasets['MinMap-pearson-in rangeINDEX'].data, po.stats)
    fractions, dominant = sf.compute_downhole_mineral_fractions(seg.data, seg.mask, 
                                                             po.datasets['MinMap-pearson-in rangeLEGEND'].data)
    if full_fractions is None:
        # First box → just take it as-is
        full_fractions = fractions      # shape (H_box, K+1)
        full_dominant  = dominant       # shape (H_box,)
    else:
        # Append this box below the existing full arrays
        full_fractions = np.vstack((full_fractions, fractions))
        full_dominant  = np.concatenate((full_dominant, dominant))
    po.reload_all()
#%%
import matplotlib.pyplot as plt
import matplotlib
def plot_fullhole_mineral_stack(
    depths: np.ndarray,
    fractions: np.ndarray,     # (H, K+1) from compute_fullhole_mineral_fractions
    legend: list[dict],        # length K, same as used in compute_fullhole_mineral_fractions
    include_unclassified: bool = True,
    ax=None,
):
    """
    Vertical stacked mineral-fraction log.

    - Depth on Y axis (downwards).
    - Fractions stacked horizontally from 0 → 1.
    - Columns 0..K-1 in `fractions` correspond to legend entries (in order).
    - Column K is the 'unclassified' remainder (optional to plot).
    """
    depths = np.asarray(depths)
    frac = np.asarray(fractions)

    H, C = frac.shape
    K = len(legend)

    if C != K + 1:
        raise ValueError(f"fractions should have K+1 columns (got {C}, expected {K+1})")

    if depths.shape[0] != H:
        raise ValueError("depths and fractions must have the same number of rows")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 10))

    # Make sure depth increases downward visually
    if depths[0] > depths[-1]:
        depths = depths[::-1]
        frac = frac[::-1, :]

    # Which columns to plot
    cols_to_plot = list(range(K))  # legend classes
    if include_unclassified:
        cols_to_plot.append(K)      # last column = remainder

    # Build cumulative left/right bounds
    frac_use = frac[:, cols_to_plot]          # (H, M)
    cum = np.cumsum(frac_use, axis=1)        # (H, M)
    left = np.hstack([np.zeros((H, 1)), cum[:, :-1]])
    right = cum

    # Colour map: match _show_index_with_legend logic
    cmap = matplotlib.colormaps.get("tab20") or matplotlib.colormaps["tab10"]

    for band_idx, col_idx in enumerate(cols_to_plot):
        if col_idx < K:
            # Regular mineral class
            cid = int(legend[col_idx]["index"])   # library ID
            name = str(legend[col_idx]["label"])
            color = cmap(cid % 20)
        else:
            # Unclassified remainder
            cid = None
            name = "Unclassified"
            color = (0.3, 0.3, 0.3, 1.0)  # dark grey

        ax.fill_betweenx(
            depths,
            left[:, band_idx],
            right[:, band_idx],
            step="pre",
            facecolor=color,
            edgecolor="none",
            label=name,
        )

    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of row width")
    ax.set_ylabel("Depth")
    ax.grid(True, axis="x", alpha=0.2)

    # One legend entry per band, no sorting / reordering
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        handlelength=1.8,
        handletextpad=0.6,
    )

    plt.tight_layout()
    return ax
from app.interface import tools as t
fullhole_depths = None
for po in hole:
    t.unwrapped_output(po)
    depths = po.DholeDepths
    if fullhole_depths is None:
        # First box → just take it as-is
        fullhole_depths = depths      # shape (H_box, K+1)
    else:
        # Append this box below the existing full arrays
        fullhole_depths = np.concatenate((fullhole_depths, depths))
    po.commit_temps()
    po.save_all()   
    po.reload_all()


#%%
plot_fullhole_mineral_stack(
    fullhole_depths,
    full_fractions,
    hole[0].datasets['MinMap-pearson-in rangeLEGEND'].data
    )

#%%
np.save('D:/DownHole_TestData/a_clonminch_depths.npy', fullhole_depths)
np.save('D:/DownHole_TestData/a_clonminch_fractions.npy', full_fractions)
np.save('D:/DownHole_TestData/a_clonminch_dominant.npy', full_dominant)
