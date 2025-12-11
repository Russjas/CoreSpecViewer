## Batch Clustering Utility

Minimal headless tool that reuses the project k-means wrapper (`kmeans_spectral_wrapper`) to cluster arrays and save a label image.

### Supported inputs
- `.npy` or `.npz` numeric arrays (2D or 3D). 3D is treated as (H, W, Bands); 2D as (H, W).
- `.json` with either:
  - top-level `data` array,
  - top-level list/array,
  - dict of numeric lists (columns are stacked).

### Usage
```bash
python batch_cluster.py <input> -k 5 --max-iter 10 -o <output.png>
```
Example on processed data:
```bash
python batch_cluster.py data/134Mcrop/2025-10-31_09-01-32_white_circ_savgol.npy -k 5 --max-iter 10 -o data/134Mcrop/outputs/savgol_clusters.png
```

### What it does
1) Loads the array and flattens to features.
2) Reshapes back to (H, W, Bands) for `kmeans_spectral_wrapper`.
3) Runs k-means; logs shape, params, class pixel counts.
4) Saves a tab20-colored PNG and logs the output path.
