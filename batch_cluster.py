"""
Lightweight batch clustering utility.

Steps:
1) Load a JSON dataset (expects either a top-level list/array or dict with
   a "data" key).
2) Run a quick MiniBatchKMeans clustering with sensible defaults.
3) Save and display a colorised cluster map as an image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-GUI backend for headless use
import matplotlib.pyplot as plt
import numpy as np
import logging

# Use the project’s existing clustering helper
from app.spectral_ops.spectral_functions import kmeans_spectral_wrapper

# simple logger
logger = logging.getLogger("batch_cluster")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _load_array(path: Path) -> np.ndarray:
    """
    Load numeric data from .npy, .npz or .json.
    """
    logger.info(f"Loading data from {path}")
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            # Prefer "data" key, otherwise first array
            if "data" in data:
                arr = data["data"]
            else:
                first_key = next(iter(data.files))
                arr = data[first_key]
        else:
            arr = data
        return np.asarray(arr, dtype=float)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict) and "data" in obj:
            return np.asarray(obj["data"], dtype=float)

        if isinstance(obj, (list, tuple)):
            arr = np.asarray(obj, dtype=float)
            if arr.ndim not in (2, 3):
                raise ValueError(
                    f"Unsupported data shape {arr.shape}; expected 2D or 3D array."
                )
            return arr

        if isinstance(obj, dict):
            cols = []
            for v in obj.values():
                if isinstance(v, (list, tuple)):
                    try:
                        a = np.asarray(v, dtype=float)
                    except (TypeError, ValueError):
                        continue
                    if a.ndim == 1 and a.size > 1:
                        cols.append(a)
            if not cols:
                raise ValueError(f"No numeric arrays found in {path}")
            min_len = min(c.size for c in cols)
            stacked = np.stack([c[:min_len] for c in cols], axis=1)
            return stacked

        raise ValueError(f"Unsupported JSON structure in {path}")

    raise ValueError(f"Unsupported file type: {suffix}")


def _prepare_features(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Flatten the array for clustering and record a target image shape.

    Returns
    -------
    features : (N, C) float array
    img_shape : (H, W) tuple for reshaping cluster labels
    """
    if arr.ndim == 3:
        h, w, c = arr.shape
        logger.info(f"Input cube shape: {arr.shape}")
        features = arr.reshape(-1, c)
        img_shape = (h, w)
    elif arr.ndim == 2:
        h, w = arr.shape
        logger.info(f"Input image shape: {arr.shape}")
        features = arr.reshape(-1, 1)
        img_shape = (h, w)
    else:
        raise ValueError(f"Unsupported data shape {arr.shape}; expected 2D or 3D array.")

    return features, img_shape


def run_clustering(
    features: np.ndarray,
    img_shape: tuple[int, int],
    clusters: int,
    iters: int,
) -> np.ndarray:
    """
    Run clustering using the project’s wrapper around spectral.kmeans.

    The wrapper expects a cube shaped (H, W, Bands), so reshape features back.
    """
    h, w = img_shape
    bands = features.shape[1]
    logger.info(f"Running kmeans on shape (H={h}, W={w}, Bands={bands}), clusters={clusters}, iters={iters}")
    cube = features.reshape(h, w, bands)
    labels, _ = kmeans_spectral_wrapper(cube, clusters=clusters, iters=iters)
    return labels


def save_and_show(labels: np.ndarray, output: Path) -> None:
    """Save a cluster label image and display it."""
    # Count pixels per class
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(f"Class pixel counts: {class_counts}")

    plt.figure(figsize=(8, 4.5), dpi=120)
    plt.imshow(labels, cmap="tab20")
    plt.axis("off")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", pad_inches=0.05)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch cluster a JSON dataset and save a label image."
    )
    parser.add_argument("input", type=Path, help="Path to dataset file (.npy/.npz/.json).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cluster_output.png"),
        help="Output image path (PNG/JPG).",
    )
    parser.add_argument(
        "-k", "--clusters", type=int, default=5, help="Number of clusters."
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Max iterations."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_array(args.input)
    features, img_shape = _prepare_features(data)
    labels = run_clustering(
        features=features,
        img_shape=img_shape,
        clusters=args.clusters,
        iters=args.max_iter,
    )
    save_and_show(labels, args.output)
    logger.info(f"Saved cluster image to: {args.output.resolve()}")


if __name__ == "__main__":
    main()