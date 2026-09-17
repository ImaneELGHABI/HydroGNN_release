"""Generate a k-means sensor set for Heusden at a given coverage fraction.

Reproduces the placement used for the existing sets (1/2/5/10/20%): k-means on
manhole planar coordinates with k = round(coverage * n_manholes), then the
manhole closest to each cluster centroid becomes a sensor. Writes one 0-based
node index per line, sorted, matching the format of the existing files.

Coordinates are read from the fold graphs (manhole.x columns 1 and 2), which is
what the model itself sees, so placement cannot drift from the training data.

Usage:  python3 preprocessing/kmeans_sensors_heusden.py 0.01
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
FOLD = REPO / "data/processed/heusden_v3/folds/heusden_v3_fold0.pt"
OUT = REPO / "data/sensors"


def main(frac: float, seed: int = 42) -> None:
    raw = torch.load(FOLD, map_location="cpu", weights_only=False)
    x = raw["graphs"][0]["manhole"].x
    xy = x[:, 1:3].numpy()
    n = xy.shape[0]
    k = int(round(frac * n))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(xy)
    idx = sorted({int(np.argmin(((xy - c) ** 2).sum(1)))
                  for c in km.cluster_centers_})
    pct = int(round(frac * 100))
    dest = OUT / f"sensors_heusden_kmeans_{pct}pct.txt"
    dest.write_text("\n".join(str(i) for i in idx) + "\n")
    print(f"{len(idx)} sensors of {n} manholes ({100*len(idx)/n:.2f}%) -> {dest.name}")
    if len(idx) < k:
        print(f"  note: {k - len(idx)} centroids mapped to an already-chosen "
              f"manhole, so the set is slightly smaller than k={k}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.30)
