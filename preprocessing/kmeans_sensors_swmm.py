"""k-means sensor sets at a given coverage for any of the three networks.

Same procedure as the existing Heusden sets: k-means on manhole planar
coordinates read from the dataset's own graphs (so placement cannot drift from
what the model sees), then the manhole nearest each centroid becomes a sensor.
Writes 0-based node indices, one per line, sorted.

Usage:  python3 preprocessing/kmeans_sensors_swmm.py bellinge 0.01
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/sensors"
DATA = {"bellinge": REPO / "data/processed/bellinge_v3_nopumpstatus.pt",
        "tuindorp": REPO / "data/processed/tuindorp_v2_rain.pt"}


def main(net: str, frac: float, seed: int = 42) -> None:
    raw = torch.load(DATA[net], map_location="cpu", weights_only=False)
    xy = raw["graphs"][0]["manhole"].x[:, 1:3].numpy()
    n = xy.shape[0]
    k = max(1, int(round(frac * n)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(xy)
    idx = sorted({int(np.argmin(((xy - c) ** 2).sum(1))) for c in km.cluster_centers_})
    pct = int(round(frac * 100))
    dest = OUT / f"sensors_{net}_kmeans_{pct}pct.txt"
    dest.write_text("\n".join(str(i) for i in idx) + "\n")
    print(f"{net}: {len(idx)} sensors of {n} manholes ({100*len(idx)/n:.2f}%) -> {dest.name}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
