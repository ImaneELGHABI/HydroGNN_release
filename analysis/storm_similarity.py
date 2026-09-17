"""Section 3.1.2: similarity between Heusden design storms.

A storm's signature is its mean depth field at each of the 12 timesteps
(averaged over runs), minus the all-record mean field. Pairs are compared
by Pearson r over the full timestep x manhole signature.

    python -m analysis.storm_similarity
"""
import json
from itertools import combinations

import numpy as np
import torch

from analysis.common import DATA, RESULTS

raw = torch.load(DATA / "processed/heusden_v3/folds/heusden_v3_fold0.pt",
                 map_location="cpu", weights_only=False)
m, s = float(raw["depth_mean"]), float(raw["depth_std"])
Y = np.stack([g["manhole"].y.view(-1).numpy() for g in raw["graphs"]]) * s + m
bui = np.array([p["bui_id"] for p in raw["event_provenance"]])
ts = np.array([p["timestep"] for p in raw["event_provenance"]])

base = Y.mean(0)
sig = {b: np.stack([Y[(bui == b) & (ts == t)].mean(0) - base
                    for t in sorted(set(ts))]).ravel()
       for b in sorted(set(bui))}

pairs = sorted(((float(np.corrcoef(sig[a], sig[b])[0, 1]), a, b)
                for a, b in combinations(sorted(sig), 2)), reverse=True)
for r, a, b in pairs[:5]:
    print(f"bui#{a} vs bui#{b}: r = {r:.4f}")
RESULTS.mkdir(exist_ok=True)
(RESULTS / "storm_similarity.json").write_text(
    json.dumps([dict(a=a, b=b, r=r) for r, a, b in pairs], indent=1))
