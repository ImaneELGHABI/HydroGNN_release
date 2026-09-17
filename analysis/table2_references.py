"""Table 2, climatology and nearest-sensor rows.

Climatology predicts each manhole's training-split mean. Nearest-sensor copies
the current reading of the sensor reached first by a breadth-first search over
manhole adjacency; manholes with no path to any sensor are left out of that row.

    python -m analysis.table2_references
"""
import gc
import json

import numpy as np
import torch

from analysis.common import (NETWORKS, RESULTS, diagnostics, manhole_adjacency,
                             nearest_sensor_map, revert_units_si, sensor_file,
                             targets_cm)
from hydrognn.train import build_fixed_mask, load_sensor_indices

out = {}
for net, cfg in NETWORKS.items():
    clim, near, unreached = [], [], []
    for k in cfg["folds"]:
        raw = torch.load(cfg["data"](k), map_location="cpu", weights_only=False)
        if net == "heusden":
            revert_units_si(raw["graphs"])
        n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
        n_te = int(raw.get("test_size", len(raw["graphs"]) - n_tr - n_val))
        dmean, dstd = float(raw["depth_mean"]), float(raw["depth_std"])
        g = raw["graphs"]
        n = g[0]["manhole"].x.size(0)

        true = targets_cm(g[n_tr + n_val: n_tr + n_val + n_te], dmean, dstd)
        clim_pred = np.broadcast_to(targets_cm(g[:n_tr], dmean, dstd).mean(0), true.shape)
        sensors = [i for i in load_sensor_indices(sensor_file(net)) if i < n]
        mask = build_fixed_mask(n, sensors).numpy()
        owner = nearest_sensor_map(manhole_adjacency(g[0], n), sensors, n)
        ok = owner >= 0
        near_pred = np.zeros_like(true)
        near_pred[:, ok] = true[:, owner[ok]]

        clim.append(diagnostics(clim_pred, true, mask))
        near.append(diagnostics(near_pred, true, mask & ok))
        unreached.append(int((mask & ~ok).sum()))
        del raw, g, true, clim_pred, near_pred
        gc.collect()

    mean = lambda rows: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
    out[net] = dict(climatology=mean(clim), nearest_sensor=mean(near),
                    nearest_sensor_unreached=unreached)
    for name in ("climatology", "nearest_sensor"):
        r = out[net][name]
        print(f"{net:9s} {name:15s} MAE {r['mae']:6.2f}  wet {r['wet_mae']:6.2f}  "
              f"RMSE {r['rmse']:6.2f}  DynR {r['dyn_ratio']:.2f}  rho {r['rho']:.2f}")
    print(f"{'':9s} masked manholes with no path to a sensor: {unreached[0]}")

RESULTS.mkdir(exist_ok=True)
(RESULTS / "table2_references.json").write_text(json.dumps(out, indent=1))
