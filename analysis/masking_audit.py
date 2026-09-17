"""Section S5: static audit of the inputs handed to the model.

Uses the trainer's own loading, masking and context-feature code. At every
non-sensor manhole it checks that
  - every dynamic depth column is exactly zero,
  - the two per-node climatology columns are exactly zero,
and that the four network-wide statistics are identical at every manhole.

    python -m analysis.masking_audit [heusden bellinge tuindorp]
"""
import sys

import torch

from analysis.common import NETWORKS, sensor_file
from hydrognn.train import (add_sensor_context_features, apply_mask,
                            build_fixed_mask, compute_climatology, load_data,
                            load_sensor_indices)


def audit(net: str) -> bool:
    cfg = NETWORKS[net]
    k = cfg["folds"][0]
    train, val, *_, didx, sfi = load_data(cfg["data"](k), "spatial", add_qmax_feature=True)
    n = train[0]["manhole"].x.size(0)
    mask = build_fixed_mask(n, load_sensor_indices(sensor_file(net)))   # True = no sensor
    mu, sd = compute_climatology(train, sensor_mask=~mask)
    graphs = val + train[::10]
    fed = [add_sensor_context_features(apply_mask(g, mask, didx, sfi), mu, sd, didx[0],
                                       include_global=True) for g in graphs]
    X = torch.stack([g["manhole"].x for g in fed])          # [graphs, nodes, features]
    d = X.size(2)
    dyn = X[:, mask][:, :, didx].abs().max().item()
    clim = X[:, mask][:, :, d - 2:].abs().max().item()
    clim_at_sensors = X[:, ~mask][:, :, d - 2:].abs().max().item()
    glob = (X[:, :, d - 6:d - 2] - X[:, :1, d - 6:d - 2]).abs().max().item()
    flag_ok = bool((X[:, mask, sfi] == 0).all() and (X[:, ~mask, sfi] == 1).all())
    ok = dyn == 0 and clim == 0 and glob == 0 and flag_ok
    print(f"{net:9s} sensors {int((~mask).sum()):3d}  graphs {len(fed):5d}  "
          f"dynamic cols {didx}  max|dynamic| {dyn:.1e}  max|climatology| {clim:.1e} "
          f"(sensors {clim_at_sensors:.2f})  global spread {glob:.1e}  flag ok {flag_ok}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [audit(net) for net in (sys.argv[1:] or ["tuindorp", "bellinge", "heusden"])]
    sys.exit(0 if all(results) else 1)
