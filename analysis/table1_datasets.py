"""Table 1: network composition and split sizes.

Component counts come from the topology exports, record counts from the
processed datasets, sensor counts from data/sensors.

    python -m analysis.table1_datasets
"""
import glob
import os

import pandas as pd
import torch

from analysis.common import NETWORKS, TOPOLOGY
from hydrognn.train import load_sensor_indices
from analysis.common import sensor_file

for net, cfg in NETWORKS.items():
    topo = pd.read_csv(TOPOLOGY[net], encoding="latin-1")
    tc = next(c for c in topo.columns if c.lower() in ("node_type", "type"))
    print(f"\n{net}: node types {topo[tc].value_counts().to_dict()}")
    for f in sorted(glob.glob(os.path.join(os.path.dirname(TOPOLOGY[net]), "*.csv"))):
        if "_node" not in os.path.basename(f) and os.path.getsize(f) > 1:
            print(f"  {os.path.basename(f):60s} {len(pd.read_csv(f, encoding='latin-1')):6d} rows")
    raw = torch.load(cfg["data"](cfg["folds"][0]), map_location="cpu", weights_only=False)
    n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
    n_te = int(raw.get("test_size", len(raw["graphs"]) - n_tr - n_val))
    print(f"  records {n_tr + n_val + n_te}: train {n_tr}, val {n_val}, test {n_te}"
          + ("  (fold 0)" if net == "heusden" else ""))
    print(f"  sensors at 1%: {len(load_sensor_indices(sensor_file(net)))}")
    del raw
