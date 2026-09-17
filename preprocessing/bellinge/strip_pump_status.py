"""Remove the per-timestep pump_status column (col 4) from pump edges.

Writes data/processed/bellinge_v3_nopumpstatus.pt, the file used in the paper.
Pump edge_dim goes from 5 to 4 (capacity, switch_on, switch_off, efficiency).
"""
import torch
from pathlib import Path

DATASETS = [
    ("bellinge", Path("data/processed/bellinge_v3.pt")),
]

for name, src in DATASETS:
    print(f"Processing {name} ...")
    d = torch.load(src, map_location="cpu", weights_only=False)

    pump_ets = [et for et in d["graphs"][0].edge_types if "pump" in et[1]]

    for g in d["graphs"]:
        for et in pump_ets:
            ea = g[et].edge_attr          # shape [E, 5]
            g[et].edge_attr = ea[:, :4]  # drop col 4 (pump_status)

    # Update edge_dims metadata
    for et in pump_ets:
        if et in d["edge_dims"]:
            d["edge_dims"][et] = 4

    dst = src.parent / (src.stem + "_nopumpstatus.pt")
    torch.save(d, dst)
    print(f"  Saved → {dst}")
    print(f"  pump edge types stripped: {pump_ets}")

print("Done.")
