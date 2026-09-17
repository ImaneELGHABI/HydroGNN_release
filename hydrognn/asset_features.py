"""Rewrite control-structure attributes so their activation threshold is usable.

OPT-IN. Nothing calls this unless --relative_thresholds is passed.

The problem it solves
---------------------
A weir's crest and a pump's switching levels are already present in the asset
attribute vectors, but in a form the network cannot act on:

  * they are absolute datum elevations in FEET (Heusden's InfoWorks export),
  * while manhole features are z-scored (mean 0, std 1),
  * and patch_invert_units divides the manhole invert without touching asset
    attributes, so after that patch the two are not even in the same unit.

To decide "is this weir spilling?" the model would have to denormalise the
manhole level back to metres, convert the crest from feet to metres, and
subtract -- across a node-type boundary, through message passing, with no
mechanism that performs arithmetic between two differently scaled quantities.
The threshold is present and unreachable, which is a sufficient explanation for
why weir- and pump-adjacent manholes are the worst-predicted group.

What it writes instead
----------------------
The activation threshold expressed as a DEPTH ABOVE THE UPSTREAM MANHOLE'S OWN
INVERT, in metres, optionally normalised by that manhole's chamber depth:

    weir : [ (crest - invert_us)/scale , width , C_d , height ]
    pump : [ Q_rated , (on - invert_us)/scale , (off - invert_us)/scale , ... ]

Now "this weir spills once the upstream manhole is 1.2 m deep" is a single
number on the same footing as the depth the model predicts, it is datum-free,
and it transfers between the ICM and SWMM networks unchanged.

NO LEAKAGE. Every quantity used here is static topology -- crest, switching
levels, invert elevation, chamber geometry. None of it is derived from
`manhole.y`, from any snapshot, or from the test split. The transform is a pure
function of the network's fixed geometry and is identical for every snapshot and
every fold.

ORDERING. Must run BEFORE patch_invert_units, because it relies on the crest and
the invert still being in the same raw units. Applying it afterwards would
subtract metres from feet; the caller is checked for this.
"""
from __future__ import annotations

from typing import List, Optional

import torch

FEET_PER_METRE = 3.280839895013123

WEIR_CREST, WEIR_WIDTH, WEIR_CD, WEIR_HEIGHT = 0, 1, 2, 3
PUMP_Q, PUMP_ON, PUMP_OFF = 0, 1, 2


def _us_invert(g, group: str) -> Optional[torch.Tensor]:
    """Invert elevation of each asset's upstream manhole, in asset-node order."""
    ei_us = g.edge_index_dict.get(("manhole", "has_us", group)) \
        if hasattr(g, "edge_index_dict") else None
    if ei_us is None:
        if ("manhole", "has_us", group) not in g.edge_types:
            return None
        ei_us = g[("manhole", "has_us", group)].edge_index
    inv = getattr(g["manhole"], "invert_level_m", None)
    if inv is None:
        return None
    inv = inv.view(-1)
    n_assets = g[group].x.size(0)
    out = torch.zeros(n_assets, dtype=inv.dtype)
    order = ei_us[1].argsort()
    out[ei_us[1][order]] = inv[ei_us[0][order]]
    return out


def relativise_asset_thresholds(graphs: List, elev_scale: float = FEET_PER_METRE,
                                chamber_depth_m: Optional[torch.Tensor] = None,
                                verbose: bool = True) -> dict:
    """Rewrite weir/pump thresholds in place, relative to the upstream invert.

    `elev_scale` converts the raw elevation unit to metres (3.2808 for the
    Heusden feet export, 1.0 for the metric SWMM networks).
    `chamber_depth_m` optionally normalises by each manhole's own depth, giving a
    dimensionless "fraction of chamber filled before this control activates".

    Returns a summary of what was rewritten, for logging.
    """
    if not graphs:
        return {}
    g0 = graphs[0]
    info: dict = {}

    for group in ("weir", "pump"):
        if group not in g0.node_types:
            continue
        us_inv = _us_invert(g0, group)
        if us_inv is None:
            continue
        cd = None
        if chamber_depth_m is not None:
            ei_us = g0[("manhole", "has_us", group)].edge_index
            cd = torch.zeros(g0[group].x.size(0), dtype=us_inv.dtype)
            order = ei_us[1].argsort()
            cd[ei_us[1][order]] = chamber_depth_m.view(-1)[ei_us[0][order]]
            cd = cd.clamp(min=0.3)

        cols = [WEIR_CREST] if group == "weir" else [PUMP_ON, PUMP_OFF]
        before = {}
        for g in graphs:
            x = g[group].x
            if x.size(1) <= max(cols):
                continue
            x = x.clone()
            for c in cols:
                if g is g0:
                    before[c] = (float(x[:, c].min()), float(x[:, c].max()))
                rel = (x[:, c] - us_inv) / elev_scale
                x[:, c] = rel / cd if cd is not None else rel
            g[group].x = x
        if before:
            x0 = g0[group].x
            info[group] = {
                "cols": cols,
                "before": before,
                "after": {c: (float(x0[:, c].min()), float(x0[:, c].max())) for c in cols},
            }

    if verbose and info:
        for grp, d in info.items():
            for c in d["cols"]:
                b, a = d["before"][c], d["after"][c]
                print(f"    {grp} col{c}: [{b[0]:.2f}, {b[1]:.2f}] raw "
                      f"-> [{a[0]:.2f}, {a[1]:.2f}] relative to upstream invert"
                      + (" (chamber-normalised)" if chamber_depth_m is not None else " (m)"))
    return info


def chamber_depth_from_topology(topology_csv, num_manholes: int,
                                elev_scale: float = FEET_PER_METRE
                                ) -> Optional[torch.Tensor]:
    """Ground level minus invert per manhole, in metres. Static geometry only."""
    import pandas as pd
    import numpy as np
    from pathlib import Path
    p = Path(topology_csv)
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="latin-1")
    tcol = next((c for c in ("node_type", "type") if c in df.columns), None)
    rows = (df[df[tcol].astype(str).str.lower() == "manhole"].reset_index(drop=True)
            if tcol else df.reset_index(drop=True))
    if len(rows) != num_manholes or "ground_level" not in rows.columns \
            or "chamber_floor" not in rows.columns:
        return None
    d = (rows["ground_level"].astype(float).values
         - rows["chamber_floor"].astype(float).values) / elev_scale
    if not np.isfinite(d).all() or (d <= 0).mean() > 0.01:
        return None
    return torch.tensor(d, dtype=torch.float32)
