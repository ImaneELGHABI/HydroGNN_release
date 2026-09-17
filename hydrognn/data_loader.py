"""HydroGNN graph builder — lines-as-nodes transform.

Converts a HeteroData graph where hydraulic assets (pipes, weirs, pumps)
are represented as manhole-to-manhole edges into a graph where each asset
becomes a dedicated node:

    (manhole_us) ──has_us──> (asset_node) ──has_ds──> (manhole_ds)

This allows the GNN to learn asset-specific representations and apply
type-pair message passing across heterogeneous asset types.

Asset categories (grouped by hydraulic behaviour):
    pipe = conduit | channel                     [free-flow conveyance]
    weir = weir | orifice | flap_valve           [threshold control]
    pump = pump                                  [active control]

Each asset node feature vector consists of the original 4-dim asset attributes
plus a one-hot encoding of the specific sub-type.

Usage:
    from hydrognn.data_loader import lines_as_nodes

    g = lines_as_nodes(g_raw)    # one snapshot
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData


# ── asset → node-type-group + sub-type one-hot --------------------------------
# Order of one-hots matches the column index used in the feature padding.
PIPE_SUBTYPES = ["conduit", "channel"]
WEIR_SUBTYPES = ["weir", "orifice", "flap_valve"]
PUMP_SUBTYPES = ["pump"]

ASSET_GROUP = {
    "conduit":     ("pipe",  PIPE_SUBTYPES.index("conduit")),
    "channel":     ("pipe",  PIPE_SUBTYPES.index("channel")),
    "weir":        ("weir",  WEIR_SUBTYPES.index("weir")),
    "orifice":     ("weir",  WEIR_SUBTYPES.index("orifice")),
    "flap_valve":  ("weir",  WEIR_SUBTYPES.index("flap_valve")),
    "pump":        ("pump",  PUMP_SUBTYPES.index("pump")),
}

ASSET_GROUP_NUM_SUBTYPES = {"pipe": len(PIPE_SUBTYPES),
                            "weir": len(WEIR_SUBTYPES),
                            "pump": len(PUMP_SUBTYPES)}


def compute_qmax_for_pipe_edges(edge_attr: torch.Tensor) -> torch.Tensor:
    """Append log1p(Q_max) as a new edge feature for pipe edges.

    Manning's full-pipe discharge for circular conduits:
        Q_max = (1/n) · A · R^(2/3) · S^(1/2)
        A = π·D²/4   R = D/4   (circular pipe flowing full)

    Inputs (edge_attr columns):
        [0] width       — diameter D (raw units, model only needs relative scale)
        [1] length_norm — normalised pipe length (unused here)
        [2] roughness   — Manning's n
        [3] gradient    — pipe slope S

    log1p compresses Q_max dynamic range (small laterals → trunk sewers
    can span 3+ orders of magnitude).
    """
    width     = edge_attr[:, 0].clamp(min=1e-3)
    roughness = edge_attr[:, 2].clamp(min=1e-3)
    gradient  = edge_attr[:, 3].clamp(min=0.0)            # ignore reverse slopes

    A = torch.pi * width.pow(2) / 4.0
    R = width / 4.0
    Q_max = (1.0 / roughness) * A * R.pow(2.0 / 3.0) * gradient.pow(0.5)
    log_q = torch.log1p(Q_max).unsqueeze(1)
    return torch.cat([edge_attr, log_q], dim=1)


def _asset_node_features(edge_attrs: torch.Tensor, group: str, subtype_idx: int) -> torch.Tensor:
    """Build the asset node feature matrix.

    Each row = [<original asset feature vector>, <one-hot of sub-type>].
    Group fixes the one-hot width: pipe→2, weir→3, pump→1.
    NaN in edge_attr (e.g. missing topology fields) is replaced with 0.
    """
    if torch.isnan(edge_attrs).any():
        edge_attrs = edge_attrs.clone()
        torch.nan_to_num_(edge_attrs, nan=0.0)
    n = edge_attrs.size(0)
    width = ASSET_GROUP_NUM_SUBTYPES[group]
    one_hot = torch.zeros(n, width, dtype=edge_attrs.dtype)
    one_hot[:, subtype_idx] = 1.0
    return torch.cat([edge_attrs, one_hot], dim=1)


def lines_as_nodes(g: HeteroData, add_qmax_feature: bool = False,
                   add_manhole_shortcut: bool = False,
                   normalize_assets: bool = False) -> HeteroData:
    """Return a fresh HeteroData with hydraulic assets hoisted from edges to nodes.

    Manhole / outfall / storage nodes are copied verbatim. Manhole-manhole asset
    edges are removed; for each one we add a new asset-type node and two new
    edge types `(manhole, has_us, asset)` + `(asset, has_ds, manhole)` plus
    their reverses.

    add_qmax_feature: append log1p(Q_max) (Manning full-pipe capacity) to pipe
    node features → pipe feat dim 7 instead of 6.

    add_manhole_shortcut: additionally emit ("manhole", "adjacent", "manhole")
    edges joining the two endpoints of every hoisted asset, in both directions.
    The asset nodes and their has_us/has_ds edges are untouched — this only adds
    a parallel path.

    normalize_assets: z-normalise each asset group's hydraulic columns (the
    one-hot subtype tail is left alone). Asset attributes are static topology --
    identical in every snapshot -- so scaling them by their own column statistics
    is deterministic feature conditioning, not a train/test statistic.

    Why it matters: manhole features are z-normalised by the preprocessor but
    edge_attr never was, so asset node features arrive on wildly different
    scales. On Heusden fold 0 the pipe block reaches absmax 454.7 against 5.2 for
    the manhole block -- an 87x mismatch that suppresses exactly the hydraulic
    information the heterogeneous design exists to exploit.

    Why: without it there is no manhole→manhole edge at all, so a neighbour's
    reading reaches a node only after passing through an asset node that
    aggregates *both* of its endpoints. When the neighbour is a sensor and you
    are masked, what returns to you is a blend of its depth with your own zeroed
    features; you never see its value cleanly. Copying an adjacent sensor — the
    most useful single operation in sparse-sensor reconstruction — is therefore
    expensive to express, while emitting a smooth prior is cheap, and the model
    measurably takes the latter (error nearly independent of hop distance, 2-3x
    worse than GraphSAGE at 1 hop, collapse when the climatology feature is
    removed). The shortcut makes a one-hop copy directly representable and also
    doubles reach: L blocks span L manhole hops instead of L/2.
    """
    out = HeteroData()

    # -- pass-through node types (their features and meta)
    for nt in g.node_types:
        for k, v in g[nt].items():
            out[nt][k] = v

    # -- collect asset edges by group, building per-group node features as we go
    grouped_features: Dict[str, List[torch.Tensor]] = {"pipe": [], "weir": [], "pump": []}
    grouped_us_idx:   Dict[str, List[torch.Tensor]] = {"pipe": [], "weir": [], "pump": []}
    grouped_ds_idx:   Dict[str, List[torch.Tensor]] = {"pipe": [], "weir": [], "pump": []}

    # Cross-type edges (subcatchment->manhole, storage<->manhole, manhole->outfall, ...)
    # are passed through unchanged — they are already heterogeneous.
    cross_type_kept: List[Tuple[Tuple[str, str, str], torch.Tensor, torch.Tensor]] = []

    for et in list(g.edge_types):
        src_t, rel_t, dst_t = et
        if rel_t.endswith("_rev"):
            # Reverse edges are regenerated after asset hoisting — skip here.
            continue
        if src_t == "manhole" and dst_t == "manhole" and rel_t in ASSET_GROUP:
            group, subtype_idx = ASSET_GROUP[rel_t]
            ei = g[et].edge_index
            ea = g[et].edge_attr if "edge_attr" in g[et] else torch.zeros(ei.size(1), 4)
            if add_qmax_feature and group == "pipe":
                ea = compute_qmax_for_pipe_edges(torch.nan_to_num(ea, nan=0.0))
            grouped_features[group].append(_asset_node_features(ea, group, subtype_idx))
            grouped_us_idx[group].append(ei[0])
            grouped_ds_idx[group].append(ei[1])
        else:
            # Keep verbatim: subcatchment runoff, storage links, outfall links, etc.
            ea = g[et].edge_attr if "edge_attr" in g[et] else None
            cross_type_kept.append((et, g[et].edge_index, ea))

    # -- materialise asset nodes
    for group in ("pipe", "weir", "pump"):
        if not grouped_features[group]:
            continue
        feats = torch.cat(grouped_features[group], dim=0)
        us = torch.cat(grouped_us_idx[group], dim=0)
        ds = torch.cat(grouped_ds_idx[group], dim=0)
        n_assets = feats.size(0)
        asset_idx = torch.arange(n_assets, dtype=torch.long)

        out[group].x = feats

        # (manhole, has_us, asset) — upstream manhole feeds the asset node
        out["manhole", "has_us", group].edge_index = torch.stack([us, asset_idx], dim=0)
        # (asset, has_ds, manhole) — asset feeds the downstream manhole
        out[group, "has_ds", "manhole"].edge_index = torch.stack([asset_idx, ds], dim=0)
        # Reverse edges for the topology fix: ds-manhole→asset, asset→us-manhole
        out["manhole", "has_us_rev", group].edge_index = torch.stack([ds, asset_idx], dim=0)
        out[group, "has_ds_rev", "manhole"].edge_index = torch.stack([asset_idx, us], dim=0)

    # -- scale asset hydraulic columns onto the same footing as manhole features
    if normalize_assets:
        for group in ("pipe", "weir", "pump"):
            if group not in out.node_types:
                continue
            x = out[group].x
            n_hyd = x.size(1) - ASSET_GROUP_NUM_SUBTYPES[group]
            if n_hyd <= 0 or x.size(0) < 2:
                continue
            blk = x[:, :n_hyd]
            mu = blk.mean(0, keepdim=True)
            sd = blk.std(0, keepdim=True)
            # Constant columns (Manning n, efficiency, ...) carry no information;
            # centring sends them to exactly 0 rather than dividing by ~0.
            sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
            x[:, :n_hyd] = (blk - mu) / sd
            out[group].x = x

    # -- direct manhole-manhole shortcut alongside the asset path
    if add_manhole_shortcut:
        us_all, ds_all = [], []
        for group in ("pipe", "weir", "pump"):
            if grouped_us_idx[group]:
                us_all.append(torch.cat(grouped_us_idx[group], dim=0))
                ds_all.append(torch.cat(grouped_ds_idx[group], dim=0))
        if us_all:
            us_c, ds_c = torch.cat(us_all), torch.cat(ds_all)
            ei = torch.cat([torch.stack([us_c, ds_c]), torch.stack([ds_c, us_c])], dim=1)
            out["manhole", "adjacent", "manhole"].edge_index = torch.unique(ei, dim=1)

    # -- reattach cross-type edges (no asset hoisting for these)
    for et, ei, ea in cross_type_kept:
        out[et].edge_index = ei
        if ea is not None:
            out[et].edge_attr = ea

    return out


def strip_asset_physics(g: HeteroData) -> HeteroData:
    """Zero every hydraulic attribute in place, keeping asset identity and topology.

    Ablation counterpart to `lines_as_nodes`. Asset node features keep their
    one-hot subtype tail and lose the raw hydraulic head (diameter, length,
    Manning n, slope, log1p(Q_max) for pipes; crest height/width, C_d for
    weirs; design flow and on/off thresholds for pumps). Cross-type asset
    edges that were never hoisted keep their `edge_attr` slot but carry no
    hydraulic values.

    Feature dimensions are unchanged, so the ablated model instantiates
    exactly the parameter count of the full model: what is removed is the
    hydraulic information, not the capacity to use it.
    """
    for group in ("pipe", "weir", "pump"):
        if group not in g.node_types:
            continue
        x = g[group].x
        n_hydraulic = x.size(1) - ASSET_GROUP_NUM_SUBTYPES[group]
        if n_hydraulic > 0:
            x[:, :n_hydraulic] = 0.0

    # Asset edges between differing node types (manhole→outfall conduits,
    # storage→manhole pumps, ...) are passed through by lines_as_nodes with
    # their hydraulic edge_attr intact; they feed the model via W_e·e_ji.
    for et in g.edge_types:
        if et[1] not in ASSET_GROUP:
            continue
        ea = getattr(g[et], "edge_attr", None)
        if ea is not None:
            ea.zero_()

    return g


def asset_node_feature_dims(add_qmax_feature: bool = False) -> Dict[str, int]:
    """Return the feature dim of each asset node type, for sanity-check / model wiring."""
    # pipe/weir/pump: 4 raw asset features + one-hot; pipe gains log1p(Q_max) when enabled
    base = {"pipe": 5 if add_qmax_feature else 4, "weir": 4, "pump": 4}
    return {grp: base[grp] + ASSET_GROUP_NUM_SUBTYPES[grp] for grp in ("pipe", "weir", "pump")}


def asset_endpoint_indices(g: HeteroData) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Return (us_manhole_idx, ds_manhole_idx) for each asset group in node-index order."""
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for group in ("pipe", "weir", "pump"):
        if group not in g.node_types:
            continue
        ei_us = g["manhole", "has_us", group].edge_index   # [2, n_assets]
        ei_ds = g[group, "has_ds", "manhole"].edge_index   # [2, n_assets]
        order_us = ei_us[1].argsort()
        order_ds = ei_ds[0].argsort()
        out[group] = (ei_us[0][order_us], ei_ds[1][order_ds])
    return out


def convert_dataset(graphs: List[HeteroData]) -> List[HeteroData]:
    """Apply lines_as_nodes to every graph in a list."""
    return [lines_as_nodes(g) for g in graphs]


if __name__ == "__main__":
    # smoke test: python -m hydrognn.data_loader path/to/legacy.pt
    import sys
    from pathlib import Path

    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                Path(__file__).parent / "examples/example_dataset.pt"

    raw = torch.load(data_file, map_location="cpu", weights_only=False)
    graphs = raw.get("train_graphs") or raw.get("graphs", [])
    g_in = graphs[0]

    if "train_graphs" in raw:
        print("Applying lines-as-nodes transform ...")
        g_out = lines_as_nodes(g_in)
        print("  before:", list(g_in.node_types))
        print("  after :", list(g_out.node_types))
    else:
        print("Graph already in lines-as-nodes format.")
        g_out = g_in

    print("Edge types:")
    for et in g_out.edge_types:
        print(f"  {et}: {g_out[et].edge_index.size(1)} edges")
    for nt in g_out.node_types:
        if "x" in g_out[nt]:
            print(f"  {nt}: {g_out[nt].x.size(0)} nodes  feat_dim={g_out[nt].x.size(1)}")
