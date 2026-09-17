"""Figure 1: topology and sensor placement for the three study networks.

Regenerates fig_network_overview_3panel, whose archived version was drawn with
the 5% k-means sensor sets (its legend read "Sensor (5% coverage)") while the
caption described the 1% sets the manuscript reports.

Sensor locations are taken from the same eval_{net}_test_1pct.npz files that
feed the spatial-error figure, so the red markers here are exactly the nodes
the reported models were given, rather than a placement re-derived from the
sensor list. Node ordering follows load_coords: manhole rows of the topology
CSV in file order, which is the convention the masks index into.

Links are split into gravity assets (conduits, channels, weirs, orifices, flap
valves) and pump mains, the distinction that matters for reading the Heusden
panel: its pump mains span up to 20 km between otherwise isolated villages.

    python -m figures.fig1_networks
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from figures.common import TOPO, COND_DIR, MH_TYPE, PRED, load_coords, _save

NETS = ["tuindorp", "bellinge", "heusden"]
LABEL = {"tuindorp": "(a) Tuindorp", "bellinge": "(b) Bellinge", "heusden": "(c) Heusden"}

# node_type spellings differ between the SWMM and InfoWorks exports
OUTFALL = {"tuindorp": ("Outfall",), "bellinge": ("Outfall",), "heusden": ("Outfall",)}
STORAGE = {"tuindorp": (), "bellinge": ("StorageUnit",), "heusden": ("Storage",)}

C_MANHOLE = "#3B7DD8"
C_SENSOR = "#D62728"
C_OUTFALL = "#2E7D32"
C_STORAGE = "#F08C1F"
C_GRAVITY = "#555555"
C_PUMP = "#E06A10"

# panel widths follow each network's own extent so no map is stretched
WIDTH_RATIOS = [0.60, 2.15, 1.40]
NODE_S = {"tuindorp": 9, "bellinge": 6, "heusden": 2.4}
SENSOR_S = {"tuindorp": 46, "bellinge": 34, "heusden": 15}
EDGE_LW = {"tuindorp": 0.7, "bellinge": 0.55, "heusden": 0.35}
# Heusden carries 164 outfalls and 80 storage units against 6,197 manholes, so
# these markers are scaled down with the node size rather than held fixed;
# at a common size they cover the topology they are meant to annotate.
OUTFALL_S = {"tuindorp": 24, "bellinge": 18, "heusden": 6}
STORAGE_S = {"tuindorp": 22, "bellinge": 16, "heusden": 5}


def _id_to_pos(net):
    df = pd.read_csv(TOPO[net], encoding="latin-1")
    return {str(r["node_id"]): (float(r["x"]), float(r["y"]))
            for _, r in df.iterrows()
            if pd.notna(r.get("x")) and pd.notna(r.get("y"))}


def _typed_edges(net):
    """Return (gravity_segments, pump_segments) as lists of [(x0,y0),(x1,y1)]."""
    id2pos = _id_to_pos(net)
    grav, pump = [], []
    for lf in sorted(glob.glob(os.path.join(COND_DIR[net], "*.csv"))):
        base = os.path.basename(lf).lower()
        if "_node" in base:
            continue
        try:
            df = pd.read_csv(lf, encoding="latin-1")
        except Exception:
            continue
        if df.empty:
            continue
        us = next((c for c in df.columns if "us_node" in c.lower()
                   or c.lower() in ("upstream_node", "from_node", "us")), None)
        ds = next((c for c in df.columns if "ds_node" in c.lower()
                   or c.lower() in ("downstream_node", "to_node", "ds")), None)
        if us is None or ds is None:
            continue
        bucket = pump if "pump" in base else grav
        for _, row in df.iterrows():
            u, v = str(row[us]), str(row[ds])
            if u in id2pos and v in id2pos:
                bucket.append([id2pos[u], id2pos[v]])
    return list(map(list, dict.fromkeys(map(tuple, grav)))), \
        list(map(list, dict.fromkeys(map(tuple, pump))))


def _nodes_of_type(net, wanted):
    df = pd.read_csv(TOPO[net], encoding="latin-1")
    tc = next(c for c in df.columns if c.lower() in ("node_type", "type"))
    sel = df[df[tc].isin(wanted)]
    return sel["x"].values.astype(float), sel["y"].values.astype(float)


def _draw(ax, net):
    grav, pump = _typed_edges(net)
    if grav:
        ax.add_collection(LineCollection(grav, colors=C_GRAVITY,
                                         linewidths=EDGE_LW[net], zorder=1, alpha=0.75))
    if pump:
        ax.add_collection(LineCollection(pump, colors=C_PUMP, linewidths=1.0,
                                         linestyles="--", zorder=2, alpha=0.95))

    cx, cy = load_coords(net)
    d = np.load(PRED / f"{net}.npz", allow_pickle=True)
    mask = d["non_sensor"]          # True = masked (unmonitored)
    n = len(mask)
    cx, cy = cx[:n], cy[:n]
    sensor = ~mask

    ax.scatter(cx[mask], cy[mask], s=NODE_S[net], c=C_MANHOLE,
               linewidths=0, zorder=3)
    ax.scatter(cx[sensor], cy[sensor], s=SENSOR_S[net], c=C_SENSOR,
               linewidths=0.4, edgecolors="white", zorder=6)

    ox, oy = _nodes_of_type(net, OUTFALL[net])
    if len(ox):
        ax.scatter(ox, oy, s=OUTFALL_S[net], c=C_OUTFALL, marker="^",
                   linewidths=0, zorder=5, alpha=0.9)
    if STORAGE[net]:
        sx, sy = _nodes_of_type(net, STORAGE[net])
        if len(sx):
            ax.scatter(sx, sy, s=STORAGE_S[net], c=C_STORAGE, marker="s",
                       linewidths=0, zorder=5, alpha=0.9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    ax.set_title(LABEL[net], fontsize=10, pad=4)
    return int(sensor.sum()), n


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3),
                             gridspec_kw=dict(width_ratios=WIDTH_RATIOS, wspace=0.02))
    counts = {}
    for ax, net in zip(axes, NETS):
        counts[net] = _draw(ax, net)
        print(f"  {net}: {counts[net][0]} sensors / {counts[net][1]} manholes "
              f"({100.0 * counts[net][0] / counts[net][1]:.1f}%)")

    cov = 100.0 * counts["heusden"][0] / counts["heusden"][1]
    handles = [
        Line2D([], [], marker="o", ls="", color=C_MANHOLE, ms=5, label="Manhole"),
        Line2D([], [], marker="o", ls="", color=C_SENSOR, ms=7,
               label=f"Sensor ({cov:.0f}\\% coverage)".replace("\\%", "%")),
        Line2D([], [], marker="^", ls="", color=C_OUTFALL, ms=6, label="Outfall"),
        Line2D([], [], marker="s", ls="", color=C_STORAGE, ms=6, label="Storage unit"),
        Line2D([], [], color=C_GRAVITY, lw=1.2, label="Gravity link"),
        Line2D([], [], color=C_PUMP, lw=1.2, ls="--", label="Pump main"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.13)
    _save(fig, "fig_network_overview_3panel_v2")
    plt.close(fig)
    print("Saved fig_network_overview_3panel_v2")


if __name__ == "__main__":
    main()
