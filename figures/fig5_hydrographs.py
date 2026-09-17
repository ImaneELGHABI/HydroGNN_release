"""Figure 5: observed and reconstructed hydrographs on the held-out test data.

For each panel the plotted manhole is the unmonitored one with the largest
depth range in the window among those with below-median error.

The storm window is centred on the peak network-mean depth; the low-flow window
is the lowest-mean window of the same length that does not overlap it. Heusden
test folds contain only design storms, so its low-flow row is a quiet period
within an event. Heusden uses fold 0's block of the concatenated predictions.

    python -m figures.fig5_hydrographs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures.common import C_GT, C_MODEL, PRED, REPO, _save

# sampling interval per network, minutes between consecutive records
DT_MIN = {"bellinge": 5, "tuindorp": 1, "heusden": 10}


def load_test(net: str):
    """Return (pred_cm, true_cm, non_sensor) from the test npz."""
    d = np.load(PRED / f"{net}.npz", allow_pickle=True)
    return d["preds"], d["trues"], d["non_sensor"]


def heusden_fold0_len() -> int:
    """Records in fold 0's test block, from the run's own metrics file."""
    f = REPO / "outputs/hydrognn/heusden_1pct_fold0/test_metrics.json"
    return int(json.loads(f.read_text())["n_test_snapshots"])


def event_blocks(true_cm: np.ndarray, mask: np.ndarray,
                 min_len: int = 6) -> list[tuple[int, int]]:
    """Split a contiguous record sequence into per-event blocks.

    A new simulation starts where the network-mean depth drops sharply; that is
    the reset between one event and the next.
    """
    means = true_cm[:, mask].mean(axis=1)
    drop = np.diff(means)
    thr = -max(3.0, float(np.std(drop)) * 3.0)
    bnds = np.concatenate([[0], np.where(drop < thr)[0] + 1, [len(means)]])
    return [(int(a), int(b)) for a, b in zip(bnds[:-1], bnds[1:])
            if b - a >= min_len]


# window length per network, in records: long enough to show a hydrograph
# shape, short enough to read on a two-column page
WIN = {"bellinge": 72, "tuindorp": 120, "heusden": 72}


def storm_window(true_cm, mask, net):
    """Fixed-length window centred on the global peak masked depth."""
    w = WIN[net]
    series = true_cm[:, mask].mean(axis=1)
    peak = int(np.argmax(series))
    t0 = max(0, min(peak - w // 2, len(series) - w))
    return t0, min(t0 + w, len(series))


def lowflow_window(true_cm, mask, net, avoid):
    """Lowest-mean window of the same length, not overlapping `avoid`.

    Heusden's held-out folds contain only design storms -- the dry-weather
    series is in the training split -- so for that network this is the quiescent
    portion of an event rather than a dry-weather period. The row is labelled
    "low flow" accordingly.
    """
    w = WIN[net]
    series = true_cm[:, mask].mean(axis=1)
    T = len(series)
    a0, a1 = avoid
    best, best_mean = None, np.inf
    for t0 in range(0, T - w + 1, max(1, w // 4)):
        t1 = t0 + w
        if t0 < a1 and a0 < t1:          # overlaps the storm window
            continue
        m = float(series[t0:t1].mean())
        if m < best_mean:
            best_mean, best = m, (t0, t1)
    return best if best is not None else (0, min(w, T))


def pick_node(true_w, pred_w, mask) -> int:
    """Most dynamic masked node among those reconstructed better than median."""
    idx = np.where(mask)[0]
    err = np.abs(pred_w[:, idx] - true_w[:, idx]).mean(0)
    rng = true_w[:, idx].max(0) - true_w[:, idx].min(0)
    ok = err <= np.median(err)
    if not ok.any():
        ok = np.ones_like(err, dtype=bool)
    cand = np.where(ok)[0]
    return int(idx[cand[np.argmax(rng[cand])]])


def panel(ax, net, pred, true, mask, window, kind, title=None,
          legend_loc="upper right"):
    t0, t1 = window
    win = slice(t0, t1)
    node = pick_node(true[win], pred[win], mask)
    t_x = np.arange(t1 - t0) * DT_MIN[net]

    gt, pr = true[win, node], pred[win, node]
    if net != "heusden":                 # depths above invert cannot be negative
        gt, pr = np.clip(gt, 0, None), np.clip(pr, 0, None)
    ax.plot(t_x, gt, color=C_GT, lw=1.8, label="Ground truth")
    ax.plot(t_x, pr, color=C_MODEL, lw=1.6, ls="--", label="HydroGNN")
    ax.set_xlabel("Time (min)", fontsize=8)
    ax.set_ylabel("Depth (cm)" if net != "heusden" else "Water level (cm)",
                  fontsize=8)
    ax.tick_params(labelsize=7.5)
    if title:
        ax.set_title(title, pad=4)
        ax.legend(loc=legend_loc, framealpha=0.9, fontsize=7)
    mae = float(np.abs(pr - gt).mean())
    print(f"  {net:9s} {kind:5s}: records {t0}-{t1}, node {node}, "
          f"range {gt.max() - gt.min():6.1f} cm, MAE {mae:5.2f} cm")


def main():
    fig = plt.figure(figsize=(10.5, 4.2))
    gs = fig.add_gridspec(2, 3, hspace=0.60, wspace=0.42)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    cols = [("bellinge", "(a) Bellinge", "upper right"),
            ("tuindorp", "(b) Tuindorp", "upper right"),
            ("heusden",  "(c) Heusden",  "upper left")]

    for col, (net, title, loc) in enumerate(cols):
        pred, true, mask = load_test(net)
        if net == "heusden":
            k = heusden_fold0_len()      # fold 0's contiguous test block
            pred, true = pred[:k], true[:k]
        w_storm = storm_window(true, mask, net)
        w_low = lowflow_window(true, mask, net, w_storm)
        for row, (kind, window) in enumerate([("low", w_low),
                                              ("storm", w_storm)]):
            panel(axes[row][col], net, pred, true, mask, window, kind,
                  title=title if row == 0 else None, legend_loc=loc)

    axes[0][0].set_ylabel("Low flow\nDepth (cm)", fontsize=8)
    axes[1][0].set_ylabel("Storm peak\nDepth (cm)", fontsize=8)
    fig.suptitle("Water level reconstruction at unmonitored manholes — "
                 "held-out test records", fontsize=9, y=1.01)
    _save(fig, "fig_hydrographs_corrected")


if __name__ == "__main__":
    main()
