"""Figures 6 and 7: per-node test MAE maps.

Figure 6 pairs Bellinge and Tuindorp; Figure 7 is Heusden, pooled over the five
test folds. Colour scales stop at each network's 90th percentile.

    python -m figures.fig6_7_error_maps
"""
import numpy as np
import matplotlib.pyplot as plt

from figures.common import PRED, _draw_spatial_error_ax, _save

PANEL_LABEL = {"heusden": "(a)", "tuindorp": "(b)", "bellinge": "(c)"}
PRETTY = {"heusden": "Heusden", "tuindorp": "Tuindorp", "bellinge": "Bellinge"}


def _title(key):
    """Title carrying the counts actually present in the evaluation file."""
    d = np.load(PRED / f"{key}.npz", allow_pickle=True)
    mask = d["non_sensor"]
    n_nodes = len(d["mae_node_cm"])
    n_sens = int((~mask).sum())
    cov = 100.0 * n_sens / n_nodes
    return (f"{PANEL_LABEL[key]} {PRETTY[key]} — {n_nodes:,} nodes, "
            f"{n_sens} sensors ({cov:.0f}\\% coverage)".replace("\\%", "%"))


def pair():
    """Bellinge and Tuindorp side by side, Heusden separately.

    Tuindorp's extent is tall and narrow, so on its own in a full-width float
    it leaves both margins empty however it is scaled; paired with Bellinge the
    row is full. The two also belong together: both are SWMM networks predicting
    a depth on a single reserved test split, at centimetre scale, whereas
    Heusden predicts a water level over five folds and is plotted alone.
    Panel widths follow each map's own aspect so neither is stretched.
    """
    fig = plt.figure(figsize=(11.0, 4.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.64, 0.92], wspace=0.26)
    for i, key in enumerate(("bellinge", "tuindorp")):
        ax = fig.add_subplot(gs[0, i])
        _draw_spatial_error_ax(ax, key)
        lbl = "(a)" if i == 0 else "(b)"
        ax.set_title(f"{lbl} {_title(key).split(' ', 1)[1]}", pad=6, fontsize=10, loc="left")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.11)
    _save(fig, "fig_spatial_error_swmm_pair")
    plt.close(fig)
    print("  Saved fig_spatial_error_swmm_pair")


def heusden():
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 8.0))
    _draw_spatial_error_ax(ax, "heusden")
    ax.set_title(_title("heusden").split(" ", 1)[1], pad=6, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_spatial_error_heusden_1pct")


if __name__ == "__main__":
    pair()
    heusden()
