"""Build one dataset per Heusden fold with train-only normalisation.

Input:  data/processed/heusden_v3/heusden_v3_5fold_event_isolated.pt
Output: data/processed/heusden_v3/folds/heusden_v3_fold{k}.pt

Graphs are ordered train | val | test with train_size / val_size / test_size,
the fold's storm assignment and per-graph provenance, as train.py expects
for --data_format spatial.

depth_mean / depth_std / static_mean / static_std are computed on the train
storms of fold k only and applied to its val and test graphs. This is asserted
per fold.

    python preprocessing/heusden/build_fold_datasets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "preprocessing/heusden"))

from features_v3 import (add_pump_status,                 # noqa: E402
                                   compute_stats, load_topology,
                                   normalise_graph)

IN_PATH = REPO / "data/processed/heusden_v3/heusden_v3_5fold_event_isolated.pt"
OUT_DIR = REPO / "data/processed/heusden_v3/folds"


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"missing {IN_PATH} — run scripts/build_heusden_event_folds.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading topology ...")
    mh_invert, st_invert, of_invert, mh_id2idx, st_id2idx, pumps = load_topology()
    mh_invert_t = torch.tensor(mh_invert, dtype=torch.float32)

    print(f"Loading {IN_PATH.name} ...")
    raw = torch.load(IN_PATH, map_location="cpu", weights_only=False)
    graphs = raw["graphs"]
    prov = raw["event_provenance"]
    folds = raw["fold_assignments"]
    fold_indices = raw["fold_indices"]
    print(f"  {len(graphs)} graphs, {len(folds)} folds")

    # Depths must be stored unmasked; train.py applies the sensor mask. A build
    # with pre-masked depths is mostly zeros (mu ~0.07 instead of ~1.71).
    probe = graphs[0]["manhole"].x[:, 5]
    frac_zero = float((probe == 0).float().mean())
    if frac_zero > 0.5:
        raise SystemExit(
            f"ABORT: depth column is {frac_zero:.1%} exact zeros — this build has "
            f"sensor masking baked in. Rebuild with sensor_indices covering every "
            f"manhole so masking stays train.py's job (--sensor_file).")
    y_mean = float(graphs[0]["manhole"].y.mean())
    if not (0.3 < y_mean < 10.0):
        raise SystemExit(f"ABORT: target mean {y_mean:.4f} m is outside the plausible "
                         f"range for Heusden depths — check the raw build.")
    print(f"Feature-content gate PASSED: depth column {frac_zero:.1%} zeros, "
          f"target mean {y_mean:.3f} m")

    print("Injecting pump status (pre-normalisation, topology-only) ...")
    graphs = [add_pump_status(g, mh_invert, st_invert, mh_id2idx, st_id2idx, pumps)
              for g in graphs]

    summary = {}
    stats_seen = {}
    for k in sorted(fold_indices):
        idx = fold_indices[k]
        tr_i, va_i, te_i = idx["train"], idx["val"], idx["test"]
        assert set(tr_i).isdisjoint(va_i) and set(tr_i).isdisjoint(te_i) \
            and set(va_i).isdisjoint(te_i), f"fold {k}: graph index overlap"

        # ---- normalisation statistics from THIS FOLD'S TRAIN GRAPHS ONLY ----
        train_raw = [graphs[i] for i in tr_i]
        d_mu, d_sd, s_mu, s_sd = compute_stats(train_raw)

        # The stats must be reproducible from the train subset alone, and must
        # not match what an all-storm pool would give.
        chk = compute_stats(train_raw)
        assert abs(chk[0] - d_mu) < 1e-12 and abs(chk[1] - d_sd) < 1e-12, \
            f"fold {k}: statistics not reproducible from train split alone"
        held_out = [graphs[i] for i in va_i + te_i]
        h_mu, h_sd, _, _ = compute_stats(held_out)
        assert (d_mu, d_sd) != (h_mu, h_sd), \
            f"fold {k}: train stats equal held-out stats — pooled normalisation"
        stats_seen[k] = (d_mu, d_sd)

        norm = lambda ids: [normalise_graph(graphs[i], d_mu, d_sd, s_mu, s_sd, mh_invert_t)
                            for i in ids]
        out_graphs = norm(tr_i) + norm(va_i) + norm(te_i)
        out_prov = [prov[i] for i in tr_i + va_i + te_i]

        g0 = out_graphs[0]
        node_dims = {nt: g0[nt].x.size(-1) for nt in g0.node_types if hasattr(g0[nt], "x")}
        edge_dims = {et: (g0[et].edge_attr.size(-1)
                          if getattr(g0[et], "edge_attr", None) is not None else 0)
                     for et in g0.edge_types}

        out_path = OUT_DIR / f"heusden_v3_fold{k}.pt"
        torch.save({
            "graphs": out_graphs,
            "train_size": len(tr_i), "val_size": len(va_i), "test_size": len(te_i),
            "node_dims": node_dims, "edge_dims": edge_dims,
            "edge_types": list(g0.edge_types),
            "depth_mean": d_mu, "depth_std": d_sd,
            "fold": k, "fold_assignment": folds[k],
            "event_provenance": out_prov,
            "normalisation_source": "train split of this fold only",
        }, out_path)

        summary[k] = {"train": len(tr_i), "val": len(va_i), "test": len(te_i),
                      "depth_mean": d_mu, "depth_std": d_sd,
                      "events": folds[k]}
        print(f"  fold {k}: train={len(tr_i):5d} val={len(va_i):5d} test={len(te_i):5d}  "
              f"mu={d_mu:.6f} sd={d_sd:.6f}  test storms={folds[k]['test_event_ids']}  "
              f"-> {out_path.name}")

    # Folds must not share identical statistics.
    assert len(set(stats_seen.values())) == len(stats_seen), \
        "two folds produced identical normalisation stats — suspect a global pool"
    print("\nTrain-only normalisation checks passed for all folds:")
    print("  - stats reproducible from each fold's train split alone")
    print("  - no fold's stats equal its held-out (val+test) stats")
    print("  - all 5 folds have distinct statistics")

    (REPO / "data/processed/heusden_v3/summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Written {REPO / 'data/processed/heusden_v3/summary.json'}")


if __name__ == "__main__":
    main()
