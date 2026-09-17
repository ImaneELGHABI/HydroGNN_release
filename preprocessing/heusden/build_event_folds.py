"""Build the Heusden dataset from the raw depnod CSVs with event-isolated folds.

The storm id is taken from each file name (`bui#NN`) and whole storms are
assigned to splits, so no storm appears in more than one split of a fold.

Fold layout (10 storms, 2 test storms per fold):

    fold 0  test {01,02}  val {03}  train {04..10}
    fold 1  test {03,04}  val {05}  train {01,02,06..10}
    fold 2  test {05,06}  val {07}  train {01..04,08,09,10}
    fold 3  test {07,08}  val {09}  train {01..06,10}
    fold 4  test {09,10}  val {01}  train {02..08}

Test_3Events/ repeats 81 depnod files from the Run_C2100_* directories. Files
are keyed by basename so each simulation is counted once.

    python preprocessing/heusden/build_event_folds.py --partition-only   # write data/heusden_folds.json
    python preprocessing/heusden/build_event_folds.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "preprocessing/heusden"))

RAW_ROOT = REPO / "data/raw/heusden"
RAW_DIRS = [
    "Run_C2100_Gebeurtenis_DWA_20250422_003112",
    "Run_C2100_Reeks_DWA_20250422_000959",
    "Test_3Events",
]
TOPOLOGY_DIR = RAW_ROOT / "Detailed_heusden_topology"
OUT_PATH = REPO / "data/processed/heusden_v3/heusden_v3_5fold_event_isolated.pt"

EVENTS = [f"{i:02d}" for i in range(1, 11)]
TEST_FOLDS = [("01", "02"), ("03", "04"), ("05", "06"), ("07", "08"), ("09", "10")]

BUI_RE = re.compile(r"bui#(\d+)")
SCEN_RE = re.compile(r"(BRC_\d+_\d+)")


def inventory() -> dict[str, list[tuple[str, Path]]]:
    """Map storm id -> [(scenario_id, path)], de-duplicated by file basename."""
    seen: set[str] = set()
    inv: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for d in RAW_DIRS:
        for f in sorted((RAW_ROOT / d).glob("*depnod*.csv")):
            if f.name in seen:
                continue                      # Test_3Events duplicates the Run_* dirs
            seen.add(f.name)
            m, s = BUI_RE.search(f.name), SCEN_RE.search(f.name)
            if not m:
                continue
            inv[m.group(1)].append((s.group(1) if s else "unknown", f))
    return dict(inv)


def build_folds() -> dict[int, dict[str, list[str]]]:
    """Whole-storm fold assignment; no storm appears in two splits of a fold."""
    folds = {}
    for k, test in enumerate(TEST_FOLDS):
        rest = [e for e in EVENTS if e not in test]
        val = [rest[k % len(rest)]]
        train = [e for e in rest if e not in val]
        folds[k] = {"train_event_ids": train,
                    "val_event_ids": val,
                    "test_event_ids": list(test)}
    return folds


def assert_disjoint(folds) -> None:
    for k, f in folds.items():
        tr, va, te = (set(f["train_event_ids"]), set(f["val_event_ids"]),
                      set(f["test_event_ids"]))
        assert tr.isdisjoint(va), f"fold {k}: train/val overlap {tr & va}"
        assert tr.isdisjoint(te), f"fold {k}: train/test overlap {tr & te}"
        assert va.isdisjoint(te), f"fold {k}: val/test overlap {va & te}"
        assert tr | va | te == set(EVENTS), f"fold {k}: does not cover all 10 storms"
    all_test = [e for f in folds.values() for e in f["test_event_ids"]]
    assert sorted(all_test) == EVENTS, "test folds must partition the 10 storms exactly"
    assert len(all_test) == len(set(all_test)), "a storm is tested twice"


def report_partition(inv, folds) -> None:
    print(f"{'storm':>6} {'files':>6} {'scenarios':>10}")
    for e in EVENTS:
        scen = sorted({s for s, _ in inv.get(e, [])})
        print(f"  bui#{e:>2} {len(inv.get(e, [])):>6} {len(scen):>10}")
    print(f"\ntotal de-duplicated depnod files: {sum(len(v) for v in inv.values())}\n")
    print(f"{'fold':>4}  {'train storms':<34} {'val':<6} {'test':<10} {'train/val/test files'}")
    for k, f in folds.items():
        nf = lambda ids: sum(len(inv.get(e, [])) for e in ids)
        print(f"{k:>4}  {','.join(f['train_event_ids']):<34} "
              f"{','.join(f['val_event_ids']):<6} {','.join(f['test_event_ids']):<10} "
              f"{nf(f['train_event_ids'])}/{nf(f['val_event_ids'])}/{nf(f['test_event_ids'])}")


def build_dataset(inv, folds, timesteps_per_sim: int, sensor_coverage: float):
    """Build every snapshot once, tagged with provenance; folds index into it."""
    from spatial_loader import SpatialHeusdenLoader
    import pandas as pd
    import numpy as np

    loader = SpatialHeusdenLoader(
        topology_dir=str(TOPOLOGY_DIR),
        data_dir=str(RAW_ROOT / RAW_DIRS[0]),
        sensor_strategy="strategic",
        sensor_coverage=sensor_coverage,
        verbose=True,
        # Asset-type relation names are mandatory here: hydrognn's ASSET_GROUP
        # keys on them, and under the loader's default "behavior" naming
        # lines_as_nodes() hoists zero asset nodes, silently handing HydroGNN a
        # graph with no pipe/weir/pump types at all.
        relation_naming="asset_type",
    )

    all_manhole_idx = list(range(loader.num_manholes))
    graphs, provenance = [], []
    files = [(e, s, p) for e in EVENTS for s, p in inv.get(e, [])]
    print(f"\nBuilding graphs from {len(files)} simulations "
          f"({timesteps_per_sim} timesteps each)...")
    for i, (event, scen, path) in enumerate(files, 1):
        try:
            df = pd.read_csv(path, encoding="latin-1", header=0, skiprows=[1])
        except Exception as e:
            print(f"  [skip] {path.name}: {type(e).__name__}: {e}")
            continue
        idxs = np.linspace(0, len(df) - 1, min(timesteps_per_sim, len(df)), dtype=int)
        for t in idxs:
            try:
                # Store depths UNMASKED (every manhole treated as observed at
                # build time). create_heterogeneous_graph zeroes the depth of any
                # node outside sensor_indices, so passing the loader's own 309
                # sensors would bake a 95%-zero depth column into the file: the
                # normalisation statistics would collapse (mu ~0.07 rather than
                # ~1.71) and train.py's apply_mask would then designate a
                # different 309 nodes as sensors, most of them already zeroed.
                # Masking is train.py's job, driven by --sensor_file.
                g = loader.create_heterogeneous_graph(
                    depnod_file=path, timestep=int(t),
                    sensor_indices=all_manhole_idx)
            except Exception:
                continue
            graphs.append(g)
            provenance.append({"bui_id": event, "scenario_id": scen,
                               "timestep": int(t), "source_file": path.name})
        if i % 50 == 0:
            print(f"  {i}/{len(files)} simulations -> {len(graphs)} graphs")

    by_event = defaultdict(list)
    for i, p in enumerate(provenance):
        by_event[p["bui_id"]].append(i)

    fold_indices = {}
    for k, f in folds.items():
        idx = {split: sorted(i for e in f[f"{split}_event_ids"] for i in by_event[e])
               for split in ("train", "val", "test")}
        # index-level isolation, not just event-level
        assert set(idx["train"]).isdisjoint(idx["val"])
        assert set(idx["train"]).isdisjoint(idx["test"])
        assert set(idx["val"]).isdisjoint(idx["test"])
        fold_indices[k] = idx
    return graphs, provenance, fold_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition-only", action="store_true",
                    help="Verify the fold partition without building graphs.")
    ap.add_argument("--timesteps-per-sim", type=int, default=13)
    ap.add_argument("--sensor-coverage", type=float, default=0.05)
    args = ap.parse_args()

    inv = inventory()
    missing = [e for e in EVENTS if e not in inv]
    if missing:
        raise RuntimeError(f"storms absent from raw data: {missing}")

    folds = build_folds()
    assert_disjoint(folds)
    report_partition(inv, folds)
    print("\nDisjointness assertions PASSED for all 5 folds "
          "(train/val, train/test, val/test; test folds partition all 10 storms).")

    if args.partition_only:
        out = REPO / "data/heusden_folds.json"
        out.write_text(json.dumps(
            {"folds": folds,
             "files_per_event": {e: len(inv[e]) for e in EVENTS}}, indent=2))
        print(f"Written {out}")
        return

    graphs, provenance, fold_indices = build_dataset(
        inv, folds, args.timesteps_per_sim, args.sensor_coverage)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "graphs": graphs,
        "event_provenance": provenance,
        "fold_assignments": folds,
        "fold_indices": fold_indices,
        "n_folds": len(folds),
    }, OUT_PATH)
    print(f"\nSaved {len(graphs)} graphs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
