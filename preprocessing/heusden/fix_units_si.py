"""Convert Heusden conduit attributes to SI in the fold datasets.

The Heusden model was exported from InfoWorks ICM in imperial units; Bellinge
and Tuindorp are already SI.

    col 0  diameter    inches   -> m         x 0.0254
    col 2  roughness   25.0     -> 0.04      the export column holds the Strickler
                                             coefficient (1/0.04 = 25)
    col 3  gradient    percent  -> fraction  / 100

Only `conduit` rows change; `channel` rows are already SI. Q_max is derived from
columns 0, 2 and 3 in lines_as_nodes, so it is corrected as well.
Fold files are rewritten in place (edge_attr only).

    python preprocessing/heusden/fix_units_si.py
"""
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_TO_M, MANNING_N, PCT_TO_FRAC = 0.0254, 0.04, 0.01


def fix(path: Path) -> None:
    d = torch.load(path, map_location="cpu", weights_only=False)
    graphs = d["graphs"]
    ets = [et for et in graphs[0].edge_types if et[1] == "conduit"]
    before = graphs[0][ets[0]].edge_attr[0].tolist()
    if abs(before[2] - MANNING_N) < 1e-9:
        print(f"  {path.name}: already SI, skipped")
        return
    n = 0
    for g in graphs:
        for et in ets:
            ea = g[et].edge_attr
            if ea is None or ea.size(1) < 4:
                continue
            ea[:, 0] *= IN_TO_M
            ea[:, 2] = MANNING_N
            ea[:, 3] *= PCT_TO_FRAC
            n += ea.size(0)
    after = graphs[0][ets[0]].edge_attr[0].tolist()
    torch.save(d, path)
    print(f"  {path.name}: {n} conduit rows over {len(graphs)} graphs")
    print(f"      before {[round(v,5) for v in before]}")
    print(f"      after  {[round(v,5) for v in after]}")


if __name__ == "__main__":
    for k in range(5):
        fix(REPO / f"data/processed/heusden_v3/folds/heusden_v3_fold{k}.pt")
