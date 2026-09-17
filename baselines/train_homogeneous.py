"""
Train homogeneous GNN baselines (GCN, GAT, GraphSAGE, ChebNet, GIN) on
Bellinge or Tuindorp using the spatial .pt format.

Evaluation protocol matches HydroGNN exactly:
- Fixed k-means sensor mask (indices from --sensor_file)
- MAE on non-sensor manholes only, de-normalized to centimetres
- Same train/val split as HydroGNN (train_size / val_size from .pt)

Usage:
    python baselines/train_homogeneous.py \
        --data_file  outputs/bellinge_v3/bellinge_v3_nopumpstatus.pt \
        --sensor_file analysis/sensor_optimization/sensors_bellinge_kmeans.txt \
        --arch       gcn \
        --output_dir outputs/baselines_homo/bellinge/gcn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, GINConv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ─── Models ─────────────────────────────────────────────────────────────────

class HomoGNN(torch.nn.Module):
    def __init__(self, arch: str, in_ch: int, hidden: int, layers: int,
                 heads: int = 4, cheb_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.arch = arch
        self.dropout = dropout

        def make_conv(in_c, out_c):
            if arch == "gcn":
                return GCNConv(in_c, out_c)
            if arch == "gat":
                h = heads if out_c == hidden else 1
                return GATConv(in_c, out_c // h, heads=h, concat=True, dropout=dropout)
            if arch == "sage":
                return SAGEConv(in_c, out_c)
            if arch == "cheb":
                return ChebConv(in_c, out_c, K=cheb_k)
            if arch == "gin":
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(in_c, out_c), torch.nn.ReLU(),
                    torch.nn.Linear(out_c, out_c))
                return GINConv(mlp)
            if arch == "gatres":
                # GATRes: multi-head graph attention with residual connections
                # and batch normalisation. Same conv as "gat"; the residual and
                # BatchNorm are applied in forward().
                return GATConv(in_c, out_c // heads, heads=heads, concat=True,
                               dropout=dropout)
            raise ValueError(arch)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(layers):
            ic = in_ch if i == 0 else hidden
            self.convs.append(make_conv(ic, hidden))
            self.norms.append(torch.nn.BatchNorm1d(hidden) if arch == "gatres"
                              else torch.nn.LayerNorm(hidden))
        # GATRes carries a residual around every block; the first block needs a
        # projection because in_ch != hidden.
        self.res_proj = (torch.nn.Linear(in_ch, hidden)
                         if arch == "gatres" else None)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 64), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x, edge_index):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.arch == "gatres":
                # Residual add before the nonlinearity. The previous form,
                # relu(norm(conv(x))) + identity, added a strictly non-negative
                # branch to the skip stream at every layer, so activations
                # compounded monotonically (0.335 -> 0.826 mean magnitude over
                # 4 layers, against 0.40 flat for plain GAT). That inflation,
                # not the method, is why this baseline scored worst.
                identity = self.res_proj(x) if i == 0 else x
                x = F.relu(norm(conv(x, edge_index)) + identity)
                x = F.dropout(x, p=self.dropout, training=self.training)
                continue
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


# ─── Data helpers ────────────────────────────────────────────────────────────

def load_sensor_mask(path: Path, n_manholes: int) -> torch.Tensor:
    indices = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        indices.append(int(line.split()[0]))
    mask = torch.ones(n_manholes, dtype=torch.bool)   # True = non-sensor (masked)
    for i in indices:
        if 0 <= i < n_manholes:
            mask[i] = False
    return mask  # True where we predict (non-sensor)


def build_random_mask(n: int, mask_rate: float, rng: torch.Generator) -> torch.Tensor:
    """True = non-sensor (predicted). Fresh random sensor placement."""
    n_sensors = max(1, int(round(n * (1.0 - mask_rate))))
    m = torch.ones(n, dtype=torch.bool)
    m[torch.randperm(n, generator=rng)[:n_sensors]] = False
    return m


def mask_x(x: torch.Tensor, pred_mask: torch.Tensor, feat_dim: int) -> torch.Tensor:
    """Apply same input masking as HydroGNN train.py.

    Columns 3..feat_dim-2 are dynamic (depth, velocity, …) — zeroed for
    non-sensor nodes.  Column feat_dim-1 is the sensor flag — 1 for sensors,
    0 for non-sensors.  Static columns 0-2 are left unchanged for all nodes.
    pred_mask: True = non-sensor (predict), False = sensor (observed).
    """
    x = x.clone()
    x[pred_mask, 3:feat_dim - 1] = 0.0   # zero dynamic cols for non-sensors
    x[:, feat_dim - 1] = 0.0
    x[~pred_mask, feat_dim - 1] = 1.0    # sensor flag: 1 = sensor, 0 = non-sensor
    return x


def homo_node_offsets(g) -> Tuple[List[str], Dict[str, int], int]:
    """Assign every node type a slice of one flat index space, manholes first.

    Returns (ordered type names, {type: offset}, total node count).
    """
    types = ["manhole"] + sorted(nt for nt in g.node_types
                                 if nt != "manhole" and "x" in g[nt])
    offsets, total = {}, 0
    for nt in types:
        offsets[nt] = total
        total += g[nt].x.size(0)
    return types, offsets, total


def get_homo_edge_index(g) -> torch.Tensor:
    """Every edge in the graph, flattened into one untyped undirected edge set.

    The homogeneous control must see the same *network* as HydroGNN — the same
    nodes and the same connectivity — and differ only in that it is told nothing
    about node or edge types. Keeping only manhole->manhole edges would instead
    hand it a smaller network: on Bellinge that discards 25 storage/outfall nodes
    and 40 of 1037 edges, so part of any HydroGNN margin would be missing
    connectivity rather than heterogeneity.
    """
    _, offsets, _ = homo_node_offsets(g)
    parts = []
    for (src_t, _, dst_t), store in g.edge_items():
        if src_t not in offsets or dst_t not in offsets:
            continue
        ei = store.edge_index
        if ei.numel() == 0:
            continue
        parts.append(torch.stack([ei[0] + offsets[src_t], ei[1] + offsets[dst_t]]))
    if not parts:
        raise RuntimeError("No edges found")
    ei = torch.cat(parts, dim=1)
    ei = torch.cat([ei, ei.flip(0)], dim=1)          # undirected
    return torch.unique(ei, dim=1)


def homo_type_stats(graphs, feat_dim: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Per-node-type mean/std for the non-manhole types, from the TRAIN graphs.

    Manhole features are already z-normalised by the preprocessing; the other
    types are not always. Tuindorp's outfall rows still hold raw RD coordinates
    (absmax 457788 against 3.47 for manholes), so folding them into the shared
    matrix unscaled would swamp every other input. Bellinge already normalises
    per node type upstream, so this is a no-op there.

    Statistics come from training graphs only.
    """
    stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    types, _, _ = homo_node_offsets(graphs[0])
    for nt in types:
        if nt == "manhole":
            continue
        xs = torch.cat([g[nt].x for g in graphs], dim=0)
        stats[nt] = (xs.mean(0), xs.std(0).clamp(min=1e-6))
    return stats


def get_homo_x(g, feat_dim: int,
               type_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
               drop_ground_level: bool = False) -> torch.Tensor:
    """Stack every node type's features into one untyped matrix.

    Types carry different numbers of attributes (manhole 10, storage 4,
    outfall 3), so shorter rows are zero-padded to a common width. No type
    indicator is added: as far as the model is concerned these are all just
    nodes, which is the point of the control.
    """
    types, _, total = homo_node_offsets(g)
    x = g["manhole"].x.new_zeros(total, feat_dim)
    # `drop_ground_level` zeroes column 0 rather than removing it, so the column
    # indices that mask_x relies on (dynamic 3..feat_dim-2, sensor flag
    # feat_dim-1) stay put. A constant column carries no information, so the
    # model has no access to elevation -- leaving the homogeneous control with
    # coordinates and masked water levels only.
    row = 0
    for nt in types:
        xi = g[nt].x
        if type_stats is not None and nt in type_stats:
            mu, sd = type_stats[nt]
            xi = (xi - mu) / sd
        n, f = xi.size(0), min(xi.size(1), feat_dim)
        x[row:row + n, :f] = xi[:, :f]
        row += n
    if drop_ground_level:
        x[:, 0] = 0.0
    return x


# ─── Training / evaluation ───────────────────────────────────────────────────

def evaluate(model, graphs, edge_index, pred_mask, depth_mean, depth_std, device,
             feat_dim: int = 10, wet_thresh_cm: float = 5.0,
             mask_full: Optional[torch.Tensor] = None, type_stats=None,
             drop_gl: bool = False):
    """Return dict of metrics; wet MAE uses depth > wet_thresh_cm as wet condition.

    The model runs over every node in the flattened graph; only manhole rows
    carry a depth target, so scoring slices the leading n_manholes predictions.
    """
    model.eval()
    n_mh = pred_mask.size(0)
    mf = pred_mask if mask_full is None else mask_full
    all_pred, all_true = [], []
    with torch.no_grad():
        for g in graphs:
            x  = mask_x(get_homo_x(g, feat_dim, type_stats), mf, feat_dim).to(device)
            y  = g["manhole"].y.squeeze(-1).to(device)
            ei = edge_index.to(device)
            m  = pred_mask.to(device)
            pred = model(x, ei)[:n_mh]
            all_pred.append(((pred[m] * depth_std + depth_mean) * 100.0).cpu())
            all_true.append(((y[m]   * depth_std + depth_mean) * 100.0).cpu())

    p = torch.cat(all_pred)
    t = torch.cat(all_true)
    err = p - t

    mae  = err.abs().mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    bias = err.mean().item()

    ss_res = err.pow(2).sum().item()
    ss_tot = (t - t.mean()).pow(2).sum().item()
    nse = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    pz = p - p.mean();  tz = t - t.mean()
    pearson_r = ((pz * tz).sum() / (pz.norm() * tz.norm() + 1e-8)).item()

    wet = t > wet_thresh_cm
    wet_mae = err[wet].abs().mean().item() if wet.sum() > 0 else float("nan")

    return dict(mae=mae, wet_mae=wet_mae, rmse=rmse, nse=nse,
                pearson_r=pearson_r, bias=bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file",   required=True)
    ap.add_argument("--sensor_file", required=True)
    ap.add_argument("--arch", required=True, choices=["gcn", "gat", "sage", "cheb", "gin", "gatres"])
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--hidden_dim",  type=int,   default=128)
    ap.add_argument("--num_layers",  type=int,   default=4)
    ap.add_argument("--heads",       type=int,   default=4)
    ap.add_argument("--cheb_k",      type=int,   default=3)
    ap.add_argument("--dropout",     type=float, default=0.1)
    ap.add_argument("--epochs",      type=int,   default=200)
    ap.add_argument("--batch_size",  type=int,   default=16)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay",type=float, default=1e-5)
    ap.add_argument("--patience",    type=int,   default=40)
    ap.add_argument("--drop_ground_level", action="store_true",
                    help="Zero the ground_level column so the homogeneous control sees "
                         "only coordinates, masked water levels and the sensor flag.")
    ap.add_argument("--random_train_mask", action="store_true",
                    help="Re-draw the sensor placement for every graph every epoch during "
                         "TRAINING; val/test keep --sensor_file. Mirrors hydrognn/train.py.")
    ap.add_argument("--mask_rate",   type=float, default=0.95)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading {args.data_file} ...")
    raw = torch.load(args.data_file, map_location="cpu", weights_only=False)
    graphs    = raw["graphs"]
    train_end = raw["train_size"]
    val_end   = train_end + raw["val_size"]
    depth_mean = float(raw.get("depth_mean", 0.0))
    depth_std  = float(raw.get("depth_std",  1.0))

    train_graphs = graphs[:train_end]
    val_graphs   = graphs[train_end:val_end]
    n_manholes   = train_graphs[0]["manhole"].x.shape[0]
    in_ch        = train_graphs[0]["manhole"].x.shape[1]
    print(f"  {len(train_graphs)} train / {len(val_graphs)} val snapshots")
    print(f"  manholes={n_manholes}  in_channels={in_ch}")

    # ── Fixed edge index ───────────────────────────────────────────────────
    edge_index = get_homo_edge_index(train_graphs[0])
    print(f"  homogeneous edges: {edge_index.shape[1]}")

    # ── Sensor mask ────────────────────────────────────────────────────────
    pred_mask = load_sensor_mask(Path(args.sensor_file), n_manholes)
    # Full-length mask over the flattened node set. Storage and outfall nodes
    # carry no depth reading, so they are unobserved exactly like a non-sensor
    # manhole: dynamic columns zeroed, sensor flag 0. They still participate in
    # message passing, they are simply never scored.
    # Named distinctly: `rng` inside the epoch loop is the shuffle index tensor.
    mask_rng = torch.Generator(); mask_rng.manual_seed(args.seed)
    type_stats = homo_type_stats(train_graphs, in_ch)
    _, _, n_total = homo_node_offsets(train_graphs[0])
    mask_full = torch.ones(n_total, dtype=torch.bool)
    mask_full[:n_manholes] = pred_mask
    print(f"  nodes total={n_total} (manholes={n_manholes}, "
          f"other={n_total - n_manholes} untyped, in-graph but unscored)")
    print(f"  non-sensor (predicted) nodes: {pred_mask.sum()}")
    print(f"  masking cols 3..{in_ch-2} (dynamic) + sensor flag col {in_ch-1} for non-sensor nodes")

    # ── Model ──────────────────────────────────────────────────────────────
    model = HomoGNN(
        arch=args.arch, in_ch=in_ch, hidden=args.hidden_dim,
        layers=args.num_layers, heads=args.heads,
        cheb_k=args.cheb_k, dropout=args.dropout,
    ).to(device)
    print(f"  model={args.arch}  params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_mae  = float("inf")
    best_metrics  = {}
    best_ckpt     = out_dir / "best_model.pth"
    no_improve    = 0
    ei_dev        = edge_index.to(device)
    m_dev         = pred_mask.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = torch.randperm(len(train_graphs))
        total_loss = 0.0
        steps = 0
        for i in range(0, len(train_graphs), args.batch_size):
            idx = rng[i:i + args.batch_size].tolist()
            batch_loss = 0.0
            for j in idx:
                g    = train_graphs[j]
                if args.random_train_mask:
                    # Fresh placement for this graph. Non-manhole nodes stay
                    # unobserved; only the manhole block is re-drawn. Scoring
                    # still uses the fixed k-means mask, so val/test are
                    # unchanged and remain comparable to the fixed-mask runs.
                    mf = torch.ones(n_total, dtype=torch.bool)
                    mf[:n_manholes] = build_random_mask(n_manholes, args.mask_rate, mask_rng)
                    tm = mf[:n_manholes].to(device)
                else:
                    mf, tm = mask_full, m_dev
                x    = mask_x(get_homo_x(g, in_ch, type_stats, drop_ground_level=args.drop_ground_level), mf, in_ch).to(device)
                y    = g["manhole"].y.squeeze(-1).to(device)
                # The model runs over every node in the flattened graph; only
                # manhole rows carry a depth target, so the loss is taken on the
                # leading n_manholes predictions.
                pred = model(x, ei_dev)[:n_manholes]
                loss = F.l1_loss(pred[tm], y[tm])
                batch_loss = batch_loss + loss
            (batch_loss / len(idx)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()
            steps      += len(idx)
        scheduler.step()

        # Validate every epoch, matching hydrognn/train.py. Checking only every
        # 5th epoch let a baseline pass its best model without recording it.
        met = evaluate(model, val_graphs, edge_index, pred_mask,
                       depth_mean, depth_std, device, feat_dim=in_ch,
                       mask_full=mask_full, type_stats=type_stats,
                       drop_gl=args.drop_ground_level)
        val_mae = met["mae"]
        print(f"  ep {epoch:4d}  train_loss={total_loss/steps:.5f}"
              f"  mae={val_mae:.4f}  wet_mae={met['wet_mae']:.4f}"
              f"  rmse={met['rmse']:.4f}  nse={met['nse']:.4f}"
              f"  r={met['pearson_r']:.4f}  bias={met['bias']:+.4f}  [cm]")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_metrics = met
            no_improve   = 0
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch, "metrics": met,
                        "args": vars(args)}, best_ckpt)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    import json
    print(f"\nBest val metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    np.save(out_dir / "val_mae.npy", np.array([best_val_mae]))
    with open(out_dir / "best_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    # ── Held-out test split ────────────────────────────────────────────────
    # Scored once, after selection is finished. The test graphs take no part in
    # early stopping or checkpoint choice, so this is a clean generalisation
    # estimate. Absent a test split we report nothing rather than silently
    # falling back to the validation graphs.
    n_test = int(raw.get("test_size", len(graphs) - val_end))
    if n_test > 0:
        test_graphs = graphs[val_end:val_end + n_test]
        best_path = out_dir / "best_model.pth"
        if best_path.exists():
            sd = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(sd.get("model_state_dict", sd.get("state_dict", sd)))
        test_metrics = evaluate(model, test_graphs, edge_index, pred_mask,
                                depth_mean, depth_std, device, feat_dim=in_ch,
                                mask_full=mask_full, type_stats=type_stats,
                           drop_gl=args.drop_ground_level)
        print(f"\nHeld-out TEST metrics ({n_test} snapshots, "
              f"best-val checkpoint):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump({"n_test_snapshots": n_test, **test_metrics}, f, indent=2)
    else:
        print("\nNo test split in this dataset — no test metrics reported.")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
