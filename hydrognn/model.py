"""HydroGNN model.

Architecture: heterogeneous GNN with type-pair weight matrices and standard
multi-head attention message passing. NO multiplicative gates. NO per-behavior
sub-modules. Every (src_type, rel, dst_type) edge gets its own `TypePairMP`
with the same forward equation but its own learned weights.

Input: HeteroData with node types {manhole, outfall, storage, pipe, weir, pump}
Output: per-manhole depth prediction (znorm scale)

The model has one job per forward pass: predict depth at every manhole node.
Physics constraints are enforced by the loss in `physics_loss.py`.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as scatter_softmax


# ── one (src,rel,dst) message-passing module --------------------------------
class TypePairMP(MessagePassing):
    """Standard GAT-style update for a single edge type.

    Forward:
        m_{j→i} = α_{ji} · ( W_src x_j + W_edge e_{ji} )       # per head
        x_i ← Σ_j m_{j→i}                                      # sum-aggregate

    `α_{ji} = softmax_i ( leakyReLU( a^T concat[W_src x_j , W_dst x_i , W_edge e] ) )`

    Type-specificity comes from W_src, W_dst,
    W_edge and `att` being instantiated separately per (src_type, rel, dst_type).
    """

    def __init__(
        self,
        in_src: int,
        in_dst: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(aggr="add")
        self.in_src = in_src
        self.in_dst = in_dst
        self.out_channels = out_channels
        self.edge_dim = int(edge_dim) if edge_dim is not None else 0
        self.heads = heads
        self.dropout = dropout

        self.lin_src = nn.Linear(in_src, out_channels * heads)
        self.lin_dst = nn.Linear(in_dst, out_channels * heads)
        if self.edge_dim > 0:
            self.lin_edge: Optional[nn.Linear] = nn.Linear(self.edge_dim, out_channels * heads)
        else:
            self.lin_edge = None

        att_in = (3 if self.edge_dim > 0 else 2) * out_channels
        self.att = nn.Parameter(torch.empty(1, heads, att_in))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight); nn.init.zeros_(self.lin_src.bias)
        nn.init.xavier_uniform_(self.lin_dst.weight); nn.init.zeros_(self.lin_dst.bias)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight); nn.init.zeros_(self.lin_edge.bias)
        nn.init.xavier_uniform_(self.att)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H, C = self.heads, self.out_channels
        N_dst = x_dst.size(0)

        Sj = self.lin_src(x_src).view(-1, H, C)            # [N_src, H, C]
        Di = self.lin_dst(x_dst).view(-1, H, C)            # [N_dst, H, C]

        if self.lin_edge is not None and edge_attr is not None:
            Ee = self.lin_edge(edge_attr).view(-1, H, C)   # [E, H, C]
        else:
            Ee = None

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        sj = Sj[src_idx]                                   # [E, H, C]
        di = Di[dst_idx]                                   # [E, H, C]

        if Ee is not None:
            cat = torch.cat([sj, di, Ee], dim=-1)          # [E, H, 3C]
        else:
            cat = torch.cat([sj, di], dim=-1)              # [E, H, 2C]

        alpha = (cat * self.att).sum(dim=-1)               # [E, H]
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = scatter_softmax(alpha, dst_idx, num_nodes=N_dst)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if Ee is not None:
            msg = sj + Ee                                  # [E, H, C]
        else:
            msg = sj                                       # [E, H, C]
        msg = alpha.unsqueeze(-1) * msg                    # [E, H, C]

        out = torch.zeros(N_dst, H, C, device=x_dst.device, dtype=msg.dtype)
        idx = dst_idx.view(-1, 1, 1).expand(-1, H, C)
        out = out.scatter_add_(0, idx, msg)                # [N_dst, H, C]
        return out.view(N_dst, H * C)


# ── one residual block over the whole heterogeneous graph -------------------
class HeteroResBlock(nn.Module):
    """One round of heterogeneous message passing + node update + residual.

    Per-node-type:
        h_τ        = node_transforms_τ(x_τ)                # Linear+LN+ReLU+Drop
        m_τ        = Σ over edges (·, ·, τ): TypePairMP(...)
        out_τ      = output_transforms_τ(m_τ)              # Linear+LN+ReLU+Drop
        x_τ_new    = h_τ + out_τ                           # proper residual
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        edge_dims: Dict[Tuple[str, str, str], int],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Per-type input transform (every block starts by re-projecting features)
        self.node_transforms = nn.ModuleDict()
        for nt, in_dim in node_dims.items():
            self.node_transforms[nt] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # One TypePairMP per (src,rel,dst) edge type the data carries
        self.convs = nn.ModuleDict()
        for et in edge_types:
            src_t, rel_t, dst_t = et
            if src_t not in node_dims or dst_t not in node_dims:
                continue
            ed = edge_dims.get(et, 0)
            self.convs[self._key(et)] = TypePairMP(
                in_src=hidden_dim, in_dst=hidden_dim, out_channels=hidden_dim // heads,
                edge_dim=ed if ed > 0 else None, heads=heads, dropout=dropout,
            )

        # Per-type output transform on the aggregated messages
        self.output_transforms = nn.ModuleDict()
        for nt in node_dims.keys():
            self.output_transforms[nt] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=1e-3),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    @staticmethod
    def _key(et: Tuple[str, str, str]) -> str:
        return f"{et[0]}__{et[1]}__{et[2]}"

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1) per-type re-projection
        h: Dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            h[nt] = self.node_transforms[nt](x) if nt in self.node_transforms else x

        # 2) accumulate messages per destination node type
        msg_acc: Dict[str, torch.Tensor] = {nt: torch.zeros_like(h[nt]) for nt in h}
        for et, ei in edge_index_dict.items():
            key = self._key(et)
            if key not in self.convs:
                continue
            src_t, _, dst_t = et
            if src_t not in h or dst_t not in h:
                continue
            ea = edge_attr_dict.get(et) if edge_attr_dict else None
            m = self.convs[key](h[src_t], h[dst_t], ei, ea)
            msg_acc[dst_t] = msg_acc[dst_t] + m

        # 3) per-type output transform + proper residual
        new_x: Dict[str, torch.Tensor] = {}
        for nt in h:
            if nt in self.output_transforms:
                new_x[nt] = h[nt] + self.output_transforms[nt](msg_acc[nt])
            else:
                new_x[nt] = h[nt] + msg_acc[nt]
        return new_x


# ── full model --------------------------------------------------------------
class GlobalLowRankField(nn.Module):
    """A learned low-rank global pathway: sensors -> k coefficients -> every node.

    Message passing carries information a fixed number of hops. At 1% sensor
    coverage on Heusden the median manhole is 9 hops from the nearest sensor
    (p90 = 16, max = 32), and lines-as-nodes halves the effective reach, so 4
    blocks touch 2 manhole hops (5.6% of nodes) and 12 blocks touch 6 (25.4%).
    Most manholes therefore never receive a sensor reading at all and are
    predicted from static features alone.

    A gappy-POD reconstruction has no such limit: each spatial mode is global, so
    one coefficient fitted to the 62 sensor readings sets a value at all 6,197
    nodes at once. That is the whole of its roughly 2x advantage over this model.

    This module is that mechanism, learned end to end:

        coefficients  a = Enc( pool over SENSOR manholes only )    [B, k]
        per-node      u_i                                          [N, k] learned
        contribution  ( a_b (*) u_i ) W                            [B, N, hidden]

    `u` are learned analogues of POD modes and `a` of the mode coefficients, but
    unlike POD the graph then adds local nonlinear corrections on top, so the two
    are complementary rather than alternatives.

    NO LEAKAGE. The pool reads the masked feature matrix at sensor rows only --
    the same columns any node can see, never `y`, and never an unmonitored node's
    features. `u` is a free parameter fitted by the training loss like any other
    weight, not a table of observed values.
    """

    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int,
                 rank: int = 16, sensor_flag_idx: int = -1, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.sensor_flag_idx = sensor_flag_idx
        self.encoder = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, rank),
        )
        self.loadings = nn.Parameter(torch.randn(num_nodes, rank) * 0.02)
        self.project = nn.Linear(rank, hidden_dim)

    def forward(self, x_manhole: torch.Tensor) -> torch.Tensor:
        n_total, f = x_manhole.shape
        n = self.num_nodes
        if n_total % n != 0:
            return x_manhole.new_zeros(n_total, self.project.out_features)
        b = n_total // n
        xb = x_manhole.view(b, n, f)

        flag = xb[..., self.sensor_flag_idx]              # 1 at sensors, 0 elsewhere
        w = flag.unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1.0)
        mean = (xb * w).sum(dim=1) / denom
        # masked max: push non-sensor rows far below any real value
        neg = torch.finfo(xb.dtype).min
        mx = torch.where(w > 0, xb, torch.full_like(xb, neg)).max(dim=1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

        coef = self.encoder(torch.cat([mean, mx], dim=-1))          # [B, k]
        field = coef.unsqueeze(1) * self.loadings.unsqueeze(0)      # [B, N, k]
        return self.project(field).reshape(n_total, -1)


class VirtualNodePath(nn.Module):
    """A master node wired to every manhole: any node is 2 hops from any sensor.

    The same reach problem the low-rank pathway addresses, solved the other way.
    After each message-passing round the manhole states are pooled into a single
    global state, which is updated and then broadcast back to every node:

        v <- v + Update( mean_i h_i )
        h_i <- h_i + Broadcast( v )

    Compared with GlobalLowRankField the trade is explicit:

      * this has NO per-node parameters, so it stays permutation-equivariant and
        transfers to a network it was not trained on. The low-rank pathway's
        loadings are indexed by node, which ties it to one network permanently.
      * but every node receives the SAME broadcast vector, modulated only by its
        own state, where the low-rank pathway gives each node its own loading and
        so its own response to the same global state -- which is what a POD mode
        actually is, and what this model is being measured against.

    Whether the extra expressiveness or the transferability matters more is an
    empirical question, which is why both exist and are run against one control.

    NO LEAKAGE: it pools hidden states derived from the masked feature matrix,
    where unmonitored nodes have their depth columns zeroed. No target is read.
    """

    def __init__(self, hidden_dim: int, num_nodes: int, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.broadcast = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, v: Optional[torch.Tensor] = None):
        n_total, d = h.shape
        n = self.num_nodes
        if n_total % n != 0:
            return h, v
        b = n_total // n
        pooled = h.view(b, n, d).mean(dim=1)                 # [B, D]
        v = pooled.new_zeros(b, d) if v is None else v
        v = v + self.update(pooled)
        return h + self.broadcast(v).repeat_interleave(n, dim=0), v


class HydroGNN(nn.Module):
    """Stack of HeteroResBlock + manhole depth head.

    Args:
        node_dims:    {node_type: input_feature_dim}
        edge_dims:    {edge_type_tuple: edge_feature_dim or 0}
        edge_types:   list of all edge types the data carries
        hidden_dim:   model width
        num_blocks:   number of message-passing rounds
        heads:        attention heads per block
        dropout:      dropout in encoders, transforms, attention
        use_softplus_output: clamp depth output to non-negative via Softplus
        target_node_type:    node type the head predicts on (default 'manhole')
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        edge_dims: Dict[Tuple[str, str, str], int],
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int = 128,
        num_blocks: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
        use_softplus_output: bool = True,
        target_node_type: str = "manhole",
        num_system_types: int = 4,
        system_type_emb_dim: int = 16,
        use_system_type_conditioning: bool = True,
        use_input_skip: bool = False,
        use_static_highway: bool = False,
        virtual_node_nodes: int = 0,
        global_lowrank_nodes: int = 0,
        global_lowrank_k: int = 16,
        sensor_flag_idx: int = -1,
        static_feature_idx: Tuple[int, ...] = (0, 1, 2),
    ):
        super().__init__()
        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.use_system_type_conditioning = use_system_type_conditioning
        self.use_input_skip = use_input_skip
        # Raw static highway: bypass the message-passing stack entirely and hand
        # [invert, x, y] straight to the readout. The measured per-node error bias
        # correlates with each node's mean water level at r = -0.944 and with
        # invert elevation at r = -0.809 while being uncorrelated with distance to
        # the nearest sensor (-0.055), i.e. the model tracks the dynamics but
        # mean-reverts on the static elevation component that every block's
        # LayerNorm re-centres away.
        #
        # This overlaps use_input_skip, which already concatenates the full raw
        # manhole vector (invert at index 0, x and y at 1-2). The highway differs
        # in passing only the three spatial columns and, when the caller supplies
        # invert_level_m, the unnormalised invert in metres rather than its
        # z-scored column.
        self.use_static_highway = use_static_highway
        self.static_feature_idx = tuple(static_feature_idx)
        self.target_in_dim = node_dims.get(target_node_type, 0)

        # Input encoder per node type
        self.input_projections = nn.ModuleDict()
        for nt, in_dim in node_dims.items():
            self.input_projections[nt] = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        # After projection, every node type's working dim is hidden_dim
        block_node_dims = {nt: hidden_dim for nt in node_dims}
        self.blocks = nn.ModuleList([
            HeteroResBlock(block_node_dims, edge_dims, edge_types,
                           hidden_dim=hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Optional FiLM conditioning on system_type for the depth head
        if self.use_system_type_conditioning:
            self.system_type_emb = nn.Embedding(num_system_types, system_type_emb_dim)
            self.system_type_to_film = nn.Sequential(
                nn.Linear(system_type_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),
            )

        # Optional virtual master node (see VirtualNodePath), one per block.
        self.virtual_nodes = None
        if virtual_node_nodes:
            self.virtual_nodes = nn.ModuleList([
                VirtualNodePath(hidden_dim, virtual_node_nodes, dropout)
                for _ in range(num_blocks)])

        # Optional global low-rank pathway (see GlobalLowRankField).
        self.global_field = None
        if global_lowrank_nodes and global_lowrank_k:
            self.global_field = GlobalLowRankField(
                num_nodes=global_lowrank_nodes,
                in_dim=node_dims.get(target_node_type, 0),
                hidden_dim=hidden_dim, rank=global_lowrank_k,
                sensor_flag_idx=sensor_flag_idx, dropout=dropout)

        tail = nn.Softplus() if use_softplus_output else nn.Identity()
        head_in = (hidden_dim
                   + (self.target_in_dim if use_input_skip else 0)
                   + (len(self.static_feature_idx) if use_static_highway else 0))
        self.depth_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            tail,
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        system_type_idx: Optional[torch.Tensor] = None,
        static_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1) encoder
        x = {nt: self.input_projections[nt](v) if nt in self.input_projections else v
             for nt, v in x_dict.items()}

        # 2) message passing
        v = None
        for bi, block in enumerate(self.blocks):
            x = block(x, edge_index_dict, edge_attr_dict)
            if self.virtual_nodes is not None:
                # Global exchange after every round, so sensor information is not
                # limited to the hop count of the stack.
                x = dict(x)
                x[self.target_node_type], v = self.virtual_nodes[bi](
                    x[self.target_node_type], v)

        # 3) depth head on target node type only
        h = x[self.target_node_type]
        if self.global_field is not None:
            # Global sensor information, reaching every node in one step rather
            # than propagating hop by hop.
            h = h + self.global_field(x_dict[self.target_node_type])
        if self.use_system_type_conditioning and system_type_idx is not None:
            emb = self.system_type_emb(system_type_idx)
            film = self.system_type_to_film(emb)
            gamma, beta = film.chunk(2, dim=-1)
            h = h * (1.0 + 0.1 * torch.tanh(gamma)) + beta
        if self.use_input_skip:
            # Concatenate input features to the post-GNN embedding before the head.
            h = torch.cat([h, x_dict[self.target_node_type]], dim=-1)
        if self.use_static_highway:
            stat = x_dict[self.target_node_type][:, list(self.static_feature_idx)]
            if static_raw is not None:
                # unnormalised invert in metres in place of its z-scored column
                stat = torch.cat(
                    [static_raw.view(-1, 1).to(stat.dtype), stat[:, 1:]], dim=-1)
            h = torch.cat([h, stat], dim=-1)
        return self.depth_head(h).squeeze(-1)


# ── helper: introspect a HeteroData to get node_dims/edge_dims/edge_types ---
def schema_from_graph(g) -> Tuple[Dict[str, int], Dict[Tuple[str, str, str], int], List[Tuple[str, str, str]]]:
    node_dims: Dict[str, int] = {}
    for nt in g.node_types:
        if "x" in g[nt] and g[nt].x is not None:
            node_dims[nt] = g[nt].x.size(-1)
    edge_dims: Dict[Tuple[str, str, str], int] = {}
    edge_types: List[Tuple[str, str, str]] = []
    for et in g.edge_types:
        edge_types.append(et)
        if "edge_attr" in g[et] and g[et].edge_attr is not None:
            edge_dims[et] = int(g[et].edge_attr.size(-1))
        else:
            edge_dims[et] = 0
    return node_dims, edge_dims, edge_types


if __name__ == "__main__":
    # smoke test: python -m hydrognn.model path/to/dataset.pt
    import sys
    from pathlib import Path

    data_file = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                Path(__file__).parent / "examples/example_dataset.pt"

    raw = torch.load(data_file, map_location="cpu", weights_only=False)
    graphs = raw.get("graphs", raw.get("train_graphs", []))
    g = graphs[0]

    # Convert legacy format on-the-fly if needed
    if ("train_graphs" in raw):
        from hydrognn.data_loader import lines_as_nodes
        g = lines_as_nodes(g)

    node_dims, edge_dims, edge_types = schema_from_graph(g)
    print("node_dims:", node_dims)
    print(f"#edge_types: {len(edge_types)}")

    model = HydroGNN(
        node_dims=node_dims, edge_dims=edge_dims, edge_types=edge_types,
        hidden_dim=64, num_blocks=2, heads=4, dropout=0.1,
    )
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    edge_attr_dict = {et: g[et].edge_attr if "edge_attr" in g[et] else None for et in g.edge_types}
    with torch.no_grad():
        y = model(g.x_dict, g.edge_index_dict, edge_attr_dict)
    print(f"output shape: {y.shape}  range=[{y.min():.3f}, {y.max():.3f}]")
