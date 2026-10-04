"""Classical-only EPOCH point-cloud student and AST structural teacher.

The inference branch consumes only an unordered (x, y) point cloud.  During
pretraining it is aligned to a formula AST encoder composed of edge-aware local
message passing and graph-local global attention.  Exactly one local update
uses a compact RBF-basis KAN-style layer; the remaining updates are ordinary
MLPs, making the requested substitution explicit and ablatable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EpochArchitecture:
    d_model: int = 384
    embedding_dim: int = 256
    heads: int = 8
    point_layers: int = 6
    graph_local_layers: int = 4
    graph_global_layers: int = 2
    point_pool_tokens: int = 4
    dropout: float = 0.08
    kan_layer: int = 1
    kan_bottleneck: int = 96
    kan_grid: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, 2 * hidden)
        self.output = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(self.dropout(value * F.silu(gate)))


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, 2 * d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, real_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attn(x)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~real_mask, need_weights=False,
        )
        x = x + self.dropout(attended)
        return x + self.dropout(self.ff(self.norm_ff(x)))


class PointCloudEncoder(nn.Module):
    """Permutation-invariant point-set encoder used alone at inference."""

    def __init__(self, architecture: EpochArchitecture) -> None:
        super().__init__()
        d = architecture.d_model
        self.stem = nn.Sequential(
            nn.Linear(2, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d)
        )
        self.blocks = nn.ModuleList([
            SetAttentionBlock(d, architecture.heads, architecture.dropout)
            for _ in range(architecture.point_layers)
        ])
        self.pool_queries = nn.Parameter(
            torch.randn(1, architecture.point_pool_tokens, d) * 0.02
        )
        self.pool_norm_q = nn.LayerNorm(d)
        self.pool_norm_kv = nn.LayerNorm(d)
        self.pool_attention = nn.MultiheadAttention(
            d, architecture.heads, dropout=architecture.dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, architecture.embedding_dim),
        )

    def forward(
        self, points: torch.Tensor, real_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if real_mask is None:
            real_mask = torch.ones(
                points.shape[:2], device=points.device, dtype=torch.bool
            )
        hidden = self.stem(points)
        for block in self.blocks:
            hidden = block(hidden, real_mask)
        queries = self.pool_queries.expand(points.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            self.pool_norm_q(queries), self.pool_norm_kv(hidden),
            self.pool_norm_kv(hidden), key_padding_mask=~real_mask,
            need_weights=False,
        )
        return self.output((queries + pooled).mean(dim=1))


class RBFKANLinear(nn.Module):
    """Compact KAN-style linear map using learned coefficients over RBF bases."""

    def __init__(self, in_features: int, out_features: int, grid: int = 8) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid = grid
        centers = torch.linspace(-2.0, 2.0, grid)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.tensor(math.log(0.65)))
        self.coefficients = nn.Parameter(
            torch.empty(out_features, in_features, grid)
        )
        self.base = nn.Linear(in_features, out_features)
        nn.init.normal_(
            self.coefficients, std=0.5 / math.sqrt(in_features * grid)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = self.log_width.exp().clamp(0.15, 2.0)
        basis = torch.exp(-((x.unsqueeze(-1) - self.centers) / width).square())
        spline = torch.einsum("nig,oig->no", basis, self.coefficients)
        return self.base(F.silu(x)) + spline


class GraphLocalUpdate(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        use_kan: bool,
        kan_bottleneck: int,
        kan_grid: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * d_model)
        if use_kan:
            self.update = nn.Sequential(
                nn.Linear(2 * d_model, kan_bottleneck),
                nn.LayerNorm(kan_bottleneck),
                RBFKANLinear(kan_bottleneck, kan_bottleneck, kan_grid),
                nn.GELU(),
                nn.Linear(kan_bottleneck, d_model),
            )
        else:
            self.update = SwiGLU(2 * d_model, 2 * d_model, dropout)
            self.project = nn.Linear(2 * d_model, d_model)
        self.use_kan = use_kan
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, own: torch.Tensor, aggregate: torch.Tensor) -> torch.Tensor:
        joined = self.norm(torch.cat((own, aggregate), dim=-1))
        if self.use_kan:
            delta = self.update(joined)
        else:
            delta = self.project(self.update(joined))
        return self.output_norm(own + self.dropout(delta))


class FormulaGraphEncoder(nn.Module):
    """Edge-aware AST message passing followed by graph-local Transformer."""

    def __init__(self, vocab_size: int, architecture: EpochArchitecture) -> None:
        super().__init__()
        d = architecture.d_model
        self.type_embedding = nn.Embedding(vocab_size, d)
        self.variable_embedding = nn.Embedding(17, d)
        self.depth_embedding = nn.Embedding(32, d)
        self.position_embedding = nn.Embedding(128, d)
        self.constant_projection = nn.Sequential(
            nn.Linear(1, d), nn.Tanh(), nn.Linear(d, d)
        )
        self.edge_direction = nn.Embedding(2, d)
        self.edge_position = nn.Embedding(32, d)
        self.message = nn.ModuleList([nn.Linear(d, d) for _ in range(
            architecture.graph_local_layers
        )])
        self.local_updates = nn.ModuleList([
            GraphLocalUpdate(
                d, architecture.dropout, layer == architecture.kan_layer,
                architecture.kan_bottleneck, architecture.kan_grid,
            )
            for layer in range(architecture.graph_local_layers)
        ])
        global_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=architecture.heads,
            dim_feedforward=4 * d,
            dropout=architecture.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            global_layer, num_layers=architecture.graph_global_layers,
            enable_nested_tensor=False,
        )
        self.pool_score = nn.Linear(d, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, architecture.embedding_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_batch = batch["node_batch"]
        graph_ptr = batch["graph_ptr"]
        local_position = (
            torch.arange(node_batch.numel(), device=node_batch.device)
            - graph_ptr[:-1][node_batch]
        )
        constant = torch.sign(batch["node_constant"]) * torch.log1p(
            batch["node_constant"].abs()
        )
        hidden = (
            self.type_embedding(batch["node_type"])
            + self.variable_embedding((batch["node_variable_slot"] + 1).clamp(0, 16))
            + self.depth_embedding(batch["node_depth"].clamp(0, 31))
            + self.position_embedding(local_position.clamp(0, 127))
            + self.constant_projection(constant[:, None])
        )
        source, target = batch["edge_index"]
        edge = (
            self.edge_direction(batch["edge_direction"])
            + self.edge_position(batch["edge_child_position"].clamp(0, 31))
        )
        for message, update in zip(self.message, self.local_updates):
            messages = message(hidden[source]) + edge
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, target, messages)
            degree = torch.zeros(
                hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype
            )
            degree.index_add_(
                0, target,
                torch.ones(target.shape[0], 1, device=hidden.device, dtype=hidden.dtype),
            )
            hidden = update(hidden, aggregate / degree.clamp_min(1))

        counts = torch.diff(graph_ptr)
        graph_count = int(counts.numel())
        max_nodes = int(counts.max())
        padded = hidden.new_zeros(graph_count, max_nodes, hidden.shape[-1])
        padded[node_batch, local_position] = hidden
        padding_mask = (
            torch.arange(max_nodes, device=hidden.device)[None, :] >= counts[:, None]
        )
        padded = self.global_encoder(padded, src_key_padding_mask=padding_mask)
        score = self.pool_score(padded).squeeze(-1).masked_fill(
            padding_mask, float("-inf")
        )
        weight = score.softmax(dim=-1)
        pooled = torch.einsum("gn,gnd->gd", weight, padded)
        return self.output(pooled)


class EpochDualEncoder(nn.Module):
    def __init__(self, vocab_size: int, architecture: EpochArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.point_encoder = PointCloudEncoder(architecture)
        self.formula_encoder = FormulaGraphEncoder(vocab_size, architecture)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(1.0, 100.0)


def positive_log_softmax_loss(
    logits: torch.Tensor, positive_mask: torch.Tensor
) -> torch.Tensor:
    """Mean -log probability mass assigned to one or more positives."""
    valid = positive_mask.any(dim=1)
    if not bool(valid.any()):
        return logits.sum() * 0
    denominator = torch.logsumexp(logits[valid], dim=1)
    numerator = torch.logsumexp(
        logits[valid].masked_fill(~positive_mask[valid], float("-inf")), dim=1
    )
    return (denominator - numerator).mean()


def cross_modal_loss(
    point_embeddings: torch.Tensor,
    point_formula_ids: torch.Tensor,
    graph_embeddings: torch.Tensor,
    graph_formula_ids: torch.Tensor,
    scale: torch.Tensor,
    queue_embeddings: torch.Tensor | None = None,
    queue_formula_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    point = F.normalize(point_embeddings, dim=-1)
    graph = F.normalize(graph_embeddings, dim=-1)
    candidates = graph
    candidate_ids = graph_formula_ids
    if queue_embeddings is not None and queue_embeddings.numel():
        candidates = torch.cat((graph, queue_embeddings.detach()), dim=0)
        candidate_ids = torch.cat((graph_formula_ids, queue_formula_ids), dim=0)
    point_logits = scale * (point @ candidates.T)
    point_positive = point_formula_ids[:, None] == candidate_ids[None, :]
    point_to_graph = positive_log_softmax_loss(point_logits, point_positive)

    graph_logits = scale * (graph @ point.T)
    graph_positive = graph_formula_ids[:, None] == point_formula_ids[None, :]
    graph_to_point = positive_log_softmax_loss(graph_logits, graph_positive)
    total = 0.5 * (point_to_graph + graph_to_point)
    return total, {
        "point_to_graph": point_to_graph.detach(),
        "graph_to_point": graph_to_point.detach(),
    }


def view_regularization(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cosine = (1.0 - F.cosine_similarity(first, second, dim=-1)).mean()
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    std_first = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
    std_second = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(0.5 - std_first).mean() + F.relu(0.5 - std_second).mean()
    )
    return cosine + 0.1 * variance, {
        "view_cosine": cosine.detach(), "variance": variance.detach()
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
