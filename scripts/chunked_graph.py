"""Run the grid-scale parts of the model in row chunks, so that no
[grid points or edges] x 768 tensor is ever materialized whole.

Why: at 0.25 deg the grid has 1,038,240 points and the mesh->grid decoder has
3,114,720 edges. The port computes one 768-wide fp32 message per edge in one go:
3.1M x 768 x 4 B = 9.6 GB per tensor, with 2-3 alive at once. That, not attention,
is the next memory wall once the dense attention scores are gone.

What gets chunked (everything here is row-wise, so results are identical up to
floating-point rounding: linear layers, activations, LayerNorm over the feature
axis, and per-receiver sums):
  * every WeatherNext2ConditionedMlp (grid encoder, edge encoders, node updates)
  * grid->mesh GNN: chunks of mesh nodes, each with all of its incoming edges
    (edges are sorted by receiver), so each node's sum sees the same edges in the
    same order as the original embedding_bag
  * mesh->grid GNN: chunks of grid points, each with its 3 incoming edges
  * output head MLP

Two algebraic rewrites, both exact per row:
  * sender_proj(x)[senders] == sender_proj(x[senders])   (gather, then project:
    avoids the full-grid projected tensor; costs ~1.6x the FLOPs of that one matmul)
  * same for receiver_proj in the decoder.

    from chunked_graph import install_chunked
    install_chunked(model, chunk=65536)     # rows per chunk; Mini (65,160 points) is
                                            # then untouched except edge encoders
"""
import types

import torch
from torch import nn
from transformers.models.weathernext2 import modeling_weathernext2 as mw


def _ranges(n: int, chunk: int):
    for s in range(0, n, chunk):
        yield s, min(s + chunk, n)


def _chunk_conditioned_mlp(module: nn.Module, chunk: int) -> None:
    """Row-chunk a ConditionedMlp along the node/edge axis (dim 1)."""
    original = module.forward                       # bound method of the original class

    def forward(self, hidden_states, conditioning, spatial_features=None):
        n = spatial_features.shape[0] if spatial_features is not None else hidden_states.shape[1]
        if n <= chunk:
            return original(hidden_states, conditioning, spatial_features)
        out = None
        for s, e in _ranges(n, chunk):
            hs = hidden_states if hidden_states.ndim == 2 else hidden_states[:, s:e]
            y = original(hs, conditioning, None if spatial_features is None else spatial_features[s:e])
            if out is None:
                out = y.new_empty(y.shape[0], n, y.shape[-1])
            out[:, s:e] = y
        return out

    module.forward = types.MethodType(forward, module)


def _residual_rows(update: nn.Module, x: torch.Tensor, conditioning, chunk: int) -> torch.Tensor:
    """x + update(x, conditioning), computed chunk by chunk along dim 1."""
    out = torch.empty_like(x)
    for s, e in _ranges(x.shape[1], chunk):
        out[:, s:e] = x[:, s:e] + update(x[:, s:e], conditioning)
    return out


def _edge_messages(eu, edge_states, sender_rows, conditioning, receiver_rows=None):
    """The port's edge update, written on already-gathered sender/receiver rows."""
    msg = eu.edge_proj(edge_states) + eu.sender_proj(sender_rows)
    if receiver_rows is not None:
        msg = msg + eu.receiver_proj(receiver_rows)
    msg = eu.out_proj(eu.act_fn(msg))
    return eu.norm(msg, conditioning)


def _grid_to_mesh_forward(chunk: int):
    def forward(self, grid_states, mesh_states, edge_features, senders, receivers, conditioning):
        b, m, d = mesh_states.shape
        n_edges = receivers.shape[0]
        offsets = torch.searchsorted(receivers, torch.arange(m + 1, device=receivers.device))
        off = offsets.tolist()
        nodes_per_chunk = max(1, chunk * m // max(n_edges, 1))
        aggregated = mesh_states.new_zeros(b, m, d, dtype=torch.float32)
        for m0, m1 in _ranges(m, nodes_per_chunk):
            e0, e1 = off[m0], off[m1]
            if e1 == e0:
                continue
            es = self.edge_encoder(edge_features[:, e0:e1], conditioning)
            msg = _edge_messages(self.edge_update, es, grid_states[:, senders[e0:e1]], conditioning)
            idx = torch.arange(e1 - e0, device=msg.device)
            local = offsets[m0:m1 + 1] - e0
            for bi in range(b):
                aggregated[bi, m0:m1] = nn.functional.embedding_bag(
                    idx, msg[bi].float(), local, mode="sum", include_last_offset=True)
        if self.aggregate_normalization is not None:
            aggregated = aggregated / self.aggregate_normalization
        aggregated = aggregated.to(mesh_states.dtype)

        mesh_states = mesh_states + self.mesh_node_update(torch.cat([mesh_states, aggregated], dim=-1), conditioning)
        grid_states = _residual_rows(self.grid_node_update, grid_states, conditioning, chunk)
        return grid_states, mesh_states
    return forward


def _mesh_to_grid_forward(chunk: int):
    checked = {}

    def forward(self, grid_states, mesh_states, edge_features, senders, receivers, conditioning):
        g = grid_states.shape[1]
        if "ok" not in checked:   # the 3-contiguous-edges-per-grid-point layout the original relies on
            expect = torch.arange(g, device=receivers.device).repeat_interleave(3)
            if receivers.shape[0] != 3 * g or not torch.equal(receivers, expect):
                raise ValueError("mesh->grid receivers are not 3 contiguous edges per grid point")
            checked["ok"] = True
        eu = self.edge_update
        out = torch.empty_like(grid_states)
        points = max(1, chunk // 3)
        for g0, g1 in _ranges(g, points):
            e0, e1 = 3 * g0, 3 * g1
            es = self.edge_encoder(edge_features[:, e0:e1], conditioning)
            gs = grid_states[:, g0:g1]
            msg = _edge_messages(eu, es, mesh_states[:, senders[e0:e1]], conditioning,
                                 receiver_rows=gs.repeat_interleave(3, dim=1))
            agg = msg.reshape(msg.shape[0], g1 - g0, 3, -1).sum(dim=2, dtype=torch.float32).to(gs.dtype)
            out[:, g0:g1] = gs + self.grid_node_update(torch.cat([gs, agg], dim=-1), conditioning)
        mesh_states = mesh_states + self.mesh_node_update(mesh_states, conditioning)
        return out, mesh_states
    return forward


def _head_forward(chunk: int):
    def forward(self, grid_states):
        b, g, _ = grid_states.shape
        prediction = None
        for s, e in _ranges(g, chunk):
            p = self.mlp(grid_states[:, s:e])
            if prediction is None:
                prediction = p.new_empty(b, g, p.shape[-1])
            prediction[:, s:e] = p
        shifted = torch.sigmoid(prediction - self.sigmoid_shift)
        prediction = torch.where(self.sigmoid_gate, shifted, prediction)
        prediction = prediction.transpose(1, 2)
        return prediction.reshape(b, -1, self.config.grid_latitudes, self.config.grid_longitudes)
    return forward


def install_chunked(model, chunk: int = 65536) -> None:
    """Patch a WeatherNext2ForWeatherForecasting in place. chunk = rows (points/edges) per chunk."""
    for module in model.modules():
        if isinstance(module, mw.WeatherNext2ConditionedMlp):
            _chunk_conditioned_mlp(module, chunk)
    core = model.model
    core.grid_to_mesh.forward = types.MethodType(_grid_to_mesh_forward(chunk), core.grid_to_mesh)
    core.mesh_to_grid.forward = types.MethodType(_mesh_to_grid_forward(chunk), core.mesh_to_grid)
    model.head.forward = types.MethodType(_head_forward(chunk), model.head)
