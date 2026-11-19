"""Swap the port's blocked mesh attention for ONE FlexAttention call over all
mesh nodes, in a chosen node ordering.

Original (port + JAX GPU path): nodes in RCM order are cut into blocks of size
= bandwidth; each block attends to itself + both neighbouring blocks, masked.
Here: the exact same neighbour sets (rebuilt from the model's own blocked mask),
as one N x N block-sparse mask, optionally with the nodes renumbered along a
Hilbert curve so neighbourhoods pack into fewer 128 x 128 tiles.

Per node the math is identical: attention depends on WHICH nodes are neighbours,
not on their order. Only floating-point summation order changes.

    from mesh_flex_attention import install
    install(model, order="hilbert", attn_dtype=torch.float32)
    install(model, order="hilbert", **TUNED)   # tuned tile sizes (scripts/12)
"""
import importlib.util
import pathlib
import types

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_here = pathlib.Path(__file__).parent
_spec = importlib.util.spec_from_file_location("orders", _here / "10_ordering_analysis.py")
orders = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orders)

_flex = torch.compile(flex_attention)


def full_mask_from_blocked(blocked: torch.Tensor, n: int) -> torch.Tensor:
    """[num_blocks, 1, B, 3B] (each block vs [prev, own, next]) -> [n, n]."""
    nb, _, bs, _ = blocked.shape
    m = torch.zeros(nb * bs, nb * bs, dtype=torch.bool, device=blocked.device)
    for b in range(nb):
        for j, nbr in enumerate((b - 1, b, b + 1)):
            if 0 <= nbr < nb:
                m[b * bs:(b + 1) * bs, nbr * bs:(nbr + 1) * bs] = blocked[b, 0, :, j * bs:(j + 1) * bs]
    if m[n:].any() or m[:, n:].any():
        raise ValueError("padding nodes take part in attention; full-mesh rewrite would change results")
    return m[:n, :n]


def node_order(order: str, mask: torch.Tensor) -> torch.Tensor:
    """Permutation (new position -> model node index)."""
    n = mask.shape[0]
    if order == "native":
        return torch.arange(n, device=mask.device)
    # The model's node index i is icosahedral vertex rcm_perm[i]; check that
    # claim against the model's own mask before using the coordinates.
    mesh = orders.icosahedral_mesh.get_last_triangular_mesh_for_sphere({10242: 5, 40962: 6}[n])
    rcm = orders.icosahedral_mesh.get_permutation_to_banded(mesh)[0]
    xyz = mesh.vertices[rcm]                                   # coords in model order
    i, j = torch.nonzero(mask[:64], as_tuple=True)             # spot check: first 64 rows
    d = np.linalg.norm(xyz[i.cpu().numpy()] - xyz[j.cpu().numpy()], axis=1)
    if d.max() > 1.0:                                          # 16 hops on this mesh << 1 rad
        raise ValueError("model node order doesn't match RCM-permuted mesh vertices")
    key = {"hilbert": orders.hilbert_key, "morton": orders.morton_key}[order](xyz)
    return torch.from_numpy(np.argsort(key, kind="stable").astype(np.int64)).to(mask.device)


# Best settings from scripts/12_flex_tuning.py (A10, fp32/TF32, Mini mesh, Hilbert order):
TUNED = {"block": 64, "kernel_options": {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1, "num_warps": 8}}


def _attend(attn, h, block_mask, dtype, kernel_options=None):
    b, n, _ = h.shape
    shape = (b, n, -1, attn.head_dim)
    q = attn.q_proj(h).view(shape).transpose(1, 2).to(dtype)
    k = attn.k_proj(h).view(shape).transpose(1, 2).to(dtype)
    v = attn.v_proj(h).view(shape).transpose(1, 2).to(dtype)
    o = _flex(q, k, v, block_mask=block_mask, scale=attn.scaling, kernel_options=kernel_options)
    return attn.o_proj(o.transpose(1, 2).reshape(b, n, -1).to(h.dtype))


def install(model, order: str = "hilbert", attn_dtype: torch.dtype = torch.float32,
            block: int = 128, kernel_options: dict | None = None) -> None:
    enc = model.model.mesh_transformer
    cache = {}

    def forward(self, hidden_states, attention_mask, conditioning, **kwargs):
        if "bm" not in cache:
            n = hidden_states.shape[1]
            mask = full_mask_from_blocked(attention_mask, n)
            perm = node_order(order, mask)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(n, device=perm.device)
            dense = mask[perm][:, perm].contiguous()          # mask in the new order
            cache["dense"] = dense

            def mod(bi, hi, q_idx, kv_idx):
                return dense[q_idx, kv_idx]
            cache["bm"] = create_block_mask(mod, B=None, H=None, Q_LEN=n, KV_LEN=n,
                                            device=dense.device, BLOCK_SIZE=block)
            cache["perm"], cache["inv"] = perm, inv
        x = hidden_states[:, cache["perm"]]
        for layer in self.layers:
            x = x + _attend(layer.self_attn, layer.input_layernorm(x, conditioning), cache["bm"],
                            attn_dtype, kernel_options)
            x = x + layer.mlp(layer.post_attention_layernorm(x, conditioning))
        return self.norm(x[:, cache["inv"]], conditioning)

    enc.forward = types.MethodType(forward, enc)
