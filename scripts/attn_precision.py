"""Run the port's attention in a lower precision than fp32.

The HF port (like the original, `upcast_attn_to_fp32=True`) converts Q, K, V to
fp32 right before attention. Fused attention kernels can only use tensor cores in
16-bit precision, so this swaps in a copy of WeatherNext2Attention.forward that
casts to a chosen dtype instead. Everything else (projections, MLPs, norms) is
unchanged. The softmax inside SDPA/Flex kernels still accumulates in fp32.
"""
import torch
from transformers.models.weathernext2 import modeling_weathernext2 as mw


def patch_attention_dtype(dtype: torch.dtype) -> None:
    def forward(self, hidden_states, attention_mask, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query = self.q_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(2, 3)
        key = self._gather_neighbouring_blocks(key)
        value = self._gather_neighbouring_blocks(value)

        attention_interface = mw.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, mw.eager_attention_forward)

        # The only change from the original: .to(dtype) instead of .float()
        query = query.reshape(-1, *query.shape[-3:]).to(dtype)
        key = key.reshape(-1, *key.shape[-3:]).to(dtype)
        value = value.reshape(-1, *value.shape[-3:]).to(dtype)

        attn_output, attn_weights = attention_interface(
            self, query, key, value, attention_mask,
            dropout=0.0, scaling=self.scaling, **kwargs)
        attn_output = attn_output.to(hidden_states.dtype).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    mw.WeatherNext2Attention.forward = forward
