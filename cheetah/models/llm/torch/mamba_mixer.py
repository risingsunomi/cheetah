from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if input_tensor.ndim == 4 else (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if input_tensor.ndim == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    return input_tensor.reshape(
        input_tensor.shape[0],
        -1,
        chunk_size,
        input_tensor.shape[2],
        input_tensor.shape[3],
    )


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    lower_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~lower_mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    keep_mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool),
        diagonal=0,
    )
    return tensor_segsum.masked_fill(~keep_mask, -torch.inf)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.group_size = max(1, group_size)

    def forward(self, hidden_states: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))

        if hidden_states.shape[-1] % self.group_size == 0:
            grouped = hidden_states.view(*hidden_states.shape[:-1], -1, self.group_size)
            variance = grouped.pow(2).mean(dim=-1, keepdim=True)
            grouped = grouped * torch.rsqrt(variance + self.variance_epsilon)
            hidden_states = grouped.reshape_as(hidden_states)
        else:
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class MambaStateCache:
    def __init__(self, config: dict, batch_size: int, *, dtype: torch.dtype, device: torch.device | str):
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.conv_kernel_size = int(config["conv_kernel"])
        self.num_layers = int(config["num_layers"])
        self.ssm_state_size = int(config["ssm_state_size"])
        self.mamba_num_heads = int(config["mamba_num_heads"])
        self.mamba_head_dim = int(config["mamba_head_dim"])
        self.intermediate_size = self.mamba_num_heads * self.mamba_head_dim
        self.n_groups = int(config["n_groups"])
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.layers_block_type = list(config.get("layers_block_type", []))
        self.conv_states: list[torch.Tensor] = []
        self.ssm_states: list[torch.Tensor] = []

        for idx in range(self.num_layers):
            if idx < len(self.layers_block_type) and self.layers_block_type[idx] == "mamba":
                self.conv_states.append(
                    torch.zeros(
                        self.batch_size,
                        self.conv_dim,
                        self.conv_kernel_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
                self.ssm_states.append(
                    torch.zeros(
                        self.batch_size,
                        self.intermediate_size,
                        self.ssm_state_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            else:
                self.conv_states.append(torch.zeros((self.batch_size, 0), device=self.device, dtype=self.dtype))
                self.ssm_states.append(torch.zeros((self.batch_size, 0), device=self.device, dtype=self.dtype))

    def update_conv_state(self, layer_idx: int, new_conv_state: torch.Tensor, *, cache_init: bool = False) -> torch.Tensor:
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(device=self.device, dtype=self.dtype)
        else:
            state = self.conv_states[layer_idx]
            state = state.roll(shifts=-1, dims=-1)
            state[:, :, -1] = new_conv_state[:, 0, :].to(device=self.device, dtype=self.dtype)
            self.conv_states[layer_idx] = state
        return self.conv_states[layer_idx]

    def update_ssm_state(self, layer_idx: int, new_ssm_state: torch.Tensor) -> torch.Tensor:
        self.ssm_states[layer_idx] = new_ssm_state.to(device=self.device, dtype=self.dtype)
        return self.ssm_states[layer_idx]

    def reset(self) -> None:
        for state in self.conv_states:
            state.zero_()
        for state in self.ssm_states:
            state.zero_()


class MambaMixer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.num_heads = int(config["mamba_num_heads"])
        self.hidden_size = int(config["embed_dim"])
        self.ssm_state_size = int(config["ssm_state_size"])
        self.conv_kernel_size = int(config["conv_kernel"])
        self.intermediate_size = self.num_heads * int(config["mamba_head_dim"])
        self.layer_idx = layer_idx
        self.use_conv_bias = bool(config.get("use_conv_bias", True))
        self.activation = str(config.get("mamba_hidden_act", "silu")).lower()
        self.act = ACT2FN[self.activation]
        self.layer_norm_epsilon = float(config.get("layer_norm_epsilon", config.get("norm_eps", 1e-6)))
        self.n_groups = int(config["n_groups"])
        self.head_dim = int(config["mamba_head_dim"])
        self.chunk_size = int(config.get("chunk_size", 256))
        self.time_step_limit = tuple(config.get("time_step_limit", (0.0, float("inf"))))
        if len(self.time_step_limit) != 2:
            self.time_step_limit = (0.0, float("inf"))

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        proj_bias = bool(config.get("mamba_proj_bias", config.get("use_bias", False)))
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=proj_bias)
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        a = torch.arange(1, self.num_heads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a))
        self.norm = MambaRMSNormGated(
            self.intermediate_size,
            group_size=max(1, self.intermediate_size // max(1, self.n_groups)),
            eps=self.layer_norm_epsilon,
        )
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=proj_bias)

    def torch_forward(
        self,
        input_states: torch.Tensor,
        cache_params: Optional[MambaStateCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        projected_states = self.in_proj(input_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states_b_c, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        if cache_params is not None and cache_position is not None and int(cache_position[0].item()) > 0:
            cache_params.update_conv_state(self.layer_idx, hidden_states_b_c, cache_init=False)
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device, dtype=dtype)
            hidden_states_b_c = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
            if self.use_conv_bias and self.conv1d.bias is not None:
                hidden_states_b_c = hidden_states_b_c + self.conv1d.bias
            hidden_states_b_c = self.act(hidden_states_b_c)
        else:
            if cache_params is not None:
                hidden_states_b_c_transposed = hidden_states_b_c.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_b_c_transposed,
                    (cache_params.conv_kernel_size - hidden_states_b_c_transposed.shape[-1], 0),
                )
                cache_params.update_conv_state(self.layer_idx, conv_states, cache_init=True)
            hidden_states_b_c = self.act(self.conv1d(hidden_states_b_c.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        hidden_states_b_c = apply_mask_to_padding_states(hidden_states_b_c, attention_mask)
        hidden_states, b_state, c_state = torch.split(
            hidden_states_b_c,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        a_log = -torch.exp(self.A_log.float())
        if cache_params is not None and cache_position is not None and int(cache_position[0].item()) > 0:
            dt = dt[:, 0, :][:, None, :]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            a_log = a_log[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            d_a = torch.exp(dt[..., None] * a_log).to(device=cache_params.device)

            b_state = b_state.reshape(batch_size, self.n_groups, -1)[..., None, :]
            b_state = b_state.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, b_state.shape[-1]).contiguous()
            b_state = b_state.reshape(batch_size, -1, b_state.shape[-1])
            d_b = dt[..., None] * b_state[..., None, :]

            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            d_bx = (d_b * hidden_states[..., None]).to(device=cache_params.device)
            cache_params.update_ssm_state(
                self.layer_idx,
                cache_params.ssm_states[self.layer_idx] * d_a + d_bx,
            )

            c_state = c_state.reshape(batch_size, self.n_groups, -1)[..., None, :]
            c_state = c_state.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, c_state.shape[-1]).contiguous()
            c_state = c_state.reshape(batch_size, -1, c_state.shape[-1])
            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=c_state.device, dtype=c_state.dtype)
            ssm_states = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
            c_state = c_state.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states, c_state).view(batch_size, self.num_heads, self.head_dim)
            d_skip = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * d_skip).to(dtype)
            y = y.reshape(batch_size, -1)[:, None, :]
        else:
            dt = torch.nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            b_state = b_state.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            c_state = c_state.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            b_state = b_state.repeat(1, 1, self.num_heads // self.n_groups, 1)
            c_state = c_state.repeat(1, 1, self.num_heads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            d_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
            hidden_states = hidden_states * dt[..., None]
            a_log = a_log.to(hidden_states.dtype) * dt
            hidden_states, a_log, b_state, c_state = [
                reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, a_log, b_state, c_state)
            ]

            a_log = a_log.permute(0, 3, 1, 2)
            a_cumsum = torch.cumsum(a_log, dim=-1)

            l_mask = torch.exp(segment_sum(a_log))
            g_intermediate = c_state[:, :, :, None, :, :] * b_state[:, :, None, :, :, :]
            g_val = g_intermediate.sum(dim=-1)
            m_intermediate = g_val[..., None] * l_mask.permute(0, 2, 3, 4, 1)[..., None]
            m_val = m_intermediate.sum(dim=-1)
            y_diag = (m_val[..., None] * hidden_states[:, :, None]).sum(dim=3)

            decay_states = torch.exp((a_cumsum[:, :, :, -1:] - a_cumsum))
            b_decay = b_state * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (b_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            if cache_params is not None and cache_position is not None and int(cache_position[0].item()) > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(segment_sum(nn.functional.pad(a_cumsum[:, :, :, -1], (1, 0)))).transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            state_decay_out = torch.exp(a_cumsum)
            c_times_states = c_state[..., None, :] * states[:, :, None, ...]
            state_decay_out = state_decay_out.permute(0, 2, 3, 1)
            y_off = c_times_states.sum(-1) * state_decay_out[..., None]
            y = y_diag + y_off
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
            y = y + d_residual
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            if cache_params is not None:
                cache_params.update_ssm_state(self.layer_idx, ssm_state)

        scan_output = self.norm(y, gate)
        return self.out_proj(scan_output.to(dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[MambaStateCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.torch_forward(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
