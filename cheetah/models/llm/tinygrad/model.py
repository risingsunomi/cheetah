import tinygrad as tg

from ...shard import Shard
from .transformer import TransformerBlock


def _resolve_start_pos(
    start_pos: int | tg.UOp | None,
    position_ids: tg.Tensor | None,
) -> int | tg.UOp:
    if start_pos is not None:
        return start_pos
    if position_ids is None:
        raise ValueError("decode path requires start_pos or scalar position_ids")
    if len(position_ids.shape) == 2:
        if position_ids.shape != (1, 1):
            raise ValueError("decode path requires scalar position_ids")
        return int(position_ids[0, 0].item())
    if len(position_ids.shape) == 1 and position_ids.shape[0] == 1:
        return int(position_ids[0].item())
    raise ValueError("decode path requires scalar position_ids")


class Model:
    def __init__(
        self,
        config: dict,
        shard: Shard,
        use_tied: bool = False
    ):
        
        self.config = config
        self.shard = shard

        print(f"loading shard: {shard}")

        self.embed_tokens = tg.nn.Embedding(
            vocab_size=self.config["vocab_size"],
            embed_size=self.config["embed_dim"]
        )

        self.norm = tg.nn.RMSNorm(self.config["embed_dim"], eps=self.config["norm_eps"])

        self.layers = [
            TransformerBlock(self.config, layer_idx=layer_idx)
            for layer_idx in range(self.shard.start_layer, self.shard.end_layer)
        ]

        # output == lm_head
        self.output = tg.nn.Linear(
            self.config["embed_dim"],
            self.config["vocab_size"],
            bias=bool(self.config.get("lm_head_bias", False)),
        )
        if use_tied:
            self.output.weight = self.embed_tokens.weight

        # TinyJit only supports fixed input signatures, so keep it on the
        # single-token decode path instead of the full variable-length forward.
        self._decode_start_pos = tg.UOp.variable(
            "start_pos",
            0,
            max(int(self.config["max_seq_len"]) - 1, 0),
        )
        self._decode_token_jit = tg.TinyJit(self._decode_token_impl)
        self._decode_hidden_jit = tg.TinyJit(self._decode_hidden_impl)

    def reset_kv_cache(self) -> None:
        for layer in self.layers:
            if layer is None:
                continue
            attn = getattr(layer, "self_attn", None)
            if attn is not None and getattr(attn, "kv_cache", None) is not None:
                attn.kv_cache.clear()

    def reset_decode_jit(self) -> None:
        # Keep TinyJit captures alive across generations. Clearing the KV cache
        # is enough for a fresh decode; recapturing every turn adds avoidable
        # warmup cost.
        for jit_runner in (self._decode_token_jit, self._decode_hidden_jit):
            reset = getattr(jit_runner, "reset", None)
            if callable(reset):
                reset()

    def _resolve_shard_window(self, shard: Shard | None) -> tuple[int, int, bool]:
        requested = shard or self.shard
        loaded_start = int(getattr(self.shard, "start_layer", 0) or 0)
        loaded_end = int(getattr(self.shard, "end_layer", loaded_start + len(self.layers)) or (loaded_start + len(self.layers)))
        requested_start = int(getattr(requested, "start_layer", loaded_start) or loaded_start)
        requested_end = int(getattr(requested, "end_layer", loaded_end) or loaded_end)
        total_layers = int(getattr(requested, "total_layers", getattr(self.shard, "total_layers", loaded_end + 1)) or (loaded_end + 1))

        if requested_end == total_layers and loaded_end == total_layers - 1:
            requested_end = loaded_end

        if requested_start < loaded_start or requested_end > loaded_end:
            raise ValueError(
                f"Requested shard {requested_start}:{requested_end} is outside loaded range {loaded_start}:{loaded_end}"
            )

        local_start = requested_start - loaded_start
        local_end = requested_end - loaded_start
        is_final = total_layers > 0 and requested_end >= total_layers - 1
        return local_start, local_end, is_final

    def _forward_impl(
        self,
        x,
        position_ids: tg.Tensor | None=None,
        attention_mask: tg.Tensor | None=None,
        hidden_state: tg.Tensor | None = None,
        start_pos: int | tg.UOp | None = None,
        layer_start: int = 0,
        layer_end: int | None = None,
        is_final: bool | None = None,
    ):
        if hidden_state is None:
            x = self.embed_tokens(x)
        else:
            x = hidden_state

        if layer_end is None:
            layer_end = len(self.layers)

        for layer in self.layers[layer_start:layer_end]:
            x = layer(x, attention_mask, position_ids, start_pos=start_pos)

        apply_output = self.shard.end_layer == self.shard.total_layers - 1 if is_final is None else bool(is_final)
        if apply_output:
            x = self.norm(x)
            x = self.output(x)
        
        return x

    def run_shard(
        self,
        x,
        *,
        position_ids: tg.Tensor | None = None,
        attention_mask: tg.Tensor | None = None,
        hidden_state: tg.Tensor | None = None,
        shard: Shard | None = None,
        start_pos: int | tg.UOp | None = None,
    ):
        layer_start, layer_end, is_final = self._resolve_shard_window(shard)
        return self._forward_impl(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            hidden_state=hidden_state,
            start_pos=start_pos,
            layer_start=layer_start,
            layer_end=layer_end,
            is_final=is_final,
        )

    def _decode_token_impl(self, x: tg.Tensor, start_pos: int | tg.UOp) -> tg.Tensor:
        return self._forward_impl(x, attention_mask=None, start_pos=start_pos).realize()

    def decode_token(
        self,
        x: tg.Tensor,
        position_ids: tg.Tensor | None = None,
        *,
        start_pos: int | tg.UOp | None = None,
    ) -> tg.Tensor:
        resolved_start_pos = _resolve_start_pos(start_pos, position_ids)
        if len(x.shape) == 2 and x.shape[1] == 1:
            if not isinstance(resolved_start_pos, tg.UOp):
                resolved_start_pos = self._decode_start_pos.bind(int(resolved_start_pos))
            return self._decode_token_jit(x, resolved_start_pos)
        return self._decode_token_impl(x, resolved_start_pos)

    def _decode_hidden_impl(self, hidden_state: tg.Tensor, start_pos: int | tg.UOp) -> tg.Tensor:
        return self._forward_impl(
            None,
            attention_mask=None,
            hidden_state=hidden_state,
            start_pos=start_pos,
        ).realize()

    def decode_hidden(
        self,
        hidden_state: tg.Tensor,
        position_ids: tg.Tensor | None = None,
        *,
        start_pos: int | tg.UOp | None = None,
    ) -> tg.Tensor:
        resolved_start_pos = _resolve_start_pos(start_pos, position_ids)
        if len(hidden_state.shape) == 3 and hidden_state.shape[1] == 1:
            if not isinstance(resolved_start_pos, tg.UOp):
                resolved_start_pos = self._decode_start_pos.bind(int(resolved_start_pos))
            return self._decode_hidden_jit(hidden_state, resolved_start_pos)
        return self._decode_hidden_impl(hidden_state, resolved_start_pos)

    def __call__(
        self,
        x,
        position_ids: tg.Tensor | None=None,
        attention_mask: tg.Tensor | None=None,
        hidden_state: tg.Tensor | None = None,
        start_pos: int | tg.UOp | None = None,
    ):
        return self.run_shard(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            hidden_state=hidden_state,
            start_pos=start_pos,
            shard=self.shard,
        )
