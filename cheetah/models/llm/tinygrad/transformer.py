import tinygrad as tg

from .attention import MultiHeadAttention
from .mlp import MLP
from .moe import MOEMLP

class TransformerBlock:
    def __init__(
        self,
        config: dict,
        layer_idx: int | None = None,
    ):
        norm_eps = config.get("norm_eps", 1e-6)
        self.input_layernorm = tg.nn.RMSNorm(config["embed_dim"], eps=norm_eps)
        if bool(config.get("moe")):
            self.mlp = MOEMLP(config)
        else:
            self.mlp = MLP(
                config.get("embed_dim"),
                config.get("intermediate_dim"),
                config.get("hidden_act"),
                config.get("mlp_bias", True),
            )
        self.self_attn = MultiHeadAttention(
            config=config,
            is_causal=any("CausalLM" in arch for arch in config.get("architectures", []))
        )
        self.post_attention_layernorm = tg.nn.RMSNorm(config.get("embed_dim"), eps=norm_eps)

        self.attn_scale = config.get("attn_scale")
        self.mlp_scale = config.get("mlp_scale")

    def __call__(self, x, attention_mask, position_ids, start_pos=None):
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, attention_mask, position_ids, start_pos=start_pos)

        if self.attn_scale:
            x = x + self.attn_scale(attn_out)
        else:
            x = x + attn_out

        mlp_out = self.mlp(self.post_attention_layernorm(x))

        if self.mlp_scale:
            out = x + self.mlp_scale(mlp_out)
        else:
            out = x + mlp_out
        
        return out
