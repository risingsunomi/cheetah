import tinygrad

class MLP:
    """
    A simple multi-layer perceptron (MLP) implementation.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        bias: bool = True,
    ):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.gate_proj = tinygrad.nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = tinygrad.nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = tinygrad.nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.activation = activation

    def activation_fn(self, x: tinygrad.Tensor) -> tinygrad.Tensor:
        if self.activation == "relu":
            return x.relu()
        elif self.activation == "silu":
            return x.silu()
        elif self.activation == "gelu":
            return x.gelu()
        elif self.activation == "selu":
            return x.selu()
        return x

    def __call__(self, x):
        return self.down_proj(
            self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        )
