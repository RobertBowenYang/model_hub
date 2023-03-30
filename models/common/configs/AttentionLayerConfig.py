from .BaseConfig import BaseConfig


class AttentionLayerConfig(BaseConfig):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        inner_dim: int,
    ):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim