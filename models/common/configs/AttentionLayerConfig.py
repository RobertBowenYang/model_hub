from .BaseConfig import BaseConfig


class AttentionLayerConfig(BaseConfig):
    def __init__(
        self,
        num_heads: int = 12,
        hidden_dim: int = 768,
        inner_dim: int = 768,
    ):
        super(AttentionLayerConfig, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim