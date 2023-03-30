

from .BaseConfig import BaseConfig
from .AttentionLayerConfig import AttentionLayerConfig


class TransformerLayerConfig(BaseConfig):
    def __init__(
        self,
        # Attention Params
        num_heads: int = 12,
        hidden_dim: int = 768,
        inner_dim: int = 768,
        # ffn
        intermediate_dim: int = None,
        ffn_activation: str = "relu",
        # layernorm
        layer_norm_epsilon: float = 1.0e-5,
    ) -> None:
        super(TransformerLayerConfig, self).__init__()

        self.attention_layer_config = AttentionLayerConfig(
            num_heads,
            hidden_dim,
            inner_dim,
        )

        self.hidden_dim = hidden_dim
        self.intermediate_dim = self.hidden_dim * 4 if intermediate_dim is None else intermediate_dim
        self.ffn_activation = ffn_activation
        self.layer_norm_epsilon = layer_norm_epsilon