from models.common.configs.BaseConfig import BaseConfig
from models.common.configs.EmbeddingLayerConfig import EmbeddingLayerConfig
from models.common.configs.TransformerLayerConfig import TransformerLayerConfig

class BertConfig(BaseConfig):
    def __init__(
        self,
        # Embedding Params
        vocab_size: int = 50257,
        embedding_dim: int = 768,
        position_embedding_type: str = "learned",
        max_sequence_length: int = 2048,
        # Attention Params
        num_heads: int = 12,
        inner_dim: int = 768,
        intermediate_dim: int = None,
        ffn_activation: str = "relu",
        # other params
        hidden_dim: int = 768,
        num_hidden_layers: int = 12,
        layer_norm_epsilon: float = 1.0e-5,
    ):
        # Embedding Params
        self.embedding_layer_config = EmbeddingLayerConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            position_embedding_type=position_embedding_type,
            max_sequence_length=max_sequence_length,
        )

        # Attention Params
        self.attention_layer_config = TransformerLayerConfig(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            inner_dim=inner_dim,
            intermediate_dim=intermediate_dim,
            ffn_activation=ffn_activation,
            layer_norm_epsilon=layer_norm_epsilon,
        )

        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon