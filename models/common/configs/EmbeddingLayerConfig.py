from .BaseConfig import BaseConfig


class EmbeddingLayerConfig(BaseConfig):
    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 768,
        position_embedding_type: str = "learned",
        max_sequence_length: int = 2048,
    ):
        super(EmbeddingLayerConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.position_embedding_type = position_embedding_type
        self.max_sequence_length = max_sequence_length
