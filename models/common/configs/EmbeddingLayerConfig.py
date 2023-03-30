from .BaseConfig import BaseConfig


class EmbeddingLayerConfig(BaseConfig):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        position_embedding_type: str,
        max_sequence_length: int,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.position_embedding_type = position_embedding_type
        self.max_sequence_length = max_sequence_length
