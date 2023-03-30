import torch

from enum import Enum
from models.common.configs import EmbeddingLayerConfig

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, config: EmbeddingLayerConfig) -> None:
        super(EmbeddingLayer, self).__init__()
        self.position_embedding_type = config.position_embedding_type

        # word embeddings
        self.word_embedding_layer = torch.nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        if self.position_embedding_type == "learned":
            self.position_embedding_layer = torch.nn.Embedding(
                config.max_sequence_length,
                config.embedding_dim,
            )
        else:
            raise ValueError(f"Position Embedding of type {self.position_embedding_type} is not yet supported")

    def get_word_embeddings(self, input_ids):
        word_embeddings = self.word_embedding_layer(input_ids)
        return word_embeddings

    def get_position_embeddings(self, input_ids, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1]).expand((input_ids.shape[0], -1))

        position_embeddings = None
        if self.position_embedding_type == "learned":
            position_embeddings = self.position_embedding_layer(position_ids)

        return position_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """ Return embeddings

        Parameters:
            input_ids (torch.Tensor): shape [batch_size, sequence_len]
            position_ids (torch.Tensor): shape [batch_size, sequence_len]. Will be created based input_ids if not provided.

        Return:
            embeddings (torch.Tensor): shape [batch_size, sequence_len, embedding_dim]
        """

        embeddings = self.word_embedding_layer(input_ids)
        embeddings += self.get_position_embeddings(input_ids, position_ids=position_ids)

        return embeddings
