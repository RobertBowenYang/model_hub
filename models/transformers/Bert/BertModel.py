

import torch

from models.common.configs import BertConfig

from models.common.transformer_components import EmbeddingLayer
from BertEncoder import BertEncoder

class BertModel(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super(BertModel, self).__init__()
        self.config = config

        self.embedding_layer = EmbeddingLayer(self.config.embedding_layer_config)
        self.encoder = BertEncoder(config)

        self.layer_norm = torch.nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids):

        embeddings = self.embedding_layer(input_ids)
        encoder_output = self.encoder(embeddings)
        output = self.layer_norm(encoder_output)

        return output
