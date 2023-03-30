

import torch

from models.common.configs import BertConfig
from models.common.transformer_components import TransformerLayer

class BertEncoder(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super(BertEncoder, self).__init__()

        self.layers = torch.nn.ModuleList(
            [TransformerLayer(config.attention_layer_config) for i in range(config.num_hidden_layers)]
        )

    def forward(self, embeddings):

        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states
        
