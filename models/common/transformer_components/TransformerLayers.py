
import torch

from models.common.configs import TransformerLayerConfig
from .AttentionLayers import MultiheadAttentionLayer

from models.common.model_utils.activations import get_activation_fn

class TransformerLayer(torch.nn.Module):

    def __init__(self, config: TransformerLayerConfig) -> None:
        super(TransformerLayer, self).__init__()

        # attention
        self.attention_layer = MultiheadAttentionLayer(config.attention_layer_config)

        # ffn. TODO: create layer
        self.ffn_layer_1 = torch.nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.activation_fn = get_activation_fn(config.ffn_activation)
        self.ffn_layer_2 = torch.nn.Linear(config.intermediate_dim, config.hidden_dim)

        # layernorms
        self.layer_norm_1 = torch.nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.layer_norm_2 = torch.nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    def self_attention_fn(self, x):
        attention_output = self.attention_layer(
            x,
            x,
            x,
        )
        return attention_output

    def ffn_fn(self, x):
        ffn_output = self.ffn_layer_1(x)
        ffn_output = self.activation_fn(ffn_output)
        ffn_output = self.ffn_layer_2(ffn_output)
        return ffn_output

    def forward(self, hidden_states):
        # use pre-norm by default. TODO: Add post-norm

        # pre-norm
        attention_output = self.self_attention_fn(
            self.layer_norm_1(
                hidden_states
            )
        )
        residual_1 = hidden_states + attention_output

        ffn_output = self.ffn_fn(
            self.layer_norm_2(
                residual_1
            )
        )
        residual_2 = residual_1 + ffn_output

        return residual_2
        


