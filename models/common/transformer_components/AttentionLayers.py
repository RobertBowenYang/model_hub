import torch
from models.common.configs import AttentionLayerConfig


class MultiheadAttentionLayer(torch.nn.Module):
    def __init__(self, config: AttentionLayerConfig) -> None:
        super(MultiheadAttentionLayer, self).__init__()
        self.num_heads = config.num_heads
        self.inner_dim = config.inner_dim
        self.head_dim = self.inner_dim // self.num_heads

        self.query_projection = torch.nn.Linear(config.hidden_dim, self.inner_dim)
        self.key_projection = torch.nn.Linear(config.hidden_dim, self.inner_dim)
        self.value_projection = torch.nn.Linear(config.hidden_dim, self.inner_dim)

        self.output_projection = torch.nn.Linear(self.inner_dim, config.hidden_dim)

    def split_by_heads(self, x):
        # x, shape [batch_size, sequence_length, inner_dim]
        batch_size, sequence_length = x.shape[:2]
        x = x.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3) # [batch_size, num_heads, sequence_length, head_dim]
        return x

    def combine_by_heads(self, x):
        # x, shape [batch_size, num_heads, query_len, head_dim]
        batch_size, num_heads, query_len, head_dim = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch_size, query_len, num_heads * head_dim)

    def forward(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
    ):

        # project onto query, key, value
        query = self.query_projection(query_input)
        key = self.key_projection(key_input)
        value = self.value_projection(value_input)

        # split by heads
        query = self.split_by_heads(query)
        key = self.split_by_heads(key)
        value = self.split_by_heads(value)

        # calculate attention scores
        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", query, key) # [batch_size, num_heads, query_len, key_len]
        # scale by head_dim. TODO: other scaling (ex. by layer_num)
        attention_scores = attention_scores  / float(self.head_dim) ** 0.5
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)

        # weighted aggregation on value
        attention_output = torch.einsum("bhqk, bhkd -> bhqd", attention_scores, value)
        attention_output = self.combine_by_heads(attention_output)

        # to output
        attention_output = self.output_projection(attention_output)

        return attention_output
