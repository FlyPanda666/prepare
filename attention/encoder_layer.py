import torch.nn as nn
from layer_norm import LayerNorm
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.ffn = PositionWiseFeedForward(
            ffn_hidden=ffn_hidden, d_model=d_model, dropout=dropout
        )
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None):
        _x = x
        x, _ = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
